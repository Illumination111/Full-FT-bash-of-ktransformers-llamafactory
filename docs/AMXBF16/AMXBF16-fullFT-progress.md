# AMX BF16 Full-FT 代码进度与训练链路

> 主体初始代码基准：KTransformers PR [#2086](https://github.com/kvcache-ai/ktransformers/pull/2086)，head [`1e95053`](https://github.com/kvcache-ai/ktransformers/commit/1e95053b15b32e6db8193fd852d62d051c6e7ef5)，初次核对时间 2026-07-17。
>
> 2026-07-21 增量核对：堆叠 PR [#2094](https://github.com/kvcache-ai/ktransformers/pull/2094)，head [`a3d1f15`](https://github.com/kvcache-ai/ktransformers/commit/a3d1f15c074a4056cfb252100d1584f52d8ccb27)。本文的 checkpoint、forward cache 和 backward repack 相关段落已更新到该 head；未标注 2026-07-21 增量的 kernel、dWeight 和 direct reload 说明仍沿用上述 #2086 基线。
>
> 本地 `fullft-development` 已快进到 `a3d1f15`，与 `upstream/pr/2094` 一致，KTransformers tracked working tree clean。针对本次增量运行 `test_sft_authoritative_grad.py` 与 `test_sft_checkpoint_reuse.py`，结果为 `15 passed`；这两个提交不改 C++，因此本轮没有重编译扩展或运行新的端到端 benchmark。

## 当前结论

PR #2086 已经在代码结构上接通 AMX BF16 CPU MoE 专家基础权重的 Full-FT 链路；PR #2094 的 2026-07-21 增量又补充了 checkpoint 前向复用和异步 repack 并发保护：

1. 基础 Gate、Up、Down 权重成为 optimizer 可见的 CPU BF16 `nn.Parameter`；
2. C++ backward 计算 `dX`、router 权重梯度以及三组基础权重梯度；
3. 自定义 Autograd Function 把 C++ 写入的基础权重梯度返回给 PyTorch；
4. optimizer 更新权威 BF16 参数后，下一轮 forward 前重新生成 AMX forward BufferB；
5. backward 所需的转置 BufferB 可以从 forward BufferB 直接 repack，并可跨层异步准备；
6. 当前 AMX BF16 绑定复用 inference 的 `GemmKernel224BF16`，基础权重 dW 使用独立的 `BF16DWeightKernel`；
7. C++/Python 已加入分阶段 profiler，可分别观察 forward、backward、dWeight、base-weight reload 和 backward repack。
8. 单进程纯 Full 与纯 LoRA 可在非重入 checkpoint 重算时复用第一次 CPU MoE 输出，避免再次提交 CPU expert forward；
9. wrapper 显式跟踪异步 backward repack，在任何新的 CPU pool forward/backward 任务提交前按需等待，修复 reentrant checkpoint 下的 forward cache underflow 竞态。

2026-07-21 的 checkpoint 增量已有 15 项本地 Python 回归通过，但这些测试主要验证 Python 状态机、执行顺序和 fake backend 下的梯度语义，不能替代真实 C++/NUMA 并发、完整 Full/LoRA/Hybrid reference、15-step 模式回归和最新 TPS 报告。因此当前仍不能把“聚焦测试通过”扩大为完整端到端验收。

## 2026-07-21：PR #2094 checkpoint 增量

今天有两个直接相关的提交，均修改 Python 控制流、测试或文档，没有修改 C++ kernel。

### [`4240f6f`](https://github.com/kvcache-ai/ktransformers/commit/4240f6fd44b3175d4dd7fca36609887a2fc1327f)：让纯 LoRA 复用 checkpoint forward

`wrapper.py` 把复用能力判断抽成 `_supports_checkpoint_forward_reuse(full_weight_grad, lora_rank)`。支持矩阵为：

| 模式 | `full_weight_grad` | `lora_rank` | 是否复用 |
| --- | ---: | ---: | --- |
| 纯 Full | `True` | `0` | 是 |
| 纯 LoRA | `False` | `> 0` | 是 |
| Hybrid | `True` | `> 0` | 否 |
| 无 Full、无 LoRA | `False` | `0` | 否 |

复用还要求单进程、`AMXBF16_SFT`，且 `KT_REUSE_CHECKPOINT_FORWARD` 没有设为 `0`。开启后，第一次 forward 保留 C++ backward cache，并额外缓存一次 CPU MoE output；非重入 checkpoint 重算仍会重算 decoder layer 和 router，但不会再次执行 GPU→CPU staging 与 CPU expert forward，而是直接取回缓存的 CPU output：

```text
checkpoint first forward
  -> CPU expert forward(save_for_backward=True)
  -> 保留 C++ backward cache + CPU output

checkpoint recompute
  -> 重算 GPU 主干/router
  -> 跳过 CPU pool submit
  -> get_checkpoint_output()
  -> 使用 first forward 留下的 C++ cache 执行 backward
```

这项改动是以额外保留 CPU cache/output 为代价，消除一次重复 CPU expert forward；不能再把该模式描述为“第一次 forward 完全不保留 CPU expert cache”。Hybrid、分布式、其他 backend 或显式关闭复用时仍走旧的“首次跳过 cache、重算时重新执行 CPU forward”路径。

### [`a3d1f15`](https://github.com/kvcache-ai/ktransformers/commit/a3d1f15c074a4056cfb252100d1584f52d8ccb27)：串行化 checkpoint 重算与异步 repack

Qwen3-30B LoRA 的 reentrant checkpoint 路径曾稳定触发：

```text
[KT-MOE ERROR] Forward cache stack underflow! cache_stack_top_=0
```

根因不是 checkpoint 开关本身，而是执行时序竞态：Layer N backward 完成后异步为 Layer N-1 repack backward 权重，checkpoint 随即重算 Layer N-1 forward；两者并发进入同一个不支持并发提交的 `NumaJobDistributor::do_numa_job()`。Python 已把 cache depth 记为 1，但 C++ forward cache 没有正确 push，随后 backward pop 时发生 underflow。

修复在 `BaseSFTMoEWrapper` 中增加 `_backward_repack_pending`：

1. `submit_backward_repack()` 成功提交后将其设为 `True`；
2. 同步/异步 forward、backward 及 CUDA-stream forward 在提交新的 CPU pool 任务前调用 `_wait_for_pending_backward_repack()`；
3. 仅在 pending 时执行 `wait_backward_repack()`，完成或对象已失效后清除标记；
4. 无 pending repack 的正常路径不增加 C++ wait。

关键顺序因此固定为：

```text
repack_submit -> repack_wait -> pool_submit
```

新增回归测试检查上述事件顺序。PR 内 RCA 记录的诊断 step 为 `loss=3.219`、`grad_norm=3.221`，Layer 46/47 cache 均正常完成 `0 -> 1 -> 0`；相关两组 Python 测试在本地合计 `15 passed`。

## Full-FT 与 LoRA 的 forward、训练耗时和 TPS 对比

> 本节是 2026-07-21 对后续开发分支 `a3d1f15` 的补充核对，用于回答 Full-FT 与 LoRA 的性能口径问题；它不属于上文 `1e95053` 基准已经具备的代码事实。涉及 checkpoint forward 复用、authoritative optimizer gradient 和 direct Full-FT cache 等细节时，应以该后续实现为准。

### 结论

Full-FT 和 LoRA 会执行大体相同的 Transformer 主干与 routed expert 基础 forward，但两者不是同一条完整计算路径：

- Full-FT 计算基础 expert，并让 router 和 Gate、Up、Down 基础权重参与训练；纯 Full 模式设置 `full_weight_grad=True`、`lora_rank=0`；
- LoRA 仍然必须执行完整的冻结基础 expert，同时额外计算低秩增量；router 在当前 LoRA 路径中冻结；
- 因此两者的纯 forward 时间可以很接近，但理论上不要求完全一致；
- 如果 TPS 覆盖整个训练 step，即 forward、backward、梯度通信和 optimizer step，则更不应假定一致。通常 LoRA TPS 应高于 Full-FT，但当共同的 CPU expert、GPU attention、PCIe/NUMA 搬运或 checkpoint 重算占据关键路径时，两者也可能接近。

### 两条 forward 路径的共同部分

两种模式都要完成：

1. GPU router 计算 top-k expert id 和 routing weight；
2. hidden state、expert id 和 routing weight 提交到 KT CPU expert；
3. CPU 执行基础 Gate、Up、Down expert 计算；
4. 同步 CPU 输出，并与可选 GPU expert 输出合并；
5. 训练时通过 `KTMoEFunction` 接入 backward。

冻结基础权重只表示 optimizer 不更新它，不表示 forward 可以跳过它。用一个线性层表示，两种模式分别是：

```text
Full-FT: y = W x
LoRA:    y = W x + scale × B(Ax)
```

对于 MoE expert，LoRA 增量分别附加在 Gate、Up、Down 投影上。因此 LoRA forward 相比 Full-FT 多出低秩 A/B GEMM；当 rank `r` 远小于 hidden size `H` 和 intermediate size `I` 时，这部分 FLOPs 通常远小于基础 expert GEMM，但仍有 kernel 调度和内存访问成本。

### Full-FT forward 的特有工作

当前实现中，Full-FT 有以下额外行为：

- router 不处于 `torch.no_grad()` 中，top-k selected weight 的梯度可继续回到 router；
- 自定义 Autograd 节点关联 `gate_proj_buf`、`up_proj_buf` 和 `down_proj_buf`；
- optimizer 更新 CPU 基础权重后，下一次该层 forward 入口可能执行 `kt.sft.base_weight_reload`，重新生成 AMX forward BufferB；
- AMX BF16 纯 Full 路径可以使用 `direct_fullft_cache`，减少部分 forward-cache 搬运和转换成本。

因此不能只根据“Full 没有 LoRA A/B GEMM”推断它的实测 forward 一定更快；router 建图、base-weight reload 和不同 cache 路径也会影响墙钟时间。

### LoRA forward 的特有工作

LoRA 模式中基础 expert 权重冻结，但基础分支和低秩分支都要执行：

```text
Gate = Wg x + scale × Bg(Ag x)
Up   = Wu x + scale × Bu(Au x)
Z    = SiLU(Gate) ⊙ Up
Out  = Wd Z + scale × Bd(Ad Z)
```

当前实现会在 LoRA 模式下用 `torch.no_grad()` 包裹 router，以避免为冻结 router 创建无用的 Autograd 节点；自定义 Autograd 节点改为关联 LoRA 参数。LoRA 参数更新后，KT 内核还需要读取更新后的 adapter buffer；指针为 dirty 时，forward 入口会刷新 LoRA pointers。

### 为什么纯 forward 时间可能接近，但不应要求相等

两种模式的大头通常都是共同的基础 MoE GEMM、GPU attention、CPU/GPU staging 和同步。LoRA rank 较小时，额外低秩计算可能只占很小比例；与此同时 Full-FT 又可能承担 router Autograd 和 base-weight reload。因此以下结果都可能合理：

- LoRA 略慢：低秩 GEMM、adapter 内存访问和调度开销占优；
- Full-FT 略慢：router 建图或 base-weight reload 占优；
- 两者非常接近：共同的基础 expert 或跨设备搬运主导关键路径。

所以，对只包含 model forward 的测量，可以预期“接近”，不能预期“严格相同”。比较时还必须把首次 warm-up、每 step 的 dirty weight reload、checkpoint 初次 forward 和 checkpoint recompute 分开，否则一个总的 `forward` 数字会混合不同工作。

### 为什么完整训练 TPS 通常不一致

训练 TPS 常见定义为 `有效训练 token 数 / 总训练墙钟时间`，其分母通常包含：

```text
forward + loss + backward + gradient communication
        + optimizer.step + zero_grad + checkpoint recompute
```

两种模式都要计算 expert `dX`，因为梯度仍需传回前一层。主要区别在参数梯度和 optimizer：

| 工作 | Full-FT | LoRA |
| --- | --- | --- |
| 基础 expert forward | 必须 | 必须 |
| LoRA A/B forward | 无 | 有 |
| expert `dX` | 必须 | 必须 |
| router gradient | 有 | 当前路径冻结 |
| 基础 `dW_gate/dW_up/dW_down` | 有 | 无 |
| LoRA `dA/dB` | 无 | 有 |
| optimizer 参数及 state | 完整基础权重规模 | 低秩参数规模 |
| optimizer 后基础权重 reload | 有 | 无 |

基础权重梯度规模为 `O(d_in × d_out)`，LoRA 参数梯度规模约为 `O(r × (d_in + d_out))`。Full-FT 还要更新完整 CPU expert 参数并读写相应 optimizer state，随后重新物化 AMX BufferB。因此在其他条件相同、且这些工作处于关键路径时，通常应有：

```text
LoRA training TPS > Full-FT training TPS
```

这不是无条件保证。以下因素会压缩差距：

- CPU 基础 expert forward 或共同的 `dX` 是瓶颈；
- GPU attention、PCIe/NUMA 数据搬运或多 rank 通信是瓶颈；
- gradient checkpointing 使两种模式都重算大量 forward；
- batch/sequence 太小，线程调度和同步占比高；
- LoRA rank 较大、target modules 较多；
- Full-FT 使用 direct cache、共享 backward BufferB 或 authoritative gradient 等优化；
- TPS 统计没有覆盖 optimizer step，或两组实验的 padding、有效 token 数、路由分布不同。

### 实验判断和 profiler 口径

公平 A/B 必须保持模型、数据顺序、有效 token 数、micro-batch、GAS、序列长度、精度、checkpoint、CPU 线程、NUMA 绑定和分布式配置一致，并经过相同 warm-up。建议至少分别报告：

```text
纯 model forward TPS
forward + backward TPS
完整 train-step TPS（包含 optimizer）
```

同时对齐下列 profiler 区间：

```text
kt.sft.routing
kt.sft.base_weight_reload
kt.sft.submit_and_gpu_experts
kt.sft.autograd_apply_and_cpu_sync
kt.sft.checkpoint_recompute
kt.sft.cpu_backward
optimizer.step
```

结果可按以下原则解释：

- 纯 forward 相差很小是合理现象；
- 完整训练 TPS 中 LoRA 明显更高，符合通常预期；
- 完整训练 TPS 接近，说明共同计算、通信或数据搬运更可能是瓶颈，不能据此认为两条路径相同；
- Full-FT 明显快于 LoRA 时，应检查 LoRA rank、target modules、adapter 同步和 kernel profile；
- 两边每 step 耗时几乎逐毫秒相同，应核实 Full 模式是否真正得到 `full_weight_grad=True, lora_rank=0`，以及 TPS 是否只统计了共同的 forward 区间。

## 全量微调本身的数据流：GPU 主干与 CPU AMX 专家协同

### “全量”指参数都参与训练，不是所有计算都搬到 CPU

在这条 Full-FT 路径中，模型仍然是一张连续的 PyTorch Autograd 计算图，但不同部分放在不同硬件上执行：

- embedding、attention、RMSNorm、LM head、router/gate 等普通模型模块仍由 GPU 执行；它们的参数、激活、梯度和 optimizer state 通常位于 GPU 显存；
- 被 KTransformers 替换的 routed expert 基础权重放在 CPU 内存，并作为 CPU BF16 `nn.Parameter` 参加同一个 optimizer；
- expert 的 Gate、Up、Down 大矩阵乘由 CPU 核心上的 AMX 执行，路由重排、SiLU、逐元素乘、NUMA 合并等辅助工作由 CPU 执行；
- 一次 MoE 层调用需要把 GPU 产生的 hidden state 和路由结果送到 CPU，再把 CPU expert 输出送回 GPU；backward 以相反方向交换上游梯度、`dX` 和 router 梯度；
- optimizer 最终同时更新 GPU 上的普通模型参数和 CPU 上的 expert 参数。因此这里的“全量微调”描述的是训练参数覆盖范围，而不是要求所有参数驻留在同一个设备。

本文中的 **router** 是在 token 与 expert 之间做 top-k 选择的网络（部分模型源码也把它命名为 `gate`）；**Gate projection** 则是每个 expert 内部与 Up、Down 并列的三组权重之一。两者位置、输入输出和梯度路径都不同。

在 LlamaFactory 的 full 模式没有额外冻结配置时，逻辑上的参数更新关系是：

```text
GPU 参数：embedding / attention / norm / router / LM head / GPU-side 分支 ...
   ▲                         同一 loss / 同一 Autograd 图
   │
   └──────────────┬───────────────────────────────────────────────┐
                  │                                               │
CPU 参数：routed expert 的 Gate / Up / Down BF16 基础权重          │
                  ▲                                               │
                  └────────────── 同一个训练 step 的 optimizer ────┘
```

这里还要区分 CPU、AMX 和内存三个概念。AMX 不是一块带有独立显存的加速卡，而是 CPU 核心内部的矩阵执行单元和 tile 寄存器。权重、激活和梯度长期驻留的是 CPU DRAM 或 cache；worker 运行 kernel 时才把当前 panel 装入 AMX tile，累加后写回 CPU buffer。所谓“提高 tile 常驻和复用”是让一个 kernel/任务的内层计算尽量复用已经装入 tile 的数据，不表示整层权重能跨 kernel 或跨训练 step 永久留在 AMX 寄存器中。

### 各硬件和内存区域分别保存什么

| 位置 | 主要常驻对象 | 在训练中的职责 |
| --- | --- | --- |
| GPU 显存 | 非 expert 参数、router 参数、GPU optimizer state、当前层 hidden state、router logits/top-k 结果、GPU Autograd 激活和梯度 | 执行模型主干、router 和可选 shared/LoRA expert；承接 CPU expert 前后的 Autograd |
| CPU 内存 | Gate/Up/Down 权威 BF16 Parameter、expert 梯度、CPU optimizer state、GPU↔CPU staging buffer、forward cache、AMX BufferA/BufferB/BufferC 和 scratch | 保存体积最大的 routed expert；完成路由重排、AMX 计算、梯度生成和 CPU 参数更新 |
| CPU NUMA-local 内存/cache | 每个 NUMA 的 intermediate slice、对应 forward BufferB、worker scratch | 避免远端 socket 反复读大权重；并行计算各自的 `I` 维分片 |
| AMX tile 寄存器 | 当前微内核处理的 activation/weight tile 和 FP32 accumulator | 在一个任务的 K reduction 内高吞吐完成 BF16 矩阵乘；任务结束后结果回写 CPU 内存 |

CPU 内存中的对象还分成两类，不能把它们看成同一份权重：

1. **权威训练状态**：CPU BF16 Parameter、`.grad` 以及 optimizer 的一阶/二阶状态。checkpoint 保存的是这类 Parameter；optimizer 只更新权威 Parameter。
2. **计算派生状态**：面向 AMX 访问顺序的 forward BufferB、其转置方向的 backward BufferB，以及每个 batch 的 BufferA、forward cache 和临时结果。这些对象服务于计算，不是新的可训练 Parameter。

### 一次 forward：数据从 GPU 主干进入 CPU expert，再回到 GPU

设当前 MoE 层共有 `T = batch_size × sequence_length` 个 token，每个 token 选择 `K` 个 expert，hidden size 为 `H`，expert intermediate size 为 `I`。单卡主路径如下：

```text
DataLoader/CPU batch
        │  token id 等输入送入 GPU
        ▼
GPU：Embedding → Attention/Norm → hidden state [T,H]
        │
        ├─ GPU router：logits → top-k expert id [T,K]
        │                       top-k weight [T,K]
        │
        ├──────── hidden/id/weight：GPU → CPU staging buffer ────────┐
        │                                                           ▼
        │       CPU：token-major → expert-major dispatch，得到约 T×K 条 routed row
        │                                                           │
        │                              ┌─ AMX Gate：G = X Wg^T ─────┤
        │                              └─ AMX Up：  U = X Wu^T ─────┤
        │                                                           ▼
        │                         CPU：Z = SiLU(G) ⊙ U
        │                                                           │
        │                              AMX Down：Y = Z Wd^T          │
        │                                                           ▼
        │                  CPU：乘 routing weight、按原 token 合并
        │                                                           │
        ├── 可并行：GPU shared expert / GPU LoRA expert              │
        │                                                           │
        ◀────────────── routed expert output [T,H]：CPU → GPU ──────┘
        │
        ▼
GPU：CPU routed 输出 + 可选 GPU-side expert 输出 → residual/下一层
```

该流程中 Gate、Up、Down 的作用分别是：

- **Gate**：把 `X:[*,H]` 投影到 `I` 维并经过 SiLU，形成每个 intermediate channel 的门控幅度；
- **Up**：把同一个 `X` 投影到 `I` 维，提供被门控的内容；
- **逐元素融合**：`Z = SiLU(G) ⊙ U`，只有这里把 Gate 与 Up 两条支路合在一起；
- **Down**：把 `Z:[*,I]` 投影回 `H` 维，使 expert 输出能够回到 Transformer residual stream。

CPU 不会对所有 expert、所有 token 做稠密计算。router 在 GPU 上先给出 top-k，CPU 再按 expert 重新排列约 `T×K` 条 routed row；每个 expert 只消费分到自己的 token。重排后的连续布局使相同 expert 的权重 panel 能被连续任务复用，也使 Gate/Up 可以共享同一批输入 `X`。

当前训练实现的 GPU→CPU staging 是显式边界：hidden state 转成 BF16、expert id 转成 INT64、routing weight 转成 FP32 后写入普通 CPU buffer，代码还会在提交 CPU 任务前同步相应 CUDA device。`submit_forward()` 随后异步派发 CPU expert；如果模型配置了 GPU-side shared expert 或独立 LoRA expert，GPU 可以在这段时间计算这些分支，最后在 `sync_forward()` 处等待 routed expert 并相加。没有可并行 GPU 分支时，GPU 主干仍必须等待 CPU expert 输出，不能仅凭“异步 submit”认为整段 CPU 时间都被隐藏。

### expert forward cache 与 gradient checkpoint 的数据代价

普通训练为了 backward，会在 CPU 保存 expert-major 的 `X`、Gate/Up 输出、`Z`、Down 输出及路由 offset。其规模跟实际 routed row 数而不是原 token 数成正比；top-k 为 `K` 时，主要中间激活约按 `T×K` 扩张。

截至 PR #2094 `a3d1f15`，非重入 checkpoint 有两种策略：

1. **默认复用路径**：单进程纯 Full 或纯 LoRA、`AMXBF16_SFT` 且未设置 `KT_REUSE_CHECKPOINT_FORWARD=0` 时，第一次 forward 保留 C++ backward cache 和一份 CPU output。重算仍执行 GPU decoder/router，但跳过 hidden/id/weight 的第二次 GPU→CPU staging 和 CPU expert forward；缓存 output 仍需从 CPU 再复制到原 output device。
2. **fallback 路径**：Hybrid、分布式、其他 backend 或显式关闭复用时，第一次 forward 不保留大 cache；backward 触发重算后再次执行 CPU expert forward，以计算换内存。

因此 checkpoint 的交换不再只有一种：复用路径用更多 CPU 常驻 cache/output 换掉重复 CPU forward，fallback 路径则继续用重复 forward 和跨设备流量换取更低的长期 activation 占用。分析 profile 时必须先确认 `reuse_checkpoint_forward` 是否开启，不能把所有 `checkpoint 重算` 都解释为完整重复 CPU expert 数据流。

### 一次 backward：上游梯度进入 CPU，dX、router 梯度回到 GPU

forward 输出回到 GPU 后，loss 仍在 GPU 上形成。GPU Autograd 反向走到 KT MoE 节点时，数据按以下方向流动：

```text
GPU：来自后续层的 dO [T,H]
        │
        └────────────── dO：GPU → CPU grad_output buffer
                                      │
                                      ▼
CPU：按 forward 路由展开 dY_e = routing_weight_e × dO
      │
      ├─ router selected-weight grad = <expert_output_e, dO>
      │
      ├─ AMX Down dX：        dZ = dY Wd
      │
      ├─ CPU 激活反向：       dG、dU
      │
      ├─ AMX Gate/Up dX：     dX_e = dG Wg + dU Wu
      │
      └─ AMX dWeight：        dWg = dG^T X
                              dWu = dU^T X
                              dWd = dY^T Z
        │
        ├──────── dX [T,H]：CPU → GPU，继续 attention/前层 backward
        ├──────── selected router grad [T,K]：CPU → GPU
        │                               └─ GPU Autograd 继续经过 top-k weight/router
        └──────── dWg/dWu/dWd：留在 CPU，连接到 CPU expert Parameter.grad
```

这里有两类完全不同的梯度流：

- `dX` 和 selected routing-weight gradient 必须回到 GPU，因为它们是 GPU 计算图上游节点的输入；
- `dW_gate`、`dW_up`、`dW_down` 不需要在单卡场景往返 GPU。C++ 直接写 CPU BF16 gradient buffer，自定义 Autograd Function 再把这些 buffer 返回到相应 CPU Parameter 的 `.grad`。

backward BufferB 也是 CPU 内存内的布局变化，不是权重传到另一个设备。Down dX 需要 `Wd` 的反向方向，Gate/Up dX 需要 `Wg/Wu` 的反向方向，因此代码把 forward packed BufferB 转置 repack 到 backward BufferB。共享 backward BufferB 时，当前层结束后可提前为 backward 顺序的下一层提交 repack，并尝试与 GPU 上其他层的 attention backward 重叠；真正使用前仍必须等待它就绪。

### NUMA 如何共同完成一个 expert

在多 socket CPU 上，内存访问不是等价的。当前 TP/NUMA wrapper 沿 expert 的 `I` 维切分 Gate/Up 输出和 Down 输入，每个 NUMA node 使用本地 worker、BufferB 和 scratch 处理自己的 slice：

```text
同一个 routed X [*,H]
   ├─ NUMA 0：Gate/Up 的 I0 slice → Z0 → Down partial Y0
   ├─ NUMA 1：Gate/Up 的 I1 slice → Z1 → Down partial Y1
   └─ ...

Forward：Y = Y0 + Y1 + ...
Backward：dX = dX0 + dX1 + ...，router partial grad 也需要合并
dWeight：各 NUMA 直接写 [E,I,H] 或 [E,H,I] 中互不重叠的 I slice
```

因此 NUMA 间不可避免地要合并最终 `H` 维输出和 `dX`，但大体积权重读取、AMX pack 和 dWeight 写入应尽量停留在各自 node。optimizer 更新后 direct BF16 reload 也是从完整 CPU Parameter 中按 stride 读取每个 `I` slice，直接生成对应 NUMA 的 forward BufferB；它减少临时分片和 memcpy，但仍要读取全部新权重并完成一次 pack。

### optimizer：两类设备上的参数在同一个 step 收敛到新版本

backward 完成后，PyTorch 看到的是同一计算图中的混合设备参数：GPU 普通参数持有 GPU gradient，CPU expert Parameter 持有 CPU gradient。optimizer 按 Parameter 所在设备更新它们；expert 的 optimizer state 留在 CPU，GPU 参数的 optimizer state 留在 GPU。state 的精度、是否使用 fused kernel 等细节由实际 Trainer/optimizer 配置决定，不应从数据所在设备反推。

一次参数更新及下一轮重新物化的顺序是：

```text
全部 backward 完成
  ├─ 单卡：CPU expert dW 留在 CPU；GPU 参数梯度留在 GPU
  └─ 多 rank：KT 管理梯度按分布式路径同步，CPU gradient 可能暂存到 GPU 做 all-reduce 再拷回
        ↓
grad clip / optimizer.step()
  ├─ GPU：更新 attention/router/... 参数及其 optimizer state
  └─ CPU：更新 Gate/Up/Down 权威 BF16 Parameter 及其 optimizer state
        ↓
把 expert base weight 标记为 dirty
        ↓
下一轮该 MoE 层 forward 入口
  └─ CPU 权威 BF16 Parameter → direct pack → 各 NUMA forward BufferB
```

optimizer 更新的是 row-major 权威 Parameter，而 AMX forward 读取的是 BufferB，所以二者之间必须有 dirty/reload 边界。BF16 direct reload 消除的是中间 TP 临时分片，不会消除“读取更新后的 Gate/Up/Down，并重新生成 AMX 布局”本身。只要权重每个 step 都改变，这段 CPU 内存流量就仍然存在。

### 每层真正跨设备和不跨设备的数据

| 阶段 | 数据 | 方向 | 说明 |
| --- | --- | --- | --- |
| forward 输入 | hidden `[T,H]`、id `[T,K]`、weight `[T,K]` | GPU → CPU | routed expert 的固定边界；当前代码在 CPU submit 前有 CUDA 同步 |
| forward 输出 | merged expert output `[T,H]` | CPU → GPU | 回到 residual stream，并与可选 GPU-side expert 输出相加 |
| backward 输入 | `dO [T,H]` | GPU → CPU | CPU expert backward 的上游梯度 |
| backward 输出 | `dX [T,H]`、selected router grad `[T,K]` | CPU → GPU | 继续 GPU 主干和 router 的 Autograd |
| expert dWeight | 三组完整参数梯度 | CPU 内部 | 单卡不需要经过 GPU；C++ buffer 直接接到 CPU Parameter.grad |
| optimizer 后 reload | CPU Parameter → forward BufferB | CPU 内部 | 是布局物化，不是 CPU→GPU 传输，也不是数值 requant |
| 多 rank 同步 | CPU expert grad → GPU all-reduce → CPU | CPU ↔ GPU/网络 | 只在分布式梯度同步路径发生，可能成为额外大流量 |

所以这套 Full-FT 系统的关键并不是“CPU 代替 GPU 完成整个模型”，而是：GPU 保持 Transformer 主干和训练控制流，CPU 内存容纳 routed expert 的参数、梯度与 optimizer 状态，CPU worker 用 AMX 完成 expert 的三个主要 GEMM，两边通过每层 activation/gradient staging 接成一张 Autograd 图。性能上需要同时观察 GPU↔CPU staging、AMX Gate/Up/Down 与 dWeight、NUMA 合并、checkpoint 重算、optimizer CPU 内存带宽和每 step BufferB reload；只优化其中一个 kernel 不代表整条训练数据流已经没有瓶颈。

## 代码地图

| 层次 | 关键代码 | 实际职责 |
| --- | --- | --- |
| 配置 | [`python/sft/config.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/config.py) | 解析 `lora/full/hybrid`、设置 `full_weight_grad`、配置 OMP/PyTorch CPU 线程 |
| 模型替换 | [`python/sft/wrapper.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/wrapper.py) | 创建 KT layer wrapper、提取专家权重、初始化 Full-FT 参数和梯度 buffer |
| 权威参数 | [`python/sft/base.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/base.py) | 保存 `gate/up/down_proj_buf` 和对应 C++ 写入的梯度 buffer |
| Python forward | [`python/sft/layer.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/layer.py) | router、dirty weight reload、CPU submit、checkpoint cache 策略、Autograd 接入 |
| Autograd | [`python/sft/autograd.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/autograd.py) | checkpoint 重算触发、C++ backward、跨 rank 输出分发、返回基础权重梯度、提交下一层 repack |
| Optimizer 接入 | [`python/sft/lora.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/lora.py) | 收集 KT trainable 参数、跨 rank 同步梯度、optimizer 后标记权重 dirty |
| AMX Python backend | [`python/sft/amx.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/amx.py) | 创建 C++ MoE、传递数据指针、同步 forward/backward、调用 `update_base_weights()` |
| Pybind | [`ext_bindings.cpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/ext_bindings.cpp) | 将 AMX BF16 SFT 绑定到 `GemmKernel224BF16 + AMX_BF16_MOE_TP`，暴露 backward/repack/profile/reload 接口 |
| 单 NUMA/TP 内核 | [`operators/amx/sft_moe.hpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/sft_moe.hpp) | expert dispatch、forward cache、dX、router grad、LoRA grad、base dW、forward/backward BufferB |
| dWeight driver | [`operators/amx/la/bf16_dweight.hpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/la/bf16_dweight.hpp) | dW operand 转置打包、线程局部 scratch、AMX/AVX512-BF16 GEMM、FP32→BF16 写回 |
| BF16 布局 | [`operators/amx/la/amx_raw_buffers.hpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/la/amx_raw_buffers.hpp) | forward BufferB pack、strided pack、unpack 和直接 BB→BB 转置 |
| TP/NUMA wrapper | [`operators/moe-sft-tp.hpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/moe-sft-tp.hpp) | intermediate 维分片、NUMA 并发、梯度合并/直写、共享 backward BB、direct BF16 reload |
| Profiler | [`operators/sft_profile.hpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/sft_profile.hpp)、[`python/sft/profiler.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/python/sft/profiler.py) | C++ stage counter、跨层收集、格式化和 reset |

## 1. 权重、派生布局和梯度对象

Full-FT 同时维护三种不同语义的权重对象：

```text
权威、可训练的 PyTorch BF16 参数
gate_proj_buf / up_proj_buf / down_proj_buf
              │
              │ base-weight reload + pack
              ▼
AMX Forward BufferB
              │
              │ direct BufferB transpose/repack
              ▼
AMX Backward BufferB
```

- PyTorch BF16 参数是 optimizer 更新和 checkpoint 保存时的权威副本；
- forward BufferB 是从权威参数生成的 AMX/VNNI 派生布局；
- backward BufferB 是计算 base dX 所需的转置布局；启用 `share_backward_bb` 时，它来自跨层共享内存池，不是每层永久保存一份；
- BF16 模式没有 INT8/INT4 的 scale 计算和数值再量化。日志中的 `requant` 对 BF16 实际表示“权重 reload + AMX pack”。

基础权重形状为：

```python
gate_proj_buf: [E, I, H]  # BF16 CPU nn.Parameter
up_proj_buf:   [E, I, H]
down_proj_buf: [E, H, I]
```

对应的 C++ 写入梯度 buffer 为：

```python
grad_gate_proj_buf: [E, I, H]  # BF16 CPU Tensor
grad_up_proj_buf:   [E, I, H]
grad_down_proj_buf: [E, H, I]
```

其中 `E` 是专家数，`H` 是 hidden size，`I` 是完整 MoE intermediate size。`init_full_weight_grad_buffers()` 不预先把这些梯度 buffer 赋给 `.grad`；C++ 写入后，由 `KTMoEFunction.backward()` 返回，PyTorch 再完成 Parameter 的梯度连接。

## 2. 模式初始化和 optimizer 参数接入

`KTSFTConfig` 支持：

```text
kt_train_mode = "lora" | "full" | "hybrid"
kt_full_weight_grad = kt_train_mode in ("full", "hybrid")
```

- `full`：基础专家权重参与训练，允许 `lora_rank=0`；
- `lora`：基础权重冻结，只训练 LoRA；
- `hybrid`：基础权重和 LoRA 同时训练。

`wrapper.py` 从原模型权重初始化上述 Parameter/buffer。`get_kt_trainable_params()` 在 Full/Hybrid 模式下返回基础权重 Parameter，并追加存在的 LoRA Parameter；这些对象随后被加入 Trainer 创建的 optimizer。

当前 PR 保持 PyTorch fused AdamW 路径，没有合入实验性的 DeepSpeedCPUAdam。`config.py` 还会处理 Accelerate 把 GPU job 的 `OMP_NUM_THREADS` 设为 1 的情况：优先使用 `ACCELERATE_KT_OMP_NUM_THREADS`，其次保留显式且大于 1 的 `OMP_NUM_THREADS`，否则按进程 affinity 可见的物理核心数设置环境变量和 `torch.set_num_threads()`。

## 3. Forward 代码路径

### 3.1 Router 和 Autograd 条件

`KTMoELayerWrapper.forward()` 首先调用 `_compute_routing()`：

- LoRA-only 下 router 被视为冻结对象，代码使用 `torch.no_grad()`；
- Full/Hybrid 下使用正常 Autograd 上下文，因此 router linear、routing probability 和 top-k weight 路径可以获得梯度；
- top-k expert id 是离散选择，不对 expert index 求导；C++ 返回的是已选 expert routing weight 的梯度。

### 3.2 optimizer 更新后的 lazy weight reload

optimizer 完成后，`update_kt_lora_pointers()` 设置：

```python
wrapper.wrapper._base_weights_dirty = True
```

下一轮 layer forward 开始时执行：

```python
if full_weight_grad and wrapper._base_weights_dirty:
    wrapper.update_base_weights()
    wrapper._base_weights_dirty = False
```

因此 base-weight reload 的物理执行位置在下一轮 forward 入口，逻辑上属于上一次 optimizer 更新后的参数物化。它必须发生在新一轮 forward 前，保证一次 forward/backward 始终使用同一个权重版本。

### 3.3 CPU submit 和 expert-major dispatch

Python 将 hidden state、top-k id 和 routing weight 交给 CPU backend。AMX BF16 路径使用：

```text
hidden state   → BF16 CPU
expert id      → INT64 CPU
routing weight → FP32 CPU
```

C++ `forward_sft()` 统计每个 expert 的 token 数，将 token-major 输入整理为 expert-major 连续区域。激活通过 `BufferA::from_mat()` 转换成 AMX tile 布局；代码中沿用的 “Quantize input” 注释在 BF16 模式下应理解为 activation pack，不是数值量化。

单个 SwiGLU expert 的基础计算为：

\[
G=XW_g^T,\qquad U=XW_u^T
\]

\[
Z=\operatorname{SiLU}(G)\odot U,\qquad Y=ZW_d^T
\]

随后按 routing weight 加权并把多个 expert 贡献合并。TP/NUMA 模式按 `I` 维切分 Gate/Up 输出和 Down 输入，每个 NUMA 产生部分 Down 输出，wrapper 最后对这些输出求和。

### 3.4 Forward cache 与 gradient checkpoint

backward 需要缓存：

- expert-major 输入 `X`；
- Gate/Up 输出；
- 激活后中间量 `Z`；
- 每个 expert 的 Down 输出；
- expert id、routing weight、局部 token 数和 offset。

非重入 gradient checkpoint 下，`KTMoEFunction` 保存一个 sentinel tensor；backward 访问 `ctx.saved_tensors` 时会触发 decoder layer 重算。是否在重算时再次运行 CPU MoE，由 `reuse_checkpoint_forward` 决定：

- 开启复用时，`first_forward` 以 `save_for_backward=True` 执行并通过 `cache_checkpoint_output()` 保存 CPU output；`recompute` 发现 `_kt_has_cached_forward=True` 后跳过 `submit_forward()`，通过 `get_checkpoint_output()` 取回结果，原 C++ cache 直接供真正的 MoE backward 使用；
- 未开启复用时，`first_forward` 把 `save_for_backward_submit` 设为 `False`，避免保留整层 C++ cache；`recompute` 再以 `save_for_backward=True` 执行 CPU forward 并填充 cache。

复用只对单进程纯 Full/纯 LoRA 的 `AMXBF16_SFT` 默认开启；Hybrid 与分布式仍使用 fallback。无论哪条路径，真正的 MoE backward 都必须在对应 cache 就绪后才能开始。

## 4. Backward 数学和执行顺序

设 token 对 expert `e` 的 routing weight 为 `r_e`，上游梯度为 `dO`。C++ 先形成该 expert 的：

\[
dY_e=r_e\,dO
\]

并由 expert 输出与 `dO` 计算所选 routing weight 的梯度。随后执行：

\[
dZ=dY W_d
\]

\[
dU=dZ\odot \operatorname{SiLU}(G)
\]

\[
dG=dZ\odot U\odot \operatorname{SiLU}'(G)
\]

\[
dX=dG W_g+dU W_u
\]

Full/Hybrid 还计算三组基础权重梯度：

\[
dW_g=dG^T X,\qquad dW_u=dU^T X,\qquad dW_d=dY^T Z
\]

`base_grad_output_bf16_ptr_` 保存计算 `dW_d` 所需的 route-weighted `dY`。工作缓冲区随后会被 Gate/Up backward 复用，因此不能直接拿会被覆盖的 `grad_output_bf16_ptr_` 计算 Down dW。

### 4.1 backward BufferB 生命周期

base dX 需要与 forward 相反的权重方向：

```text
Forward Gate/Up BufferB:  [I, H]
Backward Gate/Up BufferB: [H, I]

Forward Down BufferB:     [H, I]
Backward Down BufferB:    [I, H]
```

当前 `GemmKernel224BF16::BufferB` 提供 `from_bb_transposed()`，可以直接从 packed forward BufferB 生成 packed transposed BufferB，不必先 `to_mat()` 到完整 BF16 workspace 再 `from_mat_transposed()`。

启用 `share_backward_bb` 时：

1. 当前层完成后，`KTMoEFunction.backward()` 对 backward 顺序中的下一层调用 `submit_backward_repack()`，wrapper 将 `_backward_repack_pending` 设为 `True`；
2. C++ 独立线程在 GPU attention 等其他 backward 工作期间准备下一层布局；
3. `KTMoEFunction.backward()` 仍在 checkpoint 重算前显式 `wait_backward_repack()`；此外 `BaseSFTMoEWrapper` 的所有 sync/async forward、backward CPU pool 提交入口都有统一 pending guard，防止 reentrant 重算从其他入口绕过等待；
4. 当前层 backward 检查共享池 owner，必要时执行同步 fallback repack；等待完成后 pending 标记被清除。

这条路径只影响 backward 权重布局，不能与 optimizer 后的 base-weight reload 混为同一个阶段。新增 pending guard 是 Python 侧的最小串行化修复；长期仍可考虑让 `NumaJobDistributor` 自身支持并发提交，或为 repack 使用独立线程池。

### 4.2 `BF16DWeightKernel`

当前 AMX BF16 SFT 绑定使用 `GemmKernel224BF16`，因此 `backward_base_weight_grad()` 进入新的 `BF16DWeightKernel` 分支。旧 `GemmKernel224BF` 分支仍作为兼容代码保留，但不是当前 `AMXBF16_SFT_MOE` 绑定的主路径。

新 dWeight driver 的结构为：

```text
每个激活 expert
  ├─ Gate/Up：每个 i_tile 共用一个任务
  └─ Down：    每个 i_tile 一个任务

任务内：
  转置打包 activation/gradient panel
      ↓
  GemmKernel224BF16 AMX 或 AVX512-BF16 driver
      ↓
  FP32 32×32 accumulator
      ↓
  转换并写入 BF16 dW
```

任务数是：

```text
activated_expert × (2 × ceil(I_local / 32))
```

每个 worker 使用 `thread_local BF16DWeightScratch`。scratch 按本轮最大 route 数对应的 `padded_k` 扩容并复用，包含两份 BufferA、一份 BufferB 和两份对齐的 FP32 C tile。Gate/Up 共用 input B panel；Down 为固定 `i_tile` 预打包 intermediate B panel。该设计降低任务领取次数和重复 panel pack，并让通用 AMX driver负责完整 K reduction。

内层 profiler 只在 `KT_SFT_PROFILE` 启用时计时。`worker_cpu.pack_a/pack_b/kernel_gate_up/kernel_down/store` 是各 worker 累加的 CPU 工作时间，不是外层关键路径墙钟，因此不能与 `backward.base_weight_grad` 直接做父子百分比相加。

## 5. TP/NUMA 梯度语义

`TP_MOE_SFT::backward()` 按完整 `I` 维为每个 TP 计算 offset。

基础 dW 的目标指针直接指向最终梯度 Tensor 中互不重叠的 TP slice：

```text
Gate/Up dW: [E, I_full, H] 的 I slice
Down dW:    [E, H, I_full] 的 I slice
```

因此基础 dW 不需要跨 TP 求和；各 NUMA 直接写自己的 slice。wrapper 在 dispatch 前对三组完整基础梯度做并行分块清零。

需要跨 TP 合并的结果包括：

- `grad_input`：不同 `I` slice 对同一个输入的贡献相加；
- router `grad_weights`：按 TP 部分输出累加；
- LoRA reduce-type 梯度。

LoRA copy-type 梯度也采用 TP slice 直写，reduce-type 梯度使用 active-expert 范围的稀疏 FP32 partial buffer 后再归并。这与基础 dW 的直写原则相同，但 buffer 形状和归并规则不同。

## 6. C++ 梯度如何接回 PyTorch

`KTMoEFunction.apply()` 把三组基础 Parameter 作为最后三个 Tensor 输入传入。C++ backward 通过 `amx.py` 提供的 data pointer 写入 `grad_*_proj_buf`，`KTMoEFunction.backward()` 再在返回 tuple 的对应位置返回这三个 Tensor：

```python
return (
    grad_input,
    None,
    grad_weights,
    ...,
    grad_gate_proj,
    grad_up_proj,
    grad_down_proj,
)
```

PyTorch 随后把它们累积到 `gate_proj_buf.grad`、`up_proj_buf.grad` 和 `down_proj_buf.grad`。多进程训练中，`sync_kt_lora_gradients()` 把 CPU 基础梯度复制到 GPU，执行 `all_reduce(SUM) / world_size`，再复制回 CPU，确保各 rank optimizer 使用一致梯度。

## 7. optimizer 后的 BF16 direct reload

`AMXSFTMoEWrapper.update_base_weights()` 的首选路径为：

```python
self.moe.set_base_weight_pointers(
    gate_proj_buf.data.data_ptr(),
    up_proj_buf.data.data_ptr(),
    down_proj_buf.data.data_ptr(),
)
self.cpu_infer.submit(self.moe.load_weights_task())
self.cpu_infer.sync()
```

在当前 `GemmKernel224BF16` 上，`kSupportsDirectBf16Reload=True`。`TP_MOE_SFT::load_weights()`：

1. 计算每个 NUMA 的 `intermediate_offset`；
2. 在 NUMA worker 上调用 `load_forward_weights_from_full_bf16()`；
3. Gate/Up 从 `[E, I_full, H]` 的连续 I slice 使用 `from_mat_strided(..., stride=H)`；
4. Down 从 `[E, H, I_full]` 的列 slice 使用 `from_mat_strided(..., stride=I_full)`；
5. 直接写入各 NUMA 已有的 forward BufferB。

相较 fallback 路径，这会消除：

```text
temp_gate/temp_up/temp_down 分配
→ 完整 TP 分片 memcpy
→ 从临时连续分片 pack
→ delete[] 临时分片
```

但它没有消除全量权重 pack：每轮 optimizer 更新后，三组最新 BF16 权重仍必须读出并写成 AMX forward BufferB。因此不能把这项修改表述成“直接释放全部 requant 时长”。新 profiler 使用 `weights.base_reload.direct_pack` 单独记录这条路径。

若 `share_backward_bb=False`，forward pack 后还会同步生成 backward BufferB；若为 `True`，reload 跳过持久 backward pack，后续按 backward 顺序动态 repack。

非 `GemmKernel224BF16` kernel 仍走旧 fallback：分配 TP 临时分片、复制、pack、可选 backward pack、释放。Python 中对象重建 fallback 也仍存在；其中 `del old_moe` 没有清除 `self.moe` 对旧对象的引用，不能证明旧 C++ 对象会在新对象创建前释放。当前 AMX BF16 正常绑定具备 `set_base_weight_pointers`，通常不会进入该 fallback，但该生命周期问题尚未从代码中消失。

## 8. 三种容易混淆的 pack

| 操作 | 输入 → 输出 | 触发位置 | 是否仍存在 |
| --- | --- | --- | --- |
| Activation pack | row-major BF16 activation → AMX BufferA | 每次 forward/backward GEMM 前 | 是，属于计算准备 |
| Base-weight reload/direct pack | optimizer 更新后的权威 BF16 weight → forward BufferB | 下一轮 forward 入口 | 是；direct path 只去掉 TP 临时分片 |
| Backward repack | forward BufferB → transposed backward BufferB | backward 前或跨层异步 | 是；BF16 支持直接 BB→BB 转置 |

三者处理的对象和生命周期不同。特别是 direct backward repack 的优化不能作为消除 `update_base_weights()` 全部耗时的依据。

## 9. 一次训练迭代的真实时序

```text
第 t 轮 forward
  │
  ├─ 若 W_t dirty：direct BF16 reload/pack → forward BufferB
  ├─ router
  ├─ CPU submit、expert dispatch、activation pack
  ├─ Gate/Up → SwiGLU → Down → routing merge
  └─ 普通训练保存 cache；checkpoint 按模式选择：
       ├─ 复用路径：保留 C++ cache + CPU output
       └─ fallback：首次 forward 跳过 C++ cache

loss
  │
  ▼
第 t 轮 backward
  │
  ├─ 若 backward BB repack pending：先等待，再提交新的 CPU pool 任务
  ├─ checkpoint：访问 sentinel，触发整层重算
  │    ├─ 复用路径：跳过 CPU expert forward，读取 cached CPU output/cache
  │    └─ fallback：重新执行 CPU expert forward 并填充 cache
  ├─ route-weighted dY 和 router grad
  ├─ Down dX → activation backward → Gate/Up dX
  ├─ BF16DWeightKernel：dW_gate / dW_up / dW_down
  ├─ TP/NUMA merge 或 slice 直写
  ├─ C++ gradient buffer 返回 PyTorch Autograd
  └─ 异步提交 backward 顺序下一层的 BufferB repack

optimizer/post-step
  │
  ├─ 多 rank KT gradient sync
  ├─ grad clip / fused AdamW.step / zero-grad 等 Trainer 操作
  ├─ W_t → W_{t+1}
  └─ _base_weights_dirty = True

第 t+1 轮 forward
  └─ W_{t+1} direct reload/pack → 新 forward BufferB
```

核心一致性约束是：

\[
\boxed{\text{同一次 forward 与 backward 必须使用同一个权重版本 }W_t}
\]

## 10. 当前 profiler 如何对应代码

设置 `KT_SFT_PROFILE=1` 必须发生在创建 C++ MoE 对象之前。之后可通过 Python 的 `collect_kt_sft_profile()`、`format_kt_sft_profile()` 和 `reset_kt_sft_profile()` 收集各层数据。

关键 stage 与代码含义：

| Stage | 对应代码 |
| --- | --- |
| `forward.initial.total` / `forward.recompute.total` | C++ `forward_sft()` 外层；基于 `save_for_backward` 选择标签 |
| `kt.sft.checkpoint_cached_cpu_moe` | checkpoint 重算命中 CPU output 复用；出现该标记时没有再次提交 C++ CPU expert forward |
| `kt.sft.wait_backward_repack` / `kt.sft.submit_backward_repack` | checkpoint/backward 使用共享 BufferB 前的等待，以及对下一层的异步 repack 提交 |
| `backward.down.*` | route-weighted dY、Down base dX 和 Down LoRA |
| `backward.gate_up.*` | Gate/Up base dX 和 LoRA |
| `backward.base_weight_grad.matmat` | dWeight worker-pool 的外层墙钟 |
| `backward.base_weight_grad.worker_cpu.*` | worker 累加的 pack/kernel/store CPU 时间，不是关键路径墙钟 |
| `backward.repack` | forward BufferB → backward BufferB |
| `weights.base_reload.direct_pack` | 当前 BF16 直接从完整权威参数生成 TP forward BufferB |
| `weights.base_reload.partition/forward_pack/cleanup` | 非 direct kernel 的 fallback reload |

Python 还使用 `torch.profiler.record_function` 标出 routing、base weight reload、CPU forward sync、checkpoint recompute、CPU backward、wait/submit repack，便于与 GPU timeline 对齐。

一个已知的标签限制是：C++ 用 `save_for_backward` 区分 initial 和 recompute。fallback checkpoint 的第一次 forward 为 `False`、重算为 `True`，标签成立；但启用 forward reuse 后，第一次 forward 为保存 C++ cache 也会传 `True`，而真正的重算会跳过 C++ forward。非 checkpoint 普通训练初次 forward 同样可能传 `True`。这不影响训练数值，但分析 profile 时不能只按 `forward.initial/recompute` 名称归因，必须结合 `kt.sft.checkpoint_cached_cpu_moe` 和 checkpoint 模式。

## 11. 聚焦测试和验收缺口

本文涉及的两个基线新增或扩展了以下测试：

- [`test_bf16_dweight.cpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/test/test_bf16_dweight.cpp)：以 BF16 输入和 FP32 标量 reference 检查 route/tail case，阈值为 relative L2 `<= 0.01`、cosine `>= 0.999`；`--benchmark` 比较 common driver 与 legacy tile loop，并限制单 tile driver 不得回退超过 5%。
- [`test_raw_bf16_repack.cpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/test/test_raw_bf16_repack.cpp)：检查普通 pack/unpack、转置、直接 BB→BB 转置和 strided source pack 的 BF16 bitwise 一致性。
- [`test_sft_profiler.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/test/per_commit/test_sft_profiler.py)：用 fake MoE 检查 profile 收集、聚合、格式化、reset 和 worker CPU stage 显示。
- [`test_sft_checkpoint_reuse.py`](https://github.com/kvcache-ai/ktransformers/blob/a3d1f15c074a4056cfb252100d1584f52d8ccb27/kt-kernel/test/per_commit/test_sft_checkpoint_reuse.py)：检查纯 Full/纯 LoRA 的复用能力矩阵，并验证非重入 checkpoint 只执行一次 CPU expert forward、仍保持输入与 routing-weight 梯度正确。
- [`test_sft_authoritative_grad.py`](https://github.com/kvcache-ai/ktransformers/blob/a3d1f15c074a4056cfb252100d1584f52d8ccb27/kt-kernel/test/per_commit/test_sft_authoritative_grad.py)：增加异步 repack 与新 CPU pool submit 的事件顺序测试，要求 `repack_submit -> repack_wait -> pool_submit`。这两组 Python 测试在本地合计 `15 passed`。

这些测试说明新组件有明确的局部验证入口，但仍有以下验收缺口：

1. AMX C++ test 只在 `KTRANSFORMERS_CPU_DEBUG` 打开时由 CMake 的 AMX test glob 构建；本轮两个 Python-only commit 没有重新构建或执行这些硬件测试。
2. checkpoint/repack Python test 使用 fake wrapper/task runner，不能覆盖真实 `NumaJobDistributor`、pybind、NUMA 并发和硬件计数器。
3. `test_bf16_dweight` 验证单个 tile driver，不等价于验证 `TP_MOE_SFT::backward()` 的完整 expert/TP pointer、清零和 Autograd 生命周期。
4. 本地仍需在 `a3d1f15` 保存并复现 Full、LoRA-only、Hybrid/Router、reentrant/non-reentrant checkpoint 和多 rank 真实模型回归；聚焦 Python 测试不能单独证明当前 head 的端到端行为。
5. 需要以 PyTorch reference 对比 forward、dX、selected router gradient 和三组 dW，并检查多 step loss、grad norm、NaN/Inf、checkpoint 保存/恢复。
6. 性能结论必须基于最新 head 的同配置 A/B；LoRA forward reuse 从代码上消除一次 CPU forward，但仍需用 CPU cache 占用、checkpoint wall time 和 TPS 数据量化收益。

因此当前状态应表述为：

> **Full-FT 结构链路完整；PR #2094 已把 checkpoint forward reuse 扩展到纯 LoRA，并修复 reentrant 重算与异步 repack 的提交竞态。相关 Python 回归为 15 passed，真实 NUMA 端到端模式回归和最新 TPS 证据仍需在本地补齐。**

## 参考提交

- [`9dc9d93`：staged SFT profiler](https://github.com/kvcache-ai/ktransformers/commit/9dc9d93b7bf1b00124475cde5ff6cae97741e28d)
- [`c6f4211`：SFT 复用 inference BF16 kernel](https://github.com/kvcache-ai/ktransformers/commit/c6f4211346e1aa7b42f66551c33f9a131a2e4dc3)
- [`109b403`：Full-FT 细粒度 profiler](https://github.com/kvcache-ai/ktransformers/commit/109b403b633865f75c1a86411f3c2749c229725d)
- [`34d2102`：BF16 dWeight driver 与 direct BF16 TP reload](https://github.com/kvcache-ai/ktransformers/commit/34d2102082c69fdbcb49e97fa653ce4352a2cd18)
- [`ea84e6e`：BF16 dWeight AMX benchmark](https://github.com/kvcache-ai/ktransformers/commit/ea84e6edf44d2576dc3078d60f5b2b808b42d095)
- [`273e670`：worker CPU profiler 标签修正](https://github.com/kvcache-ai/ktransformers/commit/273e670890c7b966ce537cd1bb4acf69c8202767)
- [`1e95053`：本文初始 #2086 基线](https://github.com/kvcache-ai/ktransformers/commit/1e95053b15b32e6db8193fd852d62d051c6e7ef5)
- [`4240f6f`：纯 LoRA checkpoint forward reuse](https://github.com/kvcache-ai/ktransformers/commit/4240f6fd44b3175d4dd7fca36609887a2fc1327f)
- [`a3d1f15`：checkpoint 重算与异步 repack 串行化；当前 #2094 head](https://github.com/kvcache-ai/ktransformers/commit/a3d1f15c074a4056cfb252100d1584f52d8ccb27)

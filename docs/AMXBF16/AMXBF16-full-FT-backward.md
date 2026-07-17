# AMX Full-FT Backward 代码设计与内部计时分析

## 1. 文档范围

本文分析 KTransformers 在 Qwen3-30B-A3B Full Fine-Tuning 中的 CPU/AMX MoE expert backward，重点覆盖：

- `gate_proj`、`up_proj`、`down_proj` 基座权重的梯度计算；
- expert activation 梯度如何沿 Down、激活函数、Gate/Up 向前一层传播；
- gradient checkpoint、backward BufferB repack、NUMA 并行和梯度清零的作用；
- 2026-07-16 内部打点运行的耗时分解；
- 当前瓶颈、测量限制和后续优化顺序。

主要测试日志：

```text
/mnt/data2/wbw/FFTtest/Qwen3-30B-A3B/test_log/
  20260716_175359_1gpu_AMX_BF16_FULL
```

测试配置为单卡、AMX BF16、2 NUMA、batch=1、GAS=1、序列长度 4096、15 steps、跳过前 5 个 warmup steps、OMP=96、PyTorch fused AdamW。内部计时模式为 `summary`，覆盖全部 48 层。

## 2. MoE MLP 的 forward 与 backward

对一个 routed expert，忽略 batch 和路由维度后，forward 可写为：

```text
G = X · Wgateᵀ
U = X · Wupᵀ
Z = SiLU(G) ⊙ U
Y = Z · Wdownᵀ
```

其中：

- `X`：expert 输入，形状 `[M, H]`；
- `G/U/Z`：expert 中间结果，形状 `[M, I]`；
- `Y`：expert 输出，形状 `[M, H]`；
- `M`：当前 expert 实际接收的 routed token 数；
- `H=2048`：hidden size；
- `I=768`：完整 intermediate size；2 NUMA/TP 下每个子核使用 `I_local=384`。

Backward 分成两类数学目标。

### 2.1 向前一层传播的 activation 梯度

```text
dZ = dY · Wdown

dG = dZ ⊙ U ⊙ SiLU'(G)
dU = dZ ⊙ SiLU(G)

dX = dG · Wgate + dU · Wup
```

这条路径对应日志中的：

```text
Down dX -> activation backward -> Gate/Up dX
```

最终的 `dX` 会继续传给上一层 Transformer、attention 和 normalization。缺少这条路径，autograd 会在 MoE 层中断。

### 2.2 交给 optimizer 的基座权重梯度

```text
dWgate = dGᵀ · X
dWup   = dUᵀ · X
dWdown = dYᵀ · Z
```

输出布局为：

```text
grad_gate_proj [experts, I, H]
grad_up_proj   [experts, I, H]
grad_down_proj [experts, H, I]
```

这条路径就是 `backward_base_weight_grad()`。它只在 Full/Hybrid FT 中执行；LoRA-only 的 `full_weight_grad=false`，不会计算基座 dW。

## 3. Backward 的代码调用和执行顺序

单层主要调用关系如下：

```text
KTMoEFunction.backward()
│
├─ wait_backward_repack()
│    等待当前层 dX 所需的转置基座权重 BufferB
│
├─ ctx.saved_tensors
│    触发 non-reentrant checkpoint decoder-layer 重算
│
└─ wrapper.backward()
     │
     ├─ grad_output GPU/原设备 -> CPU buffer
     ├─ 创建并提交 CPUInfer backward task
     ├─ CPUInfer sync
     │    │
     │    └─ TP_MOE_SFT::backward()
     │         ├─ 清零临时 buffer 和完整基座梯度
     │         ├─ 两个 NUMA 子核并行 backward
     │         │    ├─ restore routed input
     │         │    ├─ backward_down_amx()
     │         │    ├─ backward_activation()
     │         │    ├─ backward_gate_up_amx()
     │         │    ├─ router gradient
     │         │    └─ backward_base_weight_grad()
     │         ├─ 合并两个 NUMA 的 grad_input
     │         ├─ 合并 router/LoRA partial gradients
     │         └─ final NUMA barrier
     │
     └─ CPU gradients -> autograd 返回值
```

相关实现：

- [`KTMoEFunction.backward()`](../../ktransformers/kt-kernel/python/sft/autograd.py)
- [`AMXSFTMoEWrapper.backward()`](../../ktransformers/kt-kernel/python/sft/base.py)
- [`TP_MOE_SFT::backward()`](../../ktransformers/kt-kernel/operators/moe-sft-tp.hpp)
- [`AMX_SFT_MOE_TP::backward()`](../../ktransformers/kt-kernel/operators/amx/sft_moe.hpp)

## 4. 各核心阶段的实际作用

### 4.1 `checkpoint_recompute`

训练启用了 non-reentrant gradient checkpointing。第一次 forward 不永久保留全部 decoder-layer activation；进入 MoE backward 时，访问 `ctx.saved_tensors` 会触发 checkpoint unpack hook，重新执行该 decoder layer 的 forward。

重算为 C++ backward 恢复：

- expert 输入 `X`；
- router expert ID、position 和 routing weight；
- Gate 输出 `G`；
- Up 输出 `U`；
- 激活后中间值 `Z`；
- 其余 C++ forward cache。

它不是梯度公式本身，而是典型的“内存换计算”：少保存 activation，代价是在 backward 中增加一次 forward-like 重算。

### 4.2 `Down dX`

`backward_down_amx()` 计算：

```text
dZ = dY · Wdown
```

实际工作包括：

1. 根据路由把 token 级 `dY` 分发到对应 expert；
2. 乘上 routing weight；
3. 将动态 `dY` 转成 AMX `BufferA`；
4. 使用预转置的 `down_backward_bb_` 作为 `BufferB`；
5. AMX GEMM 生成 `grad_intermediate=dZ`；
6. 保存一份不可变的 route-weighted `dY`，供稍后的 `dWdown=dYᵀ·Z` 使用。

表格中的 “Down dX” 是 projection 术语；严格说它生成的是 Down projection 输入 `Z` 的梯度 `dZ`，不是整层最终的 `dX`。

### 4.3 `activation_backward`

该阶段根据 forward cache 中的 `G`、`U` 和 Down dX 得到的 `dZ`，计算：

```text
dG = dZ ⊙ U ⊙ SiLU'(G)
dU = dZ ⊙ SiLU(G)
```

当前实现使用 AVX512，一次处理 32 个 BF16 元素，并为不足向量宽度的尾部保留标量路径。它在内部打点中单独记录，稳定合计约 47 ms/step。

### 4.4 `Gate/Up dX`

`backward_gate_up_amx()` 计算：

```text
dXgate = dG · Wgate
dXup   = dU · Wup
dX     = dXgate + dXup
```

Gate/Up dX 使用已经预转置的 `gate_backward_bb_`、`up_backward_bb_`，因此权重一侧可以直接作为 AMX `BufferB`。两个 NUMA 分别计算本地 intermediate slice 对 `dX` 的贡献，顶层 TP wrapper 随后执行 BF16 -> FP32 相加 -> BF16 写回，合并成完整 `grad_input`。

### 4.5 `base-weight dW`

`backward_base_weight_grad()` 计算三组基座权重梯度：

```text
dWgate = dGᵀ · X
dWup   = dUᵀ · X
dWdown = dYᵀ · Z
```

与 dX 不同，dW 的两个 operand 都是当前 step 动态产生的数据，不能像基座权重 BufferB 一样跨 step 长期缓存。因此它需要在本次调用内完成动态 panel packing。

Qwen3-30B-A3B 每 NUMA/TP 分区：

```text
I_local = 384  -> 12 个 i_tile
H       = 2048 -> 64 个 h_tile
tile    = 32 × 32 × 32
experts = 128
```

优化前任务粒度为：

```text
(expert, projection, i_tile, h_tile)
128 × 2 × 12 × 64 = 196,608 tasks/layer/NUMA
```

优化后任务粒度为：

```text
(expert, projection, i_tile)
128 × 2 × 12 = 3,072 tasks/layer/NUMA
```

每个 task 在本线程中循环 64 个 `h_tile`，使固定 panel 得以复用：

- Gate/Up：固定 `i_tile` 的 Gate A、Up A 全 K panel 只打包一次；
- Down：固定 `i_tile` 的 intermediate B 全 K panel 只打包一次；
- 每个 `h_tile` 的 input B panel 由 Gate/Up 共用；
- 线程局部 scratch 按需扩容、跨 task 复用，并对齐到 64 byte。

平均一个 expert 约有：

```text
4096 tokens × top-k 8 / 128 experts ≈ 256 routed tokens
```

即约 8 个 K tile。旧代码每个 K tile 后将 32×32 FP32 C tile 写到内存，并在下一 K tile 前重新加载：

```text
8 store_c + 7 load_c
```

新代码把完整 K reduction 放在：

```cpp
clean_c();
for (int kt = 0; kt < k_tiles; ++kt) {
  load_a_b();
  run_tile();
}
store_c();
```

之间，使 C 在完整 K 循环内常驻 AMX tile 寄存器，最终只执行一次 `store_c()`。

Gate 和 Up 各自占用 `GemmKernel224BF` 的完整四个 C tile，无法同时常驻。因此使用两个完整 K pass：先完整计算 Gate 并写回，再完整计算 Up。该设计用少量本地循环控制和额外 B reload，换取显著减少的 C tile load/store。

详细 diff 设计见 [`AMXBF16-full-FT-gitdiff.md`](AMXBF16-full-FT-gitdiff.md)。

### 4.6 `cpp_grad_buffer_clear`

顶层 TP backward 在进入 NUMA 计算前清零：

- 各 NUMA 的临时 `grad_input`；
- 各 NUMA 的 router gradient partial buffer；
- LoRA partial buffer，当前 Full rank=0 时基本为空；
- 完整 `grad_gate_proj`；
- 完整 `grad_up_proj`；
- 完整 `grad_down_proj`。

基座梯度清零是主要部分。dW 只为当前激活 experts 创建任务；如果某个 expert 当前 step 没有 token而梯度 buffer 未清零，它可能保留上一 step 的梯度并被 optimizer 错误更新。

代码把 buffer 切成 2 MiB chunk，通过共享 WorkerPool 并行 `memset`。全部 CPU expert BF16 梯度约为：

```text
28.991B parameters × 2 bytes ≈ 54 GiB/step
```

后续可以尝试在当前层全部 experts 都激活时跳过基座梯度预清零，或只清未激活 expert；但必须验证 inactive expert、GAS、多 microbatch 和 PyTorch `.grad` 累积语义。

### 4.7 `backward_repack_wait`

该阶段不是等待 `base-weight dW` 内的动态 panel packing。

Forward 和 dX 需要不同权重布局：

```text
forward: X  · Wᵀ
dX:      dY · W
```

因此 dX 需要：

```text
gate_backward_bb_
up_backward_bb_
down_backward_bb_
```

即基座权重的转置 AMX `BufferB`。

配置 `share_backward_bb=true` 时，每层不永久保存一整套 backward BufferB，而是每 NUMA 共用 shared backward-BB pool。完成当前层 backward 后，Python 为反向顺序中的下一层异步提交 repack，希望和 GPU attention backward 或其他工作重叠；进入下一层前调用 `wait_backward_repack()`。

因此 `repack_wait` 只记录异步 repack 未被隐藏的尾部等待：

- repack 已完成时，wait 接近零；
- repack 未完成时，主 backward 线程通过 `join()` 阻塞；
- 它服务于 Down/Gate/Up dX，不服务于 dW；
- 它也不同于 optimizer 后的 `update_base_weights/requant`。

## 5. 内部计时设计和口径

Recorder 使用 Python `perf_counter_ns` 和 C++ `steady_clock` 测量已有函数/编排边界，不在 AMX tile 或 WorkerPool task 内逐次打点。

一个 step 内：

- 记录 48 次 layer backward；
- Python 层将同名字段在 48 层上求和；
- C++ NUMA 子阶段对两个 NUMA 取 `max`，近似关键路径；
- `summary` 模式只输出 step aggregate，不输出每层、每 NUMA trace；
- 不新增 CUDA synchronization。

必须遵守以下口径：

1. `outer_backward`、`autograd_total`、`wrapper_total`、`cpp_total` 和 `numa_critical_total` 是嵌套视图，不能相加。
2. `numa_critical_*` 每个字段独立对 NUMA 取最大值，最慢 NUMA 可能因阶段而变化，所以子阶段最大值之和可能略大于 `numa_critical_total`。
3. GPU-facing Python 阶段是 host-wall 边界；异步 CUDA 工作可能在后续已有同步点才被观察到。
4. `base_weight_grad_ns` 覆盖整个 dW 函数，但 AMX tile、packing、task pickup 和缓存行为在该边界内部仍是不透明的。

原始打点报告：

- [`backward_timing.md`](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/backward_internal/backward_timing.md)
- [`backward_step.csv`](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/backward_internal/backward_step.csv)
- [`backward_timing.json`](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/backward_internal/backward_timing.json)

## 6. 稳定 step 的内部耗时

稳定区间为 steps 6-15，共10步。`outer_backward=6.946782 s` 与 step probe 的 `backward_sec=6.947 s` 一致；`cpp_total=3.844781 s` 与 `cpuinfer_sync=3.846785 s` 只差约2 ms，说明边界计时内部自洽。

```text
outer backward                         6.947 s
├─ 非 KT residual                     0.463 s
└─ 48 层 KT autograd                  6.483 s
   ├─ checkpoint recompute            2.079 s
   ├─ backward repack wait            0.313 s
   └─ wrapper backward                4.078 s
      ├─ grad_output -> CPU           0.102 s
      ├─ C++/CPUInfer                 3.845 s
      │  ├─ 全量梯度 buffer 清零      0.340 s
      │  ├─ NUMA 并行 backward        3.479 s
      │  │  ├─ base-weight dW         2.036 s
      │  │  ├─ Down base dX           0.665 s
      │  │  ├─ Gate/Up base dX        0.643 s
      │  │  └─ restore/activation/
      │  │     router/prepare         约 0.18 s
      │  └─ merge/barrier             约 0.025 s
      └─ return gradients             0.124 s
```

| 阶段 | 每 step | 每层均值 | 占 outer backward | 作用 |
|---|---:|---:|---:|---|
| base-weight dW | 2.036 s | 42.41 ms | 29.3% | 生成 `dWgate/dWup/dWdown` |
| checkpoint 重算 | 2.079 s | 43.32 ms | 29.9% | 恢复 forward activation/cache |
| Down dX | 0.665 s | 13.86 ms | 9.6% | 计算 `dZ=dY·Wdown` |
| Gate/Up dX | 0.643 s | 13.39 ms | 9.3% | 计算并合成 `dX` |
| 全量梯度清零 | 0.340 s | 7.08 ms | 4.9% | 消除旧 step/inactive expert 残留 |
| backward repack wait | 0.313 s | 6.52 ms | 4.5% | 等待 dX 所需转置 BufferB |

`base-weight dW` 约占 NUMA 并行墙钟的 58.5%，占整个 C++ backward 的 53.0%，但只占完整 outer backward 的 29.3%。因此它仍是最大的 CPU 子函数，却不能代表完整 backward 的全部成本。

稳定性：

- outer backward 范围：6.904-6.996 s，CV约0.38%；
- base-weight dW 范围：2.026-2.051 s，CV约0.34%；
- repack wait 范围：0.287-0.343 s，CV约4.4%，相对更容易受流水重叠影响。

## 7. dW 与 dX 的有效吞吐对比

理论分析给出的 CPU expert backward 为 29.687 TFLOP/step。线性层的 dX 和 dW 各约占一半，即 14.843 TFLOP。

据此估算：

```text
dW effective throughput
  ≈ 14.843 TFLOP / 2.036 s
  ≈ 7.29 TFLOPS

dX effective throughput
  ≈ 14.843 TFLOP / (0.665 + 0.643) s
  ≈ 11.35 TFLOPS
```

大致相同的有用矩阵 FLOPs 下，dW 用时是 dX 的 1.56 倍。主要原因是：

- dX 权重 BufferB 已预转置，可直接复用；
- dW 两个 operand 均为动态数据，需要每 step packing；
- routed expert 的 `M` 较小且不规则；
- dW 还必须写出完整 BF16 参数梯度；
- 尾部 expert-token 块需要 padding。

这里的 TFLOPS 是模型“有用 FLOPs”估算，不等于硬件实际 AMX 指令吞吐，也没有计入 padding、packing、激活、路由和数据搬运。要判断 AMX feed 是否仍是主要限制，还需要 `perf` 或 VTune 的 AMX、cycles、cache 和 top-down 计数器。

理论 FLOPs 文件：[`flops_analysis.json`](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/flops_analysis.json)。

## 8. 首步异常：梯度 buffer first-touch

首步内部数据：

```text
cpp_grad_buffer_clear = 9.165 s
outer_backward        = 18.212 s
```

稳定阶段：

```text
cpp_grad_buffer_clear = 0.340 s
outer_backward        = 6.947 s
```

清零相差约27倍，而首步 NUMA AMX backward 仅从稳定的约3.48 s增加到约3.74 s。因此首步 backward 异常主要不是 dW kernel，而是约54 GiB大梯度内存的首次物理页分配、缺页和 NUMA first-touch。

稳定清零的等效带宽约为：

```text
54 GiB / 0.340 s ≈ 159 GiB/s
```

首步不应计入稳定 AMX kernel 性能；但如果关注首次迭代延迟，应单独优化 Parameter/gradient 的预分配和 NUMA first-touch。

## 9. 与前一组无内部打点运行的比较

对照运行：

```text
20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA
```

Full 训练参数相同；本次额外启用了 backward summary timing，且只执行 Full。

| 指标 | 前一组 | 本次 | 变化 |
|---|---:|---:|---:|
| backward | 6.792 s | 6.947 s | +0.155 s / +2.28% |
| forward | 1.346 s | 1.530 s | +0.184 s / +13.68% |
| requant | 5.732 s | 6.746 s | +1.014 s / +17.69% |
| optimizer | 1.925 s | 1.757 s | -0.168 s / -8.74% |
| step | 16.140 s | 17.362 s | +1.222 s / +7.57% |

不能把 backward 多出的155 ms直接判定为 recorder overhead。完整 step 的主要回退来自不在 recorder 范围内的 requant 和 forward；同时 recorder 不新增 CUDA synchronization。要测量打点本身的代价，应使用同一二进制做至少三组：

```text
off -> summary -> off -> summary
```

## 10. 剩余优化空间：先区分“必要计算”和“可避免开销”

本次稳定 step 为17.362 s，TPS为235.92：

| 阶段 | 时间 | step 占比 |
|---|---:|---:|
| backward | 6.947 s | 40.0% |
| requant/update_base_weights | 6.746 s | 38.9% |
| forward | 1.530 s | 8.8% |
| optimizer | 1.757 s | 10.1% |
| 其他 | 0.382 s | 2.2% |

Backward 内部已经识别出的四个大项为：

```text
dW 2.036 + checkpoint 2.079 + clear 0.340 + repack wait 0.313
= 4.768 s = outer backward 的 68.6%
```

这个68.6%是“已定位到可研究对象的时间”，不是可直接删除的时间：dW是 Full FT 必需计算；checkpoint 用计算换显存；清零涉及跨 microbatch 的梯度语义；`repack_wait` 也只在能把工作提前并成功重叠时才能隐藏。

### 10.1 单项 Amdahl 上限

以下估算只改变一个阶段，其他阶段保持本次实测值：

| 假设 | 节省时间 | 新 backward | 估算 TPS |
|---|---:|---:|---:|
| dW 耗时降低25% | 0.509 s | 6.438 s | 243.0 |
| dW 达到当前 dX 的有效吞吐 | 0.728 s | 6.219 s | 246.2 |
| dX 合计耗时降低20% | 0.262 s | 6.685 s | 239.5 |
| checkpoint 重算降低25% | 0.520 s | 6.427 s | 243.2 |
| 完全跳过本次全量清零 | 0.340 s | 6.607 s | 240.6 |
| 完全隐藏 repack wait | 0.313 s | 6.634 s | 240.3 |
| 完全删除 dW（不现实上限） | 2.036 s | 4.911 s | 267.3 |
| 完全删除 checkpoint（不现实上限） | 2.079 s | 4.868 s | 268.0 |

所以“还有空间”的答案是肯定的，但只继续打磨 dW，端到端 TPS 的合理单项目标更接近246 tok/s，而不是300 tok/s。Backward 与 requant 已经几乎并列，后续应同时优化两者。

### 10.2 分层判断

| 层级 | 目标 | 从本次打点可见的空间 | 判断 |
|---|---|---:|---|
| P0 | dW kernel/packing | 0.5–0.8 s | 最确定的计算内空间；dW有效吞吐7.29 TFLOPS，仅为当前 dX 的约64%。 |
| P0 | selective checkpoint | 0.5–1.0 s（若减少25%–50%重算） | 绝对时间最大，但需要额外保存 activation，收益取决于内存预算。 |
| P1 | 条件清零 | 上限0.340 s | 稳定且容易量化；必须先证明本 step 会完整覆盖相应梯度。 |
| P1 | repack 流水 | 上限0.313 s | 当前看到的是尾部等待；总工作量未知，先找提交时机和 worker-pool 竞争。 |
| P1 | dX kernel | 0.13–0.26 s（若降低10%–20%） | 已明显快于 dW，优先级低一档。 |
| P2 | host copy/返回梯度 | `101.5 + 123.9 = 225.4 ms` | 可能与异步 CUDA 在既有阻塞点发生归因迁移，需 trace 后再改。 |
| P2 | `non_kt_residual` | 0.463 s | 仍是黑盒边界，不能把全部时间视为 CPU 可优化开销。 |

表中的区间是规划量级，不是已测得收益；多项同时修改时会竞争同一内存带宽和线程池，不能简单相加。作为工程目标，可先验证 backward 从6.95 s降到约5.5–6.0 s；若不改 requant，对应完整 step 约15.9–16.4 s，即约250–257 tok/s。

## 11. 推荐的验证和优化顺序

### 11.1 先拆 dW，而不是继续凭循环层数判断

当前计时只有整个 `backward_base_weight_grad()` 边界。下一步在少量 trace 层中区分：

- Gate/Up dW 与 Down dW；
- dynamic A/B panel packing；
- AMX compute；
- FP32 C 到 BF16 的 conversion/store；
- 每个 worker 的 task 数、有效 token 数和空转时间。

不应给每个3,072-task strip都调用高开销墙钟。使用线程局部 cycle accumulator，退出 WorkerPool 后归约；同时采集 AMX tile 指令、cycles、IPC、cache miss、内存带宽以及24/48 worker扩展性。只有当 AMX compute 占比高且 tile 指令吞吐低时，才继续改 kernel；如果 packing/store 占比更高，应优先改数据布局和复用。

### 11.2 用 trace 判断 NUMA 和路由不均衡

选择层0、23、47记录两个 NUMA 的真实时间、active experts、每 expert routed tokens、`M` 分布和关键路径差异。当前 summary 对每个子阶段取 `max(NUMA0, NUMA1)`，无法判断慢的是固定 socket还是阶段间交替，子阶段最大值也可能来自不同 NUMA。

### 11.3 条件清零必须先证明覆盖语义

当128个 experts全部激活且本 step 的 dW会覆盖三组完整梯度 tensor 时，可以评估跳过 base-gradient 预清零；否则只清 inactive expert。回归必须覆盖非全激活路由、GAS>1、多 microbatch、TP part 0/1、非零 expert ID，以及连续 step inactive/active 交替。

### 11.4 checkpoint 和 repack 分别处理

- checkpoint：评估只保存 MoE 所需输入/路由 cache、保留其他模块 checkpoint 的选择性方案，并记录额外常驻内存；
- repack：检查 backward-BB repack 是否可更早提交，以及 shared repack pool 与 checkpoint recompute 是否争用 CPU worker；
- 对 `repack_wait` 波动层启用 trace，不能把312.9 ms等待直接当成312.9 ms可删除计算。

### 11.5 每项优化都做同二进制交替对照

至少执行三组稳定区间，并使用：

```text
baseline -> candidate -> baseline -> candidate
```

同时报告 backward 子项、完整 step 和 TPS。内部 recorder 自身的开销则使用同二进制 `off -> summary -> off -> summary` 测量。

## 12. 报告口径注意事项

1. 本次端到端性能采用 step probe 的 `17.362 s/step、235.92 tok/s`，而不是 `summary.md` 中基于 tqdm 日志时间戳的 `211.6 tok/s`。
2. `step_timing.md` 的 phase legend 仍残留 “CPU DeepSpeedCPUAdam” 文案；训练日志实际显示 `AdamW` 和 `AcceleratedOptimizer`，当前运行没有使用 DeepSpeedCPUAdam。
3. `numa_critical_*` 是关键路径近似，不是两个 NUMA 时间之和；各子阶段的 max 也不能保证来自同一个 NUMA。
4. 整体 backward 的理论有效吞吐不等于 AMX tile 利用率；GPU non-expert、CPU expert、checkpoint 和同步位于不同嵌套边界。
5. 本次 timing run 相比前一无内部打点 run 的 step 慢7.57%，主要变化落在 requant 和 forward；在同二进制交替实验前，不能归因给 recorder。

## 13. 总体结论

1. Backward 仍有明确优化空间：约68.6%的时间已定位到 dW、checkpoint、清零和 repack wait，但其中只有一部分可真正消除。
2. dW为2.036 s/step，是最大 CPU 子函数；它与 dX具有相近的有效 FLOPs，但耗时为dX的1.56倍。若达到当前 dX 的有效吞吐，可节省约0.728 s/step。
3. checkpoint重算为2.079 s，是最大的策略性空间；它不能无条件删除，必须用额外 activation 内存换取时间。
4. 条件清零和 repack流水各有约0.3 s理论上限，适合作为低风险、可独立验收的第二梯队。
5. 首步慢主要由约54 GiB梯度 buffer首次触页导致，不应误判为稳定 AMX dW回退。
6. 可把 backward 5.5–6.0 s作为下一阶段的验证目标，而不是承诺；若 requant不变，完整 TPS大约仍受限在250–257 tok/s。
7. 完整 step 中 requant为6.746 s，已与 backward并列。要继续显著提高 TPS，必须并行推进 backward 和 requant，而不是只增加 AMX dW 内的循环级优化。

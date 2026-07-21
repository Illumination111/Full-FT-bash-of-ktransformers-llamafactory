# AMX Full-FT Backward：当前实现、历史计时与后续分析边界

更新时间：2026-07-21

## 1. 文档范围与当前代码基线

本文分析 Qwen3-30B-A3B Full Fine-Tuning 中由 KTransformers CPU/AMX MoE experts 执行的 backward，覆盖 activation gradient、base-weight gradient、TP/NUMA merge、backward BufferB repack 和 staged profiling。

当前代码基线已经从历史的 `f209878` 更新为：

```text
ktransformers/fullft-development
HEAD = 1e95053b15b32e6db8193fd852d62d051c6e7ef5
PR   = kvcache-ai/ktransformers#2086
```

本地 tracked 工作树与 GitHub PR head 一致。2026-07-16 的性能日志产生于 `f209878` 或其本地 timing 工作树，仍可作为历史热点证据，但不能替代 `1e95053` 的重新构建和训练验收。

2026-07-20 新增了一次完整 Full-FT 内部打点运行，但运行时版本核验表明它仍不是 `1e95053`：`Kllama` 环境中的 `backward_timing.py` 和 `autograd.py` 与 `stash@{0}` 对应文件的 SHA-256 完全一致，C++ extension 也是 2026-07-16 构建。因此该运行是更新、更稳定的 **`f209878 + KT_BACKWARD_TIMING` 历史证据**，而不是 current head 验收。

## 2. 数学目标

单个 routed expert 的 forward：

```text
G = X · Wgateᵀ
U = X · Wupᵀ
Z = SiLU(G) ⊙ U
Y = Z · Wdownᵀ
```

activation backward：

```text
dZ = dY · Wdown
dG = dZ ⊙ U ⊙ SiLU'(G)
dU = dZ ⊙ SiLU(G)
dX = dG · Wgate + dU · Wup
```

Full/Hybrid FT 的 base-weight gradient：

```text
dWgate = dGᵀ · X
dWup   = dUᵀ · X
dWdown = dYᵀ · Z
```

完整梯度布局：

```text
grad_gate_proj [experts, I, H]
grad_up_proj   [experts, I, H]
grad_down_proj [experts, H, I]
```

Qwen3-30B-A3B 使用 `H=2048`、完整 `I=768`；2 NUMA/TP 时每个子核处理 `I_local=384`。LoRA-only 的 `full_weight_grad=false`，不会生成三组 base dW，但仍执行公共 base dX。

## 3. 当前 backward 调用顺序

```text
KTMoEFunction.backward()
│
├─ wait_backward_repack()
├─ ctx.saved_tensors
│    └─ non-reentrant checkpoint 重算 decoder layer
├─ wrapper.backward()
│    ├─ grad_output -> CPU buffer
│    ├─ CPUInfer submit/sync
│    │    └─ TP_MOE_SFT::backward()
│    │         ├─ buffer clear
│    │         ├─ NUMA-local backward 并行
│    │         │    ├─ cache restore
│    │         │    ├─ Down base dX
│    │         │    ├─ activation backward
│    │         │    ├─ Gate/Up base dX
│    │         │    ├─ router gradient
│    │         │    └─ BF16 base dW
│    │         └─ grad_input/LoRA/router merge
│    └─ CPU gradients -> output device
└─ submit next backward repack
```

主要实现：

- `kt-kernel/python/sft/autograd.py`
- `kt-kernel/python/sft/base.py`
- `kt-kernel/operators/moe-sft-tp.hpp`
- `kt-kernel/operators/amx/sft_moe.hpp`
- `kt-kernel/operators/amx/la/bf16_dweight.hpp`

## 4. 关键阶段

### 4.1 Checkpoint recompute

访问 `ctx.saved_tensors` 触发 non-reentrant checkpoint unpack hook，重新生成 expert 输入、路由、Gate/Up output、activation intermediate 和 C++ forward cache。它用计算换 activation 内存，不是可无条件删除的冗余。

当前 Python 代码同时保留 `torch.profiler.record_function` 边界；C++ profiler 还区分 initial forward 与 recompute forward。

当前配置中，LLaMA-Factory 默认的 `disable_gradient_checkpointing=false` 会调用
`model.gradient_checkpointing_enable(use_reentrant=false)`；FSDP2 也会把 reentrant 模式强制关闭。随后每个
`Qwen3MoeDecoderLayer.__call__()` 都由 `torch.utils.checkpoint.checkpoint()` 包住整个 decoder layer。

开启 checkpoint 时，一个 layer 的生命周期如下：

1. 初始 forward 仍执行 RMSNorm、attention、第二个 RMSNorm、router 和 KT MoE，但 non-reentrant checkpoint
   的 saved-tensor hook 用 holder 代替这些算子原本要保存的内部 activation；主要只留下 layer 输入等重算入口。
2. `KTMoELayerWrapper` 检测到 `first_forward` hook，把 C++ 提交参数改为
   `save_for_backward=false`。CPU expert 会算出 forward output，但不保留 input、Gate/Up output、activation
   intermediate 和 down output，避免所有 layer 同时持有大缓存。
3. `KTMoEFunction.forward()` 保存一个标量 sentinel。该 sentinel 也被 checkpoint hook 接管。
4. backward 逆序到该 MoE 时，`KTMoEFunction.backward()` 先访问 `ctx.saved_tensors`。解包 sentinel 会触发
   整个 decoder layer 的重算；由于 sentinel 位于 MoE forward 内，重算至少会再次经过 attention、norm、router
   和 KT MoE。
5. 重算阶段 hook mode 不再是 `first_forward`，所以 KT 以 `save_for_backward=true` 再执行一次 CPU expert
   forward，并填充 C++ backward cache。随后 `wrapper.backward()` 立即消费并 pop 这个 cache。

这种严格的“重算一层、立刻反传一层”顺序使两个 NUMA 节点各自只需要一份跨 48 层共享的 cache pool；源码也明确
标注 `share_cache_pool` 仅在 gradient checkpoint 下安全。

关闭 checkpoint 后，decoder layer 不再经过 checkpoint wrapper，第一次 forward 就建立普通 PyTorch autograd
图并保留 attention、RMSNorm、router 等 backward 所需 activation。`_checkpoint_hook_mode()` 返回 `none`，所以
KT 第一次 CPU expert forward 直接使用 `save_for_backward=true` 并保存 C++ cache。backward 访问 sentinel 时不再
触发重算，`kt.sft.checkpoint_recompute` 只剩近似空的取值开销，随后直接消费初始 forward 留下的 cache。

因此完整关闭不能只删除 `ctx.saved_tensors` 或只跳过重算。正确的配置/生命周期组合是：

- 训练配置设置 `disable_gradient_checkpointing: true`；
- 保持训练 KV cache 关闭；LLaMA-Factory 当前会在 trainable 模式中自动设置 `config.use_cache=false`；
- KT 必须使用每层独立 cache，即 `kt_share_cache_pool=false`。当前配置处理在 checkpoint 关闭时会回落到该值，但
  仍应在启动日志中确认所有 layer 都打印 `share_cache_pool false`；若仍为 true，后层 forward 会覆盖前层 cache，
  产生错误梯度而不只是内存不足；
- 当前 benchmark 每层每个 step 只有一份未消费 cache，可把 `kt_max_cache_depth` 从 2 降为 1 来减半 cache
  内存；必须用 cache stack 平衡检查及短训验证该假设。

### 4.2 Down、activation、Gate/Up dX

- Down：`dZ=dY·Wdown`，并保存稍后计算 `dWdown` 所需的 route-weighted `dY`；
- activation：以缓存的 `G/U` 计算 `dG/dU`；
- Gate/Up：`dX=dG·Wgate+dU·Wup`；
- 两个 NUMA 计算各自 intermediate slice 对 `dX` 的贡献，顶层再合并为完整 `grad_input`。

dX 的权重一侧可以使用提前准备的转置 base-weight BufferB，因此它与两个 operand 都是动态数据的 dW 有不同的数据复用条件。

### 4.3 Buffer clear

TP wrapper 在 NUMA 计算前清零临时 grad buffer、router/LoRA partial buffer 和完整 base-gradient tensor。约 28.991B CPU expert 参数对应约 54 GiB BF16 gradient。

只有当前 step 确实覆盖完整目标区域时才能跳过或缩小清零。任何优化都必须覆盖 inactive expert、GAS、多 microbatch、TP slice 和跨 step 残留语义。

### 4.4 Backward repack

Forward 使用 `X·Wᵀ`，dX 使用 `dY·W`，因此需要不同的 BufferB 布局。`share_backward_bb=true` 时各层不永久保存整套 backward BufferB，而是在共享 pool 中异步准备下一层。

`wait_backward_repack()` 只表示没有被前序工作隐藏的尾部等待，不是 base dW 动态 panel packing，也不同于 optimizer 后的 base-weight reload。

## 5. 当前 BF16 dWeight 实现

### 5.1 `f209878` 的历史原型

旧原型在 `backward_base_weight_grad()` 内证明了三项方向：

1. 任务从 `(expert, projection, i_tile, h_tile)` 合并为 fixed-`i_tile` strip；
2. Gate/Up 或 Down 的固定动态 panel 在线程局部 scratch 中预打包和复用；
3. FP32 C tile 跨完整 K reduction 常驻 AMX tile，最后只 store 一次。

它把每层、每 NUMA 的理论任务量从：

```text
128 × 2 × 12 × 64 = 196,608
```

降到：

```text
128 × 2 × 12 = 3,072
```

历史 Full/LoRA A/B 证明该方向有效，但该函数内手写实现不再是新 head 的完整描述。

### 5.2 `1e95053` 的 current driver

今日 GitHub head 把 BF16 dW 抽为 `BF16DWeightKernel`，并复用 inference 的 `GemmKernel224BF16` driver：

- 只为 active experts 创建任务；
- Gate/Up 共用 input panel；
- Down fixed-`i_tile` 复用 intermediate panel；
- 线程局部 scratch 按需扩容并保持对齐；
- 完整 K reduction 由通用 AMX/AVX512 driver 完成；
- Gate/Up、Down、pack A、pack B、kernel 和 store 分别累计 timing；
- 测试覆盖 reference、tail shape、Qwen shape 和 common-driver 性能门槛。

当前 staged profiler 中：

```text
backward.base_weight_grad
backward.base_weight_grad.offsets
backward.base_weight_grad.matmat
backward.base_weight_grad.worker_cpu.pack_a
backward.base_weight_grad.worker_cpu.pack_b
backward.base_weight_grad.worker_cpu.kernel_gate_up
backward.base_weight_grad.worker_cpu.kernel_down
backward.base_weight_grad.worker_cpu.store
```

`worker_cpu.*` 是各 worker 的累计 CPU 时间，不是外层墙钟；不同 worker 并行执行时不能与 `backward.base_weight_grad` 直接相加或比较占比。

## 6. Base-weight reload 的变化

Optimizer 更新权威 CPU BF16 Parameter 后，下一次 forward 前仍需生成 AMX BufferB。新 head 可以从完整 Parameter 按 TP/NUMA stride 直接 pack，减少旧路径的临时分片分配和 memcpy，并分别记录：

```text
weights.base_reload
weights.base_reload.partition
weights.base_reload.forward_pack
weights.base_reload.direct_pack
weights.base_reload.backward_pack
weights.base_reload.cleanup
```

Direct pack 没有消除完整 BufferB pack，也不能在新训练结果出现前宣称 requant/reload 已经被消除。

## 7. 当前 profiler 口径

`1e95053` 已提交的唯一 C++ timing 源是 `SFTProfiler`：

- `KT_SFT_PROFILE=1` 必须在 MoE 对象创建前设置；
- C++ 使用 `steady_clock` 和原子累计计数；
- wrapper 与 `tp.<index>` scope 分别表示顶层和 NUMA-local 子核；
- `get_profile_stats(reset=False)` 获取累计快照；
- Python `collect_kt_sft_profile()` 聚合各层，`format_kt_sft_profile()` 输出表格；
- PyTorch profiler 另有 checkpoint、CPU forward/backward 和 repack 边界。

以下字段是嵌套视图，不能相加：

```text
wrapper.backward.total
wrapper 下的 backward 子阶段
tp.<index>.backward.total
tp.<index> 下的子阶段
worker_cpu.* 累计时间
```

旧 `KT_BACKWARD_TIMING` summary/trace recorder 已从当前源码工作树移出并保存在 stash，但 2026-07-20 运行所用的 `Kllama` site-packages 和已构建 extension 仍是该旧 timing 版本。它的逐 step/layer CSV/JSON 能力尚未合入新 profiler；历史输出仍可读，但不能和新字段逐列混合。

## 8. 历史性能证据

### 8.1 `f209878` 前后 A/B

配置相同的相邻 `full -> lora` 会话：单卡、AMX BF16、2 NUMA、batch=1、GAS=1、seq=4096、15 steps、warmup 5、OMP=96、PyTorch fused AdamW。

| Full 稳定指标 | 修改前 | `f209878` 后 | 变化 |
|---|---:|---:|---:|
| Step | 19.25197 s | 16.13997 s | -16.16% |
| TPS | 212.76 | 253.78 | +19.28% |
| Backward | 9.28085 s | 6.79183 s | -26.82% |
| Optimizer | 1.79075 s | 1.92533 s | +7.52% |
| Requant | 6.32860 s | 5.73156 s | -9.43% |

同期 LoRA-only backward 只变化 -2.74%，支持主要收益来自 Full-only dW 路径。Forward/requant 也存在运行间变化，所以不能把完整 TPS 增益全部归给 dW。

### 8.2 旧内部 timing runs

两次使用旧 recorder 的 Full-FT 运行配置一致，均统计稳定 steps 6–15：

| 运行 | Forward | Outer backward | Backward/Forward | Step | TPS |
|---|---:|---:|---:|---:|---:|
| `20260716_175359...` | 1.530 s | 6.947 s | 4.54× | 17.362 s | 235.92 |
| `20260720_130854...` | 1.408 s | 6.671 s | 4.74× | 17.032 s | 240.49 |

最新 `20260720_130854...` 的稳定 backward 分解如下。顶层 `outer = autograd + non-KT residual`，而 wrapper、C++ 和 NUMA 字段是嵌套视图，不能跨层相加：

```text
outer backward                 6.671 s
├─ autograd total              6.128 s
│  ├─ checkpoint recompute     1.991 s
│  ├─ wrapper total            3.886 s
│  ├─ backward repack wait     0.238 s
│  └─ submit/other             0.013 s
└─ non-KT residual             0.542 s

wrapper/C++ 嵌套视图：
  C++ total                    3.720 s
  ├─ NUMA parallel wall        3.359 s
  ├─ gradient buffer clear     0.336 s
  └─ prepare/merge/barrier     0.025 s

NUMA critical-path 近似：
  base-weight dW               1.999 s
  Down dX                      0.637 s
  Gate/Up dX                   0.635 s
  activation/router/restore    约 0.161 s
```

最新稳定 backward 范围为 6.635–6.710 s，forward 范围为 1.383–1.429 s，说明目标缺口不是随机抖动。两次 timing run 仍一致地把 checkpoint 和 dW 指向最大子项，也显示首步清零受约 54 GiB gradient first-touch 强烈影响。但二者都早于当前 8 个 commit，不能用于宣称 `BF16DWeightKernel` 或 direct reload 的当前耗时。

### 8.3 `backward <= 3 × forward` 的预算

以最新同一步计时口径计算：

```text
forward                     = 1.408484 s
3 × forward                 = 4.225452 s
backward                    = 6.670751 s
必须减少                    = 2.445299 s（backward 的 36.66%）
```

仅关闭 gradient checkpointing 的算术上界是移除 1.990752 s recompute，使 backward 降至约 4.680 s，即 3.32× forward，**仍不能单独达标**。在此基础上，即使完全消除 0.335884 s 的 buffer clear，仍为 4.344 s（3.08×），还差约 0.119 s。

因此，一个可解释但尚未实测的达标预算是：关闭 checkpointing，加上消除完整 buffer clear，再隐藏约一半 repack wait；结果约为 4.225 s。等价地，最后 0.119 s 也可由约 6% 的 dW 墙钟优化补足。这只是阶段预算，不是性能预测：关闭 checkpointing 可能增加 GPU activation 峰值，清零不能在未证明覆盖语义前删除，把 repack 移到 reload 只会改变计时归属而不一定改善 step。

该运行的 `monitor.csv` 记录 GPU 峰值 40,528 MiB、总容量 49,140 MiB，即 39.58 GiB / 47.99 GiB，真实表面余量
是 8.41 GiB。旧写法把 MiB→GiB 的峰值与十进制总容量混在一起，因而把余量高估为 9.57 GB。

关闭 checkpoint 的 host cache 增量可以直接由当前 C++ 分配公式计算。Full-FT 的 `lora_rank=0`，每个 NUMA、
每个 cache slot 为：

```text
input          = 4096 × 2048 × BF16                  =  16 MiB
gate/up/inter  = 3 × 4096 × 8 × 384 × BF16          =  72 MiB
down output    = 4096 × 8 × 2048 × BF16             = 128 MiB
合计                                                   216 MiB / NUMA / slot
两 NUMA                                                432 MiB / layer / slot
```

当前 checkpoint + shared pool + depth=2 只占 864 MiB（0.844 GiB）。关闭后若保持 depth=2，48 层独立 pool
合计 40.50 GiB，净增 39.66 GiB（42.58 GB）；若同步设 depth=1，则合计 20.25 GiB，净增 19.41 GiB
（20.84 GB）。以观测到的进程树 RAM 峰值 1056.99 GB 粗加，预计分别约为 1099.57 GB 或 1077.83 GB；这不含
allocator 碎片和其他生命周期变化，但相对于 2 TB 主存并非容量风险。

GPU activation 没有同样精确的静态值，因为它取决于 CUDA SDPA 后端、autograd 保存集合、梯度清零方式和 allocator
复用。按当前 BF16、SDPA/GQA、`B=1,S=4096,H=2048,Q=32 heads,KV=4 heads,D=128` 估算，每层完整保存集合约
0.24–0.28 GiB；扣除 checkpoint 已保存的 layer boundary 后，48 层常驻 activation 集合预计净增约
11–13 GiB。由于 forward 结束时参数梯度通常尚未重新分配，且 backward 中 activation 释放与梯度生成相反，
CUDA 峰值的实际净增可能只有约 6–10 GiB；若 SDPA 回退、K/V 被 repeat 或 allocator 碎片明显，则可能接近
11–15 GiB。现有 8.41 GiB 余量处在该区间中间，因此 seq=4096 的 no-checkpoint 可能刚好运行，也可能 OOM，不能
在短程实测前承诺。验收至少要记录 `torch.cuda.max_memory_allocated/reserved`，而不能只依赖 2.4 秒一次的 NVML
采样。

如果必须保留完整 checkpointing，则需要同时大幅优化 dW、dX、重算、清零和 CPU/GPU 重叠，达到 3× 的风险明显
更高，不应承诺由单个 AMX kernel 修改完成。

### 8.4 与 DeepSpeed 的 backward/optimizer 计时边界

同样使用单卡、BF16、batch=1、GAS=1、seq=4096 和稳定 steps 6–15 的 DeepSpeed ZeRO-3 CPU Offload 对照记录：

```text
step total                         27.2399 s
forward                             3.2919 s
accelerator.backward 计时桶         23.9330 s
显式 Trainer optimizer 计时桶          0 s
```

这里的 0 s 不是 CPU optimizer 没有成本。Accelerate 1.11.0 的 `DeepSpeedEngineWrapper.backward()` 先调用 `engine.backward()`，然后在 gradient-accumulation 边界调用 `engine.step()`。本次 GAS=1，所以每一步的 DeepSpeedCPUAdam、ZeRO step、gradient clipping、zero-grad 和 scheduler 都被记在 23.933 s 的 `accelerator.backward` 桶内。日志明确显示 `Adam Optimizer #0 is created with AVX512 arithmetic capability.`，因此不能把外层 optimizer 的 0 解读为真实 CPUAdam 时间。

DeepSpeed 运行的 `wall_clock_breakdown=false`，现有日志无法把 23.933 s 拆成纯 GPU backward、ZeRO gradient partition/offload 和 CPUAdam/step。因此也不能把它与 KT 的 6.671 s 纯 outer backward 直接做倍数比较。

只有粗粒度“forward 之后到权重已可用于下一 step”可以在现有边界上对齐：

```text
DeepSpeed backward + ZeRO + CPUAdam/step                     23.9330 s
KT backward + clip + optimizer + post-optim + base reload    15.6101 s
差值                                                            8.3230 s
```

这个对齐说明 KT 的完整更新路径比该 DeepSpeed 基线短 8.323 s，但不能说明差值是 CPUAdam 单项导致。KT 内部可继续拆成 6.671 s backward、2.257 s AdamW、0.332 s post-optim 和 6.327 s base reload。

2026-07-21 已在 FFTtest 的 `step_timing_probe.py` 落地 DeepSpeed 运行时包装：分别累计 `DeepSpeedEngine.backward()`、`DeepSpeedEngine.step()`、ZeRO optimizer `step()` 和底层 `DeepSpeedCPUAdam.step()`，并保存 wrapper/engine/ZeRO 三层残差与调用次数。它不会把嵌套字段加入 TPS phase 总和。`exact` 模式在 instrumented DeepSpeed 边界同步 CUDA，适合拆分 completed-work 墙钟，但会改变异步重叠；`low_overhead` 不同步，适合估计探针扰动。专用入口 `run_deepspeed_full_ft_probe.sh` 强制 Full-FT，默认 35 steps、warmup 5、OMP=96。当前仅通过合成调用链与 dry-run，尚无新的 30B Full-FT 结果，因此上述 23.933 s 历史合并桶仍不能被反推拆分。

## 9. 新旧结果不能混用

| 对象 | 可以证明 | 不能证明 |
|---|---|---|
| `f209878` A/B | strip/panel/C-residency 方向有效 | `1e95053` 的 TPS |
| 旧 timing runs（含 2026-07-20） | 安装环境中 `f209878 + KT_BACKWARD_TIMING` 的 checkpoint/dW/clear/repack 热点与 3× 预算 | 新 staged profiler 的字段值或 `1e95053` 性能 |
| `1e95053` 源码 | 新 driver、profiler、direct reload 已进入 GitHub 树 | 构建、正确性或端到端性能已通过 |
| GitHub mergeable | PR 可生成合并结果 | CI 或 Full-FT 训练通过 |

## 10. 当前验证缺口

2026-07-20 虽完成了一次旧安装环境的短训和内部打点，但没有重新构建当前源码；因此新 head 仍需在后续独立验收：

1. Release extension 构建；
2. staged profiler Python 测试；
3. raw BF16 repack、dWeight reference 与 benchmark；
4. Full/Hybrid/LoRA；
5. checkpoint on/off 与 shared backward BB；
6. TP/NUMA slice、inactive expert 和多 microbatch；
7. Qwen3 Full-FT 短训与稳定 TPS。

在这些验证完成前，文档只把 `1e95053` 标为“代码已同步”，不标为“运行时已验收”。

## 11. 后续分析顺序

新 profiler 已提供旧文档计划中的 dW 子阶段，因此下一次不需要再增加第二套 C++ timing。应按以下顺序使用现有字段：

1. 同时读取外层 base-weight dW 墙钟和 worker CPU 累积时间，避免把并行时间误当关键路径；
2. 比较 Gate/Up kernel、Down kernel、pack A/B、store，确定计算、packing 或内存写出谁占主导；
3. 对比 `tp.0` 与 `tp.1` 的 tokens、routed rows、active experts 和阶段时间，定位 NUMA/路由不均衡；
4. 分离 direct reload、forward pack、backward pack，验证新 reload 路径的真实收益；
5. 再决定是否研究 selective checkpoint、条件清零、repack 提交时机或 dW kernel。

每项性能结论必须使用同 commit、同构建、同配置的交替运行，并同时报告 step、TPS、backward 和 reload；不同 commit 的历史数字只能作为方向性参考。

## 12. 总体结论

1. 本地 backward 代码已同步到 GitHub `1e95053`，当前工作树不再包含重复的旧 timing 实现。
2. 今日 head 将 `f209878` 的有效原型演进为通用 `BF16DWeightKernel`，并加入更细粒度 staged profiling。
3. 最新旧环境实测为 6.671 s backward / 1.408 s forward，即 4.74×；达到 3× 需要减少至少 2.445 s（36.66%）。
4. 关闭 checkpointing 单独只能推导到约 3.32×；结合清零缩减以及至少约 0.119 s 的 repack/dW/重叠收益，才有达标可能。
5. 若必须保留完整 checkpointing，3× 仍属高风险多阶段优化目标，而不是单一 dW kernel 可保证的结果。
6. 新 direct reload 减少中间分片和 memcpy，不等于取消 BufferB pack，也不能用移动阶段计时来替代 step 改善。
7. 当前最重要的版本边界是：源码已同步到 `1e95053`；最新打点来自旧安装环境，current head 的构建、测试和 Qwen3 端到端结果仍未执行。
8. DeepSpeed 的历史 23.933 s `accelerator.backward` 同时包含 `engine.backward()` 和 `engine.step()`；其显式 optimizer=0 是旧探针边界造成的未捕获，不能与 KT 2.257 s AdamW 直接比较。新四层探针已实现但尚未产生 Full-FT 实测。

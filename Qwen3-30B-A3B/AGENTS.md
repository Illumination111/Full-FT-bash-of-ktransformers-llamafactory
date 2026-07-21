# Qwen3-30B-A3B Full FT 性能分析与 AMX backward 优化方案

> 当前代码状态：2026-07-17 已把 `/mnt/data2/wbw/ktransformers` fast-forward 到 PR #2086 head `1e95053`，本地、`origin/fullft-development`、`upstream/pr-2086` 三者一致且 tracked 工作树干净。同步前的 6+2 个 `KT_BACKWARD_TIMING` 文件保存在 `stash@{0}`，没有混入当前代码树。`OMP=96 + PyTorch fused AdamW` 与 **9.281 → 6.792 s backward、212.76 → 253.78 TPS** 都是 `f209878` 阶段的历史实测；今日 head 新增通用 `SFTProfiler`、`BF16DWeightKernel` 与 direct BF16 reload，但本轮没有对它做构建或训练，不能沿用历史 TPS 作为新 head 结果。GPU routed expert 仍仅为分析设计，尚未修改。

## Agent 工作说明：AMX BF16 文档同步

以下规则是修改 `kt-kernel` AMX 代码、运行 Qwen3-30B-A3B 测试和分析性能数据时的强制要求。一次工作可能同时触发多个文档；满足多个条件时必须更新全部对应文档，不能只选择其中一个。

1. **Backward 文档：** 修改 `kt-kernel` 中的 AMX 部分代码，或进行 Qwen3-30B-A3B 测试时，凡涉及 backward 的代码设计、正确性、打点、性能归因、测试结果或结论，都必须同步更新 [`docs/AMXBF16/AMXBF16-full-FT-backward.md`](../docs/AMXBF16/AMXBF16-full-FT-backward.md)。
2. **Git diff 文档：** 只要本地代码与 GitHub 上选定的代码树状态不同，或本轮对代码做过修改，就必须同步更新 [`docs/AMXBF16/AMXBF16-full-FT-gitdiff.md`](../docs/AMXBF16/AMXBF16-full-FT-gitdiff.md)。文档中应明确比较对象、base/head 或 commit、working-tree 修改和实际 diff，不能把未提交改动、个人分支改动和官方更新混为一谈。
3. **TPS 文档：** 进行 TPS 测试，或新增、修改、引用任何与 TPS、step time、阶段耗时和吞吐归因有关的内容时，都必须同步更新 [`docs/AMXBF16/AMXBF16-full-FT-TPS-bench.md`](../docs/AMXBF16/AMXBF16-full-FT-TPS-bench.md)。必须写清测试配置、warmup/stable 区间、计时口径和对照运行，不能混用 step probe、tqdm 平滑值或不同配置的结果。

每次更新上述任一 Markdown 前，必须完整通读该文件，而不是只读取或修改相关段落；一次工作触发多个文件时，必须逐个完整检查。更新时不得只在文末追加新记录，而保留已经失效、矛盾或会误导后续 Agent 的旧结论。必须同时完成：

- 修正因代码、GitHub 代码树、测试配置或新日志发生变化而变错的结论、数字、路径和描述；
- 合并重复内容，保证同一指标只保留清晰且口径一致的权威结论；
- 缩写已经更新且重要性下降的试错过程、旧方案和旧结论，只保留理解当前设计、回归风险和历史决策所必需的信息；
- 保留尚未被新证据证明的限制条件，明确区分“实测”“推导”“估算”和“待验证”，不能把相关性写成严格因果；
- 检查三个文档之间的配置、commit、时间、TPS、backward 分解和 Git diff 描述是否相互一致。

## 2026-07-17 PR #2086 同步状态

> 代码树同步已经完成；构建、测试和端到端训练验收没有在本轮执行。下面先记录当前权威状态，再保留同步前现场和 8 个 commit 的历史细节，避免把 stash、旧 head 和 GitHub head 混为一谈。

### 当前权威状态

- 本地仓库：`/mnt/data2/wbw/ktransformers`，分支 `fullft-development`。
- 本地 HEAD、`origin/fullft-development`、`upstream/pr-2086` 均为 `1e95053b15b32e6db8193fd852d62d051c6e7ef5`。
- `git status --short --branch` 只显示分支跟踪关系，tracked working-tree diff 为零。
- PR #2086 为 open、非 draft、mergeable；head 没有 commit status，不能等同于 CI 或训练通过。
- 同步前 6 个 tracked 修改和 2 个 untracked timing 文件已保存为 `stash@{0}: pre-1e95053 KT_BACKWARD_TIMING integration 2026-07-17`。
- 本轮最终要求是不继续测试或整合 stash；只同步 GitHub 当日 commit 并更新 `FFTtest` 文档。

### 同步前比较基线（历史现场）

- 本地仓库：`/mnt/data2/wbw/ktransformers`。
- 同步前本地分支为 `fullft-development`，当时 HEAD 为 `f2098786f02ae4cd1d3f6a2968e15df7dfdf83fc`。
- GitHub fork 跟踪分支：`origin/fullft-development`；官方 PR：[`kvcache-ai/ktransformers#2086`](https://github.com/kvcache-ai/ktransformers/pull/2086)。
- 2026-07-17 刷新远端后，`origin/fullft-development` 与 `upstream` 的 PR #2086 head 均为 `1e95053b15b32e6db8193fd852d62d051c6e7ef5`，两者代码树一致。
- `f209878` 是 `1e95053` 的祖先；同步时已执行 fast-forward，没有分叉。
- PR 状态为 Open、`MERGEABLE`，但 `mergeStateStatus=UNSTABLE`，head 没有 status check；因此 GitHub 能合并不等于最新代码已经通过 CI 或 Full-FT 端到端验收。

### 同步前工作树独有内容（现已进入 stash）

同步前本地不是干净的 `f209878`，包含 6 个 tracked 修改和 2 个 untracked 新文件；这些文件现在只存在于恢复 stash，不在当前工作树：

```text
M  kt-kernel/ext_bindings.cpp
M  kt-kernel/operators/amx/sft_moe.hpp
M  kt-kernel/operators/moe-sft-tp.hpp
M  kt-kernel/python/sft/amx.py
M  kt-kernel/python/sft/autograd.py
M  kt-kernel/python/sft/base.py
?? kt-kernel/operators/sft-backward-timing.hpp
?? kt-kernel/python/sft/backward_timing.py
```

这些本地独有改动共同实现 `KT_BACKWARD_TIMING` 专用 backward 打点：

- C++ 在 TP wrapper 和每个 NUMA 子核记录梯度 buffer 清零、NUMA 并行墙钟、Down backward、activation backward、Gate/Up backward、router gradient、base-weight dW 和各类 merge；
- pybind 暴露 timing level 与最近一次 C++ backward timing；
- Python 记录 checkpoint 重算、repack wait/submit、GPU→CPU `grad_output`、CPUInfer task build/submit/sync、结果回传等边界；
- recorder 按 step/layer 汇总，并计划在退出时输出 JSON、CSV 和 Markdown。

这套打点比 GitHub 新增的通用 `KT_SFT_PROFILE` 更专注于逐 step/layer 的 backward orchestration，但 C++ 阶段明显重叠。若以后恢复，只能复用 GitHub `SFTProfiler`，不能把两套 C++ timing 直接叠加。

### GitHub 相对本地 HEAD 新增的 8 个 commit

按 GitHub PR 的 committed time，以下 8 个 commit 在 2026-07-17 加入 PR；其中 `9dc9d93` 的 author time 是 2026-07-16。提交者均为 `yyj`：

1. `9dc9d93` `[feat](kt-kernel): add staged SFT profiling`
   - 新增通用 C++ `SFTProfiler`、Python 收集/格式化/reset API 和 fake-MoE profiler test；
   - 覆盖 forward、backward、TP/NUMA、backward repack 和 base-weight reload。
2. `c6f4211` `[fix](kt-kernel): reuse inference BF16 kernel for SFT`
   - 将 `AMXBF16_SFT_MOE` 从旧 `GemmKernel224BF + AMX_MOE_TP` 切换为 inference 共用的 `GemmKernel224BF16 + AMX_BF16_MOE_TP`；
   - 为 BF16 raw BufferB 增加 unpack、转置 pack 和直接 `forward BufferB → transposed backward BufferB`；
   - 新增 raw BF16 repack 聚焦测试。
3. `109b403` `[perf](kt-kernel): add fine-grained Full-FT profiling`
   - 细化 initial/recompute forward、dWeight、reload partition/pack/cleanup 等 stage；
   - 在 PyTorch profiler 中标记 routing、CPU forward sync、checkpoint 重算、CPU backward 和 repack wait/submit。
4. `34d2102` `[perf](kt-kernel): batch BF16 Full-FT weight gradients`
   - 新增 `BF16DWeightKernel`，按 active expert 与 `i_tile` 合并任务，复用线程局部 BF16 panel/scratch，并调用通用 `GemmKernel224BF16` AMX/AVX512 driver完成完整 K reduction；
   - Gate/Up 共用 input panel，Down 固定 `i_tile` 复用 intermediate panel，降低任务派发和重复 pack；
   - optimizer 更新后可从完整 CPU BF16 Parameter 按 stride 直接 pack 到各 TP/NUMA forward BufferB，去掉 `temp_gate/temp_up/temp_down` 分片分配和 memcpy；
   - direct reload **没有消除全量 BufferB pack**，不能表述成完全释放 requant/reload 时间；
   - 新增 dWeight、strided pack 和 direct repack 聚焦测试。
5. `ea84e6e` `[test](kt-kernel): benchmark BF16 dWeight AMX driver`
   - 扩展 dWeight test，比较 common driver 与 legacy tile loop，并设置性能回退门槛。
6. `273e670` `[fix](kt-kernel): label dWeight store as worker CPU time`
   - 修正 profiler 标签，明确 inner dWeight store 是各 worker 累积 CPU 时间，不能当作外层关键路径墙钟。
7. `61e63a8` `[docs](sft): record BF16 Full-FT performance`
   - 临时加入 Qwen3 BF16 Full-FT 性能报告。
8. `1e95053` `[docs](sft): remove Qwen3 Full-FT performance report`
   - 立即删除上一提交的报告和索引；两个 docs commit 对最终代码树的净变化为零。

相对本地 `f209878`，最终 GitHub head 的净代码变化为 15 个文件、约 `+1524/-112`，核心是通用 profiler、inference BF16 kernel/BufferB 布局复用、新 dWeight driver、direct BF16 reload 及其测试。

### 同步前确认的整合冲突（历史）与当前验证边界

同步前不能在脏工作树中直接 `git pull`。曾在隔离的 `1e95053` worktree 上对旧 tracked diff 执行 `git apply --check`，以下 4 个文件不能直接应用：

```text
kt-kernel/ext_bindings.cpp
kt-kernel/operators/amx/sft_moe.hpp
kt-kernel/operators/moe-sft-tp.hpp
kt-kernel/python/sft/autograd.py
```

原因是本地 `KT_BACKWARD_TIMING` 与远端 `KT_SFT_PROFILE` 同时修改了 pybind、TP/NUMA backward 和 Autograd 打点边界。`amx.py` 与 `base.py` 的 tracked patch 检查未报冲突，但仍需在整合后重新检查接口。两个本地 untracked 新文件在 GitHub head 中没有同名文件。

此次验证在用户要求停止前只得到以下有限结果：

- `f209878..1e95053` 的 `git diff --check` 通过；
- 远端 head 中本轮修改的 Python 文件 `py_compile` 通过；
- profiler pytest 初次收集导入了 Kllama 环境中旧的 `site-packages/kt_kernel`，隔离 Python package 再测又因尚未构建对应 `.so` 而停止；这属于测试环境未完成，既不能记为代码测试通过，也不能记为 PR 功能失败；
- 没有执行 Release extension 完整构建、AMX C++ 聚焦测试、Full/LoRA/Hybrid 回归或 Qwen3 训练。

为避免占用磁盘，检查期间创建的隔离 worktree、Python build 目录和临时目录均已删除。随后原始 6+2 文件被保存进 stash，本地分支完成 fast-forward；本轮没有继续执行 build、pytest、AMX 聚焦测试或 Qwen3 训练。

### 若以后恢复 timing stash

1. 先刷新 PR head 并确认当前分支干净；不要直接在主工作树 `stash pop`。
2. 在独立分支或 worktree 查看 stash，只保留逐 step、microbatch、layer/NUMA trace 和输出归档能力。
3. C++ 阶段必须复用 `SFTProfiler`、`get_profile_stats()` 和现有 Python profiler API，不恢复重复 timing struct/pybind。
4. 构建与临时目录放到 `/mnt/data2`；根分区仍接近满载。
5. 完成 profiler、raw BF16 repack、dWeight、Full/LoRA/Hybrid、checkpoint on/off 和 Qwen3 短训后，才能宣称整合版可运行。

## 总体结论

2026-07-14 原始 KTransformers 基线的稳定吞吐约为 DeepSpeed 的 **29.6%**，即 DeepSpeed 快 **3.38 倍**。该基线的 optimizer/update 路径是最大单项瓶颈，但根因不是缺少 DeepSpeed AVX512 Adam，而是 Accelerate 将 GPU 启动的 `OMP_NUM_THREADS` 默认为 1，使 PyTorch CPU fused AdamW、dense gradient accumulation 和 zero-grad 基本串行。

最终对照 `20260715_162820_1gpu_AMX_BF16_FULL` 保持原 PyTorch fused AdamW、只把 OMP 提升到 96，稳定 step 从 92.029 s 降到 20.405 s，TPS 从 44.51 提升到 200.74。相同 OMP 下，DeepSpeedCPUAdam 的 optimizer 为 6.804 s，而 PyTorch fused AdamW 只有 1.955 s；因此 CPUAdam 替换在当前 BF16/双 NUMA/144 tensor 布局下是 **3.48 倍的 optimizer 回退**，不应合入。

2026-07-14 原始基线还支付约 **15.05 s/step 的 post-optimizer 开销**和 **7.83 s/step 的 48 层 expert 权重重新打包/量化开销**。在该基线口径下，即使将 optimizer 时间不现实地降为零，仍需 **41.11 s/step（99.6 tok/s）**，慢于 DeepSpeed 的 **27.24 s/step（150.4 tok/s）**。后续 OMP=96 实验把 post-optimizer 降至 0.409 s，证明原15.05 s并不是不可消除的固定成本，而是受到ATen/OpenMP并行度强烈影响。

现有日志也不能证明 KTransformers 的 backward 比 DeepSpeed 慢数倍：KTransformers 的稳定 backward 为 **16.703 s/step**；DeepSpeed 记录为 **23.933 s/step**，但后者还包含 `engine.step()`、ZeRO 和 CPUAdam 更新，二者不是同一口径。若单独拆出 DeepSpeed 的纯 GPU backward，它很可能明显更快，但当前日志无法给出准确倍数。

即使 DeepSpeed 的纯 backward 更快，原因也不是 AVX512 Adam。AVX512 Adam 只作用于 optimizer/update，不参与模型梯度计算。纯 backward 的差异主要来自 DeepSpeed 在 GPU Tensor Core 上执行大批量 GEMM，而 KTransformers 在 CPU AMX 上处理按 expert 路由后的小矩阵。

KTransformers 优化前 AMX dW 的核心问题不是线程或任务数量不足，而是 AMX 指令占周期过低。每个 `32×32` 输出 tile 周围存在频繁的标量装填、补零、转置、任务调度，以及 FP32 C tile 的反复回写和重载。线程可以保持忙碌，但多数周期没有执行 AMX 点积。2026-07-16 的最小有效修改已针对这些开销落地，并在完整 Full FT 中取得明确收益。

`f209878` 的历史最小有效优化没有修改 AMX 汇编核、Python、autograd、TP 布局或权重格式；它在 `backward_base_weight_grad()` 内用 fixed-`i_tile` strip、线程局部预打包 panel 和 C tile 跨 K 常驻验证了方向。当前 `1e95053` 已把这一路径演进为独立 `BF16DWeightKernel`，复用 inference `GemmKernel224BF16` driver，并加入 active-expert batch task、细粒度 profiler 与聚焦测试；不能再把旧 113/86 行函数内 diff 描述成当前实现。

另外两项候选优化的结论是：

- **单卡 GPU 分担部分 routed experts：架构上可行，但当前 Full-FT SFT 没有真正实现，且常驻方案受显存限制，预计只能减少几个百分点的 CPU expert 计算，暂不作为第一优先级。**
- **KT 使用 DeepSpeedCPUAdam：可行性原型已验证但性能失败，代码已撤销。** 当前继续由 PyTorch fused AdamW 同时处理 CUDA non-expert 与 CPU expert 参数；CPUAdam A/B 本身没有修改 AMX GEMM 或矩阵打包代码，后续 dW packing 修改与 optimizer 保持解耦。

## 实测对比

两次测试的模型、单卡、batch=1、GAS=1、序列长度 4096、BF16、15 steps 和后 10 个稳定 steps 均一致。以下优先采用逐 step 墙钟探针；summary 中基于 tqdm 平滑值的结果为 39.8 vs 137.4 tok/s，结论相同（DeepSpeed 快 3.45 倍）。

| 稳定区间指标 | KTransformers AMX BF16 | DeepSpeed ZeRO-3 CPU Offload | 说明 |
| --- | ---: | ---: | --- |
| Step time | 92.029 s | 27.240 s | DeepSpeed 快 3.38 倍 |
| TPS | 44.51 | 150.37 | KTransformers 为 29.6% |
| Forward | 1.488 s | 3.292 s | KT 的 CPU expert forward 并不慢 |
| Backward 桶 | 16.703 s | 23.933 s | DeepSpeed 此桶还包含 `engine.step()`，不可直接横比 |
| 显式 optimizer | 50.923 s | 0 s（未捕获） | DeepSpeed 的 0 不是零成本 |
| Post optimizer | 15.047 s | 包含在 DeepSpeed engine 内 | KT 主要是大梯度清零等 |
| Expert requant/repack | 7.828 s | 0 s | KT 每步重建 48 层 AMX 权重格式 |

稳定区间监控采样显示出流水线利用率差异：KTransformers 的 GPU SM 平均利用率约 **1.2%**，DeepSpeed 约 **62.4%**。KTransformers 的长时间 CPU optimizer、清梯度和 requant 阶段使 GPU 基本空闲；DeepSpeed 则把参数卸载、GPU 计算和 ZeRO step 统一管理。

## 为什么 optimizer 很重要

日志确认 DeepSpeed 使用了专门优化的 CPU Adam：

```text
Adam Optimizer #0 is created with AVX512 arithmetic capability.
```

虽然 DeepSpeed JSON 没有显式 `optimizer` 段，但 Accelerate 在 `offload_optimizer.device=cpu` 且默认 `zero_force_ds_cpu_optimizer=true` 时，会自动把 Trainer 创建的 PyTorch AdamW 映射为 `DeepSpeedCPUAdam`。其 C++ 实现使用 AVX512 和 OpenMP；本机安装的实现说明其目标即为相对 `torch.optim.Adam(W)` 提速 5–7 倍。

KTransformers 侧则是另一条路径：

- Transformers 默认创建 `adamw_torch_fused`；
- KT 在 optimizer 创建后，再注入 48 层 × 3 投影共 **144 个 CPU BF16 expert Parameter**；
- 这些 expert 参数约为 `30.532B - 1.541B = 28.991B`，仅 BF16 权重即约 **58.0 GB**，每步 AdamW 还要遍历梯度及一、二阶状态；
- 实测 optimizer 为 **50.923 s/step，占 KT step 的 55.3%**，是第一瓶颈。

KT 日志中的 `Injected 144 fused expert LoRA params` 是兼容入口遗留文案；在本次 `full/lora_rank=0` 运行中，注入的是 expert 基座权重。模型树显示约 1.55B trainable params，是因为约 29B expert 参数由 KT wrapper 外置管理，并不表示本次只训练了 5% 参数。

DeepSpeed 探针显示 `optimizer=0` 是计时边界问题：Accelerate 的 DeepSpeed wrapper 在 `accelerator.backward()` 中调用 `engine.backward()` 和 `engine.step()`，最终的 CPUAdam/ZeRO 更新被计入 23.933 s 的 backward 桶。当前日志无法从该桶中精确拆出 CPUAdam 时间。

## 为什么不只是 AVX512 Adam

1. **原始OMP=1基线的梯度清零约15.05 s/step。** KT Trainer 为保持 KT 梯度 buffer/view 有效，显式执行 `optimizer.zero_grad(set_to_none=False)`。post-optimizer 桶还包含轻量的 scheduler 和 dirty 标记，原15 s基本可归因于近串行遍历并清零大规模CPU expert梯度；OMP=96后该桶实测为0.409 s。下一次C++ backward仍有自己的buffer写入/清零逻辑，存在进一步合并内存遍历的空间。
2. **AMX 权重重新打包约 7.83 s/step。** optimizer 更新权威 BF16 expert 权重后，KT 在下一次 forward 前对 48 层逐层执行 `update_base_weights()`，把权重重新转换为 AMX BufferB 使用的格式。DeepSpeed 直接用权威参数在 GPU 上计算，没有这项派生副本维护成本。
3. **两者并非只替换 optimizer 的同构 A/B。** KT 将 MoE experts 放在双 NUMA CPU 上用 AMX 计算，GPU 处理 non-expert；DeepSpeed 将全模型参数/状态卸载到 CPU，但按层预取参数并在 GPU 上完成 expert 计算。参数布局、梯度生命周期、预取和更新调度均不同。
4. **KT backward 仍有优化余量。** 归档的 FLOPs 分析给出 KT backward 有效吞吐约 6.34 TFLOPS、乐观下界利用率约 5.5%。它不是本次总差距的第一来源，但说明 AMX MoE 路由碎片、NUMA 布局和 dW kernel 尚未充分利用机器峰值。

在2026-07-14原始基线中，KT与DeepSpeed的step差为 **64.79 s**，KT optimizer桶为50.92 s；即便完全删除该桶，KT仍比DeepSpeed多13.87 s/step。该推导只说明旧配置不应把全部差距归因于Adam，不能外推到后来同时启用OMP=96的结果。

## 2026-07-15 OMP 与 optimizer 的最终交叉归因

### 完整同线程 A/B

| 运行 | CPU optimizer | `OMP_NUM_THREADS` | Step | TPS | Forward | Backward | Optimizer | Post-optim | Requant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260714_164510...` | PyTorch fused AdamW | 1 | 92.029 s | 44.51 | 1.488 s | 16.703 s | 50.923 s | 15.047 s | 7.828 s |
| `20260715_151914...` | DeepSpeedCPUAdam | 1 | 未完成稳定区间 | - | - | 16.259/16.293 s | - | - | - |
| `20260715_155237...` | DeepSpeedCPUAdam | 96 | 24.566 s | 166.73 | 1.474 s | 9.352 s | 6.804 s | 0.409 s | 6.491 s |
| `20260715_162820...` | **PyTorch fused AdamW** | **96** | **20.405 s** | **200.74** | 1.497 s | **9.239 s** | **1.955 s** | **0.348 s** | 7.328 s |

最终对照证明：

- backward 的 `16.703 → 9.239 s` 来自 OMP/ATen 并行度，CPUAdam 对同一步 backward 没有直接作用；相同 OMP 下两种 optimizer 的 backward 只差 1.2%。
- 原 fused AdamW 在 OMP=96 时从 50.923 s 降到 1.955 s，说明旧瓶颈是单线程配置，不是 AdamW 实现天然缓慢。
- DeepSpeedCPUAdam 在相同 OMP 下为 6.804 s，比 fused AdamW 慢 3.48 倍，并使完整 step 多 4.161 s、TPS 低 20.4%。
- legacy AdamW 的 requant 比 CPUAdam 多 0.837 s，可能来自更新后 cache/NUMA 状态差异；即使计入该损失，legacy AdamW 仍明显更快。需要重复 run 才能判断这 0.837 s 是否稳定。

Trainer 在 backward 完成后才调用 `optimizer.step()`，所以 optimizer 不可能直接改变同一步 AMX dX/dW。真正受 OMP 影响的是 Full FT 中大规模 CPU tensor 的 `AccumulateGrad`、fused AdamW、`zero_grad(set_to_none=False)`、初始化/first-touch 以及 AMX 前后的 ATen 操作。AMX dW 核心 WorkerPool 并未因 optimizer A/B 改变。

### 为什么 AVX512 CPUAdam 没有优势

AVX512 不是 DeepSpeedCPUAdam 独占能力。本机 PyTorch 2.9.1 同样以 AVX512、OpenMP 和 oneDNN/MKL 构建；默认 `adamw_torch_fused` 按 device/dtype 聚合 tensor，再调用原生 `_fused_adamw_` 多 tensor 接口。

DeepSpeedCPUAdam 则在 Python 中逐个遍历 144 个 expert tensor，每个 tensor 调一次 C++ `adam_update`；C++ 又按 128M 元素 tile 重复建立 OpenMP parallel-for。对当前 BF16 Parameter + BF16 moments、双 NUMA 和超大 tensor 布局，AVX512 逐元素指令不足以抵消 per-tensor 调度、重复 parallel region、NUMA/first-touch 和数据流量成本。现有日志不能把多出的 4.849 s 精确分摊到其中某一项，但已经足以判定该实现不适合当前默认路径。

optimizer 仍是内存带宽受限，不能按 FLOPs 与 forward 比较。仓库保守 32 B/parameter 模型给出的 optimizer 下界约 1.510 s；fused AdamW 实测 1.955 s、optimizer/forward 为 1.31 倍，已经接近该保守 roofline。CPUAdam 实测 6.804 s、optimizer/forward 为 4.62 倍，在同机同线程对照下不再具有实用竞争力。

### 代码决策

- 删除未提交的 `KTHybridOptimizer`、DeepSpeedCPUAdam JIT loader、CPU/GPU parameter 分流、嵌套 optimizer checkpoint 和相关环境变量。
- 恢复 PR #2086 原有路径：KT 的 144 个 CPU expert Parameter 继续注入 Transformers 默认 PyTorch fused AdamW。
- 不引入 DeepSpeed 运行时依赖，不改变 checkpoint state 格式，也不修改 AMX GEMM/weight packing。
- 保留 monitor FIFO SIGPIPE 修复和逐 step 计时探针；它们解决的是独立的测试启动/归因问题。

### OMP 自动配置

Accelerate 对 GPU launch 在未指定时会写入 `OMP_NUM_THREADS=1`。KT SFT 现在按当前进程 CPU affinity 可见的 `(physical_package_id, core_id)` 去重计数物理核心，并同时设置环境变量和 `torch.set_num_threads()`。本机为 2 sockets × 48 physical cores，因此自动值是 96，而不是 192 个 SMT logical threads。

优先级如下：

1. `ACCELERATE_KT_OMP_NUM_THREADS`：KT 专用强制覆盖，包含有意测试 `=1`；
2. 已存在且大于 1 的 `OMP_NUM_THREADS`；
3. affinity/cpuset 可见物理核心数；sysfs topology 不可用时回退到可见逻辑 CPU 数。

测试 runner 在启动 Accelerate 之前执行相同探测，并按 `FFT_OMP_NUM_THREADS`、`ACCELERATE_KT_OMP_NUM_THREADS`、现有 `OMP_NUM_THREADS`、自动探测的顺序解析一次；随后把同一个值同时传给 `OMP_NUM_THREADS` 和 KT 专用覆盖变量。这样日志展示值、OpenMP 初始值和 KT 运行时值始终一致，也不会再被 Accelerate 默认降为 1。

正式 Full-FT 性能命令无需写死 OMP：

```bash
cd /mnt/data2/wbw/FFTtest/Qwen3-30B-A3B

bash run_finetune_perf_test_bf16.sh \
  --mode full \
  --gpus 1 \
  --batch-size 1 \
  --gas 1 \
  --steps 15 \
  --warmup-steps 5 \
  --learning-rate 1.0e-5
```

需要固定线程做实验时使用：

```bash
FFT_OMP_NUM_THREADS=48 bash run_finetune_perf_test_bf16.sh --mode full --gpus 1
```

验收日志应显示 `OpenMP threads : 96`（本机自动值），optimizer 只显示 `AdamW`/`AcceleratedOptimizer`，不应出现 `DeepSpeedCPUAdam` 或 `KTHybridOptimizer`。

## 可行性评估：单卡 GPU 分担 routed experts

### 结论

**可以实现，但当前代码不能通过设置 `kt_num_gpu_experts` 直接启用。** 单卡不需要 expert parallel 或 all-to-all，只需将选中的 routed expert 放在 `cuda:0`，CPU 和 GPU 分别计算互斥的 expert-token 对，最后对输出求和。当前 forward 已采用“先提交 CPU expert、再计算 GPU shared/LoRA expert、最后同步 CPU”的结构，因此具备扩展为 CPU/GPU 重叠执行的基础。

但对于本次 Full FT 配置，**常驻少量 GPU experts 的收益预计有限**：监控记录 GPU 峰值为 39.57 GB、总显存为 49.14 GB，只剩约 9.57 GB，且还必须为临时 workspace 和 allocator 碎片留余量。若不做 optimizer/gradient offload，安全起点应为每层 2 个 GPU experts，逐步测试到 3–4 个，而不是一次放入大量 experts。

### 为什么当前配置项没有生效

现有实现已经有部分接口，但 SFT Full FT 链路未接通：

- [`build_kt_device_map()`](../../ktransformers/kt-kernel/python/sft/wrapper.py#L69) 可以把前 N 个 expert 映射到 `cuda:0`；
- [`KTMoEWrapper` 创建处](../../ktransformers/kt-kernel/python/sft/wrapper.py#L334) 仍硬编码 `gpu_experts_mask=None, num_gpu_experts=0`；
- [`AMXSFTMoEWrapper.load_weights()`](../../ktransformers/kt-kernel/python/sft/amx.py#L206) 没有把 `self.num_gpu_experts` 写入 C++ SFT config；
- C++ SFT forward/backward 已有 `expert_id < config_.num_gpu_experts` 的跳过逻辑，例如 [`forward_sft()`](../../ktransformers/kt-kernel/operators/amx/sft_moe.hpp#L969)，但因为 config 始终为 0，实际上不会跳过任何 expert；
- 单卡 GPU 分支目前只计算 shared experts 和 LoRA experts，见 [`_submit_and_compute_gpu()`](../../ktransformers/kt-kernel/python/sft/layer.py#L445)，没有 routed expert MLP；
- Qwen3-MoE 在 transformers 5.6 中使用整层融合参数 `[E, 2I, H]` 和 `[E, H, I]`。KT 加载后会把整组原 expert 权重替换成 zero-storage placeholder，见 [`_clear_original_expert_weights()`](../../ktransformers/kt-kernel/python/sft/weights.py#L111)，因此必须在清理前单独保存选中 expert 的紧凑 GPU Parameter，不能继续调用已清理的原模块。

因此，**只设置 `ACCELERATE_KT_NUM_GPU_EXPERTS=N` 会造成“device map 看似支持、实际 routed expert 仍全部由 CPU 计算”的结果，不能作为有效测试。**

### 显存和预期减压比例

Qwen3-30B-A3B 的一个 expert 在一层中有：

```text
3 × H × I = 3 × 2048 × 768 = 4,718,592 个参数
BF16 权重约 9.44 MB/层
```

若同一 expert 编号常驻全部 48 层，其存储下界为：

| 存储内容 | 每个 expert 编号跨 48 层 |
| --- | ---: |
| 仅 BF16 参数 | 0.453 GB |
| BF16 参数 + 梯度 + 两个 BF16 Adam states | 1.812 GB |
| BF16 参数 + 梯度 + 两个 FP32 Adam states | 2.718 GB |

以上还不含激活、临时 grouped-GEMM workspace 和 allocator 碎片。因此，本次约 9.57 GB 的观测余量只适合从 2 个 expert/层开始验证。

如果路由经过负载均衡且近似均匀，N 个 GPU experts 只能减少约 `N/128` 的 CPU expert-token 计算：

| 每层 GPU experts | CPU expert 运算理论减少量 |
| ---: | ---: |
| 2 | 1.56% |
| 3 | 2.34% |
| 4 | 3.13% |

只有实际路由明显偏斜、选中的 hot experts 覆盖率远高于 `N/128` 时，收益才会更大。当前测试没有保存逐层 expert 命中直方图，因此实施前必须先记录 `topk_ids`，按“每个 expert 实际 token 占比”而不是 expert 编号选择。

### 推荐实现边界

最小可验证版本应限定为单卡、固定 expert 集合：

1. 加载时从融合权重中抽取选中 expert，创建紧凑的 GPU `gate_up_proj/down_proj` Parameter；未选中的 expert 继续由 CPU BF16/AMX 管理。
2. 优先支持任意逐层 bool mask；若为了最小改动沿用 C++ 的“前 N 个 expert”判断，则必须同时重排 expert 权重和 router 输出列，并在保存 checkpoint 时恢复映射，不能直接假设前 N 个最热。
3. forward 中先提交 CPU 任务，再使用 fused/grouped GEMM 计算 GPU expert-token 对，按 routing weight `index_add`，最后与 CPU 输出相加。直接复用 HF 的逐 expert Python 循环可能变成 kernel-launch 瓶颈，只适合正确性原型。
4. CPU C++ forward/backward 必须跳过 GPU mask；被跳过位置的 CPU output、`grad_input` 和 `grad_weights` 必须显式为零，避免当前 `torch.empty` buffer 中的未初始化值进入求和。
5. GPU 分支交给 PyTorch autograd 产生 input、router weight 和 GPU expert dW；CPU 自定义 autograd 只返回 CPU expert 的贡献，两侧梯度相加。
6. GPU expert 参数交给 GPU AdamW，CPU expert 参数交给 CPU optimizer；不能让同一个 expert 同时存在两个可更新的权威副本。
7. forward 可以复用当前 submit/compute/sync 顺序实现重叠；backward 需要显式 CUDA stream/event 和 CPU async submit，不能依赖 autograd 分支的偶然调度顺序。

若 resident 版本的路由覆盖率不足，下一阶段可以考虑“CPU master + 按层预取 GPU mirror + GPU 梯度回传 CPU”的流式方案。它能用较少显存覆盖更多 experts，但会引入每层 PCIe 权重/梯度传输、prefetch、checkpoint 重算和一致性管理，已经接近 DeepSpeed ZeRO-Offload 的层级调度复杂度，不属于最小修改。

### 优先级判断

该方案利用了当前长周期内空闲的 GPU，技术方向成立，但本次显存余量下的常驻 expert 数太少。建议只有在以下条件同时满足时再进入实现：

- 路由统计表明少数 hot experts 覆盖了显著高于 `N/128` 的 token；
- GPU expert forward+backward 能完全或大部分隐藏在剩余 CPU expert 时间内；
- 增加 2–4 个 experts 后显存峰值仍至少保留 1–2 GB 安全余量。

否则，应优先优化 AMX dW 和 CPU optimizer，它们的潜在收益更大、修改边界也更清晰。

## CPUAdam 原型归档与保留的启动修复

DeepSpeedCPUAdam 原型证明了架构上可以用双 optimizer 分流 CPU/GPU Parameter，但同线程 A/B 已证明它在本工作负载中无性能价值。相关 Python adapter、CPU-only JIT loader、环境变量和嵌套 checkpoint 格式全部撤销，不作为 PR #2086 的组成部分。

这一结论不改变 AMX 与 optimizer 的接口解耦事实：AMX 负责 expert forward/dX/dW，optimizer 逐元素更新权威 CPU BF16 Parameter，下一次 forward 前 `update_base_weights()` 重打包派生 BufferB。2026-07-16 的 AMX dW packing 优化正是在 fused AdamW 不变的前提下独立实现和测量。

### monitor FIFO SIGPIPE 修复继续保留

`20260715_145106...` 在调用 Accelerate 前退出，最终定位为 monitor FIFO reader 尚未 ready 时父 shell 写入，触发 `SIGPIPE/exit 141`。修复使用 reader-ready marker、monitor PID 检查和明确的 FIFO 写错误。该问题从旧 runner 即存在，与 CPUAdam 无关，因此继续保留。

### 当前运行时验收

- `Kllama` 环境不再需要 DeepSpeedCPUAdam Python adapter 或 CPU-only JIT extension；
- KT 原生 AMX 扩展没有因 optimizer A/B 改动，不需要为撤销 CPUAdam 重新编译；
- OMP runtime 逻辑属于 Python 修改，本地源码可以直接做单元测试；安装环境正式运行前仍需重新安装或同步 Python package；
- 训练日志只应出现 `AdamW` 和 `AcceleratedOptimizer`；任何 `KTHybridOptimizer`/`Using DeepSpeedCPUAdam` 都表示环境仍残留旧原型。

运行中可使用以下只读监控命令：

```bash
cd /mnt/data2/wbw/FFTtest/Qwen3-30B-A3B
RUN_DIR=$(find test_log -maxdepth 1 -type d -name '*_1gpu_AMX_BF16_FULL' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)

tail -F "${RUN_DIR}/full_ft/monitor.log" \
        "${RUN_DIR}/full_ft/phase4/train_full_ft.log" \
        "${RUN_DIR}/full_ft/phase4/step_timing/step_timing.csv"
```

## 2026-07-16 AMX Full-FT dW 最小有效修改实测

### 对照口径

目标运行是 [`20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA`](test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA)，对照运行是紧邻的 [`20260715_203338_1gpu_AMX_BF16_FULL_THEN_LORA`](test_log/20260715_203338_1gpu_AMX_BF16_FULL_THEN_LORA)。两者的 `session_config.json` 文本 diff 为空，均为单卡、AMX BF16、2 NUMA、batch=1、GAS=1、seq=4096、15 steps、跳过前 5 steps、OMP=96、PyTorch fused AdamW；两次会话也都按 `full -> lora` 顺序串行运行。

以下以 `step_timing.json` 的 10 个稳定 step 为主口径。它直接覆盖 dataloader 到 log/save 的 TPS cycle，适合阶段归因；`summary.md` 中另一个基于 tqdm 日志时间戳的 TPS 会包含平滑和日志间隔，新 Full 得到 226.8 tok/s，而 step probe 得到 253.78 tok/s。本文所有 A/B 百分比统一使用后者，避免混用。

| Full 稳定指标 | 修改前 | 修改后 | 变化 |
| --- | ---: | ---: | ---: |
| Step | 19.25197 s | 16.13997 s | **-3.11200 s / -16.16%** |
| TPS | 212.76 | 253.78 | **+41.02 / +19.28%** |
| Forward | 1.47821 s | 1.34553 s | -0.13268 s / -8.98% |
| Backward | 9.28085 s | 6.79183 s | **-2.48902 s / -26.82%** |
| Optimizer | 1.79075 s | 1.92533 s | +0.13458 s / +7.52% |
| Requant/update | 6.32860 s | 5.73156 s | -0.59704 s / -9.43% |

结论是 **改进有效**。目标 backward 桶单步减少 2.489 s，占完整 step 3.112 s 减少量的约 **80.0%**，而 optimizer 没有同步变快。按日志的同一理论 FLOPs 模型，backward 有效吞吐从 11.412 提升到 15.594 TFLOPS（+36.65%），roofline 下界效率从 9.95% 提升到 13.59%，自动判断由“偏低”变为“正常”；该指标是整个 backward 的模型估算，不等价于硬件 AMX tile 利用率。新的稳定 backward 范围为 6.755–6.829 s，step 范围为 16.008–16.290 s，抖动很小；训练退出码为 0，15 步 loss 从 0.7773 降到 0.2754，记录的 grad norm 为 1.685–5.276，健康检查判定无 SIGSEGV/NaN。

不能把全部端到端收益都归因于 dW diff：requant 同时减少 0.597 s、forward 减少 0.133 s，这两个阶段不在本次 C++ 修改范围内，应视为运行间 cache/NUMA/系统波动或其他非隔离因素。反过来，optimizer 还回退了 0.135 s。现有证据足以确认 dW 优化方向和实际收益，但若要给单函数做严格归因，仍应增加 `backward_base_weight_grad()` 独立计时并做至少 3 次 revert/patch 交叉运行。

### LoRA 旁路对照

同一组会话中的 LoRA-only 不会调用 `backward_base_weight_grad()`，因为 [`backward()`](../../ktransformers/kt-kernel/operators/amx/sft_moe.hpp#L1903) 仅在 `full_weight_grad=true` 时进入该函数。因此 LoRA 可作为“机器和公共 backward 路径变化”的旁路对照：

| LoRA 稳定指标 | 修改前会话 | 修改后会话 | 变化 |
| --- | ---: | ---: | ---: |
| Step | 6.08625 s | 5.86640 s | -0.21984 s / -3.61% |
| TPS | 672.99 | 698.21 | +25.22 / +3.75% |
| Forward | 1.63101 s | 1.54067 s | -0.09033 s / -5.54% |
| Backward | 4.35296 s | 4.23380 s | -0.11916 s / -2.74% |

LoRA 的小幅整体变快说明两次运行确有约 3%–5% 的公共波动，但远小于 Full backward 的 26.82%。以 backward 的绝对差做保守差分，Full 仍有约 `2.489 - 0.119 = 2.370 s/step` 的额外减少，且该差异恰好落在只有 Full 执行的 dW 路径，因果方向一致。新 LoRA 也稳定完成，说明此次 Full-only diff 没有造成明显的 LoRA 回归。

### `f209878` 历史修改和验证状态

当时的修改仅涉及 `sft_moe.hpp` 的 AMX BF16 dW 分支，commit diff 为 113 additions / 86 deletions：

- 任务从 `(expert, projection, i_tile, h_tile)` 合并为 `(expert, projection, i_tile)`，Qwen 每层每 NUMA 的任务数从 196,608 降为 3,072，任务内部遍历 64 个 `h_tile`；
- 固定 strip 的 Gate/Up gradient A panel 或 Down intermediate B panel 只预打包一次；每个 `h_tile` 的 input B panel由 Gate/Up 共用；
- 三个 `thread_local std::vector<ggml_bf16_t>` 仅增不减地复用 scratch，并手工对齐到 64 byte；
- Down 保留 C tile 贯穿全部 K，Gate/Up 因共用四个 C tile 而分成两个完整 K pass；每个输出 tile 最后只 `store_c()` 一次；
- 当时没有改变 `GemmKernel224BF` 的 `32×32×32` tile 形状、底层 tile config、外部接口、TP 布局、权重格式、WorkerPool 或 Python/autograd。

`f209878` 的 Release 构建和临时 reference 对照曾覆盖 TP part 0/1、非零 expert、`M={1,31,32,33,65,256}` 以及 Qwen 维度 `H=2048, I_local=384`。当前 `1e95053` 已改用 `BF16DWeightKernel`/`GemmKernel224BF16`，本轮未重新执行这些验证。当前代码 diff 和历史证据边界见 [`docs/AMXBF16/AMXBF16-full-FT-gitdiff.md`](../docs/AMXBF16/AMXBF16-full-FT-gitdiff.md)。

### 参考 LoRA backward 后的方向判断

Full FT 的方向是正确的，但不能把 LoRA rank-8 的具体内核原样搬到 Full dW：

- 公共 base dX 在 [`backward_down_amx()`](../../ktransformers/kt-kernel/operators/amx/sft_moe.hpp#L4452) 和 [`backward_gate_up_amx()`](../../ktransformers/kt-kernel/operators/amx/sft_moe.hpp#L5212) 中使用通用 AMX `BufferA::from_mat()`、预先转置的 base-weight `BufferB` 和 `amx::mat_mul()`；Full 与 LoRA 都会经过这部分。
- LoRA adapter backward 的主要优化是融合 Gate/Up 的 `u` 与 grad-B、按 token/dimension 分块、线程局部 FP32 scratch、rank=8 AVX2 FMA、AVX512 scatter/fused-add，以及共享 buffer pool；它主要减少低秩小矩阵周围的读取、调度和归约，不是改变 AMX tile 寄存器形状。
- `prepare_lora_backward_weights()` 仍会把两组 Down LoRA 转置权重转换成 AMX `BufferB`，但当前 live backward 中 `down_lora_a_t_bb_`/`down_lora_b_t_bb_` 除准备和分配外没有消费点；实际 adapter 梯度/grad-input 走 `avx::lora_*`。因此准确答案是：**LoRA 保留了 AMX BufferB 打包准备接口，公共 base dX 也使用标准 AMX packing，但当前 LoRA-specific backward 优化没有调整底层 AMX tile 或 live tile packing 循环。**
- Full dW 是大矩阵外积，两个 operand 都来自本次 activation/gradient，不能像 base dX 那样长期缓存权重 `BufferB`。本次按 strip 复用动态 panel、合并任务、让 C 跨 K 常驻，正是适合 Full dW 的对应做法，也与 LoRA “融合、分块、复用 scratch、减少中间读写”的优化原则一致。

当前 head 已加入 dW 函数级及 pack/kernel/store staged timing，不应再恢复第二套 C++ 墙钟。下一次运行应先使用现有 `SFTProfiler` 区分外层墙钟与 worker CPU 累积时间，再采集 AMX/top-down 计数器；历史 6.792 s backward 与 5.732 s requant 只能作为 `f209878` 规划基线。

## AMX backward 优化前热点结构

`backward_base_weight_grad()` 计算三组基座权重梯度：

```text
grad_gate = grad_gate_out^T × input         [I, H]
grad_up   = grad_up_out^T   × input         [I, H]
grad_down = grad_output^T   × intermediate  [H, I]
```

当前 AMX kernel 的输出 tile 为 `32×32`，K tile 为 32。Qwen3-30B-A3B 在每个 NUMA/TP 分区上的维度为：

```text
I_local = 384  → 12 个 i_tile
H       = 2048 → 64 个 h_tile
experts = 128
```

优化前一个任务只计算一个 `(expert, projection, i_tile, h_tile)`，因此每层、每个 NUMA 约投递：

```text
128 × 2 × 12 × 64 = 196,608 个任务
```

这远多于 48 个 NUMA worker，但任务过细会带来以下开销：

- 每个任务都通过共享原子计数器领取工作；
- 每个输出 tile 重新执行 BF16 panel 的标量复制、补零和转置；
- Gate/Up 的相同 input panel 被重复打包；
- 固定 `i_tile` 的 Gate/Up 梯度 panel 在 64 个 `h_tile` 上重复打包；
- Down 的 intermediate panel 在 64 个 `h_tile` 上重复打包；
- 每个 K 分块都将 FP32 C tile 从 AMX 寄存器写到内存，下一分块再载入。

平均每个 expert 约有：

```text
4096 tokens × top-k 8 / 128 experts ≈ 256 tokens
```

即约 8 个 K 分块。优化前单个输出 tile、单个投影会进行约 8 次 `store_c` 和 7 次 `load_c`；每次 C tile 为 `32×32×4 = 4 KiB`。这些数据通常落在缓存中，但仍占用 tile load/store 指令、缓存端口和执行周期。

## 已实施的 AMX 最小有效修改方案

### 第一步：C tile 跨完整 K 循环常驻

修改位置仅限 [`sft_moe.hpp`](../../ktransformers/kt-kernel/operators/amx/sft_moe.hpp#L1948) 中的 AMX dW 分支。

Down 优化前等价于：

```cpp
for (int kt = 0; kt < k_tiles; ++kt) {
    pack_a_b(kt);
    if (kt == 0) clean_c();
    else         load_c(c0);
    run_tile();
    store_c(c0);
}
```

改为：

```cpp
clean_c();
for (int kt = 0; kt < k_tiles; ++kt) {
    pack_a_b(kt);
    load_a_b();
    run_tile();
}
store_c(c0);
```

Gate 和 Up 共用同一组 AMX C tile 寄存器，不能同时常驻，因此拆成两个完整的 K pass：

```cpp
clean_c();
for (int kt = 0; kt < k_tiles; ++kt)
    run_gate_tile(kt);
store_c(c_gate);

clean_c();
for (int kt = 0; kt < k_tiles; ++kt)
    run_up_tile(kt);
store_c(c_up);
```

每个输出 tile、每个投影的 C 内存操作由约 15 次降为 1 次。每个输出元素的 K 累加顺序仍然保持从小到大，数值语义不变。

这一修改行数最少、风险最低，但单独实施未必足以让 AMX 高占用，因为 operand 打包和任务调度仍然过于频繁。

### 第二步：任务合并为固定 `i_tile` 的 strip

将任务单位由：

```text
(expert, projection, i_tile, h_tile)
```

改为：

```text
(expert, projection, i_tile)
```

每个任务内部循环全部 64 个 `h_tile`。每层、每个 NUMA 的任务数变为：

```text
128 × 2 × 12 = 3,072
```

48 个 worker 平均仍有 64 个任务，足以负载均衡，同时任务领取次数减少 64 倍。

实际采用的循环结构：

```cpp
if (projection == GATE_UP) {
    pack_all_k_gate_a(i_tile);
    pack_all_k_up_a(i_tile);

    for (int ht = 0; ht < h_tiles; ++ht) {
        pack_all_k_input_b(ht);

        clean_c();
        for (int kt = 0; kt < k_tiles; ++kt)
            run(packed_gate_a[kt], packed_input_b[kt]);
        store_gate_c();

        clean_c();
        for (int kt = 0; kt < k_tiles; ++kt)
            run(packed_up_a[kt], packed_input_b[kt]);
        store_up_c();
    }
} else {
    pack_all_k_intermediate_b(i_tile);

    for (int ht = 0; ht < h_tiles; ++ht) {
        pack_all_k_grad_output_a(ht);

        clean_c();
        for (int kt = 0; kt < k_tiles; ++kt)
            run(packed_grad_output_a[kt],
                packed_intermediate_b[kt]);
        store_down_c();
    }
}
```

### 第三步：使用线程局部的预打包 panel

在该函数内部使用按需扩容、跨调用复用的线程局部 scratch：

```cpp
thread_local std::vector<ggml_bf16_t> packed_a0;
thread_local std::vector<ggml_bf16_t> packed_a1;
thread_local std::vector<ggml_bf16_t> packed_b;
```

| 路径 | 固定 strip 后只打包一次 | 每个 `h_tile` 打包一次 |
| --- | --- | --- |
| Gate/Up | Gate A、Up A 的全部 K panel | input B 的全部 K panel，供 Gate/Up 共用 |
| Down | intermediate B 的全部 K panel | grad_output A 的全部 K panel |

平均 `M=256` 时，三个 panel scratch 合计最大约 48 KiB/worker，不需要新增类成员、持久化大 buffer 或接口参数。M、I、H 非 32 整数倍时继续使用现有补零逻辑。

## 为什么不首先修改 WorkerPool

[`WorkerPool::process_tasks()`](../../ktransformers/kt-kernel/cpu_backend/worker_pool.cpp#L178) 中计算了 guided block，随后又强制设置：

```cpp
block = 1;
```

删除这一行看似是最小修改，但不应作为第一步：

- 不减少 operand 重复打包；
- 不减少 AMX C tile 的回写和重载；
- 修改会影响所有共享 WorkerPool 的算子；
- 大 block 可能降低尾部负载均衡。

应先通过 dW strip task 在局部自然减少调度开销。若之后性能分析仍显示共享原子计数器是热点，再单独增加可配置 grain size，而不是直接全局删除 `block = 1`。

## 推荐优化和验证顺序

### 总体训练性能

1. **保持 PyTorch fused AdamW，并让 OMP 自动匹配物理核心。** CPUAdam A/B 已完成且确认回退，不再作为默认优化方向。
2. **重新测量 direct reload。** `f209878` 历史运行中 fused AdamW 为 1.925 s、requant 为 5.732 s；`1e95053` 已改为 strided direct pack，但尚无同 commit 实测，先用现有 reload stage 验证再决定 selective update。
3. **消除重复的全量梯度清零。** 精确验证 C++ buffer 清零、autograd `.grad` 和 GAS 生命周期后，尝试 `set_to_none=True`、只清活跃 expert，或把清零融合进下一次 backward；不能在未验证梯度累积语义时直接删除。
4. **使用现有 staged profiler 测量 dW。** 新 head 已区分外层 dW、pack A/B、Gate/Up/Down kernel 与 worker store；不要新增重复 C++ timing。
5. **重新建立新 head 基准。** 对 `f209878` 与 `1e95053` 做至少 3 次交叉运行并增加稳定 steps，记录实际 OMP、CPU affinity、NUMA first-touch 和 profiler on/off。

### AMX backward 修改

1. **历史已完成（`f209878`）：** C tile 常驻、fixed-`i_tile` strip、线程局部预打包 panel，并完成 reference 与 Full/LoRA 短测。
2. **当前代码已同步（`1e95053`）：** `BF16DWeightKernel`、common driver、active-expert batch task 和 staged profiler 已进入本地代码树。
3. **当前待验证：** 新 driver 的 reference/benchmark、Full/Hybrid/LoRA、checkpoint on/off 与 Qwen3 短训；本轮没有执行测试。
4. **计时接口已存在：** 复用 `SFTProfiler` 的 dW/pack/kernel/store 字段，区分外层墙钟和 worker CPU 累积时间。
5. 分别以每 NUMA 12/24/48 worker 测试 dW 扩展性。
6. 使用平台可用的 AMX/tile、cycles、cache miss 和 top-down 计数器；事件名应以本机 `perf list` 为准，也可使用 Intel VTune。
7. 若 24→48 worker 扩展仍明显变差，再分析 NUMA 带宽和 WorkerPool 调度，不预先修改全局调度器。

“满载 AMX”应定义为 dW 阶段 AMX 指令占周期显著上升、24→48 worker 仍有良好扩展，而不是只看 CPU 利用率。对于 MoE 路由形成的 `M≈256` 小矩阵，不应承诺达到理论峰值或 100% AMX 利用率。

## 推荐实施边界

- **历史最小方案：** `f209878` 的 113/86 行函数内原型，仅用于解释旧 A/B。
- **当前实现边界：** `1e95053` 使用独立 `BF16DWeightKernel`、inference BF16 driver、staged profiler 和 direct reload，修改面已超过旧单函数边界。
- **仍未包含：** optimizer 替换、条件 zero-grad、全局 WorkerPool 改造和 GPU routed experts。

## 证据文件

- [KTransformers summary](test_log/20260714_164510_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft/summary.md)
- [KTransformers step timing](test_log/20260714_164510_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft/phase4/step_timing/step_timing.md)
- [DeepSpeed summary](test_log/20260714_154355_1gpu_DEEPSPEED_Z3_OFFLOAD_BF16_FULL_THEN_LORA/full_ft/summary.md)
- [DeepSpeed step timing](test_log/20260714_154355_1gpu_DEEPSPEED_Z3_OFFLOAD_BF16_FULL_THEN_LORA/full_ft/phase4/step_timing/step_timing.md)
- [DeepSpeed config](configs/deepspeed_zero3_offload_bf16.json)
- [OMP=96 + DeepSpeedCPUAdam summary](test_log/20260715_155237_1gpu_AMX_BF16_FULL/full_ft/summary.md)
- [OMP=96 + PyTorch fused AdamW summary](test_log/20260715_162820_1gpu_AMX_BF16_FULL/full_ft/summary.md)
- [OMP=96 + PyTorch fused AdamW step timing](test_log/20260715_162820_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/step_timing.md)
- [AMX dW 修改前 Full/LoRA 对照](test_log/20260715_203338_1gpu_AMX_BF16_FULL_THEN_LORA/comparison.md)
- [AMX dW 修改后 Full/LoRA 对照](test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/comparison.md)
- [AMX dW 修改前 Full step timing](test_log/20260715_203338_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft/phase4/step_timing/step_timing.md)
- [AMX dW 修改后 Full step timing](test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft/phase4/step_timing/step_timing.md)
- [AMX dW 修改前 LoRA step timing](test_log/20260715_203338_1gpu_AMX_BF16_FULL_THEN_LORA/lora_ft/phase4/step_timing/step_timing.md)
- [AMX dW 修改后 LoRA step timing](test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/lora_ft/phase4/step_timing/step_timing.md)
- [Full FT AMX dW 代码 diff 说明](../docs/AMXBF16/AMXBF16-full-FT-gitdiff.md)
- [KT Full-FT 参数与梯度 buffer](../../ktransformers/kt-kernel/python/sft/base.py)
- [KT optimizer 参数注入与 dirty 标记](../../ktransformers/kt-kernel/python/sft/lora.py)

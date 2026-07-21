# Qwen3-30B-A3B AMX BF16 Full FT TPS Benchmark

对应 PR：[kvcache-ai/ktransformers#2086](https://github.com/kvcache-ai/ktransformers/pull/2086)

更新时间：2026-07-21

## 1. 当前代码与性能结论必须分开

当前 GitHub、本地分支和官方 PR ref 已同步到：

```text
1e95053b15b32e6db8193fd852d62d051c6e7ef5
```

本轮没有对该 head 重新构建 extension 或安装 Python package。因此当前性能结论分为两层：

- **当前代码状态：** `1e95053` 已包含 staged profiler、通用 `BF16DWeightKernel`、inference BF16 kernel/BufferB 复用和 direct BF16 reload；
- **最新可用实测：** 2026-07-20 新运行仍加载 2026-07-16 构建的 extension，以及与 `stash@{0}` 哈希一致的 `backward_timing.py` / `autograd.py`，属于 `f209878 + KT_BACKWARD_TIMING` 安装环境，不能标成 `1e95053` 的 benchmark。

当前最适合作为“旧 head、无 backward 内部 recorder”的历史性能基线是：

```text
20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft
commit family: f209878

16.140 s/step
253.78 tok/s
backward 6.792 s
requant  5.732 s
```

最新历史内部打点运行是：

```text
20260720_130854_1gpu_AMX_BF16_FULL/full_ft
17.032 s/step
240.49 tok/s
forward  1.408 s
backward 6.671 s（4.74 × forward）
requant  6.327 s
```

该运行使用已经移出当前工作树的旧 `KT_BACKWARD_TIMING` recorder，作用是历史归因，不是新 staged profiler 的结果。

## 2. GitHub 同步阶段

| 阶段 | 状态 | 说明 |
|---|---|---|
| 刷新 PR/fork ref | 完成 | GitHub head、`origin/fullft-development`、`upstream/pr-2086` 均为 `1e95053` |
| 保存本地 timing 现场 | 完成 | 原 6+2 文件保存在 `stash@{0}`，不属于当前代码树 |
| Fast-forward | 完成 | `f209878 -> 1e95053`，8 commits |
| 工作树一致性 | 完成 | tracked working tree clean |
| 新 head 性能测试 | 未执行 | 2026-07-20 运行仍使用旧安装环境，未重建 `1e95053` |

完整 Git 对象和文件差异见 [AMXBF16-full-FT-gitdiff.md](AMXBF16-full-FT-gitdiff.md)。

## 3. 历史 benchmark 配置

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3-30B-A3B，48 层，128 experts/layer，top-k=8 |
| GPU / CPU | 1 GPU；2 × Intel Xeon Platinum 8488C；2 NUMA |
| 后端 / 精度 | AMX BF16 / BF16 |
| KT workers / OMP | 96 / `OMP_NUM_THREADS=96` |
| Batch / GAS | 1 / 1 |
| 序列长度 | 4096 |
| tokens/step | 4096 |
| 学习率 | `1.0e-5` |
| 步数 | 15；跳过前 5 步，统计 steps 6–15 |
| Gradient checkpointing | 开启 |
| Optimizer | PyTorch fused AdamW / AcceleratedOptimizer |

2026-07-14 的部分旧报告使用 1024 tokens/step，不能与这里的 4096-token run 只按 step 秒数直接比较。

## 4. Benchmark 演进

| 运行 | 代码/用途 | Step | TPS | Backward | Optimizer | Requant |
|---|---|---:|---:|---:|---:|---:|
| `20260715_203338.../full_ft` | `f209878` dW 优化前对照 | 19.252 s | 212.76 | 9.281 s | 1.791 s | 6.329 s |
| `20260716_143647.../full_ft` | `f209878` 优化后、无内部 recorder | **16.140 s** | **253.78** | **6.792 s** | 1.925 s | **5.732 s** |
| `20260716_175359.../full_ft` | 旧 recorder 的 backward 归因 | 17.362 s | 235.92 | 6.947 s | **1.757 s** | 6.746 s |
| `20260720_130854.../full_ft` | 旧 recorder 的最新稳定归因 | 17.032 s | 240.49 | **6.671 s** | 2.257 s | 6.327 s |
| `1e95053` | 今日 GitHub head | 未测 | 未测 | 未测 | 未测 | 未测 |

前两组相邻 `full -> lora` 会话的 `session_config.json` 文本一致。`f209878` 后：

- Full backward：`9.28085 -> 6.79183 s`，-26.82%；
- Full step：`19.25197 -> 16.13997 s`，-16.16%；
- TPS：`212.76 -> 253.78 tok/s`，+19.28%；
- 同期 LoRA-only backward：-2.74%。

LoRA 旁路支持收益集中在 Full-only base dW，但 forward 和 requant 也有运行间变化，因此不能把全部 TPS 增益都严格归因给一个函数。

## 5. DeepSpeed 与最新 KT Full-FT 对比

对照运行为：

```text
DeepSpeed: 20260714_154355_1gpu_DEEPSPEED_Z3_OFFLOAD_BF16_FULL_THEN_LORA/full_ft
KT:        20260720_130854_1gpu_AMX_BF16_FULL/full_ft
```

两者都是 Qwen3-30B-A3B Full-FT、单卡、BF16、batch=1、GAS=1、seq=4096、15 steps，并统一用 step probe 的 steps 6–15 稳定均值。DeepSpeed 运行未记录 `OMP_NUM_THREADS`/实际 CPU affinity；KT 明确为 OMP=96。因此总体 TPS 可作为完整系统对比，但 CPU optimizer 的线程级归因不是严格同构 A/B。

| 稳定区间指标 | DeepSpeed ZeRO-3 CPU Offload | 最新 KT 旧安装环境 | KT 相对 DeepSpeed |
|---|---:|---:|---:|
| Step | 27.2399 s | **17.0316 s** | **-10.2083 s / -37.48%** |
| TPS | 150.37 tok/s | **240.49 tok/s** | **1.599× / +59.94%** |
| Forward | 3.2919 s | **1.4085 s** | -1.8834 s |
| 探针记录的 backward 桶 | 23.9330 s | 6.6708 s | **不可直接比较** |
| 显式 optimizer 桶 | 0 s（未捕获） | 2.2567 s | **DeepSpeed 的 0 不是零成本** |
| Post optimizer | 包含在 `engine.step()` | 0.3319 s | 边界不同 |
| KT base-weight reload | 无派生 AMX BufferB | 6.3271 s | KT 独有布局维护成本 |

最新 KT TPS 仍然属于 `f209878 + KT_BACKWARD_TIMING` 旧安装环境，不是当前 `1e95053` 源码的 benchmark。上表回答“日志中最新 KT 结果”，不改变新 head 仍待重建和重测的版本边界。

### 5.1 DeepSpeed 的 `optimizer=0` 为什么不成立

DeepSpeed 日志确认启用了 CPU offload 和 AVX512 `DeepSpeedCPUAdam`：

```text
Adam Optimizer #0 is created with AVX512 arithmetic capability.
```

本次环境的 Accelerate 1.11.0 `DeepSpeedEngineWrapper.backward()` 在 GAS 边界依次调用 `engine.backward()` 和 `engine.step()`。本次 GAS=1，所以每个 step 的 CPUAdam、gradient clipping、zero-grad 以及 DeepSpeed 内部 scheduler 都在探针的 `accelerator.backward()` 时间窗口内完成。Trainer 外层 optimizer 接口在 DeepSpeed 模式下是 no-op，所以显式 `optimizer_sec` 记为 0。

DeepSpeed 配置还设置了 `wall_clock_breakdown: false`，日志没有 `engine.backward` 与 `engine.step` 的子边界。因此当前只能得到：

```text
GPU backward + ZeRO gradient partition/offload + CPUAdam + clip/zero/scheduler
= 23.9330 s/step
```

不能从这份历史日志得出 DeepSpeed CPUAdam 本身是 0 s，也不能用 `23.933 / 2.257 = 10.6×` 宣称 KT optimizer 快 10.6 倍。2026-07-21 已在 `step_timing_probe.py` 实现方案二的运行时包装，但尚未执行新的 Full-FT 训练，因此上表数值和结论不变。新探针记录四个嵌套原始墙钟：

```text
accelerator.backward（既有 backward_sec）
├── ds_engine_backward_sec
└── ds_engine_step_sec
    └── ds_zero_optimizer_step_sec
        └── ds_cpu_adam_step_sec（同一步内多次调用求和）
```

同时输出 wrapper、engine-step 除 ZeRO 之外、ZeRO 除 CPUAdam 之外三个非负残差和各层调用次数。所有 `ds_*` 都是诊断字段，不加入 `ATTRIBUTED_PHASES`，也不能彼此相加；否则会在 TPS 分母中重复计算。`DS_PROBE_MODE=exact` 在 DeepSpeed 边界执行 CUDA 同步以获得 completed-work 墙钟，可能打断异步重叠；`low_overhead` 只记录 host wall，扰动更小但 GPU 异步阶段不够精确。正式归因应至少做 `off/exact` 或 `low_overhead/exact` 同配置对照。

### 5.2 KT optimizer 的真实时间应拆为三层

最新 KT 训练日志实际显示 `AdamW` 和 `AcceleratedOptimizer`，并向其注入 144 个 CPU BF16 expert Parameter；`step_timing.md` 中“CPU DeepSpeedCPUAdam”是旧模板文案，不代表本次运行使用 CPUAdam。在 OMP=96 下：

| KT 更新子阶段 | 耗时 | Step 占比 | 含义 |
|---|---:|---:|---|
| `optimizer.step()` | 2.2567 s | 13.25% | PyTorch fused AdamW 更新权威 Parameter |
| post optimizer | 0.3319 s | 1.95% | scheduler、`zero_grad` 和 KT pointer/dirty 管理 |
| `update_base_weights()` | 6.3271 s | 37.15% | 下一次 forward 前重建 AMX BufferB |
| 更新链合计 | **8.9158 s** | **52.35%** | 不能全部命名为“optimizer 计算” |

因此 KT 的显式 AdamW 不是当前第一单项瓶颈；AMX 权重 reload 是 optimizer 本体的 2.80 倍，占整个更新链的 71.0%。只不现实地删掉 `optimizer.step()` 的 Amdahl 上限是把 TPS 从 240.49 提到约 277.2 tok/s；若连 post optimizer 和 reload 也全部删掉，算术上限才是约 504.7 tok/s。后者只是阶段预算，不是可实现预测。

与前两次同家族 KT 日志相比，显式 optimizer 均值从 1.757/1.925 s 变为 2.257 s，即比最快的旧 timing run 慢 28.4%、比无内部 recorder 基线慢 17.2%。但最新稳定 steps 中 optimizer 只在 2.168–2.334 s 之间，说明这次运行内部稳定；跨运行差异仍需交替重复测试才能归因给系统负载、NUMA/first-touch 或代码。

### 5.3 可比的是“完整 step 路径”，不是显式 optimizer 列

把两边统一到“forward 结束后，直到权重已可用于下一 step”的粗粒度边界：

```text
DeepSpeed: backward/ZeRO/CPUAdam 合并桶                       = 23.9330 s
KT: backward + clip + optimizer + post-optim + base reload          = 15.6101 s
差值                                                                  =  8.3230 s
```

这 8.323 s 占 KT 整体 10.208 s step 优势的 81.5%；另外约 1.883 s（18.5%）来自 KT 更快的 forward。这说明最新 KT 已在完整更新周期上明显快于旧 DeepSpeed 基线，但由于 DeepSpeed 合并桶未拆分，不能进一步声称这 8.323 s 中有多少是 CPUAdam、纯 GPU backward 或 ZeRO offload 的差异。

另外，DeepSpeed 原生 Full-FT 日志记录 30.532B trainable parameters 并将可训练参数保持为 FP32；KT 把约 28.991B expert 参数外置为 CPU BF16 Parameter，训练日志的可见模型树只显示约 1.55B，随后再注入 144 个 expert Parameter。两者保持 Full-FT 语义，但 optimizer 的 dtype、tensor 布局和数据流不同，即使下次拆出单独 CPUAdam 时间，也不能当成只替换 optimizer 实现的微基准。

## 6. 旧内部 timing 的历史拆解与 3× 目标

`20260720_130854...` 的 step probe：

| 阶段 | 平均耗时 | Step 占比 |
|---|---:|---:|
| Backward | 6.671 s | 39.2% |
| Requant / update_base_weights | 6.327 s | 37.1% |
| Forward | 1.408 s | 8.3% |
| Optimizer | 2.257 s | 13.3% |
| Post optimizer | 0.332 s | 1.9% |
| 其余 | 0.037 s | 0.2% |
| 总计 | 17.032 s | 100% |

旧 recorder 的 backward 子项：

| 子项 | 每 step | Outer backward 占比 |
|---|---:|---:|
| checkpoint 重算 | 1.991 s | 29.8% |
| base-weight dW | 1.999 s | 30.0% |
| Down dX | 0.637 s | 9.5% |
| Gate/Up dX | 0.635 s | 9.5% |
| 全量梯度清零 | 0.336 s | 5.0% |
| backward repack wait | 0.238 s | 3.6% |

这些数据说明旧 head 的 dW、checkpoint 和 reload 都值得继续研究。它们不提供新 driver 或 direct reload 的实测增益。

以同一 step probe 口径，`3 × forward = 4.225 s`，当前 backward 还需减少 2.445 s（36.66%）。移除全部 checkpoint recompute 的算术上界仍是 4.680 s（3.32×）；再完全消除 buffer clear 后约为 4.344 s（3.08×），还需约 0.119 s。该最后缺口相当于隐藏一半 repack wait，或把 dW 墙钟再降低约 6%。这些是预算推导，不是未测试配置的结果。

关闭 checkpointing 是否可用取决于显存：该运行峰值为 39.57/49.14 GB，表面余量约 9.57 GB，但尚无 no-checkpoint 峰值或 OOM 实测。若关闭 checkpoint 不能运行，则必须用 selective checkpoint 加上 dW/dX、清零和 CPU/GPU overlap 的组合优化；完整保留 checkpoint 时，3× 目标风险较高。

## 7. 今日 head 可能改变的阶段

`f209878..1e95053` 的代码变化可能影响：

1. **dW：** 通用 `BF16DWeightKernel`、active-expert batch task、common AMX driver；
2. **Backward 可观测性：** Gate/Up、Down、pack A/B、kernel、store 的 staged timing；
3. **Reload：** 从完整 BF16 Parameter 按 stride direct pack，减少临时分片和 memcpy；
4. **Forward/dX：** SFT 复用 inference BF16 kernel 和 BufferB 布局转换。

“可能影响”是源码推断，不是性能结论。尤其 direct reload 仍执行完整 BufferB pack，不能预先写成 requant 已消失。

## 8. 新 head 后续 benchmark 口径

未来测试 `1e95053` 或其后继 commit 时，应固定：

- 单卡、2 NUMA、BF16、batch=1、GAS=1、seq=4096；
- OMP=96 和实际 CPU affinity；
- 15 steps 以上，至少跳过前 5 步；
- TPS 使用 step probe 的 `tokens_per_step / stable_mean_step_sec`；
- 同时报告 forward、backward、optimizer、post-optimizer、base reload；
- DeepSpeed optimizer 专项使用 `DS_PROBE_MODE=exact`，报告 engine backward、engine step、ZeRO step、CPUAdam step、三个残差和调用次数；
- `KT_SFT_PROFILE=0/1` 用同一二进制交替测试 profiler overhead；
- profiler 的 wrapper、`tp.<index>` 和 worker CPU time 不得相加；
- 使用相同 commit 做 Full/LoRA，必要时再做 Hybrid/checkpoint on/off。

若要严格比较新旧 dW/reload，应采用：

```text
old -> new -> old -> new
```

并记录系统负载、NUMA first-touch 和每层路由分布。单次 `f209878` 与单次 `1e95053` 之间的差异不能全部归因给源码。

DeepSpeed Full-FT-only 探针入口为：

```bash
cd /mnt/data2/wbw/FFTtest/Qwen3-30B-A3B

bash run_deepspeed_full_ft_probe.sh
```

默认是单卡、batch=1、GAS=1、seq=4096、35 steps、跳过前 5 steps、`DS_PROBE_MODE=exact` 和 OMP=96；只执行 `--mode full`。可用 `STEPS`、`WARMUP_STEPS`、`FFT_OMP_NUM_THREADS` 等环境变量调整，或设置 `DS_PROBE_MODE=low_overhead` 做扰动对照。当前只完成合成调用链测试和 runner dry-run，没有执行 30B Full-FT，因此不能在本文填入新的 TPS/optimizer 数字。

## 9. 正确性和报告边界

- 历史两组 Full 运行完成 15/15 steps，loss 和 grad norm 有限；这是旧 head 证据。
- 15-step 健康检查不替代收敛、TP、多 GPU、Hybrid 或逐元素 PyTorch reference。
- 最新 PR head 没有 GitHub commit status。
- 本轮没有运行新 head 的 build、聚焦测试或训练；文档不得把历史 `253.78 tok/s` 标为 `1e95053` 结果。
- 首步 backward 的异常主要受约 54 GiB gradient buffer first-touch 影响，稳定性能只统计 warmup 后区间。

### 可视化输出约定

自 2026-07-20 起，共用 `Qwen3-30B-A3B/analyze.py` 的测试 runner 只生成三张连续编号的性能图：

```text
01_gpu_memory.png
02_cpu_ram.png
03_tps.png
```

Loss、grad norm、NaN/Inf 和各阶段汇总仍由分析脚本解析并写入 `summary.md`，但不再单独生成旧 `03_training_loss.png`、`04_grad_norm.png`、`05_nan_inf_timeline.png`、`06_phase_summary.png`。历史 `test_log` 中的这四类图已删除，保留的旧 `07_tps.png` 已统一改名为 `03_tps.png`；重新分析旧目录时也会清理遗留文件，避免同时出现新旧编号。

## 10. 结果文件

- [历史无内部打点基线 summary](../../Qwen3-30B-A3B/test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft/summary.md)
- [历史内部 timing step report](../../Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/step_timing.md)
- [历史 backward internal timing](../../Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/backward_internal/backward_timing.md)
- [历史 FLOPs analysis](../../Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/flops_analysis.md)
- [2026-07-20 最新旧环境 step report](../../Qwen3-30B-A3B/test_log/20260720_130854_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/step_timing.md)
- [2026-07-20 最新旧环境 backward timing](../../Qwen3-30B-A3B/test_log/20260720_130854_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/backward_internal/backward_timing.md)
- [DeepSpeed 对照 step report](../../Qwen3-30B-A3B/test_log/20260714_154355_1gpu_DEEPSPEED_Z3_OFFLOAD_BF16_FULL_THEN_LORA/full_ft/phase4/step_timing/step_timing.md)
- [DeepSpeed 对照训练日志](../../Qwen3-30B-A3B/test_log/20260714_154355_1gpu_DEEPSPEED_Z3_OFFLOAD_BF16_FULL_THEN_LORA/full_ft/phase4/train_full_ft.log)
- [当前 Git diff 与同步状态](AMXBF16-full-FT-gitdiff.md)
- [当前 backward 设计与计时边界](AMXBF16-full-FT-backward.md)

## 11. 当前结论

1. 代码同步已完成，`ktransformers` 当前为干净的 `1e95053`。
2. 最新旧安装环境实测为 240.49 tok/s、6.671 s backward 和 1.408 s forward；backward/forward 为 4.74×。
3. 压到 3× 需要至少减少 2.445 s；关闭 checkpointing 单独不够，必须再缩减清零并取得至少约 0.119 s 的 repack/dW/重叠收益。
4. 新 head 同时改变 dW、BF16 kernel/BufferB 复用和 reload，下一次必须重新建立同 commit benchmark。
5. 在新 benchmark 出现前，本文保留历史数值用于规划，但所有 `1e95053` 性能栏均明确为“未测”。
6. 同一旧 step-probe 口径下，最新 KT 为 240.49 tok/s，是 DeepSpeed 150.37 tok/s 的 1.599 倍；历史 DeepSpeed CPUAdam 仍折叠在 23.933 s 合并桶内。新的四层探针代码已经就绪，但在 Full-FT 实测产出前仍不能单独比较 optimizer 耗时。
7. KT 显式 AdamW 为 2.257 s，而 optimizer + post-optim + AMX base reload 合计 8.916 s（52.35% step）；后续的更大空间在 reload/更新链，不是只替换 Adam 实现。

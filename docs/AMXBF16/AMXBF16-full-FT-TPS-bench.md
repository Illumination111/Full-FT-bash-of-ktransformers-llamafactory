# Qwen3-30B-A3B AMX BF16 Full FT TPS Benchmark

对应 PR：[kvcache-ai/ktransformers#2086](https://github.com/kvcache-ai/ktransformers/pull/2086)

当前 PR head：`f2098786f02ae4cd1d3f6a2968e15df7dfdf83fc`（`fullft-development`）

更新时间：2026-07-17

## 1. 当前结论

当前最适合作为“无 backward 内部 recorder 的性能基线”的运行是：

```text
20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft
```

其稳定区间为：

```text
16.140 s/step
253.78 tok/s
backward 6.792 s
requant  5.732 s
```

最新内部打点运行是：

```text
20260716_175359_1gpu_AMX_BF16_FULL/full_ft
```

其 step probe 结果为 `17.362 s/step、235.92 tok/s`。这组运行用于 backward 归因，不应在没有同二进制 `off/summary` 交替实验时被解释为 recorder 固定损耗。相比前一运行，主要回退发生在 requant（+1.014 s）和 forward（+0.184 s），backward 只增加0.155 s。

当前端到端瓶颈不再是旧报告中的51.6 s optimizer，而是几乎并列的：

- backward：6.947 s，占40.0%；
- requant/update_base_weights：6.746 s，占38.9%。

因此后续 TPS 优化必须同时覆盖 backward 和 requant。只把 dW 提升到当前 dX 的有效吞吐，预计约节省0.728 s，TPS约从235.9提升到246.2 tok/s；这不是足以单独改变全局瓶颈的量级。

## 2. 代码和 GitHub 状态

截至本次核验，PR #2086仍为 open、可合并、非 draft，base 为`main`，head 为`fullft-development@f209878`。GitHub PR 快照包含9个 commits、16个 changed files、约`+1215/-160`。

`f209878`已经提交了 AMX Full-FT dW strip/panel/C-residency 优化；当前本地 worktree 另有未提交的 backward timing 代码。提交态和本地态的边界见 [AMXBF16-full-FT-gitdiff.md](AMXBF16-full-FT-gitdiff.md)。

Full FT 链路仍为：

```text
CPU BF16 gate/up/down Parameter
  -> C++/AMX backward 写三组 base gradient
  -> AdamW/AcceleratedOptimizer 更新权重
  -> update_base_weights 重新量化到 AMX BufferB
  -> 下一 step forward
```

## 3. 当前测试配置

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3-30B-A3B，48层，128 experts/layer，top-k=8 |
| GPU / CPU | 1 GPU；2 × Intel Xeon Platinum 8488C；2 NUMA |
| 后端 / 精度 | AMXBF16 / BF16 |
| KT workers | 96，`OMP_NUM_THREADS=96` |
| Batch / GAS | 1 / 1 |
| 序列长度 | 4096 |
| tokens/step | 4096 |
| 学习率 | `1.0e-5` |
| 步数 | 15；跳过前5步，统计6–15共10步 |
| Gradient checkpointing | 开启 |
| 内部打点运行 | `KT_BACKWARD_TIMING=summary`，48层全量 |

这里的配置来自 `20260716_175359.../session_config.json`。历史2026-07-14报告使用1024 tokens/step，不能和当前4096-token结果只按秒数直接比较。

## 4. Benchmark 演进

| 运行 | 用途 | Step | TPS | Backward | Optimizer | Requant |
|---|---|---:|---:|---:|---:|---:|
| `20260715_203338.../full_ft` | dW优化前、同配置对照 | 19.252 s | 212.76 | 9.281 s | 1.791 s | 6.329 s |
| `20260716_143647.../full_ft` | dW优化后、当前无内部打点基线 | **16.140 s** | **253.78** | **6.792 s** | 1.925 s | 5.732 s |
| `20260716_175359.../full_ft` | backward内部归因 | 17.362 s | 235.92 | 6.947 s | **1.757 s** | 6.746 s |

前两组相邻 `full -> lora` 会话配置文本一致。dW优化后：

- Full backward：`9.28085 -> 6.79183 s`，降低26.82%；
- Full step：`19.25197 -> 16.13997 s`，降低16.16%；
- TPS：`212.76 -> 253.78`，提高19.28%；
- 同期不执行 Full-only dW 的 LoRA backward只变化-2.74%。

这支持收益主要来自`f209878`中的 Full-only dW优化，但 forward和requant也有运行间变化，所以不能把全部TPS增益都归给一个函数。

2026-07-14的`88.83 s/step、11.53 tok/s`结果仅保留为历史故障演进记录。当时 optimizer为51.588 s且序列长度为1024；后续OMP线程和optimizer路径已变化，它不再代表当前瓶颈或当前性能。

## 5. 最新稳定 step 拆解

数据来自 `20260716_175359...` 的 step probe：

| 阶段 | 平均耗时 | Step占比 | 作用 |
|---|---:|---:|---|
| Backward | 6.947 s | 40.0% | GPU autograd、checkpoint重算、CPU AMX expert dX/dW及同步 |
| Requant / update_base_weights | 6.746 s | 38.9% | 将更新后的BF16 expert权重重新量化为AMX BufferB |
| Forward | 1.530 s | 8.8% | GPU non-expert与CPU AMX expert forward；不含单列requant |
| Optimizer | 1.757 s | 10.1% | 日志实际为AdamW + AcceleratedOptimizer |
| Post optimizer | 0.347 s | 2.0% | pointer sync、scheduler、zero-grad等 |
| 其余 | 0.035 s | 0.2% | grad clip、data、log和未归因小项 |
| 总计 | 17.362 s | 100% | `4096 / 17.362 = 235.92 tok/s` |

`step_timing.md`的 phase legend仍残留“CPU DeepSpeedCPUAdam”模板文案；训练日志显示本次没有使用DeepSpeedCPUAdam。

## 6. Backward 对 TPS 的限制

最新内部打点给出的稳定均值：

| Backward子项 | 每step | Outer backward占比 |
|---|---:|---:|
| checkpoint重算 | 2.079 s | 29.9% |
| base-weight dW | 2.036 s | 29.3% |
| Down dX | 0.665 s | 9.6% |
| Gate/Up dX | 0.643 s | 9.3% |
| 全量梯度清零 | 0.340 s | 4.9% |
| backward repack wait | 0.313 s | 4.5% |
| non-KT residual | 0.463 s | 6.7% |

dW、checkpoint、清零和repack wait合计4.768 s，占outer backward的68.6%，但不是都能删除：dW是必要计算，checkpoint用时间换内存，清零受梯度累积语义约束，repack只有在成功重叠时才能隐藏。

单项Amdahl估算：

| 假设 | 节省/step | 估算TPS |
|---|---:|---:|
| dW耗时降低25% | 0.509 s | 243.0 |
| dW达到当前dX有效吞吐 | 0.728 s | 246.2 |
| checkpoint重算降低25% | 0.520 s | 243.2 |
| 完全跳过本次全量清零 | 0.340 s | 240.6 |
| 完全隐藏repack wait | 0.313 s | 240.3 |

更完整的设计、计时口径和下一步优化顺序见 [AMXBF16-full-FT-backward.md](AMXBF16-full-FT-backward.md)。

## 7. 当前优化优先级

1. 对dW继续拆分Gate/Up、Down、packing、AMX compute、BF16 store，并采集AMX/cycles/cache/内存带宽计数器；目标不是减少源码中的for，而是减少动态packing、任务派发和tile load/store。
2. 评估selective checkpoint，用额外activation内存换取0.5–1.0 s量级的重算时间。
3. 在证明完整覆盖语义后做条件清零；理论上限0.340 s。
4. 提前提交backward repack并分析与checkpoint共享worker pool的竞争；理论等待上限0.313 s。
5. 独立拆解requant。即使backward降到5.5–6.0 s，当前6.746 s requant仍会成为第一瓶颈。

下一阶段可把backward `5.5–6.0 s`作为验证目标而非承诺。在requant不变时，完整step约为15.9–16.4 s，对应约250–257 tok/s。

## 8. 正确性和报告口径

- 两个最新Full运行均完成15/15 steps并正常退出；最新运行loss从0.7773下降到0.2930，记录的grad norm为有限值。
- 15-step健康检查不能替代完整收敛验证；TP、多GPU、hybrid以及逐元素PyTorch reference仍需单独回归。
- 内部计时没有新增CUDA synchronization；GPU-facing Python host-wall边界可能把异步工作归到后续既有阻塞点。
- TPS统一采用step probe：`tokens_per_step / stable_mean_step_sec`。最新值为235.92 tok/s；`summary.md`中的211.6 tok/s来自tqdm日志时间戳，不作为本报告主口径。
- 首步backward 18.212 s主要由约54 GiB梯度buffer first-touch造成；稳定性能只统计steps 6–15。

### 可视化输出约定

自 2026-07-20 起，共用 `Qwen3-30B-A3B/analyze.py` 的测试 runner 只生成三张连续编号的性能图：

```text
01_gpu_memory.png
02_cpu_ram.png
03_tps.png
```

Loss、grad norm、NaN/Inf 和各阶段汇总仍由分析脚本解析并写入 `summary.md`，但不再单独生成旧 `03_training_loss.png`、`04_grad_norm.png`、`05_nan_inf_timeline.png`、`06_phase_summary.png`。历史 `test_log` 中的这四类图已删除，保留的旧 `07_tps.png` 已统一改名为 `03_tps.png`；重新分析旧目录时也会清理遗留文件，避免同时出现新旧编号。

## 9. 结果文件

- [最新 step timing](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/step_timing.md)
- [最新 backward internal timing](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/backward_internal/backward_timing.md)
- [最新 FLOPs analysis](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/full_ft/phase4/step_timing/flops_analysis.md)
- [最新 session config](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_175359_1gpu_AMX_BF16_FULL/session_config.json)
- [无内部打点基线 summary](../../FFTtest/Qwen3-30B-A3B/test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA/full_ft/summary.md)
- [dW优化设计和Git diff](AMXBF16-full-FT-gitdiff.md)

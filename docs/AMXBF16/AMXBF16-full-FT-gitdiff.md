# PR #2086、本地代码树与 AMX Full-FT Git Diff

更新时间：2026-07-21

## 1. 当前结论

本地仓库已经同步到 GitHub 当日最新 PR head，tracked 代码树保持一致：

```text
仓库          /mnt/data2/wbw/ktransformers
分支          fullft-development
本地 HEAD     1e95053b15b32e6db8193fd852d62d051c6e7ef5
fork tracking origin/fullft-development = 1e95053
官方 PR ref   upstream/pr-2086           = 1e95053
working tree  clean
```

GitHub PR 为 [kvcache-ai/ktransformers#2086](https://github.com/kvcache-ai/ktransformers/pull/2086)。2026-07-17 刷新时状态为 open、非 draft、mergeable；head 没有 commit status。`mergeable=true` 只说明 GitHub 可以生成合并结果，不代表新 head 已完成 CI、扩展构建或 Full-FT 训练验收。

同步前的 6 个 tracked 修改和 2 个 untracked timing 文件没有混入当前工作树，已完整保存在：

```text
stash@{0}: pre-1e95053 KT_BACKWARD_TIMING integration 2026-07-17
```

该 stash 是恢复副本，不属于 `1e95053`，也不能计入 PR diff。不要在未重新设计 profiler 接口前直接 `stash pop`。

## 2. 本轮同步阶段

### 阶段 A：刷新与核对

- 通过 GitHub PR 元数据确认 head 仍为 `1e95053`；
- 重新 fetch `origin/fullft-development` 与 `refs/pull/2086/head`；
- 确认旧本地 HEAD `f209878` 是新 head 的祖先，可 fast-forward，落后 8 个 commit，没有分叉；
- 确认原工作树仍是记录中的 6 个修改和 2 个新文件。

### 阶段 B：保存本地现场

使用包含 untracked 文件的 stash 保存原 timing 实验：

```text
6 tracked files: 247 insertions, 2 deletions
2 untracked files: 373 lines
合计 8 files: 620 insertions, 2 deletions
```

原实现单独维护 `KT_BACKWARD_TIMING` C++ struct、pybind API 和 Python recorder。GitHub 新增的 `KT_SFT_PROFILE` 已覆盖并细化大部分 C++ 阶段，两套实现不能直接叠加。

### 阶段 C：fast-forward 并保持一致

本地分支从 `f209878` fast-forward 到 `1e95053`。按本轮最终要求，没有继续合并 stash，也没有在 `ktransformers` 中保留新的未提交代码；因此当前 tracked 工作树与 fork/PR head 一致。

### 阶段 D：阶段性文档迁移

本轮只更新 `FFTtest` 下的分析文档，明确新旧 commit、历史性能证据和未完成验证。没有修改已经同步干净的 `ktransformers` 代码树，也没有按用户最新要求继续运行测试。

## 3. 三种 Git 比较口径

### 3.1 PR 相对官方 base

PR base 为 `upstream/main@7c021b4`，PR head 为 `1e95053`；共同祖先是 `8e46e58`。三点 diff：

```text
git diff 7c021b4...1e95053
25 files changed, 2727 insertions(+), 260 deletions(-)
17 commits
```

必须使用三点 diff 或共同祖先统计 PR 内容。直接用两点 `git diff upstream/main fullft-development` 会同时混入官方 main 在共同祖先之后、但个人分支没有包含的提交。

### 3.2 旧本地 head 到今日 GitHub head

本轮真正同步的范围是：

```text
f209878..1e95053
15 files changed, 1524 insertions(+), 112 deletions(-)
8 commits
```

文件级净变化：

| 文件 | + | - | 主要作用 |
|---|---:|---:|---|
| `operators/sft_profile.hpp` | 239 | 0 | 通用 staged profiler 与阶段枚举 |
| `operators/amx/la/bf16_dweight.hpp` | 168 | 0 | 共用 BF16 dWeight driver、scratch 和 worker timing |
| `operators/amx/sft_moe.hpp` | 272 | 9 | 新 dW 调度、细粒度阶段、BF16 kernel/BufferB 复用 |
| `operators/moe-sft-tp.hpp` | 136 | 58 | TP/NUMA profiler、direct reload 与 merge 路径 |
| `operators/amx/la/amx_raw_buffers.hpp` | 112 | 5 | raw BF16 unpack、转置 pack 与 strided/direct pack |
| `python/sft/profiler.py` | 149 | 0 | profile 收集、聚合和格式化 |
| `python/sft/autograd.py` | 20 | 13 | PyTorch profiler 边界 |
| `python/sft/layer.py` | 29 | 25 | forward/recompute/repack profile 边界 |
| 三个聚焦测试 | 387 | 0 | profiler、raw repack、dWeight reference/benchmark |
| 其他 binding/CMake/export | 12 | 2 | API 暴露与构建接入 |

### 3.3 当前 working tree

当前 `git status --short --branch` 仅显示：

```text
## fullft-development...origin/fullft-development
```

因此当前 working-tree diff 为零。stash 不会出现在 `git diff` 中；它只表示存在可恢复的历史现场。

## 4. 2026-07-17 新增的 8 个 commit

| commit | 作用 | 当前树净影响 |
|---|---|---|
| `9dc9d93` | staged SFT profiling | 新增 C++/Python profiler、pybind API 和测试 |
| `c6f4211` | SFT 复用 inference BF16 kernel | 统一 `GemmKernel224BF16`，增加 raw BufferB 转换能力 |
| `109b403` | Full-FT 细粒度 profiling | 区分 initial/recompute forward、dWeight、reload 与 PyTorch 边界 |
| `34d2102` | batch BF16 Full-FT weight gradients | 新 `BF16DWeightKernel`、批量 dW 任务与 direct BF16 reload |
| `ea84e6e` | dWeight benchmark | 增加 common driver 与 legacy 路径的性能门槛 |
| `273e670` | profiler 标签修正 | 将 inner store 明确标为 worker CPU 累积时间 |
| `61e63a8` | 临时性能文档 | 添加报告 |
| `1e95053` | 删除临时性能文档 | 与上一 commit 对最终树净变化为零 |

所有 8 个 commit 的 committed date 都是 2026-07-17；`9dc9d93` 的 author date 较早不改变这一同步口径。

## 5. 当前 head 的实现变化

### 5.1 C++ profiler 成为唯一已提交计时源

`SFTProfiler` 由 `KT_SFT_PROFILE` 在对象创建时启用，使用原子累计值记录：

- NUMA-local forward、backward、Down、Gate/Up、activation、router；
- base-weight dW 的 offsets、matmul、pack A/B、Gate/Up kernel、Down kernel、store；
- TP forward/backward、buffer clear、NUMA compute、grad merge；
- backward repack 与 base-weight reload 的 partition/pack/cleanup。

pybind 暴露 `get_profile_stats(reset=False)` 和 `reset_profile_stats()`；Python 提供 `collect_kt_sft_profile()`、`reset_kt_sft_profile()` 与格式化表格。`tp.<index>` 是各 NUMA-local 子核的累计 scope；并行 worker 的 CPU 累积时间不能当成外层墙钟。

### 5.2 `f209878` dW 原型被通用 driver 取代

`f209878` 证明了 fixed-`i_tile` strip、线程局部 panel 和 C tile 跨 K 常驻的方向有效。新 head 没有简单保留旧函数内手写循环，而是抽出 `BF16DWeightKernel`：

- 复用 inference 的 `GemmKernel224BF16` driver；
- 只为 active experts 创建任务；
- Gate/Up 共用 input panel；Down 固定 `i_tile` 复用 intermediate panel；
- 使用线程局部对齐 scratch；
- 将 pack、kernel、store 分别累计到 staged profiler；
- 保留 fallback 路径及更细的正确性/性能测试。

因此旧 `f209878` 的源码 diff 只能作为历史设计依据，不能再描述成 `1e95053` 的当前实现。

### 5.3 BF16 weight reload 改为 direct pack

新 head 可从完整 CPU BF16 Parameter 按 TP/NUMA stride 直接 pack 到 forward BufferB，并生成 backward 转置 BufferB，减少旧路径的临时 gate/up/down 分片分配和 memcpy。

这不等于“消除了 requant/reload”：完整 BufferB pack 仍会执行，只是数据路径和中间分配更直接。新 profiler 已把 direct pack、forward pack、backward pack、partition 和 cleanup 分开。

## 6. 性能证据的归属

`20260715_203338...` 与 `20260716_143647...` 的 A/B 证明 `f209878` 原型有效：

| 指标 | 修改前 | `f209878` 后 | 变化 |
|---|---:|---:|---:|
| Step | 19.25197 s | 16.13997 s | -16.16% |
| TPS | 212.76 | 253.78 | +19.28% |
| Backward | 9.28085 s | 6.79183 s | -26.82% |

LoRA-only backward 同期只变化 -2.74%，支持收益集中在 Full-only dW 路径。但这些运行都早于 `9dc9d93..1e95053`，不能直接当作新 head 的性能结果。

`20260716_175359...` 的内部 timing 也来自旧的本地 `KT_BACKWARD_TIMING` 实现。它可以继续作为历史热点证据，但字段不能与新 `SFTProfiler` 输出逐列混用。

## 7. 验证状态

本轮目标是同步代码树与迁移文档，按用户最新要求没有继续运行测试。当前可以确认的是 Git 与代码树状态，不是运行时正确性：

- 已确认本地、fork tracking ref、官方 PR ref 三者同为 `1e95053`；
- 已确认 working tree clean；
- 已确认 PR head 没有 commit status；
- 没有在本轮对 `1e95053` 执行 extension build、profiler pytest、AMX dWeight/repack 测试或 Qwen3 训练。

因此不得写成“新 head 已构建通过”或“新 head TPS 为 253.78”。

## 8. 后续恢复边界

若以后继续整合旧 timing stash，应遵循：

1. 先从 stash 导出/查看 diff，不直接 pop 到当前开发分支；
2. 复用 `SFTProfiler` stage、pybind 和 Python API，不恢复重复 C++ timing struct；
3. 只保留通用 profiler 尚未提供的逐 step、microbatch、layer/NUMA trace 和输出归档能力；
4. 另建分支或 worktree，完成冲突整合后再测试；
5. 测试完成前保持 `full/hybrid/lora`、checkpoint on/off 与 direct reload 的验证缺口为“未验证”。

## 9. 只读核对命令

```bash
cd /mnt/data2/wbw/ktransformers

git status --short --branch
git rev-parse HEAD origin/fullft-development upstream/pr-2086
git log --reverse --oneline f209878..1e95053
git diff --shortstat f209878..1e95053
git diff --numstat f209878..1e95053
git diff --shortstat 7c021b4...1e95053
git stash list
git stash show --stat --include-untracked stash@{0}
```

这些命令分别核对当前同步状态、今日 8 个 commit、完整 PR 三点 diff 和本地恢复副本，四种对象不能混为一谈。

## 10. FFTtest working-tree 变更

以下变更发生在独立仓库 `/mnt/data2/wbw/FFTtest`，不改变 `/mnt/data2/wbw/ktransformers@1e95053`；因此不能混入 PR #2086 的 25-file 三点 diff。

### 10.1 2026-07-20 可视化精简

比较基线为：

```text
仓库          /mnt/data2/wbw/FFTtest
分支          agent/add-expert-gradient-probe
HEAD/base     d76dc2bc386e15e289480b235e19b580df2eb50b
提交范围      本节列出的可视化变更
```

本轮相关 tracked diff 为 4 个测试/分析脚本、`39 insertions / 245 deletions`：

- `Qwen3-30B-A3B/analyze.py` 删除 Loss、grad norm、NaN/Inf 和 phase summary 四组绘图函数，只保留 GPU 显存、CPU 内存与 TPS；TPS 从 `07_tps.png` 重编号为 `03_tps.png`，分析旧目录时会清理旧图名；
- `run_finetune_perf_test_bf16.sh` 与 `run_finetune_perf_test_bf16_deepspeed.sh` 的摘要链接同步指向 `03_tps.png`；
- `run_full_ft_test_1gpu_bf16_frozen.sh` 不再引用已移除的 `04_grad_norm.png`，Router 稳定性改为引用训练日志中的数值检查。

`test_log/` 被 `.gitignore` 排除，因此历史图片的删除与 `07_tps.png -> 03_tps.png` 重命名不会出现在 tracked Git diff 中。进入本轮前，`AGENTS.md` 和多份 AMX 文档已经存在其他未提交修改；本轮保留这些修改，没有将其误记为上述 4 个脚本的可视化 diff。

### 10.2 2026-07-21 DeepSpeed backward/optimizer 探针

本轮只修改 FFTtest 的 Python/runner，不修改 DeepSpeed、Accelerate、LLaMA-Factory 或 ktransformers 安装树。相对 `FFTtest@d76dc2b`，本轮探针代码范围为：

```text
Qwen3-30B-A3B/step_timing_probe.py                       +316 / -6
Qwen3-30B-A3B/run_finetune_perf_test_bf16_deepspeed.sh   +22 / -2
Qwen3-30B-A3B/run_deepspeed_full_ft_probe.sh              52 lines（新文件）
代码范围合计                                             +390 / -8
```

实现内容：

- `step_timing_probe.py` 在 Accelerate 完成 DeepSpeed engine 构造后，运行时包装 `DeepSpeedEngine.backward/step`、ZeRO optimizer `step` 与底层 CPU optimizer `step`；
- 新字段作为嵌套 diagnostics 写入既有 `step_timing.json/csv/md`，不加入 TPS attributed phase，另输出三个残差和调用次数；
- `DS_PROBE_MODE=off|low_overhead|exact` 控制关闭、host-wall 低扰动和 CUDA-boundary 精确模式；通用 DeepSpeed runner 默认 `off`，避免历史 benchmark 行为被静默改变；
- `run_deepspeed_full_ft_probe.sh` 强制 `--mode full`，拒绝用户覆盖为 LoRA/both，默认 exact、OMP=96、35 steps/warmup 5，并允许环境变量做 A/B；
- session config、控制台和 summary 记录 probe mode 与 OMP，避免 optimizer 线程配置再次缺失。

验证边界：`py_compile`、两个 Bash 的 `bash -n`、合成四层调用链/调用次数/残差/三种输出检查和 Full-only dry-run 已通过；没有启动 Qwen3-30B-A3B Full-FT，也没有产生新的 TPS、backward 或 CPUAdam 实测。`exact` 的 CUDA 同步会影响异步重叠，必须用同配置 `off` 或 `low_overhead` 伴随运行评估扰动。

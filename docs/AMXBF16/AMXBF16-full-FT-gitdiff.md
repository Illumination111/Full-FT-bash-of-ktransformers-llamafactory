# Full FT AMX backward dW 最小有效修改与 Git Diff

## 结论

本次 dW 修改有效，而且收益主要出现在预期的 Full-only base-weight dW 路径。必须区分三个状态：

1. **GitHub PR head**：PR #2086当前 head 为`fullft-development@f2098786f02ae4cd1d3f6a2968e15df7dfdf83fc`，仍为open；
2. **已提交 dW 优化**：`f209878 [perf](kt-kernel): optimize AMX Full-FT weight gradients`，相对父提交为`1 file changed, 113 insertions(+), 86 deletions(-)`；
3. **当前本地未提交修改**：是后续 backward timing instrumentation，不属于`f209878`，也不在当前GitHub PR head中。

在配置完全相同的相邻两组 `full -> lora` 会话中，Full FT 稳定 backward 从 **9.28085 s** 降到 **6.79183 s（-26.82%）**，完整 step 从 **19.25197 s** 降到 **16.13997 s（-16.16%）**，TPS 从 **212.76** 提升到 **253.78（+19.28%）**。同期不调用该 dW 函数的 LoRA backward 仅变化 **-2.74%**，支持主要收益来自本次代码，而不是机器整体等比例变快。

`f209878`的性能修改只在：

```text
/mnt/data2/wbw/ktransformers/kt-kernel/operators/amx/sft_moe.hpp
```

核心函数是`backward_base_weight_grad()`的AMX BF16分支，提交 diff 为：

```text
1 file changed, 113 insertions(+), 86 deletions(-)
```

该提交没有修改Python/autograd、外部接口、TP布局、权重格式、全局WorkerPool、optimizer、requant，也没有改变`GemmKernel224BF`的`32×32×32` AMX tile形状。

## 当前本地 worktree 与 GitHub head 的差异

当前`/mnt/data2/wbw/ktransformers`的`HEAD`与GitHub PR head相同，均为`f209878`；但本地另有未提交计时代码：

```text
kt-kernel/ext_bindings.cpp                    +40
kt-kernel/operators/amx/sft_moe.hpp           +67
kt-kernel/operators/moe-sft-tp.hpp            +68
kt-kernel/python/sft/amx.py                     +9
kt-kernel/python/sft/autograd.py               +32
kt-kernel/python/sft/base.py                 +31/-2
kt-kernel/operators/sft-backward-timing.hpp    79 lines, untracked
kt-kernel/python/sft/backward_timing.py        294 lines, untracked
```

已跟踪文件合计`247 insertions, 2 deletions`，另有373行未跟踪计时支持代码。这些修改只增加可开关的summary/trace边界计时和报告导出，不是本节113/86行dW kernel优化的一部分。

测试仓库`FFTtest`还存在未提交的`monitor.py`、`run_finetune_perf_test_bf16.sh`和`step_timing_probe.py`修改，用于传递计时环境变量、归档内部计时结果和补充step口径；它们同样不在ktransformers的GitHub PR代码树内。

## 调用边界

三组 Full FT 基座权重梯度为：

```text
grad_gate = grad_gate_out^T × input         [I, H]
grad_up   = grad_up_out^T   × input         [I, H]
grad_down = grad_output^T   × intermediate  [H, I]
```

调用处有明确的 Full-only 条件：

```cpp
if (sft_config_.full_weight_grad && grad_gate_proj && grad_up_proj && grad_down_proj) {
  backward_base_weight_grad(cache, full_intermediate_size,
                            grad_gate_proj, grad_up_proj, grad_down_proj);
}
```

因此：

- Full FT 执行公共 base dX、activation backward、LoRA/adapter 条件路径以及本函数的 base dW；
- LoRA-only 执行公共 base dX 和 adapter 梯度，但 `full_weight_grad=false`，不会进入本函数；
- 本次 diff 不会直接加速 LoRA-only，LoRA 日志只能作为公共路径和环境波动的旁路对照。

## 优化前的问题

Qwen3-30B-A3B 每个 NUMA/TP 分区：

```text
I_local = 384  -> 12 个 i_tile
H       = 2048 -> 64 个 h_tile
experts = 128
tile    = 32 × 32 × 32
```

优化前一个任务对应一个输出 tile：

```text
(expert, projection, i_tile, h_tile)
128 × 2 × 12 × 64 = 196,608 tasks / layer / NUMA
```

平均每个 expert 约 256 tokens，即 8 个 K tile。旧循环在每个 K tile 后 `store_c()`，下一 K tile 前再 `load_c()`；同一 Gate/Up input panel、固定 `i_tile` 的梯度 panel，也会在 64 个 `h_tile` 之间重复打包。

最小有效修改同时解决三件事：

1. 任务合并成固定 `i_tile` strip；
2. 动态 operand panel 在任务内预打包并复用；
3. FP32 C tile 在完整 K reduction 内常驻 AMX tile 寄存器。

下面的 `diff` 块抽取关键行，并对少数长行重新换行以便阅读；未省略任何设计层面的修改。文件级原始 diff 请使用文末只读命令查看。

## Git Diff 重点 1：对齐所需类型

```diff
@@
 #include <chrono>
 #include <climits>
 #include <cmath>
+#include <cstdint>
 #include <cstdio>
```

`std::uintptr_t` 用于把 `thread_local std::vector` 的数据指针向上对齐到 64 byte。没有替换全局 allocator，也没有新增持久化类成员。

## Git Diff 重点 2：任务从 output tile 合并为 fixed-i strip

```diff
@@ void backward_base_weight_grad(...)
 const int i_tiles = (I + TILE_M - 1) / TILE_M;
 const int h_tiles = (H + TILE_N - 1) / TILE_N;
-const int tiles_per_projection = i_tiles * h_tiles;
-const int tasks_per_expert = tiles_per_projection * 2;
+// Keep enough fixed-i strips for load balancing while amortizing
+// task pickup and panel packing over H.
+const int tasks_per_expert = i_tiles * 2;
 const int total_tasks = activated_expert * tasks_per_expert;

 pool->do_work_stealing_job(
     total_tasks, [](int _) { T::config(); },
-    [&, i_tiles, h_tiles, tiles_per_projection, tasks_per_expert](int task_id) {
+    [&, i_tiles, h_tiles, tasks_per_expert](int task_id) {
       const int expert_task = task_id / tasks_per_expert;
       const int local_task = task_id % tasks_per_expert;
-      const bool do_down = local_task >= tiles_per_projection;
-      const int tile_id = do_down ? local_task - tiles_per_projection : local_task;
+      const bool do_down = local_task >= i_tiles;
+      const int i_tile = local_task % i_tiles;
```

变化后的任务是：

```text
(expert, projection, i_tile)
128 × 2 × 12 = 3,072 tasks / layer / NUMA
```

任务领取次数理论上减少 64 倍。每个 NUMA 有 48 workers 时仍平均有 64 tasks/worker，保留了足够的 work-stealing 粒度。

## Git Diff 重点 3：线程局部、按需扩容、64-byte 对齐的 panel scratch

```diff
@@
-alignas(64) ggml_bf16_t a_tile[TILE_M * TILE_K];
-alignas(64) ggml_bf16_t b_tile[TILE_N * TILE_K];
+const int k_tiles = (m + TILE_K - 1) / TILE_K;
+constexpr size_t A_TILE_ELEMENTS = TILE_M * TILE_K;
+constexpr size_t B_TILE_ELEMENTS = TILE_N * TILE_K;
+constexpr size_t TILE_ALIGNMENT = 64;
+constexpr size_t ALIGNMENT_PADDING =
+    TILE_ALIGNMENT / sizeof(ggml_bf16_t);
+const size_t packed_a_elements =
+    (size_t)k_tiles * A_TILE_ELEMENTS;
+const size_t packed_b_elements =
+    (size_t)k_tiles * B_TILE_ELEMENTS;
+
+thread_local std::vector<ggml_bf16_t> packed_a0_storage;
+thread_local std::vector<ggml_bf16_t> packed_a1_storage;
+thread_local std::vector<ggml_bf16_t> packed_b_storage;
+auto resize_aligned = [](std::vector<ggml_bf16_t>& storage,
+                         size_t elements) {
+  const size_t required = elements + ALIGNMENT_PADDING;
+  if (storage.size() < required) storage.resize(required);
+  const auto raw =
+      reinterpret_cast<std::uintptr_t>(storage.data());
+  const auto aligned =
+      (raw + TILE_ALIGNMENT - 1) &
+      ~(std::uintptr_t)(TILE_ALIGNMENT - 1);
+  return reinterpret_cast<ggml_bf16_t*>(aligned);
+};
+
+ggml_bf16_t* packed_a0 =
+    resize_aligned(packed_a0_storage, packed_a_elements);
+ggml_bf16_t* packed_b =
+    resize_aligned(packed_b_storage, packed_b_elements);
 alignas(64) float c0[TILE_M * TILE_N];
```

scratch 特性：

- 每个 WorkerPool thread 各自拥有，不共享写入，无需锁；
- 只在当前遇到的 `k_tiles` 更大时扩容，跨任务和跨函数调用复用容量；
- 尾部 M/I/H 继续先清零再填有效范围，保持 padding 语义；
- 平均 `M=256` 时，每个 packed panel 是 `8 × 32 × 32 × 2 B = 16 KiB`，三个 panel 合计约 48 KiB/worker。

## Git Diff 重点 4：Down 固定 B panel，并让 C 跨完整 K 常驻

Down 的固定 `i_tile` 对应 intermediate B。它先一次性打包全部 K panel，然后在 `h_tile` 循环中只重新打包 grad-output A：

```diff
@@ Down path
+const ggml_bf16_t* grad_output =
+    base_grad_output_bf16_ptr_[expert_idx];
+const ggml_bf16_t* intermediate =
+    cache.intermediate_cache + pos_start * I;
+std::memset(packed_b, 0,
+            packed_b_elements * sizeof(ggml_bf16_t));
+for (int kt = 0; kt < k_tiles; kt++) {
+  const int k_start = kt * TILE_K;
+  const int k_count = std::min(TILE_K, m - k_start);
+  ggml_bf16_t* b_tile =
+      packed_b + (size_t)kt * B_TILE_ELEMENTS;
+  for (int col = 0; col < i_count; col++) {
+    for (int kk = 0; kk < k_count; kk++) {
+      b_tile[col * TILE_K + kk] =
+          intermediate[(size_t)(k_start + kk) * I +
+                       i_start + col];
+    }
+  }
+  amx::transpose_16x16_32bit(
+      reinterpret_cast<__m512i*>(b_tile));
+  amx::transpose_16x16_32bit(
+      reinterpret_cast<__m512i*>(
+          b_tile + amx::GemmKernel224BF::TILE_N * TILE_K));
+}
```

最关键的 C tile diff：

```diff
@@ old K loop
-T::load_b(b_tile, TILE_K * sizeof(ggml_bf16_t));
-T::load_a(a_tile, TILE_K * sizeof(ggml_bf16_t));
-if (k_start == 0) {
-  T::clean_c();
-} else {
-  T::load_c(c0, TILE_N * sizeof(float));
-}
-T::run_tile();
-T::store_c(c0, TILE_N * sizeof(float));
+// Keep the full 32x32 FP32 C tile resident for the complete K reduction.
+T::clean_c();
+for (int kt = 0; kt < k_tiles; kt++) {
+  T::load_b(packed_b + (size_t)kt * B_TILE_ELEMENTS,
+            TILE_K * sizeof(ggml_bf16_t));
+  T::load_a(packed_a0 + (size_t)kt * A_TILE_ELEMENTS,
+            TILE_K * sizeof(ggml_bf16_t));
+  T::run_tile();
+}
+T::store_c(c0, TILE_N * sizeof(float));
```

平均 8 个 K tile 时，每个输出 tile 从 8 次 `store_c` + 7 次 `load_c` 变成最后 1 次 `store_c`。

## Git Diff 重点 5：Gate/Up 共用 input B，分别占用完整 C tile 集合

固定 `i_tile` 后，Gate A 和 Up A 的全部 K panel 各打包一次：

```diff
@@ Gate/Up fixed-i panels
+ggml_bf16_t* packed_a1 =
+    resize_aligned(packed_a1_storage, packed_a_elements);
+std::memset(packed_a0, 0,
+            packed_a_elements * sizeof(ggml_bf16_t));
+std::memset(packed_a1, 0,
+            packed_a_elements * sizeof(ggml_bf16_t));
+for (int kt = 0; kt < k_tiles; kt++) {
+  const int k_start = kt * TILE_K;
+  const int k_count = std::min(TILE_K, m - k_start);
+  ggml_bf16_t* gate_a_tile =
+      packed_a0 + (size_t)kt * A_TILE_ELEMENTS;
+  ggml_bf16_t* up_a_tile =
+      packed_a1 + (size_t)kt * A_TILE_ELEMENTS;
+  for (int row = 0; row < i_count; row++) {
+    for (int kk = 0; kk < k_count; kk++) {
+      gate_a_tile[row * TILE_K + kk] =
+          grad_gate_output_[
+              (pos_start + k_start + kk) * I + i_start + row];
+      up_a_tile[row * TILE_K + kk] =
+          grad_up_output_[
+              (pos_start + k_start + kk) * I + i_start + row];
+    }
+  }
+}
```

每个 `h_tile` 的 input B 只打包一次，Gate 和 Up 两个 pass 共用。Gate 和 Up 都需要 `GemmKernel224BF` 的全部四个 C tile，不能同时常驻，所以正确结构是两个完整 K pass：

```diff
@@ Gate/Up C-resident passes
+// Gate and up each consume all four C tiles,
+// so retain C across K in two separate passes.
+T::clean_c();
+for (int kt = 0; kt < k_tiles; kt++) {
+  T::load_b(packed_b + (size_t)kt * B_TILE_ELEMENTS,
+            TILE_K * sizeof(ggml_bf16_t));
+  T::load_a(packed_a0 + (size_t)kt * A_TILE_ELEMENTS,
+            TILE_K * sizeof(ggml_bf16_t));
+  T::run_tile();
+}
+T::store_c(c0, TILE_N * sizeof(float));
+
+T::clean_c();
+for (int kt = 0; kt < k_tiles; kt++) {
+  T::load_b(packed_b + (size_t)kt * B_TILE_ELEMENTS,
+            TILE_K * sizeof(ggml_bf16_t));
+  T::load_a(packed_a1 + (size_t)kt * A_TILE_ELEMENTS,
+            TILE_K * sizeof(ggml_bf16_t));
+  T::run_tile();
+}
+T::store_c(c1, TILE_N * sizeof(float));
```

不能把 Gate 和 Up 合成一次 `clean_c()` 后交错运行，因为两次 `run_tile()` 都写同一组 C tile，会把两个 projection 累加到一起。

## 数值和并发不变量

本次修改保持以下不变量：

- 每个 `(expert, projection, i_tile, h_tile)` 输出区域仍只有一个任务写，不需要原子加或互斥锁；
- 每个输出元素的 K 累加顺序仍按 `kt=0..k_tiles-1`，没有分线程 reduction；
- `m`、`I`、`H` 非 32 整数倍时，先清零整块 scratch，再只复制有效元素；
- Gate/Up/Down 最终写回的逻辑 layout 和 BF16 转换没有变化；
- 非 AMX BF16 的 fallback 路径没有变化；
- TP 分区仍使用原 `I=full_intermediate_size/tp_part` 和原 expert/gradient pointer。

构建与 reference 验证：

- Release 构建通过；
- TP part 0/1、非零 expert 通过；
- `M={1,31,32,33,65,256}` 通过；
- Qwen 维度 `H=2048, I_local=384` 的 `M=33/256` 通过；
- Gate/Up/Down dW 均与 reference 对照通过。

## 性能证据

对照会话：

- 修改前：`FFTtest/Qwen3-30B-A3B/test_log/20260715_203338_1gpu_AMX_BF16_FULL_THEN_LORA`
- 修改后：`FFTtest/Qwen3-30B-A3B/test_log/20260716_143647_1gpu_AMX_BF16_FULL_THEN_LORA`
- 两个 `session_config.json` 的文本 diff 为空。

Full 稳定区间：

| 指标 | 修改前 | 修改后 | 变化 |
| --- | ---: | ---: | ---: |
| Step | 19.25197 s | 16.13997 s | -16.16% |
| TPS | 212.76 | 253.78 | +19.28% |
| Forward | 1.47821 s | 1.34553 s | -8.98% |
| Backward | 9.28085 s | 6.79183 s | **-26.82%** |
| Optimizer | 1.79075 s | 1.92533 s | +7.52% |
| Requant | 6.32860 s | 5.73156 s | -9.43% |

目标 backward 节省 2.48902 s，占完整 step 节省 3.11200 s 的约 80.0%。同一理论 FLOPs 模型下，backward 有效吞吐从 11.412 提升到 15.594 TFLOPS（+36.65%），roofline 下界效率从 9.95% 提升到 13.59%，日志判断由“偏低”变为“正常”。这仍是整个 backward 的模型估算，不是硬件 AMX tile 利用率；forward 和 requant 也有运行间变化，所以不能把全部 TPS 增益都归因于 dW。

LoRA 旁路对照：

| 指标 | 修改前会话 | 修改后会话 | 变化 |
| --- | ---: | ---: | ---: |
| Step | 6.08625 s | 5.86640 s | -3.61% |
| TPS | 672.99 | 698.21 | +3.75% |
| Forward | 1.63101 s | 1.54067 s | -5.54% |
| Backward | 4.35296 s | 4.23380 s | -2.74% |

LoRA backward 的公共波动约 0.119 s；Full backward 则减少 2.489 s。简单绝对差分仍留下约 **2.370 s/step** 的 Full-only 额外收益。

新 Full 运行 15 steps、退出码 0；稳定 backward 为 6.755–6.829 s，稳定 step 为 16.008–16.290 s。loss 从 0.7773 降到 0.2754，grad norm 为 1.685–5.276。日志 summary 的基础健康检查判定无 SIGSEGV/NaN。

## 和 LoRA backward 优化的关系

### LoRA 当前优化了什么

当前 LoRA backward 主要包括：

- base dX：`BufferA::from_mat()` + 预转置 base-weight `BufferB` + `amx::mat_mul()`；
- Gate/Up adapter：融合 `input -> u` 与 grad-B，按 token 分块，rank=8 使用 AVX2 FMA；
- Down adapter：按 token 或输出维分块，使用线程局部 FP32 accumulator，最后分块写回；
- route scatter/fused add：AVX512 BF16/FP32 转换和 FMA；
- 中间结果：共享 pool、per-expert view 和复用 scratch，避免反复大对象分配。

这些优化的共同原则是融合、分块、复用数据和减少中间内存流量。

### LoRA 是否调整了 AMX tile 打包逻辑

需要区分三层：

1. **底层 AMX tile config/形状：没有调整。** 仍是 `GemmKernel224BF` 的固定 tile 和通用 `amx::mat_mul()`。
2. **公共 base dX packing：有标准打包，但不是 LoRA-specific 改动。** Full 和 LoRA 都把动态 grad 转成 `BufferA`，把 base weight 预先转置成 `BufferB`。
3. **LoRA adapter backward 的 live 实现：主要走 AVX，不是自定义 AMX tile packing。** `prepare_lora_backward_weights()` 仍准备 `down_lora_a_t_bb_`/`down_lora_b_t_bb_`，但当前文件中这两个对象除分配和准备外没有消费点；实际 adapter backward 使用 `avx::lora_*`。

所以准确结论是：**LoRA 保留了 AMX BufferB 打包准备设施，公共 base dX 依赖标准 AMX packing；但当前 LoRA-specific backward 优化没有改变底层 AMX tile，也没有调整 live AMX tile packing 循环。**

### Full FT 方向是否正确

正确。Full dW 的两个 operand 都来自当前 step 的 activation/gradient，不能像 base dX 的权重 `BufferB` 那样跨 step 长期缓存。本次选择在单次 dW 内：

- 固定 strip 后复用不变 panel；
- Gate/Up 共用 input B；
- 合并 task，降低调度频率；
- 让 C 跨完整 K 常驻；
- 使用线程局部 scratch 避免共享和反复分配。

这与 LoRA 的优化原则一致，又符合 Full dW 的数据生命周期。实测 Full backward -26.82%、LoRA 旁路仅 -2.74%，也验证了方向。

函数级和编排边界计时已在本地worktree实现。`20260716_175359...`稳定结果显示`backward_base_weight_grad()`为2.036 s/step，占完整outer backward的29.3%；checkpoint重算为2.079 s，requant为6.746 s。因此下一步应继续拆分dW中的Gate/Up、Down、dynamic packing、AMX compute与BF16 store，并采集AMX/cycles/cache/top-down计数器；不要在缺少这些证据时先改tile形状或全局WorkerPool。

## 查看完整 Git Diff

本文主体描述的是已提交的`f209878`，不是当前未提交计时 diff。分别使用以下只读命令查看：

```bash
cd /mnt/data2/wbw/ktransformers
git show --stat --oneline f209878
git diff f209878^ f209878 -- kt-kernel/operators/amx/sft_moe.hpp
git status --short
git diff --stat
git diff -- kt-kernel/operators/amx/sft_moe.hpp
```

前两个命令查看已提交dW优化；后三个命令查看GitHub head之外的本地计时修改。未跟踪文件不会出现在`git diff --stat`中，需结合`git status --short`。

当前实现主体位于：

```text
kt-kernel/operators/amx/sft_moe.hpp:1926  backward_base_weight_grad()
kt-kernel/operators/amx/sft_moe.hpp:1949  AMX BF16 optimized branch
kt-kernel/operators/amx/sft_moe.hpp:2001  Down strip
kt-kernel/operators/amx/sft_moe.hpp:2052  Gate/Up strip
```

## 10. FFTtest 可视化 working-tree 变更

2026-07-20 的可视化精简发生在独立仓库 `/mnt/data2/wbw/FFTtest`，不改变 `/mnt/data2/wbw/ktransformers` 代码树。比较基线为：

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

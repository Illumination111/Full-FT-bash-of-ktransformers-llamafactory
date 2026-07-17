# kt-kernel fullft-development 分支作用与逐段代码分析

## 1. 分析对象与结论

分析对象：

- GitHub 比较页：[kvcache-ai/ktransformers main...Illumination111:fullft-development](https://github.com/kvcache-ai/ktransformers/compare/main...Illumination111:fullft-development)
- 官方 main：<code>7c021b430c36a408032c20bbf3833dc1bce6efa4</code>
- 个人分支：<code>06f06065697b1e50ba6eefc10ab85b3cf37e61a6</code>
- 共同祖先：<code>8e46e5896c3d993a1285052f2618f5a9f01882d4</code>
- 分析日期：2026-07-13 UTC

本文使用三点比较，也就是只分析共同祖先到个人分支的变化：

~~~bash
git diff upstream/main...fullft-development
~~~

不能把 <code>git diff upstream/main fullft-development</code> 的双点结果全部归为个人修改，因为两个分支已经分叉：个人分支领先 6 个提交，同时落后官方 main 3 个提交。落后的三个官方提交是 RAWINT4 归一化、端口检测和 sglang 子模块更新，不属于 Full FT 分支开发内容。

### 核心结论

这个分支的主要作用，是把官方 kt-kernel 以 LoRA 为中心的 SFT expert 路径扩展成可训练 CPU/AMX MoE expert 基座权重的完整 Full Fine-Tuning 路径。被训练的基座权重是每层 expert 的：

- <code>gate_proj</code>
- <code>up_proj</code>
- <code>down_proj</code>

它建立的闭环是：

~~~text
LLaMA-Factory finetuning_type
  → KT 的 lora / full / hybrid 模式
  → CPU BF16 gate/up/down nn.Parameter
  → Python autograd 显式输入
  → C++/AMX backward 计算基座权重梯度
  → TP 按完整 F 维切片写回
  → autograd 返回三个梯度
  → optimizer 更新 BF16 Parameter
  → 标记基座权重 dirty
  → 下一次 forward 前重新量化到 AMX BufferB
~~~

因此，它解决的不是“让 loss 能下降”这一表面问题，而是让 KT 接管的 CPU experts 真正成为 optimizer 可见、可求梯度、可更新、可重新量化的训练参数。

需要限定结论：该分支已经证明基座梯度链路和权重更新链路连通，但现有测试中梯度最大值仍达到约 <code>1e16～1e17</code>，尚未完成与 PyTorch 参考梯度的逐元素数值验证，所以不能据此宣称 Full FT 的数值正确性已经完全关闭。

## 2. 修改规模

GitHub 三点比较共 16 个文件，<code>+1582/-160</code>。其中新增的 <code>kt-kernel/AGENTS.md</code> 占 546 行；排除文档后，生产代码为 15 个文件，<code>+1036/-160</code>。

### 2.1 改动最多的两个 hpp

| 排名 | 文件 | 新增 | 删除 | churn | 主要职责 |
|---:|---|---:|---:|---:|---|
| 1 | <code>operators/amx/sft_moe.hpp</code> | 285 | 5 | 290 | 单个 NUMA/TP AMX 子核的真实 backward、基座梯度、工作区和 AMX tile 并行 |
| 2 | <code>operators/moe-sft-tp.hpp</code> | 62 | 7 | 69 | 顶层 TP/NUMA 调度、完整梯度清零、TP slice 计算和 Python binding |
| 3 | <code>operators/common.hpp</code> | 9 | 0 | 9 | Full FT 配置和三个梯度指针 |

后文按零上下文 diff 的 hunk 边界，逐段覆盖第 1 个文件的 20 段修改和第 2 个文件的 12 段修改。大函数所在 hunk 还会继续拆成更细的逻辑块。

### 2.2 提交演化

| 提交 | 生产代码作用 |
|---|---|
| <code>e99b5e1</code> | 初次接入 Full FT；15 个代码文件，<code>+809/-155</code>，打通 Python Parameter、autograd、C++ 指针和 optimizer/requant 主链路 |
| <code>2d81e86</code> | 两个 hpp 中修复 Full FT 开关被 config slicing 丢失、TP slice/global stride、统一清零和纯 Full 模式零秩问题；生产代码 <code>+65/-15</code> |
| <code>20f645c</code> | 修复上一提交把 token-major 路由缓存误当 expert-major token list 所导致的 SIGSEGV；生产代码 <code>+5/-10</code> |
| <code>5ce0767</code> | 保存 down 梯度需要的只读 dY 快照，并用 AMX BF16 tile 与 NUMA subpool 并行替代慢速标量计算；生产代码 <code>+185/-8</code> |

后三个修复提交的生产代码只集中在这两个 hpp，合计 <code>+255/-33</code>。这也是它们成为本分支核心修改文件的原因。

## 3. 全部分支改动的作用

| 文件 | 代码变化 | 作用与修改原因 |
|---|---:|---|
| <code>operators/common.hpp</code> | +9/-0 | 在 <code>MOESFTConfig</code> 中加入 <code>full_weight_grad</code> 和 gate/up/down 三个零拷贝梯度指针，使 C++ 配置能够表达“计算基座权重梯度”并直接写入 Python tensor 内存。 |
| <code>ext_bindings.cpp</code> | +22/-5 | 扩展 backward task 参数和 pybind 配置字段；暴露 <code>set_base_weight_pointers</code>，让 optimizer 更新后的 BF16 权重在原 C++ 对象内重新量化。 |
| <code>operators/moe-sft-tp.hpp</code> | +62/-7 | 在顶层统一清零完整梯度，按 TP offset 分割三组基座梯度，并把完整 F 作为子核写回 stride。 |
| <code>operators/amx/sft_moe.hpp</code> | +285/-5 | 计算三组基座权重梯度；修复路由布局、down dY 生命周期和零秩 LoRA；加入 AMX BF16/FP32 累加与线程池并行。 |
| <code>python/experts.py</code> | +6/-0 | wrapper/factory 增加并向下传递 <code>full_weight_grad</code>。 |
| <code>python/sft/config.py</code> | +8/-0 | 增加 <code>kt_train_mode</code> 与 <code>kt_full_weight_grad</code>；从环境变量读取模式，并把 full/hybrid 映射为基座梯度开启。 |
| <code>python/sft/wrapper.py</code> | +71/-17 | 将 LLaMA-Factory 训练类型映射到 KT 模式；保留合法的 <code>lora_rank=0</code>；创建基座 Parameter；清除 model tree 内重复 expert 权重。 |
| <code>python/sft/base.py</code> | +89/-26 | 保存 Full FT 状态与 dirty 标志；创建三组 CPU BF16 Parameter 和 C++ 直写梯度 buffer；允许无 LoRA 的 Full FT；定义基座权重同步接口。 |
| <code>python/sft/layer.py</code> | +107/-19 | Full FT 时强制走 autograd；把三个 Parameter 显式传入自定义 Function；optimizer 后触发 requant；保留 router 梯度；同期兼容 transformers v5 TopKRouter/GLM4 路由。 |
| <code>python/sft/autograd.py</code> | +51/-16 | forward 增加三个基座 Parameter 输入，backward 返回 C++ 已写好的 gate/up/down 梯度，令 PyTorch 设置 Parameter.grad。 |
| <code>python/sft/lora.py</code> | +137/-23 | 分离收集 LoRA 参数和 Full FT 基座参数；兼容旧 Trainer 注入入口；纯 Full 跳过 LoRA buffer；分布式同步基座梯度；step 后标 dirty。 |
| <code>python/sft/amx.py</code> | +139/-27 | 把模式和梯度 data pointer 写入 C++ task；零秩时传空 LoRA 指针；实现原对象 requant 和完整重建 fallback。 |
| <code>python/sft/weights.py</code> | +26/-12 | 把 model tree 中的重复 expert 权重替换为带 <code>_kt_zero_storage</code> 标记的零存储 placeholder，避免重复显存/内存和参数统计。 |
| <code>python/sft/arch.py</code> | +13/-1 | 增加 GLM4 MoE 架构识别，属于同期兼容性扩展，并非 Qwen3 Full FT 梯度 bug 的直接修复。 |
| <code>python/sft/__init__.py</code> | +11/-2 | 导出新的 KT trainable 参数收集 API；其余变化主要是格式整理。 |

### 3.1 三种模式的预期语义

- full：训练 expert 基座权重；通常 <code>lora_rank=0</code>。
- hybrid：同时训练 expert 基座权重和 LoRA；允许 <code>lora_rank&gt;0</code>。
- lora：只训练 LoRA；不应计算、同步或更新基座权重。

## 4. operators/amx/sft_moe.hpp：20 段修改逐段分析

该文件处在最内层，负责每个 NUMA/TP 子核的真实 forward/backward。以下编号 S01～S20 与：

~~~bash
git diff --unified=0 8e46e58..fullft-development -- \
  kt-kernel/operators/amx/sft_moe.hpp
~~~

得到的 20 个 hunk 一一对应。

### S01：增加每个 expert 的只读 dY 快照指针

代表代码：

~~~cpp
std::vector<ggml_bf16_t*> base_grad_output_bf16_ptr_;
~~~

- 作用：为每个 active expert 保存 route-weighted <code>grad_output</code> 的独立地址，专供 <code>down_proj</code> 基座梯度计算。
- 修改原因：原 <code>grad_output_bf16_ptr_</code> 是工作区，之后会被 gate/up 的 grad-input 路径覆盖。测试中 gate/up 为 48/48 非零而 down 为 0/48，说明 down 读取时原 dY 已失效。

### S02：增加快照池的基地址

代表代码：

~~~cpp
void* base_grad_output_bf16_pool_ = nullptr;
~~~

- 作用：保存整块快照内存的池指针，后续从中为 active experts 切片。
- 修改原因：仅增加 per-expert 指针不提供实际存储；必须有独立于可复用工作区的底层内存，才能保证 dY 生命周期跨过 gate/up backward。

### S03：记录快照池大小

代表代码：

~~~cpp
size_t base_grad_output_bf16_pool_bytes_ = 0;
~~~

- 作用：把快照池纳入统一 backward pool 的容量计算。
- 修改原因：该类使用手工池化内存；若不记录字节数，新增快照会与后续区域重叠或越界。

### S04：扩大 forward pool，容纳标量 fallback 的三个 FP32 累加矩阵

代表代码：

~~~cpp
const size_t bytes = 3 * I * H * sizeof(float);
work_required = std::max(work_required, bytes);
~~~

- 作用：在 Full FT 开启时，保证 forward pool 至少能容纳 gate、up、down 三个 FP32 累加器。down 的 <code>H×I</code> 元素数与 gate/up 的 <code>I×H</code> 相同。
- 修改原因：纯 Full 模式可能使用 <code>lora_rank=0</code>，原 pool 只按 LoRA 工作区大小申请，容量可能为零或明显不足；标量 fallback 复用它做三个累加器时会发生越界或内存破坏。

### S05：把 dY 快照大小计入 backward pool 总容量

代表代码：

~~~cpp
required += round_up(base_grad_output_bf16_pool_bytes_, kAmxAlignment);
~~~

- 作用：在一次统一分配中为快照预留对齐空间。
- 修改原因：避免新增快照与已有 BufferA、BufferC、LoRA grad pool 共享同一地址区间。

### S06：从统一 backward pool 分配快照区域

代表代码：

~~~cpp
assign(&base_grad_output_bf16_pool_, base_grad_output_bf16_pool_bytes_);
~~~

- 作用：把 S05 预留的容量映射成实际可用地址。
- 修改原因：统一池的容量计算和地址切片必须同时修改；只做其中一步会得到空指针或区域重叠。

### S07：LoRA scaling 对零秩安全

代表代码：

~~~cpp
lora_scaling_ = rank > 0 ? alpha / rank : 0.0f;
~~~

- 作用：在纯 Full FT 的 <code>lora_rank=0</code> 下把 LoRA scaling 设为 0。
- 修改原因：官方路径假设 LoRA rank 为正，直接 <code>alpha / rank</code> 会除零；纯 Full 模式必须允许完全没有 LoRA。

### S08：增加 Full FT 开关 setter

代表代码：

~~~cpp
void set_full_weight_grad(bool enabled);
~~~

- 作用：允许顶层 <code>TP_MOE_SFT</code> 在子核构造后显式恢复 Full FT 开关。
- 修改原因：顶层基类只保存 <code>GeneralMOEConfig</code>，把派生的 <code>MOESFTConfig</code> 传入时发生 object slicing，<code>full_weight_grad</code> 在子核内恢复为默认 false。初版即使 Python 和顶层配置为 true，基座梯度函数仍被静默跳过。

### S09：扩展子核 backward 接口

代表代码：

~~~cpp
void backward(...,
              void* grad_gate_proj,
              void* grad_up_proj,
              void* grad_down_proj);
~~~

- 作用：让 AMX 子核直接收到三个 Python BF16 梯度 tensor 的 TP slice 地址。
- 修改原因：原接口只返回输入、router 和 LoRA 梯度，没有任何通道能把 expert 基座梯度写回 Python/autograd。

### S10：在普通 backward 完成后触发基座梯度计算

代表代码：

~~~cpp
if (full_weight_grad && gate_ptr && up_ptr && down_ptr) {
    backward_base_weight_grad(...);
}
~~~

- 作用：复用已经得到的 <code>grad_gate_output_</code>、<code>grad_up_output_</code>、中间激活和保存的 dY，计算三组基座权重梯度。
- 修改原因：基座梯度依赖前面的 down backward 和 gate/up backward 结果；同时必须要求开关开启且三个输出指针全部有效，避免 LoRA 模式误写内存和空指针访问。

### S11：新增并逐步修正基座权重梯度核心函数

这一个 diff hunk 最终新增约 234 行，是整个分支的 C++ 核心。它可继续拆成以下逻辑块。

#### S11-a：明确本地 I 与完整 F 的布局契约

~~~text
H = hidden_size
I = 当前 TP 子核的 local intermediate_size
F = 未切分的完整 intermediate_size

gate/up 输出布局：[E, F, H]
down 输出布局：   [E, H, F]
~~~

- 作用：循环只计算本地 I 个通道，但 expert stride 和 down 的 row stride 始终使用完整 F。
- 修改原因：初版以 I 作为完整 tensor stride；TP 大于 1 时不同 expert/TP slice 的地址会错位。函数同时检查 <code>F &gt;= I</code>，尽早拒绝不合法配置。

#### S11-b：预计算 expert-major packed token offset

代表逻辑：

~~~text
expert_offsets[task] = 前面 active experts 的 token 数之和
~~~

- 作用：把 <code>grad_gate_output_</code>、<code>grad_up_output_</code> 和 <code>intermediate_cache</code> 中的 packed 区域定位到当前 expert。
- 修改原因：这些 backward buffer 是 expert-major packed 布局，不是原始 token-major 路由表。预计算也把标量 fallback 原来每个 expert 重扫前序 expert 的 O(A²) offset 计算降为 O(A)。

#### S11-c：只在匹配的 AMX BF16 kernel 上进入 tile 快路径

代表逻辑：

~~~cpp
if constexpr (T 是 GemmKernel224BF 且 AMX 可用) {
    static_assert(tile 为 32×32×32);
}
~~~

- 作用：为 AMX BF16 kernel 使用固定 32×32×32 tile，同时保留其他模板实例的标量 fallback。
- 修改原因：tile pack、转置和寄存器布局依赖具体 kernel；无条件使用会破坏其他量化/分组 kernel。

#### S11-d：按 expert、projection、output tile 生成线程池任务

代表逻辑：

~~~text
每个 expert 的任务 =
  所有 gate/up output tiles
  + 所有 down output tiles
~~~

- 作用：将任务提交给当前 TP/NUMA 的 subpool，gate/up 在同一任务中共享 input tile，down 使用独立任务。
- 修改原因：旧实现由每个 NUMA 调度线程串行执行 token×I×H 三重标量循环，测试 backward 平均约 754.484 秒；tile 并行版本降到约 22.755 秒，约 33.2 倍加速。

#### S11-e：AMX 计算 gate/up 基座梯度

数学含义：

~~~text
dW_gate = dGateᵀ × X    → [I, H]
dW_up   = dUpᵀ × X      → [I, H]
~~~

- 作用：对 token 维 K 分块，把不足 32 的边缘补零；gate/up 共用 X 的 B tile，分别使用 FP32 <code>c0/c1</code> 跨 K block 累加，最后转成 BF16。
- 修改原因：outer product 本质可表达为 GEMM；复用 AMX tile 能同时消除三重标量循环瓶颈，并保持 FP32 累加精度。

#### S11-f：AMX 计算 down 基座梯度

数学含义：

~~~text
dW_down = dY_weightedᵀ × intermediate → [H, I]
~~~

- 作用：从只读 <code>base_grad_output_bf16_ptr_</code> 取 route-weighted dY，从 forward cache 取激活，按 H×I tile 计算并以完整 F 作为行 stride 写回。
- 修改原因：down 必须使用乘过 router weight 且尚未被 gate/up 工作区覆盖的 dY；此前直接读工作 buffer 导致 down 48 层全部为零。

#### S11-g：每个 tile 写独占输出区域

- 作用：gate/up/down task 根据 expert 和 tile 坐标写入互不重叠的最终梯度 slice，无需锁或原子加。
- 修改原因：如果仍按 expert 整块并行或让 TP 子核共享起始地址，会产生写竞争；独占 tile 与顶层 TP slice 一起建立无重叠写入契约。

#### S11-h：保留非 AMX 标量 fallback

- 作用：对不支持该 AMX matmul 的模板实例继续使用 FP32 gate/up/down 累加器，并读取 expert-major 的 <code>m_local_input_ptr_</code> 和只读 dY 快照。
- 修改原因：Full FT 不能只在一个编译期模板中有实现；fallback 还修正了早期错误读取 <code>m_local_pos_cache[expert][t]</code> 的问题。该缓存真实布局为 <code>[token][route_slot]</code>，错误解释曾在第一个 backward 触发 SIGSEGV。

### S12：快照池大小与原 route-weighted grad_output 池相同

代表代码：

~~~cpp
base_grad_output_bf16_pool_bytes_ = grad_output_bf16_pool_bytes_;
~~~

- 作用：按相同的 active token 容量、hidden size 和对齐开销分配一份完整副本。
- 修改原因：两者保存完全相同形状的数据，只是生命周期不同；等尺寸可保证每个 active expert 的对齐 slice 都能复制。

### S13：不支持标准 AMX kernel 时把快照池大小置零

代表代码：

~~~cpp
base_grad_output_bf16_pool_bytes_ = 0;
~~~

- 作用：在 unsupported kernel 分支保持统一的明确初始化。
- 修改原因：防止该字段保留旧值或未初始化值，从而错误扩张/切分 backward pool；该分支将走原有或标量实现。

### S14：按 expert 数量扩展快照指针数组

代表代码：

~~~cpp
base_grad_output_bf16_ptr_.resize(expert_num);
~~~

- 作用：使 <code>expert_idx</code> 可直接索引快照地址。
- 修改原因：Full FT 只为本步 active experts 分配实际 slice，但指针容器仍必须覆盖全部 expert id。

### S15：初始化每个 expert 的快照指针为空

代表代码：

~~~cpp
base_grad_output_bf16_ptr_[i] = nullptr;
~~~

- 作用：在第一次 backward 动态分配前建立安全状态。
- 修改原因：避免 inactive expert 或未完成地址切分时意外使用未定义指针。

### S16：建立快照池切片游标

代表代码：

~~~cpp
char* cursor = static_cast<char*>(base_grad_output_bf16_pool_);
~~~

- 作用：在 <code>backward_down_amx</code> 中从独立快照池起始地址顺序切片。
- 修改原因：快照只需要覆盖本步 active experts，应沿实际 token 容量动态布局，而不是为所有 experts 按最大长度永久分配。

### S17：为每个 active expert 分配对齐快照 slice

代表逻辑：

~~~text
slice bytes = align64(local_max_m × H × sizeof(BF16))
~~~

- 作用：令每个 expert 的 <code>base_grad_output_bf16_ptr_[expert]</code> 指向互不重叠、按 AMX M_STEP 向上取整的区域。
- 修改原因：与工作 grad_output 的 per-expert 布局保持一致，避免 SIMD/AMX 对齐问题和 expert 间覆盖。

### S18：在工作 buffer 被覆盖前复制 route-weighted dY

代表代码：

~~~cpp
memcpy(base_dy[expert], working_dy[expert], num_tokens * H * sizeof(BF16));
~~~

- 作用：scatter 并乘 router weight 后，立即为 active experts 并行保存只读副本。
- 修改原因：<code>backward_gate_up_amx</code> 后续复用 <code>grad_output_bf16_ptr_</code> 保存 gate/up grad-input；若到基座梯度函数中才读取，down 所需 dY 已被覆盖。只在 Full FT 开启时复制，避免 LoRA 模式承担额外内存带宽。

### S19：只在 rank 大于零时准备 LoRA backward 权重

代表代码：

~~~cpp
if (lora_rank_ > 0 && lora_ptrs_valid) {
    prepare_lora_backward_weights();
}
~~~

- 作用：纯 Full 模式仍执行 base gate/up grad-input，但不进入 LoRA 权重准备。
- 修改原因：<code>lora_rank=0</code> 是新支持的合法状态；即使某些指针对象存在，也不能构造或量化零秩 LoRA buffer。

### S20：base pass 后对零秩 LoRA 提前返回

代表代码：

~~~cpp
if (SkipLoRA || lora_rank_ <= 0 || lora_ptrs_invalid) return;
~~~

- 作用：先完成 gate/up 基座权重对 grad-input 的贡献，再跳过所有 LoRA remainder。
- 修改原因：纯 Full FT 需要普通基座 backward，但不需要 LoRA 的 fused buffer、grad A/B 和缩放；显式零秩判断既防止非法临时区访问，也减少无效计算。

## 5. operators/moe-sft-tp.hpp：12 段修改逐段分析

该文件位于 AMX 子核之外，是 TP/NUMA 顶层 wrapper。以下编号 M01～M12 与：

~~~bash
git diff --unified=0 8e46e58..fullft-development -- \
  kt-kernel/operators/moe-sft-tp.hpp
~~~

得到的 12 个 hunk 一一对应。

### M01：构造后向所有 TP 子核传播 Full FT 开关

代表代码：

~~~cpp
for (int i = 0; i < tp_count; i++) {
    tps[i]->set_full_weight_grad(config.full_weight_grad);
}
~~~

- 作用：保证每个 NUMA/TP 子核的 <code>sft_config_.full_weight_grad</code> 与顶层配置一致。
- 修改原因：<code>TP_MOE</code> 基类保存的是 <code>GeneralMOEConfig</code>，构造子核时丢失 SFT 派生字段。该传播位于 <code>if constexpr (!kSkipLoRA)</code> 之外，确保纯 Full、零秩 LoRA 路径也能启用基座梯度。

### M02：关闭 BF16 partitioning 的直接 printf

修改形式：

~~~cpp
// printf("TP_MOE_SFT: From BF16 with partitioning\n");
~~~

- 作用：不再每次从 BF16 权重分区/加载时打印固定调试信息。
- 修改原因：该 hunk 不改变算法。提交没有单独记录原因；结合 Full FT 每次 optimizer step 后都可能触发 48 层重新量化，合理推断是为了避免日志洪泛和同步输出开销。

### M03：扩展顶层 backward 接口

代表代码：

~~~cpp
void backward(...,
              void* grad_gate_proj = nullptr,
              void* grad_up_proj = nullptr,
              void* grad_down_proj = nullptr);
~~~

- 作用：顶层 TP wrapper 接收完整的三组基座梯度 tensor。
- 修改原因：Python binding 只能调用这一层；若顶层接口没有这些参数，就无法将 Python buffer 继续分发给 AMX 子核。

### M04：集中判定是否需要基座权重梯度

代表逻辑：

~~~cpp
need_base_weight_grad =
    full_weight_grad && gate_ptr && up_ptr && down_ptr;
~~~

- 作用：后续清零、切片和传参共享同一条件。
- 修改原因：必须同时满足模式和三个输出 buffer 有效；统一条件避免某一步判定不一致，例如已经对空指针做 offset，却在子核处才发现功能未启用。

### M05：更新清零阶段的职责说明

修改后的注释含义：

~~~text
并行清零 per-TP partials 和最终 base-weight gradients
~~~

- 作用：明确最终三组基座梯度不再依赖 Python caller 预先清零，而由顶层 TP wrapper 负责。
- 修改原因：跨 step 时 inactive experts 本步不会被子核覆盖；若最终 tensor 不统一清零，就会保留上一 step 的梯度。由多个 TP 子核各自清完整 tensor 又会产生竞争，因此职责必须上移到唯一的顶层。

### M06：按 2 MiB chunk 并行清零三组完整梯度

代表逻辑：

~~~text
base_grad_bytes = E × F × H × sizeof(BF16)
依次把 gate、up、down 切成 2 MiB ClearSeg
统一提交 work-stealing memset
~~~

- 作用：在子核 dispatch 之前，把 gate/up 的 <code>[E,F,H]</code> 和 down 的 <code>[E,H,F]</code> 全部清零。三者元素数相同，所以可以复用同一个字节数。
- 修改原因：消除 inactive expert 的跨 step 残留，并避免多个 TP/NUMA 子核同时清同一完整 buffer。大 tensor 分块后可以利用线程池内存带宽。

### M07：为每个 TP 建立三组 slice 指针数组

代表代码：

~~~cpp
std::vector<ggml_bf16_t*> tp_grad_gate_proj(tp_count, nullptr);
std::vector<ggml_bf16_t*> tp_grad_up_proj(tp_count, nullptr);
std::vector<ggml_bf16_t*> tp_grad_down_proj(tp_count, nullptr);
~~~

- 作用：保存每个 TP 子核应该收到的独立起始地址。
- 修改原因：初版把三组完整 tensor 的相同首地址传给所有 TP，多个子核会写入同一位置，造成覆盖和竞争。

### M08：按 TP offset 计算 slice，并验证完整覆盖 F

代表公式：

~~~text
gate/up slice = base + tp_offset × H
down slice    = base + tp_offset

tp_offset 最终必须等于 F
~~~

- 作用：gate/up 沿 <code>[F,H]</code> 的 F 维切片，因此一个通道偏移 H 个元素；down 沿每一行的 F 维切片，因此首地址只偏移 <code>tp_offset</code>。子核内部再用完整 F 作为 expert/row stride。
- 修改原因：修复 TP slice 错位、不同 TP 写相同区域和 expert stride 使用 local I 的问题。指针运算位于 <code>need_base_weight_grad</code> 条件内，也避免对 null 做非零 offset 的未定义行为。覆盖检查能发现 TP 配置出现缺口或总长度不等于 F。

### M09：dispatch 时传入完整 F 和当前 NUMA 的独立 slice

代表逻辑：

~~~text
tps[numa_id]->backward(
    ...,
    full_intermediate_size,
    tp_grad_gate_proj[numa_id],
    tp_grad_up_proj[numa_id],
    tp_grad_down_proj[numa_id])
~~~

- 作用：每个 NUMA 子核只写自己的 local I 通道，但按完整 F 布局跨 expert/row。
- 修改原因：顶层负责“起点”，子核负责“local 计算 + global stride”；两侧缺一不可。只传独立起点但仍以 I 为 stride，会在 <code>expert_idx&gt;0</code> 或 down 的下一行发生错位。

### M10：Python backward binding 增加三个 intptr 参数

代表代码：

~~~cpp
void backward_binding(...,
                      intptr_t grad_gate_proj,
                      intptr_t grad_up_proj,
                      intptr_t grad_down_proj);
~~~

- 作用：pybind/CPUInfer task 可以把 Python tensor 的 <code>data_ptr()</code> 作为整数地址传入。
- 修改原因：C++ task 队列接口使用可复制的 <code>intptr_t</code> 表示外部地址；新增梯度必须进入同一 ABI 链路。

### M11：binding 将三个地址转成 void pointer 并调用顶层 backward

代表逻辑：

~~~cpp
backward(...,
         (void*)grad_gate_proj,
         (void*)grad_up_proj,
         (void*)grad_down_proj);
~~~

- 作用：完成 Python binding 到 C++ 顶层算法的最后一段传递。
- 修改原因：M10 只改变入口签名；若调用处仍使用旧参数列表，地址不会进入清零、切片和 AMX 子核。

### M12：增加原对象基座权重指针更新接口

代表代码：

~~~cpp
config.gate_proj = gate;
config.up_proj = up;
config.down_proj = down;
weights_loaded = false;
~~~

- 作用：optimizer 更新 CPU BF16 Parameter 后，替换顶层配置中的权重地址，并把状态标为需要重新 load/partition/quantize。
- 修改原因：AMX forward 使用的是由 BF16 权重量化生成的内部 BufferB；只更新 Parameter 不会改变下一次 forward。复用现有 C++ 对象重新量化约 0.6 秒/层，而完整重建约 1.9 秒/层，因此该接口既完成正确性闭环，也减少重建开销。

## 6. 两个 hpp 之间的接口契约

定义：

- E：expert 数量。
- H：hidden size。
- F：完整 intermediate size。
- I：当前 TP 子核的 local intermediate size。
- O：当前 TP slice 在 F 维的起点。

完整 Python 梯度 tensor：

~~~text
gate/up: [E, F, H]
down:    [E, H, F]
~~~

顶层 <code>moe-sft-tp.hpp</code> 计算首地址：

~~~text
gate/up = base + O × H
down    = base + O
~~~

子核 <code>sft_moe.hpp</code> 写回：

~~~text
gate/up = slice + expert × F × H + local_i × H + h
down    = slice + expert × H × F + h × F + local_i
~~~

这个分工的关键点是：

1. 循环范围使用 local I。
2. expert stride 和 down row stride 使用 global F。
3. 顶层在 dispatch 前统一清完整 tensor。
4. 各 TP 只写自己的 slice，且 tile task 只写自己的输出块。
5. 不能先对 null 做 offset，再在子核内检查。

## 7. 修改原因的故障证据链

### 7.1 初始 Full FT 接入后，专家权重仍不更新

初始实现已经把基座 Parameter 注入 optimizer 并产生 requant 调用，但 15 步测试仍出现梯度和权重变化为零。代码根因之一是：

~~~text
MOESFTConfig
  → 转成 GeneralMOEConfig 构造 TP 基类
  → full_weight_grad 字段被切掉
  → 子核中的开关保持 false
  → backward_base_weight_grad 被跳过
~~~

这对应 S08 和 M01。

### 7.2 修复开关和 TP stride 后出现 SIGSEGV

GDB 原始日志记录：

- 进程以 Signal 11 停止。
- 栈顶位于当时 <code>sft_moe.hpp:1968</code> 的 <code>backward_base_weight_grad</code>。
- 两个 NUMA 子核都进入相同错误路径。

根因不是最终梯度写回 stride，而是初版将 <code>m_local_pos_cache</code> 当成 <code>[expert][token]</code>。其真实结构是 <code>[token][route_slot]</code>；用 expert id 取第一维并用 expert 内 token 序号取第二维，超过 top-k 后会越界，随后用垃圾 token id 读取 input。

最终实现不再反查该路由表，而使用 backward 已恢复和打包好的：

- <code>m_local_input_ptr_[expert]</code>
- expert-major 的 gate/up/intermediate buffer

这对应 S11-b 和 S11-h。

### 7.3 SIGSEGV 消失后，down 梯度仍为零

三步测试中：

~~~text
gate_proj: 48/48 非零
up_proj:   48/48 非零
down_proj:  0/48 非零
总计：     96/144 非零
~~~

C++ grad buffer 与 Parameter.grad 的非零数一致，因此问题发生在 C++ 计算前的数据生命周期，而不是 autograd 返回时丢失。

根因是 down 需要的 route-weighted dY 先写入 <code>grad_output_bf16_ptr_</code>，随后这个工作区被 gate/up grad-input 覆盖。S01～S03、S05～S06、S12～S18 共同实现了独立快照的完整生命周期。

### 7.4 保存快照并 AMX 并行后的结果

五步验证中：

~~~text
C++ grad buffer：144/144 非零
Parameter.grad： 144/144 非零
gate/up/down：分别 48/48、48/48、48/48
抽样权重：9/12 tensor 发生变化
down 抽样：4/4 发生变化
最大权重差：约 3.8147e-05
backward 平均：754.484 s → 22.755 s
~~~

这些结果证明：

- Python Parameter → C++ grad → autograd → optimizer 的结构链路已经连通。
- down dY 生命周期修复有效。
- optimizer 后重新量化确实使用了更新后的基座权重。
- AMX/NUMA 并行消除了原标量实现的主要性能瓶颈。

但同一验证中最大梯度仍达到约 <code>1.1259e17</code>。AdamW 对梯度的归一化可能让权重变化保持有限，因此“权重有合理小变化”不能反推“原始梯度数值正确”。

## 8. 设计影响与边界

### 8.1 内存所有权发生变化

Full FT 模式下的权威权重变成 wrapper 中的 CPU BF16：

~~~text
gate_proj_buf / up_proj_buf / down_proj_buf
~~~

HF model tree 中原 expert 权重被替换成 zero-storage placeholder。这样可以避免同一组大权重同时存在于：

- HF model 参数树；
- optimizer 可见的 BF16 Parameter；
- AMX 量化 BufferB。

因此，不能再通过 HF model tree 中 placeholder 的数值比较来判断专家是否更新；必须检查 <code>*_proj_buf</code>。

### 8.2 AMX forward 使用派生副本

optimizer 更新的是 BF16 Parameter，而 AMX forward 使用内部量化 BufferB。每个 optimizer step 后必须：

~~~text
mark _base_weights_dirty
  → update_base_weights()
  → set_base_weight_pointers()
  → load_weights_task()
  → 重新分区并量化
~~~

缺少任何一步都会出现“Parameter 在变，但 forward 仍使用旧 expert 权重”的假训练。

### 8.3 分布式代价

当前 Python 实现会把 CPU 基座梯度复制到 GPU 做 all-reduce，再复制回 CPU。它保证各 rank optimizer 更新一致，但三组 expert 全量 BF16 梯度很大，通信和两次设备拷贝可能成为新瓶颈。报告中的 AMX backward 加速不等于端到端 step 已经同等加速。

### 8.4 兼容性改动与 Full FT 核心要区分

GLM4 架构识别、transformers v5 TopKRouter 包装、路由 top-k 复现属于同期兼容性修改。它们扩大了模型覆盖面，也会影响 router/PEFT 行为，但不是解决 Full FT 基座梯度为零、SIGSEGV 或 down 梯度为零的必要条件。

## 9. 尚未关闭的风险

1. 数值尺度异常：现有梯度最大值约 <code>1e16～1e17</code>，应逐阶段检查 route weight、激活导数、AMX tile pack/transpose 和 BF16 转换。
2. 缺少小尺寸参考：需要用 PyTorch 对 gate/up/down 的 outer product 做逐元素对照，而不只是检查 finite/nonzero。
3. TP 覆盖验证不足：至少需要 TP=1、TP=2，并包含 <code>expert_idx&gt;0</code>，因为 expert 0 会掩盖错误 expert stride。
4. 跨 step 残留：应连续两步激活不同 experts，确认 M06 的统一清零使 inactive expert 为零。
5. 模式回归：需要分别验证 full、hybrid、lora；尤其确保 <code>full_weight_grad=false</code> 时不复制/使用 dY 快照、不写基座梯度且原 LoRA 结果不变。当前快照池容量仍被计入普通 backward pool，可继续评估 LoRA 模式是否应避免这部分额外内存。
6. 上游同步：个人分支落后官方 main 3 个提交，虽然当前看与 Full FT 核心无关，合并后仍应重新构建并回归。
7. SkipLoRA 入口：Python <code>_is_skip_lora</code> 分支当前给三个基座梯度参数传 0；纯 Full 依赖使用普通 AMX SFT 方法并以 <code>lora_rank=0</code> 跳过 LoRA。若未来希望“SkipLoRA 方法名 + Full FT”组合工作，需要单独统一这条入口语义。
8. 分布式梯度对象：基座 all-reduce 修改的是 C++ grad buffer；应显式测试多卡下它与 <code>Parameter.grad</code> 是否共享/保持同步，避免单卡正确而多卡 optimizer 仍读取旧梯度。

## 10. 最终判断

该分支的真实价值可以概括为三点：

1. 功能上，它第一次让 KT CPU/AMX experts 的 gate/up/down 基座权重进入 PyTorch 的可训练闭环，而不再只有 LoRA 或其他 GPU 参数能更新。
2. 正确性修复上，它解决了 config slicing、TP slice/global stride、路由缓存布局误读、down dY 工作区覆盖和零秩 LoRA 等一系列跨层问题。
3. 性能上，它把基座 outer-product 梯度从每 NUMA 单线程标量三重循环改为 AMX BF16 tile + FP32 累加 + NUMA subpool 并行，使已记录的 backward 均值从约 754.5 秒降到约 22.8 秒。

当前最准确的状态描述是：Full FT 的结构链路、三组梯度非零、权重更新和 requant 已被验证；数值尺度及逐元素参考正确性尚未验证完成。

## 11. 复核命令与证据位置

代码统计与逐段 diff：

~~~bash
cd /mnt/data2/wbw/ktransformers
git merge-base upstream/main fullft-development
git diff --shortstat upstream/main...fullft-development
git diff --numstat upstream/main...fullft-development
git diff --unified=0 8e46e58..fullft-development -- \
  kt-kernel/operators/amx/sft_moe.hpp
git diff --unified=0 8e46e58..fullft-development -- \
  kt-kernel/operators/moe-sft-tp.hpp
~~~

本地原始验证证据：

~~~text
/mnt/data2/wbw/FFTtest/Qwen3-30B-A3B/test_log/
  20260713_105328_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
    phase4/gdb_sigsegv.log
  20260713_120636_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
    expert_gradient_check.txt
    expert_weight_change_check.txt
    phase4/step_timing/
  20260713_140101_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
    expert_gradient_check.txt
    expert_weight_change_check.txt
    phase4/step_timing/
~~~

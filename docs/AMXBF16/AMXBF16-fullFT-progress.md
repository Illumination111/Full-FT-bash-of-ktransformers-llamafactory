# AMX BF16 Full-FT 代码进度与训练链路

> 代码基准：KTransformers PR [#2086](https://github.com/kvcache-ai/ktransformers/pull/2086)，head [`1e95053`](https://github.com/kvcache-ai/ktransformers/commit/1e95053b15b32e6db8193fd852d62d051c6e7ef5)，核对时间 2026-07-17。
>
> 本文只描述该 commit 上的代码事实。PR 当前仍为 Open，GitHub 判定可合并但状态为 `UNSTABLE`，且当前 head 没有 CI status check；因此“结构链路已接通”不等于“该 head 已完成端到端验收”。

## 当前结论

PR #2086 已经在代码结构上接通 AMX BF16 CPU MoE 专家基础权重的 Full-FT 链路：

1. 基础 Gate、Up、Down 权重成为 optimizer 可见的 CPU BF16 `nn.Parameter`；
2. C++ backward 计算 `dX`、router 权重梯度以及三组基础权重梯度；
3. 自定义 Autograd Function 把 C++ 写入的基础权重梯度返回给 PyTorch；
4. optimizer 更新权威 BF16 参数后，下一轮 forward 前重新生成 AMX forward BufferB；
5. backward 所需的转置 BufferB 可以从 forward BufferB 直接 repack，并可跨层异步准备；
6. 当前 AMX BF16 绑定复用 inference 的 `GemmKernel224BF16`，基础权重 dW 使用独立的 `BF16DWeightKernel`；
7. C++/Python 已加入分阶段 profiler，可分别观察 forward、backward、dWeight、base-weight reload 和 backward repack。

当前仍不能宣称功能已经完整验收：最新 dWeight 和 direct BF16 reload 提交只有聚焦测试源码与 profiler 测试，PR head 没有 CI 结果，也没有保留一份针对 `1e95053` 的完整 Full-FT reference、15-step 模式回归和 TPS 报告。

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

非重入 gradient checkpoint 下，第一次 forward 处于 `first_forward` hook 时，Python 把 `save_for_backward_submit` 设为 `False`，避免保留整层缓存；`KTMoEFunction` 保存一个 sentinel tensor。backward 访问 `ctx.saved_tensors` 时触发整层重算，重算 forward 再以 `save_for_backward=True` 填充 C++ cache，然后才调用真正的 MoE backward。

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

1. Python backward 先 `wait_backward_repack()`，避免 checkpoint 重算和仍在运行的 repack 同时使用共享池；
2. 当前层 backward 检查共享池 owner，必要时同步 fallback repack；
3. 当前层完成后，`KTMoEFunction.backward()` 对 backward 顺序中的下一层调用 `submit_backward_repack()`；
4. C++ 独立线程在 GPU attention 等其他 backward 工作期间准备下一层布局。

这条路径只影响 backward 权重布局，不能与 optimizer 后的 base-weight reload 混为同一个阶段。

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
  └─ 普通训练保存 cache；checkpoint 首次 forward 跳过 cache

loss
  │
  ▼
第 t 轮 backward
  │
  ├─ 等待共享 backward BB repack
  ├─ checkpoint：访问 sentinel，触发整层重算并填充 cache
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
| `backward.down.*` | route-weighted dY、Down base dX 和 Down LoRA |
| `backward.gate_up.*` | Gate/Up base dX 和 LoRA |
| `backward.base_weight_grad.matmat` | dWeight worker-pool 的外层墙钟 |
| `backward.base_weight_grad.worker_cpu.*` | worker 累加的 pack/kernel/store CPU 时间，不是关键路径墙钟 |
| `backward.repack` | forward BufferB → backward BufferB |
| `weights.base_reload.direct_pack` | 当前 BF16 直接从完整权威参数生成 TP forward BufferB |
| `weights.base_reload.partition/forward_pack/cleanup` | 非 direct kernel 的 fallback reload |

Python 还使用 `torch.profiler.record_function` 标出 routing、base weight reload、CPU forward sync、checkpoint recompute、CPU backward、wait/submit repack，便于与 GPU timeline 对齐。

一个已知的标签限制是：C++ 用 `save_for_backward` 区分 initial 和 recompute。在 checkpoint 流程中第一次 forward 为 `False`、重算为 `True`，标签成立；但非 checkpoint 的普通训练初次 forward 也会传 `True`，可能被记录成 `forward.recompute.total`。这不影响训练数值，但分析非 checkpoint profile 时不能按名称直接归因。

## 11. 聚焦测试和验收缺口

当前 head 新增或扩展了以下测试：

- [`test_bf16_dweight.cpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/test/test_bf16_dweight.cpp)：以 BF16 输入和 FP32 标量 reference 检查 route/tail case，阈值为 relative L2 `<= 0.01`、cosine `>= 0.999`；`--benchmark` 比较 common driver 与 legacy tile loop，并限制单 tile driver 不得回退超过 5%。
- [`test_raw_bf16_repack.cpp`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/operators/amx/test/test_raw_bf16_repack.cpp)：检查普通 pack/unpack、转置、直接 BB→BB 转置和 strided source pack 的 BF16 bitwise 一致性。
- [`test_sft_profiler.py`](https://github.com/kvcache-ai/ktransformers/blob/1e95053b15b32e6db8193fd852d62d051c6e7ef5/kt-kernel/test/per_commit/test_sft_profiler.py)：用 fake MoE 检查 profile 收集、聚合、格式化、reset 和 worker CPU stage 显示。

这些测试说明新组件有明确的局部验证入口，但仍有以下验收缺口：

1. AMX C++ test 只在 `KTRANSFORMERS_CPU_DEBUG` 打开时由 CMake 的 AMX test glob 构建；当前 GitHub PR head 没有 CI check 证明它们已经执行。
2. Python profiler test 不加载真实 C++ extension，不能覆盖 pybind、NUMA 并发和硬件计数器。
3. `test_bf16_dweight` 验证单个 tile driver，不等价于验证 `TP_MOE_SFT::backward()` 的完整 expert/TP pointer、清零和 Autograd 生命周期。
4. `1e95053` 需要重新执行 Full、LoRA-only、Hybrid/Router、checkpoint on/off 和多 rank 回归；早于最新 kernel/dWeight/direct-reload 提交的日志不能单独证明当前 head。
5. 需要以 PyTorch reference 对比 forward、dX、selected router gradient 和三组 dW，并检查多 step loss、grad norm、NaN/Inf、checkpoint 保存/恢复。
6. 性能结论必须基于最新 head 的同配置 A/B；代码减少了临时 TP copy，并不自动证明 TPS 已提高或 base reload 已完全消失。

因此当前状态应表述为：

> **Full-FT 结构链路完整，最新 BF16 dWeight 和 direct reload 已有聚焦测试代码；端到端正确性、模式回归、CI 和最新 TPS 证据仍待补齐。**

## 参考提交

- [`9dc9d93`：staged SFT profiler](https://github.com/kvcache-ai/ktransformers/commit/9dc9d93b7bf1b00124475cde5ff6cae97741e28d)
- [`c6f4211`：SFT 复用 inference BF16 kernel](https://github.com/kvcache-ai/ktransformers/commit/c6f4211346e1aa7b42f66551c33f9a131a2e4dc3)
- [`109b403`：Full-FT 细粒度 profiler](https://github.com/kvcache-ai/ktransformers/commit/109b403b633865f75c1a86411f3c2749c229725d)
- [`34d2102`：BF16 dWeight driver 与 direct BF16 TP reload](https://github.com/kvcache-ai/ktransformers/commit/34d2102082c69fdbcb49e97fa653ce4352a2cd18)
- [`ea84e6e`：BF16 dWeight AMX benchmark](https://github.com/kvcache-ai/ktransformers/commit/ea84e6edf44d2576dc3078d60f5b2b808b42d095)
- [`273e670`：worker CPU profiler 标签修正](https://github.com/kvcache-ai/ktransformers/commit/273e670890c7b966ce537cd1bb4acf69c8202767)
- [`1e95053`：当前 PR head](https://github.com/kvcache-ai/ktransformers/commit/1e95053b15b32e6db8193fd852d62d051c6e7ef5)

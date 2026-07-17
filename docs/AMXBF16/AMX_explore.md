# KT-Kernel AMX 代码与 Python 调用链详解

> 调查对象：`/mnt/data2/wbw/ktransformers/kt-kernel` 当前工作树。本文中的相对路径均以该目录为根目录。
>
> 本文聚焦三类代码：AMX/AVX-512 底层算子、在其上构建的 MoE 推理与 SFT 实现、Python 到 C++ AMX 后端的接口和任务调度。纯 CUDA/KML/llamafile/AVX2 实现不展开，只在后端选择或回退关系中说明。

## 1. 先给出整体结论

KT-Kernel 的 AMX 路径不是一个单独的 GEMM 函数，而是下面这组分层组件共同组成的：

```text
Python 模型封装
  python/experts.py、python/sft/*
        │
        ▼
Python AMX 后端与权重加载
  python/utils/amx.py、python/utils/loader.py
        │  整数形式的 Tensor.data_ptr()
        ▼
pybind11 接口
  ext_bindings.cpp
        │  返回 (函数地址, 参数对象地址) 任务
        ▼
异步执行与 NUMA/线程池
  cpu_backend/cpuinfer.h、worker_pool.h、task_queue.h
        │
        ▼
TP/NUMA MoE 调度
  operators/moe-tp.hpp、operators/moe-sft-tp.hpp
        │
        ▼
AMX MoE 算法层
  operators/amx/*-moe.hpp、moe_base.hpp、sft_moe.hpp
        │
        ▼
缓冲区、量化、排布和微内核
  operators/amx/la/amx_buffers.hpp、amx_kernels.hpp 等
        │
        ▼
Intel AMX tile 指令 / AVX-512 回退
```

几个最重要的判断：

1. Python 只负责选择后端、准备权重和工作缓冲区，并把地址传入 C++；Python 本身不操作 AMX tile。
2. `ext_bindings.cpp` 暴露的 `*_task()` 通常不立即计算，而是创建一个可交给 `CPUInfer` 的任务对象。
3. `TP_MOE`/`TP_MOE_SFT` 把中间维 `F` 分给多个 NUMA 子池，每个子实例执行局部计算，最后合并结果。
4. `operators/amx/la/` 才是最接近硬件的层：配置 tile、量化和排布 A/B、执行 tile dot-product、把 FP32 累加结果写回 C。
5. “类名或目录名含 AMX”不等于每次都执行 AMX 指令。是否可用由编译宏决定；某些短向量路径和 LoRA 路径明确使用 AVX-512。
6. Python/C++ 边界大量使用裸指针和异步任务，因此 Tensor、mmap 权重、pinned buffer 的生命周期是正确性的组成部分，而不只是性能细节。

## 2. 数据形状和术语

本文沿用源码中的命名：

| 符号 | 含义 |
| --- | --- |
| `E` / `expert_num` | 专家总数 |
| `T` / `qlen` | 本次送入 CPU MoE 的 token 数 |
| `topk` / `num_experts_per_tok` | 每个 token 命中的专家数 |
| `H` / `hidden_size` | 隐藏维度 |
| `F` / `intermediate_size` | 单个专家的中间维度 |
| `M/N/K` | GEMM 的行数、输出列数、归约维度；在不同投影中会对应 `T/H/F` 的不同组合 |
| gate/up/down | MoE 的三个基础投影：`gate(x)`、`up(x)`、`down(activation)` |
| prefill | `qlen` 较大、同一专家通常收到多个 token，适合矩阵乘 |
| decode | `qlen` 很小，常退化为矩阵-向量乘，内存带宽和小批量开销更关键 |
| TP | 此处主要是按 `F` 在 NUMA 子池间切分的 tensor parallel，不是 Python 进程级模型并行 |
| SFT | 监督微调路径，支持 LoRA 以及可选的基础权重全量梯度 |

单个专家的前向公式可概括为：

```text
gate = x · W_gate^T
up   = x · W_up^T
h    = activation(gate, up)
y_e  = h · W_down^T
y    = Σ routing_weight(token, e) · y_e
```

标准激活是 `silu(gate) * up`。源码还支持：

- DeepSeek V4 风格的 gate 单侧、up 双侧截断，由 `swiglu_limit` 控制；
- MiniMax M3 的 `gate * sigmoid(gate * alpha) * (up + 1)`，由 `swiglu_alpha` 控制。

## 3. Python 到 AMX 的完整调用链

### 3.1 扩展加载阶段

```text
import kt_kernel
  └─ python/__init__.py
      └─ python/_cpu_detect.py::initialize()
          ├─ 探测 amx_tile / amx_bf16 / amx_int8、AVX-512、AVX2
          ├─ 读取 KT_KERNEL_CPU_VARIANT（若用户强制指定）
          └─ 加载匹配的 _kt_kernel_ext_<variant>.so
                └─ sys.modules["kt_kernel_ext"] = extension
```

这里解决的是“加载哪个编译变体”。加载到 AMX 变体之后，具体格式仍可能在 `python/utils/amx.py` 中做更细的 AMX、AVX-VNNI 或 AVX2 选择。

### 3.2 推理前向链

以 `method="AMXINT4"` 为例：

```text
KTMoEWrapper(..., mode="inference", method="AMXINT4")
  └─ python/experts.py::_create_inference_wrapper
      └─ AMXMoEWrapper
          ├─ SafeTensorLoader 取得 gate/up/down 浮点权重
          ├─ 构造 pybind 的 MOEConfig，字段中写入 tensor.data_ptr()
          ├─ AMXInt4_MOE(config)
          └─ cpu_infer.submit(moe.load_weights_task(...))
                  C++ 在线量化并重排为 BufferB 的 AMX 友好布局

forward:
BaseMoEWrapper.submit_forward(...)
  └─ moe.forward_task(qlen_ptr, topk, ids_ptr, weights_ptr,
                      input_ptr, output_ptr, incremental)
      └─ ext_bindings.cpp 创建不透明任务参数
          └─ CPUInfer.submit / submit_with_cuda_stream
              └─ TP_MOE::forward
                  ├─ 每个 NUMA TP 子实例 AMX_MOE_BASE::forward
                  ├─ 按专家聚集 token
                  ├─ gate/up GEMM
                  ├─ SwiGLU
                  ├─ down GEMM
                  └─ TP_MOE::merge_results，按路由权重归并
```

`qlen` 是通过指针传入而非按值传入，便于 CUDA stream host callback 在真正执行任务时读取当前有效长度。输入、输出、专家 id 和路由权重同样只是地址；异步完成以前不能销毁或复用承载这些地址的 Tensor。

### 3.3 原生压缩格式的推理链

`RAWINT4`、`BF16`、`FP8`、`FP8_PERCHANNEL`、`MXFP4`、`MXFP8` 使用 `NativeMoEWrapper`。与 `AMXINT4/8` 的主要差别是：

- 权重在检查点里已经是目标格式，loader 负责字段命名、形状、scale/zero 的统一；
- C++ 主要做分区、复制和 AMX-friendly pack，而不是从 BF16/FP16 做通用在线量化；
- Python 会根据编译产物、CPU flag 和环境变量选择 AMX/AVX-512 或 AVX2 类；
- loader 必须保持 safetensors mmap 和 backing tensor 存活，直到所有层完成 C++ 复制/pack。

### 3.4 SFT 前向和反向链

```text
python/sft/wrapper.py
  ├─ 识别模型架构和 MoE 层
  ├─ 提取/加载专家基础权重
  └─ 用 KTMoELayerWrapper 替换原模型 MoE
        │
        ▼
python/sft/layer.py::forward
  ├─ GPU 上计算 router/top-k
  ├─ 准备 CPU pinned buffer
  └─ KTMoEFunction.apply(...)
        │
        ├─ forward: python/sft/autograd.py
        │    └─ AMXSFTMoEWrapper.forward[_async]
        │         └─ C++ forward_sft_task
        │              └─ TP_MOE_SFT -> AMX_SFT_MOE_TP
        │
        └─ backward: python/sft/autograd.py
             └─ AMXSFTMoEWrapper.backward[_async]
                  └─ C++ backward_task
                       ├─ down 投影反传
                       ├─ activation 反传
                       ├─ gate/up 投影反传
                       ├─ LoRA A/B 梯度
                       ├─ grad_input / grad_router_weight
                       └─ 可选基础 gate/up/down 全量梯度
```

SFT 路径仍把基础专家 GEMM 放在 AMX/NUMA 侧，但 LoRA 的小秩乘法大量使用 `la/avx_kernels.hpp` 中的 AVX-512 BF16 内核。全量训练时，Python 暴露 BF16 基础权重 `Parameter`；基础权重被优化器修改后，`set_base_weight_pointers()` 加 `load_weights_task()` 会把新 BF16 权重重新量化/pack 成下次 AMX 前向使用的布局。

## 4. 方法名、Python 类、绑定类和 C++ 实现的对应关系

### 4.1 推理

| Python `method` | Python wrapper | pybind 类 | 高层 C++ 实现 | 核心 kernel/格式 |
| --- | --- | --- | --- | --- |
| `AMXINT8` | `AMXMoEWrapper` | `AMXInt8_MOE` | `AMX_MOE_TP` | `GemmKernel224Int8`，在线对称 INT8 |
| `AMXINT4` | `AMXMoEWrapper` | `AMXInt4_MOE` | `AMX_MOE_TP` | `GemmKernel224Int4`，在线对称 INT4 |
| `RAWINT4` | `NativeMoEWrapper` | `AMXInt4_KGroup_MOE`（优先） | `AMX_K2_MOE_TP` | `GemmKernel224Int4SmallKGroup`，预量化小 K-group |
| `BF16` | `NativeMoEWrapper` | `AMXBF16_MOE`（AMX/AVX-512 变体） | `AMX_BF16_MOE_TP` | `GemmKernel224BF16` |
| `FP8` | `NativeMoEWrapper` | `AMXFP8_MOE` | `AMX_FP8_MOE_TP` | `GemmKernel224FP8`，block scale |
| `FP8_PERCHANNEL` | `NativeMoEWrapper` | `AMXFP8PerChannel_MOE` | `AMX_FP8_PERCHANNEL_MOE_TP` | `GemmKernel224FP8PerChannel` |
| `MXFP4` | `NativeMoEWrapper` | `AMXFP4_KGroup_MOE`（优先） | `AMX_FP4_MOE_TP` | E2M1 4-bit + UE8M0 group scale |
| `MXFP8` | `NativeMoEWrapper` | `AMXMXFP8_KGroup_MOE`（有 AMX 时优先） | `AMX_MXFP8_MOE_TP` | E4M3fn + UE8M0 group scale |

`ext_bindings.cpp` 还绑定了 `AMXInt4_1_MOE`、`AMXInt4_1KGroup_MOE`，分别对应有 zero-point 的 INT4 和 AWQ 风格低 K-group 路径。它们主要被专项示例/测试或上层集成直接调用，并不是当前 `KTMoEWrapper` 推理工厂列出的主方法名。

### 4.2 SFT

| Python `method` | pybind 类 | C++ 模板实例 |
| --- | --- | --- |
| `AMXBF16_SFT` | `AMXBF16_SFT_MOE` | `AMX_SFT_MOE_TP<GemmKernel224BF>` |
| `AMXINT8_SFT` | `AMXInt8_SFT_MOE` | `AMX_SFT_MOE_TP<GemmKernel224Int8>` |
| `AMXINT4_SFT` | `AMXInt4_SFT_MOE` | `AMX_SFT_MOE_TP<GemmKernel224Int4>` |
| 上述三种加 `_SkipLoRA` | 对应 `*_SkipLoRA` 类 | 同一模板，第三个模板参数为 `true` |

`python/experts.py` 的可选集合还保留了 INT4_1/K-group SFT 名称，但 `python/sft/amx.py` 的类映射以及 `ext_bindings.cpp` 中对应绑定目前没有启用。因此这些名称通过第一层校验后仍会在真正创建 SFT backend 时失败；不能把“工厂允许该字符串”理解为“编译产物一定提供实现”。

## 5. C++ AMX 底层：`operators/amx/la/`

这一目录负责硬件使能、tile 配置、数据布局、量化、微内核和少量 AVX-512 辅助计算。理解 AMX 性能问题时应先看这里。

### 5.1 `operators/amx/la/amx_config.hpp`

作用：AMX/AVX-512 的编译兼容层和 tile 运行时配置入口。

关键内容：

- 定义 `ALWAYS_INLINE`、`RESTRICT` 等微内核常用宏；
- 在编译器缺少原生 AVX512-BF16/VBMI intrinsic 时提供等价兼容实现；
- Linux x86 上通过 `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` 请求 tile data 权限；
- `enable_amx()` 是每个使用 AMX 的执行线程进入 tile 代码前的重要前置条件；
- `TileConfig` 对应 Intel 64-byte tile configuration：palette、每个 tile 的 `rows` 和 `colsb`；
- 封装 `_tile_loadconfig`，以及按 tile 编号分派 `_tile_loadd`/`_tile_stored`。

注意：AMX 权限和 tile 配置带有线程上下文属性，不能只在主线程初始化一次后假设工作线程自动继承全部状态。各 kernel 的 `init()`/入口会再次调用 `enable_amx()`。

### 5.2 `operators/amx/la/amx.hpp`

作用：底层 AMX API 的聚合头和通用入口。

关键内容：

- 包含 config、buffer、kernel 等底层实现；
- 提供 `AMX_DISPATCH_QTYPES`，把 ggml 数据类型映射到具体模板实例；
- `init_tile()` 负责按数据类型准备 tile；
- `gemm()` 是给通用调用者使用的类型分派入口；
- `recommended_nth()` 根据矩阵形状给出线程数建议；
- 实现 AVX-512 向量化 `exp` 和 `act_fn()`，后者统一处理标准 SwiGLU、DeepSeek V4 截断和 MiniMax M3 `swigluoai`。

`amx-example.cpp` 走的就是这里的通用 `amx::gemm()`；高层 MoE 为减少动态分派，更多直接实例化具体 `GemmKernel*`。

### 5.3 `operators/amx/la/amx_kernels.hpp`

作用：传统 BF16/INT8/INT4 量化格式的 AMX 矩阵乘微内核，是底层计算主体。

重要结构：

- `AMX_AVAILABLE`：由 `HAVE_AMX` 等编译条件形成的编译期常量；高层用它选择 tile 路径还是 AVX-512 路径；
- `GemmKernel133`、`GemmKernel133BF`：使用 1 个 A tile、3 个 B tile、3 个 C tile 的 1-3-3 排布；
- `GemmKernel224BF`：2 个 A、2 个 B、4 个 C 的 BF16 2-2-4 排布；
- `GemmKernel224Int8`：INT8 dot-product，结合 A/B scale 恢复到 FP32；
- `GemmKernel224Int4`：对称 INT4 权重路径；
- `GemmKernel224Int4_1`：带 zero/min 修正的非对称 INT4；
- `GemmKernel224Int4KGroup`、`GemmKernel224Int4_1KGroup`：按 K 维分组缩放的版本；
- `GemmKernel224Int4_1_LowKGroup`：适配 AWQ 小 group 的非对称路径；
- `GemmKernel224Int4SmallKGroup`：K2/RAWINT4 对称小 group 路径；
- `GemmKernel<TA, TB, TC>` 特化：把 ggml block 类型映射到上述 kernel。

微内核通常做四件事：加载/配置 tile；从 `BufferA`/`BufferB` 取得已经排布的数据；执行 `_tile_dpbf16ps` 或 `_tile_dpbssd` 一类 dot-product；结合 scale、sum、zero-point 做反量化并累计到 FP32 `BufferC`。

模板参数中的 `amx_or_avx` 会让若干 wrapper 在 `AMX_AVAILABLE` 为真时走 AMX，否则调用同文件或辅助代码里的 AVX-512 mat-vec 实现。decode 的小 `M` 场景不一定适合完整 tile GEMM，所以源码同时保留 `mat_mul`、`vec_mul` 和 K-group 变体。

### 5.4 `operators/amx/la/amx_raw_kernels.hpp`

作用：面向原生压缩检查点格式的 BF16/FP8 微内核。

主要类型：

- `GemmKernel224BF16`：直接消费原生 BF16 B buffer；矩阵乘走 AMX，向量乘明确调用 AVX-512 浮点 mat-vec；
- `GemmKernel224FP8`：FP8 byte 权重配合 block/group scale，计算前按块恢复可参与 BF16 dot-product 的值；
- `GemmKernel224FP8PerChannel`：每个输出 channel 一个 scale，服务 GLM-4.7 风格权重。

这里最能说明“AMX backend 不等于所有形状都执行 tile”。例如 BF16 `vec_mul` 刻意采用 AVX-512；FP8 的 group wrapper 也会根据模板常量和编译能力选择 AVX/AMX 分支。

### 5.5 `operators/amx/la/amx_buffers.hpp`

作用：传统量化路径 A/B/C 三类矩阵的存储、量化、pack 和归约。kernel 能否连续、高复用地喂给 tile，很大程度由这里决定。

核心 buffer：

- `BufferAImpl`：输入 A 的分块、量化和 scale；
- `BufferAWithSumImpl`：除量化 A 外还记录行和，用于带 zero-point 的 B 修正；
- `BufferAWithSumKGroupImpl`：按 K-group 保存 A 的 scale/sum；
- `BufferAKGroupImpl`、`BufferASmallKGroupImpl`：为分组量化和小 group kernel 准备 A；
- `BufferBInt4Impl`：对称 INT4 B 的 nibble 解包/pack 与 scale；
- `BufferBKGroupImpl`：按 K-group 组织 B；
- `BufferBInt4WithZeroImpl`：带 zero-point 的 INT4；
- `BufferBInt4KGroupImpl`、`BufferBInt4WithZeroKGroupImpl`、`BufferBInt4WithZeroLowKGroupImpl`：各种 group/zero 组合；
- `BufferCImpl`：FP32 累加输出；
- `BufferCReduceImpl`：在多个局部块或线程结果之间做归约。

这些类还承担维度 padding、64-byte 对齐、tile 行列次序转换、不同 block 的地址计算。权重 load 阶段的耗时主要就在量化和创建 B buffer；运行阶段则尽量只做顺序加载与 dot-product。

### 5.6 `operators/amx/la/amx_raw_buffers.hpp`

作用：为原生 BF16/FP8 格式提供比通用量化 buffer 更直接的布局。

主要类型：

- `BufferABF16Impl`：把输入转换/pack 为 BF16 A tile；
- `BufferBBF16Impl`：原生 BF16 权重 B 的 pack 和访问；
- `BufferBFP8Impl`：FP8 block 权重及 scale 的布局；
- `BufferBFP8PerChannelImpl`：FP8 per-channel 权重和 scale；
- `BufferCFP32Impl`：FP32 输出；
- `BufferCFP32ReduceImpl`：局部 FP32 输出归约。

它与 `amx_raw_kernels.hpp` 成对使用：前者定义数据如何落内存，后者定义 tile/向量指令如何读取和计算。

### 5.7 `operators/amx/la/amx_quantization.hpp`

作用：AMX 路径需要的参考量化格式、量化/反量化和向量解包工具。

关键内容：

- 对齐的 `q4_0`、`q8_0` block 数据结构；
- FP32/BF16 到 INT8、INT4 的量化和反量化参考函数；
- `Dequantizer` 帮助类和 scale/min 解码；
- AVX-512 nibble 解包、dot-product 兼容辅助；
- 既用于生产 pack，也用于测试中生成参考结果和检查误差。

### 5.8 `operators/amx/la/amx_utils.hpp`

作用：AMX tile 和 AVX-512 寄存器级调试/转置辅助。

包括 tile 内容打印、寄存器 dump，以及 `8x8`、`16x4`、`16x8`、`16x16` 等小矩阵转置。它们服务于 B 权重重排、反向权重转置和底层测试；不是面向 Python 的公共 API。

### 5.9 `operators/amx/la/avx_kernels.hpp`

作用：SFT 中 LoRA 和转置相关的小矩阵 AVX-512 BF16 内核。

主要负责：

- LoRA A 前向；
- LoRA B 前向并融合加到基础专家输出；
- A/B 转置权重版本；
- LoRA backward matmul 和权重梯度；
- 支持 SFT cache/反传的转置及局部规约。

LoRA rank 通常远小于 `H/F`，用 AMX tile 未必能覆盖其准备开销，因此该文件使用 AVX-512 是有意的混合实现，而不是 AMX 功能缺失。

### 5.10 `operators/amx/la/utils.hpp` 与 `operators/amx/utils.hpp`

两个文件都提供 SIMD 基础辅助，但层级略有差异：

- `la/utils.hpp`：线性代数层常用的 AVX-512 BF16 copy、FP32/BF16 转换、向量 `exp`、绝对值最大值等；在没有原生 AVX512-BF16 intrinsic 时用位操作实现 round-to-nearest-even；
- `operators/amx/utils.hpp`：AMX 高层目录可直接使用的一组同类轻量 helper。当前内容主要是 32 个 BF16 的复制/转换、向量指数和最大绝对值。

两者名字相近，阅读 include 时要根据相对路径区分；它们不是 AMX tile kernel，而是数据转换和激活计算的 SIMD 支撑。

### 5.11 `operators/amx/la/pack.hpp`

作用：通用多维索引到二维 packed layout 的坐标工具。

`Packed2DLayout` 允许把若干逻辑维指定为 row 或 column digit，计算高维坐标、二维坐标和线性 offset 之间的转换。文件尾部的 `main()` 受 `PACKED2D_DEMO` 宏保护，只是自测示例。该文件更像布局实验/通用工具，当前不是 Python AMX 主调用链的关键节点。

### 5.12 `operators/amx/la/amx-example.cpp`

作用：最小化演示通用 AMX GEMM API。

它初始化 ggml，创建 FP32 A/B，转 BF16，调用 `amx::init_tile()`、`recommended_nth()` 和并行 `amx::gemm()`，最后与朴素 FP32 结果比较。适合先理解 tile 权限、数据类型、leading dimension 和 64-byte padding；不包含 MoE 路由、权重加载或 Python 绑定。

## 6. C++ AMX MoE 算法层：`operators/amx/`

这一层把通用的“路由—专家计算—归并”流程与不同权重格式组合起来。各格式文件普遍采用 CRTP：`AMX_MOE_BASE<T, Derived>` 实现公共流程，`Derived` 只提供 buffer 类型、权重加载和 `do_gemm()` 等格式特化逻辑。

### 6.1 `operators/amx/moe_base.hpp`

作用：所有现代 AMX MoE 推理实现共享的 CRTP 基类。

它维护的主要状态包括：

- expert-major 的本地 BF16 输入、gate/up/down 输出；
- `m_local_pos_`、`m_local_num_` 等 token 到专家的路由表；
- 每个专家的 A/B/C buffer 对象及其连续 backing pool；
- 当前 TP 分片编号、WorkerPool、共享内存和配置；
- prefill/decode 所需的临时空间。

前向的核心阶段是：

1. 扫描 `[qlen, topk]` 的专家 id，跳过无效专家和 GPU resident 专家；
2. 把 token 输入按专家聚集到连续缓冲区；
3. 为 gate/up 创建 A buffer，并调用 Derived 的 kernel；
4. 在 FP32/BF16 临时结果上执行激活；
5. 对 down 投影再次建 A buffer并计算；
6. 按原 token 位置和 routing weight scatter/reduce 到局部输出。

prefill 会尽量形成矩阵乘；decode 针对小 token 数选择更轻的 vector 路径。基类只描述流程，权重的量化格式、B buffer 和 kernel 由 `T`/`Derived` 决定。

文件尾部的 `TP_MOE<AMX_MOE_BASE<...>>` 特化定义该类如何接入通用 TP 框架和合并各分片结果。

### 6.2 `operators/amx/moe.hpp`

作用：通用在线量化 INT8/INT4/INT4_1 AMX MoE 实现。

`AMX_MOE_TP<T>` 为 `GemmKernel224Int8`、`GemmKernel224Int4`、`GemmKernel224Int4_1` 等模板实例提供：

- 对应 `BufferA`、`BufferB`、`BufferC` 类型别名；
- gate/up/down 权重从浮点源加载、量化、pack；
- 权重 cache 的读写；
- `do_gemm()` 对矩阵/向量和不同量化类型的分派；
- TP 分片后的权重范围和结果合并。

Python `AMXMoEWrapper` 的 `AMXINT8`/`AMXINT4` 最终进入这里。在线量化发生在 `load_weights_task()` 阶段，不在每次 forward 重做。

### 6.3 `operators/amx/bf16-moe.hpp`

作用：原生 BF16 专家权重的 MoE 实现。

特点：

- 使用 `BufferABF16Impl`、`BufferBBF16Impl`、`BufferCFP32Impl`；
- 权重不需要 INT 量化，load 阶段主要做 BF16 copy、转置/pack 和 TP 分区；
- 大矩阵使用 `GemmKernel224BF16` 的 AMX BF16 路径；小向量路径可使用 AVX-512；
- 提供 TP 专用的预分区权重加载；
- 包含 GPU expert offload 场景下的本地结果 copy/merge 支持。

这是精度最直观、也最适合对照其他量化格式误差的 AMX MoE 实现。

### 6.4 `operators/amx/fp8-moe.hpp`

作用：blockwise FP8 专家权重实现。

检查点保存 FP8 byte 权重和 FP32 scale，通常按 128 个 K 元素分组。该文件负责：

- 识别 gate/up/down 的 FP8 和 scale 地址；
- 把原生布局重排进 `BufferBFP8Impl`；
- 在 TP 切分时同步切分权重和 scale；
- 调用 `GemmKernel224FP8`；
- 在 AMX 编译不可用时保留 AVX-512 BF16 兼容执行路径。

### 6.5 `operators/amx/fp8-perchannel-moe.hpp`

作用：每输出通道一个 scale 的 FP8 MoE，主要面向 GLM-4.7-FP8 类检查点。

它与 blockwise FP8 文件的流程相似，但 scale 的索引维度、pack 和 kernel 修正不同，使用 `BufferBFP8PerChannelImpl` 与 `GemmKernel224FP8PerChannel`。不能把 block scale loader 生成的元数据直接交给 per-channel 实现。

### 6.6 `operators/amx/fp4-moe.hpp`

作用：MXFP4（E2M1）预量化 MoE。

关键点：

- 每个 byte 打包两个 4-bit E2M1 值；
- 每个 group 配 UE8M0 类型 scale，源码用适合向量/AMX 计算的表示保存；
- 文件内定义 `GemmKernel224MXFP4SmallKGroup`，完成 nibble 解码、scale 应用和 BF16 dot-product；
- `AMX_FP4_MOE_TP` 负责 DeepSeek V4 等模型的权重加载、TP 分区、激活截断参数传递；
- 支持 `swiglu_limit` 对 gate/up 的非对称截断。

这里的“FP4”是 MXFP4 格式，不应与 ggml `q4_0` 或普通整数 INT4 混为一谈。

### 6.7 `operators/amx/mxfp8-moe.hpp`

作用：MXFP8 E4M3fn + UE8M0 group scale 的 MoE，主要面向 MiniMax M3。

文件内定义 `GemmKernel224MXFP8SmallKGroup` 和格式转换/LUT 逻辑；`AMX_MXFP8_MOE_TP` 负责 pack、TP 权重分区以及 forward。它还把 `swiglu_alpha` 和 `swiglu_limit` 传给统一激活函数，从而执行 `swigluoai`。

Python 后端选择比其他格式更严格：除编译出类以外，还会检查主机 `amx_tile/amx_bf16`；缺少时优先选 AVX2 MXFP8，而不是盲目实例化名字中含 AMX 的类。

### 6.8 `operators/amx/k2-moe.hpp`

作用：K2/RAWINT4 对称小 K-group MoE。

该路径直接消费预量化 4-bit 权重和 group scale，不使用 zero-point。它以 `GemmKernel224Int4SmallKGroup` 为核心，负责：

- 验证 group size 和维度；
- 从压缩 safetensors 复制/pack nibble 权重；
- 对 gate/up/down scale 做相同的 TP 分区；
- 处理已预分区 NUMA 权重；
- 用 CRTP 基类完成前向。

当前 Python `RAWINT4` 在 AMX/AVX-512 后端可用时会选择该实现。

### 6.9 `operators/amx/awq-moe.hpp`

作用：AWQ 风格的非对称 INT4 小 K-group MoE。

与 `k2-moe.hpp` 的核心区别是每组除了 scale 还有 zero-point，因此 A buffer 需要保留 sum，kernel 用 `sum(A) * zero` 做零点修正。该文件包含：

- scale/zero/packed weight 的加载和布局转换；
- 在线量化、反量化和精度校验辅助；
- `GemmKernel224Int4_1_LowKGroup` 的调用；
- TP 权重分区、cache 读写和结果合并。

它对应绑定类 `AMXInt4_1KGroup_MOE`，主要由 AWQ 专项示例和集成代码使用。

### 6.10 `operators/amx/sft_moe.hpp`

作用：AMX MoE 的训练核心，覆盖 LoRA SFT、混合训练和基础权重全量梯度。该文件接近六千行，是 AMX 训练路径最集中的实现。

主要组成：

- `ForwardCache`：保存反向需要的路由、输入、gate/up/activation 等数据；
- NUMA 共享 cache/buffer pool：避免每层、每步反复分配大块内存；
- `AMX_SFT_MOE_TP<T, BaseMOE, SkipLoRA>`：在基础 MoE 类上增加前向缓存和训练接口；
- LoRA 前向：基础 gate/up/down 结果与 LoRA A/B 结果融合；
- LoRA 反向：分别计算 gate/up/down 的 A/B 梯度和输入梯度；
- 基础权重反向：计算 gate、up、down 的完整 `dW`，并写到 Python 提供的 FP32/BF16 梯度缓冲区；
- router 权重梯度：保留未乘路由权重的专家输出，按 token/expert 计算 `d routing_weight`；
- backward weight 准备：生成转置、预量化、AMX BufferB 形式的反向权重；
- 基础权重更新接口：替换 BF16 源指针、重新量化/pack；
- LoRA pointer 更新、保存/加载 backward weight、repack 和诊断统计。

反向按 down、activation、gate/up 三阶段推进。基础大矩阵运算尽可能使用 AMX，LoRA 小秩矩阵调用 `avx_kernels.hpp`。`SkipLoRA=true` 的模板实例完全跳过 LoRA 计算，但仍可服务基础权重全量梯度场景。

文件中包含若干 NaN/Inf 统计与调试 helper；部分打印函数入口直接 `return`，说明它们是保留诊断设施，不属于正常运行输出。

## 7. C++ 配置、TP、绑定和异步调度文件

### 7.1 `operators/common.hpp`

作用：跨后端配置结构和接口定义。AMX 相关的三个核心结构是：

- `QuantConfig`：量化方法、bits、group size、zero-point、per-channel 标志；
- `GeneralMOEConfig`：`E/topk/H/F`、WorkerPool、GPU expert mask、三组权重/scale/zero 裸指针、TP 预分区二维指针表、反向权重表、cache/加载选项、激活参数；
- `MOESFTConfig`：在通用配置上增加 LoRA rank/alpha、六个 LoRA 权重指针、是否计算 full weight gradient、三个基础权重梯度指针。

`should_skip_expert()` 统一处理越界和 GPU expert；`compute_num_gpu_experts()` 从 mask 计算 offload 数。这里的 pointer 字段只是地址，不拥有内存。

### 7.2 `operators/moe-tp.hpp`

作用：推理 MoE 的 NUMA/TP 外壳。

`MOE_TP_PART` concept 规定子实例必须有输入/输出类型、构造函数和 `forward()`。`TP_MOE_Common<T>`：

- 从 `WorkerPoolConfig.subpool_count` 得到 TP 数；
- 默认把 `intermediate_size` 等分给各 TP；
- 在对应 NUMA 节点创建子实例；
- 为每个分片分配本地输出；
- 并行调用 `tps[numa_id]->forward()`；
- 通过具体特化的 `merge_results()` 合并局部结果；
- 维护 `weights_loaded`，防止未加载权重就 forward。

它按值把 `GeneralMOEConfig` 切成多个 `tp_config`，因此新加配置字段时必须确认复制和各格式特化是否仍保留正确语义。

### 7.3 `operators/moe-sft-tp.hpp`

作用：SFT 的 TP/NUMA 外壳，比推理版本多负责缓存、梯度和参数更新。

`TP_MOE_SFT`/相关特化完成：

- 为每个 NUMA 分片创建 `AMX_SFT_MOE_TP`；
- 分发前向并合并局部基础/LoRA 输出；
- 为反向临时数据使用共享 pool；
- 分发 `grad_output`，合并 `grad_input` 与 router gradient；
- 保持完整 `F` stride，避免 TP 局部中间维破坏 Python 端全量梯度布局；
- full-weight-grad 模式下只清零一次全局梯度，再由各分片写自己的切片；
- 显式下发 `set_full_weight_grad`，补偿 `GeneralMOEConfig` 切片可能丢失 SFT 派生字段的问题；
- 分区并更新基础权重、LoRA pointer、backward weight；
- 暴露异步 repack、准备/保存 backward 权重等训练维护动作。

### 7.4 `ext_bindings.cpp`

作用：整个 Python/C++ 边界，使用 pybind11 暴露配置、WorkerPool、CPUInfer、MoE 类和任务接口。

AMX 相关内容分为四组：

1. include 与条件编译：只有 x86 且 `USE_AMX_AVX_KERNEL` 构建才包含 AMX MoE 文件；
2. 配置绑定：`MOEConfig`/`MOESFTConfig` 的标量和 pointer 字段；pointer setter 把 Python 整数转换为 `void*`；
3. `MOEBindings`、`MOESFTBindings`：把 `load_weights`、`forward`、`backward` 等成员函数及参数封装为 CPUInfer 可执行的 `(func, args)`；
4. 类注册：`bind_moe_module()` 与 `bind_moe_sft_module()` 把具体 C++ 模板实例注册成 Python 类。

推理绑定包括：

```text
AMXInt8_MOE                 AMXInt4_MOE
AMXInt4_1_MOE               AMXInt4_1KGroup_MOE
AMXInt4_KGroup_MOE          AMXBF16_MOE
AMXFP8_MOE                  AMXFP8PerChannel_MOE
AMXFP4_KGroup_MOE           AMXMXFP8_KGroup_MOE
```

SFT 绑定包括 BF16、INT8、INT4 及各自 `SkipLoRA`；INT4_1 和 K-group SFT 注册代码目前被注释。

pybind 层不对每个 `data_ptr()` 的 dtype、shape、device 和容量做完整验证。Python wrapper 的校验和 buffer 规划必须与 C++ 假定严格一致，否则错误常表现为越界写、静默数值错误或异步阶段崩溃。

### 7.5 `cpu_backend/cpuinfer.h`

作用：Python 任务进入 C++ 后的执行门面。

- 构造时创建 `WorkerPool` 和 `TaskQueue`；
- `submit(pair)` 取出函数地址和参数地址，把当前 `CPUInfer*` 注入参数对象后调用任务 trampoline；
- `submit_with_cuda_stream()` 用 GPU runtime 的 host callback，让 CPU 任务与 CUDA/MUSA/ROCm/MACA stream 排序；
- `sync()` 等待 TaskQueue，`allow_n_pending` 可允许流水线中保留少量未完成任务；
- 初始化 ggml FP16 到 FP32 查表。

任务参数通常由 binding 在堆上创建，并由 trampoline/队列侧管理；调用者不应自行解释这对整数。

### 7.6 `cpu_backend/worker_pool.h` 与 `cpu_backend/worker_pool.cpp`

作用：固定线程、NUMA 子池和任务并行执行。

`worker_pool.h` 声明 `WorkerPoolConfig`、线程状态、`InNumaPool`、`NumaBackend` 和 `WorkerPool` API；配置包含总线程数、子池数和线程映射。`worker_pool.cpp` 实现工作线程创建/退出、hwloc CPU 亲和性、NUMA memory policy、原子 task index 和 work stealing，并让调用线程也作为编号 0 的 worker 参与执行。TP 实例通过子 backend 把任务送到本地 NUMA 节点。

AMX kernel 在这些工作线程上执行，因此 tile 权限初始化、权重内存 placement 和线程 pinning 都在此层与性能发生联系。

### 7.7 `cpu_backend/task_queue.h` 与 `cpu_backend/task_queue.cpp`

作用：CPUInfer 的异步任务队列和同步语义。

`task_queue.h` 声明单消费者 linked-node 队列、`enqueue()`、`sync()`、pending 计数和后台 worker。`task_queue.cpp` 创建 dummy head 和专用 worker thread；生产者用原子 `tail.exchange()` 追加节点，消费者执行 `std::function` 后递减 pending 并通知 condition variable。`sync(allow_n_pending)` 等待 pending 不超过阈值，析构时通知、join 并释放余下节点。

Python 的 `submit_*`/`sync_*` 之所以能流水化，靠的是这里而不是 pybind 自身。

### 7.8 `cpu_backend/shared_mem_buffer.h` 与 `cpu_backend/shared_mem_buffer.cpp`

作用：按对象和内存请求统一分配、复用大块共享/NUMA 感知临时内存。

`shared_mem_buffer.h` 声明 `MemoryRequest`：一组“大小 + 把子块地址写回哪个成员指针”的 callback；还声明普通和按 NUMA id 分组的 `SharedMemBuffer`。`shared_mem_buffer.cpp` 汇总一个请求内的大小，以 64-byte `posix_memalign` 创建 backing buffer；若后来出现更大的请求，会扩容并重新回填此前登记对象的指针。NUMA 版本按节点持有独立 buffer，并检查调用线程当前节点。

`AMX_MOE_BASE` 和 TP wrapper 用它分配局部输入输出、A/C backing pool 等。这里的多个对象会复用可容纳最大请求的共享 backing buffer，并非每个对象永久独占一块。连续、对齐和可复用的内存既减少 allocator 开销，也让 tile/AVX load 更稳定。修改最大 `qlen`、TP 数或 buffer 类型后，要重新核算这些请求的大小与并发复用关系。

## 8. Python 推理接口逐文件解释

### 8.1 `python/__init__.py`

作用：包入口和二进制扩展安装点。

导入包时先调用 `_cpu_detect.initialize()`，得到 extension module 与 `__cpu_variant__`，然后把 extension 同时放进：

```python
sys.modules["kt_kernel_ext"]
sys.modules[f"{__name__}.kt_kernel_ext"]
```

这样历史代码的 `import kt_kernel_ext` 和包内相对导入都指向同一个动态库。之后导出 `KTMoEWrapper`、GPU expert mask helper；`AMXSFTMoEWrapper` 使用 `__getattr__` 延迟导入，避免普通推理导入时强制加载训练依赖。

### 8.2 `python/_cpu_detect.py`

作用：运行时 CPU ISA 探测和多变体扩展选择。

主要流程：

- 读取 Linux CPU flag/平台信息，判断 AMX、AVX-512、AVX2 能力；
- 接受 `KT_KERNEL_CPU_VARIANT=amx|avx512|avx2` 强制覆盖；
- 枚举包内 `_kt_kernel_ext_amx...so` 等候选动态库；
- 按能力和兼容顺序尝试加载，失败时给出下一变体或清晰错误；
- 返回已加载 module 和实际 variant。

这里做的是动态库级选择。即便 variant 是 `amx`，某个 kernel 仍可能在 C++ 内按形状走 AVX-512；反过来，加载 `avx512` 变体不会包含真正的 AMX tile 指令。

### 8.3 `python/experts.py`

作用：用户可见的统一 MoE 工厂 `KTMoEWrapper`。

它完成：

- 校验 `mode="inference"` 或 `mode="sft"` 与 method 字符串；
- 推理时把 `AMXINT4/AMXINT8` 交给 `AMXMoEWrapper`；
- 把 `RAWINT4/FP8/BF16/FP8_PERCHANNEL/GPTQ_INT4/MXFP4/MXFP8` 交给 `NativeMoEWrapper`，由后者二次选 AMX/AVX2；
- 其他 method 分给 llamafile 或 moe_kernel 后端；
- SFT 时统一创建 `AMXSFTMoEWrapper`；
- 只对 MXFP4/MXFP8 传递 `swiglu_limit`，避免其他格式意外应用截断；
- 转发基础 wrapper 的静态资源清理/缓存接口。

这是理解“用户给一个 method 后到底去了哪里”的第一站，但不是后端最终判定点。

### 8.4 `python/experts_base.py`

作用：推理 wrapper 的通用资源、buffer 和提交逻辑。

关键职责：

- 维护共享 `CPUInfer`、`WorkerPool`，避免每层重复创建大量线程；
- 根据最大 prefill 长度创建 pinned CPU 输入、输出、expert id、routing weight ring buffer；
- 生成和持有 GPU expert mask、physical-to-logical 映射；
- 将 GPU 输入/路由信息异步复制到 pinned CPU buffer；
- 调用 `self.moe.forward_task(...)`，把各 Tensor 的 `data_ptr()` 传给 C++；
- 用 `submit_with_cuda_stream()` 把 host callback 插到 GPU stream，或直接 `CPUInfer.submit()`；
- `sync` 后把 CPU 结果复制/累加回 GPU；
- 管理 chunked prefill 和多 buffer 槽，防止流水线尚未结束时提前复用内存。

真正跨语言的关键调用可简化成：

```python
task = self.moe.forward_task(
    qlen_tensor.data_ptr(), topk,
    expert_ids_cpu.data_ptr(), routing_weights_cpu.data_ptr(),
    input_cpu.data_ptr(), output_cpu.data_ptr(), incremental,
)
self.cpu_infer.submit(task)
```

所有地址都没有 Python 引用计数保护，wrapper 保留 Tensor 实例才保证任务执行期间地址有效。

### 8.5 `python/utils/amx.py`

作用：推理 AMX/native 后端的具体 Python 实现，是 Python 调用 AMX 最直接的文件。

文件开头通过 `getattr(kt_kernel_ext.moe, name, None)` 获取所有可能的 pybind 类。使用 `getattr` 是为了让同一 Python 包可搭配不同 ISA 构建；缺失的类在选择时转为可读错误，而不是 import 阶段崩溃。

`AMXMoEWrapper`：

- 服务 `AMXINT4`、`AMXINT8`；
- 用 `SafeTensorLoader` 读取浮点 gate/up/down；
- 创建 `MOEConfig`，填写形状、pool、最大长度、GPU expert mask 和权重指针；
- 实例化 `AMXInt4_MOE` 或 `AMXInt8_MOE`；
- 提交 `load_weights_task(physical_to_logical_map_ptr)`，让 C++ 在线量化/pack；
- 支持预先按 NUMA 切好的权重二维指针表。

`NativeMoEWrapper`：

- 服务 `RAWINT4`、`FP8`、`FP8_PERCHANNEL`、`BF16`、`GPTQ_INT4`、`MXFP4`、`MXFP8`；
- `_create_loader()` 为每种格式创建正确的 loader；
- 为 RAWINT4 根据 `KT_RAWINT4_BACKEND`、编译类和 CPU flag 选择 AMX、AVX-VNNI-256 或 AVX2；
- 为 MXFP4/MXFP8 根据 `KT_MXFP4_BACKEND`/`KT_MXFP8_BACKEND` 做 AMX/AVX2 选择；
- 为 BF16/FP8 在 AMX/AVX-512 类不可用时选择 AVX2；
- 整理每个 expert 的 weight/scale/zero 指针数组，构造 `MOEConfig` 后提交 load；
- 在最后一层加载完成后释放 mmap loader，降低文件句柄和虚拟地址占用；
- 暴露把内部 weight scale 写回用户 buffer 的辅助接口。

注意 `AMXFP4_KGroup_MOE` 这类名字在编译为 AVX-512 umbrella、但未启用 `HAVE_AMX` 时仍可能存在；此时类内部依赖 `AMX_AVAILABLE` 走 AVX-512 路径。名字表达的是实现家族，不是一次调用的硬件指令证明。

### 8.6 `python/utils/loader.py`

作用：把不同检查点格式规范成 C++ MoE 期望的权重、scale、zero 和布局。

相关 loader 包括：

- `SafeTensorLoader`：普通浮点权重，供在线 AMX INT4/8 量化；
- `BF16SafeTensorLoader`：原生 BF16；
- FP8 loader：读取 byte 权重及 block scale；
- FP8 per-channel loader：处理每输出通道 scale；
- `CompressedSafeTensorLoader`：RAWINT4/K2 压缩格式；
- GPTQ loader：qweight/qzeros/scales 的 GPTQ 约定；
- `MXFP4SafeTensorLoader`：E2M1 nibble 与 UE8M0 scale；
- `MXFP8SafeTensorLoader`：E4M3fn byte 与 UE8M0 scale；
- `GGUFLoader`：兼容以 GGUF 保存的专家权重。

loader 还要处理模型间不同的键名、gate/up 融合或分离、expert 维度、转置方向和 scale shape。它们返回的不只是数值，还是 C++ 解释裸地址时所依赖的布局协议。

### 8.7 `python/utils/__init__.py`

作用：集中重导出 `AMXMoEWrapper`、`NativeMoEWrapper` 和常用 loader。它不增加执行逻辑，但决定外部代码可以从 `kt_kernel.utils` 稳定导入哪些组件。

### 8.8 同目录中不属于 AMX 主链的文件

- `python/utils/llamafile.py`：llamafile MoE wrapper，接口形状与 AMX wrapper 相似，但 C++ backend 不同；
- `python/utils/moe_kernel.py`：项目自有非 AMX `moe_kernel` wrapper。

它们会被 `experts.py` 的其他 method 选中，因此排查工厂分派时有参考价值；但不会调用本文的 `operators/amx/` kernel。

## 9. Python SFT 接口逐文件解释

### 9.1 `python/sft/__init__.py`

作用：SFT 子包的公共导出入口。对外提供配置、模型包装、LoRA/权重工具以及 AMX SFT wrapper，并尽量把可选依赖的影响限制在训练路径。

### 9.2 `python/sft/config.py`

作用：训练配置和环境变量到 KT-Kernel 参数的映射。

它定义/整理 method、LoRA rank/alpha、CPU 线程、NUMA 子池、最大 token、full/hybrid 模式等。还会根据物理 core 设置 OMP/线程环境；full 或 hybrid 训练会映射到 `full_weight_grad`，决定 C++ 是否为三个基础投影计算 `dW`。

### 9.3 `python/sft/arch.py`

作用：模型架构适配。

它识别 Qwen、DeepSeek、GLM 等模型中 MoE 层、router、expert 列表和投影命名，返回统一的架构描述；还负责定位应留在 GPU 的非专家模块，以及给出不支持架构时的专用异常。这个文件不提交 AMX 任务，但决定交给 AMX 的权重语义和 forward 接口是否正确。

### 9.4 `python/sft/wrapper.py`

作用：模型级 SFT 改造和加载编排。

主要过程：

1. 解析配置并识别模型架构；
2. 从现有模型或 checkpoint 提取专家基础权重；
3. 对每个 MoE 层创建 `KTMoEWrapper(mode="sft")`；
4. 调用 AMX SFT wrapper 加载/pack 权重；
5. full-FT 时创建连续 BF16 基础 `Parameter`；
6. 用 `KTMoELayerWrapper` 替换 Hugging Face 原 MoE module；
7. 在安全时释放原 expert 参数的 storage，避免 CPU/GPU 双份占用；
8. 暴露 `load_model`、插件入口和 checkpoint 辅助。

分布式模式通常由 rank 0 持有 CPU AMX backend，其他 rank 通过 gather/scatter 协作。

### 9.5 `python/sft/weights.py`

作用：专家权重提取、检查点读取和旧参数清理。

它兼容“每个 expert 一个 module”和“多个 expert 融合 tensor”两种结构，把 gate/up/down 统一到 C++ 需要的 expert-major 形状；必要时做反量化/类型转换。全量训练还用它创建/维护连续基础权重容器。清理旧权重必须发生在 AMX load 已复制完成，或 C++ 已明确持有新的 BF16 source pointer 之后。

### 9.6 `python/sft/layer.py`

作用：替代原模型 MoE 层的 `torch.nn.Module`。

前向中它：

- 调用原 router 或适配后的 router 得到 logits/top-k；
- 协调 GPU experts 与 CPU AMX experts 的重叠计算；
- 检查基础权重是否被优化器修改，必要时触发重新量化；
- 调用 `KTMoEFunction.apply` 把 autograd 边界交给自定义 Function；
- 将 CPU/GPU 专家结果合并为模型期望的输出。

这个文件把 PyTorch module 语义转换为 AMX wrapper 的低层 buffer 语义。

### 9.7 `python/sft/autograd.py`

作用：PyTorch 自定义 autograd 桥。

`KTMoEFunction.forward()` 提交/同步 AMX SFT 前向并保存 backward 所需的 wrapper/buffer 标识。分布式时处理变长 `qlen` gather 到 rank 0 和结果 scatter。

`backward()` 调用 C++ backward，并返回：

- 输入梯度；
- router/routing weight 相关梯度；
- LoRA 参数梯度对应的 autograd 返回值或由连续 buffer 承载的梯度；
- full-FT 模式下由 C++ 直接写入三个基础权重 `.grad` backing buffer 的结果。

它还处理 gradient checkpointing：重算 forward 时要避免重复破坏缓存或错误提交可保存状态。

### 9.8 `python/sft/base.py`

作用：SFT wrapper 抽象基类和 pinned buffer 生命周期管理。

`KExpertsSFTBuffer` 是 grow-only 的 CPU pinned buffer 集合，容纳当前 batch 的：

- `qlen`、输入/输出；
- expert ids、routing weights；
- grad output/input、router gradient；
- LoRA 中间值和梯度地址；
- forward cache 关联状态。

`BaseSFTMoEWrapper` 定义 `_make_forward_task()`、`_make_backward_task()` 模板方法，并实现同步/异步提交、CPU/GPU copy、ring slot 复用、full weight Parameter/grad 管理。子类 `AMXSFTMoEWrapper` 只需把这些 buffer 地址翻译成具体 pybind task。

### 9.9 `python/sft/amx.py`

作用：SFT Python 层到 AMX pybind 类的直接适配器。

文件开头的 `_SFT_METHOD_TO_CLASS` 映射 BF16/INT8/INT4 及 `SkipLoRA` 到相应 extension 类。构造时若类为 `None`，说明当前编译变体没有该 SFT backend。

核心接口：

- `_make_forward_task()`：把 `qlen` pointer、top-k、expert ids、routing weights、input/output 和 `save_for_backward` 传给 `forward_sft_task()`；
- `_make_backward_task()`：传入 grad output/input、六组 LoRA grad、router grad，以及可选的三个 full base grad pointer；缺失项用 `0`；
- `load_weights()`：构造 `MOESFTConfig`，设置 pool、形状、基础权重、scale/zero、LoRA 和 backward 权重地址，实例化 C++ 类并提交 load；
- `update_lora_weights()`：更新六个零拷贝 LoRA pointer；
- `update_base_weights()`：把 Python BF16 参数地址下发给 `set_base_weight_pointers()`，再提交 `load_weights_task()` 重新量化/pack；
- 准备、保存、加载和 repack backward weight；
- 校验 NUMA 预分区 tensor 的个数、device、连续性和 scale 长度。

### 9.10 `python/sft/lora.py`

作用：LoRA 参数组织、PEFT 接入和梯度同步。

它为六组专家 LoRA 权重建立连续 CPU view 和 grad buffer，收集应交给 optimizer 的参数，跟踪 pointer/版本是否变化，并在变化后通知 AMX SFT wrapper。还负责分布式梯度同步、checkpoint 保存加载，以及 full-FT 基础参数与 LoRA 参数的统一参数列表。

### 9.11 `python/sft/dist_utils.py`

作用：分布式和 checkpointing 辅助。

包括变长 token 在各 rank 间 gather/scatter、rank 0 AMX 计算协调、checkpoint 模式判断、ZeRO-3 参数 gather context 等。它不含 AMX intrinsic，但决定只有某一 rank 执行 CPU task 时数据和梯度如何保持一致。

## 10. 构建和运行时开关逐文件解释

### 10.1 `CMakeLists.txt`

AMX 相关的主要选项：

```text
KTRANSFORMERS_CPU_USE_AMX_AVX512
KTRANSFORMERS_CPU_USE_AMX
```

前者启用 AMX/AVX-512 实现家族并定义 `USE_AMX_AVX_KERNEL`；后者进一步定义 `HAVE_AMX`，添加 `-mamx-tile -mamx-bf16 -mamx-int8`。也就是说，只打开 umbrella 可能编译出同名类，但 `AMX_AVAILABLE=false`，内部走 AVX-512。

CMake 还为底层 AMX/LoRA 测试创建可选 executable，并统一添加 AVX512F/BW/VL/BF16/VBMI 等所需 flag。

### 10.2 `setup.py`

作用：Python package 的 CMake 构建驱动和多 CPU 变体打包。

它探测本机能力，把：

```text
CPUINFER_ENABLE_AMX    -> KTRANSFORMERS_CPU_USE_AMX
CPUINFER_ENABLE_AVX512 -> KTRANSFORMERS_CPU_USE_AMX_AVX512
```

传给 CMake，并可一次生成 AMX、AVX-512、AVX2 等多个 extension 变体。运行时再由 `_cpu_detect.py` 选择。因此“构建机支持 AMX”和“目标机实际加载 AMX 变体”是两个独立阶段。

### 10.3 `cmake/DetectCPU.cmake`

作用：从系统 CPU 特征检测 AMX/AVX 能力，并在用户未显式设置时初始化 CMake 选项。交叉编译、容器屏蔽 flag 或构建机与部署机不同的场景中，不应完全依赖自动结果。

### 10.4 `cmake/FindSIMD.cmake`

作用：通过小型 C 程序探测 AVX、AVX2、FMA、AVX-512 编译和运行支持，主要服务通用 SIMD 配置和 MSVC flag。它不单独探测 AMX tile，但会影响 AMX family 所依赖的 AVX 回退能力。

### 10.5 `CMakePresets.json`

作用：给不同构建目标提供 AMX、AVX-512、AVX2 选项组合。阅读 preset 时应同时看两个 AMX 选项，不能只看到 `KTRANSFORMERS_CPU_USE_AMX_AVX512=ON` 就判断包含 tile 指令。

### 10.6 `pyproject.toml`

作用：Python 构建系统、包元数据和可传递环境选项的声明。它本身不编译 kernel，实际编译逻辑仍在 `setup.py`/CMake。

### 10.7 `install.sh`

作用：安装前的 CPU 探测、环境变量设置和构建入口。脚本在自动模式下设置 `CPUINFER_ENABLE_AMX`，手动模式验证用户组合。部署到不同 CPU 时应关注它记录的选择是否与运行主机一致。

## 11. 测试与示例逐文件索引

### 11.1 `operators/amx/test/` 底层测试

这些文件多数是独立 executable 或诊断程序，不进入 Python extension 的生产调用链。

| 文件 | 作用 |
| --- | --- |
| `operators/amx/test/amx-test.cpp` | 综合 AMX kernel harness：使能 tile、构造多种量化矩阵、比较参考结果并测 latency/throughput。适合验证机器和编译器的 AMX 基础能力。 |
| `operators/amx/test/amx-bkgroup-test.cpp` | 单独检查 K-group B buffer 的 pack、地址计算和 kernel 消费结果。 |
| `operators/amx/test/amx-kgroup-test.cpp` | 检查 K-group A/B buffer、量化正确性和整条 K-group GEMM。 |
| `operators/amx/test/amx-c-reduce-test.cpp` | 验证 `BufferCReduce` 的创建、局部结果归约、精度和性能。 |
| `operators/amx/test/analyze-error.cpp` | 对 K-group 结果误差按 block/位置展开，帮助区分量化误差与布局错误。 |
| `operators/amx/test/debug-kgroup.cpp` | K-group 路径的简化调试入口，输出中间 buffer/结果。 |
| `operators/amx/test/debug-kgroup-details.cpp` | 更细粒度查看 scale、sum、zero、block offset 和局部 dot-product。 |
| `operators/amx/test/debug-specific-dims.cpp` | 针对曾出问题的特定 `M/N/K` 形状输出内部状态。 |
| `operators/amx/test/test-specific-dims.cpp` | 将特定非整块维度作为回归用例，检查 padding 和尾块处理。 |
| `operators/amx/test/test-kgroup-128.cpp` | group size 128 的专项正确性/边界测试。 |
| `operators/amx/test/test-kgroup-kernel.cpp` | 绕过高层 MoE，直接测试 K-group 微内核。 |
| `operators/amx/test/verify-kgroup.cpp` | 以参考反量化/GEMM 系统验证 K-group 数值误差。 |
| `operators/amx/test/mat-test.hpp` | 测试共用矩阵容器、随机数据、参考 GEMM、结果比较和 benchmark helper。 |
| `operators/amx/test/timer.hh` | 测试共用计时与性能输出工具。 |
| `operators/amx/test/avx-test.cpp` | 大容量 AVX-512 内存带宽/向量访问基准，用于判断瓶颈是否在内存子系统；不是 AMX tile 测试。 |
| `operators/amx/test/mmq.h` | 独立 ggml 风格 AMX quantized matmul 的声明与数据结构。 |
| `operators/amx/test/mmq.cpp` | 上述独立 MMQ 实现，更多是原型/对照路径，不是当前 CRTP MoE 主 kernel。 |
| `operators/amx/test/mmq-test.cpp` | MMQ 的精度与性能测试。 |
| `operators/amx/test/test_lora_kernel.cpp` | 验证 `avx_kernels.hpp` 中基础 LoRA 前向/反向小矩阵 kernel。 |
| `operators/amx/test/test_lora_fused_add.cpp` | 验证 LoRA B 结果融合加到基础输出的正确性和性能，覆盖不同 shape/stride。 |
| `operators/amx/test/test_lora_fused_add_wt.cpp` | 验证使用转置权重布局的 fused-add 版本。 |
| `operators/amx/test/test_repack.cpp` | 检查前向 BufferB 权重还原为 BF16、转置、再 pack 成反向 BufferB 的过程。SFT backward 权重错误时优先看此测试。 |
| `operators/amx/test/thread_test.sh` | 以不同线程数重复运行 benchmark，观察扩展性、NUMA 和带宽饱和点。 |

### 11.2 `examples/` Python 示例

| 文件 | 作用 |
| --- | --- |
| `examples/test_moe_amx.py` | 最通用的 AMXINT8/AMXINT4 Python 正确性和性能示例，覆盖 wrapper、load task 和 forward task。 |
| `examples/bench_moe_amx_int8.py` | 可配置的 INT8 MoE benchmark，适合测线程数、token 数、专家形状。 |
| `examples/test_awq_moe_amx.py` | 非对称 AWQ INT4_1/K-group 权重、scale、zero 的加载与结果检查。 |
| `examples/test_k2_moe_amx.py` | RAWINT4/K2 对称小 K-group 示例。 |
| `examples/test_bf16_moe.py` | 原生 BF16 backend 的精度/性能基线。 |
| `examples/test_fp8_moe.py` | blockwise FP8 权重与 scale 加载、forward 对照。 |
| `examples/test_fp8_perchannel_moe.py` | per-channel FP8 路径专项测试。 |
| `examples/test_fp4_moe_amx.py` | 合成 MXFP4 权重的 AMX/AVX-512 family 测试。 |
| `examples/test_fp4_moe_v4.py` | 面向真实 DeepSeek V4 类 checkpoint，覆盖 MXFP4 与 `swiglu_limit`。 |
| `examples/test_mxfp8_moe_m3.py` | 面向 MiniMax M3 checkpoint，覆盖 MXFP8 与 `swigluoai` 激活。 |
| `examples/test_fp4_moe_avx2.py` | MXFP4 的 AVX2 对照/回退示例，可用于和 AMX family 比较。 |
| `examples/test_mxfp8_moe_avx2.py` | MXFP8 的 AVX2 对照/回退示例。 |

### 11.3 `test/per_commit/` 回归测试

| 文件 | 作用 |
| --- | --- |
| `test/per_commit/test_moe_amx_accuracy_int8.py` | AMX INT8 数值回归。 |
| `test/per_commit/test_moe_amx_accuracy_int4.py` | 对称 AMX INT4 数值回归。 |
| `test/per_commit/test_moe_amx_accuracy_int4_1.py` | 非对称 INT4_1 数值回归。 |
| `test/per_commit/test_moe_amx_accuracy_int4_1k.py` | INT4_1 K-group 数值回归。 |
| `test/per_commit/test_moe_amx_bench_int8.py` | INT8 性能回归。 |
| `test/per_commit/test_moe_amx_bench_int4.py` | INT4 性能回归。 |
| `test/per_commit/test_moe_amx_bench_int4_1.py` | INT4_1 性能回归。 |
| `test/per_commit/test_moe_amx_bench_int4_1k.py` | INT4_1 K-group 性能回归。 |
| `test/per_commit/test_moe_rawint4_accuracy.py` | `NativeMoEWrapper` RAWINT4 后端选择与数值回归；在 AMX 机器上可覆盖 K2 AMX 路径。 |
| `test/per_commit/test_moe_gptq_int4_accuracy.py` | GPTQ INT4 loader/后端对照；当前主执行类来自 AVX2/VNNI，不是 AMX kernel。 |
| `test/per_commit/test_moe_avx2_accuracy_bf16.py` | BF16 AVX2 fallback 对照。 |
| `test/per_commit/test_moe_avx2_accuracy_fp8.py` | FP8 AVX2 fallback 对照。 |

## 12. 几个关键实现机制

### 12.1 2-2-4 tile 排布为什么常见

AMX 有 8 个 tile 寄存器。`GemmKernel224*` 把它们分为：

```text
TMM0, TMM1        两块 A
TMM2, TMM3        两块 B
TMM4..TMM7        四块 C = A(2) × B(2)
```

一次加载两块 A 和两块 B，可以形成四个独立 C 累加块，恰好用满 8 个 tile。K 维在循环中推进，C tile 最后 store，再结合 scale/zero 做 FP32 修正。1-3-3 是另一种寄存器分配，适用于不同 N 分块和复用关系。

### 12.2 为什么先 pack 权重

原始 PyTorch 权重通常是 `[E, out, in]` 的连续 tensor，但 tile 指令希望在固定 K block 内直接读取适当交错的 B 行/列。若 forward 每次现场转置、解 nibble、重排，内存访问开销会吞掉 AMX 的算力。

因此 load 阶段把每个专家权重转换成 `BufferB`：

```text
checkpoint layout
  -> loader 统一键名/shape
  -> C++ TP 切片
  -> quantize（若需要）
  -> transpose/interleave/pad
  -> 64-byte aligned BufferB
  -> forward 只做规则 block load
```

代价是权重更新后必须 repack；这也是 full-FT 中 `update_base_weights()` 不能省略的原因。

### 12.3 prefill 与 decode 的不同

prefill 中，一个专家经常积累多个 token，`M` 足够大，A pack 和 tile 配置能被多次 dot-product 摊薄。decode 中每个专家可能只有一个 token，矩阵-向量内存访问占主导，AVX-512 `vec_mul` 有时比强行填满 AMX tile 更合适。

所以性能判断应至少按以下维度分组：`qlen`、命中专家数、每专家 token 数、`H/F`、量化格式和 NUMA 数。只报告一个总 tokens/s 很难定位 kernel 问题。

### 12.4 TP/NUMA 如何切权重

一般情况下每个 TP 子实例获得 `F / tp_count`：

- gate/up 的输出维按 `F` 切；
- down 的输入维按 `F` 切；
- 每个子实例产生对 `H` 输出的局部贡献；
- 外层把这些局部结果相加，再按 routing weight 聚合。

权重、scale、zero 必须用相同边界切分。group 量化还要求切分点满足 group/block 对齐。预分区二维指针表的第一维通常对应 TP/NUMA，第二维对应 expert。

### 12.5 Python 裸指针协议

Python 把 `Tensor.data_ptr()` 转成整数；pybind 再转为 `void*`。这条路径没有 DLPack/pybind Tensor 对象为其自动维持生命周期。必须同时满足：

- Tensor 在任务完成前仍存活；
- Tensor 未被 resize、迁移 device 或替换 storage；
- dtype 与 C++ cast 一致，例如 expert id 是 `int64_t`、routing weight 是 `float`、AMX 输入通常是 BF16；
- tensor 连续、容量至少覆盖 C++ 根据 `qlen/H/F/topk` 计算的范围；
- CPU 任务不能直接解引用 CUDA pointer，GPU 数据必须先进入 pinned CPU buffer；
- mmap loader 不能在 C++ 完成权重复制以前关闭；
- 异步 ring buffer 槽不能在上一次任务结束前复用。

这也是 Python wrapper 看似有很多 buffer/cache bookkeeping 的根本原因。

### 12.6 前向权重、反向权重和训练源权重

SFT/full-FT 中同一投影可能同时有三种表示：

| 表示 | 用途 |
| --- | --- |
| Python BF16 `Parameter` | optimizer 的源权重，用户可保存和更新 |
| forward packed/quantized `BufferB` | AMX 前向快速读取 |
| transposed backward packed `BufferB` | 计算 `grad_input` 等反向矩阵乘 |

optimizer 更新的是第一种，后两种不会自动变化。训练编排必须在正确时机更新 pointer、重新量化 forward buffer，并重新准备或 repack backward buffer。

## 13. 易错点和排障建议

### 13.1 先确认“实际后端”，不要只看类名

按以下顺序确认：

1. `kt_kernel.__cpu_variant__` 是什么；
2. 构建是否定义 `USE_AMX_AVX_KERNEL` 和 `HAVE_AMX`；
3. Python `NativeMoEWrapper` 最终选择了哪个 pybind 类；
4. 对应 C++ kernel 的 `AMX_AVAILABLE` 是否为 true；
5. 当前形状调用的是 `mat_mul` 还是 AVX `vec_mul`。

### 13.2 AMX 权限失败

若编译和 CPU flag 都正确但 tile 指令仍失败，检查：

- 内核是否支持 AMX xstate；
- `arch_prctl` 请求是否成功；
- 执行 kernel 的 WorkerPool 线程是否调用过 `enable_amx()`；
- 虚拟机/容器是否暴露 AMX flag 和 xstate 权限。

### 13.3 数值错位但不崩溃

优先检查布局协议：

- gate/up/down 是否转置方向一致；
- TP slice 是否与 scale/zero slice 同步；
- group size 是否与 kernel 模板一致；
- FP8 是 blockwise 还是 per-channel；
- MXFP4 nibble 高低半字节顺序；
- `ld`、padding、尾块尺寸；
- routing expert id 是否已经应用 physical/logical 映射。

这类问题通常先跑 `test-specific-dims.cpp`、K-group debug 测试或单格式 Python example，比从完整模型定位更快。

### 13.4 SFT 梯度异常

建议按下面顺序隔离：

1. BF16 base、`SkipLoRA`、单层、单 NUMA；
2. 只看 `grad_input`；
3. 打开 LoRA，分别比较 A/B 梯度；
4. 打开 router gradient；
5. 最后打开 `full_weight_grad` 和多 TP；
6. 基础权重更新后验证 forward/backward 两份 packed weight 都已刷新。

`test_lora_kernel.cpp`、两个 fused-add 测试和 `test_repack.cpp` 分别覆盖这条链的底层部件。

### 13.5 性能异常

除 kernel 本身外，应同时检查：

- WorkerPool 是否绑定到预期 core/NUMA；
- 权重是否在使用它的 NUMA 节点；
- `threadpool_count` 是否与 socket/NUMA 拓扑匹配；
- `F` 切分是否满足 block/group 对齐；
- decode 是否实际走了 mat-vec；
- load/repack 时间是否被误计入 forward；
- pinned H2D/D2H 和 GPU host callback 是否与 CPU 计算正确重叠；
- 多个层是否错误共享或争用临时 pool。

## 14. 推荐阅读顺序

若目标是快速建立全局认识：

1. `python/experts.py`
2. `python/utils/amx.py`
3. `ext_bindings.cpp` 中 `bind_moe_module` 和 AMX 类注册
4. `operators/moe-tp.hpp`
5. `operators/amx/moe_base.hpp`
6. 选择一种格式，例如 `bf16-moe.hpp` 或 `moe.hpp`
7. `operators/amx/la/amx_buffers.hpp`
8. `operators/amx/la/amx_kernels.hpp`
9. `operators/amx/la/amx_config.hpp`

若目标是 SFT/全量梯度：

1. `python/sft/wrapper.py`
2. `python/sft/layer.py`
3. `python/sft/autograd.py`
4. `python/sft/base.py`
5. `python/sft/amx.py`
6. `operators/moe-sft-tp.hpp`
7. `operators/amx/sft_moe.hpp`
8. `operators/amx/la/avx_kernels.hpp`
9. `operators/amx/test/test_repack.cpp`

若目标是新增一种量化格式，最好以 `bf16-moe.hpp` 或 `fp8-moe.hpp` 为骨架，同时成对设计 `BufferB` 和 `GemmKernel`，再补 loader、pybind 注册、Python method 分派和独立数值测试。只增加 kernel 而不定义完整布局/生命周期协议，无法接入 Python MoE 主链。

## 15. 总结

KT-Kernel 的 AMX 实现可以概括为“Python 负责编排和内存，C++ 负责异步/NUMA 调度，Buffer 负责布局，Kernel 负责 tile 计算”。

- 推理公共流程集中在 `moe_base.hpp`，各 `*-moe.hpp` 处理具体权重格式；
- SFT 公共 Python 流程集中在 `python/sft/`，核心 C++ 训练逻辑在 `sft_moe.hpp` 和 `moe-sft-tp.hpp`；
- 真正的 AMX tile 配置、buffer 和微内核位于 `operators/amx/la/`；
- `ext_bindings.cpp`、`CPUInfer` 和 WorkerPool 把 Python 裸指针任务安全地送到正确 NUMA 线程；
- 编译变体、实际类选择和单次 shape 的 AMX/AVX 分支必须分开判断；
- 权重 pack、异步 Tensor 生命周期和 TP/group 对齐与算术 kernel 同等重要。

# Qwen3.5-35B-A3B 的 APTMoE 替代测试方案

## 结论

可以使用 Qwen3-30B-A3B，但原模型只能作为 **APTMoE 通路基线**，不能直接作为
Qwen3.5-35B-A3B 的等价性能替代。推荐以 APTMoE 已支持的 Qwen3 代码路径为骨架，
构造一个不加载预训练权重的 **Qwen3.5 组件同构 deployment proxy**：

- GPU 使用真实 Qwen3.5 的 30 个 Gated DeltaNet 和 10 个 gated full-attention；
- CPU 部署 40×256 个独立 SwiGLU(2048, 512) routed experts；
- router、shared expert 和 shared-expert gate 随 stage 在 GPU 执行，只有 routed
  experts 允许留在 CPU；
- 所有参数参与 BF16 forward/backward/update，但权重随机初始化；
- 只比较部署、搬运、显存/内存和组件耗时，不比较 loss、收敛或 checkpoint。

KTransformers 与 DeepSpeed 仍使用当前 LLaMA-Factory 入口运行真实
`Qwen3_5MoeForCausalLM`。APTMoE proxy 不是 LLaMA-Factory 后端，结果必须单列。

## 当前测试要求

现有 sweep 的正式口径是：

- 文本模型：从多模态 checkpoint 提取 `text_config`，加载
  `Qwen3_5MoeForCausalLM`，排除 vision 和 MTP；
- 精度与训练：BF16、`finetuning_type=full`、真实 forward/backward/optimizer；
- 序列长度：server 为 32、64、128、256、512、1024、2048、4096；
  consumer 为 16、32、64、128、256、512、1024、2048；
- 每档 15 个 optimizer steps，前 5 步只作性能 warmup；
- server：8 GPU、global batch 8；consumer：2 GPU、global batch 2、1 TiB
  cgroup、无 swap、NUMA 0/1 interleave；
- 只记录无 `cuda.synchronize()` 的 host-wall forward/backward/optimizer/step
  边界时间，不启用资源采样器或内部 profiler。

当前 APTMoE 不能满足这个正式口径：

1. 它有自有模型、数据和 pipeline runtime，并不是 LLaMA-Factory 的 backend。
2. 底层 `PipelineRuntime` 在 `lora_mode=False` 时已经会将全部模型参数交给
   optimizer，具备全参数更新能力；但当前 YAML/SFT 入口只接受
   `finetuning_type: lora`，需要把这条已有能力接到 full SFT 入口。
3. Qwen3.5 虽能被识别，但会因 hybrid attention、attention output gate、fused
   experts 和 shared expert 未实现而主动终止。
4. 当前训练循环包含逐 micro-batch `torch.cuda.synchronize()` 和 barrier，与
   sweep 的无同步计时契约不同。
5. 当前 APTMoE 的 Qwen3-30 lookup table 对应 9 MiB expert，不能用于 Qwen3.5
   的 6 MiB expert。

因此 proxy 是替代的系统实验，不是“第三个等价训练后端”。

这里的 proxy full-FT 不是只按公式估算计算量，也不是 forward-only simulator。每个
计时 step 都必须真实执行：

```text
BF16 forward -> loss -> backward -> 全参数 optimizer.step()
```

随机初始化的 attention、router、shared expert 和 CPU routed expert 都要产生梯度并
发生数值更新。所谓“没有实际效果”仅表示它没有预训练知识、loss/收敛没有模型质量
意义、输出 checkpoint 也不能作为 Qwen3.5 使用；默认可以不保存这些无业务价值的
权重。若跳过 backward 或 optimizer update，则只能叫 inference/compute benchmark，
不能计为 BF16 全量微调 TPS。

## 精确参数对比

以下数字由本机两个 checkpoint 的 `config.json` 和 Transformers meta model
交叉验证得到，均只包含文本 CausalLM。

| 项目 | Qwen3.5-35B-A3B | Qwen3-30B-A3B | Qwen3 / Qwen3.5 |
|---|---:|---:|---:|
| 文本总参数 | 34,660,610,688 | 30,532,122,624 | 88.09% |
| routed expert 参数 | 32,212,254,720 | 28,991,029,248 | 90.00% |
| BF16 routed expert 权重 | 60 GiB | 54 GiB | 90.00% |
| 层数 | 40 | 48 | 120.00% |
| expert/层 | 256 | 128 | 50.00% |
| expert 总数 | 10,240 | 6,144 | 60.00% |
| 单 expert 形状 | SwiGLU(2048,512) | SwiGLU(2048,768) | 不同 |
| 单 expert BF16 大小 | 6 MiB | 9 MiB | 150.00% |
| top-k | 8 | 8 | 相同 |
| 每 token active expert 计算量 | 8×512 + shared 512 | 8×768 | Qwen3 高 33.33% |
| attention 参数 | 1,284,188,800 | 905,981,952 | 70.55% |
| attention 类型 | 30 linear + 10 full | 48 full GQA | 不同复杂度 |
| shared expert | 每层 512 + scalar gate | 无 | 缺失 |

Qwen3-30 的 expert 总存储只少 10%，看似接近；但它的单 expert 搬运粒度大 50%、
expert 个数少 40%，每 token CPU expert 计算反而高 33.33%。这三项都会改变 APTMoE
的热门 expert 选择、PCIe 搬运和 CPU/GPU 分工。

attention 的偏差更不能用一个缩放系数修正：Qwen3 的 48 层均为二次复杂度 GQA，
Qwen3.5 只有 10 层 full attention，另外 30 层是线性复杂度 Gated DeltaNet。
因此原样 Qwen3-30 只适合验证“能否运行”和排查 pipeline bug。

## 推荐的三级方案

### A. 原样 Qwen3-30：通路基线

无需修改 Qwen3 模型结构和权重加载器，可以继续使用 APTMoE 已有的 Qwen3-30
配置和 checkpoint。APTMoE runtime 已有全参数 optimizer 路径；若要求通过当前
YAML/SFT 数据入口运行 full update，仍须完成下文的入口、保存和计时接线。允许报告：

- APTMoE 能否完成 BF16 forward/backward/update；
- pipeline、offload、通信和保存是否工作；
- 标注为 `qwen3_30b_operational_baseline` 的独立数据。

不得把它放进 Qwen3.5 三后端 TPS 表，也不要按总参数比 34.66/30.53 对 TPS 做线性
换算。

### B. 最小 Qwen3-shaped proxy：只对齐 CPU expert

沿用 `Qwen3Stage`、router 和 APTMoE dispatch，修改：

```text
num_hidden_layers:       48  -> 40
num_experts:             128 -> 256
moe_intermediate_size:   768 -> 512
num_experts_per_tok:     8   -> 8
hidden_size:             2048（不变）
shared expert:           无  -> SwiGLU(2048, 512) + scalar sigmoid gate
```

这样 routed-expert 参数、单 expert 6 MiB 搬运粒度、expert 数量和 top-k 均精确
对齐。若 attention 仍使用 Qwen3 GQA，结果只能用于 CPU expert/offload 指标；
GPU attention 和端到端 TPS 必须标为不对齐。

### C. 组件同构 proxy：推荐

在 B 的基础上，直接复用已安装 Transformers 的：

- `Qwen3_5MoeGatedDeltaNet`：layer 0/1/2、4/5/6……；
- `Qwen3_5MoeAttention`：layer 3、7、11……39；
- `Qwen3_5MoeTextRotaryEmbedding`；
- `Qwen3_5MoeRMSNorm`。

参考实现见
[`qwen35_aptmoe_proxy_components.py`](qwen35_aptmoe_proxy_components.py)。
它在 meta device 上已经验证：

```text
linear layer token mixer:       33,718,464 params
full-attention token mixer:     27,263,488 params
routed experts/layer:          805,306,368 params
router/layer:                      524,288 params
shared expert + gate/layer:       3,147,776 params
```

这条路线无需处理 Qwen3.5 checkpoint shard、fused expert 键拆分、视觉塔、MTP 或
保存格式，改造量明显小于完整模型适配，同时 GPU token mixer 与 CPU experts 的
结构和参数完全一致。

本机 APTMoE 环境当前可以导入 Qwen3.5 类，但
`is_fast_path_available == False`，会退回 Transformers 的 PyTorch
linear-attention 实现。因此：

- 参数/显存和 CPU expert 测试现在即可使用；
- GPU attention 性能测试前必须安装与当前 PyTorch/CUDA 兼容的
  `flash-linear-attention` 和 `causal-conv1d`；
- 入口必须调用参考代码的 `require_linear_attention_fastpath()`，并在 manifest
  记录 Transformers、FLA、causal-conv1d、PyTorch、CUDA 和实际 SDPA backend；
- fast path 不可用时只能输出 `attention_performance_valid: false`，不得静默跑
  fallback 后参与性能比较。

## 已落地的适配结构

### 1. 不修改独立 APTMoE checkout

实现没有改动当前已有用户修改的 `/mnt/data2/wbw/APTMoE-baseline`。入口
[`aptmoe_qwen35_proxy_train.py`](aptmoe_qwen35_proxy_train.py) 将其加入
`PYTHONPATH`，复用以下原始能力：

- `PipelineRuntime` 的 pipeline P2P、AdamW 和 scheduler；
- `ModelShard` 的 stage load/drop 与 checkpoint recompute；
- `CommScheduler` 的独立 CUDA load/drop streams；
- `OffloadInputBegin/End` 的 CPU expert 可微输入/输出桥；
- APTMoE 的 SFT dataset/tokenizer。

因此适配代码可由 FFTtest 独立提交，APTMoE checkout 的 dirty worktree 不会被混入。

### 2. 参数精确的单层 stage

[`aptmoe_proxy/model.py`](aptmoe_proxy/model.py) 每个 pipeline stage 放一层，共
40 stages：

```python
token_mixer = Qwen35TokenMixer(text_config, layer_idx)
router = APTQwen35Router(text_config, layer_idx, route_replay)
experts = ModuleList([
    APTQwen35RoutedExpert(2048, 512, layer_id=layer_idx, expert_id=eid)
    for eid in range(256)
])
shared = Qwen35SharedExpert(2048, 512)
```

stage 0 额外包含 embedding，stage 39 包含 final norm、LM head 和交叉熵。
每个 routed expert 保留目标的 fused `gate_up_proj(2048→1024)` 加
`down_proj(512→2048)` 两 tensor 布局，而非拆成两个 512-wide GEMM。
`inter_stage_only=True` 是有意选择：每 stage 恰好一层，不需要 APTMoE 面向多层
stage 的 predictor queue。`FwdStageLoad` 按上一 microbatch 的实测 popularity
运行原 placement solver，当前 replay route 在 router 后动态补载缺失 hot experts，
避免用 oracle look-ahead 虚高 TPS。token mixer/router/shared/hot experts 经 APTMoE
load stream 上卡，cold experts 保持 CPU home。top-8 dispatch 逐 expert 加权并
`index_add_`，不是只跑 top-1。动态补载会完整出队当前层的缺失 experts，不把
cold-start 队列遗留到稳定窗口；历史 popularity 在原始 forward 后固定，checkpoint
反向重算产生的第二次路由不会被重复计数。

### 3. 真实 full update 与审计

入口以 `lora_mode=False` 创建原 APTMoE `PipelineRuntime`，其 optimizer 对全部
34,660,610,688 个参数构造 `torch.optim.AdamW`。adapter 的训练循环执行真实
forward、loss、backward、梯度裁剪和 `optimizer.step()`，并保持 APTMoE 原有的
“每个 gradient-accumulation microbatch 完整 forward+backward”语义。

[`aptmoe_proxy/runtime.py`](aptmoe_proxy/runtime.py) 强制审计：

```text
optimizer 中的 Parameter 对象集合 == 本 rank 所有 proxy Parameter 对象集合
所有 rank optimizer 参数合计 == 34,660,610,688
embedding / LM head / norms / token mixer / router /
  routed experts / shared expert+gate 均观察到非零梯度
除 norm 外的上述类别均观察到代表权重数值变化
Adam exp_avg、exp_avg_sq 的实际 dtype == BF16，home device == CPU
```

结果写入 `full_update_verification.json`；任一项失败，runner 返回失败。当前
APTMoE baseline 对 BF16 参数直接使用 AdamW，本机实测两个 moment 也是 BF16。
norm 仍必须在 optimizer scope 中且有非零梯度，但不把数值变化设为硬门槛：其初值
为 1，`1e-5` 更新小于 BF16 在 1 附近的量化间隔。
若未来理想适配版改用 FP32 moments/master weights，proxy 必须同步修改，否则
optimizer 内存流量和 TPS 不可比。随机 model-only 权重只在显式
`--save-random-weights` 时保存，且路径被限制在 `APTMoE-simulate/`。

### 4. 重新生成 lookup table

在目标机器上分别 profile：

- 6 MiB BF16 expert 的 H2D、D2H；
- 0～实际最大 token 数的 CPU SwiGLU(2048,512) forward/backward；
- 256-way router；
- linear-attention 与 full-attention 的 H2D、GPU forward/backward；
- 首尾 stage 的 embedding、final norm 和 LM head H2D。

server 和 consumer 若 CPU/GPU/PCIe 拓扑不同，应分别生成 lookup table。禁止复用
Qwen3-30 的 9 MiB expert 表。placement 保留原 APTMoE 的累计
`CPU expert forward / H2D load < 1` 判据，但固定加载项会按当前层选择实测的
linear-attention 或 full-attention H2D 时间，并在 stage 0/39 加入 embedding 或
final-norm/LM-head，而不是对所有层共用一个 `load_MHA`。
曲线覆盖上限至少为该 profile 的最大 `global_batch × sequence`：server 32,768，
consumer 4,096；否则长序列 formal run 会在分配模型前拒绝该 lookup。

### 5. 路由输入

已提供的 hook 在真实 Qwen3.5 KTransformers/DeepSpeed 被排除的 warmup forward
捕获每层、每 token 的 top-8 expert ID，各 rank 退出时落盘，再合并成
`[patterns, 40, global_batch × sequence, 8]`。默认 5 个 pattern，GAS>1 时乘以
GAS；proxy 按每个 accumulation microbatch 依次循环重放，以覆盖多个 batch 的
路由局部性和稀疏 Adam state 物化。由此既能还原 256-bin counts，又能保持
expert ID 和 token dispatch；
router 仍计算完整 logits/softmax，并从 replay ID gather 可求导 gate scores。
trace 不包含 token ID 或样本文本，所有 D2H copy 都落在被排除的 warmup 中。

固定 seed 的 Zipf synthetic trace 和随机 router 都只允许显式 smoke。formal runner
要求 metadata source 为 `merged_exact_qwen35_router_trace`，否则在模型分配前终止。

### 6. 计时与结果隔离

adapter 将每 rank 的 APTMoE action list 分成 forward/backward 两段，按原语义让每个
gradient-accumulation microbatch 依次完成两段，并累计 host wall time。完整 step
wall time仍以 optimizer-step 外层计时。timed window 内不调用全局
`torch.cuda.synchronize()` 或 barrier；stage load/compute/drop 通过 CUDA event
`wait_event` 排序，在 backward 尾部只等待本 rank 各 stage 的 `_StageDropEvent`，
确保下一 microbatch/optimizer 看到已回到 CPU 的参数。

输出至少增加：

```json
{
  "benchmark_class": "deployment_proxy",
  "target_model": "Qwen3.5-35B-A3B-text",
  "weight_source": "deterministic_random_initialization",
  "checkpoint_compatible": false,
  "llamafactory_backend": false,
  "real_forward_backward_optimizer_update": true,
  "result_validity": "formal_deployment_proxy|smoke_only",
  "route": {"mode": "replayed_qwen35_topk_indices"},
  "placement": {"mode": "profiled_compute_load"}
}
```

结果目录使用 `APTMOE_BF16_DEPLOYMENT_PROXY_SWEEP`，聚合器不得把其 TPS 放入
KTransformers/DeepSpeed 的 exact-model 对比行。建议表格分成：

1. exact full-FT：KTransformers、DeepSpeed；
2. deployment proxy：APTMoE；
3. operational baseline：原样 Qwen3-30（可选）。

## 当前机器与存储预算

### 当前实测

截至 2026-07-23，当前执行环境为：

| 项目 | 实测 |
|---|---:|
| `/mnt/data2/wbw` 文件系统可用 | 833,158,369,280 bytes = 775.94 GiB |
| `/tmp` 可用 | 25.02 GiB，不适合作为大型 CUDA 扩展编译目录 |
| `/mnt/data3/models/Qwen3.5-35B-A3B` | 66.98 GiB，已有且只读 |
| `/mnt/data3/models/Qwen3-30B-A3B` | 56.89 GiB，已有且只读 |
| 当前 `Aptmoe` conda 环境 | 约 14 GiB |
| `FFTtest/Qwen3.5-35B-A3B` | 约 11 MiB（含本次脚本，不含 test_log） |
| 主机内存 | 2.0 TiB，无 swap |

源 checkpoint 不需要复制到当前路径。proxy 只读取不足一个 MiB 的 `config.json`，
随后在 RAM 中随机初始化参数。当前会话中 `nvidia-smi` 无法连接 NVIDIA driver，
所以只能完成静态审计；真正 smoke/full sweep 必须在 GPU 对该作业可见的执行节点上
进行。

### 磁盘预算

默认 `save_random_weights: false` 时，64.56 GiB 随机权重及 optimizer state 都是
运行时内存，不会形成同等大小的磁盘文件。建议按下面方式预留：

| 运行方式 | 当前路径新增磁盘预算 |
|---|---:|
| 复用现有 `Aptmoe` 环境、无 checkpoint | 30 GiB 配额 |
| 克隆一份独立环境后再安装 fast path、无 checkpoint | 50 GiB 配额 |
| 单个 BF16 model-only checkpoint | 额外 64.56 GiB |
| 单个 model + 两份 BF16 Adam moments | 额外 193.68 GiB |
| 单个 model + 两份 FP32 Adam moments | 额外 322.80 GiB |
| 再含 FP32 master weights | 额外 451.92 GiB |

无 checkpoint 的 30 GiB 包含约 20 GiB 的 CUDA 扩展构建/缓存上限、1 GiB
manifest/route/lookup/日志以及约 9 GiB 安全余量；清理构建中间件后的长期占用应远低于
该上限。若要克隆现有约 14 GiB 的 conda 环境，则使用 50 GiB 档。route、lookup、
随机权重、编译目录、Triton cache 和 torch extensions 都应放到 Git-ignored 的
`/mnt/data2/wbw/FFTtest/APTMoE-simulate/`，不要使用只剩 25.02 GiB 的 `/tmp`。

不建议 sweep 保存权重。8 个序列长度只保存一个 profile 的 model-only checkpoint
就需要 516.48 GiB；server 和 consumer 共 16 份需要 1,032.97 GiB，已经超过当前
775.94 GiB 可用空间。若确有调试需要，只保存一个最终 model-only checkpoint；runner
把它限制在 `APTMoE-simulate/random_weights/`，且 checkpoint I/O 位于计时窗口之外。

### 运行时 RAM 容量投影

Qwen3.5 text-only proxy 的精确参数量为 34,660,610,688。以下按“所有参数的
gradient 和 Adam state 均已 materialize”计算聚合 tensor payload，不包含
activation、CUDA allocator、pipeline buffer、pinned staging 和临时 kernel
workspace：

| 状态 | 聚合大小 |
|---|---:|
| BF16 model weights | 64.56 GiB |
| BF16 gradients | 64.56 GiB |
| 两份 BF16 Adam moments | 129.12 GiB |
| 当前 APTMoE 口径合计 | 258.24 GiB |
| 改用两份 FP32 moments 后合计 | 387.36 GiB |
| FP32 moments + FP32 master 后合计 | 516.48 GiB |

64.56 GiB model weights 是持久参数下界；稀疏 routed experts 的 gradient/moment
只在被路由后逐步建立，因此短 run 可能低于 258.24 GiB。258.24 GiB 是当前 BF16
state 策略的全 materialization 规划值，但实际 RSS 仍可能因 activation、allocator
和临时 buffer 超过它，必须通过 smoke run 测量。当前主机 2 TiB RAM、consumer
1 TiB cgroup 从参数状态看有余量，但不能由此推断 GPU 一定可装下 token mixer、
activation 和 expert staging。

## 分阶段执行方案

### 0. 冻结比较契约

先固定两个 profile：

| profile | world size | APTMoE pipeline global batch | 序列长度 |
|---|---:|---:|---|
| server | 8 | 8 | 32、64、128、256、512、1024、2048、4096 |
| consumer | 2 | 2 | 16、32、64、128、256、512、1024、2048 |

APTMoE 的 rank 是 pipeline rank，不是数据并行 replica。公共 wrapper 已明确传入
`--global-batch-size`，server/consumer 分别为 8/2，并强制它等于
`num_gpus × per_device_batch_size`；不会把 batch 1 误当成 global batch。两边都使用
AdamW、相同 learning rate、BF16 model/gradient，以及运行时审计确认的相同
optimizer-state dtype。

### 1. 准备 fast path

在独立或现有 `Aptmoe` 环境中安装与 PyTorch 2.9.1/CUDA 12.8 匹配的
`flash-linear-attention` 和 `causal-conv1d`。将构建临时目录和 kernel cache 指向
上述 workspace 路径。随后运行：

1. `require_linear_attention_fastpath()`；
2. 一次 linear-attention 和 full-attention BF16 forward/backward；
3. 记录实际 FLA、causal-conv1d、SDPA、PyTorch、CUDA 和 driver 版本。

任一项 fallback 或 GPU 不可见时停止，不进入 TPS sweep。

### 2. 验证已接入的 APTMoE runtime

使用已提供 adapter 检查以下强制项：

1. 包含 exact embedding、40 个 decoder layer、final norm、248,320-way LM head
   和交叉熵，不能只测 attention+expert；
2. GPU token mixer 复用 Qwen3.5 的实际 fast-path 组件；
3. 10,240 个 6 MiB routed experts 使用与理想适配版相同的独立 expert 布局和
   APTMoE load/drop/prefetch 路径；
4. replay 每层、每 token 的 top-8 expert ID；被选 gate weight 仍从 router logits
   gather，使 router 保持可求导；
5. `lora_mode=False`，optimizer 参数总数必须等于 34,660,610,688；
6. proxy 不加载模型 shard；默认不保存，显式保存也只能写到 simulation root。

adapter 不执行 APTMoE baseline 中额外的“step 0”，也不保留逐 micro-batch 全局
`cuda.synchronize()`/barrier。每档恰好执行 15 个真实 update，初始化、NUMA
first-touch、optimizer-state materialization 和 full-update audit 都落在前 5 个
warmup step，后 10 个 update 才进入稳定 TPS。

### 3. 重新 profile APTMoE

分别在 server/consumer 拓扑生成 lookup table，包括 6 MiB expert H2D/D2H、
CPU expert forward/backward、256-way router、两类 attention forward/backward。
显式记录 CPU core pinning、NUMA policy、PCIe 拓扑、pinned-memory 上限和
`prefetch_portion`。Qwen3-30 的 9 MiB lookup table 只用于通路检查。

### 4. 静态审计与 smoke

执行顺序为：

1. 生成 `proxy_manifest.json`，与实际 `named_parameters()` 分类统计逐项比较；
2. meta-device 审计总参数及组件参数；
3. consumer 2-GPU、seq=16、2 个 warmup + 2 个 update；
4. 验证 token mixer、router、shared expert、至少一个 CPU routed expert 的
   `grad != None` 且权重发生改变；
5. seq=128 重跑，检查 OOM、NaN、deadlock、state dtype/home device 和 route replay；
6. 通过后再进行 8-GPU smoke。

### 5. 正式 sweep

server 依次跑 32、64、128、256、512、1024、2048、4096，consumer 依次跑
16、32、64、128、256、512、1024、2048；每档执行 15 个真实 optimizer updates，
前 5 个 warmup、后 10 个计时。consumer 使用 1 TiB cgroup、关闭 swap并 NUMA 0/1
interleave；server 使用 8 GPU。每档单独进程启动，输出 canonical step records、
route/placement/fast-path manifest 和 `full_update_verification.json`，但不输出
checkpoint。

## 与理想 Qwen3.5 APTMoE 的 TPS 关系

结论是：**组件同构 proxy 可以作为很接近的 TPS 估算，但原样 Qwen3-30 不可以；
在真正适配版出现前，也不能宣称已经实测“一致”。**

随机权重本身通常不会改变 dense BF16 GEMM/attention/expert kernel 的形状和字节量。
只要以下条件同时成立，proxy 与“同一 APTMoE runtime 的完整 Qwen3.5 适配版”执行的
关键路径基本相同：

- attention 类型、实现、fast path 和 gradient checkpointing 完全一致；
- expert 数量、单 expert 形状、shared expert、load/drop/prefetch 及 CPU/GPU placement
  完全一致；
- 使用真实 route replay，expert ID 的时间局部性和每个 expert token 数一致；
- embedding、LM head、loss、backward、gradient clipping 和 optimizer update 均未省略；
- global batch、有效 token 数、数据 padding、pipeline schedule、NUMA/core pinning一致；
- optimizer 类型、moment dtype、state materialization 和 master-weight 策略一致；
- checkpoint 加载、初始化、保存和 profiler 均排除在计时窗口之外。

在这些门槛通过后，可把下面数值作为**工程验收带，而不是当前实测结论**：

| profile | proxy TPS / 理想适配版 TPS |
|---|---:|
| server 8-GPU | 0.95～1.05 |
| consumer 2-GPU | 0.90～1.10 |

consumer 放宽到 ±10%，因为每 rank 承担约 20 层，CPU expert、NUMA first-touch、
page placement、pinned buffer 和预取调度对整体 TPS 更敏感。若做到完全相同的 route
replay、optimizer 预热和 NUMA 预触碰，通常应向 ±5%～8% 收敛。以上区间需要等真正
Qwen3.5 APTMoE 适配版可运行后，用相同机器、相同 trace 做 A/B 才能确认。

以下任一情况都不能称为“几乎一致”：

- linear attention 走 Transformers 慢速 fallback；
- 使用原样 Qwen3-30 的 48 层 full GQA 或 9 MiB expert；
- 让随机 router 自由路由而不 replay；
- proxy 用逐 expert dispatch，而理想版用 fused/batched expert kernel，或反之；
- 一边使用 BF16 moments，另一边使用 FP32 moments/master weights；
- 省略 248,320-way LM head、loss、backward 或 optimizer update；
- 把首次分配、checkpoint I/O 或同步 profiler 计入一边但不计入另一边。

因此最终报告建议写成
`APTMoE Qwen3.5 component-isomorphic deployment-proxy TPS estimate`，并同时给出
server/consumer 的 proxy TPS、route/fast-path/placement audit 和上述误差带；不能把
它直接改名为真实 Qwen3.5 full-FT TPS。

## 已提供的代码与用法

参数审计与 dry-run：

```bash
cd /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B

/mnt/data2/wbw/conda/envs/Aptmoe/bin/python qwen35_proxy_spec.py \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --qwen3-reference-model-path /mnt/data3/models/Qwen3-30B-A3B \
  --output /mnt/data2/wbw/FFTtest/APTMoE-simulate/spec.json \
  --summary

bash run_finetune_perf_test_bf16_aptmoe.sh \
  --profile both --seq-lengths 32 --skip-dataset-check --dry-run
```

文件说明：

- [`qwen35_proxy_spec.py`](qwen35_proxy_spec.py)：从两个模型的 config 计算精确
  参数与偏差，生成 proxy contract；
- [`qwen35_aptmoe_proxy_components.py`](qwen35_aptmoe_proxy_components.py)：
  真实 Qwen3.5 token mixer 与精确 expert 组件；
- [`aptmoe_proxy/model.py`](aptmoe_proxy/model.py)：APT route replay、top-8
  dispatch、CPU expert bridge 和 40-stage 模型；
- [`aptmoe_proxy/runtime.py`](aptmoe_proxy/runtime.py)：pipeline full update、
  参数更新审计和 canonical timing；
- [`aptmoe_proxy/placement.py`](aptmoe_proxy/placement.py)：与 APTMoE
  compute/load 判据一致的 host-profiled placement；
- [`qwen35_route_capture.py`](qwen35_route_capture.py) 与
  [`merge_qwen35_route_traces.py`](merge_qwen35_route_traces.py)：真实路由采集和合并；
- [`profile_aptmoe_qwen35_proxy.py`](profile_aptmoe_qwen35_proxy.py)：6 MiB
  expert、CPU curve、router 和 token mixer lookup profiler；
- [`generate_synthetic_qwen35_routes.py`](generate_synthetic_qwen35_routes.py)：
  只供 smoke 的确定性 Zipf 路由；
- [`aptmoe_qwen35_proxy_train.py`](aptmoe_qwen35_proxy_train.py)：分布式入口；
- [`configs/aptmoe_qwen35_deployment_proxy_bf16.yaml`](configs/aptmoe_qwen35_deployment_proxy_bf16.yaml)：
  已实现 adapter 的配置契约。

正式运行前，proxy manifest、实际 `named_parameters()` 分类统计和运行时 placement
audit 三者必须一致；任何一项不一致都在 optimizer step 或结果聚合前终止。完整命令
见 [`README_PERF_SWEEP.md`](README_PERF_SWEEP.md)。

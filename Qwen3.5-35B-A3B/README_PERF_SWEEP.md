# Qwen3.5-35B-A3B BF16 全量微调与 APTMoE Proxy TPS Sweep

这组脚本不包含 LoRA。KTransformers/DeepSpeed 测真实文本模型全量微调；
APTMoE 测随机权重的组件同构 full-update proxy。目标配置固定为本地
`/mnt/data3/models/Qwen3.5-35B-A3B`，训练与后端混合精度均显式设为
BF16。默认对 `32,64,128,256,512,1024,2048,4096` 八种 sequence
length 分别运行 15 个 optimizer steps，去除前 5 个 warmup steps 后计算
稳定 TPS。这里的 5 步是性能统计排除窗口；训练配置的学习率 warmup 为 0。

## 文本-only 模型契约

本地 checkpoint 是多模态的 `Qwen3_5MoeForConditionalGeneration`。仅设置
`freeze_vision_tower: true` 仍会构造视觉塔，因此 Torch 2.9 仍能检测到其中的
`Conv3d`；`20260722_181421_KTRANSFORMERS_BF16_FULL_SWEEP` 正是因此在训练开始前
终止。

现在 KTransformers 和 DeepSpeed 入口会强制执行以下流程：

- 从源配置提取 `text_config`，构造 `Qwen3_5MoeForCausalLM`；
- 由 Transformers 将 checkpoint 的 `model.language_model.*` 映射到文本模型，
  不加载 `model.visual.*` 和 `mtp.*`；
- 只加载 tokenizer，不创建 `AutoProcessor`；
- 使用无多模态 plugin 的 `qwen3` 文本模板；
- optimizer 创建前检查模型类型、`Conv3d` 数量和多模态参数数量；任一不符合即终止。

因此这里的“全量微调”是全部文本 CausalLM 参数的全量微调，不包括视觉塔。数据预检
也禁止样本出现 `image(s)`、`video(s)` 或 `audio(s)` 字段。每个
`run_config.json` 会记录 `modality: text_only` 和实际加载架构。

## 计时与性能干扰约束

三个后端只记录每个 optimizer step 的以下数据：

- forward host wall time；
- backward host wall time；
- optimizer host wall time；
- 完整 optimizer-step wall time 和由其计算的 TPS。

计时器只在三个阶段的 API 边界读取 `time.perf_counter()`，不会调用
`torch.cuda.synchronize()`。逐 step 数据缓存在内存中，训练结束后才统一写入文件。
脚本强制设置 `DS_PROBE_MODE=off`、`KT_BACKWARD_TIMING=off`、
`KT_SFT_PROFILE=0`，并且不启动 CPU、磁盘、GPU 或内存采样进程。

因此三个阶段是训练 API 的 host wall time，不应解释为纯 GPU kernel 时间。DeepSpeed
的 optimizer 时间对应 `DeepSpeedEngine.step()` 整段，包含 ZeRO/offload 的更新工作，
但不会进一步探测 CPUAdam 或 ZeRO 内部子阶段。

## Profile

| Profile | GPU | 全局 batch | 每卡 batch | 内存与 NUMA |
|---|---:|---:|---:|---|
| server | 8 | 8 | 1 | 不加 cgroup 上限，使用主机现有约 2T 内存 |
| consumer | 2 | 2 | 1 | cgroup v2 `MemoryMax=1T`、`MemorySwapMax=0`，NUMA 0/1 等比例 interleave，满载目标各 512G |

consumer 默认优先使用用户级 transient systemd scope。若管理员已经把当前
shell 放入有效上限恰好为 1 TiB 的 cgroup，可使用
`--consumer-cgroup-mode prelimited`；若需要由系统级 systemd 创建 scope，使用
`--consumer-cgroup-mode system`。脚本不会用 `ulimit` 冒充整棵进程的内存硬限制。

资源校验程序在模型加载前检查 cgroup、swap 和 NUMA policy，写出
`resource_contract.json` 后用 `exec` 替换自身；训练期间不会留下采样或包装进程。

## CPU 线程

DeepSpeed 和 APTMoE 默认每个训练 rank 使用：

```text
floor(当前进程可见物理核心数 / profile 的 GPU/rank 数)
```

KTransformers 使用 rank0 集中式 CPU MoE backend，因此不再把 96 个物理核平均
切给所有 GPU rank。默认给每个非 owner rank 2 线程、给系统和通信辅助线程预留
2 核，其余全部交给 rank0：

```text
server（8 卡）：rank0 80；rank1～7 各 2；计划合计 94
consumer（2 卡）：rank0 92；rank1 2；计划合计 94
```

训练入口会在导入 PyTorch 前按 global rank 分别设置 `OMP_NUM_THREADS`、
`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、
`BLIS_NUM_THREADS` 和 `ACCELERATE_KT_OMP_NUM_THREADS`。`kt_num_threads`
单独使用 owner 的线程数。

可分别覆盖普通 rank 和 KT owner：

```bash
bash run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --cpu-threads 2 \
  --kt-owner-threads 80
```

也可以使用 `FFT_CPU_THREADS=2` 和 `FFT_KT_OWNER_THREADS=80`。显式参数
优先于对应环境变量。

## 启动

```bash
# KTransformers AMX BF16
bash run_finetune_perf_test_bf16_ktransformers.sh --profile server
bash run_finetune_perf_test_bf16_ktransformers.sh --profile consumer

# 原生 LLaMA-Factory + DeepSpeed ZeRO-3 CPU offload BF16
bash run_finetune_perf_test_bf16_deepspeed.sh --profile server
bash run_finetune_perf_test_bf16_deepspeed.sh --profile consumer

# 仅检查所有生成配置与资源包装命令
bash run_finetune_perf_test_bf16_ktransformers.sh --profile both --dry-run
```

`--profile both` 按 server、consumer 顺序运行。可以通过
`--seq-lengths 32,64` 缩小调试范围；正式对比应保留默认八档。

## APTMoE deployment proxy（已实现，非等价后端）

APTMoE 官方 artifact 没有 Qwen3.5 的通用 Hugging Face/LLaMA-Factory 后端。本目录
现在提供一个独立 adapter，它导入 `/mnt/data2/wbw/APTMoE-baseline`，不修改该
checkout：

- 40 层、34,660,610,688 个参数，与 Qwen3.5 text CausalLM 的组件参数量一致；
- GPU 使用 Transformers 的 30 个 Gated DeltaNet 和 10 个 gated full-attention；
- CPU home 部署 40×256 个独立 6 MiB BF16 routed experts；
- 保留 router、shared expert、248,320-way LM head、loss、backward、梯度裁剪和
  AdamW；每 step 是真实的全参数可更新训练路径；
- 权重由固定 seed 随机初始化，不加载 Qwen3.5 checkpoint shard，默认也不保存。

所以它测的是 `APTMoE Qwen3.5 component-isomorphic deployment-proxy TPS`，不产生
有意义的 loss、模型效果或可用 checkpoint。KTransformers、DeepSpeed 仍通过
LLaMA-Factory 测真实 Qwen3.5；APTMoE proxy 明确记录
`llamafactory_backend: false`，汇总器会永久分表。

完整设计、参数/内存推导和误差边界见
[`APTMOE_DEPLOYMENT_PROXY.md`](APTMOE_DEPLOYMENT_PROXY.md)。

### 1. 无 GPU 参数审计

```bash
cd /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B

PYTHONPATH="$PWD:/mnt/data2/wbw/APTMoE-baseline" \
  /mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
  aptmoe_qwen35_proxy_train.py \
  --aptmoe-root /mnt/data2/wbw/APTMoE-baseline \
  --deployment-profile server \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --output-dir /mnt/data2/wbw/FFTtest/APTMoE-simulate/audit \
  --step-timing-output-dir /mnt/data2/wbw/FFTtest/APTMoE-simulate/audit/timing \
  --sequence-length 32 --num-gpus 8 --global-batch-size 8 \
  --per-device-batch-size 1 --steps 2 --warmup-steps 1 \
  --precision bf16 --text-only --audit-only
```

### 2. 采集真实 Qwen3.5 路由

在任一真实后端 sweep 上加 `--capture-aptmoe-routes`。hook 只在被排除的
warmup forward 把 top-8 expert ID 搬到 CPU（默认 5 个 pattern，GAS>1 时相应
增加），训练退出后自动合并各 rank：

```bash
bash run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --capture-aptmoe-routes
```

输出为
`APTMoE-simulate/routes/qwen35/{server,consumer}/seq_<长度>.npz`。也可在
DeepSpeed wrapper 上执行同一参数。trace 形状为
`[patterns, 40, global_batch×sequence, 8]`，proxy 按每个 accumulation
microbatch 依次循环重放，覆盖多个 batch 的路由局部性和 optimizer-state
物化。formal run 要求 pattern 数严格等于 `warmup_steps×GAS`，确保所有 route
cache 和稀疏 state 首次触达都被排除。正式 proxy 会校验 trace 的来源、
层数、token 数、top-k、expert 范围和重复 ID；synthetic trace 不能冒充正式结果。

### 3. 在目标拓扑生成 lookup

当前 `Aptmoe` 环境尚缺兼容的 `flash-linear-attention` 和 `causal-conv1d`，
`is_fast_path_available == false`。正式 GPU attention TPS 前必须先安装兼容
PyTorch 2.9.1/CUDA 12.8 的版本。然后分别在 server/consumer 实际 CPU
线程、NUMA、cgroup 和 PCIe 拓扑下运行：

```bash
export CUDA_CACHE_PATH=/mnt/data2/wbw/FFTtest/APTMoE-simulate/cache/cuda
export TORCH_EXTENSIONS_DIR=/mnt/data2/wbw/FFTtest/APTMoE-simulate/cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data2/wbw/FFTtest/APTMoE-simulate/cache/triton

/mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
  profile_aptmoe_qwen35_proxy.py \
  --deployment-profile server \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --output /mnt/data2/wbw/FFTtest/APTMoE-simulate/lookups/qwen35/server.json \
  --simulation-root /mnt/data2/wbw/FFTtest/APTMoE-simulate \
  --sequence-length 128 --max-tokens 32768 --cpu-threads 12

/mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
  profile_aptmoe_qwen35_proxy.py \
  --deployment-profile consumer \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --output /mnt/data2/wbw/FFTtest/APTMoE-simulate/lookups/qwen35/consumer.json \
  --simulation-root /mnt/data2/wbw/FFTtest/APTMoE-simulate \
  --sequence-length 128 --max-tokens 8192 --cpu-threads 48
```

lookup 覆盖 6 MiB expert H2D/D2H、CPU expert forward/backward 曲线、
256-way router、两种 Qwen3.5 token mixer，以及首尾 stage 的
embedding/final-norm/LM-head 搬运。不能复用 Qwen3-30B 的 9 MiB 表。
`max-tokens` 必须至少覆盖该 profile 最大的 `global_batch×sequence`；不足时正式
runner 会在模型分配前拒绝 lookup，而不会静默 clamp。

### 4. Smoke 与正式运行

仅验证 pipeline/参数路径时，必须显式打开全部 smoke fallback：

```bash
bash run_finetune_perf_test_bf16_aptmoe.sh \
  --profile consumer \
  --seq-lengths 32 \
  --steps 4 --warmup-steps 2 \
  --aptmoe-allow-synthetic-routing \
  --aptmoe-allow-unprofiled-placement \
  --aptmoe-allow-linear-attention-fallback
```

这类结果固定标为 `SMOKE_ONLY`。真实路由、profile lookup 和 linear-attention
fast path 都准备好后，正式命令不带任何 fallback：

```bash
bash run_finetune_perf_test_bf16_aptmoe.sh --profile server
bash run_finetune_perf_test_bf16_aptmoe.sh --profile consumer
```

默认使用 `/mnt/data2/wbw/conda/envs/Aptmoe` 和
`/mnt/data2/wbw/APTMoE-baseline`，均可通过 `--aptmoe-python`、
`--aptmoe-root` 覆盖。正式运行若缺 route、lookup 或 fast path，会在大规模参数
分配前终止。

随机参数只在 RAM 中构造。仅当显式传入 `--keep-model-output` 时，才按 rank 写出
约 64.56 GiB 的 model-only 随机权重；路径固定在
`/mnt/data2/wbw/FFTtest/APTMoE-simulate/random_weights/`。整个
`APTMoE-simulate/` 已被 Git 忽略。不要对 8 个长度和两个 profile 全量保存，
16 份约 1,032.97 GiB，超过当前约 776 GiB 可用空间。

## 结果

真实后端写入 `test_log/<timestamp>_<backend>_BF16_FULL_SWEEP/`；APTMoE 写入
`test_log/<timestamp>_APTMOE_BF16_DEPLOYMENT_PROXY_SWEEP/`。其中：

- 每个 sequence 的训练配置、完整日志和 `step_timing.json/csv/md`；
- `sweep_results.csv`：稳定 TPS，以及 forward、backward、optimizer 平均耗时；
- `summary.md`：按 profile 汇总的对比表；
- `dataset_validation.json`：Qwen3.5 tokenizer 下的数据长度与 BF16 模型校验；
- `run_config.json`：源架构、文本加载架构以及 `text_only` 模态契约；
- 每个 sequence 的 `resource_contract.json`：训练开始前实际生效的 cgroup、swap
  和 NUMA policy。它不是运行期资源采样结果。
- APTMoE 另写 `proxy_manifest.json` 和 `full_update_verification.json`，记录精确
  参数分类、路由/placement/fast-path 来源、optimizer scope、梯度、权重变化以及
  BF16 moment 的 CPU home device。

默认跳过 LLaMA-Factory 在训练结束后的完整模型保存，避免每个 sequence 重复写出
几十 GB 权重；它不属于 optimizer-step TPS 窗口。APTMoE 同样默认不保存随机权重。
需要保留时显式传入 `--keep-model-output`；APTMoE 只允许写入 Git-ignored 的
`APTMoE-simulate/random_weights/`。

TPS 公式为：

```text
TPS = GPU 数 × 每卡 batch × sequence length × GAS
      / 去除 warmup 后的平均 optimizer-step wall time
```

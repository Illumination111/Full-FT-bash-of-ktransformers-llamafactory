# Qwen3.5-35B-A3B 三后端文本-only BF16 全量微调脚本调用与参数配置

三个后端共用同一套参数解析、序列长度 sweep、TPS 计时和资源限制逻辑。BF16 是硬编码配置，不能切换为 FP16/FP32。

本地 Qwen3.5 checkpoint 是多模态的，但本测试强制只加载
`Qwen3_5MoeForCausalLM` 文本子模型：视觉塔、`Conv3d`、多模态 processor 和视觉
参数均不会进入模型或 optimizer。这里的“全量微调”专指全部文本 CausalLM 参数。

## 1. 一步启动完整测试

下面的命令可在任意目录直接执行，无需先 `cd`。每条命令都会依次完成：

```text
server：8 GPU、全局 batch 8、使用主机约 2T 内存
consumer：2 GPU、全局 batch 2、限制 1 TiB 内存
sequence length：32、64、128、256、512、1024、2048、4096
每个长度：15 steps，排除前 5 个 warmup steps
精度：BF16
模型：text-only Qwen3_5MoeForCausalLM
```

`--profile both` 会先跑 server，再跑 consumer，不会同时运行两个 profile。默认 CPU
线程按可见物理核心数除以训练 rank 数自动计算；在当前 96 物理核心主机上，server
为 12 线程/rank，consumer 为 48 线程/rank。

### KTransformers：一条命令跑完整测试

该命令自动使用 `Kllama` Conda 环境和 AMXBF16 后端：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --seq-lengths 32,64,128,256,512,1024,2048,4096 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --kt-distributed-checkpoint-reuse on \
  --consumer-cgroup-mode auto \
  --continue-on-error
```

`--kt-distributed-checkpoint-reuse on` 使多卡 gradient-checkpoint 重算阶段复用第一次
forward 已生成的 CPU routed-expert 缓存。GPU attention、router 等非 CPU MoE 部分仍按
checkpoint 语义正常重算。该开关仅作用于 KTransformers，且默认值就是 `on`；命令中显式
写出是为了让性能测试条件清晰可复现。

KTransformers 和 DeepSpeed 生成的 LLaMA-Factory 配置都固定使用非重入式 checkpoint：
`gradient_checkpointing_kwargs: {use_reentrant: false}`。这也避免 Transformers `Trainer`
的二次初始化把 KTransformers 所需的非重入式 checkpoint 静默改回 LLaMA-Factory 的
reentrant 默认值。APTMoE 使用外部 adapter；做三后端严格对比时，该 adapter 也应采用
等价的非重入式 checkpoint 设置。

### DeepSpeed：一条命令跑完整测试

该命令自动使用 `Deepspeed` Conda 环境、ZeRO-3、参数和 optimizer CPU offload：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --seq-lengths 32,64,128,256,512,1024,2048,4096 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --consumer-cgroup-mode auto \
  --continue-on-error
```

DeepSpeed 的 CPUAdam/ZeRO 内部探针不可启用。脚本会强制使用
`DS_PROBE_MODE=off`，optimizer 时间只记录完整的 `DeepSpeedEngine.step()`。

### APTMoE：一条命令跑完整测试

APTMoE 目前必须先准备已经完成 Qwen3.5 全量训练适配的 Python 入口。将下面命令中
两个 `/path/to/...` 替换为真实路径后即可启动：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_aptmoe.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --seq-lengths 32,64,128,256,512,1024,2048,4096 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --consumer-cgroup-mode auto \
  --aptmoe-python /path/to/aptmoe-env/bin/python \
  --aptmoe-entrypoint /path/to/qwen35_aptmoe_bf16_adapter.py \
  --continue-on-error
```

APTMoE adapter 必须接受脚本传入的 `--text-only`，并保证只构造
`Qwen3_5MoeForCausalLM`，不得加载视觉塔或 processor。它还必须生成
`<step-timing-output-dir>/step_timing.json`。具体参数契约见
[README_PERF_SWEEP.md](../Qwen3.5-35B-A3B/README_PERF_SWEEP.md)。未提供 adapter
时只能执行 `--dry-run`，真实训练会明确终止。

### 只运行单个 profile

如果只想测试一种机器规格，可直接复制下面的短命令；其余参数使用上述默认值：

```bash
# KTransformers server：8 卡、约 2T 内存
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7 \
  --kt-distributed-checkpoint-reuse on

# KTransformers consumer：2 卡、1 TiB 内存
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --devices 0,1 \
  --kt-distributed-checkpoint-reuse on

# DeepSpeed server
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7

# DeepSpeed consumer
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile consumer \
  --devices 0,1
```

### 三后端统一计时规则

三套脚本只记录每个 optimizer step 的：

- forward host wall time；
- backward host wall time；
- optimizer host wall time；
- 完整 optimizer-step wall time 和据此计算的 TPS。

计时器只读取 `time.perf_counter()`，不调用 `torch.cuda.synchronize()`；逐 step
记录先缓存在内存中，训练结束后统一写出。所有后端都强制设置：

```text
DS_PROBE_MODE=off
KT_BACKWARD_TIMING=off
KT_SFT_PROFILE=0
FFT_DISABLE_PERF_PROBES=1
```

脚本不会在训练期间启动 CPU、磁盘、GPU 或内存采样进程。这里的阶段耗时是训练
API 的 host wall time，不应解释为纯 GPU kernel 时间。

## 2. Profile 固定配置

| Profile | GPU | 每卡 batch | 全局 micro-batch | 每个 optimizer step 的有效 batch | 内存 |
|---|---:|---:|---:|---:|---|
| server | 8 | 1 | 8 | `8 × GAS` | 不设置 cgroup 上限 |
| consumer | 2 | 1 | 2 | `2 × GAS` | 恰好 1 TiB，禁止 swap |

GPU 数量和 batch 不提供单独的 `--gpus`、`--batch-size` 参数，避免测试时破坏 server/consumer 定义。

consumer 使用：

```text
MemoryMax=1T
MemorySwapMax=0
numactl --interleave=0,1
```

训练启动前会验证实际 cgroup 上限和 NUMA policy，然后校验程序通过 `exec`
替换为训练进程，不会在性能测试期间保留采样进程。默认自动选择 cgroup 模式；
如果用户级 systemd 不可用，可以使用：

```bash
# 通过系统级 systemd 创建限制，可能需要管理员权限
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --consumer-cgroup-mode system

# 当前 shell 已经位于恰好 1 TiB 的 cgroup 中
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --consumer-cgroup-mode prelimited
```

## 3. 公共参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--profile` | `server` | `server`、`consumer` 或 `both` |
| `--seq-lengths` | 八档全部 | 逗号分隔的 sequence length |
| `--steps` | `15` | 每个 sequence 的 optimizer steps |
| `--warmup-steps` | `5` | 前 N 步不计入稳定 TPS |
| `--gas` | `1` | Gradient accumulation steps |
| `--learning-rate` | `1.0e-5` | 全量微调学习率 |
| `--cpu-threads` | 物理核心数/rank 数 | 每个训练 rank 的统一 CPU 线程数 |
| `--devices` | 自动选择 | 物理 GPU 列表 |
| `--model-path` | `/mnt/data3/models/Qwen3.5-35B-A3B` | 模型目录 |
| `--dataset-dir` | `FFTtest/dataset` | LLaMA-Factory 数据集目录 |
| `--dataset-name` | `fft_real_100` | `dataset_info.json` 中的数据集名称 |
| `--log-base` | 当前目录下 `test_log` | 测试结果根目录 |
| `--kt-distributed-checkpoint-reuse` | `on` | KTransformers 多卡 checkpoint 重算复用第一次 CPU MoE forward；可设为 `off` 做 A/B 对照 |
| `--continue-on-error` | 关闭 | 某个长度失败后继续后续测试 |
| `--keep-model-output` | 关闭 | 保留最终模型；默认跳过完整权重保存 |
| `--skip-dataset-check` | 关闭 | 跳过 tokenizer 长度校验 |
| `--dry-run` | 关闭 | 只生成配置并打印命令 |

默认 sequence length 为：

```text
32,64,128,256,512,1024,2048,4096
```

例如只测试 512、2048 和 4096：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --seq-lengths 512,2048,4096
```

修改步数与 warmup：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile consumer \
  --steps 20 \
  --warmup-steps 5
```

这里 `--warmup-steps` 只控制 TPS 统计排除窗口；学习率 warmup 固定为 0。

也可以通过环境变量设置同一开关；显式命令行参数优先：

```bash
FFT_KT_DISTRIBUTED_CHECKPOINT_REUSE=off \
  bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server
```

每个 sequence 的 `run_config.json` 会记录
`kt_distributed_checkpoint_forward_reuse: true/false`。开启后，训练日志应包含：

```text
Checkpoint forward reuse: enabled=True, distributed_opt_in=True, world_size=2
Distributed checkpoint forward reuse active: layer=0, world_size=2
```

server 模式对应的 `world_size` 应为 `8`。第一行表示各 rank 对开关达成一致，第二行
表示 checkpoint 的第二次 forward 已实际进入缓存复用分支，而不仅是配置已开启。

## 4. GPU 选择

Server 指定 8 张卡：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7
```

Consumer 使用 GPU 6、7：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --devices 6,7
```

`--profile both` 提供 8 张卡时，server 使用全部 8 张，consumer 使用列表中的前两张：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7
```

也支持外部 `CUDA_VISIBLE_DEVICES`，但显式 `--devices` 优先。

## 5. Conda 与 CPU 线程配置

```bash
# 覆盖默认 Conda 环境
FFT_CONDA_ENV=Kllama \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh --profile server

FFT_CONDA_ENV=Deepspeed \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh --profile server
```

DeepSpeed 和 APTMoE 默认使用
`floor(当前进程可见物理核心数 / profile 的 GPU/rank 数)`。KTransformers
只有 global rank0 创建 CPU MoE backend，因此改为 rank-aware 分配：非 owner
rank 各 2 线程，预留 2 个物理核，其余核心交给 rank0。在当前 96 核主机上，
8 卡为 `80 + 7 × 2 = 94`，双卡为 `92 + 1 × 2 = 94`。

训练入口会按 global rank 设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、
`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、`BLIS_NUM_THREADS` 和
`ACCELERATE_KT_OMP_NUM_THREADS`；生成的 Accelerate 配置同时把
`kt_num_threads` 设置为 owner 线程数。

命令行覆盖方式：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --cpu-threads 2 \
  --kt-owner-threads 80
```

也可以用同一个环境变量控制任意后端：

```bash
FFT_CPU_THREADS=12 \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh --profile server
```

显式 `--cpu-threads` 优先于 `FFT_CPU_THREADS`；`--kt-owner-threads` 优先于
`FFT_KT_OWNER_THREADS`。旧的后端专用
`FFT_OMP_NUM_THREADS`、`FFT_DS_OMP_NUM_THREADS` 和
`FFT_APTMOE_OMP_NUM_THREADS` 不再使用。

## 6. 配置文件位置

- 公共训练模板：[train_full_bf16_qwen35.yaml](../Qwen3.5-35B-A3B/configs/train_full_bf16_qwen35.yaml)
- KTransformers 8 卡：[accelerate_ktransformers_bf16_8gpu.yaml](../Qwen3.5-35B-A3B/configs/accelerate_ktransformers_bf16_8gpu.yaml)
- KTransformers 2 卡：[accelerate_ktransformers_bf16_2gpu.yaml](../Qwen3.5-35B-A3B/configs/accelerate_ktransformers_bf16_2gpu.yaml)
- DeepSpeed ZeRO-3：[deepspeed_zero3_offload_bf16.json](../Qwen3.5-35B-A3B/configs/deepspeed_zero3_offload_bf16.json)
- 公共启动逻辑：[run_finetune_perf_sweep_bf16_common.sh](../Qwen3.5-35B-A3B/run_finetune_perf_sweep_bf16_common.sh)
- 文本-only 加载契约：[qwen35_text_only.py](../Qwen3.5-35B-A3B/qwen35_text_only.py)
- 统一粗粒度计时器：[step_phase_timer.py](../Qwen3.5-35B-A3B/step_phase_timer.py)
- 一次性资源校验/启动器：[resource_scope_exec.py](../Qwen3.5-35B-A3B/resource_scope_exec.py)
- 计时契约校验器：[validate_step_timing.py](../Qwen3.5-35B-A3B/validate_step_timing.py)

每个 sequence 都会在日志目录生成独立的 `train_config.yaml`；KTransformers 还会
生成写入实际 `kt_num_threads` 的 `accelerate_config.yaml`。这些才是当次测试的最终配置。

建议正式测试前先执行：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --dry-run
```

结果写入：

```text
test_log/<timestamp>_<backend>_BF16_FULL_SWEEP/
├── summary.md
├── sweep_results.csv
├── dataset_validation.json
├── server_8gpu_batch8/seq_*/
│   ├── resource_contract.json
│   └── step_timing/step_timing.{json,csv,md}
└── consumer_2gpu_batch2/seq_*/
```

稳定 TPS 公式为：

```text
TPS = GPU数 × 每卡batch × sequence length × GAS
      / 去除warmup后的平均optimizer-step时间
```

# Qwen3.5-35B-A3B 文本-only BF16 全量微调 TPS Sweep

这组脚本只测试全量微调，不包含 LoRA。模型固定为本地
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

三个后端使用完全相同的线程策略和环境变量。默认每个训练 rank 使用：

```text
floor(当前进程可见物理核心数 / profile 的 GPU/rank 数)
```

同一个数值会写入 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、
`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 和 `BLIS_NUM_THREADS`；
KTransformers 的 `kt_num_threads` 与 `ACCELERATE_KT_OMP_NUM_THREADS` 也使用该值。
可通过公共参数覆盖：

```bash
bash run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --cpu-threads 12
```

也可以设置对三个后端都生效的 `FFT_CPU_THREADS=12`。显式
`--cpu-threads` 优先于环境变量。

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

## APTMoE 适配器契约

APTMoE 官方 artifact 没有可直接用于 Qwen3.5 的通用 Hugging Face 全量训练入口，
因此脚本不会把 simulator TPS 当成模型训练 TPS。完成 Qwen3.5 runtime 移植后，
用下面方式接入：

```bash
bash run_finetune_perf_test_bf16_aptmoe.sh \
  --profile server \
  --aptmoe-python /path/to/aptmoe-env/bin/python \
  --aptmoe-entrypoint /path/to/qwen35_aptmoe_bf16_adapter.py
```

适配器必须接受公共脚本传入的这些参数：

```text
--model-path --dataset-dir --dataset-name --output-dir
--step-timing-output-dir --sequence-length --num-gpus
--global-batch-size --per-device-batch-size
--gradient-accumulation-steps --steps --warmup-steps
--learning-rate --precision
--text-only
```

它必须把多模态源 checkpoint 加载为 `Qwen3_5MoeForCausalLM`，不得创建视觉塔或
processor，并执行真实 Qwen3.5-35B-A3B 文本模型 BF16 全量 forward、backward 和
optimizer update。`--text-only` 是强制标志。适配器还必须遵守与另外两个后端相同
的无探针计时规则。在
`--step-timing-output-dir/step_timing.json` 中至少写出：

- `timing_mode: coarse_host_wall_no_cuda_sync`；
- 四个 `instrumentation` 标志均为 `false`；
- 每个 step 的 `forward_sec`、`backward_sec`、`optimizer_sec`、
  `step_total_sec`；
- `num_stable_steps`、`aggregate_stable` 和
  `tps_attribution.stable_tps`。

APTMoE 进程会收到和其他后端相同的 CPU 线程环境变量，以及
`FFT_DISABLE_PERF_PROBES=1`。

## 结果

每次启动在 `test_log/<timestamp>_<backend>_BF16_FULL_SWEEP/` 下生成：

- 每个 sequence 的训练配置、完整日志和 `step_timing.json/csv/md`；
- `sweep_results.csv`：稳定 TPS，以及 forward、backward、optimizer 平均耗时；
- `summary.md`：按 profile 汇总的对比表；
- `dataset_validation.json`：Qwen3.5 tokenizer 下的数据长度与 BF16 模型校验；
- `run_config.json`：源架构、文本加载架构以及 `text_only` 模态契约；
- 每个 sequence 的 `resource_contract.json`：训练开始前实际生效的 cgroup、swap
  和 NUMA policy。它不是运行期资源采样结果。

默认跳过 LLaMA-Factory 在训练结束后的完整模型保存，避免每个 sequence 重复写出
几十 GB 权重；它不属于 optimizer-step TPS 窗口。需要保留最终权重时显式传入
`--keep-model-output`。

TPS 公式为：

```text
TPS = GPU 数 × 每卡 batch × sequence length × GAS
      / 去除 warmup 后的平均 optimizer-step wall time
```

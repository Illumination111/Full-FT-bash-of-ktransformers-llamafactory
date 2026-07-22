# Qwen3.5-35B-A3B BF16 全量微调 TPS Sweep

这组脚本只测试全量微调，不包含 LoRA。模型固定为本地
`/mnt/data3/models/Qwen3.5-35B-A3B`，训练与后端混合精度均显式设为
BF16。默认对 `32,64,128,256,512,1024,2048,4096` 八种 sequence
length 分别运行 15 个 optimizer steps，去除前 5 个 warmup steps 后计算
稳定 TPS。这里的 5 步是性能统计排除窗口；训练配置的学习率 warmup 为 0，避免
把两种 warmup 混为一谈。

## Profile

| Profile | GPU | 全局 batch | 每卡 batch | 内存与 NUMA |
|---|---:|---:|---:|---|
| server | 8 | 8 | 1 | 不加 cgroup 上限，使用主机现有约 2T 内存 |
| consumer | 2 | 2 | 1 | cgroup v2 `MemoryMax=1T`、`MemorySwapMax=0`，NUMA 0/1 等比例 interleave，满载目标各 512G |

consumer 默认优先使用用户级 transient systemd scope。若管理员已经把当前
shell 放入有效上限恰好为 1 TiB 的 cgroup，可使用
`--consumer-cgroup-mode prelimited`；若需要由系统级 systemd 创建 scope，使用
`--consumer-cgroup-mode system`。脚本不会用 `ulimit` 冒充整棵进程的内存硬限制。

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

APTMoE [官方 artifact](https://github.com/yuanxinnn/APTMoE) 没有可直接用于
Qwen3.5 的通用 Hugging Face 全量训练入口，因此脚本不会把其 simulator TPS
当成模型训练 TPS。完成 Qwen3.5 runtime
移植后，用下面方式接入：

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
```

它必须做真实 Qwen3.5-35B-A3B BF16 全量前向、反向和 optimizer update，并在
`--step-timing-output-dir/step_timing.json` 写出与
`Qwen3-30B-A3B/step_timing_probe.py` 相同的 schema。至少要包含
`num_stable_steps`、`aggregate_stable.step_total_sec.mean_sec` 和
`tps_attribution.stable_tps`。

## 结果

每次启动在 `test_log/<timestamp>_<backend>_BF16_FULL_SWEEP/` 下生成：

- 每个 sequence 的训练配置、完整日志、逐步 timing JSON/CSV/Markdown；
- `sweep_results.csv`：稳定 TPS、阶段耗时、整段 CPU/磁盘/GPU 利用率；
- `summary.md`：按 profile 汇总的对比表；
- `dataset_validation.json`：Qwen3.5 tokenizer 下的数据长度与 BF16 模型校验。
- 每个 sequence 的 `resource_contract.json`、`resource_memory.csv`、
  `resource_summary.json`：训练 scope 内实际生效的 cgroup 上限、swap 上限以及
  NUMA 0/1 匿名内存峰值。consumer 若检测到有效上限并非恰好 1 TiB、swap 不为
  0 或 NUMA policy 不是 interleave，会在加载模型前终止。

默认跳过 LLaMA-Factory 在训练结束后的完整模型保存，避免每个 sequence 重复写出
几十 GB 权重；它不属于 optimizer-step TPS 窗口。需要保留最终权重时显式传入
`--keep-model-output`。

TPS 公式为：

```text
TPS = GPU 数 × 每卡 batch × sequence length × GAS
      / 去除 warmup 后的平均 optimizer-step wall time
```

系统利用率是整个 sequence run（含加载和预处理）的监控指标，稳定 TPS 则只取
warmup 之后的 optimizer steps，两者在报告中明确分开。

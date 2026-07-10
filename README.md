# Full FT bash of ktransformers & LLaMA-Factory

基于 **KTransformers + LLaMA-Factory** 的 MoE 全量微调（Full Fine-Tuning / FFT）自动化测试脚本集合。用于验证 AMX 后端下的训练通路、显存/内存占用、TPS，以及若干已知风险点（梯度累积、MoE 负载、保存 I/O 等）。

> **重要**：`finetuning_type=full`  alone 不够。真正打开 KT expert 全量梯度需要：
> ```bash
> export ACCELERATE_KT_TRAIN_MODE=full
> ```
> 各 `run_full_ft_*.sh` 脚本已自动导出该变量。

---

## 目录结构

```
.
├── dataset/                 # 共享真实 Full-FT 数据集（fft_real_100）
├── Qwen3-30B-A3B/           # Qwen3-30B-A3B 测试脚本与配置
├── GLM-4.5-Air/             # GLM-4.5-Air 测试脚本与配置
└── Qwen3.5-35B-A3B/         # Qwen3.5-35B-A3B 测试脚本与配置
```

每个模型目录大致包含：

| 内容 | 说明 |
|------|------|
| `run_*.sh` | 一键测试入口 |
| `configs/` | Accelerate + LLaMA-Factory 训练配置模板 |
| `monitor.py` / `analyze.py` | 运行时监控与日志分析 |
| `test_log/` | 运行产物（本地生成，默认不入库） |

---

## 环境依赖

运行前请确认：

1. **Conda 环境**（默认名 `Kllama`，可用 `FFT_CONDA_ENV` 覆盖）
   - 已安装 `accelerate`、`llamafactory`、`kt-kernel` 等
2. **本机路径**（脚本内写死，按需改脚本顶部变量）
   - LLaMA-Factory：`/mnt/data2/wbw/LLaMA-Factory`
   - 模型权重：见各脚本中的 `MODEL_PATH*`
3. **GPU**：多数脚本面向 1× / 4× RTX 4090；CPU 侧依赖 AMX + NUMA 布局

```bash
# 可选：指定 conda 环境名
export FFT_CONDA_ENV=Kllama
```

---

## 数据集

### 共享 Full-FT 数据集（推荐）

路径：`dataset/`

```bash
cd dataset
python gen_dataset.py   # 生成 fft_real_100.json + dataset_info.json
```

- 100 条 alpaca 样本，每条 tokenize 后 **> 7000 tokens**
- 训练 `cutoff_len=1024`，截断后每步恰好 1024 token，便于测 TPS / 显存

`run_full_ft_*.sh`（Qwen3-30B / GLM）默认使用该目录。

### Qwen3-30B 压力数据集

路径：`Qwen3-30B-A3B/data/`（`fft_stress_100`）

供 `run_fft_test_4gpu_*.sh` 使用。

---

## 快速开始

先做 dry-run 检查环境与配置，再正式跑：

```bash
# 示例：Qwen3-30B 真实 Full-FT（1 GPU AMXBF16）
cd Qwen3-30B-A3B
bash run_full_ft_test_1gpu_bf16.sh --dry-run
bash run_full_ft_test_1gpu_bf16.sh --phase4-steps 15
```

日志与报告写入对应模型目录下的 `test_log/<时间戳>_.../`。

---

## 脚本一览

### Qwen3-30B-A3B

| 脚本 | 模式 | 后端 | GPU | 说明 |
|------|------|------|-----|------|
| `run_full_ft_test_1gpu_bf16.sh` | **真实 Full-FT** | AMXBF16 | 默认 1（OOM 可回退 2） | `ACCELERATE_KT_TRAIN_MODE=full`，测稳定性 / TPS |
| `run_full_ft_test_1gpu_bf16_frozen.sh` | Full-FT 消融 | AMXBF16 | 同上 | 冻结非 expert，只训 expert base buffer |
| `run_fft_test_4gpu_bf16.sh` | FFT 管线验证 | AMXBF16 | 固定 4 | Phase1 + Phase4 |
| `run_fft_test_4gpu_int8.sh` | FFT 管线验证 | AMXINT8 | 固定 4 | 需预量化 INT8 权重 |

#### 真实 Full-FT（推荐入口）

```bash
cd Qwen3-30B-A3B

# 默认：1 GPU，Phase4=15 steps，GAS=1
bash run_full_ft_test_1gpu_bf16.sh

# 常用参数
bash run_full_ft_test_1gpu_bf16.sh \
  --gpus 1 \
  --phase4-steps 15 \
  --gas 1 \
  --dry-run
```

| 参数 | 含义 | 默认 |
|------|------|------|
| `--gpus N` | GPU 数；`N>=2` 时切到 2GPU accelerate 配置 | `1` |
| `--phase4-steps N` | Phase4 训练步数 | `15` |
| `--gas N` / `--gradient-accumulation-steps N` | 梯度累积步数 | `1` |
| `--dry-run` | 只检查环境与配置，不真正训练 | off |

**Expert-only 消融**（验证 expert base 梯度是否生效）：

```bash
bash run_full_ft_test_1gpu_bf16_frozen.sh [--gpus 1] [--phase4-steps 15] [--gas 1] [--dry-run]
```

若只训 expert base 时 `train_loss` 几乎不动，通常说明 expert 全量梯度通路有问题。

#### 4GPU 管线测试

```bash
bash run_fft_test_4gpu_bf16.sh [--skip-phase1] [--skip-phase4] \
  [--only-phase1] [--only-phase4] [--phase4-steps 50] [--dry-run]

bash run_fft_test_4gpu_int8.sh  # 参数同上；需 AMXINT8 权重路径可用
```

阶段说明：

- **Phase 0a/0b**：权重内存拆解、数据集截断分析（始终跑）
- **Phase 1**：短跑（约 3 steps）验证初始化 / 前向 / 反向
- **Phase 4**：稳定性 + TPS（默认 50 steps；前 `WARMUP_SKIP=5` 步不计入 TPS）

---

### GLM-4.5-Air

| 脚本 | 后端 | GPU | 说明 |
|------|------|-----|------|
| `run_full_ft_test_1gpu_bf16.sh` | AMXBF16 | 固定 4（FSDP） | 真实 Full-FT；脚本名含 1gpu，实际为 4 卡 FSDP |
| `run_full_ft_test_4gpu_int8.sh` | AMXINT8 | 固定 4（FSDP） | 在线量化，无需单独 INT8 缓存 |

```bash
cd GLM-4.5-Air

bash run_full_ft_test_1gpu_bf16.sh \
  [--skip-phase4] [--only-phase4] \
  [--gpu 0] [--gpu-ids 0,1,2,3] \
  [--phase4-steps 15] [--dry-run]

bash run_full_ft_test_4gpu_int8.sh   # 参数同上
```

| 参数 | 含义 |
|------|------|
| `--gpu N` | 起始 GPU id，再按 4 卡连续展开 |
| `--gpu-ids a,b,c,d` | 显式指定 4 个 CUDA 设备 |
| `--gpus N` | 仅支持 `4` |
| `--phase4-steps N` | Phase4 步数（默认 15） |
| `--skip-phase4` / `--only-phase4` / `--dry-run` | 阶段控制 / 干跑 |

主要产物：`monitor.csv`、内存拆解文本、`phase4_analysis.txt`、plots。

---

### Qwen3.5-35B-A3B

```bash
cd Qwen3.5-35B-A3B

bash run_fft_test.sh \
  [--skip-phase1] [--skip-phase2] [--skip-phase3] [--skip-phase4] \
  [--only-phase1|2|3|4] \
  [--gpus 4] [--phase4-steps 50] [--dry-run]
```

| 阶段 | 作用 |
|------|------|
| Phase 1 | 基础验证（约 3 steps） |
| Phase 2 | 梯度累积压力（约 8 steps，`accumulation=4`） |
| Phase 3 | 高频保存 I/O（约 6 steps，`save_steps=2`） |
| Phase 4 | 稳定性延伸（默认 50 steps） |

后端配置为 **AMXINT4 + 4 GPU**（见 `configs/accelerate_fft_amxint4_4gpu.yaml`）。

---

## TPS 计算约定

Full-FT / Phase4 脚本统一约定：

- `cutoff_len = 1024`（样本足够长，截断后每步固定长度）
- 丢弃前 `WARMUP_SKIP = 5` 个 step
- 公式：

```text
TPS = NUM_GPUS × CUTOFF_LEN × GAS / avg_stable_step_time
```

---

## 输出目录

每次运行会在模型目录下创建：

```text
test_log/<YYYYMMDD_HHMMSS>_<gpu标签>_<后端>/
├── monitor.csv / monitor.log
├── summary.md
├── phase4/
│   ├── train.log
│   ├── train_config.yaml
│   └── step_timing/          # 部分脚本
└── plots/                    # 若分析脚本生成
```

`test_log/` 体积大且为实验记录，默认由 `.gitignore` 排除。

---

## 常见注意点

1. **必须设置 `ACCELERATE_KT_TRAIN_MODE=full`** 才是真实 expert 全量梯度；仅改 YAML 的 `finetuning_type=full` 仍可能走 fused-expert-LoRA 路径。
2. 训练结束 LLaMA-Factory 可能写出很大的 checkpoint；部分脚本会在阶段结束后删除 `model_output/` 以省磁盘。
3. 路径、模型名、conda 环境均按本机布局硬编码；换机器时先改脚本顶部的 `LLAMA_FACTORY_DIR` / `MODEL_PATH*` / `DATA_DIR`。
4. 建议先 `--dry-run`，确认 Python、GPU、权重路径、accelerate 配置无误后再正式跑。

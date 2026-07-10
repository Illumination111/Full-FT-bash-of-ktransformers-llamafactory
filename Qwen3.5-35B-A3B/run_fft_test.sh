#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B KTransformers 全量微调（FFT）自动化测试脚本
#
# 用法：
#   bash run_fft_test.sh [--skip-phase1] [--skip-phase2] [--skip-phase3] \
#                         [--only-phase4] [--gpus 4] [--dry-run]
#
# 测试阶段：
#   Phase 1 — 基础验证（3 步）：验证 FFT 初始化/前向/反向流程
#   Phase 2 — 梯度累积压力（8 步，accumulation=4）：暴露 P1 梯度覆盖 bug
#   Phase 3 — 高频保存 I/O（6 步，save_steps=2）：暴露 P3 磁盘吞吐瓶颈
#   Phase 4 — 稳定性延伸（50 步）：监控收敛性、MoE 负载、P2/P4/P5/P6/P7
#
# 暴露的关键问题：
#   P1: 梯度覆盖非累加（grad accumulation 失效）
#   P2: backward_base_weight_grad 无向量化，CPU 极慢
#   P3: 模型保存超出磁盘吞吐极限（~61 GB expert 权重）
#   P4: MoE 负载不均衡，路由分布偏斜
#   P5: C++ 梯度索引潜在 bug（越界/NaN）
#   P6: Router 在 full 模式下梯度不稳定
#   P7: update_base_weights() re-quantize 每步开销
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# 路径配置
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${SCRIPT_DIR}/test_log"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze.py"

LLAMA_FACTORY_DIR="/mnt/data2/wbw/LLaMA-Factory"
MODEL_UNFUSED="/mnt/data2/models/Qwen3.5-35B-A3B-Unfused"
MODEL_AMXINT4="/mnt/data2/models/Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH"

ACCEL_CONFIG="${CONFIGS_DIR}/accelerate_fft_amxint4_4gpu.yaml"
TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_fft_qwen35.yaml"

MONITOR_FIFO="/tmp/fft_monitor_events.fifo"
MONITOR_PID=""

# Conda 环境（含 accelerate、kt-kernel、llamafactory 等）
CONDA_ENV="${FFT_CONDA_ENV:-Kllama}"

# 自动探测 conda 环境目录（按优先级查找）
_find_conda_python() {
    local env="$1"
    # 候选路径列表
    local candidates=(
        "/mnt/data2/wbw/conda/envs/${env}/bin/python3"
        "/mnt/data2/wbw/miniconda3/envs/${env}/bin/python3"
        "/opt/conda/envs/${env}/bin/python3"
        "$(conda run -n "${env}" which python3 2>/dev/null || true)"
    )
    for p in "${candidates[@]}"; do
        [[ -x "${p}" ]] && { echo "${p}"; return 0; }
    done
    echo "python3"
}
PYTHON="$(_find_conda_python "${CONDA_ENV}")"
CONDA_BIN_DIR="$(dirname "${PYTHON}")"

# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #
SKIP_PHASE1=0; SKIP_PHASE2=0; SKIP_PHASE3=0; SKIP_PHASE4=0
NUM_GPUS=4
DRY_RUN=0
# Phase 4 step count:
#   50 steps gives enough loss / grad-norm data points to assess convergence
#   and MoE stability, while also exercising P2/P7 pressure for meaningful
#   timing data. Each step can take 2-60+ min (P2 issue), so 50 steps is
#   intentionally a long-running stress test. Use --phase4-steps N to shorten.
PHASE4_STEPS=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-phase1) SKIP_PHASE1=1 ;;
        --skip-phase2) SKIP_PHASE2=1 ;;
        --skip-phase3) SKIP_PHASE3=1 ;;
        --skip-phase4) SKIP_PHASE4=1 ;;
        --only-phase1) SKIP_PHASE2=1; SKIP_PHASE3=1; SKIP_PHASE4=1 ;;
        --only-phase2) SKIP_PHASE1=1; SKIP_PHASE3=1; SKIP_PHASE4=1 ;;
        --only-phase3) SKIP_PHASE1=1; SKIP_PHASE2=1; SKIP_PHASE4=1 ;;
        --only-phase4) SKIP_PHASE1=1; SKIP_PHASE2=1; SKIP_PHASE3=1 ;;
        --gpus) NUM_GPUS="$2"; shift ;;
        --phase4-steps) PHASE4_STEPS="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# --------------------------------------------------------------------------- #
# 颜色输出
# --------------------------------------------------------------------------- #
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()    { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC}  $*"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*" >&2; }
phase_banner() {
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
    echo ""
}

# --------------------------------------------------------------------------- #
# 环境检查
# --------------------------------------------------------------------------- #
check_env() {
    phase_banner "环境检查"

    # Python
    log "Python 版本: $(python3 --version 2>&1)"

    # GPU
    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi 未找到，请确认 CUDA 驱动已安装"
        exit 1
    fi
    ACTUAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    log "检测到 GPU 数量: ${ACTUAL_GPUS}"
    if [[ "${ACTUAL_GPUS}" -lt "${NUM_GPUS}" ]]; then
        error "系统 GPU 数量 (${ACTUAL_GPUS}) 少于请求的 ${NUM_GPUS}"
        exit 1
    fi

    # 模型路径
    for path in "${MODEL_UNFUSED}" "${MODEL_AMXINT4}"; do
        if [[ ! -d "${path}" ]]; then
            error "模型路径不存在: ${path}"
            exit 1
        fi
        ok "模型路径存在: ${path}"
    done

    # LLaMA-Factory
    if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
        error "LLaMA-Factory 目录不存在: ${LLAMA_FACTORY_DIR}"
        exit 1
    fi
    ok "LLaMA-Factory: ${LLAMA_FACTORY_DIR}"

    # conda 环境
    log "Python 可执行文件: ${PYTHON}"
    log "Conda 环境: ${CONDA_ENV}"

    # accelerate
    if ! "${PYTHON}" -c "import accelerate" &>/dev/null; then
        error "accelerate 未在 ${PYTHON} 中安装"
        error "请激活 conda 环境: conda activate ${CONDA_ENV}"
        exit 1
    fi
    ok "accelerate $(${PYTHON} -c 'import accelerate; print(accelerate.__version__)')"

    # psutil / pynvml / matplotlib / pandas
    for pkg in psutil pynvml matplotlib pandas; do
        if "${PYTHON}" -c "import ${pkg}" &>/dev/null; then
            ok "${pkg} 已安装"
        else
            warn "${pkg} 未安装（部分功能受限），建议: ${PYTHON} -m pip install ${pkg}"
        fi
    done

    # 磁盘空间（需要至少 200 GB 用于 checkpoint）
    AVAIL_GB=$(df -BG "${LOG_BASE%/*}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    log "日志目录可用空间: ${AVAIL_GB:-未知} GB"
    if [[ -n "${AVAIL_GB}" ]] && [[ "${AVAIL_GB}" -lt 150 ]]; then
        warn "可用磁盘空间较少 (${AVAIL_GB} GB)，Phase 3 checkpoint 保存可能失败"
    fi

    ok "环境检查通过"
}

# --------------------------------------------------------------------------- #
# 资源与耗时估算（Qwen3.5-35B-A3B 全量微调）
# --------------------------------------------------------------------------- #
estimate_resources() {
    phase_banner "资源与耗时估算 (Qwen3.5-35B-A3B FFT)"

    echo -e "${BOLD}== 模型结构 ==${NC}"
    echo "  架构    : Qwen3_5MoeForConditionalGeneration"
    echo "  层数    : 40   |  Experts/层: 256  |  活跃 Expert/token: 8"
    echo "  hidden  : 2048  |  MoE intermediate: 512  |  Head dim: 256"
    echo "  注意力  : 混合 linear/full attention (每 4 层一次 full)"
    echo "  词表    : 248320  |  最大长度: 262144  |  测试序列: 512"
    echo ""

    echo -e "${BOLD}== GPU 显存估算 (4 × RTX 4090, 24 GB each) ==${NC}"
    echo "  Expert 权重由 CPU AMX 内核处理，GPU 只保存非 Expert 参数："
    echo "  · Attention (40层 × ~36 MB): ~1.4 GB"
    echo "  · Embedding + LM Head: ~1.0 GB"
    echo "  · Shared expert (每层 1 个): ~0.3 GB"
    echo "  · Norm 等杂项: ~0.2 GB"
    echo "  FSDP 分片后每 GPU 约: 1~2 GB (静态参数)"
    echo "  前向激活 (batch=1, seq=512): ~2~4 GB / GPU"
    echo "  梯度 + optimizer state (非 Expert): ~2~3 GB / GPU"
    echo "  ──────────────────────────────────────────────"
    echo "  预估 GPU 峰值显存 / 卡: ~5~10 GB  (远低于 24 GB 上限)"
    echo "  ⚠ 注意: FSDP FULL_STATE_DICT 保存时 rank-0 将聚合全部参数，"
    echo "    可能瞬间增加到 ~5 GB 额外显存 (Phase 3)"
    echo ""

    echo -e "${BOLD}== CPU 内存估算 ==${NC}"
    echo "  AMXINT4 量化 Expert 权重 (已预转换): ~22 GB"
    echo "  BF16 Expert base weight buffers (full 模式 nn.Parameter):"
    echo "    256 experts × 3 proj × (512×2048) × 2 bytes × 40 层"
    echo "    = 256 × 3 × 2,097,152 × 2 × 40 / 1e9 ≈ 61 GB"
    echo "  BF16 Expert 梯度 buffer (grad_*_proj_buf): ~61 GB"
    echo "  AdamW optimizer states (m + v for experts): ~61 × 2 = 122 GB"
    echo "  ──────────────────────────────────────────────"
    echo "  Expert 相关 CPU 内存合计: ~266 GB"
    echo "  非 Expert 参数 + 系统开销: ~20~30 GB"
    echo "  ⚠ 预估 CPU 内存峰值: ~290~300 GB"
    echo "  ⚠ Phase 3 checkpoint gather 时峰值可能更高 (~350+ GB)"
    echo "  ⚠ 建议系统 RAM >= 384 GB；若 < 256 GB 将在 full 模式初始化时 OOM"
    echo ""

    echo -e "${BOLD}== 磁盘空间估算 (Phase 3 高频保存) ==${NC}"
    echo "  单次 checkpoint 写入量 (FULL_STATE_DICT):"
    echo "    · BF16 Expert weights: ~61 GB"
    echo "    · Non-expert model weights: ~3 GB"
    echo "    · AdamW optimizer state: ~64 × 2 = ~128 GB"
    echo "    · 合计约: ~190~250 GB / checkpoint"
    echo "  Phase 3 触发 3 次保存: ~570~750 GB"
    echo "  Phase 4 通常只保存 1 次 (save_steps=500): ~250 GB"
    echo "  ──────────────────────────────────────────────"
    echo "  建议预留磁盘空间: >= 1 TB"
    echo "  当前可用空间: $(df -BG "${LOG_BASE%/*}" 2>/dev/null | tail -1 | awk '{print $4}' || echo '未知')"
    echo ""

    echo -e "${BOLD}== 耗时估算 ==${NC}"
    echo "  模型加载 & 初始化 (AMXINT4 权重 ~22 GB):"
    echo "    · 磁盘读取 22 GB @ 1-3 GB/s: ~7~22 秒"
    echo "    · AMX 内核初始化 + FSDP: ~2~5 分钟"
    echo "  每步耗时 (seq=512, batch=1, P2 问题):"
    echo "    · Forward (AMX forward_sft): ~10~30 秒"
    echo "    · Backward (backward_base_weight_grad, 无向量化!):"
    echo "      朴素三重循环: 256 experts × 40 层 × (512×2048) = 极大计算量"
    echo "      ⚠ 保守估算: 5~60 分钟/步  (这正是 P2 要暴露的瓶颈)"
    echo "    · update_base_weights (re-quantize, ~0.6s/层 × 40): ~24 秒"
    echo "    · Total per step: 6 分钟 (乐观) ~ 90 分钟 (悲观)"
    echo "  ──────────────────────────────────────────────"
    local p4_opt=$(echo "scale=0; ${PHASE4_STEPS} * 6" | bc 2>/dev/null || echo "N/A")
    local p4_pes=$(echo "scale=0; ${PHASE4_STEPS} * 90" | bc 2>/dev/null || echo "N/A")
    echo "  Phase 1 (3 步):        0.5~5 小时"
    echo "  Phase 2 (12 步):       1~18 小时"
    echo "  Phase 3 (6 步 + ckpt): 3~12 小时  (ckpt 保存是主要瓶颈)"
    echo "  Phase 4 (${PHASE4_STEPS} 步):      ${p4_opt}~${p4_pes} 分钟"
    echo ""
    echo "  ⚠ 全套测试预计总耗时: 1 天 ~ 数天"
    echo "  💡 建议先用 --only-phase1 验证基础可行性"
    echo "  💡 可用 --phase4-steps 20 缩短 Phase 4 (当前: ${PHASE4_STEPS} 步)"
    echo "  💡 可用 --skip-phase3 跳过磁盘 I/O 测试以节省时间"
    echo ""
}

# --------------------------------------------------------------------------- #
# 创建日期目录 + 事件 FIFO
# --------------------------------------------------------------------------- #
setup_run_dir() {
    RUN_TS=$(date '+%Y%m%d_%H%M%S')
    LOG_DIR="${LOG_BASE}/${RUN_TS}"
    mkdir -p "${LOG_DIR}/plots"
    log "日志目录: ${LOG_DIR}"

    # 创建 FIFO（用于向 monitor.py 发送事件）
    [[ -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
    mkfifo "${MONITOR_FIFO}"

    export LOG_DIR RUN_TS MONITOR_FIFO
}

# --------------------------------------------------------------------------- #
# 向监控 FIFO 发送事件（非阻塞）
# --------------------------------------------------------------------------- #
send_event() {
    local msg="$1"
    if [[ -p "${MONITOR_FIFO}" ]]; then
        echo "${msg}" >> "${MONITOR_FIFO}" 2>/dev/null || true
    fi
}

# --------------------------------------------------------------------------- #
# 启动 / 停止监控
# --------------------------------------------------------------------------- #
start_monitor() {
    log "启动系统监控进程..."
    "${PYTHON}" "${MONITOR_SCRIPT}" \
        --out "${LOG_DIR}/monitor.csv" \
        --fifo "${MONITOR_FIFO}" \
        --interval 2 \
        --disk-mount /mnt/data2 \
        >> "${LOG_DIR}/monitor.log" 2>&1 &
    MONITOR_PID=$!
    log "监控进程 PID: ${MONITOR_PID}"
    sleep 1
}

stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        log "停止监控进程 (PID: ${MONITOR_PID})..."
        kill -TERM "${MONITOR_PID}" || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

# --------------------------------------------------------------------------- #
# 构造训练配置临时副本
# --------------------------------------------------------------------------- #
# 参数：$1=phase名, 其余=$2+ 为 key=value 替换对
make_phase_config() {
    local phase_name="$1"; shift
    local phase_dir="${LOG_DIR}/${phase_name}"
    mkdir -p "${phase_dir}"
    local cfg="${phase_dir}/train_config.yaml"

    # 先复制基础配置
    cp "${TRAIN_CONFIG_BASE}" "${cfg}"

    # 替换 output_dir
    sed -i "s|output_dir: .*|output_dir: ${phase_dir}/model_output|g" "${cfg}"

    # 处理额外 key=value 替换
    while [[ $# -gt 0 ]]; do
        local kv="$1"; shift
        local key="${kv%%=*}"
        local val="${kv#*=}"
        # 若 key 存在则替换，否则追加
        if grep -q "^${key}:" "${cfg}"; then
            sed -i "s|^${key}: .*|${key}: ${val}|g" "${cfg}"
        else
            echo "${key}: ${val}" >> "${cfg}"
        fi
    done

    echo "${cfg}"
}

# --------------------------------------------------------------------------- #
# 运行单个 accelerate 训练命令
# --------------------------------------------------------------------------- #
# 参数：$1=phase名 $2=步数描述(用于日志) $3=train_config
run_train() {
    local phase_name="$1"
    local desc="$2"
    local train_cfg="$3"
    local phase_log="${LOG_DIR}/${phase_name}/train.log"
    local exit_code=0

    log "启动训练 [${phase_name}]: ${desc}"
    send_event "phase:${phase_name}"
    send_event "event:train_start"

    # 构造 GPU 可见性
    local gpus_str=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',')

    # 使用 conda 环境中的 accelerate
    local accelerate_bin="${CONDA_BIN_DIR}/accelerate"
    [[ ! -x "${accelerate_bin}" ]] && accelerate_bin="accelerate"

    local cmd=(
        env
        USE_KT=1
        ACCELERATE_USE_KT=true
        CUDA_VISIBLE_DEVICES="${gpus_str}"
        "${accelerate_bin}" launch
        --config_file "${ACCEL_CONFIG}"
        -m llamafactory.cli train
        "${train_cfg}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "[DRY-RUN] 命令: ${cmd[*]}"
        log "[DRY-RUN] 训练配置: $(cat "${train_cfg}")"
        return 0
    fi

    # 在 LLaMA-Factory 目录中运行（确保相对路径数据集可找到）
    pushd "${LLAMA_FACTORY_DIR}" > /dev/null
    set +e
    "${cmd[@]}" 2>&1 | tee "${phase_log}"
    exit_code=${PIPESTATUS[0]}
    set -e
    popd > /dev/null

    send_event "event:train_end"
    return "${exit_code}"
}

# --------------------------------------------------------------------------- #
# 分析训练日志（提取 loss、grad_norm、NaN 检测）
# --------------------------------------------------------------------------- #
analyze_log() {
    local phase_name="$1"
    local log_file="${LOG_DIR}/${phase_name}/train.log"
    local result_file="${LOG_DIR}/${phase_name}/log_analysis.txt"

    [[ ! -f "${log_file}" ]] && return

    {
        echo "=== Phase: ${phase_name} 日志分析 ==="
        echo "--- 训练 Loss（最后 20 条）---"
        grep -i "loss" "${log_file}" | tail -20 || echo "(未找到 loss 记录)"

        echo ""
        echo "--- 梯度范数（所有记录）---"
        grep -i "grad_norm\|gradient_norm" "${log_file}" | tail -30 || echo "(未找到 grad_norm 记录)"

        echo ""
        echo "--- NaN / Inf 出现次数 ---"
        local nan_count inf_count
        nan_count=$(grep -ci "nan" "${log_file}" 2>/dev/null || echo 0)
        inf_count=$(grep -ci "inf" "${log_file}" 2>/dev/null || echo 0)
        echo "NaN 出现: ${nan_count} 行; Inf 出现: ${inf_count} 行"

        echo ""
        echo "--- KT 相关警告/错误 ---"
        grep -i "ktransformer\|kt_kernel\|amx\|moe\|expert\|backward\|grad_proj" \
             "${log_file}" 2>/dev/null | grep -i "warn\|error\|fail\|bug" | tail -20 \
             || echo "(未发现 KT 相关告警)"

        echo ""
        echo "--- update_base_weights 调用时序（P7：re-quantize 耗时）---"
        grep -i "update_base_weights\|re-quantize\|set_base_weight" "${log_file}" | tail -20 \
             || echo "(未检测到 update_base_weights 日志)"

        echo ""
        echo "--- checkpoint 保存事件（P3：磁盘 I/O）---"
        grep -i "saving\|checkpoint\|model_output" "${log_file}" | tail -20 \
             || echo "(未检测到 checkpoint 保存日志)"

        echo ""
        echo "--- 进程退出码分析 ---"
        if grep -qi "segmentation fault\|sigsegv\|core dumped" "${log_file}"; then
            echo "⚠ 检测到 SIGSEGV / core dump（P5: C++ 梯度索引越界）"
        fi
        if grep -qi "cuda out of memory\|oom" "${log_file}"; then
            echo "⚠ 检测到 CUDA OOM"
        fi
        if grep -qi "ddp_timeout\|timeout expired" "${log_file}"; then
            echo "⚠ 检测到 DDP 超时（P2/P7: CPU 计算过慢）"
        fi
    } | tee "${result_file}"
}

# --------------------------------------------------------------------------- #
# Phase 1：基础验证（3 步）
# 目标：验证 FFT 全流程初始化 → forward → backward → optimizer step 能否跑通
# 暴露：P5（C++ 越界崩溃）、P6（Router 梯度爆炸导致 NaN）
# --------------------------------------------------------------------------- #
run_phase1() {
    phase_banner "Phase 1: 基础验证（3 步）"
    local cfg
    cfg=$(make_phase_config "phase1" \
        "max_steps=3" \
        "save_steps=500" \
        "gradient_accumulation_steps=1" \
        "logging_steps=1")

    log "测试目标: 验证 FFT 流程能否不崩溃地完成 3 个训练步"
    log "配置文件: ${cfg}"
    log "暴露问题: P5（C++ 梯度索引越界）、P6（Router 梯度不稳定）"

    local exit_code=0
    run_train "phase1" "3步基础验证" "${cfg}" || exit_code=$?

    echo ""
    if [[ "${exit_code}" -eq 0 ]]; then
        ok "Phase 1 完成，退出码 0"
    else
        error "Phase 1 失败，退出码 ${exit_code}"
        # 分类错误类型
        local log_f="${LOG_DIR}/phase1/train.log"
        if [[ -f "${log_f}" ]]; then
            if grep -qi "segmentation fault\|sigsegv" "${log_f}"; then
                error "→ 确认 P5: C++ 梯度索引越界导致 SIGSEGV"
            elif grep -qi "nan\|inf" "${log_f}"; then
                error "→ 疑似 P6: Router 或 Expert 梯度出现 NaN/Inf"
            elif grep -qi "out of memory" "${log_f}"; then
                error "→ GPU/CPU OOM：full 模式内存占用超限"
            fi
        fi
    fi
    analyze_log "phase1"
    echo "${exit_code}" > "${LOG_DIR}/phase1/exit_code.txt"
    return "${exit_code}"
}

# --------------------------------------------------------------------------- #
# Phase 2：梯度累积压力测试（8 步，accumulation=4）
# 目标：暴露 P1（梯度覆盖）—— 期望：accum=4 时梯度范数 ≈ 单步的同一量级
#       如果 bug 存在，accum=4 时的梯度范数等于最后一个 micro-batch 的梯度，
#       比 accum=1 基准小约 4× 且不稳定
# 辅助：对比 accum=1 基准（从 Phase 1 结果获取）
# --------------------------------------------------------------------------- #
run_phase2() {
    phase_banner "Phase 2: 梯度累积压力测试（8 步，accumulation=4）"

    # 子测试 2a：accumulation=1 基准（4 步）
    log "Phase 2a: accumulation=1 基准（4 步）"
    local cfg_2a
    cfg_2a=$(make_phase_config "phase2a" \
        "max_steps=4" \
        "gradient_accumulation_steps=1" \
        "save_steps=500" \
        "logging_steps=1")
    run_train "phase2a" "accum=1 基准(4步)" "${cfg_2a}" || true
    analyze_log "phase2a"

    # 子测试 2b：accumulation=4（8 步）
    log "Phase 2b: accumulation=4（8 步）"
    local cfg_2b
    cfg_2b=$(make_phase_config "phase2b" \
        "max_steps=8" \
        "gradient_accumulation_steps=4" \
        "save_steps=500" \
        "logging_steps=1")

    log "测试目标: 对比 accumulation=1 vs 4 时的梯度范数"
    log "暴露问题: P1（C++ backward 梯度覆盖非累加）"
    run_train "phase2b" "accum=4 压力(8步)" "${cfg_2b}" || true
    analyze_log "phase2b"

    # 对比分析
    {
        echo "=== Phase 2 梯度累积对比分析 ==="
        echo ""
        echo "--- Phase 2a (accum=1) 梯度范数 ---"
        grep -i "grad_norm" "${LOG_DIR}/phase2a/train.log" 2>/dev/null \
            | awk -F'grad_norm' '{print "  " NR ": " $2}' | head -20 || echo "  (无数据)"
        echo ""
        echo "--- Phase 2b (accum=4) 梯度范数 ---"
        grep -i "grad_norm" "${LOG_DIR}/phase2b/train.log" 2>/dev/null \
            | awk -F'grad_norm' '{print "  " NR ": " $2}' | head -20 || echo "  (无数据)"
        echo ""
        echo "诊断说明："
        echo "  若 accum=4 的梯度范数比 accum=1 小约 4× → 确认 P1: 梯度被覆盖而非累加"
        echo "  若梯度范数相近 → P1 bug 在此版本已修复或未触发"
    } | tee "${LOG_DIR}/phase2_comparison.txt"

    ok "Phase 2 完成"
}

# --------------------------------------------------------------------------- #
# Phase 3：高频保存 I/O 测试（6 步，save_steps=2）
# 目标：暴露 P3（保存 ~61 GB expert 权重时超出磁盘吞吐）
#       FSDP FULL_STATE_DICT 会在保存时聚合所有 expert BF16 权重到 rank 0
# 期望：磁盘写速峰值 >> 持续写速，保存耗时 >> 训练步耗时
# --------------------------------------------------------------------------- #
run_phase3() {
    phase_banner "Phase 3: 高频保存 I/O 测试（6 步，save_steps=2）"
    local cfg
    cfg=$(make_phase_config "phase3" \
        "max_steps=6" \
        "gradient_accumulation_steps=1" \
        "save_steps=2" \
        "save_only_model=false" \
        "logging_steps=1")

    log "测试目标: 触发 3 次 checkpoint 保存，监控磁盘写入速率"
    log "暴露问题: P3（保存 ~61 GB Expert 权重超出磁盘吞吐极限）"
    log "注意: 此 phase 可能耗时极长（磁盘 I/O 瓶颈），请等待..."
    warn "预计单次保存耗时 5~60 分钟取决于磁盘速度"

    send_event "phase:phase3"
    send_event "event:phase3_start"

    local exit_code=0
    run_train "phase3" "高频保存(6步,save_steps=2)" "${cfg}" || exit_code=$?

    if [[ "${exit_code}" -ne 0 ]]; then
        local log_f="${LOG_DIR}/phase3/train.log"
        if [[ -f "${log_f}" ]]; then
            if grep -qi "timeout\|deadline exceeded" "${log_f}"; then
                error "→ 确认 P3: checkpoint 保存超时（DDP 超时）"
            elif grep -qi "no space\|disk full\|enospc" "${log_f}"; then
                error "→ 确认 P3: 磁盘空间不足，保存失败"
            elif grep -qi "killed\|oom" "${log_f}"; then
                error "→ 保存时 OOM（RAM 不足以聚合全部权重）"
            fi
        fi
    fi
    analyze_log "phase3"
    echo "${exit_code}" > "${LOG_DIR}/phase3/exit_code.txt"

    # 从 monitor.csv 提取保存期间的磁盘写速峰值
    {
        echo "=== Phase 3 磁盘 I/O 分析 ==="
        if [[ -f "${LOG_DIR}/monitor.csv" ]]; then
            python3 - <<'EOF'
import csv, sys
rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r.get("phase","").startswith("phase3"):
            try:
                rows.append(float(r.get("disk_write_mbps", 0)))
            except ValueError:
                pass
if rows:
    print(f"  Phase 3 disk write -- peak: {max(rows):.1f} MB/s  mean: {sum(rows)/len(rows):.1f} MB/s")
    print(f"  samples: {len(rows)}")
else:
    print("  (no phase3 disk data)")
EOF
            "${LOG_DIR}/monitor.csv"
        else
            echo "  monitor.csv not found"
        fi
    } | tee "${LOG_DIR}/phase3_io_summary.txt"
}

# --------------------------------------------------------------------------- #
# Phase 4：稳定性延伸测试（50 步）
# 目标：系统观测 50 步内的：
#   P2  CPU backward 耗时（步均时间 >> LoRA 模式）
#   P4  MoE 路由负载分布（从 loss 辅助损失推断）
#   P5  NaN/Inf 是否出现
#   P6  Router loss 趋势
#   P7  每步 update_base_weights 频率（日志中的 re-quantize 记录）
# --------------------------------------------------------------------------- #
run_phase4() {
    phase_banner "Phase 4: 稳定性延伸测试（${PHASE4_STEPS} 步）"
    local cfg
    cfg=$(make_phase_config "phase4" \
        "max_steps=${PHASE4_STEPS}" \
        "gradient_accumulation_steps=1" \
        "save_steps=500" \
        "logging_steps=1")

    log "步数: ${PHASE4_STEPS}（可用 --phase4-steps N 调整）"
    log "测试目标: 观测 ${PHASE4_STEPS} 步内的训练稳定性和性能特征"
    log "暴露问题: P2（CPU 慢）、P4（MoE 负载）、P5（NaN）、P6（Router）、P7（re-quantize）"
    warn "P2 预期: 每步可能耗时 6~90 分钟（backward_base_weight_grad 无向量化）"

    # 记录开始时间
    local t_start
    t_start=$(date +%s)
    send_event "phase:phase4"

    local exit_code=0
    run_train "phase4" "稳定性测试(50步)" "${cfg}" || exit_code=$?

    local t_end
    t_end=$(date +%s)
    local total_sec=$((t_end - t_start))

    {
        echo "=== Phase 4 性能分析 ==="
        echo "总耗时: ${total_sec} 秒"
        local actual_steps
        actual_steps=$(grep -c "{'loss'" "${LOG_DIR}/phase4/train.log" 2>/dev/null || \
                       grep -c '"loss"' "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        if [[ "${actual_steps}" -gt 0 ]]; then
            local avg_sec=$(echo "scale=1; ${total_sec} / ${actual_steps}" | bc 2>/dev/null || echo "N/A")
            echo "steps_completed: ${actual_steps} (expected ${PHASE4_STEPS})"
            echo "avg_sec_per_step: ${avg_sec}"
            echo ""
            echo "P2 diagnosis (CPU backward speed):"
            echo "  avg_sec > 120s  => backward_base_weight_grad is a serious bottleneck (no vectorization)"
            echo "  avg_sec > 300s  => DDP timeout risk"
        fi

        echo ""
        echo "--- MoE router aux loss (P4: load balance) ---"
        grep -i "aux_loss\|router_loss\|balance_loss\|load_balance" \
             "${LOG_DIR}/phase4/train.log" 2>/dev/null | tail -20 || echo "  (no router aux loss found)"

        echo ""
        echo "--- Router parameter norm trend (P6) ---"
        grep -i "router\|gate.*norm\|norm.*gate" \
             "${LOG_DIR}/phase4/train.log" 2>/dev/null | tail -10 || echo "  (no router norm records)"

        echo ""
        echo "--- update_base_weights call count (P7) ---"
        local upd_count
        upd_count=$(grep -ci "update_base_weights\|re-quantize\|syncing updated" \
                    "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "  update_base_weights triggered: ${upd_count} times"

        echo ""
        echo "--- NaN/Inf statistics (P5) ---"
        local nan_cnt inf_cnt
        nan_cnt=$(grep -ci " nan" "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        inf_cnt=$(grep -ci " inf" "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "  NaN lines: ${nan_cnt};  Inf lines: ${inf_cnt}"
        if [[ "${nan_cnt}" -gt 0 ]] || [[ "${inf_cnt}" -gt 0 ]]; then
            warn "=> numerical anomaly detected, check P5 (C++ grad bug) and P6 (Router grad explosion)"
        fi
    } | tee "${LOG_DIR}/phase4_analysis.txt"

    analyze_log "phase4"
    echo "${exit_code}" > "${LOG_DIR}/phase4/exit_code.txt"
}

# --------------------------------------------------------------------------- #
# 汇总报告
# --------------------------------------------------------------------------- #
generate_summary() {
    phase_banner "生成汇总报告"
    local summary="${LOG_DIR}/SUMMARY.md"
    {
        echo "# Qwen3.5-35B-A3B FFT 测试汇总"
        echo ""
        echo "**测试时间**: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "**日志目录**: ${LOG_DIR}"
        echo "**GPU 数量**: ${NUM_GPUS}"
        echo ""
        echo "## 各 Phase 退出码"
        echo ""
        echo "| Phase | 描述 | 退出码 | 状态 |"
        echo "|-------|------|--------|------|"

        for ph in phase1 phase2a phase2b phase3 phase4; do
            local ec_file="${LOG_DIR}/${ph}/exit_code.txt"
            if [[ -f "${ec_file}" ]]; then
                local ec
                ec=$(cat "${ec_file}")
                local status="✓ 通过"
                [[ "${ec}" -ne 0 ]] && status="✗ 失败(${ec})"
                echo "| ${ph} | - | ${ec} | ${status} |"
            else
                echo "| ${ph} | - | - | 跳过 |"
            fi
        done

        echo ""
        echo "## 关键问题检测结果"
        echo ""
        echo "| 编号 | 问题描述 | 状态 |"
        echo "|------|----------|------|"

        # P1 梯度覆盖
        local p1_status="待分析（运行 analyze.py）"
        echo "| P1 | 梯度累积覆盖 bug | ${p1_status} |"

        # P2 CPU 慢
        local p2_status="待分析"
        if [[ -f "${LOG_DIR}/phase4_analysis.txt" ]]; then
            p2_status=$(grep "步均耗时" "${LOG_DIR}/phase4_analysis.txt" 2>/dev/null | head -1 || echo "待分析")
        fi
        echo "| P2 | CPU backward 速度瓶颈 | ${p2_status} |"

        # P3 磁盘
        local p3_status="待分析"
        if [[ -f "${LOG_DIR}/phase3_io_summary.txt" ]]; then
            p3_status=$(grep "峰值" "${LOG_DIR}/phase3_io_summary.txt" 2>/dev/null | head -1 || echo "待分析")
        fi
        echo "| P3 | 磁盘吞吐瓶颈 | ${p3_status} |"

        # P4 MoE 负载
        echo "| P4 | MoE 路由负载均衡 | 见 phase4_analysis.txt |"

        # P5 NaN
        local p5_status="正常"
        for ph in phase1 phase2a phase2b phase3 phase4; do
            local lf="${LOG_DIR}/${ph}/train.log"
            if [[ -f "${lf}" ]] && grep -qi "nan\|sigsegv\|segmentation" "${lf}"; then
                p5_status="⚠ 检测到 NaN 或崩溃"
                break
            fi
        done
        echo "| P5 | C++ 梯度索引 bug / NaN | ${p5_status} |"

        # P6 Router
        echo "| P6 | Router 梯度稳定性 | 见 analyze.py 梯度范数图 |"

        # P7 re-quantize
        echo "| P7 | update_base_weights 耗时 | 见 phase4_analysis.txt |"

        echo ""
        echo "## 后续分析"
        echo ""
        echo '```bash'
        echo "# 生成可视化图表："
        echo "python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
        echo '```'
        echo ""
        echo "生成图表位置: \`${LOG_DIR}/plots/\`"
    } > "${summary}"

    log "汇总报告: ${summary}"
    cat "${summary}"
}

# --------------------------------------------------------------------------- #
# 清理
# --------------------------------------------------------------------------- #
cleanup() {
    stop_monitor
    [[ -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
}

trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
main() {
    echo ""
    echo -e "${BOLD}Qwen3.5-35B-A3B KTransformers 全量微调测试套件${NC}"
    echo -e "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    check_env
    estimate_resources
    setup_run_dir
    start_monitor

    local overall_exit=0

    # Phase 1
    if [[ "${SKIP_PHASE1}" -eq 0 ]]; then
        if ! run_phase1; then
            warn "Phase 1 失败，后续 Phase 仍继续执行（用于全面暴露问题）"
            overall_exit=1
        fi
    else
        log "Phase 1 已跳过"
    fi

    # Phase 2
    if [[ "${SKIP_PHASE2}" -eq 0 ]]; then
        run_phase2 || overall_exit=1
    else
        log "Phase 2 已跳过"
    fi

    # Phase 3
    if [[ "${SKIP_PHASE3}" -eq 0 ]]; then
        run_phase3 || overall_exit=1
    else
        log "Phase 3 已跳过"
    fi

    # Phase 4
    if [[ "${SKIP_PHASE4}" -eq 0 ]]; then
        run_phase4 || overall_exit=1
    else
        log "Phase 4 已跳过"
    fi

    stop_monitor

    generate_summary

    # 触发可视化分析
    if [[ -f "${ANALYZE_SCRIPT}" ]]; then
        log "运行可视化分析..."
        "${PYTHON}" "${ANALYZE_SCRIPT}" --log-dir "${LOG_DIR}" \
            >> "${LOG_DIR}/analyze.log" 2>&1 && \
            ok "可视化图表已生成: ${LOG_DIR}/plots/" || \
            warn "analyze.py 执行失败，请手动运行: python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
    fi

    echo ""
    if [[ "${overall_exit}" -eq 0 ]]; then
        ok "全部 Phase 完成，日志目录: ${LOG_DIR}"
    else
        warn "部分 Phase 存在问题，请查看: ${LOG_DIR}/SUMMARY.md"
    fi

    return "${overall_exit}"
}

main "$@"

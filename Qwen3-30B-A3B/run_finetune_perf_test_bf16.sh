#!/usr/bin/env bash
# =============================================================================
# Qwen3-30B-A3B KTransformers Fine-Tuning Performance Test (Full / LoRA)
#
# Full and LoRA runs use the same dataset, batch size, GAS, GPU topology,
# learning rate, max steps and post-warmup TPS interval.  Only the fine-tuning
# method and LoRA-specific rank/alpha settings differ.  In `both` mode the
# full-weight run always completes before the LoRA run starts.
#
# Usage:
#   bash run_finetune_perf_test_bf16.sh [--mode both|full|lora]
#        [--gpus 1] [--batch-size 1] [--gas 1]
#        [--steps 15] [--warmup-steps 5] [--dry-run]
#
#   --gas N   gradient_accumulation_steps (default: 1; e.g. --gas 16)
#
# Test phase:
#   Performance run — measure TPS, forward/backward/optimizer timing and
#                     runtime GPU/RAM usage.  Warmup steps are excluded from
#                     every stable timing and FLOPs sanity-check metric.
#
# TPS Measurement (Phase 4):
#   Accurate TPS is computed by skipping WARMUP_SKIP steps (default: 5) and
#   using exactly cutoff_len=4096 tokens per micro-batch (all samples hit cutoff).
#   One optimizer step processes GAS micro-batches per GPU.
#   Formula: TPS = NUM_GPUS * BATCH_SIZE * CUTOFF_LEN * GAS
#                  / avg_stable_step_time
# =============================================================================

set -euo pipefail

# Keep run-directory names and all child-process logs in the expected local time.
export TZ="${FFT_TIMEZONE:-Asia/Shanghai}"

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
# Shared benchmark dataset directory (not the per-model data/ directory).
DATA_DIR="/mnt/data2/wbw/FFTtest/dataset"
DATASET_NAME="fft_real_100"
GEN_DATASET_SCRIPT="${DATA_DIR}/gen_dataset.py"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze.py"
FLOPS_ANALYZE_SCRIPT="${SCRIPT_DIR}/flops_timing_analysis.py"

LLAMA_FACTORY_DIR="/mnt/data2/wbw/LLaMA-Factory"
MODEL_PATH="/mnt/data3/models/Qwen3-30B-A3B"    # Original HF BF16 weights (AMXBF16 reads directly)

# The selected accelerate config is shared by both fine-tuning methods.
ACCEL_CONFIG_1GPU="${CONFIGS_DIR}/accelerate_fft_amxbf16_1gpu.yaml"
ACCEL_CONFIG_2GPU="${CONFIGS_DIR}/accelerate_fft_amxbf16_2gpu.yaml"
ACCEL_CONFIG_4GPU="${CONFIGS_DIR}/accelerate_fft_amxbf16_4gpu.yaml"
ACCEL_CONFIG="${ACCEL_CONFIG_1GPU}"

TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_finetune_perf_qwen3_30b.yaml"

# TPS measurement constants (must match train config)
CUTOFF_LEN=4096
WARMUP_SKIP=5
PHASE4_STEPS=15
TRAIN_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
LEARNING_RATE="1.0e-5"

# Backward internal timing: summary keeps only per-step aggregates; trace also
# retains per-layer/per-NUMA rows.  AMX tile/task internals remain opaque.
BACKWARD_TIMING_MODE="${FFT_BACKWARD_TIMING:-summary}"
BACKWARD_TIMING_LAYERS="${FFT_BACKWARD_TIMING_LAYERS:-all}"

# LoRA-only settings. `lora_target=all` is fixed so the FLOPs model matches the
# actual adapter placement (all Linear modules except lm_head).
LORA_RANK=8
LORA_ALPHA=16

# Roofline assumptions.  Defaults match RTX 4090 BF16 Tensor Core peak and the
# 2x48-core Xeon 8488C AMX / 16-channel DDR5 host used by this test.
GPU_BF16_TFLOPS="${GPU_BF16_TFLOPS:-82.58}"
CPU_BF16_TFLOPS="${CPU_BF16_TFLOPS:-373.56}"
GPU_MEMORY_GBPS="${GPU_MEMORY_GBPS:-1008.0}"
CPU_MEMORY_GBPS="${CPU_MEMORY_GBPS:-614.4}"

MONITOR_FIFO=""
MONITOR_PID=""

# Conda environment
CONDA_ENV="${FFT_CONDA_ENV:-Kllama}"

_find_conda_python() {
    local env="$1"
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

_detect_available_physical_cores() {
    "${PYTHON}" - <<'PYEOF'
import os
from pathlib import Path

try:
    cpu_ids = set(os.sched_getaffinity(0))
except (AttributeError, OSError):
    cpu_ids = set(range(os.cpu_count() or 1))

cores = set()
for cpu_id in cpu_ids:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    try:
        package_id = int((topology / "physical_package_id").read_text().strip())
        core_id = int((topology / "core_id").read_text().strip())
    except (OSError, ValueError):
        continue
    cores.add((package_id, core_id))

print(max(1, len(cores) if cores else len(cpu_ids)))
PYEOF
}

# Accelerate otherwise turns an unset OMP_NUM_THREADS into one thread for GPU
# launches. Default to the physical cores available through this process's CPU
# affinity; keep explicit values for controlled scaling experiments.
if [[ -n "${FFT_OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS="${FFT_OMP_NUM_THREADS}"
    OMP_THREAD_SOURCE="FFT_OMP_NUM_THREADS"
elif [[ -n "${ACCELERATE_KT_OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS="${ACCELERATE_KT_OMP_NUM_THREADS}"
    OMP_THREAD_SOURCE="ACCELERATE_KT_OMP_NUM_THREADS"
elif [[ -n "${OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS="${OMP_NUM_THREADS}"
    OMP_THREAD_SOURCE="OMP_NUM_THREADS"
else
    OMP_NUM_THREADS="$(_detect_available_physical_cores)"
    OMP_THREAD_SOURCE="affinity-visible physical cores"
fi
if [[ ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: OpenMP thread count must be a positive integer, got '${OMP_NUM_THREADS}'" >&2
    exit 2
fi
export OMP_NUM_THREADS
# Pass the already-resolved runner value through the KT-specific override so
# Accelerate cannot replace it with its implicit single-thread default.
export ACCELERATE_KT_OMP_NUM_THREADS="${OMP_NUM_THREADS}"

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
NUM_GPUS=1                                                      # Single-card by default
DRY_RUN=0
FINETUNE_SELECTION="both"
BACKEND_LABEL="AMX_BF16"
TIMING_MODULE_DIR="${SCRIPT_DIR}"
TRAIN_ENTRY_MODULE="finetune_train_with_timing"
SESSION_DIR=""
LOG_DIR=""
TRAIN_LOG_NAME=""
FT_MODE=""
FT_LOG_TAG=""
FT_DISPLAY_NAME=""

usage() {
    cat <<'EOF'
Usage: bash run_finetune_perf_test_bf16.sh [options]

Fine-tuning selection:
  --mode both|full|lora       Run both (full first), full only, or LoRA only (default: both)

Shared comparison parameters:
  --gpus N                    GPU count; supported values: 1, 2 or 4 (default: 1)
  --batch-size N              Per-device train batch size (default: 1)
  --gas N                     Gradient accumulation steps (default: 1)
  --steps N                   Optimizer steps per method (default: 15)
  --warmup-steps N            Initial steps excluded from TPS/timing (default: 5)
  --learning-rate VALUE       Shared learning rate (default: 1.0e-5)

LoRA parameters:
  --lora-rank N               LoRA rank (default: 8)
  --lora-alpha N              LoRA alpha (default: 16)

Roofline assumptions:
  --gpu-bf16-tflops VALUE     Peak BF16 TFLOPS per GPU (default: 82.58)
  --cpu-bf16-tflops VALUE     Total host AMX BF16 TFLOPS (default: 373.56)
  --gpu-memory-gbps VALUE     Memory bandwidth per GPU (default: 1008.0)
  --cpu-memory-gbps VALUE     Total host memory bandwidth (default: 614.4)

Other:
  --backward-timing MODE     off, summary or trace (default: summary)
  --backward-layers LIST     Trace rows for all or comma-separated layer ids (default: all)
  --dry-run                   Generate configs and print commands without training
  -h, --help                  Show this help
EOF
}

require_positive_int() {
    local flag="$1" value="$2"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid ${flag} value: ${value} (must be a positive integer)" >&2
        exit 1
    fi
}

require_positive_number() {
    local flag="$1" value="$2"
    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || \
       ! awk -v v="${value}" 'BEGIN { exit !(v > 0) }'; then
        echo "Invalid ${flag} value: ${value} (must be > 0)" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|--finetuning-mode) FINETUNE_SELECTION="$2"; shift ;;
        --gpus) NUM_GPUS="$2"; shift ;;
        --batch-size|--per-device-train-batch-size) TRAIN_BATCH_SIZE="$2"; shift ;;
        --gas|--gradient-accumulation-steps)
            GRAD_ACCUM_STEPS="$2"; shift ;;
        --steps|--max-steps) PHASE4_STEPS="$2"; shift ;;
        --warmup-steps|--warmup-skip) WARMUP_SKIP="$2"; shift ;;
        --learning-rate) LEARNING_RATE="$2"; shift ;;
        --lora-rank) LORA_RANK="$2"; shift ;;
        --lora-alpha) LORA_ALPHA="$2"; shift ;;
        --gpu-bf16-tflops) GPU_BF16_TFLOPS="$2"; shift ;;
        --cpu-bf16-tflops) CPU_BF16_TFLOPS="$2"; shift ;;
        --gpu-memory-gbps) GPU_MEMORY_GBPS="$2"; shift ;;
        --cpu-memory-gbps) CPU_MEMORY_GBPS="$2"; shift ;;
        --backward-timing) BACKWARD_TIMING_MODE="$2"; shift ;;
        --backward-layers) BACKWARD_TIMING_LAYERS="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

case "${FINETUNE_SELECTION}" in
    both|full|lora) ;;
    *) echo "Invalid --mode: ${FINETUNE_SELECTION} (expected both, full, or lora)" >&2; exit 1 ;;
esac
case "${BACKWARD_TIMING_MODE}" in
    off|summary|trace) ;;
    *) echo "Invalid --backward-timing: ${BACKWARD_TIMING_MODE} (expected off, summary, or trace)" >&2; exit 1 ;;
esac
if [[ "${BACKWARD_TIMING_LAYERS}" != "all" ]] && \
   ! [[ "${BACKWARD_TIMING_LAYERS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid --backward-layers: ${BACKWARD_TIMING_LAYERS} (expected all or comma-separated layer ids)" >&2
    exit 1
fi
require_positive_int "--gpus" "${NUM_GPUS}"
require_positive_int "--batch-size" "${TRAIN_BATCH_SIZE}"
require_positive_int "--gas" "${GRAD_ACCUM_STEPS}"
require_positive_int "--steps" "${PHASE4_STEPS}"
require_positive_int "--lora-rank" "${LORA_RANK}"
require_positive_int "--lora-alpha" "${LORA_ALPHA}"
if ! [[ "${WARMUP_SKIP}" =~ ^[0-9]+$ ]] || [[ "${WARMUP_SKIP}" -ge "${PHASE4_STEPS}" ]]; then
    echo "Invalid --warmup-steps: ${WARMUP_SKIP} (must be >= 0 and < --steps ${PHASE4_STEPS})" >&2
    exit 1
fi
if [[ "${NUM_GPUS}" -eq 1 ]]; then
    ACCEL_CONFIG="${ACCEL_CONFIG_1GPU}"
elif [[ "${NUM_GPUS}" -eq 2 ]]; then
    ACCEL_CONFIG="${ACCEL_CONFIG_2GPU}"
elif [[ "${NUM_GPUS}" -eq 4 ]]; then
    ACCEL_CONFIG="${ACCEL_CONFIG_4GPU}"
else
    echo "Unsupported --gpus ${NUM_GPUS}: matching configs exist only for 1, 2 and 4 GPUs" >&2
    exit 1
fi
for item in \
    "--learning-rate:${LEARNING_RATE}" \
    "--gpu-bf16-tflops:${GPU_BF16_TFLOPS}" \
    "--cpu-bf16-tflops:${CPU_BF16_TFLOPS}" \
    "--gpu-memory-gbps:${GPU_MEMORY_GBPS}" \
    "--cpu-memory-gbps:${CPU_MEMORY_GBPS}"; do
    require_positive_number "${item%%:*}" "${item#*:}"
done

RUN_LABEL="${NUM_GPUS}gpu_${BACKEND_LABEL}_${FINETUNE_SELECTION^^}"

# --------------------------------------------------------------------------- #
# Colored output
# --------------------------------------------------------------------------- #
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()    { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC}  $*"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*" >&2; }
run_timestamp() { date '+%Y%m%d_%H%M%S'; }
run_time_display() { date '+%Y-%m-%d %H:%M:%S %Z %z'; }
phase_banner() {
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
    echo ""
}

# --------------------------------------------------------------------------- #
# Environment check
# --------------------------------------------------------------------------- #
check_env() {
    phase_banner "Environment Check"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        warn "Dry-run: GPU driver, model, dataset and Python package checks are skipped"
        for cfg_file in "${TRAIN_CONFIG_BASE}" "${ACCEL_CONFIG}" "${FLOPS_ANALYZE_SCRIPT}"; do
            [[ -e "${cfg_file}" ]] || { error "Required file not found: ${cfg_file}"; exit 1; }
        done
        ok "Static environment check passed"
        return 0
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi not found, please verify CUDA driver is installed"
        exit 1
    fi
    local gpu_list
    gpu_list="$(nvidia-smi -L 2>/dev/null || true)"
    ACTUAL_GPUS=$(printf '%s\n' "${gpu_list}" | sed '/^$/d' | wc -l)
    if [[ "${ACTUAL_GPUS}" -lt "${NUM_GPUS}" ]]; then
        error "System GPU count (${ACTUAL_GPUS}) is less than requested ${NUM_GPUS}"
        exit 1
    fi

    # Model path (AMXBF16 reads original HF BF16 weights directly)
    if [[ ! -d "${MODEL_PATH}" ]]; then
        error "Model path not found: ${MODEL_PATH}"
        exit 1
    fi

    if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
        error "LLaMA-Factory directory not found: ${LLAMA_FACTORY_DIR}"
        exit 1
    fi

    # Dataset check (every sample >7000 tokens, so cutoff 4096 is fully occupied)
    if [[ ! -f "${DATA_DIR}/${DATASET_NAME}.json" ]]; then
        warn "Dataset not found: ${DATA_DIR}/${DATASET_NAME}.json"
        warn "Generating dataset automatically..."
        "${PYTHON}" "${GEN_DATASET_SCRIPT}" || {
            error "Dataset generation failed. Please run: python3 ${GEN_DATASET_SCRIPT}"
            exit 1
        }
    fi

    # Config files (must exist before any phase starts)
    for cfg_file in "${TRAIN_CONFIG_BASE}" "${ACCEL_CONFIG}"; do
        if [[ ! -f "${cfg_file}" ]]; then
            error "Required config not found: ${cfg_file}"
            exit 1
        fi
    done

    if ! "${PYTHON}" -c "import accelerate" &>/dev/null; then
        error "accelerate not installed in ${PYTHON}"
        exit 1
    fi

    if [[ "${BACKWARD_TIMING_MODE}" != "off" ]] && \
       ! "${PYTHON}" -c 'from kt_kernel.sft.backward_timing import get_backward_timing_recorder; from kt_kernel.kt_kernel_ext import moe; assert hasattr(moe.AMXBF16_SFT_MOE, "set_backward_timing_level")' &>/dev/null; then
        error "Backward timing support is not installed in ${CONDA_ENV}. Rebuild/install ktransformers/kt-kernel first."
        exit 1
    fi

    for pkg in psutil pynvml matplotlib pandas; do
        if ! "${PYTHON}" -c "import ${pkg}" &>/dev/null; then
            warn "${pkg} not installed (some features limited)"
        fi
    done

    ok "Environment check passed (${ACTUAL_GPUS} GPU(s), Python: ${PYTHON})"
}

# --------------------------------------------------------------------------- #
# Resource estimation
# --------------------------------------------------------------------------- #
estimate_resources() {
    log "Run config: mode=${FINETUNE_SELECTION}, GPUs=${NUM_GPUS}, batch=${TRAIN_BATCH_SIZE}, GAS=${GRAD_ACCUM_STEPS}, steps=${PHASE4_STEPS}, warmup=${WARMUP_SKIP}, OMP=${OMP_NUM_THREADS}, backward_timing=${BACKWARD_TIMING_MODE}, layers=${BACKWARD_TIMING_LAYERS}"
}

# --------------------------------------------------------------------------- #
# Create one session directory, then a distinct task directory per method.
# --------------------------------------------------------------------------- #
setup_session_dir() {
    RUN_TS=$(run_timestamp)
    RUN_TIME="$(run_time_display)"
    RUN_TIMEZONE="$(date '+%Z %z')"
    local order_label="${FINETUNE_SELECTION^^}"
    [[ "${FINETUNE_SELECTION}" == "both" ]] && order_label="FULL_THEN_LORA"
    RUN_LABEL="${NUM_GPUS}gpu_${BACKEND_LABEL}_${order_label}"
    SESSION_DIR="${LOG_BASE}/${RUN_TS}_${RUN_LABEL}"
    mkdir -p "${SESSION_DIR}"
    log "Session directory: ${SESSION_DIR}"

    export RUN_TS RUN_TIME RUN_TIMEZONE RUN_LABEL SESSION_DIR
}

setup_task_dir() {
    FT_MODE="$1"
    case "${FT_MODE}" in
        full)
            FT_LOG_TAG="full_ft"
            FT_DISPLAY_NAME="全量微调 (Full FT)"
            ;;
        lora)
            FT_LOG_TAG="lora_ft"
            FT_DISPLAY_NAME="LoRA 微调"
            ;;
        *) error "Internal error: invalid fine-tuning mode ${FT_MODE}"; return 1 ;;
    esac

    LOG_DIR="${SESSION_DIR}/${FT_LOG_TAG}"
    TRAIN_LOG_NAME="train_${FT_LOG_TAG}.log"
    MONITOR_FIFO="/tmp/qwen3_ft_monitor_$$_${FT_LOG_TAG}.fifo"
    mkdir -p "${LOG_DIR}/plots"
    log "Task log directory (${FT_MODE}): ${LOG_DIR}"

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        [[ -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
        mkfifo "${MONITOR_FIFO}"
    fi

    export LOG_DIR MONITOR_FIFO FT_MODE FT_LOG_TAG FT_DISPLAY_NAME TRAIN_LOG_NAME
}

send_event() {
    local msg="$1"
    [[ "${DRY_RUN}" -eq 1 ]] && return 0
    if [[ -p "${MONITOR_FIFO}" ]]; then
        echo "${msg}" >> "${MONITOR_FIFO}" 2>/dev/null || true
    fi
}

# --------------------------------------------------------------------------- #
# Start / stop monitoring
# --------------------------------------------------------------------------- #
start_monitor() {
    "${PYTHON}" "${MONITOR_SCRIPT}" \
        --out "${LOG_DIR}/monitor.csv" \
        --fifo "${MONITOR_FIFO}" \
        --interval 2 \
        --pid $$ \
        >> "${LOG_DIR}/monitor.log" 2>&1 &
    MONITOR_PID=$!
    ok "System monitor started (PID=${MONITOR_PID})"
    sleep 1
    # Re-announce root pid after FIFO reader is up.
    send_event "pid:$$"
}

stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        log "Stopping monitor (PID: ${MONITOR_PID})..."
        kill -TERM "${MONITOR_PID}" || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
    [[ -n "${MONITOR_FIFO}" && -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
    return 0
}

# --------------------------------------------------------------------------- #
# Delete model_output/ for a phase to reclaim disk space.
# LLaMA-Factory always calls trainer.save_model() at the end of run_sft(),
# regardless of save_strategy.  Each saved checkpoint is ~115 GB for this
# model, so we delete it immediately after a phase no longer needs it.
#
# --------------------------------------------------------------------------- #
cleanup_model_output() {
    local phase_name="$1"
    local model_dir="${LOG_DIR}/${phase_name}/model_output"
    if [[ -d "${model_dir}" ]]; then
        log "[cleanup] Removing ${model_dir} to free disk space..."
        rm -rf "${model_dir}"
        ok "[cleanup] ${phase_name}/model_output deleted"
    fi
}

# --------------------------------------------------------------------------- #
# Build per-phase training config as a temporary copy
# --------------------------------------------------------------------------- #
make_phase_config() {
    local phase_name="$1"; shift
    local phase_dir="${LOG_DIR}/${phase_name}"
    mkdir -p "${phase_dir}"
    local cfg="${phase_dir}/train_config.yaml"

    if [[ ! -f "${TRAIN_CONFIG_BASE}" ]]; then
        error "Base training config not found: ${TRAIN_CONFIG_BASE}"
        return 1
    fi

    cp "${TRAIN_CONFIG_BASE}" "${cfg}"
    sed -i "s|output_dir: .*|output_dir: ${phase_dir}/model_output|g" "${cfg}"
    # Sync overwrite_output_dir path too
    sed -i "s|overwrite_output_dir: .*|overwrite_output_dir: true|g" "${cfg}"

    while [[ $# -gt 0 ]]; do
        local kv="$1"; shift
        local key="${kv%%=*}"
        local val="${kv#*=}"
        if grep -q "^${key}:" "${cfg}"; then
            sed -i "s|^${key}: .*|${key}: ${val}|g" "${cfg}"
        else
            echo "${key}: ${val}" >> "${cfg}"
        fi
    done

    echo "${cfg}"
}

# Keep the complete child-process output in the phase log while showing only
# optimizer-step progress and actionable failures in the terminal.
filter_training_status() {
    local expected_steps="$1"
    tr '\r' '\n' | awk -v total="${expected_steps}" '
        {
            lower = tolower($0)
            is_step_progress = index($0, "/" total) && \
                ($0 ~ /%.*\|/ || $0 ~ /(s\/it|it\/s)/)
            is_failure = lower ~ /traceback \(most recent call last\)/ || \
                lower ~ /[[:alpha:]_]*error:|exception:|keyboardinterrupt/ || \
                lower ~ /cuda out of memory|segmentation fault|core dumped|nccl.*error|fatal error|killed/
            if (is_step_progress || is_failure) {
                print
                fflush()
            }
        }
    '
}

# --------------------------------------------------------------------------- #
# Run a single accelerate training command.  There is deliberately no
# mode-dependent OOM fallback: changing GPU count for only one method would
# invalidate a Full-vs-LoRA throughput comparison.
# --------------------------------------------------------------------------- #
run_train() {
    local phase_name="$1"
    local desc="$2"
    local train_cfg="$3"
    local phase_log="${LOG_DIR}/${phase_name}/${TRAIN_LOG_NAME}"
    local exit_code=0

    log "Starting training [${phase_name}]: ${desc} (GPU=${NUM_GPUS}, OMP=${OMP_NUM_THREADS})"
    log "Full training output: ${phase_log}"
    send_event "phase:${phase_name}"
    send_event "event:train_start"

    local gpus_str
    gpus_str=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',')

    local accelerate_bin="${CONDA_BIN_DIR}/accelerate"
    [[ ! -x "${accelerate_bin}" ]] && accelerate_bin="accelerate"

    local timing_dir="${LOG_DIR}/${phase_name}/step_timing"
    local cmd=(
        env
        USE_KT=1
        ACCELERATE_USE_KT=true
        ACCELERATE_KT_TRAIN_MODE="${FT_MODE}"
        KT_FINETUNE_MODE="${FT_MODE}"
        KT_STEP_TIMING=1
        KT_STEP_TIMING_OUT_DIR="${timing_dir}"
        KT_STEP_TIMING_WARMUP_SKIP="${WARMUP_SKIP}"
        KT_STEP_TIMING_TOKENS_PER_STEP="$((NUM_GPUS * TRAIN_BATCH_SIZE * CUTOFF_LEN * GRAD_ACCUM_STEPS))"
        KT_BACKWARD_TIMING="${BACKWARD_TIMING_MODE}"
        KT_BACKWARD_TIMING_OUT_DIR="${timing_dir}/backward_internal"
        KT_BACKWARD_TIMING_WARMUP_SKIP="${WARMUP_SKIP}"
        KT_BACKWARD_TIMING_LAYERS="${BACKWARD_TIMING_LAYERS}"
        OMP_NUM_THREADS="${OMP_NUM_THREADS}"
        ACCELERATE_KT_OMP_NUM_THREADS="${ACCELERATE_KT_OMP_NUM_THREADS}"
        PYTHONPATH="${TIMING_MODULE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        CUDA_VISIBLE_DEVICES="${gpus_str}"
        "${accelerate_bin}" launch
        --config_file "${ACCEL_CONFIG}"
        -m "${TRAIN_ENTRY_MODULE}" train
        "${train_cfg}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "[DRY-RUN] Command: ${cmd[*]}"
        return 0
    fi

    pushd "${LLAMA_FACTORY_DIR}" > /dev/null
    set +e
    "${cmd[@]}" 2>&1 | tee "${phase_log}" | filter_training_status "${PHASE4_STEPS}"
    exit_code=${PIPESTATUS[0]}
    set -e
    popd > /dev/null

    send_event "event:train_end"

    if [[ "${exit_code}" -eq 0 ]]; then
        ok "Training [${phase_name}] finished"
    else
        error "Training [${phase_name}] failed with exit code ${exit_code}; see ${phase_log}"
    fi

    return "${exit_code}"
}

# --------------------------------------------------------------------------- #
# Analyze training log (extract loss, grad_norm, NaN detection)
# --------------------------------------------------------------------------- #
analyze_log() {
    local phase_name="$1"
    local log_file="${LOG_DIR}/${phase_name}/${TRAIN_LOG_NAME}"
    local result_file="${LOG_DIR}/${phase_name}/log_analysis.txt"

    [[ ! -f "${log_file}" ]] && return

    {
        echo "=== Phase: ${phase_name} Log Analysis ==="
        echo "--- Training Loss (last 20 entries) ---"
        grep -i "loss" "${log_file}" | tail -20 || echo "(no loss records found)"

        echo ""
        echo "--- Gradient Norm (all records) ---"
        grep -i "grad_norm\|gradient_norm" "${log_file}" | tail -30 || echo "(no grad_norm records)"

        echo ""
        echo "--- NaN / Inf occurrence count ---"
        local nan_count inf_count
        nan_count=$(grep -ci "nan" "${log_file}" 2>/dev/null || true)
        inf_count=$(grep -ci "inf" "${log_file}" 2>/dev/null || true)
        echo "NaN lines: ${nan_count};  Inf lines: ${inf_count}"

        echo ""
        echo "--- KT-related warnings/errors ---"
        grep -i "ktransformer\|kt_kernel\|amx\|moe\|expert\|backward\|grad_proj" \
             "${log_file}" 2>/dev/null | grep -i "warn\|error\|fail\|bug" | tail -20 \
             || echo "(no KT-related alerts)"

        echo ""
        echo "--- update_base_weights timing (P7: re-quantize overhead) ---"
        grep -i "update_base_weights\|re-quantize\|set_base_weight\|TP_MOE_SFT" "${log_file}" | tail -20 \
             || echo "(no update_base_weights log detected)"

        echo ""
        echo "--- Process exit code analysis ---"
        if grep -qi "segmentation fault\|sigsegv\|core dumped" "${log_file}"; then
            echo "⚠ SIGSEGV / core dump detected (P5: C++ gradient index out of bounds)"
        fi
        if grep -qi "cuda out of memory\|oom" "${log_file}"; then
            echo "⚠ CUDA OOM detected"
        fi
        if grep -qi "ddp_timeout\|timeout expired" "${log_file}"; then
            echo "⚠ DDP timeout detected (P2/P7: CPU computation too slow)"
        fi
    } > "${result_file}"
}

# --------------------------------------------------------------------------- #
# Memory component attribution
#
# monitor.csv records runtime totals.  Component ownership such as "optimizer"
# or "gradients" is not exposed by nvidia-smi or psutil, so these reports combine
# runtime totals with model-structure estimates.  Treat component columns as
# attribution estimates, and monitor totals as the measured source of truth.
# --------------------------------------------------------------------------- #
generate_memory_component_estimate() {
    log "Generating memory component estimate..."

    "${PYTHON}" - "${MODEL_PATH}" "${NUM_GPUS}" "${CUTOFF_LEN}" "${TRAIN_BATCH_SIZE}" "${LOG_DIR}" "${FT_MODE}" "${LORA_RANK}" \
        > "${LOG_DIR}/memory_component_estimate.txt" <<'PYEOF'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
num_gpus = int(sys.argv[2])
cutoff_len = int(sys.argv[3])
batch_size = int(sys.argv[4])
log_dir = Path(sys.argv[5])
mode = sys.argv[6]
lora_rank = int(sys.argv[7])

def gb(n: float) -> float:
    return n / 1e9

def gib_to_bytes(n: float) -> float:
    return n * (1024 ** 3)

cfg = json.loads((model_path / "config.json").read_text())
cache_path = model_path / ".moe_analysis_cache.json"
cache = {}
if cache_path.exists():
    try:
        cache = json.loads(cache_path.read_text()).get("result", {})
    except Exception:
        cache = {}

H = int(cfg["hidden_size"])
shared_i = int(cfg.get("intermediate_size", 0) or 0)
moe_i = int(cfg["moe_intermediate_size"])
E = int(cfg["num_experts"])
L = int(cfg["num_hidden_layers"])
top_k = int(cfg["num_experts_per_tok"])
V = int(cfg["vocab_size"])

expert_params = E * 3 * H * moe_i * L
expert_weight_bytes = expert_params * 2
if mode == "full":
    expert_trainable_params = expert_params
else:
    # gate/up: H->moe_i; down: moe_i->H, for all expert adapters.
    expert_trainable_params = L * E * lora_rank * (4 * H + 2 * moe_i)
expert_grad_bytes = expert_trainable_params * 4
expert_adam_bytes = expert_trainable_params * 8

rest_gib = float(cache.get("rest_size_gb", 0.0) or 0.0)
if rest_gib > 0:
    non_expert_weight_bytes = gib_to_bytes(rest_gib)
    non_expert_source = ".moe_analysis_cache.json: rest_size_gb"
else:
    # Conservative fallback: embeddings, lm_head, attention, router/norm.
    head_dim = int(cfg.get("head_dim", H // int(cfg["num_attention_heads"])))
    nh = int(cfg["num_attention_heads"])
    nkv = int(cfg["num_key_value_heads"])
    q = H * (nh * head_dim)
    k = H * (nkv * head_dim)
    v = H * (nkv * head_dim)
    o = (nh * head_dim) * H
    p_embed = V * H
    p_lm = 0 if cfg.get("tie_word_embeddings", False) else V * H
    p_attn = L * (q + k + v + o)
    p_router_norm = L * (H * E + 2 * H)
    non_expert_weight_bytes = (p_embed + p_lm + p_attn + p_router_norm) * 2
    non_expert_source = "config fallback"

gpu_model_weights = gb(non_expert_weight_bytes / num_gpus)
if mode == "full":
    gpu_trainable_params = non_expert_weight_bytes / 2
else:
    head_dim = int(cfg.get("head_dim", H // int(cfg["num_attention_heads"])))
    nh = int(cfg["num_attention_heads"])
    nkv = int(cfg["num_key_value_heads"])
    q_out = nh * head_dim
    kv_out = nkv * head_dim
    attn_dims = (H + q_out) + 2 * (H + kv_out) + (q_out + H)
    shared_dims = 2 * (H + shared_i) + (shared_i + H)
    router_dims = H + E
    gpu_trainable_params = L * lora_rank * (attn_dims + shared_dims + router_dims)
gpu_gradients = gb(gpu_trainable_params * 4 / num_gpus)
gpu_optimizer = gb(gpu_trainable_params * 8 / num_gpus)
# Rough activation/workspace estimate scaled from the observed 1024-token,
# batch=1 baseline. GAS does not retain multiple microbatch activation sets.
gpu_activations = 2.0 * (cutoff_len / 1024.0) * batch_size

estimate = {
    "notes": [
        "Measured totals come from monitor.csv.",
        "Component columns are model-structure estimates; nvidia-smi/psutil do not expose optimizer-vs-gradient ownership.",
        "Optimizer estimates assume AdamW m+v FP32 states.",
        "Gradient estimates assume FP32 gradients for trainable tensors.",
    ],
    "model": {
        "path": str(model_path),
        "finetuning_mode": mode,
        "lora_rank": lora_rank if mode == "lora" else 0,
        "hidden_size": H,
        "num_layers": L,
        "num_experts": E,
        "active_experts_per_token": top_k,
        "cutoff_len": cutoff_len,
        "per_device_batch_size": batch_size,
        "num_gpus": num_gpus,
        "non_expert_source": non_expert_source,
    },
    "gpu_per_card_gb": {
        "model_weights": gpu_model_weights,
        "gradients": gpu_gradients,
        "optimizer_adamw_m_v": gpu_optimizer,
        "activations_workspace": gpu_activations,
        "estimated_full_step_total": gpu_model_weights + gpu_gradients + gpu_optimizer + gpu_activations,
    },
    "cpu_host_gb": {
        "expert_model_weights": gb(expert_weight_bytes),
        "expert_gradients": gb(expert_grad_bytes),
        "expert_optimizer_adamw_m_v": gb(expert_adam_bytes),
        "numa_work_buffers_estimate_low": 10.0,
        "numa_work_buffers_estimate_high": 30.0,
        "estimated_full_step_total_low": gb(expert_weight_bytes + expert_grad_bytes + expert_adam_bytes) + 10.0,
        "estimated_full_step_total_high": gb(expert_weight_bytes + expert_grad_bytes + expert_adam_bytes) + 30.0,
    },
}

(log_dir / "memory_component_estimate.json").write_text(json.dumps(estimate, indent=2))

print("=== Memory Component Attribution Estimate ===")
print(f"model_path          : {model_path}")
print(f"finetuning_mode     : {mode}")
print(f"architecture        : {cfg.get('architectures', ['?'])[0]}")
print(f"layers/experts/topk : {L} / {E} / {top_k}")
print(f"cutoff_len          : {cutoff_len}")
print(f"per_device_batch    : {batch_size}")
print(f"num_gpus            : {num_gpus}")
print()
print("--- GPU per card estimate ---")
for k, v in estimate["gpu_per_card_gb"].items():
    print(f"{k:30s}: {v:8.2f} GB")
print()
print("--- CPU host estimate ---")
for k, v in estimate["cpu_host_gb"].items():
    print(f"{k:30s}: {v:8.2f} GB")
print()
print("Outputs:")
print(f"  {log_dir / 'memory_component_estimate.json'}")
print(f"  {log_dir / 'memory_component_estimate.txt'}")
PYEOF
    ok "Memory estimate saved: ${LOG_DIR}/memory_component_estimate.txt"
}

generate_memory_component_timeline() {
    log "Generating runtime memory timeline..."

    "${PYTHON}" - "${LOG_DIR}" <<'PYEOF' > "${LOG_DIR}/memory_component_observed.txt"
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

log_dir = Path(sys.argv[1])
monitor_path = log_dir / "monitor.csv"
estimate_path = log_dir / "memory_component_estimate.json"
timeline_path = log_dir / "memory_component_timeline.csv"

if not monitor_path.exists():
    print("monitor.csv not found; no runtime attribution timeline generated")
    raise SystemExit(0)
if not estimate_path.exists():
    print("memory_component_estimate.json not found; no attribution estimates available")
    raise SystemExit(0)

rows = list(csv.DictReader(monitor_path.open()))
if not rows:
    print("monitor.csv is empty")
    raise SystemExit(0)

estimate = json.loads(estimate_path.read_text())
gpu_est = estimate["gpu_per_card_gb"]
cpu_est = estimate["cpu_host_gb"]

def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default

gpu_cols = sorted(
    [k for k in rows[0] if re.match(r"^gpu\d+_mem_used_mb$", k)],
    key=lambda x: int(x.split("_")[0][3:]),
)
has_proc_ram = "proc_ram_gb" in rows[0]
proc_gpu_cols = {
    c: f"proc_gpu{c.split('_')[0][3:]}_mem_mb"
    for c in gpu_cols
}
use_proc_gpu = all(proc_gpu_cols[c] in rows[0] for c in gpu_cols) if gpu_cols else False

def measured_ram(row):
    if has_proc_ram:
        return f(row, "proc_ram_gb")
    return f(row, "ram_used_gb")

def measured_gpu_gb(row, host_col):
    if use_proc_gpu:
        return f(row, proc_gpu_cols[host_col]) / 1024.0
    return f(row, host_col) / 1024.0

baseline_ram = measured_ram(rows[0])
baseline_gpu = {c: measured_gpu_gb(rows[0], c) for c in gpu_cols}

out_fields = [
    "timestamp", "elapsed_sec", "phase", "event",
    "ram_used_gb", "ram_delta_from_start_gb",
    "ram_scope",
    "cpu_expert_model_weights_est_gb",
    "cpu_expert_gradients_est_gb",
    "cpu_expert_optimizer_adamw_est_gb",
    "cpu_numa_workspace_est_low_gb",
    "cpu_numa_workspace_est_high_gb",
    "cpu_other_or_not_yet_allocated_gb",
]
for c in gpu_cols:
    g = c.split("_")[0]
    out_fields += [
        f"{g}_mem_used_gb",
        f"{g}_delta_from_start_gb",
        f"{g}_model_weights_est_gb",
        f"{g}_gradients_est_gb",
        f"{g}_optimizer_adamw_est_gb",
        f"{g}_activations_workspace_est_gb",
        f"{g}_other_or_not_yet_allocated_gb",
    ]
out_fields.append("attribution_note")

peak = {"ram_used_gb": 0.0, "ram_delta_gb": 0.0}
phase_peaks = defaultdict(lambda: {"ram": 0.0, "gpu": defaultdict(float)})
ram_scope = "process_tree" if has_proc_ram else "host"
gpu_scope = "process_tree" if use_proc_gpu else "host_card"

with timeline_path.open("w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=out_fields)
    writer.writeheader()
    for row in rows:
        phase = row.get("phase", "")
        in_phase4 = phase == "phase4"
        ram_used = measured_ram(row)
        ram_delta = max(ram_used - baseline_ram, 0.0)
        cpu_known_low = (
            cpu_est["expert_model_weights"]
            + cpu_est["expert_gradients"]
            + cpu_est["expert_optimizer_adamw_m_v"]
            + cpu_est["numa_work_buffers_estimate_low"]
        ) if in_phase4 else 0.0
        out = {
            "timestamp": row.get("timestamp", ""),
            "elapsed_sec": row.get("elapsed_sec", ""),
            "phase": phase,
            "event": row.get("event", ""),
            "ram_used_gb": f"{ram_used:.2f}",
            "ram_delta_from_start_gb": f"{ram_delta:.2f}",
            "ram_scope": ram_scope,
            "cpu_expert_model_weights_est_gb": f"{cpu_est['expert_model_weights'] if in_phase4 else 0.0:.2f}",
            "cpu_expert_gradients_est_gb": f"{cpu_est['expert_gradients'] if in_phase4 else 0.0:.2f}",
            "cpu_expert_optimizer_adamw_est_gb": f"{cpu_est['expert_optimizer_adamw_m_v'] if in_phase4 else 0.0:.2f}",
            "cpu_numa_workspace_est_low_gb": f"{cpu_est['numa_work_buffers_estimate_low'] if in_phase4 else 0.0:.2f}",
            "cpu_numa_workspace_est_high_gb": f"{cpu_est['numa_work_buffers_estimate_high'] if in_phase4 else 0.0:.2f}",
            "cpu_other_or_not_yet_allocated_gb": f"{max(ram_delta - cpu_known_low, 0.0):.2f}",
            "attribution_note": (
                f"totals measured as {ram_scope}/{gpu_scope}; "
                "component columns are estimates"
            ),
        }
        peak["ram_used_gb"] = max(peak["ram_used_gb"], ram_used)
        peak["ram_delta_gb"] = max(peak["ram_delta_gb"], ram_delta)
        phase_peaks[phase]["ram"] = max(phase_peaks[phase]["ram"], ram_used)

        for c in gpu_cols:
            g = c.split("_")[0]
            used = measured_gpu_gb(row, c)
            delta = max(used - baseline_gpu[c], 0.0)
            gpu_known = (
                gpu_est["model_weights"]
                + gpu_est["gradients"]
                + gpu_est["optimizer_adamw_m_v"]
                + gpu_est["activations_workspace"]
            ) if in_phase4 else 0.0
            out[f"{g}_mem_used_gb"] = f"{used:.2f}"
            out[f"{g}_delta_from_start_gb"] = f"{delta:.2f}"
            out[f"{g}_model_weights_est_gb"] = f"{gpu_est['model_weights'] if in_phase4 else 0.0:.2f}"
            out[f"{g}_gradients_est_gb"] = f"{gpu_est['gradients'] if in_phase4 else 0.0:.2f}"
            out[f"{g}_optimizer_adamw_est_gb"] = f"{gpu_est['optimizer_adamw_m_v'] if in_phase4 else 0.0:.2f}"
            out[f"{g}_activations_workspace_est_gb"] = f"{gpu_est['activations_workspace'] if in_phase4 else 0.0:.2f}"
            out[f"{g}_other_or_not_yet_allocated_gb"] = f"{max(delta - gpu_known, 0.0):.2f}"
            peak[f"{g}_used_gb"] = max(peak.get(f"{g}_used_gb", 0.0), used)
            peak[f"{g}_delta_gb"] = max(peak.get(f"{g}_delta_gb", 0.0), delta)
            phase_peaks[phase]["gpu"][g] = max(phase_peaks[phase]["gpu"][g], used)
        writer.writerow(out)

print("=== Runtime Memory Attribution Timeline ===")
print(f"timeline_csv              : {timeline_path}")
print(f"ram_scope                 : {ram_scope}")
print(f"gpu_scope                 : {gpu_scope}")
print(f"baseline_ram_gb           : {baseline_ram:.2f}")
print(f"peak_ram_used_gb          : {peak['ram_used_gb']:.2f}")
print(f"peak_ram_delta_gb         : {peak['ram_delta_gb']:.2f}")
for c in gpu_cols:
    g = c.split("_")[0]
    print(f"baseline_{g}_gb           : {baseline_gpu[c]:.2f}")
    print(f"peak_{g}_used_gb          : {peak.get(f'{g}_used_gb', 0.0):.2f}")
    print(f"peak_{g}_delta_gb         : {peak.get(f'{g}_delta_gb', 0.0):.2f}")

print()
print("Component attribution is estimate-based:")
print("  GPU: non-expert model weights + gradients + AdamW states + activation/workspace estimate.")
print("  CPU: expert BF16 weights + expert gradients + AdamW states + NUMA work buffer estimate.")
print("  Exact optimizer/gradient ownership requires instrumentation inside the training process.")
PYEOF
    ok "Memory timeline saved: ${LOG_DIR}/memory_component_timeline.csv"
}

# --------------------------------------------------------------------------- #
# Phase 4: Stability extension + accurate TPS measurement
# --------------------------------------------------------------------------- #
run_phase4() {
    phase_banner "${FT_DISPLAY_NAME}: TPS + Step Timing (${PHASE4_STEPS} steps)"
    local cfg
    local config_overrides=(
        "max_steps=${PHASE4_STEPS}"
        "per_device_train_batch_size=${TRAIN_BATCH_SIZE}"
        "gradient_accumulation_steps=${GRAD_ACCUM_STEPS}"
        "learning_rate=${LEARNING_RATE}"
        "finetuning_type=${FT_MODE}"
        "cutoff_len=${CUTOFF_LEN}"
        "save_strategy='no'"
        "logging_steps=1"
    )
    if [[ "${FT_MODE}" == "lora" ]]; then
        config_overrides+=(
            "lora_rank=${LORA_RANK}"
            "lora_alpha=${LORA_ALPHA}"
            "lora_target=all"
        )
    fi
    cfg=$(make_phase_config "phase4" "${config_overrides[@]}")

    log "Benchmark ready: mode=${FT_MODE}, steps=${PHASE4_STEPS}, warmup=${WARMUP_SKIP}, tokens/step=$((NUM_GPUS * TRAIN_BATCH_SIZE * CUTOFF_LEN * GRAD_ACCUM_STEPS)), backward_timing=${BACKWARD_TIMING_MODE}"

    local t_start
    t_start=$(date +%s)
    send_event "phase:phase4"

    local exit_code=0
    run_train "phase4" "${FT_MODE} TPS benchmark (${PHASE4_STEPS} steps)" "${cfg}" || exit_code=$?

    echo "${exit_code}" > "${LOG_DIR}/phase4/exit_code.txt"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return "${exit_code}"
    fi

    local t_end
    t_end=$(date +%s)
    local total_sec=$((t_end - t_start))

    {
        echo "=== Phase 4 Performance Analysis ==="
        echo "Total wall time: ${total_sec} seconds"

        local actual_steps
        actual_steps=$(grep -c "{'loss'" "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" 2>/dev/null || true)
        if [[ "${actual_steps}" -eq 0 ]]; then
            actual_steps=$(grep -c '"loss"' "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" 2>/dev/null || true)
        fi
        echo "steps_completed: ${actual_steps} (expected ${PHASE4_STEPS})"

        if [[ "${actual_steps}" -gt 0 ]]; then
            local avg_all
            avg_all=$(echo "scale=1; ${total_sec} / ${actual_steps}" | bc 2>/dev/null || echo "N/A")
            echo "avg_sec_per_step (all incl. warmup): ${avg_all}"
        fi

        echo ""
        # ------------------------------------------------------------------- #
        # Accurate TPS: parse tqdm per-step speed, skip first WARMUP_SKIP steps
        # ------------------------------------------------------------------- #
        echo "--- Accurate TPS (warmup excluded) ---"
        "${PYTHON}" - \
            "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" \
            "${CUTOFF_LEN}" \
            "${NUM_GPUS}" \
            "${WARMUP_SKIP}" \
            "${GRAD_ACCUM_STEPS}" \
            "${TRAIN_BATCH_SIZE}" <<'PYEOF'
import re, sys, statistics

log_path   = sys.argv[1]
cutoff_len = int(sys.argv[2])
num_gpus   = int(sys.argv[3])
warmup_n   = int(sys.argv[4])
gas        = int(sys.argv[5])
batch_size = int(sys.argv[6])

try:
    log_text = open(log_path, errors="replace").read()
except FileNotFoundError:
    print("  train.log not found; TPS unavailable")
    sys.exit(0)

# Match tqdm progress lines: "N/total [..., X.XXs/it]" or "Xs/it"
# Handles both "2.78s/it" and "12.42s/it" formats
pattern = re.compile(r'(\d+)/\d+\s+\[[\d:]+<[\d:]+,\s*([\d.]+)\s*s/it\]')
steps = []
for m in pattern.finditer(log_text):
    step_num  = int(m.group(1))
    step_time = float(m.group(2))
    steps.append((step_num, step_time))

# De-duplicate: keep last occurrence of each step number
seen = {}
for step_num, step_time in steps:
    seen[step_num] = step_time
steps_dedup = sorted(seen.items())

tokens_per_step = num_gpus * batch_size * cutoff_len * gas
stable = [(s, t) for s, t in steps_dedup if s > warmup_n]

print(f"  total_steps_logged : {len(steps_dedup)}")
print(f"  warmup_steps_skip  : {warmup_n}")
print(f"  stable_steps       : {len(stable)}")
print(f"  gas                : {gas}")
print(f"  batch_size         : {batch_size}")
print(f"  tokens_per_step    : {tokens_per_step}  ({num_gpus} GPU x batch={batch_size} x {cutoff_len} x GAS={gas})")

if not stable:
    print("  (not enough steps to compute stable TPS)")
    sys.exit(0)

times = [t for _, t in stable]
avg_t = sum(times) / len(times)
med_t = statistics.median(times)
min_t = min(times)
max_t = max(times)

print(f"")
print(f"  step_time_avg    : {avg_t:.2f} s")
print(f"  step_time_median : {med_t:.2f} s")
print(f"  step_time_min    : {min_t:.2f} s")
print(f"  step_time_max    : {max_t:.2f} s")
print(f"")
print(f"  TPS (avg)        : {tokens_per_step / avg_t:.1f} tokens/sec")
print(f"  TPS (median)     : {tokens_per_step / med_t:.1f} tokens/sec")
print(f"  TPS (peak)       : {tokens_per_step / min_t:.1f} tokens/sec")
PYEOF

        echo ""
        echo "--- P2 diagnosis (CPU backward speed) ---"
        echo "  avg_sec > 120s  => backward_base_weight_grad is a serious bottleneck"
        echo "  avg_sec > 300s  => DDP timeout risk"

        echo ""
        echo "--- MoE router aux loss (P4) ---"
        grep -i "aux_loss\|router_loss\|balance_loss" \
             "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" 2>/dev/null | tail -10 || echo "  (no router aux loss found)"

        echo ""
        echo "--- update_base_weights call count (P7) ---"
        local upd_count
        upd_count=$(grep -ci "update_base_weights\|re-quantize\|TP_MOE_SFT" \
                    "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" 2>/dev/null || true)
        echo "  update_base_weights triggered: ${upd_count} times"

        echo ""
        echo "--- NaN/Inf statistics (P5) ---"
        local nan_cnt inf_cnt
        nan_cnt=$(grep -ci " nan" "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" 2>/dev/null || true)
        inf_cnt=$(grep -ci " inf" "${LOG_DIR}/phase4/${TRAIN_LOG_NAME}" 2>/dev/null || true)
        echo "  NaN lines: ${nan_cnt};  Inf lines: ${inf_cnt}"

    } > "${LOG_DIR}/phase4_analysis.txt"

    local stable_steps step_time_avg tps_avg
    stable_steps=$(awk -F: '/stable_steps/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' \
        "${LOG_DIR}/phase4_analysis.txt")
    step_time_avg=$(awk -F: '/step_time_avg/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' \
        "${LOG_DIR}/phase4_analysis.txt")
    tps_avg=$(awk -F: '/TPS \(avg\)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' \
        "${LOG_DIR}/phase4_analysis.txt")
    log "Performance result: exit=${exit_code}, wall=${total_sec}s, stable_steps=${stable_steps:-0}, step_avg=${step_time_avg:-N/A}, TPS=${tps_avg:-N/A}"
    if [[ "${nan_cnt}" -gt 0 ]] || [[ "${inf_cnt}" -gt 0 ]]; then
        warn "Numerical anomaly detected (NaN=${nan_cnt}, Inf=${inf_cnt}); see ${LOG_DIR}/phase4_analysis.txt"
    fi

    analyze_log "phase4"
    generate_flops_analysis || true
    cleanup_model_output "phase4"
    return "${exit_code}"
}

generate_flops_analysis() {
    local timing_dir="${LOG_DIR}/phase4/step_timing"
    log "Generating FLOPs/timing analysis..."
    if "${PYTHON}" "${FLOPS_ANALYZE_SCRIPT}" \
        --timing-json "${timing_dir}/step_timing.json" \
        --model-config "${MODEL_PATH}/config.json" \
        --mode "${FT_MODE}" \
        --seq-len "${CUTOFF_LEN}" \
        --batch-size "${TRAIN_BATCH_SIZE}" \
        --gas "${GRAD_ACCUM_STEPS}" \
        --gpus "${NUM_GPUS}" \
        --lora-rank "${LORA_RANK}" \
        --gpu-bf16-tflops "${GPU_BF16_TFLOPS}" \
        --cpu-bf16-tflops "${CPU_BF16_TFLOPS}" \
        --gpu-memory-gbps "${GPU_MEMORY_GBPS}" \
        --cpu-memory-gbps "${CPU_MEMORY_GBPS}" \
        --output-dir "${timing_dir}" \
        > "${timing_dir}/flops_analysis.log" 2>&1; then
        ok "FLOPs/timing analysis saved: ${timing_dir}/flops_analysis.md"
    else
        warn "FLOPs/timing analysis failed; see ${timing_dir}/flops_analysis.log"
        return 1
    fi
}

# --------------------------------------------------------------------------- #
# Summary report
# --------------------------------------------------------------------------- #
generate_summary() {
    log "Generating summary report..."
    local summary="${LOG_DIR}/summary.md"
    # 不再生成英文 SUMMARY.md；若残留则删除
    rm -f "${LOG_DIR}/SUMMARY.md"

    {
        echo "# Qwen3-30B-A3B ${FT_DISPLAY_NAME}性能测试报告"
        echo ""
        echo "**测试时间**: ${RUN_TIME}"
        echo "**运行标签**: ${RUN_LABEL}"
        echo "**微调方式**: ${FT_MODE}"
        echo "**日志目录**: \`${LOG_DIR}\`"
        echo "**训练日志**: \`${LOG_DIR}/phase4/${TRAIN_LOG_NAME}\`"
        echo "**GPU 数量**: ${NUM_GPUS}（后端: AMXBF16，2 NUMA）"
        echo "**模型**: Qwen3MoeForCausalLM（纯文本，48 层，128 experts）"
        echo "**数据集**: ${DATASET_NAME}（100 条，样本 >7000 tokens，截断至 ${CUTOFF_LEN}）"
        echo "**Batch/GAS/学习率**: ${TRAIN_BATCH_SIZE} / ${GRAD_ACCUM_STEPS} / ${LEARNING_RATE}"
        echo "**性能步数**: ${PHASE4_STEPS}（前 ${WARMUP_SKIP} 步 warmup，后 $((PHASE4_STEPS - WARMUP_SKIP)) 步计算 TPS）"
        echo ""
        echo "## 1. Phase 退出码"
        echo ""
        echo "| Phase | 退出码 | 状态 |"
        echo "|-------|--------|------|"

        for ph in phase4; do
            local ec_file="${LOG_DIR}/${ph}/exit_code.txt"
            if [[ -f "${ec_file}" ]]; then
                local ec
                ec=$(cat "${ec_file}")
                local status="通过"
                [[ "${ec}" -ne 0 ]] && status="失败(${ec})"
                echo "| ${ph} | ${ec} | ${status} |"
            else
                echo "| ${ph} | - | 跳过 |"
            fi
        done

        echo ""
        echo "## 2. TPS 摘要"
        echo ""
        if [[ -f "${LOG_DIR}/phase4_analysis.txt" ]]; then
            grep -E "TPS|tokens_per_step|stable_steps|step_time" \
                "${LOG_DIR}/phase4_analysis.txt" 2>/dev/null \
                | sed 's/^/  /' || echo "  （运行 Phase 4 后可获得 TPS 数据）"
        else
            echo "  （无 phase4_analysis.txt）"
        fi

        echo ""
        echo "## 3. 内存归因"
        echo ""
        echo "- 静态分量估计: \`${LOG_DIR}/memory_component_estimate.txt\`"
        echo "- 运行时时间线: \`${LOG_DIR}/memory_component_timeline.csv\`"
        echo "- 观测摘要: \`${LOG_DIR}/memory_component_observed.txt\`"
        echo ""
        if [[ -f "${LOG_DIR}/memory_component_observed.txt" ]]; then
            grep -E "peak_|baseline_|timeline_csv" "${LOG_DIR}/memory_component_observed.txt" 2>/dev/null \
                | sed 's/^/  /' || true
        fi

        echo ""
        echo "## 4. ${FT_DISPLAY_NAME}逐步计时"
        echo ""
        echo "- 逐步 JSON: \`${LOG_DIR}/phase4/step_timing/step_timing.json\`"
        echo "- 逐步 CSV: \`${LOG_DIR}/phase4/step_timing/step_timing.csv\`"
        echo "- 逐步 Markdown: \`${LOG_DIR}/phase4/step_timing/step_timing.md\`"
        echo "- Job 计时 JSON: \`${LOG_DIR}/phase4/step_timing/job_timing.json\`"
        echo "- 理论 FLOPs JSON: \`${LOG_DIR}/phase4/step_timing/flops_analysis.json\`"
        echo "- 理论 FLOPs Markdown: \`${LOG_DIR}/phase4/step_timing/flops_analysis.md\`"
        echo "- backward 内部计时模式: \`${BACKWARD_TIMING_MODE}\`（层过滤: \`${BACKWARD_TIMING_LAYERS}\`）"
        echo "- backward summary: \`${LOG_DIR}/phase4/step_timing/backward_internal/backward_timing.json\` / \`backward_step.csv\` / \`backward_timing.md\`"
        if [[ "${BACKWARD_TIMING_MODE}" == "trace" ]]; then
            echo "- backward trace: \`${LOG_DIR}/phase4/step_timing/backward_internal/backward_layer.csv\` / \`backward_numa.csv\`"
        fi
        if [[ "${NUM_GPUS}" -gt 1 ]]; then
            echo "- 多进程运行时上述 backward 文件在扩展名前带 \`.rankN\` 后缀，避免各 rank 覆盖。"
        fi
        "${PYTHON}" - <<PYEOF | sed 's/^## /### /'
import json
import sys
from pathlib import Path

sys.path.insert(0, "${TIMING_MODULE_DIR}")
from step_timing_probe import (
    parse_job_timing_from_train_log,
    render_summary_timing_section,
)

log_dir = Path("${LOG_DIR}")
step_path = log_dir / "phase4" / "step_timing" / "step_timing.json"
train_log = log_dir / "phase4" / "${TRAIN_LOG_NAME}"
out_dir = log_dir / "phase4" / "step_timing"
out_dir.mkdir(parents=True, exist_ok=True)

step_timing = None
if step_path.exists():
    step_timing = json.loads(step_path.read_text())

job_timing = parse_job_timing_from_train_log(train_log)
(out_dir / "job_timing.json").write_text(json.dumps(job_timing, indent=2, ensure_ascii=False))
print(render_summary_timing_section(step_timing, job_timing, finetune_mode="${FT_MODE}"))
PYEOF

        if [[ -f "${LOG_DIR}/phase4/step_timing/flops_analysis.md" ]]; then
            echo ""
            sed -e '1s/^# /## 5. /' -e '2,$s/^## /### /' \
                "${LOG_DIR}/phase4/step_timing/flops_analysis.md"
        else
            echo ""
            echo "## 5. 理论 FLOPs 与耗时校验"
            echo ""
            echo "- FLOPs 报告不可用；检查训练是否完成 warmup 后的稳定 steps。"
        fi

        echo ""
        echo "## 6. 基础健康检查"
        echo ""
        echo "| 编号 | 问题 | 检测结论 |"
        echo "|------|------|---------|"

        local p2_status="待分析"
        if [[ -f "${LOG_DIR}/phase4_analysis.txt" ]]; then
            p2_status=$(grep "step_time_avg\|avg_sec_per_step" \
                        "${LOG_DIR}/phase4_analysis.txt" 2>/dev/null | head -1 | xargs || true)
            [[ -z "${p2_status}" ]] && p2_status="待分析"
        fi
        echo "| P2 | CPU backward 速度瓶颈 | ${p2_status} |"

        local p5_status="正常"
        for ph in phase4; do
            local lf="${LOG_DIR}/${ph}/${TRAIN_LOG_NAME}"
            if [[ -f "${lf}" ]] && grep -qi "sigsegv\|segmentation" "${lf}"; then
                p5_status="⚠ 检测到崩溃"
                break
            fi
        done
        echo "| P5 | SIGSEGV / NaN | ${p5_status} |"
        echo "| P7 | update_base_weights 开销 | 见 phase4_analysis.txt / step_timing |"

        echo ""
        echo "## 7. 可视化输出"
        echo ""
        echo "- GPU 显存: \`${LOG_DIR}/plots/01_gpu_memory.png\`"
        echo "- CPU 内存: \`${LOG_DIR}/plots/02_cpu_ram.png\`"
        echo "- TPS: \`${LOG_DIR}/plots/03_tps.png\`"
    } > "${summary}"

    ok "Summary saved: ${summary}"
}

# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
cleanup() {
    stop_monitor
    [[ -n "${MONITOR_FIFO}" && -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
    return 0
}

trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# Run one method from setup through reporting.  This function returns only
# after all child processes for the method have stopped, which guarantees that
# Full and LoRA never execute concurrently.
# --------------------------------------------------------------------------- #
run_finetune_task() {
    local requested_mode="$1"
    local task_exit=0
    setup_task_dir "${requested_mode}"

    phase_banner "START ${FT_DISPLAY_NAME} (sequential task)"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        generate_memory_component_estimate
        start_monitor
    fi

    run_phase4 || task_exit=$?

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        stop_monitor
        generate_memory_component_timeline

        if [[ -f "${ANALYZE_SCRIPT}" ]]; then
            log "Running visualization analysis for ${FT_MODE}..."
            "${PYTHON}" "${ANALYZE_SCRIPT}" --log-dir "${LOG_DIR}" \
                >> "${LOG_DIR}/analyze.log" 2>&1 && \
                ok "Charts generated: ${LOG_DIR}/plots/" || \
                warn "analyze.py failed, run manually: python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
        fi

        # analyze.py also writes summary.md; write our mode-aware summary last.
        generate_summary
    fi

    if [[ "${task_exit}" -eq 0 ]]; then
        ok "${FT_DISPLAY_NAME} completed — ${LOG_DIR}"
    else
        warn "${FT_DISPLAY_NAME} failed (${task_exit}) — ${LOG_DIR}"
    fi
    return "${task_exit}"
}

generate_session_config() {
    "${PYTHON}" - \
        "${SESSION_DIR}" "${FINETUNE_SELECTION}" "${NUM_GPUS}" \
        "${TRAIN_BATCH_SIZE}" "${GRAD_ACCUM_STEPS}" "${CUTOFF_LEN}" \
        "${PHASE4_STEPS}" "${WARMUP_SKIP}" "${LEARNING_RATE}" \
        "${LORA_RANK}" "${LORA_ALPHA}" "$(basename "${ACCEL_CONFIG}")" \
        "${OMP_NUM_THREADS}" "${BACKWARD_TIMING_MODE}" "${BACKWARD_TIMING_LAYERS}" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) / "session_config.json"
obj = {
    "selection": sys.argv[2],
    "execution_order": ["full", "lora"] if sys.argv[2] == "both" else [sys.argv[2]],
    "shared_training_parameters": {
        "num_gpus": int(sys.argv[3]),
        "per_device_train_batch_size": int(sys.argv[4]),
        "gradient_accumulation_steps": int(sys.argv[5]),
        "cutoff_len": int(sys.argv[6]),
        "max_steps": int(sys.argv[7]),
        "warmup_skip": int(sys.argv[8]),
        "stable_step_interval": [int(sys.argv[8]) + 1, int(sys.argv[7])],
        "learning_rate": sys.argv[9],
        "accelerate_config": sys.argv[12],
        "omp_num_threads": int(sys.argv[13]),
        "backward_timing_mode": sys.argv[14],
        "backward_timing_layers": sys.argv[15],
    },
    "lora_only_parameters": {
        "lora_rank": int(sys.argv[10]),
        "lora_alpha": int(sys.argv[11]),
        "lora_target": "all",
    },
}
out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
PYEOF
}

generate_session_summary() {
    "${PYTHON}" - "${SESSION_DIR}" "${FINETUNE_SELECTION}" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
selection = sys.argv[2]
modes = ["full", "lora"] if selection == "both" else [selection]
tags = {"full": "full_ft", "lora": "lora_ft"}
log_names = {"full": "train_full_ft.log", "lora": "train_lora_ft.log"}
common_keys = (
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_steps",
    "cutoff_len",
)

def simple_yaml(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*(?:#.*)?$", line)
        if m:
            values[m.group(1)] = m.group(2).strip("'\"")
    return values

rows = []
configs = {}
for mode in modes:
    task = root / tags[mode]
    cfg = simple_yaml(task / "phase4" / "train_config.yaml")
    configs[mode] = cfg
    timing_path = task / "phase4" / "step_timing" / "step_timing.json"
    flops_path = task / "phase4" / "step_timing" / "flops_analysis.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    flops = json.loads(flops_path.read_text()) if flops_path.exists() else {}
    stable = timing.get("aggregate_stable") or {}
    sanity = flops.get("observed_phase_sanity") or {}
    rows.append({
        "mode": mode,
        "exit": (task / "phase4" / "exit_code.txt").read_text().strip()
            if (task / "phase4" / "exit_code.txt").exists() else "-",
        "stable_steps": timing.get("num_stable_steps", "-"),
        "tps": (timing.get("tps_attribution") or {}).get("stable_tps"),
        "forward": (stable.get("forward_sec") or {}).get("mean_sec"),
        "backward": (stable.get("backward_sec") or {}).get("mean_sec"),
        "optimizer": (stable.get("optimizer_sec") or {}).get("mean_sec"),
        "judgements": "/".join(
            (sanity.get(p) or {}).get("status_cn", "-")
            for p in ("forward", "backward", "optimizer")
        ),
        "train_log": task / "phase4" / log_names[mode],
    })

consistent = True
mismatches = []
if len(modes) == 2:
    for key in common_keys:
        left, right = configs["full"].get(key), configs["lora"].get(key)
        if left != right:
            consistent = False
            mismatches.append(f"{key}: full={left!r}, lora={right!r}")

def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"

session_cfg = json.loads((root / "session_config.json").read_text())
shared = session_cfg["shared_training_parameters"]
lines = [
    "# Qwen3-30B-A3B 微调性能测试会话",
    "",
    f"- 选择: **{selection}**",
    f"- 串行顺序: `{' -> '.join(session_cfg['execution_order'])}`",
    f"- 公共参数一致性: **{'通过' if consistent else '失败'}**",
    f"- GPU/batch/GAS: {shared['num_gpus']} / {shared['per_device_train_batch_size']} / {shared['gradient_accumulation_steps']}",
    f"- steps/warmup/稳定区间: {shared['max_steps']} / {shared['warmup_skip']} / {shared['stable_step_interval'][0]}-{shared['stable_step_interval'][1]}",
    f"- learning_rate: {shared['learning_rate']}",
    "",
]
if mismatches:
    lines += ["## 参数不一致", ""] + [f"- {x}" for x in mismatches] + [""]
lines += [
    "## 稳定区间对比",
    "",
    "| 方式 | 退出码 | 稳定 steps | TPS | forward 均值(s) | backward 均值(s) | optimizer 均值(s) | FLOPs 判断(fwd/bwd/opt) |",
    "|------|-------:|-------------:|----:|----------------:|-----------------:|------------------:|-------------------------|",
]
for row in rows:
    lines.append(
        f"| {row['mode']} | {row['exit']} | {row['stable_steps']} | {fmt(row['tps'], 2)} | "
        f"{fmt(row['forward'])} | {fmt(row['backward'])} | {fmt(row['optimizer'])} | {row['judgements']} |"
    )
lines += ["", "## 训练日志", ""]
for row in rows:
    lines.append(f"- {row['mode']}: `{row['train_log']}`")
lines += [
    "",
    "> FLOPs 判断是基于可配置理论峰值的 roofline sanity check；低利用率表示需要定位热点，不等同于单独证明实现错误。",
    "",
]
(root / "comparison.md").write_text("\n".join(lines) + "\n")
PYEOF
    ok "Session comparison saved: ${SESSION_DIR}/comparison.md"
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
main() {
    echo ""
    echo -e "${BOLD}Qwen3-30B-A3B KTransformers Fine-Tuning Performance Test (AMXBF16)${NC}"
    echo -e "Time: $(run_time_display)"
    echo ""

    check_env
    estimate_resources
    setup_session_dir
    generate_session_config

    local overall_exit=0
    case "${FINETUNE_SELECTION}" in
        both)
            run_finetune_task full || overall_exit=1
            run_finetune_task lora || overall_exit=1
            ;;
        full|lora)
            run_finetune_task "${FINETUNE_SELECTION}" || overall_exit=1
            ;;
    esac

    generate_session_summary

    echo ""
    if [[ "${overall_exit}" -eq 0 ]]; then
        ok "Selected fine-tuning task(s) completed — ${SESSION_DIR}"
    else
        warn "One or more tasks failed — see ${SESSION_DIR}/comparison.md"
    fi
    return "${overall_exit}"
}

main "$@"

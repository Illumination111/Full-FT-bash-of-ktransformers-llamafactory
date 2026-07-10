#!/usr/bin/env bash
# =============================================================================
# Qwen3-30B-A3B Full-FT ablation: train EXPERT BASE WEIGHTS ONLY
#
# Based on run_full_ft_test_1gpu_bf16.sh, with one critical difference:
#   * Freeze ALL non-expert parameters (attention, embed, lm_head, router, LoRA).
#   * Only KT expert base buffers remain trainable:
#       gate_proj_buf / up_proj_buf / down_proj_buf
#   * Still uses ACCELERATE_KT_TRAIN_MODE=full (kt_full_weight_grad=True).
#
# Purpose (see docs/debug_FFT.md):
#   If expert base grads are broken (full_weight_grad lost in TP slicing),
#   train_loss should stay flat when only expert base is trainable.
#   A flat loss curve is strong supporting evidence for that root cause.
#
# Usage:
#   bash run_full_ft_test_1gpu_bf16_frozen.sh [--gpus 1] [--phase4-steps 15] \
#                                              [--gas 1] [--dry-run]
#
#   --gas N   gradient_accumulation_steps (default: 1)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# REAL full fine-tuning switch + expert-base-only freeze ablation
# --------------------------------------------------------------------------- #
export ACCELERATE_KT_TRAIN_MODE=full
# Force pure full mode (no hybrid LoRA buffers on experts)
export ACCELERATE_KT_LORA_RANK=0
# Enable freeze monkey-patch (see freeze_non_expert_base.py)
export KT_FREEZE_NON_EXPERT_BASE=1

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${SCRIPT_DIR}/test_log"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
# Shared dataset directory (real full-FT dataset lives here, not per-model data/)
DATA_DIR="/mnt/data2/wbw/FFTtest/dataset"
DATASET_NAME="fft_real_100"
GEN_DATASET_SCRIPT="${DATA_DIR}/gen_dataset.py"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze.py"

LLAMA_FACTORY_DIR="/mnt/data2/wbw/LLaMA-Factory"
MODEL_PATH="/mnt/data3/models/Qwen3-30B-A3B"    # Original HF BF16 weights (AMXBF16 reads directly)

# Default: 1 GPU; auto-fallback to 2 GPU on OOM
ACCEL_CONFIG_1GPU="${CONFIGS_DIR}/accelerate_fft_amxbf16_1gpu.yaml"
ACCEL_CONFIG_2GPU="${CONFIGS_DIR}/accelerate_fft_amxbf16_2gpu.yaml"
ACCEL_CONFIG="${ACCEL_CONFIG_1GPU}"

TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_full_ft_qwen3_30b.yaml"

# TPS measurement constants (must match train config)
CUTOFF_LEN=1024
WARMUP_SKIP=5    # Steps to discard before computing stable TPS
GRAD_ACCUM_STEPS=1   # Default GAS; override via --gas N

MONITOR_FIFO="/tmp/fft_monitor_events.fifo"
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

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
NUM_GPUS=1                                                      # Single-card by default
DRY_RUN=0
PHASE4_STEPS=15
OOM_FALLBACK_DONE=0   # Prevent infinite retry on OOM
BACKEND_LABEL="AMX_BF16_FULLFT_EXPERTONLY"
# In-training expert base-weight probe (samples gate/up/down_proj_buf).
# Do NOT compare HF checkpoint expert weights: those are zero-storage placeholders.
EXPERT_CHECK_SAMPLES="${EXPERT_CHECK_SAMPLES:-12}"
EXPERT_CHECK_SEED="${EXPERT_CHECK_SEED:-20260709}"
EXPERT_CHECK_ATOL="${EXPERT_CHECK_ATOL:-0.0}"
PROBE_MODULE_DIR="${SCRIPT_DIR}"
# Expert-base-only entry: installs freeze_non_expert_base before LLaMA-Factory
TRAIN_ENTRY_MODULE="fft_train_expert_only"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) NUM_GPUS="$2"; shift
                if [[ "${NUM_GPUS}" -ge 2 ]]; then
                    ACCEL_CONFIG="${ACCEL_CONFIG_2GPU}"
                fi ;;
        --phase4-steps) PHASE4_STEPS="$2"; shift ;;
        --gas|--gradient-accumulation-steps)
            GRAD_ACCUM_STEPS="$2"; shift
            if ! [[ "${GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
                echo "Invalid --gas value: ${GRAD_ACCUM_STEPS} (must be positive integer)"
                exit 1
            fi ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

RUN_LABEL="${NUM_GPUS}gpu_${BACKEND_LABEL}"

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

    log "Python version: $(${PYTHON} --version 2>&1)"

    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi not found, please verify CUDA driver is installed"
        exit 1
    fi
    local gpu_list
    gpu_list="$(nvidia-smi -L 2>/dev/null || true)"
    ACTUAL_GPUS=$(printf '%s\n' "${gpu_list}" | sed '/^$/d' | wc -l)
    log "Detected GPUs: ${ACTUAL_GPUS}"
    if [[ "${ACTUAL_GPUS}" -lt "${NUM_GPUS}" ]]; then
        error "System GPU count (${ACTUAL_GPUS}) is less than requested ${NUM_GPUS}"
        exit 1
    fi

    # Model path (AMXBF16 reads original HF BF16 weights directly)
    if [[ ! -d "${MODEL_PATH}" ]]; then
        error "Model path not found: ${MODEL_PATH}"
        exit 1
    fi
    ok "Model path (AMXBF16): ${MODEL_PATH}"

    if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
        error "LLaMA-Factory directory not found: ${LLAMA_FACTORY_DIR}"
        exit 1
    fi
    ok "LLaMA-Factory: ${LLAMA_FACTORY_DIR}"

    # Dataset check (real full-FT dataset: every sample >7000 tokens, cutoff 1024)
    if [[ ! -f "${DATA_DIR}/${DATASET_NAME}.json" ]]; then
        warn "Dataset not found: ${DATA_DIR}/${DATASET_NAME}.json"
        warn "Generating dataset automatically..."
        "${PYTHON}" "${GEN_DATASET_SCRIPT}" || {
            error "Dataset generation failed. Please run: python3 ${GEN_DATASET_SCRIPT}"
            exit 1
        }
    fi
    ok "Dataset: ${DATA_DIR}/${DATASET_NAME}.json"

    # Config files (must exist before any phase starts)
    for cfg_file in "${TRAIN_CONFIG_BASE}" "${ACCEL_CONFIG}"; do
        if [[ ! -f "${cfg_file}" ]]; then
            error "Required config not found: ${cfg_file}"
            exit 1
        fi
    done
    ok "Training config: ${TRAIN_CONFIG_BASE}"
    ok "Accelerate config: ${ACCEL_CONFIG}"

    log "Python executable: ${PYTHON}"
    log "Conda env: ${CONDA_ENV}"
    log "GPU count: ${NUM_GPUS} (accelerate config: $(basename ${ACCEL_CONFIG}))"

    if ! "${PYTHON}" -c "import accelerate" &>/dev/null; then
        error "accelerate not installed in ${PYTHON}"
        exit 1
    fi
    ok "accelerate $(${PYTHON} -c 'import accelerate; print(accelerate.__version__)')"

    for pkg in psutil pynvml matplotlib pandas; do
        if "${PYTHON}" -c "import ${pkg}" &>/dev/null; then
            ok "${pkg} installed"
        else
            warn "${pkg} not installed (some features limited)"
        fi
    done

    ok "Environment check passed"
}

# --------------------------------------------------------------------------- #
# Resource estimation
# --------------------------------------------------------------------------- #
estimate_resources() {
    phase_banner "Resource and Time Estimation (Qwen3-30B-A3B FFT, AMXBF16)"

    echo -e "${BOLD}== Model Architecture ==${NC}"
    echo "  Architecture : Qwen3MoeForCausalLM  (pure text MoE)"
    echo "  Layers       : 48   |  Experts/layer: 128  |  Active experts/token: 8"
    echo "  Quantization : AMXBF16 (native BF16, no quantization)"
    echo "  CPU layout   : 2 NUMA nodes, kt_tp_enabled=true (48 threads each)"
    echo "  Weight source: ${MODEL_PATH}"
    echo ""

    echo -e "${BOLD}== GPU Memory Estimation (${NUM_GPUS} x RTX 4090, 24 GB each) ==${NC}"
    echo "  AMXBF16: expert weights on CPU (BF16, ~58 GB); GPU holds non-expert params only"
    echo "  Non-expert GPU params (attention + embed + shared expert): ~6 GB total"
    if [[ "${NUM_GPUS}" -eq 1 ]]; then
        echo "  Single-GPU FSDP (no sharding): full ~6 GB params on 1 card"
        echo "  + activations (seq=1024, grad-ckpt): ~4-6 GB"
        echo "  + gradients: ~6 GB"
        echo "  + AdamW optimizer (non-expert): ~12 GB"
        echo "  ──────────────────────────────────────────────"
        echo "  Estimated peak VRAM / card: ~20-24 GB  (close to 24 GB limit)"
        warn "  Single card may OOM — script will auto-retry with 2 GPUs if that happens"
    else
        echo "  2-GPU FSDP sharding: ~3 GB params per card"
        echo "  + activations (seq=1024, grad-ckpt): ~4-6 GB"
        echo "  + gradients: ~3 GB"
        echo "  + optimizer: ~6 GB"
        echo "  ──────────────────────────────────────────────"
        echo "  Estimated peak VRAM / card: ~12-18 GB  (comfortable)"
    fi
    echo ""

    echo -e "${BOLD}== CPU Memory Estimation (2 NUMA nodes) ==${NC}"
    echo "  BF16 expert weights (per NUMA copy): ~58 GB × 2 NUMA ≈ ~116 GB"
    echo "  BF16 expert gradient buffers: ~58 GB"
    echo "  AdamW expert optimizer states (m+v): ~116 GB"
    echo "  NUMA backward working buffers (tp_part 0+1): ~30-50 GB"
    echo "  ──────────────────────────────────────────────"
    if [[ "${NUM_GPUS}" -eq 1 ]]; then
        echo "  Estimated CPU memory peak: ~400-450 GB  (1 process, 2 NUMA)"
    else
        echo "  Estimated CPU memory peak: ~500-600 GB  (2 processes × 2 NUMA overhead)"
    fi
    echo ""

    echo -e "${BOLD}== TPS Measurement ==${NC}"
    echo "  Dataset: 100 samples, all truncated to exactly ${CUTOFF_LEN} tokens"
    echo "  GAS (gradient_accumulation_steps): ${GRAD_ACCUM_STEPS}"
    echo "  Tokens per optimizer step: ${NUM_GPUS} GPU(s) × ${CUTOFF_LEN} × GAS=${GRAD_ACCUM_STEPS} = $((NUM_GPUS * CUTOFF_LEN * GRAD_ACCUM_STEPS)) tokens"
    echo "  Warmup steps excluded: ${WARMUP_SKIP}"
    echo "  TPS = $((NUM_GPUS * CUTOFF_LEN * GRAD_ACCUM_STEPS)) / avg_stable_step_time"
    echo ""
}

# --------------------------------------------------------------------------- #
# Create dated run directory + event FIFO
# --------------------------------------------------------------------------- #
setup_run_dir() {
    RUN_TS=$(run_timestamp)
    RUN_TIME="$(run_time_display)"
    RUN_TIMEZONE="$(date '+%Z %z')"
    RUN_LABEL="${NUM_GPUS}gpu_${BACKEND_LABEL}"
    LOG_DIR="${LOG_BASE}/${RUN_TS}_${RUN_LABEL}"
    mkdir -p "${LOG_DIR}/plots"
    log "Log directory: ${LOG_DIR}"

    [[ -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
    mkfifo "${MONITOR_FIFO}"

    export LOG_DIR RUN_TS RUN_TIME RUN_TIMEZONE RUN_LABEL MONITOR_FIFO
}

send_event() {
    local msg="$1"
    if [[ -p "${MONITOR_FIFO}" ]]; then
        echo "${msg}" >> "${MONITOR_FIFO}" 2>/dev/null || true
    fi
}

# --------------------------------------------------------------------------- #
# Start / stop monitoring
# --------------------------------------------------------------------------- #
start_monitor() {
    log "Starting system monitor process (process-tree root PID=$$)..."
    "${PYTHON}" "${MONITOR_SCRIPT}" \
        --out "${LOG_DIR}/monitor.csv" \
        --fifo "${MONITOR_FIFO}" \
        --interval 2 \
        --pid $$ \
        >> "${LOG_DIR}/monitor.log" 2>&1 &
    MONITOR_PID=$!
    log "Monitor PID: ${MONITOR_PID} (tracking script tree $$)"
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
}

# --------------------------------------------------------------------------- #
# Delete model_output/ for a phase to reclaim disk space.
# LLaMA-Factory always calls trainer.save_model() at the end of run_sft(),
# regardless of save_strategy.  Each saved checkpoint is ~115 GB for this
# model, so we delete it immediately after a phase no longer needs it.
#
# Expert-update verification does NOT depend on this checkpoint: it samples
# KT gate/up/down_proj_buf during training (see expert_buf_probe.py). HF
# checkpoint expert weights are zero-storage placeholders and are not used.
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

# --------------------------------------------------------------------------- #
# Run a single accelerate training command
# Supports OOM auto-fallback from 1 GPU to 2 GPUs (one retry only).
# --------------------------------------------------------------------------- #
run_train() {
    local phase_name="$1"
    local desc="$2"
    local train_cfg="$3"
    local phase_log="${LOG_DIR}/${phase_name}/train.log"
    local exit_code=0

    log "Starting training [${phase_name}]: ${desc}"
    log "  accelerate config : $(basename ${ACCEL_CONFIG})"
    log "  GPUs              : ${NUM_GPUS}"
    send_event "phase:${phase_name}"
    send_event "event:train_start"

    local gpus_str
    gpus_str=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',')

    local accelerate_bin="${CONDA_BIN_DIR}/accelerate"
    [[ ! -x "${accelerate_bin}" ]] && accelerate_bin="accelerate"

    local probe_out="${LOG_DIR}/${phase_name}/expert_buf_probe.json"
    local timing_dir="${LOG_DIR}/${phase_name}/step_timing"
    local cmd=(
        env
        USE_KT=1
        ACCELERATE_USE_KT=true
        ACCELERATE_KT_TRAIN_MODE=full
        ACCELERATE_KT_LORA_RANK=0
        KT_FREEZE_NON_EXPERT_BASE=1
        KT_EXPERT_BUF_PROBE=1
        KT_EXPERT_BUF_PROBE_OUT="${probe_out}"
        KT_EXPERT_BUF_PROBE_SAMPLES="${EXPERT_CHECK_SAMPLES}"
        KT_EXPERT_BUF_PROBE_SEED="${EXPERT_CHECK_SEED}"
        KT_EXPERT_BUF_PROBE_ATOL="${EXPERT_CHECK_ATOL}"
        KT_STEP_TIMING=1
        KT_STEP_TIMING_OUT_DIR="${timing_dir}"
        KT_STEP_TIMING_WARMUP_SKIP="${WARMUP_SKIP}"
        KT_STEP_TIMING_TOKENS_PER_STEP="$((NUM_GPUS * CUTOFF_LEN * GRAD_ACCUM_STEPS))"
        KT_STALL_WATCH=1
        KT_STALL_WATCH_OUT_DIR="${timing_dir}"
        KT_STALL_WATCH_IDLE_SEC="${STALL_WATCH_IDLE_SEC:-5}"
        KT_STALL_WATCH_PHASES="${STALL_WATCH_PHASES:-optimizer,backward,post_optim}"
        PYTHONPATH="${PROBE_MODULE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
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
    "${cmd[@]}" 2>&1 | tee "${phase_log}"
    exit_code=${PIPESTATUS[0]}
    set -e
    popd > /dev/null

    send_event "event:train_end"

    # ---- OOM auto-fallback: 1 GPU → 2 GPUs (one attempt only) ----
    if [[ "${exit_code}" -ne 0 ]] && [[ "${OOM_FALLBACK_DONE}" -eq 0 ]]; then
        if grep -qi "out of memory\|cuda error.*out of memory\|oom\|cudamalloc failed" "${phase_log}" 2>/dev/null; then
            if [[ "${NUM_GPUS}" -eq 1 ]]; then
                warn "OOM detected on single GPU — switching to 2-GPU config and retrying..."
                NUM_GPUS=2
                ACCEL_CONFIG="${ACCEL_CONFIG_2GPU}"
                OOM_FALLBACK_DONE=1

                # Rename current log, retry in the same phase dir
                mv "${phase_log}" "${phase_log%.log}_oom_1gpu.log"
                log "Retrying [${phase_name}] with 2 GPUs..."

                gpus_str=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',')
                cmd=(
                    env
                    USE_KT=1
                    ACCELERATE_USE_KT=true
                    ACCELERATE_KT_TRAIN_MODE=full
                    ACCELERATE_KT_LORA_RANK=0
                    KT_FREEZE_NON_EXPERT_BASE=1
                    KT_EXPERT_BUF_PROBE=1
                    KT_EXPERT_BUF_PROBE_OUT="${probe_out}"
                    KT_EXPERT_BUF_PROBE_SAMPLES="${EXPERT_CHECK_SAMPLES}"
                    KT_EXPERT_BUF_PROBE_SEED="${EXPERT_CHECK_SEED}"
                    KT_EXPERT_BUF_PROBE_ATOL="${EXPERT_CHECK_ATOL}"
                    KT_STEP_TIMING=1
                    KT_STEP_TIMING_OUT_DIR="${timing_dir}"
                    KT_STEP_TIMING_WARMUP_SKIP="${WARMUP_SKIP}"
                    KT_STEP_TIMING_TOKENS_PER_STEP="$((NUM_GPUS * CUTOFF_LEN * GRAD_ACCUM_STEPS))"
                    KT_STALL_WATCH=1
                    KT_STALL_WATCH_OUT_DIR="${timing_dir}"
                    KT_STALL_WATCH_IDLE_SEC="${STALL_WATCH_IDLE_SEC:-5}"
                    KT_STALL_WATCH_PHASES="${STALL_WATCH_PHASES:-optimizer,backward,post_optim}"
                    PYTHONPATH="${PROBE_MODULE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
                    CUDA_VISIBLE_DEVICES="${gpus_str}"
                    "${accelerate_bin}" launch
                    --config_file "${ACCEL_CONFIG}"
                    -m "${TRAIN_ENTRY_MODULE}" train
                    "${train_cfg}"
                )
                pushd "${LLAMA_FACTORY_DIR}" > /dev/null
                set +e
                "${cmd[@]}" 2>&1 | tee "${phase_log}"
                exit_code=${PIPESTATUS[0]}
                set -e
                popd > /dev/null

                if [[ "${exit_code}" -eq 0 ]]; then
                    ok "2-GPU retry succeeded for [${phase_name}]"
                else
                    error "2-GPU retry also failed for [${phase_name}] (exit code ${exit_code})"
                fi
            fi
        fi
    fi

    return "${exit_code}"
}

# --------------------------------------------------------------------------- #
# Analyze training log (extract loss, grad_norm, NaN detection)
# --------------------------------------------------------------------------- #
analyze_log() {
    local phase_name="$1"
    local log_file="${LOG_DIR}/${phase_name}/train.log"
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
        nan_count=$(grep -ci "nan" "${log_file}" 2>/dev/null || echo 0)
        inf_count=$(grep -ci "inf" "${log_file}" 2>/dev/null || echo 0)
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
    } | tee "${result_file}"
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
    phase_banner "Memory Component Attribution Estimate"

    "${PYTHON}" - "${MODEL_PATH}" "${NUM_GPUS}" "${CUTOFF_LEN}" "${LOG_DIR}" <<'PYEOF' \
        | tee "${LOG_DIR}/memory_component_estimate.txt"
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
num_gpus = int(sys.argv[2])
cutoff_len = int(sys.argv[3])
log_dir = Path(sys.argv[4])

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
moe_i = int(cfg["moe_intermediate_size"])
E = int(cfg["num_experts"])
L = int(cfg["num_hidden_layers"])
top_k = int(cfg["num_experts_per_tok"])
V = int(cfg["vocab_size"])

expert_params = E * 3 * H * moe_i * L
expert_weight_bytes = expert_params * 2
expert_grad_bytes = expert_params * 4
expert_adam_bytes = expert_params * 8

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
gpu_gradients = gb((non_expert_weight_bytes / 2) * 4 / num_gpus)
gpu_optimizer = gb((non_expert_weight_bytes / 2) * 8 / num_gpus)
gpu_activations = 2.0

estimate = {
    "notes": [
        "Measured totals come from monitor.csv.",
        "Component columns are model-structure estimates; nvidia-smi/psutil do not expose optimizer-vs-gradient ownership.",
        "Optimizer estimates assume AdamW m+v FP32 states.",
        "Gradient estimates assume FP32 gradients for trainable tensors.",
    ],
    "model": {
        "path": str(model_path),
        "hidden_size": H,
        "num_layers": L,
        "num_experts": E,
        "active_experts_per_token": top_k,
        "cutoff_len": cutoff_len,
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
print(f"architecture        : {cfg.get('architectures', ['?'])[0]}")
print(f"layers/experts/topk : {L} / {E} / {top_k}")
print(f"cutoff_len          : {cutoff_len}")
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
}

generate_memory_component_timeline() {
    phase_banner "Runtime Memory Attribution Timeline"

    "${PYTHON}" - "${LOG_DIR}" <<'PYEOF' | tee "${LOG_DIR}/memory_component_observed.txt"
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
}

verify_expert_weight_changes() {
    phase_banner "Expert Base-Weight Probe (in-training proj_buf sample)"

    local probe_json="${LOG_DIR}/phase4/expert_buf_probe.json"
    local report_txt="${LOG_DIR}/expert_weight_change_check.txt"
    local report_json="${LOG_DIR}/expert_weight_change_check.json"

    "${PYTHON}" - \
        "${probe_json}" \
        "${report_json}" <<'PYEOF' | tee "${report_txt}"
import json
import shutil
import sys
from pathlib import Path

probe_path = Path(sys.argv[1])
report_json = Path(sys.argv[2])

print("=== Expert Base-Weight Probe (in-training) ===")
print("method                 : sample gate/up/down_proj_buf at train_begin vs train_end")
print("note                   : does NOT use HF checkpoint expert weights")
print(f"probe_json             : {probe_path}")

if not probe_path.exists():
    obj = {
        "status": "ERROR",
        "reason": "in-training probe JSON missing; callback may not have been installed or training failed before train_end",
        "method": "in_training_proj_buf_sample",
        "sampled_tensors": 0,
        "changed_tensors": 0,
    }
    report_json.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"status                 : {obj['status']}")
    print(f"reason                 : {obj['reason']}")
    raise SystemExit(0)

data = json.loads(probe_path.read_text())
shutil.copyfile(probe_path, report_json)

print(f"status                 : {data.get('status')}")
print(f"reason                 : {data.get('reason', '')}")
print(f"sampled_tensors        : {data.get('sampled_tensors', 0)}")
print(f"changed_tensors        : {data.get('changed_tensors', 0)}")
print(f"nonfinite_tensors      : {data.get('nonfinite_tensors', 0)}")
agg = data.get("aggregate") or {}
for key in (
    "changed_tensor_fraction",
    "mean_rel_l2_delta",
    "max_rel_l2_delta",
    "mean_max_abs_delta",
    "max_abs_delta",
):
    if key in agg:
        print(f"{key:23s}: {agg[key]:.8e}")
print(f"json                   : {report_json}")

print()
print("--- Sampled expert buffers ---")
for r in data.get("records", []):
    if r.get("status") != "OK":
        print(f"{r.get('tensor')} | {r.get('status')} | {r.get('reason', '')}")
        continue
    print(
        f"{r['tensor']} | changed={r.get('changed')} | "
        f"rel_l2={r.get('rel_l2_delta', float('nan')):.8e} | "
        f"max_abs={r.get('max_abs_delta', float('nan')):.8e} | "
        f"changed_frac={r.get('changed_element_fraction', 0.0):.8e} | "
        f"finite_after={r.get('finite_after')}"
    )
PYEOF
}

# --------------------------------------------------------------------------- #
# Phase 4: Stability extension + accurate TPS measurement
# --------------------------------------------------------------------------- #
run_phase4() {
    phase_banner "Phase 4: Stability Extension + TPS Benchmark (${PHASE4_STEPS} steps)"
    local cfg
    cfg=$(make_phase_config "phase4" \
        "max_steps=${PHASE4_STEPS}" \
        "gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
        "save_strategy='no'" \
        "logging_steps=1")

    log "Steps: ${PHASE4_STEPS}  |  GPUs: ${NUM_GPUS}  |  cutoff_len: ${CUTOFF_LEN}  |  GAS: ${GRAD_ACCUM_STEPS}"
    log "TPS measurement: skipping first ${WARMUP_SKIP} warmup steps"
    log "tokens/optimizer-step = ${NUM_GPUS} × ${CUTOFF_LEN} × ${GRAD_ACCUM_STEPS} = $((NUM_GPUS * CUTOFF_LEN * GRAD_ACCUM_STEPS))"
    log "Exposes: P2 (CPU speed), P4 (MoE load), P5 (NaN), P6 (Router), P7 (re-quantize)"

    local t_start
    t_start=$(date +%s)
    send_event "phase:phase4"

    local exit_code=0
    run_train "phase4" "stability + TPS benchmark (${PHASE4_STEPS} steps)" "${cfg}" || exit_code=$?

    local t_end
    t_end=$(date +%s)
    local total_sec=$((t_end - t_start))

    {
        echo "=== Phase 4 Performance Analysis ==="
        echo "Total wall time: ${total_sec} seconds"

        local actual_steps
        actual_steps=$(grep -c "{'loss'" "${LOG_DIR}/phase4/train.log" 2>/dev/null || \
                       grep -c '"loss"' "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
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
            "${LOG_DIR}/phase4/train.log" \
            "${CUTOFF_LEN}" \
            "${NUM_GPUS}" \
            "${WARMUP_SKIP}" \
            "${GRAD_ACCUM_STEPS}" <<'PYEOF'
import re, sys, statistics

log_path   = sys.argv[1]
cutoff_len = int(sys.argv[2])
num_gpus   = int(sys.argv[3])
warmup_n   = int(sys.argv[4])
gas        = int(sys.argv[5])

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

tokens_per_step = num_gpus * cutoff_len * gas
stable = [(s, t) for s, t in steps_dedup if s > warmup_n]

print(f"  total_steps_logged : {len(steps_dedup)}")
print(f"  warmup_steps_skip  : {warmup_n}")
print(f"  stable_steps       : {len(stable)}")
print(f"  gas                : {gas}")
print(f"  tokens_per_step    : {tokens_per_step}  ({num_gpus} GPU x {cutoff_len} x GAS={gas})")

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
             "${LOG_DIR}/phase4/train.log" 2>/dev/null | tail -10 || echo "  (no router aux loss found)"

        echo ""
        echo "--- update_base_weights call count (P7) ---"
        local upd_count
        upd_count=$(grep -ci "update_base_weights\|re-quantize\|TP_MOE_SFT" \
                    "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "  update_base_weights triggered: ${upd_count} times"

        echo ""
        echo "--- NaN/Inf statistics (P5) ---"
        local nan_cnt inf_cnt
        nan_cnt=$(grep -ci " nan" "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        inf_cnt=$(grep -ci " inf" "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "  NaN lines: ${nan_cnt};  Inf lines: ${inf_cnt}"
        if [[ "${nan_cnt}" -gt 0 ]] || [[ "${inf_cnt}" -gt 0 ]]; then
            warn "=> numerical anomaly detected, check P5 and P6"
        fi

    } | tee "${LOG_DIR}/phase4_analysis.txt"

    analyze_log "phase4"
    echo "${exit_code}" > "${LOG_DIR}/phase4/exit_code.txt"
    verify_expert_weight_changes
    cleanup_model_output "phase4"
}

# --------------------------------------------------------------------------- #
# Summary report
# --------------------------------------------------------------------------- #
generate_summary() {
    phase_banner "生成中文 Summary 报告"
    local summary="${LOG_DIR}/summary.md"
    # 不再生成英文 SUMMARY.md；若残留则删除
    rm -f "${LOG_DIR}/SUMMARY.md"

    {
        echo "# Qwen3-30B-A3B Full-FT 消融：仅训练 Expert Base Weights"
        echo ""
        echo "**测试时间**: ${RUN_TIME}"
        echo "**运行标签**: ${RUN_LABEL}"
        echo "**日志目录**: \`${LOG_DIR}\`"
        echo "**GPU 数量**: ${NUM_GPUS}（后端: AMXBF16，2 NUMA）"
        echo "**模型**: Qwen3MoeForCausalLM（纯文本，48 层，128 experts）"
        echo "**数据集**: ${DATASET_NAME}（100 条，样本 >7000 tokens，截断至 ${CUTOFF_LEN}）"
        echo "**GAS**: ${GRAD_ACCUM_STEPS}"
        echo "**消融设置**: \`KT_FREEZE_NON_EXPERT_BASE=1\` — 冻结 attention / embed / lm_head / router / LoRA；仅 \`gate/up/down_proj_buf\` 可训练"
        echo "**预期**: 若 \`full_weight_grad\` TP 丢失成立，则 train_loss 应基本不降，且 expert probe \`changed=0\`"
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
        echo "## 4. Full-FT 逐步计时与停顿检测"
        echo ""
        echo "- 逐步 JSON: \`${LOG_DIR}/phase4/step_timing/step_timing.json\`"
        echo "- 逐步 CSV: \`${LOG_DIR}/phase4/step_timing/step_timing.csv\`"
        echo "- 逐步 Markdown: \`${LOG_DIR}/phase4/step_timing/step_timing.md\`"
        echo "- Job 计时 JSON: \`${LOG_DIR}/phase4/step_timing/job_timing.json\`"
        echo "- 停顿检测: \`${LOG_DIR}/phase4/step_timing/stall_events.md\`"
        echo ""
        if [[ -f "${LOG_DIR}/phase4/step_timing/stall_events.json" ]]; then
            echo "### 停顿检测（Stall Watch）"
            echo ""
            "${PYTHON}" - "${LOG_DIR}/phase4/step_timing/stall_events.json" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
print(f"- 状态: {data.get('status')}，事件数: {data.get('num_events', 0)}")
for i, ev in enumerate(data.get("events") or [], 1):
    vm = ev.get("vmstat_delta") or {}
    top = sorted(vm.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top_s = "，".join(f"{k}={v}" for k, v in top) if top else "无 vmstat 增量"
    print(
        f"- 事件{i}: 阶段=`{ev.get('phase')}` step≈{ev.get('global_step_hint')} "
        f"墙钟={ev.get('wall_sec'):.1f}s CPU增量={ev.get('cpu_delta_sec')} "
        f"空闲比={ev.get('idle_ratio')}｜{top_s}"
    )
if not data.get("events"):
    print("- 本轮未检测到停顿事件")
PYEOF
            echo ""
        fi
        "${PYTHON}" - <<PYEOF
import json
import sys
from pathlib import Path

sys.path.insert(0, "${PROBE_MODULE_DIR}")
from step_timing_probe import (
    parse_job_timing_from_train_log,
    render_summary_timing_section,
)

log_dir = Path("${LOG_DIR}")
step_path = log_dir / "phase4" / "step_timing" / "step_timing.json"
train_log = log_dir / "phase4" / "train.log"
out_dir = log_dir / "phase4" / "step_timing"
out_dir.mkdir(parents=True, exist_ok=True)

step_timing = None
if step_path.exists():
    step_timing = json.loads(step_path.read_text())

job_timing = parse_job_timing_from_train_log(train_log)
(out_dir / "job_timing.json").write_text(json.dumps(job_timing, indent=2, ensure_ascii=False))
print(render_summary_timing_section(step_timing, job_timing))
PYEOF

        echo ""
        echo "## 5. Expert 基座权重变化检查"
        echo ""
        echo "- 方法: 训练中采样 KT \`gate/up/down_proj_buf\`（非 HF checkpoint）"
        echo "- 文本报告: \`${LOG_DIR}/expert_weight_change_check.txt\`"
        echo "- JSON 报告: \`${LOG_DIR}/expert_weight_change_check.json\`"
        echo "- 原始 probe: \`${LOG_DIR}/phase4/expert_buf_probe.json\`"
        echo ""
        if [[ -f "${LOG_DIR}/expert_weight_change_check.json" ]]; then
            "${PYTHON}" - "${LOG_DIR}/expert_weight_change_check.json" <<'PYEOF'
import json
import sys

data = json.load(open(sys.argv[1]))
agg = data.get("aggregate", {})
status = data.get("status", "UNKNOWN")
status_cn = {"OK": "通过", "FAIL": "未通过", "ERROR": "错误", "SKIPPED": "跳过", "PASS": "通过", "PARTIAL": "部分通过"}.get(status, status)
print(f"- 状态: **{status_cn}**（{status}）")
reason = data.get("reason", "")
if reason:
    if "none of the sampled" in reason:
        reason = "抽样的 expert 基座 buffer 均未超过 atol 变化"
    elif "probe JSON missing" in reason:
        reason = "缺少 in-training probe JSON（callback 未安装或训练未到 train_end）"
    print(f"- 原因: {reason}")
print(f"- 方法: {data.get('method', 'unknown')}")
print(f"- 抽样张量: {data.get('sampled_tensors', 0)}")
print(f"- 发生变化: {data.get('changed_tensors', 0)}")
print(f"- 非有限值: {data.get('nonfinite_tensors', 0)}")
if agg:
    print(f"- changed_tensor_fraction: {agg.get('changed_tensor_fraction', 0.0):.6e}")
    print(f"- mean_rel_l2_delta: {agg.get('mean_rel_l2_delta', 0.0):.6e}")
    print(f"- max_abs_delta: {agg.get('max_abs_delta', 0.0):.6e}")
PYEOF
        else
            echo "- （未运行 expert 权重检查）"
        fi

        echo ""
        echo "## 5b. Train Loss 轨迹（expert-base-only 消融关键）"
        echo ""
        echo "- 若冻结正确且基座梯度未写入：loss 应近似平坦"
        echo "- 原始日志: \`${LOG_DIR}/phase4/train.log\`"
        echo ""
        if [[ -f "${LOG_DIR}/phase4/train.log" ]]; then
            "${PYTHON}" - "${LOG_DIR}/phase4/train.log" <<'PYEOF'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
# HF / LLaMA-Factory logging: {'loss': 1.23, 'grad_norm': ..., 'epoch': ...}
pat = re.compile(r"'loss'\s*:\s*([0-9.eE+-]+)")
losses = [float(m.group(1)) for m in pat.finditer(text)]
# de-dup consecutive identical (tqdm may reprint)
dedup = []
for v in losses:
    if not dedup or abs(dedup[-1] - v) > 1e-12:
        dedup.append(v)
if not dedup:
    print("- （未能从 train.log 解析到 loss）")
else:
    first, last = dedup[0], dedup[-1]
    mn, mx = min(dedup), max(dedup)
    drop = first - last
    rel = drop / max(abs(first), 1e-12)
    print(f"- steps_logged: {len(dedup)}")
    print(f"- loss_first: {first:.6f}")
    print(f"- loss_last: {last:.6f}")
    print(f"- loss_min: {mn:.6f}")
    print(f"- loss_max: {mx:.6f}")
    print(f"- abs_drop (first-last): {drop:.6f}")
    print(f"- rel_drop: {rel:.4%}")
    flat = abs(drop) < 0.02 and (mx - mn) < 0.05
    print(f"- flat_heuristic: {'YES (supports broken expert-base grads)' if flat else 'NO (loss moved; check freeze / other paths)'}")
    # print compact trajectory
    show = dedup if len(dedup) <= 20 else dedup[:5] + ["..."] + dedup[-5:]
    print("- trajectory: " + ", ".join(f"{x:.4f}" if isinstance(x, float) else x for x in show))
PYEOF
        else
            echo "- （无 phase4/train.log）"
        fi

        echo ""
        echo "## 6. 关键问题检测"
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
            local lf="${LOG_DIR}/${ph}/train.log"
            if [[ -f "${lf}" ]] && grep -qi "sigsegv\|segmentation" "${lf}"; then
                p5_status="⚠ 检测到崩溃"
                break
            fi
        done
        echo "| P5 | C++ 梯度索引 bug / NaN | ${p5_status} |"
        echo "| P6 | Router 梯度稳定性 | 见 plots/04_grad_norm.png |"
        echo "| P7 | update_base_weights 开销 | 见 phase4_analysis.txt / step_timing |"

        echo ""
        echo "## 7. 后续分析"
        echo ""
        echo '```bash'
        echo "python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
        echo '```'
        echo ""
        echo "> 注: 若随后运行 analyze.py，将用更完整的中文报告覆盖本文件（仍只保留 summary.md）。"
    } > "${summary}"

    log "Summary: ${summary}"
    cat "${summary}"
}

# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
cleanup() {
    stop_monitor
    [[ -p "${MONITOR_FIFO}" ]] && rm -f "${MONITOR_FIFO}"
}

trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
main() {
    echo ""
    echo -e "${BOLD}Qwen3-30B-A3B Full-FT Ablation: Expert-Base-Only (AMXBF16)${NC}"
    echo -e "Time: $(run_time_display)"
    echo -e "Run label: ${RUN_LABEL}"
    echo -e "KT train mode: ${ACCELERATE_KT_TRAIN_MODE}  |  freeze_non_expert_base: ${KT_FREEZE_NON_EXPERT_BASE}"
    echo -e "Trainable: gate/up/down_proj_buf ONLY (attention/router/LoRA frozen)"
    echo -e "Entry module: ${TRAIN_ENTRY_MODULE}"
    echo -e "Dataset: ${DATASET_NAME} @ ${DATA_DIR}  (every sample >7000 tokens)"
    echo -e "GPUs: ${NUM_GPUS}  |  cutoff_len: ${CUTOFF_LEN}  |  GAS: ${GRAD_ACCUM_STEPS}  |  warmup_skip: ${WARMUP_SKIP}"
    echo ""

    check_env
    estimate_resources
    setup_run_dir
    generate_memory_component_estimate
    start_monitor

    local overall_exit=0

    run_phase4 || overall_exit=1

    stop_monitor

    generate_memory_component_timeline
    generate_summary

    if [[ -f "${ANALYZE_SCRIPT}" ]]; then
        log "Running visualization analysis..."
        "${PYTHON}" "${ANALYZE_SCRIPT}" --log-dir "${LOG_DIR}" \
            >> "${LOG_DIR}/analyze.log" 2>&1 && \
            ok "Charts generated: ${LOG_DIR}/plots/" || \
            warn "analyze.py failed, run manually: python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
    fi

    echo ""
    if [[ "${overall_exit}" -eq 0 ]]; then
        ok "All phases completed — log directory: ${LOG_DIR}"
    else
        warn "Some phases had issues — check: ${LOG_DIR}/summary.md"
    fi

    return "${overall_exit}"
}

main "$@"

#!/usr/bin/env bash
# Shared Qwen3.5-35B-A3B text-only native-BF16 full-FT sequence sweep.
# Invoke through one of the three backend-specific wrapper scripts.

set -Eeuo pipefail

export TZ="${FFT_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"
LLAMA_FACTORY_DIR="${FFT_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
MODEL_PATH="${FFT_MODEL_PATH:-/mnt/data3/models/Qwen3.5-35B-A3B}"
DATASET_DIR="${FFT_DATASET_DIR:-${FFT_ROOT}/dataset}"
DATASET_NAME="${FFT_DATASET_NAME:-fft_real_100}"
TRAIN_ENTRY_MODULE="finetune_train_with_timing"
TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_full_bf16_qwen35.yaml"
DEEPSPEED_CONFIG="${CONFIGS_DIR}/deepspeed_zero3_offload_bf16.json"
VALIDATOR="${SCRIPT_DIR}/validate_benchmark_dataset.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_sweep_results.py"
TIMING_VALIDATOR="${SCRIPT_DIR}/validate_step_timing.py"
RESOURCE_EXEC="${SCRIPT_DIR}/resource_scope_exec.py"

if [[ $# -lt 1 ]]; then
    echo "Internal error: backend argument is required" >&2
    exit 2
fi
BACKEND="$1"
shift
case "${BACKEND}" in
    ktransformers|deepspeed|aptmoe) ;;
    *) echo "Unsupported backend: ${BACKEND}" >&2; exit 2 ;;
esac

PROFILE="server"
SEQUENCE_LENGTHS_CSV="32,64,128,256,512,1024,2048,4096"
STEPS=15
WARMUP_STEPS=5
GRAD_ACCUM_STEPS=1
LEARNING_RATE="1.0e-5"
DEVICES_OVERRIDE=""
DRY_RUN=0
CONTINUE_ON_ERROR=0
KEEP_MODEL_OUTPUT=0
SKIP_DATASET_CHECK=0
CPU_THREADS_OVERRIDE="${FFT_CPU_THREADS:-}"

# Consumer resource contract: an aggregate 1 TiB cgroup hard limit, no swap,
# and equal interleaving across the two 1-TiB NUMA nodes. The interleave policy
# targets 512 GiB per node when the cgroup reaches its limit.
CONSUMER_MEMORY_LIMIT="1T"
CONSUMER_MEMORY_LIMIT_BYTES=1099511627776
CONSUMER_NUMA_NODES="${FFT_CONSUMER_NUMA_NODES:-0,1}"
CONSUMER_CGROUP_MODE="${FFT_CONSUMER_CGROUP_MODE:-auto}"

APTMOE_ENTRYPOINT="${FFT_APTMOE_ENTRYPOINT:-}"
APTMOE_PYTHON="${FFT_APTMOE_PYTHON:-}"

RUN_ROOT=""

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Profiles:
  --profile server|consumer|both
      server   : 8 GPUs, global batch 8, no memory cgroup cap (host ~2T)
      consumer : 2 GPUs, global batch 2, hard 1T cgroup cap, NUMA 0/1 interleave

Sweep and training (full fine-tuning, BF16 only):
  --seq-lengths LIST           Comma list drawn from 32,64,128,256,512,1024,2048,4096
  --steps N                    Optimizer steps per sequence (default: 15)
  --warmup-steps N             Initial steps excluded from stable TPS (default: 5)
  --gas N                      Gradient accumulation steps (default: 1)
  --learning-rate VALUE        Learning rate (default: 1.0e-5)
  --cpu-threads N              CPU threads per training rank; default: physical cores / ranks
  --devices LIST               Physical GPU list; each profile uses its first N entries
  --model-path PATH            Default: /mnt/data3/models/Qwen3.5-35B-A3B
  --dataset-dir PATH           LLaMA-Factory dataset directory
  --dataset-name NAME          Registered dataset name (default: fft_real_100)
  --log-base PATH              Result directory base

Consumer memory policy:
  --consumer-cgroup-mode MODE  auto, user, system, or prelimited (default: auto)
  --consumer-numa-nodes LIST   Equal-interleave nodes (default: 0,1)

APTMoE adapter (aptmoe wrapper only):
  --aptmoe-entrypoint PATH     Qwen3.5 text-only full-FT adapter implementing the documented CLI
  --aptmoe-python PATH         Python from the APTMoE runtime environment

Other:
  --continue-on-error          Continue remaining sequence lengths after a failed run
  --keep-model-output          Keep generated final model output (large)
  --skip-dataset-check         Skip tokenizer length validation
  --dry-run                    Generate configs/commands without training
  -h, --help                   Show this help

Timing records only per-step forward, backward, optimizer, and total wall time.
Backend-internal profilers, forced CUDA synchronization, and resource samplers are disabled.
The multimodal source checkpoint is loaded as Qwen3_5MoeForCausalLM; no visual tower is constructed.
TPS = GPUs * per-device batch * sequence length * GAS / post-warmup mean step time.
EOF
}

need_value() {
    local flag="$1" count="$2"
    if [[ "${count}" -lt 2 ]]; then
        echo "Missing value for ${flag}" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) need_value "$1" "$#"; PROFILE="$2"; shift ;;
        --seq-lengths) need_value "$1" "$#"; SEQUENCE_LENGTHS_CSV="$2"; shift ;;
        --steps) need_value "$1" "$#"; STEPS="$2"; shift ;;
        --warmup-steps) need_value "$1" "$#"; WARMUP_STEPS="$2"; shift ;;
        --gas) need_value "$1" "$#"; GRAD_ACCUM_STEPS="$2"; shift ;;
        --learning-rate) need_value "$1" "$#"; LEARNING_RATE="$2"; shift ;;
        --cpu-threads) need_value "$1" "$#"; CPU_THREADS_OVERRIDE="$2"; shift ;;
        --devices) need_value "$1" "$#"; DEVICES_OVERRIDE="$2"; shift ;;
        --model-path) need_value "$1" "$#"; MODEL_PATH="$2"; shift ;;
        --dataset-dir) need_value "$1" "$#"; DATASET_DIR="$2"; shift ;;
        --dataset-name) need_value "$1" "$#"; DATASET_NAME="$2"; shift ;;
        --log-base) need_value "$1" "$#"; LOG_BASE="$2"; shift ;;
        --consumer-cgroup-mode) need_value "$1" "$#"; CONSUMER_CGROUP_MODE="$2"; shift ;;
        --consumer-numa-nodes) need_value "$1" "$#"; CONSUMER_NUMA_NODES="$2"; shift ;;
        --aptmoe-entrypoint) need_value "$1" "$#"; APTMOE_ENTRYPOINT="$2"; shift ;;
        --aptmoe-python) need_value "$1" "$#"; APTMOE_PYTHON="$2"; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1 ;;
        --keep-model-output) KEEP_MODEL_OUTPUT=1 ;;
        --skip-dataset-check) SKIP_DATASET_CHECK=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

require_positive_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got ${value}"
}

require_nonnegative_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got ${value}"
}

require_positive_number() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || \
        die "${name} must be a positive number, got ${value}"
    awk -v value="${value}" 'BEGIN { exit !(value > 0) }' || \
        die "${name} must be greater than zero, got ${value}"
}

require_positive_int "--steps" "${STEPS}"
require_nonnegative_int "--warmup-steps" "${WARMUP_STEPS}"
require_positive_int "--gas" "${GRAD_ACCUM_STEPS}"
require_positive_number "--learning-rate" "${LEARNING_RATE}"
if [[ -n "${CPU_THREADS_OVERRIDE}" ]]; then
    require_positive_int "--cpu-threads/FFT_CPU_THREADS" "${CPU_THREADS_OVERRIDE}"
fi
(( WARMUP_STEPS < STEPS )) || die "--warmup-steps must be smaller than --steps"
[[ "${PROFILE}" =~ ^(server|consumer|both)$ ]] || die "invalid --profile: ${PROFILE}"
[[ "${CONSUMER_CGROUP_MODE}" =~ ^(auto|user|system|prelimited)$ ]] || \
    die "invalid --consumer-cgroup-mode: ${CONSUMER_CGROUP_MODE}"

IFS=',' read -r -a SEQUENCE_LENGTHS <<< "${SEQUENCE_LENGTHS_CSV// /}"
(( ${#SEQUENCE_LENGTHS[@]} > 0 )) || die "--seq-lengths cannot be empty"
declare -A SEEN_SEQUENCE=()
MAX_SEQUENCE_LENGTH=0
for seq in "${SEQUENCE_LENGTHS[@]}"; do
    [[ "${seq}" =~ ^(32|64|128|256|512|1024|2048|4096)$ ]] || \
        die "unsupported sequence length: ${seq}"
    [[ -z "${SEEN_SEQUENCE[${seq}]:-}" ]] || die "duplicate sequence length: ${seq}"
    SEEN_SEQUENCE["${seq}"]=1
    (( seq > MAX_SEQUENCE_LENGTH )) && MAX_SEQUENCE_LENGTH="${seq}"
done

_find_conda_python() {
    local env_name="$1"
    local candidates=(
        "/mnt/data2/wbw/conda/envs/${env_name}/bin/python3"
        "/mnt/data2/wbw/miniconda3/envs/${env_name}/bin/python3"
        "/opt/conda/envs/${env_name}/bin/python3"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        [[ -x "${candidate}" ]] && { printf '%s\n' "${candidate}"; return 0; }
    done
    return 1
}

VALIDATOR_PYTHON="$(_find_conda_python Kllama || true)"
[[ -n "${VALIDATOR_PYTHON}" ]] || VALIDATOR_PYTHON="$(command -v python3 || true)"
[[ -x "${VALIDATOR_PYTHON}" ]] || die "No Python available for validation/aggregation"

case "${BACKEND}" in
    ktransformers)
        CONDA_ENV="${FFT_CONDA_ENV:-Kllama}"
        PYTHON="$(_find_conda_python "${CONDA_ENV}" || true)"
        ;;
    deepspeed)
        CONDA_ENV="${FFT_CONDA_ENV:-Deepspeed}"
        PYTHON="$(_find_conda_python "${CONDA_ENV}" || true)"
        ;;
    aptmoe)
        CONDA_ENV="${FFT_CONDA_ENV:-AptMoE}"
        PYTHON="${APTMOE_PYTHON}"
        [[ -n "${PYTHON}" ]] || PYTHON="$(_find_conda_python "${CONDA_ENV}" || true)"
        if [[ -z "${PYTHON}" && "${DRY_RUN}" -eq 1 ]]; then
            PYTHON="${VALIDATOR_PYTHON}"
        fi
        ;;
esac
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || \
    die "Python for backend ${BACKEND} was not found (env=${CONDA_ENV})"
CONDA_BIN_DIR="$(dirname "${PYTHON}")"

detect_physical_cores() {
    "${VALIDATOR_PYTHON}" - <<'PY'
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
        cores.add(((topology / "physical_package_id").read_text(), (topology / "core_id").read_text()))
    except OSError:
        pass
print(max(1, len(cores) if cores else len(cpu_ids)))
PY
}

PHYSICAL_CORES="$(detect_physical_cores)"

check_files_and_environment() {
    [[ -d "${MODEL_PATH}" ]] || die "model directory not found: ${MODEL_PATH}"
    [[ -d "${DATASET_DIR}" ]] || die "dataset directory not found: ${DATASET_DIR}"
    [[ -d "${LLAMA_FACTORY_DIR}/src/llamafactory" ]] || die "LLaMA-Factory source not found: ${LLAMA_FACTORY_DIR}"
    [[ -f "${TRAIN_CONFIG_BASE}" ]] || die "training template not found: ${TRAIN_CONFIG_BASE}"
    [[ -f "${SCRIPT_DIR}/step_phase_timer.py" ]] || die "coarse step phase timer not found"
    [[ -f "${SCRIPT_DIR}/qwen35_text_only.py" ]] || die "text-only model loader not found"
    [[ -f "${VALIDATOR}" && -f "${AGGREGATOR}" && -f "${TIMING_VALIDATOR}" && -f "${RESOURCE_EXEC}" ]] || \
        die "benchmark helper scripts are missing"

    case "${BACKEND}" in
        ktransformers)
            "${PYTHON}" -c 'import accelerate, ktransformers, kt_kernel, transformers' || \
                die "KTransformers dependencies are unavailable in ${CONDA_ENV}"
            ;;
        deepspeed)
            [[ -f "${DEEPSPEED_CONFIG}" ]] || die "DeepSpeed BF16 config not found"
            "${PYTHON}" -c 'import accelerate, deepspeed, transformers' || \
                die "DeepSpeed dependencies are unavailable in ${CONDA_ENV}"
            "${PYTHON}" -c \
                'import importlib.util; from deepspeed.git_version_info import installed_ops; assert installed_ops.get("cpu_adam") and importlib.util.find_spec("deepspeed.ops.adam.cpu_adam_op")' || \
                die "DeepSpeedCPUAdam must be prebuilt in ${CONDA_ENV}; benchmark-time JIT is not allowed"
            ;;
        aptmoe)
            if [[ "${DRY_RUN}" -eq 0 ]]; then
                [[ -n "${APTMOE_ENTRYPOINT}" ]] || die \
                    "APTMoE has no generic Qwen3.5 trainer; set --aptmoe-entrypoint to a ported adapter"
                [[ -f "${APTMOE_ENTRYPOINT}" ]] || die "APTMoE entrypoint not found: ${APTMOE_ENTRYPOINT}"
            elif [[ -z "${APTMOE_ENTRYPOINT}" ]]; then
                APTMOE_ENTRYPOINT="/path/to/qwen35_aptmoe_bf16_adapter.py"
                warn "APTMoE adapter is not installed; dry-run will use placeholder ${APTMOE_ENTRYPOINT}"
            fi
            ;;
    esac
}

validate_dataset() {
    if [[ "${SKIP_DATASET_CHECK}" -eq 1 ]]; then
        warn "Dataset tokenizer validation skipped by request"
        return
    fi
    log "Validating the Qwen3.5 text-only BF16 model contract and dataset lengths"
    "${VALIDATOR_PYTHON}" "${VALIDATOR}" \
        --model-path "${MODEL_PATH}" \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --required-length "${MAX_SEQUENCE_LENGTH}" \
        --output-json "${RUN_ROOT}/dataset_validation.json"
}

check_visible_gpu_capacity() {
    local requested="$1"
    [[ "${DRY_RUN}" -eq 1 ]] && return
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required for a real run"
    local actual
    actual="$(nvidia-smi -L | wc -l)"
    (( actual >= requested )) || die "requested ${requested} GPUs but only ${actual} were detected"
}

resolve_devices() {
    local requested="$1"
    local source="${DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-}}"
    if [[ -z "${source}" ]]; then
        source="$(seq 0 $((requested - 1)) | paste -sd ',')"
    fi
    source="${source// /}"
    local -a candidates
    IFS=',' read -r -a candidates <<< "${source}"
    (( ${#candidates[@]} >= requested )) || \
        die "GPU list '${source}' has fewer than ${requested} entries"
    local -a selected=("${candidates[@]:0:requested}")
    local joined
    joined="$(IFS=','; printf '%s' "${selected[*]}")"
    if (( ${#candidates[@]} > requested )); then
        log "Profile uses first ${requested} devices from ${source}: ${joined}" >&2
    fi
    printf '%s\n' "${joined}"
}

current_cgroup_memory_max() {
    local rel path
    rel="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
    path="/sys/fs/cgroup${rel}/memory.max"
    [[ -r "${path}" ]] && cat "${path}" || printf 'unknown\n'
}

check_numa_capacity() {
    command -v numactl >/dev/null || die "numactl is required for the consumer NUMA policy"
    local node total_kib
    IFS=',' read -r -a nodes <<< "${CONSUMER_NUMA_NODES// /}"
    (( ${#nodes[@]} == 2 )) || die "consumer requires exactly two NUMA nodes, got ${CONSUMER_NUMA_NODES}"
    for node in "${nodes[@]}"; do
        [[ "${node}" =~ ^[0-9]+$ ]] || die "invalid NUMA node: ${node}"
        [[ -r "/sys/devices/system/node/node${node}/meminfo" ]] || die "NUMA node ${node} is unavailable"
        total_kib="$(awk '/MemTotal/ {print $4}' "/sys/devices/system/node/node${node}/meminfo")"
        (( total_kib >= 536870912 )) || \
            die "NUMA node ${node} has less than 512 GiB total memory"
    done
}

resolve_consumer_cgroup_mode() {
    local mode="${CONSUMER_CGROUP_MODE}"
    local current_max
    current_max="$(current_cgroup_memory_max)"
    if [[ "${mode}" == "auto" ]]; then
        if [[ "${current_max}" =~ ^[0-9]+$ ]] && \
           (( current_max <= CONSUMER_MEMORY_LIMIT_BYTES )); then
            mode="prelimited"
        elif [[ "${DRY_RUN}" -eq 1 ]]; then
            mode="user"
        elif systemctl --user show-environment >/dev/null 2>&1; then
            mode="user"
        elif [[ "$(id -u)" -eq 0 ]]; then
            mode="system"
        else
            die "No delegated user systemd/cgroup is available. Ask the administrator to enable it, or launch inside a prelimited 1T cgroup and use --consumer-cgroup-mode prelimited."
        fi
    fi
    if [[ "${mode}" == "prelimited" ]]; then
        [[ "${current_max}" =~ ^[0-9]+$ ]] || die "current cgroup memory.max is not numeric: ${current_max}"
        (( current_max == CONSUMER_MEMORY_LIMIT_BYTES )) || \
            die "current cgroup memory.max=${current_max}; consumer requires exactly 1 TiB"
    fi
    printf '%s\n' "${mode}"
}

declare -a RESOURCE_PREFIX=()
MEMORY_LIMIT_LABEL=""
NUMA_POLICY_LABEL=""
RESOURCE_MODE=""

build_resource_policy() {
    local profile_name="$1"
    RESOURCE_PREFIX=()
    if [[ "${profile_name}" == "server" ]]; then
        MEMORY_LIMIT_LABEL="host-unlimited (~2T visible)"
        NUMA_POLICY_LABEL="host/default nodes 0,1"
        RESOURCE_MODE="none"
        return
    fi

    check_numa_capacity
    RESOURCE_MODE="$(resolve_consumer_cgroup_mode)"
    case "${RESOURCE_MODE}" in
        user)
            RESOURCE_PREFIX=(
                systemd-run --user --scope --quiet --collect
                --property="MemoryMax=${CONSUMER_MEMORY_LIMIT}"
                --property=MemorySwapMax=0
            )
            ;;
        system)
            RESOURCE_PREFIX=(
                systemd-run --scope --quiet --collect
                --property="MemoryMax=${CONSUMER_MEMORY_LIMIT}"
                --property=MemorySwapMax=0
            )
            ;;
        prelimited) ;;
        *) die "internal error: resource mode ${RESOURCE_MODE}" ;;
    esac
    RESOURCE_PREFIX+=(numactl "--interleave=${CONSUMER_NUMA_NODES}")
    MEMORY_LIMIT_LABEL="1TiB hard cgroup (swap disabled)"
    NUMA_POLICY_LABEL="equal interleave nodes ${CONSUMER_NUMA_NODES} (~512GiB/node at limit)"
}

profile_parameters() {
    local profile_name="$1"
    case "${profile_name}" in
        server)
            NUM_GPUS=8
            GLOBAL_BATCH_SIZE=8
            ;;
        consumer)
            NUM_GPUS=2
            GLOBAL_BATCH_SIZE=2
            ;;
        *) die "internal profile: ${profile_name}" ;;
    esac
    (( GLOBAL_BATCH_SIZE % NUM_GPUS == 0 )) || die "global batch is not divisible by GPUs"
    PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
    if [[ -n "${CPU_THREADS_OVERRIDE}" ]]; then
        CPU_THREADS_PER_RANK="${CPU_THREADS_OVERRIDE}"
    else
        CPU_THREADS_PER_RANK=$((PHYSICAL_CORES / NUM_GPUS))
        (( CPU_THREADS_PER_RANK > 0 )) || CPU_THREADS_PER_RANK=1
    fi
    CPU_THREAD_BUDGET_TOTAL=$((CPU_THREADS_PER_RANK * NUM_GPUS))
}

set_yaml_value() {
    local file="$1" key="$2" value="$3"
    if grep -q "^${key}:" "${file}"; then
        sed -i "s|^${key}:.*|${key}: ${value}|" "${file}"
    else
        printf '%s: %s\n' "${key}" "${value}" >> "${file}"
    fi
}

make_train_config() {
    local run_dir="$1" seq="$2"
    local config="${run_dir}/train_config.yaml"
    cp "${TRAIN_CONFIG_BASE}" "${config}"
    set_yaml_value "${config}" model_name_or_path "${MODEL_PATH}"
    set_yaml_value "${config}" dataset "${DATASET_NAME}"
    set_yaml_value "${config}" dataset_dir "${DATASET_DIR}"
    set_yaml_value "${config}" template "qwen3"
    set_yaml_value "${config}" cutoff_len "${seq}"
    set_yaml_value "${config}" output_dir "${run_dir}/model_output"
    set_yaml_value "${config}" per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    set_yaml_value "${config}" gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    set_yaml_value "${config}" learning_rate "${LEARNING_RATE}"
    set_yaml_value "${config}" max_steps "${STEPS}"
    set_yaml_value "${config}" bf16 "true"
    set_yaml_value "${config}" fp16 "false"
    set_yaml_value "${config}" tf32 "false"
    if [[ "${BACKEND}" == "ktransformers" ]]; then
        set_yaml_value "${config}" use_kt "true"
        set_yaml_value "${config}" kt_weight_path "${MODEL_PATH}"
    else
        set_yaml_value "${config}" use_kt "false"
        set_yaml_value "${config}" kt_weight_path "null"
        set_yaml_value "${config}" deepspeed "${DEEPSPEED_CONFIG}"
    fi
    printf '%s\n' "${config}"
}

write_run_config() {
    local path="$1" profile_name="$2" seq="$3" devices="$4" tokens="$5"
    "${VALIDATOR_PYTHON}" - \
        "${path}" "${BACKEND}" "${profile_name}" "${seq}" "${NUM_GPUS}" \
        "${GLOBAL_BATCH_SIZE}" "${PER_DEVICE_BATCH_SIZE}" "${GRAD_ACCUM_STEPS}" \
        "${tokens}" "${STEPS}" "${WARMUP_STEPS}" "${LEARNING_RATE}" \
        "${devices}" "${MODEL_PATH}" "${DATASET_NAME}" "${MEMORY_LIMIT_LABEL}" \
        "${NUMA_POLICY_LABEL}" "${RESOURCE_MODE}" "${CPU_THREADS_PER_RANK}" \
        "${CPU_THREAD_BUDGET_TOTAL}" "${DRY_RUN}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
obj = {
    "backend": sys.argv[2],
    "profile": sys.argv[3],
    "precision": "bf16",
    "modality": "text_only",
    "source_architecture": "Qwen3_5MoeForConditionalGeneration",
    "model_load_architecture": "Qwen3_5MoeForCausalLM",
    "processor_loaded": False,
    "sequence_length": int(sys.argv[4]),
    "num_gpus": int(sys.argv[5]),
    "global_batch_size": int(sys.argv[6]),
    "per_device_batch_size": int(sys.argv[7]),
    "gradient_accumulation_steps": int(sys.argv[8]),
    "tokens_per_step": int(sys.argv[9]),
    "steps": int(sys.argv[10]),
    "warmup_steps": int(sys.argv[11]),
    "learning_rate": sys.argv[12],
    "devices": sys.argv[13],
    "model_path": sys.argv[14],
    "dataset_name": sys.argv[15],
    "memory_limit": sys.argv[16],
    "numa_policy": sys.argv[17],
    "resource_mode": sys.argv[18],
    "cpu_threads_per_rank": int(sys.argv[19]),
    "cpu_thread_budget_total": int(sys.argv[20]),
    "dry_run": bool(int(sys.argv[21])),
}
out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

run_one_sequence() {
    local profile_name="$1" profile_dir="$2" seq="$3" devices="$4"
    local run_dir="${profile_dir}/seq_${seq}"
    local timing_dir="${run_dir}/step_timing"
    local train_log="${run_dir}/train.log"
    local tokens_per_step=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * seq * GRAD_ACCUM_STEPS))
    local train_config=""
    local accel_config=""
    local run_cwd="${LLAMA_FACTORY_DIR}"
    local cpu_threads="${CPU_THREADS_PER_RANK}"
    mkdir -p "${timing_dir}"
    write_run_config "${run_dir}/run_config.json" "${profile_name}" "${seq}" "${devices}" "${tokens_per_step}"

    local -a command=()
    case "${BACKEND}" in
        ktransformers)
            train_config="$(make_train_config "${run_dir}" "${seq}")"
            local accel_template="${CONFIGS_DIR}/accelerate_ktransformers_bf16_${NUM_GPUS}gpu.yaml"
            [[ -f "${accel_template}" ]] || die "accelerate config not found: ${accel_template}"
            grep -q '^  kt_num_threads:' "${accel_template}" || \
                die "kt_num_threads is missing from ${accel_template}"
            accel_config="${run_dir}/accelerate_config.yaml"
            cp "${accel_template}" "${accel_config}"
            sed -i "s|^  kt_num_threads:.*|  kt_num_threads: ${cpu_threads}|" "${accel_config}"
            local accelerate_bin="${CONDA_BIN_DIR}/accelerate"
            [[ -x "${accelerate_bin}" ]] || accelerate_bin="accelerate"
            command=(
                env
                USE_KT=1
                ACCELERATE_USE_KT=true
                ACCELERATE_KT_TRAIN_MODE=full
                KT_FINETUNE_MODE=full
                FFT_TRAINING_BACKEND=kt
                FFT_PRECISION=bf16
                FFT_TEXT_ONLY=1
                FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
                FFT_STEP_TIMING_OUT_DIR="${timing_dir}"
                FFT_STEP_TIMING_WARMUP_STEPS="${WARMUP_STEPS}"
                FFT_STEP_TIMING_TOKENS_PER_STEP="${tokens_per_step}"
                FFT_DISABLE_PERF_PROBES=1
                FFT_CPU_THREADS="${cpu_threads}"
                KT_BACKWARD_TIMING=off
                KT_SFT_PROFILE=0
                DS_PROBE_MODE=off
                ACCELERATE_KT_MODEL_MAX_LENGTH="${seq}"
                OMP_NUM_THREADS="${cpu_threads}"
                MKL_NUM_THREADS="${cpu_threads}"
                OPENBLAS_NUM_THREADS="${cpu_threads}"
                NUMEXPR_NUM_THREADS="${cpu_threads}"
                BLIS_NUM_THREADS="${cpu_threads}"
                OMP_DYNAMIC=FALSE
                MKL_DYNAMIC=FALSE
                ACCELERATE_KT_OMP_NUM_THREADS="${cpu_threads}"
                TOKENIZERS_PARALLELISM=false
                HF_DATASETS_OFFLINE=1
                TRANSFORMERS_OFFLINE=1
                CUDA_VISIBLE_DEVICES="${devices}"
                PYTHONPATH="${SCRIPT_DIR}:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
                "${accelerate_bin}" launch
                --config_file "${accel_config}"
                -m "${TRAIN_ENTRY_MODULE}" train "${train_config}"
            )
            ;;
        deepspeed)
            train_config="$(make_train_config "${run_dir}" "${seq}")"
            local torchrun_bin="${CONDA_BIN_DIR}/torchrun"
            [[ -x "${torchrun_bin}" ]] || torchrun_bin="torchrun"
            command=(
                env
                USE_KT=0
                ACCELERATE_USE_KT=false
                FFT_TRAINING_BACKEND=deepspeed
                FFT_PRECISION=bf16
                FFT_TEXT_ONLY=1
                FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
                KT_FINETUNE_MODE=full
                FFT_STEP_TIMING_OUT_DIR="${timing_dir}"
                FFT_STEP_TIMING_WARMUP_STEPS="${WARMUP_STEPS}"
                FFT_STEP_TIMING_TOKENS_PER_STEP="${tokens_per_step}"
                FFT_DISABLE_PERF_PROBES=1
                FFT_CPU_THREADS="${cpu_threads}"
                KT_BACKWARD_TIMING=off
                KT_SFT_PROFILE=0
                DS_PROBE_MODE=off
                OMP_NUM_THREADS="${cpu_threads}"
                MKL_NUM_THREADS="${cpu_threads}"
                OPENBLAS_NUM_THREADS="${cpu_threads}"
                NUMEXPR_NUM_THREADS="${cpu_threads}"
                BLIS_NUM_THREADS="${cpu_threads}"
                OMP_DYNAMIC=FALSE
                MKL_DYNAMIC=FALSE
                TOKENIZERS_PARALLELISM=false
                HF_DATASETS_OFFLINE=1
                TRANSFORMERS_OFFLINE=1
                CUDA_VISIBLE_DEVICES="${devices}"
                PYTHONPATH="${SCRIPT_DIR}:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
                "${torchrun_bin}" --standalone --nproc_per_node="${NUM_GPUS}"
                -m "${TRAIN_ENTRY_MODULE}" train "${train_config}"
            )
            ;;
        aptmoe)
            run_cwd="$(dirname "${APTMOE_ENTRYPOINT}")"
            command=(
                env
                FFT_TRAINING_BACKEND=aptmoe
                FFT_PRECISION=bf16
                FFT_TEXT_ONLY=1
                FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
                FFT_DISABLE_PERF_PROBES=1
                FFT_CPU_THREADS="${cpu_threads}"
                KT_BACKWARD_TIMING=off
                KT_SFT_PROFILE=0
                DS_PROBE_MODE=off
                OMP_NUM_THREADS="${cpu_threads}"
                MKL_NUM_THREADS="${cpu_threads}"
                OPENBLAS_NUM_THREADS="${cpu_threads}"
                NUMEXPR_NUM_THREADS="${cpu_threads}"
                BLIS_NUM_THREADS="${cpu_threads}"
                OMP_DYNAMIC=FALSE
                MKL_DYNAMIC=FALSE
                TOKENIZERS_PARALLELISM=false
                HF_DATASETS_OFFLINE=1
                TRANSFORMERS_OFFLINE=1
                CUDA_VISIBLE_DEVICES="${devices}"
                "${PYTHON}" "${APTMOE_ENTRYPOINT}"
                --model-path "${MODEL_PATH}"
                --dataset-dir "${DATASET_DIR}"
                --dataset-name "${DATASET_NAME}"
                --output-dir "${run_dir}/model_output"
                --step-timing-output-dir "${timing_dir}"
                --sequence-length "${seq}"
                --num-gpus "${NUM_GPUS}"
                --global-batch-size "${GLOBAL_BATCH_SIZE}"
                --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
                --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
                --steps "${STEPS}"
                --warmup-steps "${WARMUP_STEPS}"
                --learning-rate "${LEARNING_RATE}"
                --precision bf16
                --text-only
            )
            ;;
    esac

    local -a scoped_command=(
        "${VALIDATOR_PYTHON}" "${RESOURCE_EXEC}"
        --profile "${profile_name}"
        --numa-nodes "${CONSUMER_NUMA_NODES}"
        --output-dir "${run_dir}"
    )
    if [[ "${profile_name}" == "consumer" ]]; then
        scoped_command+=(
            --expected-memory-max "${CONSUMER_MEMORY_LIMIT_BYTES}"
            --require-swap-zero
        )
    fi
    scoped_command+=(-- "${command[@]}")
    local -a full_command=("${RESOURCE_PREFIX[@]}" "${scoped_command[@]}")
    log "${BACKEND}/${profile_name}: seq=${seq}, GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, tokens/step=${tokens_per_step}, text-only BF16, CPU threads/rank=${cpu_threads}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        print_command "${full_command[@]}"
        printf 'DRY_RUN\n' > "${run_dir}/exit_code.txt"
        return 0
    fi

    local exit_code=0
    pushd "${run_cwd}" >/dev/null
    set +e
    "${full_command[@]}" 2>&1 | tee "${train_log}"
    exit_code=${PIPESTATUS[0]}
    set -e
    popd >/dev/null

    if [[ "${exit_code}" -eq 0 ]]; then
        if [[ ! -f "${timing_dir}/step_timing.json" ]]; then
            warn "Training exited successfully but canonical rank-0 timing is missing"
            exit_code=90
        elif ! "${VALIDATOR_PYTHON}" "${TIMING_VALIDATOR}" \
            --path "${timing_dir}/step_timing.json" \
            --expected-steps "${STEPS}" \
            --warmup-steps "${WARMUP_STEPS}"; then
            warn "Timing output violates the probe-free three-phase contract"
            exit_code=92
        fi
    fi
    printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"

    if [[ "${KEEP_MODEL_OUTPUT}" -eq 0 && -d "${run_dir}/model_output" ]]; then
        log "Removing generated model output for seq=${seq}; timing/logs are retained"
        rm -rf "${run_dir}/model_output"
    fi
    if [[ "${exit_code}" -ne 0 ]]; then
        warn "${BACKEND}/${profile_name}/seq_${seq} failed with exit code ${exit_code}"
        return "${exit_code}"
    fi
    return 0
}

run_profile() {
    local profile_name="$1"
    profile_parameters "${profile_name}"
    check_visible_gpu_capacity "${NUM_GPUS}"
    local devices
    devices="$(resolve_devices "${NUM_GPUS}")"
    build_resource_policy "${profile_name}"
    local profile_dir="${RUN_ROOT}/${profile_name}_${NUM_GPUS}gpu_batch${GLOBAL_BATCH_SIZE}"
    mkdir -p "${profile_dir}"
    log "Profile ${profile_name}: devices=${devices}, memory=${MEMORY_LIMIT_LABEL}, NUMA=${NUMA_POLICY_LABEL}, CPU threads/rank=${CPU_THREADS_PER_RANK}"

    local profile_status=0 seq
    for seq in "${SEQUENCE_LENGTHS[@]}"; do
        if ! run_one_sequence "${profile_name}" "${profile_dir}" "${seq}" "${devices}"; then
            profile_status=1
            if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
                warn "Stopping profile after first failure; use --continue-on-error to keep sweeping"
                break
            fi
        fi
    done
    return "${profile_status}"
}

check_files_and_environment
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${LOG_BASE}/${RUN_TIMESTAMP}_${BACKEND^^}_BF16_FULL_SWEEP"
mkdir -p "${RUN_ROOT}"
validate_dataset

log "Qwen3.5-35B-A3B text-only full-FT sweep: backend=${BACKEND}, precision=BF16, profile=${PROFILE}"
log "Sequences: ${SEQUENCE_LENGTHS[*]}; steps=${STEPS}; warmup excluded=${WARMUP_STEPS}; GAS=${GRAD_ACCUM_STEPS}"
log "Result root: ${RUN_ROOT}"

declare -a PROFILES=()
case "${PROFILE}" in
    server) PROFILES=(server) ;;
    consumer) PROFILES=(consumer) ;;
    both) PROFILES=(server consumer) ;;
esac

overall_status=0
for selected_profile in "${PROFILES[@]}"; do
    if ! run_profile "${selected_profile}"; then
        overall_status=1
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] || break
    fi
done

"${VALIDATOR_PYTHON}" "${AGGREGATOR}" --root "${RUN_ROOT}"
log "Sweep summary: ${RUN_ROOT}/summary.md"
log "Machine-readable results: ${RUN_ROOT}/sweep_results.csv"
exit "${overall_status}"

#!/usr/bin/env bash
# Shared Qwen3.5-35B-A3B text-only native-BF16 fine-tuning sequence sweep.
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
FINETUNING_TYPE="full"
readonly -a SERVER_SEQUENCE_LENGTHS=(32 64 128 256 512 1024 2048 4096)
readonly -a CONSUMER_SEQUENCE_LENGTHS=(16 32 64 128 256 512 1024 2048)
SEQUENCE_LENGTHS_CSV=""
SEQUENCE_LENGTHS_OVERRIDE_SET=0
STEPS=15
WARMUP_STEPS=5
GRAD_ACCUM_STEPS=1
LEARNING_RATE="1.0e-5"
LORA_RANK=8
LORA_ALPHA=16
DEVICES_OVERRIDE=""
DRY_RUN=0
CONTINUE_ON_ERROR=0
KEEP_MODEL_OUTPUT=0
SKIP_DATASET_CHECK=0
CPU_THREADS_OVERRIDE="${FFT_CPU_THREADS:-}"
KT_OWNER_THREADS_OVERRIDE="${FFT_KT_OWNER_THREADS:-}"
KT_DISTRIBUTED_CHECKPOINT_REUSE="${FFT_KT_DISTRIBUTED_CHECKPOINT_REUSE:-on}"

# Consumer resource contract: an aggregate 1 TiB cgroup hard limit, no swap,
# and equal interleaving across the two 1-TiB NUMA nodes. The interleave policy
# targets 512 GiB per node when the cgroup reaches its limit.
CONSUMER_MEMORY_LIMIT="1T"
CONSUMER_MEMORY_LIMIT_BYTES=1099511627776
CONSUMER_NUMA_NODES="${FFT_CONSUMER_NUMA_NODES:-0,1}"
CONSUMER_CGROUP_MODE="${FFT_CONSUMER_CGROUP_MODE:-auto}"

APTMOE_ROOT="${FFT_APTMOE_ROOT:-/mnt/data2/wbw/APTMoE-baseline}"
APTMOE_SIMULATION_ROOT="${FFT_APTMOE_SIMULATION_ROOT:-${FFT_ROOT}/APTMoE-simulate}"
APTMOE_ENTRYPOINT="${FFT_APTMOE_ENTRYPOINT:-${SCRIPT_DIR}/aptmoe_qwen35_proxy_train.py}"
APTMOE_PYTHON="${FFT_APTMOE_PYTHON:-}"
APTMOE_ROUTE_ROOT="${FFT_APTMOE_ROUTE_ROOT:-${APTMOE_SIMULATION_ROOT}/routes/qwen35}"
APTMOE_LOOKUP_ROOT="${FFT_APTMOE_LOOKUP_ROOT:-${APTMOE_SIMULATION_ROOT}/lookups/qwen35}"
APTMOE_LOOKUP_TABLE="${FFT_APTMOE_LOOKUP_TABLE:-}"
APTMOE_ALLOW_SYNTHETIC_ROUTING=0
APTMOE_ALLOW_UNPROFILED_PLACEMENT=0
APTMOE_ALLOW_LINEAR_ATTENTION_FALLBACK=0
CAPTURE_APTMOE_ROUTES=0

RUN_ROOT=""

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Profiles:
  --profile server|consumer|both
      server   : 8 GPUs, global batch 8, no memory cgroup cap (host ~2T)
      consumer : 2 GPUs, global batch 2, hard 1T cgroup cap, NUMA 0/1 interleave

Sweep and training (BF16 only):
  --finetuning-type TYPE      full or lora (default: full)
  --lora-rank N               LoRA rank when TYPE=lora (default: 8)
  --lora-alpha N              LoRA alpha when TYPE=lora (default: 16)
                                LoRA target is fixed to all
  --seq-lengths LIST           Override the selected profile default(s)
                                server:   32,64,128,256,512,1024,2048,4096
                                consumer: 16,32,64,128,256,512,1024,2048
                                With profile=both, every override must be valid for both
  --steps N                    Optimizer steps per sequence (default: 15)
  --warmup-steps N             Initial steps excluded from stable TPS (default: 5)
  --gas N                      Gradient accumulation steps (default: 1)
  --learning-rate VALUE        Learning rate (default: 1.0e-5)
  --cpu-threads N              CPU threads per ordinary training rank
                                (for KTransformers, controls non-owner ranks)
  --kt-owner-threads N         KTransformers rank0 CPU MoE/optimizer threads
                                (default: remaining cores after non-owners and 2 reserved cores)
  --devices LIST               Physical GPU list; each profile uses its first N entries
  --model-path PATH            Default: /mnt/data3/models/Qwen3.5-35B-A3B
  --dataset-dir PATH           LLaMA-Factory dataset directory
  --dataset-name NAME          Registered dataset name (default: fft_real_100)
  --log-base PATH              Result directory base

KTransformers checkpoint reuse:
  --kt-distributed-checkpoint-reuse on|off
                                Reuse first-forward CPU MoE cache on multi-GPU
                                checkpoint recompute (default: on)

Consumer memory policy:
  --consumer-cgroup-mode MODE  auto, user, system, or prelimited (default: auto)
  --consumer-numa-nodes LIST   Equal-interleave nodes (default: 0,1)

APTMoE deployment proxy (aptmoe wrapper only):
  --aptmoe-root PATH           APTMoE-baseline checkout
  --aptmoe-simulation-root PATH
                                Local-only route/lookup/random-weight root
  --aptmoe-route-root PATH     Exact Qwen3.5 route traces, split by profile
  --aptmoe-lookup-root PATH    Host lookup tables, split by profile
  --aptmoe-lookup-table PATH   Override the profile lookup with one explicit file
  --aptmoe-entrypoint PATH     Component-isomorphic proxy runner
  --aptmoe-python PATH         Python from the APTMoE runtime environment
  --aptmoe-allow-synthetic-routing
  --aptmoe-allow-unprofiled-placement
  --aptmoe-allow-linear-attention-fallback
                                Explicit smoke-only fallbacks; never formal TPS

Route capture (ktransformers/deepspeed wrappers only):
  --capture-aptmoe-routes      Capture all excluded-warmup exact Qwen3.5 top-k
                                route patterns and merge under aptmoe-route-root

Other:
  --continue-on-error          Continue remaining sequence lengths after a failed run
  --keep-model-output          Keep generated final model output (large)
  --skip-dataset-check         Skip tokenizer length validation
  --dry-run                    Generate configs/commands without training
  -h, --help                   Show this help

Timing records only per-step forward, backward, optimizer, and total wall time.
Backend-internal profilers, forced CUDA synchronization, and resource samplers are disabled.
Exact backends load Qwen3_5MoeForCausalLM without a visual tower; the APTMoE proxy reads only target config/tokenizer.
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
        --finetuning-type) need_value "$1" "$#"; FINETUNING_TYPE="$2"; shift ;;
        --lora-rank) need_value "$1" "$#"; LORA_RANK="$2"; shift ;;
        --lora-alpha) need_value "$1" "$#"; LORA_ALPHA="$2"; shift ;;
        --seq-lengths) need_value "$1" "$#"; SEQUENCE_LENGTHS_CSV="$2"; SEQUENCE_LENGTHS_OVERRIDE_SET=1; shift ;;
        --steps) need_value "$1" "$#"; STEPS="$2"; shift ;;
        --warmup-steps) need_value "$1" "$#"; WARMUP_STEPS="$2"; shift ;;
        --gas) need_value "$1" "$#"; GRAD_ACCUM_STEPS="$2"; shift ;;
        --learning-rate) need_value "$1" "$#"; LEARNING_RATE="$2"; shift ;;
        --cpu-threads) need_value "$1" "$#"; CPU_THREADS_OVERRIDE="$2"; shift ;;
        --kt-owner-threads) need_value "$1" "$#"; KT_OWNER_THREADS_OVERRIDE="$2"; shift ;;
        --devices) need_value "$1" "$#"; DEVICES_OVERRIDE="$2"; shift ;;
        --model-path) need_value "$1" "$#"; MODEL_PATH="$2"; shift ;;
        --dataset-dir) need_value "$1" "$#"; DATASET_DIR="$2"; shift ;;
        --dataset-name) need_value "$1" "$#"; DATASET_NAME="$2"; shift ;;
        --log-base) need_value "$1" "$#"; LOG_BASE="$2"; shift ;;
        --kt-distributed-checkpoint-reuse) need_value "$1" "$#"; KT_DISTRIBUTED_CHECKPOINT_REUSE="$2"; shift ;;
        --consumer-cgroup-mode) need_value "$1" "$#"; CONSUMER_CGROUP_MODE="$2"; shift ;;
        --consumer-numa-nodes) need_value "$1" "$#"; CONSUMER_NUMA_NODES="$2"; shift ;;
        --aptmoe-root) need_value "$1" "$#"; APTMOE_ROOT="$2"; shift ;;
        --aptmoe-simulation-root) need_value "$1" "$#"; APTMOE_SIMULATION_ROOT="$2"; shift ;;
        --aptmoe-route-root) need_value "$1" "$#"; APTMOE_ROUTE_ROOT="$2"; shift ;;
        --aptmoe-lookup-root) need_value "$1" "$#"; APTMOE_LOOKUP_ROOT="$2"; shift ;;
        --aptmoe-lookup-table) need_value "$1" "$#"; APTMOE_LOOKUP_TABLE="$2"; shift ;;
        --aptmoe-entrypoint) need_value "$1" "$#"; APTMOE_ENTRYPOINT="$2"; shift ;;
        --aptmoe-python) need_value "$1" "$#"; APTMOE_PYTHON="$2"; shift ;;
        --aptmoe-allow-synthetic-routing) APTMOE_ALLOW_SYNTHETIC_ROUTING=1 ;;
        --aptmoe-allow-unprofiled-placement) APTMOE_ALLOW_UNPROFILED_PLACEMENT=1 ;;
        --aptmoe-allow-linear-attention-fallback) APTMOE_ALLOW_LINEAR_ATTENTION_FALLBACK=1 ;;
        --capture-aptmoe-routes) CAPTURE_APTMOE_ROUTES=1 ;;
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
require_positive_int "--lora-rank" "${LORA_RANK}"
require_positive_int "--lora-alpha" "${LORA_ALPHA}"
if [[ -n "${CPU_THREADS_OVERRIDE}" ]]; then
    require_positive_int "--cpu-threads/FFT_CPU_THREADS" "${CPU_THREADS_OVERRIDE}"
fi
if [[ -n "${KT_OWNER_THREADS_OVERRIDE}" ]]; then
    require_positive_int "--kt-owner-threads/FFT_KT_OWNER_THREADS" "${KT_OWNER_THREADS_OVERRIDE}"
fi
(( WARMUP_STEPS < STEPS )) || die "--warmup-steps must be smaller than --steps"
if [[ "${BACKEND}" == "aptmoe" ]] && (( WARMUP_STEPS == 0 )); then
    die "APTMoE proxy runs require at least one excluded warmup step"
fi
[[ "${PROFILE}" =~ ^(server|consumer|both)$ ]] || die "invalid --profile: ${PROFILE}"
[[ "${FINETUNING_TYPE}" =~ ^(full|lora)$ ]] || \
    die "invalid --finetuning-type: ${FINETUNING_TYPE}"
if [[ "${BACKEND}" == "aptmoe" && "${FINETUNING_TYPE}" != "full" ]]; then
    die "APTMoE deployment proxy supports only --finetuning-type full"
fi
if [[ "${FINETUNING_TYPE}" == "lora" ]]; then
    EFFECTIVE_LORA_RANK="${LORA_RANK}"
    EFFECTIVE_LORA_ALPHA="${LORA_ALPHA}"
else
    EFFECTIVE_LORA_RANK=0
    EFFECTIVE_LORA_ALPHA=0
fi
[[ "${KT_DISTRIBUTED_CHECKPOINT_REUSE}" =~ ^(on|off)$ ]] || \
    die "invalid --kt-distributed-checkpoint-reuse: ${KT_DISTRIBUTED_CHECKPOINT_REUSE}"
[[ "${CONSUMER_CGROUP_MODE}" =~ ^(auto|user|system|prelimited)$ ]] || \
    die "invalid --consumer-cgroup-mode: ${CONSUMER_CGROUP_MODE}"
if [[ "${CAPTURE_APTMOE_ROUTES}" -eq 1 ]]; then
    [[ "${BACKEND}" != "aptmoe" ]] || \
        die "--capture-aptmoe-routes must be used on an exact KTransformers or DeepSpeed run"
    (( WARMUP_STEPS > 0 )) || \
        die "--capture-aptmoe-routes requires at least one excluded warmup step"
fi
if [[ "${BACKEND}" != "aptmoe" ]] && ((
    APTMOE_ALLOW_SYNTHETIC_ROUTING ||
    APTMOE_ALLOW_UNPROFILED_PLACEMENT ||
    APTMOE_ALLOW_LINEAR_ATTENTION_FALLBACK
)); then
    die "APTMoE smoke fallback flags are only valid with the aptmoe wrapper"
fi

KT_DISTRIBUTED_CHECKPOINT_REUSE_ENABLED=0
if [[ "${BACKEND}" == "ktransformers" && "${KT_DISTRIBUTED_CHECKPOINT_REUSE}" == "on" ]]; then
    KT_DISTRIBUTED_CHECKPOINT_REUSE_ENABLED=1
fi

MAX_SEQUENCE_LENGTH=0
declare -a SEQUENCE_LENGTHS_OVERRIDE=()
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    IFS=',' read -r -a SEQUENCE_LENGTHS_OVERRIDE <<< "${SEQUENCE_LENGTHS_CSV// /}"
    (( ${#SEQUENCE_LENGTHS_OVERRIDE[@]} > 0 )) || die "--seq-lengths cannot be empty"
    declare -A SEEN_SEQUENCE=()
    for seq in "${SEQUENCE_LENGTHS_OVERRIDE[@]}"; do
        [[ "${seq}" =~ ^(16|32|64|128|256|512|1024|2048|4096)$ ]] || \
            die "unsupported sequence length: ${seq}"
        if [[ "${seq}" == "16" && "${PROFILE}" =~ ^(server|both)$ ]]; then
            die "sequence length 16 is only supported by the consumer profile"
        fi
        if [[ "${seq}" == "4096" && "${PROFILE}" =~ ^(consumer|both)$ ]]; then
            die "sequence length 4096 is only supported by the server profile"
        fi
        [[ -z "${SEEN_SEQUENCE[${seq}]:-}" ]] || die "duplicate sequence length: ${seq}"
        SEEN_SEQUENCE["${seq}"]=1
        (( seq > MAX_SEQUENCE_LENGTH )) && MAX_SEQUENCE_LENGTH="${seq}"
    done
else
    case "${PROFILE}" in
        server|both) MAX_SEQUENCE_LENGTH=4096 ;;
        consumer) MAX_SEQUENCE_LENGTH=2048 ;;
    esac
fi

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
        CONDA_ENV="${FFT_CONDA_ENV:-Aptmoe}"
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
            [[ -d "${APTMOE_ROOT}" ]] || die "APTMoE root not found: ${APTMOE_ROOT}"
            [[ -f "${APTMOE_ROOT}/Runtime/PipelineRuntime/pipeline_runtime.py" ]] || \
                die "APTMoE pipeline runtime not found under ${APTMOE_ROOT}"
            [[ -f "${APTMOE_ENTRYPOINT}" ]] || \
                die "APTMoE proxy entrypoint not found: ${APTMOE_ENTRYPOINT}"
            [[ -f "${SCRIPT_DIR}/profile_aptmoe_qwen35_proxy.py" ]] || \
                die "APTMoE lookup profiler is missing"
            [[ -f "${SCRIPT_DIR}/merge_qwen35_route_traces.py" ]] || \
                die "Qwen3.5 route merge helper is missing"
            env PYTHONPATH="${SCRIPT_DIR}:${APTMOE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
                "${PYTHON}" -c \
                'import numpy, torch, transformers; from Runtime.PipelineRuntime.pipeline_runtime import PipelineRuntime; from aptmoe_proxy import ProxyPlacementSolver, RouteController' || \
                die "APTMoE proxy dependencies are unavailable in ${CONDA_ENV}"
            ;;
    esac
    if [[ "${CAPTURE_APTMOE_ROUTES}" -eq 1 ]]; then
        "${PYTHON}" -c 'import numpy' || \
            die "route capture requires NumPy in ${CONDA_ENV}"
    fi
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
    elif [[ "${BACKEND}" == "ktransformers" ]]; then
        # Only rank0 owns the KT CPU backend. Keep the non-owner ranks
        # lightweight so the owner can use most of the host's physical cores.
        CPU_THREADS_PER_RANK=2
    else
        CPU_THREADS_PER_RANK=$((PHYSICAL_CORES / NUM_GPUS))
        (( CPU_THREADS_PER_RANK > 0 )) || CPU_THREADS_PER_RANK=1
    fi

    KT_OWNER_THREADS=0
    if [[ "${BACKEND}" == "ktransformers" ]]; then
        if [[ -n "${KT_OWNER_THREADS_OVERRIDE}" ]]; then
            KT_OWNER_THREADS="${KT_OWNER_THREADS_OVERRIDE}"
        else
            # Reserve two physical cores for the OS/NCCL helpers, then give
            # every remaining core to the single global-rank0 KT owner.
            KT_OWNER_THREADS=$((PHYSICAL_CORES - 2 - CPU_THREADS_PER_RANK * (NUM_GPUS - 1)))
            (( KT_OWNER_THREADS > 0 )) || \
                die "No physical cores remain for the KT owner after reserving non-owner ranks"
        fi
        (( KT_OWNER_THREADS <= PHYSICAL_CORES )) || \
            die "KT owner threads ${KT_OWNER_THREADS} exceed visible physical cores ${PHYSICAL_CORES}"
        CPU_THREAD_BUDGET_TOTAL=$((KT_OWNER_THREADS + CPU_THREADS_PER_RANK * (NUM_GPUS - 1)))
    else
        CPU_THREAD_BUDGET_TOTAL=$((CPU_THREADS_PER_RANK * NUM_GPUS))
    fi
    if (( CPU_THREAD_BUDGET_TOTAL > PHYSICAL_CORES )); then
        warn "Configured CPU thread budget ${CPU_THREAD_BUDGET_TOTAL} exceeds ${PHYSICAL_CORES} visible physical cores"
    fi
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
    set_yaml_value "${config}" finetuning_type "${FINETUNING_TYPE}"
    set_yaml_value "${config}" lora_rank "${EFFECTIVE_LORA_RANK}"
    if [[ "${FINETUNING_TYPE}" == "lora" ]]; then
        set_yaml_value "${config}" lora_alpha "${EFFECTIVE_LORA_ALPHA}"
        set_yaml_value "${config}" lora_target "all"
    fi
    set_yaml_value "${config}" bf16 "true"
    set_yaml_value "${config}" fp16 "false"
    set_yaml_value "${config}" tf32 "false"
    # Trainer invokes gradient_checkpointing_enable() again after model setup.
    # Keep all backends on the same non-reentrant implementation and prevent
    # that second call from silently restoring LLaMA-Factory's reentrant default.
    set_yaml_value "${config}" gradient_checkpointing_kwargs "{use_reentrant: false}"
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
    local apt_route_trace="$6" apt_lookup_table="$7"
    "${VALIDATOR_PYTHON}" - \
        "${path}" "${BACKEND}" "${profile_name}" "${seq}" "${NUM_GPUS}" \
        "${GLOBAL_BATCH_SIZE}" "${PER_DEVICE_BATCH_SIZE}" "${GRAD_ACCUM_STEPS}" \
        "${tokens}" "${STEPS}" "${WARMUP_STEPS}" "${LEARNING_RATE}" \
        "${devices}" "${MODEL_PATH}" "${DATASET_NAME}" "${MEMORY_LIMIT_LABEL}" \
        "${NUMA_POLICY_LABEL}" "${RESOURCE_MODE}" "${CPU_THREADS_PER_RANK}" \
        "${CPU_THREAD_BUDGET_TOTAL}" "${DRY_RUN}" \
        "${KT_DISTRIBUTED_CHECKPOINT_REUSE_ENABLED}" "${KT_OWNER_THREADS}" \
        "${apt_route_trace}" "${apt_lookup_table}" "${APTMOE_SIMULATION_ROOT}" \
        "${KEEP_MODEL_OUTPUT}" "${APTMOE_ALLOW_SYNTHETIC_ROUTING}" \
        "${APTMOE_ALLOW_UNPROFILED_PLACEMENT}" \
        "${APTMOE_ALLOW_LINEAR_ATTENTION_FALLBACK}" \
        "${CAPTURE_APTMOE_ROUTES}" "${APTMOE_ROOT}" \
        "${APTMOE_ENTRYPOINT}" "${FINETUNING_TYPE}" \
        "${EFFECTIVE_LORA_RANK}" "${EFFECTIVE_LORA_ALPHA}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
is_proxy = sys.argv[2] == "aptmoe"
fallback_requested = any(bool(int(value)) for value in sys.argv[28:31])
finetuning_type = sys.argv[34]
obj = {
    "backend": sys.argv[2],
    "profile": sys.argv[3],
    "benchmark_class": (
        "deployment_proxy"
        if is_proxy
        else f"exact_model_{finetuning_type}_finetune"
    ),
    "result_validity": (
        "smoke_only"
        if is_proxy and fallback_requested
        else "formal_deployment_proxy"
        if is_proxy
        else "exact_model"
    ),
    "weight_source": (
        "deterministic_random_initialization"
        if is_proxy
        else "pretrained_checkpoint"
    ),
    "checkpoint_compatible": not is_proxy,
    "llamafactory_backend": not is_proxy,
    "real_forward_backward_optimizer_update": True,
    "finetuning_type": finetuning_type,
    "lora_rank": int(sys.argv[35]) if finetuning_type == "lora" else 0,
    "lora_alpha": int(sys.argv[36]) if finetuning_type == "lora" else 0,
    "lora_target": "all" if finetuning_type == "lora" else None,
    "allow_end_to_end_qwen35_tps_claim": not is_proxy,
    "precision": "bf16",
    "modality": "text_only",
    "source_architecture": "Qwen3_5MoeForConditionalGeneration",
    "model_load_architecture": (
        "Qwen35ComponentIsomorphicAPTMoEProxy"
        if is_proxy
        else "Qwen3_5MoeForCausalLM"
    ),
    "proxy_target_architecture": (
        "Qwen3_5MoeForCausalLM" if is_proxy else None
    ),
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
    "kt_distributed_checkpoint_forward_reuse": bool(int(sys.argv[22])),
    "kt_owner_rank": 0 if int(sys.argv[23]) > 0 else None,
    "kt_owner_threads": int(sys.argv[23]) or None,
    "route_trace": sys.argv[24] or None,
    "lookup_table": sys.argv[25] or None,
    "aptmoe_simulation_root": sys.argv[26] if is_proxy else None,
    "random_weights_saved": bool(int(sys.argv[27])) if is_proxy else None,
    "allow_synthetic_routing": bool(int(sys.argv[28])) if is_proxy else None,
    "allow_unprofiled_placement": bool(int(sys.argv[29])) if is_proxy else None,
    "allow_linear_attention_fallback": (
        bool(int(sys.argv[30])) if is_proxy else None
    ),
    "route_capture_warmup_only": bool(int(sys.argv[31])) if not is_proxy else False,
    "route_capture_patterns": (
        int(sys.argv[11]) * int(sys.argv[8])
        if bool(int(sys.argv[31]))
        else None
    ),
    "aptmoe_root": sys.argv[32] if is_proxy else None,
    "aptmoe_entrypoint": sys.argv[33] if is_proxy else None,
    "result_scope": (
        "APTMoE component-isomorphic deployment throughput; no model-quality claim"
        if is_proxy
        else f"end-to-end Qwen3.5 text-model {finetuning_type}-finetune throughput"
    ),
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
    local kt_owner_threads="${KT_OWNER_THREADS}"
    local apt_route_trace=""
    local apt_lookup_table=""
    local route_capture_dir=""
    if [[ "${BACKEND}" == "aptmoe" ]]; then
        apt_route_trace="${APTMOE_ROUTE_ROOT}/${profile_name}/seq_${seq}.npz"
        apt_lookup_table="${APTMOE_LOOKUP_TABLE:-${APTMOE_LOOKUP_ROOT}/${profile_name}.json}"
        if [[ ! -f "${apt_route_trace}" ]]; then
            if [[ "${APTMOE_ALLOW_SYNTHETIC_ROUTING}" -eq 1 ]]; then
                apt_route_trace=""
            elif [[ "${DRY_RUN}" -eq 0 ]]; then
                die "formal APTMoE proxy route trace not found: ${apt_route_trace}"
            fi
        fi
        if [[ ! -f "${apt_lookup_table}" ]]; then
            if [[ "${APTMOE_ALLOW_UNPROFILED_PLACEMENT}" -eq 1 ]]; then
                apt_lookup_table=""
            elif [[ "${DRY_RUN}" -eq 0 ]]; then
                die "formal APTMoE proxy lookup table not found: ${apt_lookup_table}"
            fi
        fi
    elif [[ "${CAPTURE_APTMOE_ROUTES}" -eq 1 ]]; then
        apt_route_trace="${APTMOE_ROUTE_ROOT}/${profile_name}/seq_${seq}.npz"
        route_capture_dir="${APTMOE_ROUTE_ROOT}/${profile_name}/seq_${seq}_ranks"
    fi
    mkdir -p "${timing_dir}"
    write_run_config \
        "${run_dir}/run_config.json" "${profile_name}" "${seq}" "${devices}" \
        "${tokens_per_step}" "${apt_route_trace}" "${apt_lookup_table}"

    local -a route_capture_env=()
    if [[ "${CAPTURE_APTMOE_ROUTES}" -eq 1 ]]; then
        route_capture_env=(
            FFT_ROUTE_TRACE_DIR="${route_capture_dir}"
            FFT_ROUTE_TRACE_SEQUENCE_LENGTH="${seq}"
            FFT_ROUTE_TRACE_PATTERNS="$((WARMUP_STEPS * GRAD_ACCUM_STEPS))"
            FFT_APTMOE_SIMULATION_ROOT="${APTMOE_SIMULATION_ROOT}"
        )
    fi

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
            sed -i "s|^  kt_num_threads:.*|  kt_num_threads: ${kt_owner_threads}|" "${accel_config}"
            local accelerate_bin="${CONDA_BIN_DIR}/accelerate"
            [[ -x "${accelerate_bin}" ]] || accelerate_bin="accelerate"
            command=(
                env
                "${route_capture_env[@]}"
                USE_KT=1
                ACCELERATE_USE_KT=true
                ACCELERATE_KT_TRAIN_MODE="${FINETUNING_TYPE}"
                ACCELERATE_KT_LORA_RANK="${EFFECTIVE_LORA_RANK}"
                ACCELERATE_KT_LORA_ALPHA="${EFFECTIVE_LORA_ALPHA}"
                KT_FINETUNE_MODE="${FINETUNING_TYPE}"
                FFT_TRAINING_BACKEND=kt
                FFT_PRECISION=bf16
                FFT_FINETUNING_TYPE="${FINETUNING_TYPE}"
                FFT_TEXT_ONLY=1
                FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
                FFT_STEP_TIMING_OUT_DIR="${timing_dir}"
                FFT_STEP_TIMING_WARMUP_STEPS="${WARMUP_STEPS}"
                FFT_STEP_TIMING_TOKENS_PER_STEP="${tokens_per_step}"
                FFT_DISABLE_PERF_PROBES=1
                FFT_CPU_THREADS="${cpu_threads}"
                FFT_KT_OWNER_THREADS="${kt_owner_threads}"
                FFT_KT_NON_OWNER_THREADS="${cpu_threads}"
                KT_BACKWARD_TIMING=off
                KT_SFT_PROFILE=0
                KT_REUSE_CHECKPOINT_FORWARD="${KT_DISTRIBUTED_CHECKPOINT_REUSE_ENABLED}"
                KT_REUSE_CHECKPOINT_FORWARD_DISTRIBUTED="${KT_DISTRIBUTED_CHECKPOINT_REUSE_ENABLED}"
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
                "${route_capture_env[@]}"
                USE_KT=0
                ACCELERATE_USE_KT=false
                FFT_TRAINING_BACKEND=deepspeed
                FFT_PRECISION=bf16
                FFT_FINETUNING_TYPE="${FINETUNING_TYPE}"
                FFT_TEXT_ONLY=1
                FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
                KT_FINETUNE_MODE="${FINETUNING_TYPE}"
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
            local torchrun_bin="${CONDA_BIN_DIR}/torchrun"
            [[ -x "${torchrun_bin}" ]] || torchrun_bin="torchrun"
            local apt_output_dir="${APTMOE_SIMULATION_ROOT}/random_weights/${RUN_TIMESTAMP}/${profile_name}/seq_${seq}"
            local -a apt_optional_args=()
            [[ -z "${apt_route_trace}" ]] || \
                apt_optional_args+=(--route-trace "${apt_route_trace}")
            [[ -z "${apt_lookup_table}" ]] || \
                apt_optional_args+=(--lookup-table "${apt_lookup_table}")
            [[ "${APTMOE_ALLOW_SYNTHETIC_ROUTING}" -eq 0 ]] || \
                apt_optional_args+=(--allow-synthetic-routing)
            [[ "${APTMOE_ALLOW_UNPROFILED_PLACEMENT}" -eq 0 ]] || \
                apt_optional_args+=(--allow-unprofiled-placement)
            [[ "${APTMOE_ALLOW_LINEAR_ATTENTION_FALLBACK}" -eq 0 ]] || \
                apt_optional_args+=(--allow-linear-attention-fallback)
            [[ "${KEEP_MODEL_OUTPUT}" -eq 0 ]] || \
                apt_optional_args+=(--save-random-weights)
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
                FFT_APTMOE_SIMULATION_ROOT="${APTMOE_SIMULATION_ROOT}"
                CUDA_CACHE_PATH="${APTMOE_SIMULATION_ROOT}/cache/cuda"
                TORCH_EXTENSIONS_DIR="${APTMOE_SIMULATION_ROOT}/cache/torch_extensions"
                TRITON_CACHE_DIR="${APTMOE_SIMULATION_ROOT}/cache/triton"
                PYTHONPATH="${SCRIPT_DIR}:${APTMOE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
                "${torchrun_bin}" --standalone --nproc_per_node="${NUM_GPUS}"
                "${APTMOE_ENTRYPOINT}"
                --aptmoe-root "${APTMOE_ROOT}"
                --simulation-root "${APTMOE_SIMULATION_ROOT}"
                --deployment-profile "${profile_name}"
                --model-path "${MODEL_PATH}"
                --dataset-dir "${DATASET_DIR}"
                --dataset-name "${DATASET_NAME}"
                --output-dir "${apt_output_dir}"
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
                "${apt_optional_args[@]}"
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
    if [[ "${BACKEND}" == "ktransformers" ]]; then
        log "${BACKEND}/${profile_name}: seq=${seq}, GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, tokens/step=${tokens_per_step}, ${FINETUNING_TYPE}, text-only BF16, KT owner(rank0) threads=${kt_owner_threads}, non-owner rank threads=${cpu_threads}"
    elif [[ "${BACKEND}" == "aptmoe" ]]; then
        log "${BACKEND}/${profile_name}: seq=${seq}, GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, tokens/step=${tokens_per_step}, component-isomorphic BF16 full-update proxy, CPU threads/rank=${cpu_threads}"
    else
        log "${BACKEND}/${profile_name}: seq=${seq}, GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, tokens/step=${tokens_per_step}, text-only BF16, CPU threads/rank=${cpu_threads}"
    fi

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

    if [[ "${exit_code}" -eq 0 && "${CAPTURE_APTMOE_ROUTES}" -eq 1 ]]; then
        if ! "${PYTHON}" "${SCRIPT_DIR}/merge_qwen35_route_traces.py" \
            --input-dir "${route_capture_dir}" \
            --output "${apt_route_trace}" \
            --expected-ranks "${NUM_GPUS}" \
            --expected-patterns "$((WARMUP_STEPS * GRAD_ACCUM_STEPS))" \
            --sequence-length "${seq}" \
            --global-batch-size "${GLOBAL_BATCH_SIZE}" \
            --simulation-root "${APTMOE_SIMULATION_ROOT}"; then
            warn "Exact training succeeded but APTMoE route merge failed"
            exit_code=94
        fi
    fi

    if [[ "${exit_code}" -eq 0 ]]; then
        if [[ "${FINETUNING_TYPE}" == "lora" ]] && \
           ! grep -q "Fine-tuning method: LoRA" "${train_log}"; then
            warn "Training exited successfully but the log does not confirm LoRA mode"
            exit_code=95
        elif [[ "${BACKEND}" == "ktransformers" && "${FINETUNING_TYPE}" == "lora" ]] && \
             ! grep -Eq "Injected [1-9][0-9]* fused expert LoRA params" "${train_log}"; then
            warn "Training exited successfully but KT fused-expert LoRA optimizer injection is missing"
            exit_code=96
        elif [[ ! -f "${timing_dir}/step_timing.json" ]]; then
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
    local -a profile_sequence_lengths=()
    if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
        profile_sequence_lengths=("${SEQUENCE_LENGTHS_OVERRIDE[@]}")
    elif [[ "${profile_name}" == "server" ]]; then
        profile_sequence_lengths=("${SERVER_SEQUENCE_LENGTHS[@]}")
    else
        profile_sequence_lengths=("${CONSUMER_SEQUENCE_LENGTHS[@]}")
    fi
    profile_parameters "${profile_name}"
    check_visible_gpu_capacity "${NUM_GPUS}"
    local devices
    devices="$(resolve_devices "${NUM_GPUS}")"
    build_resource_policy "${profile_name}"
    local profile_dir="${RUN_ROOT}/${profile_name}_${NUM_GPUS}gpu_batch${GLOBAL_BATCH_SIZE}"
    mkdir -p "${profile_dir}"
    if [[ "${BACKEND}" == "ktransformers" ]]; then
        log "Profile ${profile_name}: devices=${devices}, memory=${MEMORY_LIMIT_LABEL}, NUMA=${NUMA_POLICY_LABEL}, KT owner(rank0) threads=${KT_OWNER_THREADS}, non-owner rank threads=${CPU_THREADS_PER_RANK}, planned CPU budget=${CPU_THREAD_BUDGET_TOTAL}/${PHYSICAL_CORES}"
    else
        log "Profile ${profile_name}: devices=${devices}, memory=${MEMORY_LIMIT_LABEL}, NUMA=${NUMA_POLICY_LABEL}, CPU threads/rank=${CPU_THREADS_PER_RANK}"
    fi

    log "Profile ${profile_name} sequences: ${profile_sequence_lengths[*]}"
    local profile_status=0 seq
    for seq in "${profile_sequence_lengths[@]}"; do
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
if [[ "${BACKEND}" == "aptmoe" ]]; then
    RUN_ROOT="${LOG_BASE}/${RUN_TIMESTAMP}_APTMOE_BF16_DEPLOYMENT_PROXY_SWEEP"
else
    RUN_ROOT="${LOG_BASE}/${RUN_TIMESTAMP}_${BACKEND^^}_BF16_${FINETUNING_TYPE^^}_SWEEP"
fi
mkdir -p "${RUN_ROOT}"
validate_dataset

if [[ "${BACKEND}" == "aptmoe" ]]; then
    log "Qwen3.5-35B-A3B component-isomorphic deployment proxy: backend=aptmoe, random BF16 weights, profile=${PROFILE}"
    log "Proxy artifacts (gitignored): ${APTMOE_SIMULATION_ROOT}"
else
    log "Qwen3.5-35B-A3B text-only ${FINETUNING_TYPE} sweep: backend=${BACKEND}, precision=BF16, profile=${PROFILE}"
fi
if [[ "${FINETUNING_TYPE}" == "lora" ]]; then
    log "LoRA parameters: rank=${EFFECTIVE_LORA_RANK}, alpha=${EFFECTIVE_LORA_ALPHA}, target=all"
fi
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    log "Sequence override: ${SEQUENCE_LENGTHS_OVERRIDE[*]}"
else
    case "${PROFILE}" in
        server) log "Server sequences: ${SERVER_SEQUENCE_LENGTHS[*]}" ;;
        consumer) log "Consumer sequences: ${CONSUMER_SEQUENCE_LENGTHS[*]}" ;;
        both)
            log "Server sequences: ${SERVER_SEQUENCE_LENGTHS[*]}"
            log "Consumer sequences: ${CONSUMER_SEQUENCE_LENGTHS[*]}"
            ;;
    esac
fi
log "Steps=${STEPS}; warmup excluded=${WARMUP_STEPS}; GAS=${GRAD_ACCUM_STEPS}"
if [[ "${BACKEND}" == "ktransformers" ]]; then
    log "Distributed checkpoint forward reuse: ${KT_DISTRIBUTED_CHECKPOINT_REUSE}"
fi
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

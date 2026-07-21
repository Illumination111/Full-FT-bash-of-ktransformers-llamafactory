#!/usr/bin/env bash
# Full-FT-only DeepSpeed ZeRO-3 run with nested backward/optimizer probes.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_finetune_perf_test_bf16_deepspeed.sh"

for arg in "$@"; do
    case "${arg}" in
        --mode|--mode=*|--finetuning-mode|--finetuning-mode=*)
            echo "This probe runner is Full-FT-only; do not pass ${arg}." >&2
            exit 2
            ;;
    esac
done

# exact establishes CUDA-completed boundaries around DeepSpeedEngine backward,
# engine step and ZeRO optimizer step.  Use low_overhead for a host-wall-only
# companion run that estimates probe perturbation.
export DS_PROBE_MODE="${DS_PROBE_MODE:-exact}"
case "${DS_PROBE_MODE}" in
    exact|low_overhead) ;;
    *)
        echo "DS_PROBE_MODE must be exact or low_overhead for this probe runner." >&2
        exit 2
        ;;
esac

# CPUAdam timing is sensitive to OpenMP width.  Keep it controlled and recorded;
# this host has 96 physical cores.  FFT_OMP_NUM_THREADS is the explicit A/B knob.
export OMP_NUM_THREADS="${FFT_OMP_NUM_THREADS:-${OMP_NUM_THREADS:-96}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

STEPS="${STEPS:-35}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GAS="${GAS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"

echo "Full-FT DeepSpeed probe: mode=${DS_PROBE_MODE}, OMP=${OMP_NUM_THREADS}, steps=${STEPS}, warmup=${WARMUP_STEPS}"

exec bash "${RUNNER}" \
    --mode full \
    --gpus "${GPUS}" \
    --batch-size "${BATCH_SIZE}" \
    --gas "${GAS}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    "$@"

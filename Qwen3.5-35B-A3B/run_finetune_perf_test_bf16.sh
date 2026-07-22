#!/usr/bin/env bash
# Default compatibility entrypoint: KTransformers AMX BF16.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_finetune_perf_test_bf16_ktransformers.sh" "$@"

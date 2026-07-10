#!/usr/bin/env bash
# =============================================================================
# Qwen3-30B-A3B KTransformers Full Fine-Tuning (FFT) Test Script
# Configuration: AMXBF16 + 4x RTX 4090 GPU (native BF16 precision)
# Capability note: current LLaMA-Factory KT integration is LoRA SFT oriented.
# If the base config requests finetuning_type=full with use_kt=true, training
# phases are marked Unsupported instead of forcing an out-of-contract run.
#
# Usage:
#   bash run_fft_test_4gpu_bf16.sh [--skip-phase1] [--skip-phase4] [--only-phase4]
#                                   [--phase4-steps N] [--dry-run]
#
# Phases:
#   Phase 0a — Weight memory breakdown (static analysis, always runs)
#   Phase 0b — Dataset truncation analysis per sample (always runs)
#   Phase 1  — Basic validation (3 steps): verify AMXBF16 FFT pipeline
#   Phase 4  — Stability + accurate TPS benchmark (default 50 steps)
#
# Key parameters:
#   Backend     : AMXBF16 (native BF16 MoE kernels on CPU, no pre-conversion)
#   GPUs        : 4x RTX 4090, fixed
#   Model path  : /mnt/data3/models/Qwen3-30B-A3B  (HF BF16 safetensors)
#   kt_weight   : same as model path (AMXBF16 reads BF16 directly)
#   cutoff_len  : 1024  (Qwen3 tokenizer encodes fft_stress_100 at 1482–1633
#                        tok/sample, all above 1024 → every sample truncated
#                        to exactly 1024 tokens for accurate TPS measurement)
#
# vs run_fft_test_4gpu.sh (AMXINT8):
#   - No separate INT8 weight path; BF16 weights used directly by kernel
#   - CPU expert weights: ~58 GB BF16 (vs ~29 GB INT8)
#   - No quantization/dequantization overhead per step
#   - Slightly higher CPU memory, potentially lower per-token latency
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${SCRIPT_DIR}/test_log"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
DATA_DIR="${SCRIPT_DIR}/data"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze.py"

LLAMA_FACTORY_DIR="/mnt/data2/wbw/LLaMA-Factory"

# AMXBF16 reads BF16 HF safetensors directly — single model path for both
# model_name_or_path and kt_weight_path
MODEL_PATH="/mnt/data3/models/Qwen3-30B-A3B"

# 4-GPU AMXBF16 accelerate config
ACCEL_CONFIG="${CONFIGS_DIR}/accelerate_fft_amxbf16_4gpu.yaml"
TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_fft_qwen3_30b.yaml"

NUM_GPUS=4
BACKEND_LABEL="AMX_BF16"
RUN_LABEL="${NUM_GPUS}gpu_${BACKEND_LABEL}"
# cutoff_len=1024: all fft_stress_100 samples (1482–1633 Qwen3 tokens) exceed
# 1024, so every step contributes exactly 1024 tokens → TPS measurement accurate
CUTOFF_LEN=1024
WARMUP_SKIP=5    # Steps discarded before computing stable TPS

MONITOR_FIFO="/tmp/fft_monitor_events_4gpu_bf16.fifo"
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
SKIP_PHASE1=0; SKIP_PHASE4=0
DRY_RUN=0
PHASE4_STEPS=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-phase1)   SKIP_PHASE1=1 ;;
        --skip-phase4)   SKIP_PHASE4=1 ;;
        --only-phase1)   SKIP_PHASE4=1 ;;
        --only-phase4)   SKIP_PHASE1=1 ;;
        --phase4-steps)  PHASE4_STEPS="$2"; shift ;;
        --dry-run)       DRY_RUN=1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

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

    log "Python: $(${PYTHON} --version 2>&1)"
    log "Conda env: ${CONDA_ENV}"

    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi not found"
        exit 1
    fi
    ACTUAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    log "Detected GPUs: ${ACTUAL_GPUS}"
    if [[ "${ACTUAL_GPUS}" -lt "${NUM_GPUS}" ]]; then
        error "System GPU count (${ACTUAL_GPUS}) < required ${NUM_GPUS}"
        exit 1
    fi

    if [[ ! -d "${MODEL_PATH}" ]]; then
        error "Model path not found: ${MODEL_PATH}"
        exit 1
    fi
    ok "Model path (AMXBF16, BF16 HF weights): ${MODEL_PATH}"

    if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
        error "LLaMA-Factory not found: ${LLAMA_FACTORY_DIR}"
        exit 1
    fi
    ok "LLaMA-Factory: ${LLAMA_FACTORY_DIR}"

    if [[ ! -f "${ACCEL_CONFIG}" ]]; then
        error "Accelerate config not found: ${ACCEL_CONFIG}"
        exit 1
    fi
    ok "Accelerate config: $(basename ${ACCEL_CONFIG})"

    if [[ ! -f "${DATA_DIR}/fft_stress_100.json" ]]; then
        warn "Dataset not found — generating..."
        "${PYTHON}" "${SCRIPT_DIR}/gen_dataset.py" || {
            error "Dataset generation failed"
            exit 1
        }
    fi
    ok "Dataset: ${DATA_DIR}/fft_stress_100.json"

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
# Phase 0a: Weight memory breakdown (static analysis from model config)
#
# AMXBF16 memory map:
#
#   GPU (non-expert, BF16, sharded across 4 GPUs via FSDP):
#     embed_tokens   : V × H
#     lm_head        : H × V  (not tied in Qwen3-30B-A3B)
#     attention × L  : q+k+v+o projections
#     shared_expert × L : gate+up+down = 3 × H × I
#     router × L     : H × E
#     rms_norm × L   : tiny
#
#   CPU (expert weights, BF16, 2 bytes/param):
#     128 experts × 3 projections × H × moe_I × 48 layers
#     NUMA tensor-parallel: each NUMA node computes ½ output cols
#     No quantization overhead — weights loaded directly from HF safetensors
#
#   CPU (training buffers, FP32):
#     Expert gradients   : BF16 param count × 4 bytes
#     Adam 1st moment m  : BF16 param count × 4 bytes
#     Adam 2nd moment v  : BF16 param count × 4 bytes
# --------------------------------------------------------------------------- #
analyze_weight_memory() {
    phase_banner "Phase 0a: Weight Memory Breakdown (AMXBF16, 4-GPU FSDP)"

    "${PYTHON}" - "${MODEL_PATH}" "${NUM_GPUS}" <<'PYEOF'
import json, os, sys

model_path = sys.argv[1]
num_gpus   = int(sys.argv[2])

cfg_path = os.path.join(model_path, "config.json")
if not os.path.exists(cfg_path):
    print(f"  [WARN] config.json not found at {model_path}")
    sys.exit(0)

with open(cfg_path) as f:
    cfg = json.load(f)

H     = cfg["hidden_size"]           # 2048
I     = cfg["intermediate_size"]     # 6144  (shared expert)
moe_I = cfg["moe_intermediate_size"] # 768   (per MoE expert)
nh    = cfg["num_attention_heads"]   # 32
nkv   = cfg["num_key_value_heads"]   # 4
dh    = cfg.get("head_dim", H // nh) # 128
E     = cfg["num_experts"]           # 128
L     = cfg["num_hidden_layers"]     # 48
V     = cfg["vocab_size"]            # 151936

def gb(params, dtype_bytes=2):
    return params * dtype_bytes / 1e9

# ---- Non-expert GPU params (BF16) ----
p_embed   = V * H
p_lmhead  = V * H
p_attn_q  = H * (nh * dh)
p_attn_k  = H * (nkv * dh)
p_attn_v  = H * (nkv * dh)
p_attn_o  = (nh * dh) * H
p_attn_L  = (p_attn_q + p_attn_k + p_attn_v + p_attn_o) * L
p_sh_L    = 3 * H * I * L
p_router_L= H * E * L
p_norm_L  = 2 * H * L + 2 * H

p_gpu_total = p_embed + p_lmhead + p_attn_L + p_sh_L + p_router_L + p_norm_L

print("=" * 64)
print(f"  Qwen3-30B-A3B  (qwen3_moe)  —  AMXBF16 native BF16")
print(f"  H={H}  I={I}  moe_I={moe_I}  nh={nh}  nkv={nkv}  dh={dh}")
print(f"  E={E} experts  L={L} layers  V={V}")
print("=" * 64)

print("\n── GPU Non-Expert Params (BF16, FSDP sharded across 4 GPUs) ─")
rows = [
    ("embed_tokens",            p_embed,             "V × H"),
    ("lm_head",                 p_lmhead,            "H × V (not tied)"),
    (f"attention × {L} layers", p_attn_L,            "q+k+v+o proj"),
    (f"  per layer",            p_attn_L // L,       ""),
    (f"shared_expert × {L}",    p_sh_L,              f"gate+up+down, I={I}"),
    (f"  per layer",            p_sh_L // L,         ""),
    (f"router × {L}",           p_router_L,          "H → E logits"),
    (f"rms_norm × {L}",         p_norm_L,            "layernorm params"),
]
for name, params, note in rows:
    note_s = f"  [{note}]" if note else ""
    print(f"  {name:<32} {params/1e6:8.1f}M  {gb(params,2):6.2f} GB BF16{note_s}")
print(f"  {'─'*60}")
print(f"  {'TOTAL GPU params':<32} {p_gpu_total/1e6:8.1f}M  {gb(p_gpu_total,2):6.2f} GB BF16")
print(f"  {'  per GPU (' + str(num_gpus) + 'x FSDP)':<32} {'':>8}   {gb(p_gpu_total,2)/num_gpus:6.2f} GB per card")

# ---- Expert weights (BF16, no quantization) ----
p_exp_bf16 = E * 3 * H * moe_I * L   # total BF16 bytes (2B/param)

print("\n── CPU Expert Weights (AMXBF16, native BF16, 2 NUMA nodes) ──")
print(f"  Per expert:    3 × {H} × {moe_I} = {3*H*moe_I/1e6:.2f}M params")
print(f"  Per layer:     {E} experts × {3*H*moe_I/1e6:.2f}M = {E*3*H*moe_I/1e6:.0f}M")
print(f"  All {L} layers:  {p_exp_bf16/1e9:.2f} GB BF16  (2B/param, no quantization)")
print(f"  Per NUMA node (½ compute): {p_exp_bf16/2/1e9:.2f} GB")
print(f"  [vs AMXINT8]  INT8 would be {p_exp_bf16/2/1e9:.2f}→{p_exp_bf16/4/1e9:.2f} GB")
print(f"                BF16 uses {p_exp_bf16/1e9:.1f} GB (2× INT8 memory footprint)")

# ---- CPU training buffers (FP32) ----
grad_bytes = p_exp_bf16 * 4   # FP32 grads (4B) for BF16 params
adam_m     = grad_bytes
adam_v     = grad_bytes
cpu_total  = p_exp_bf16 + grad_bytes + adam_m + adam_v

print("\n── CPU Training Buffers (FFT backward, FP32) ────────────────")
print(f"  Expert BF16 weights (loaded):   {p_exp_bf16/1e9:.1f} GB")
print(f"  Expert FP32 gradients:          {grad_bytes/1e9:.1f} GB  ({E*3*H*moe_I*L/1e6:.0f}M params × 4B)")
print(f"  Adam 1st moment (m):            {adam_m/1e9:.1f} GB")
print(f"  Adam 2nd moment (v):            {adam_v/1e9:.1f} GB")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  CPU expert subtotal:           {cpu_total/1e9:.1f} GB  (+ ~10–30 GB NUMA working buf)")

# ---- GPU VRAM per card ----
vram_params = gb(p_gpu_total, 2) / num_gpus
vram_grads  = gb(p_gpu_total, 4) / num_gpus   # FP32 master grads (FSDP)
vram_optim  = gb(p_gpu_total, 8) / num_gpus   # AdamW m+v in FP32
vram_activ  = 2.0                             # activations (seq=1024, grad-ckpt)
vram_total  = vram_params + vram_grads + vram_optim + vram_activ

print("\n── GPU VRAM per Card (4× RTX 4090, 48 GB each) ─────────────")
print(f"  Sharded model params (BF16): {vram_params:.2f} GB")
print(f"  FP32 master grads (FSDP):    {vram_grads:.2f} GB")
print(f"  AdamW states (FP32 m+v):     {vram_optim:.2f} GB")
print(f"  Activations (seq=1024, est): {vram_activ:.2f} GB")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  Estimated peak VRAM / card:  {vram_total:.1f} GB  (limit: 48 GB)")
headroom = 48.0 - vram_total
print(f"  Headroom:                    {headroom:.1f} GB  ({'OK' if headroom > 10 else 'WARN: tight'})")

print("\n── Summary Table ─────────────────────────────────────────────")
print(f"  {'Component':<38} {'Size':>9}  Location")
print(f"  {'─'*64}")
print(f"  {'Expert weights (BF16, native)':<38} {p_exp_bf16/1e9:>7.1f}G  CPU (2 NUMA nodes)")
print(f"  {'Expert FP32 gradients':<38} {grad_bytes/1e9:>7.1f}G  CPU")
print(f"  {'Expert AdamW (m+v, FP32)':<38} {(adam_m+adam_v)/1e9:>7.1f}G  CPU")
print(f"  {'Non-expert params (BF16, all 4 GPUs)':<38} {gb(p_gpu_total,2):>7.2f}G  GPU (FSDP)")
print(f"  {'  per GPU':<38} {vram_params:>7.2f}G  per card")
print(f"  {'Non-expert AdamW + grads per GPU':<38} {(vram_grads+vram_optim):>7.2f}G  per card")
print(f"  {'─'*64}")
print(f"  {'Total CPU expert footprint':<38} {cpu_total/1e9:>7.1f}G")
print(f"  {'Estimated peak VRAM per card':<38} {vram_total:>7.1f}G  / 48G")

print(f"\n── BF16 vs INT8 Comparison ───────────────────────────────────")
int8_cpu = p_exp_bf16 / 2  # INT8 weights
int8_scale = int8_cpu / 32 * 4
int8_grads = p_exp_bf16 * 4  # Same FP32 grads (precision maintained)
print(f"  {'':38} {'AMXBF16':>9}  {'AMXINT8':>9}")
print(f"  {'─'*62}")
print(f"  {'Expert weights on CPU':<38} {p_exp_bf16/1e9:>7.1f}G  {(int8_cpu+int8_scale)/1e9:>7.1f}G")
print(f"  {'Expert FP32 gradients':<38} {grad_bytes/1e9:>7.1f}G  {int8_grads/1e9:>7.1f}G")
print(f"  {'Expert AdamW (m+v)':<38} {(adam_m+adam_v)/1e9:>7.1f}G  {int8_grads*2/1e9:>7.1f}G")
print(f"  {'GPU VRAM per card (est)':<38} {vram_total:>7.1f}G  {'~13.7':>9}G")
print(f"  [Note] BF16 has ~{(p_exp_bf16-(int8_cpu+int8_scale))/1e9:.1f}G more CPU weight memory vs INT8")
PYEOF
}

# --------------------------------------------------------------------------- #
# Phase 0b: Dataset truncation analysis (per-sample token counts at cutoff=1024)
# --------------------------------------------------------------------------- #
analyze_dataset_truncation() {
    phase_banner "Phase 0b: Dataset Truncation Analysis (cutoff_len=${CUTOFF_LEN})"

    "${PYTHON}" - "${MODEL_PATH}" "${DATA_DIR}/fft_stress_100.json" "${CUTOFF_LEN}" <<'PYEOF'
import json, sys, os

tokenizer_path = sys.argv[1]
dataset_path   = sys.argv[2]
cutoff_len     = int(sys.argv[3])

with open(dataset_path) as f:
    samples = json.load(f)

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    have_tokenizer = True
    print(f"  Tokenizer: {tokenizer_path}  (vocab={tok.vocab_size})")
except Exception as e:
    tok = None
    have_tokenizer = False
    print(f"  [WARN] Tokenizer not loaded ({e}); using char÷3.5 approximation")

print(f"  Dataset : {dataset_path}  ({len(samples)} samples)")
print(f"  Cutoff  : {cutoff_len} tokens\n")

print(f"  {'#':>4}  {'raw_chars':>10}  {'est_tok':>8}  "
      f"{'full_tok':>9}  {'trunc_tok':>9}  {'trunc%':>7}  status")
print(f"  {'─'*75}")

n_trunc = 0; n_exact = 0; n_under = 0
total_full = 0; total_trunc = 0
min_tok = 10**9; max_tok = 0
results = []

for idx, s in enumerate(samples):
    instr  = s.get("instruction", "")
    inp    = s.get("input", "")
    output = s.get("output", "")
    raw_chars = len(instr) + len(inp) + len(output)

    if have_tokenizer:
        messages = [
            {"role": "user",      "content": (instr + ("\n" + inp if inp else "")).strip()},
            {"role": "assistant", "content": output},
        ]
        try:
            txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_len = len(tok(txt, add_special_tokens=False)["input_ids"])
        except Exception:
            full_len = len(tok(instr + inp + output)["input_ids"])
    else:
        full_len = int(raw_chars / 3.5)

    est_tok   = int(raw_chars / 3.5)
    trunc     = max(0, full_len - cutoff_len)
    trunc_pct = trunc / full_len * 100 if full_len > 0 else 0.0

    if trunc > 0:
        status = "TRUNC"; n_trunc += 1
    elif full_len == cutoff_len:
        status = "exact"; n_exact += 1
    else:
        status = "full";  n_under += 1

    total_full  += full_len
    total_trunc += trunc
    min_tok = min(min_tok, full_len)
    max_tok = max(max_tok, full_len)
    results.append((idx+1, raw_chars, est_tok, full_len, trunc, trunc_pct, status))

for idx, rc, et, fl, tr, pct, st in results:
    flag = " ★" if tr > 0 else ""
    print(f"  {idx:>4}  {rc:>10,}  {et:>8,}  {fl:>9,}  {tr:>9,}  {pct:>6.1f}%  {st}{flag}")

retained = total_full - total_trunc
print(f"\n  {'─'*75}")
print(f"  {len(samples)} samples: {n_trunc} TRUNC  {n_exact} exact  {n_under} under cutoff")
print(f"  tok range: min={min_tok:,}  max={max_tok:,}  avg={total_full//len(samples):,}")
print(f"  tokens retained: {retained:,} / {total_full:,}  ({retained/total_full*100:.1f}%)")
if n_trunc > 0:
    print(f"  total truncated: {total_trunc:,} tokens across {n_trunc} samples")
    if n_trunc == len(samples):
        print(f"  → ALL samples truncated to exactly {cutoff_len} tokens"
              f" — TPS denominator is {cutoff_len} × {4} GPUs = {cutoff_len*4} tok/step")
else:
    print(f"  → No truncation at cutoff_len={cutoff_len}; all samples fit in full")
PYEOF
}

# --------------------------------------------------------------------------- #
# Resource estimation
# --------------------------------------------------------------------------- #
estimate_resources() {
    phase_banner "Resource Estimation (AMXBF16, 4x RTX 4090)"

    echo -e "${BOLD}== Model (Qwen3-30B-A3B, native BF16) ==${NC}"
    echo "  Architecture : Qwen3MoeForCausalLM (MoE)"
    echo "  Layers/Experts: 48 layers, 128 experts/layer, top-8 routing"
    echo "  hidden=2048  shared_intermediate=6144  moe_intermediate=768"
    echo "  Backend      : AMXBF16 (native BF16, reads HF safetensors directly)"
    echo "  Model path   : ${MODEL_PATH}"
    echo ""
    echo -e "${BOLD}== FSDP (4x RTX 4090, 48 GB each) ==${NC}"
    echo "  Config: $(basename ${ACCEL_CONFIG})"
    echo "  FSDP-v2, reshard_after_forward=true"
    echo "  Expert weights on CPU (BF16, ~58 GB); GPU holds non-expert params only"
    echo ""
    echo -e "${BOLD}== TPS Measurement ==${NC}"
    echo "  Dataset    : fft_stress_100  (100 samples)"
    echo "  cutoff_len : ${CUTOFF_LEN} tokens  (all samples exceed this → full truncation)"
    echo "  tok/step   : ${NUM_GPUS} GPUs × ${CUTOFF_LEN} = $((NUM_GPUS * CUTOFF_LEN)) tokens"
    echo "  warmup_skip: ${WARMUP_SKIP} steps excluded from TPS"
    echo ""
}

# --------------------------------------------------------------------------- #
# Run directory + FIFO setup
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

# --------------------------------------------------------------------------- #
# Capability guard
# --------------------------------------------------------------------------- #
record_unsupported_capability() {
    local finetuning_type=""
    local use_kt=""

    finetuning_type=$(grep -E "^finetuning_type:" "${TRAIN_CONFIG_BASE}" | awk '{print $2}' | tr -d "'\"" || true)
    use_kt=$(grep -E "^use_kt:" "${TRAIN_CONFIG_BASE}" | awk '{print $2}' | tr -d "'\"" || true)

    if [[ "${use_kt}" == "true" && "${finetuning_type}" == "full" && "${NUM_GPUS}" -gt 1 ]]; then
        phase_banner "Capability Check"
        warn "Skipping training phases: current LLaMA-Factory KT integration supports KT SFT through LoRA, not full fine-tuning with multi-GPU FSDP."
        warn "Static analysis artifacts are still saved; Phase 1/4 will be marked Unsupported."
        {
            echo "Unsupported test combination"
            echo ""
            echo "Reason:"
            echo "  - TRAIN_CONFIG_BASE requests finetuning_type=full and use_kt=true."
            echo "  - Current LLaMA-Factory KT integration is designed around KT LoRA SFT."
            echo "  - Running this as a 4-GPU full-FT test exceeds the framework/backend support boundary."
            echo ""
            echo "Action:"
            echo "  - Use a LoRA KT SFT config for KTransformers tests, or run non-KT full FT with the framework's normal distributed backend."
        } > "${LOG_DIR}/unsupported.txt"
        return 0
    fi

    return 1
}

send_event() {
    local msg="$1"
    [[ -p "${MONITOR_FIFO}" ]] && echo "${msg}" >> "${MONITOR_FIFO}" 2>/dev/null || true
}

# --------------------------------------------------------------------------- #
# Real-time system monitor
# --------------------------------------------------------------------------- #
start_monitor() {
    log "Starting system monitor (interval=2s, process-tree root PID=$$)..."
    "${PYTHON}" "${MONITOR_SCRIPT}" \
        --out "${LOG_DIR}/monitor.csv" \
        --fifo "${MONITOR_FIFO}" \
        --interval 2 \
        --pid $$ \
        >> "${LOG_DIR}/monitor.log" 2>&1 &
    MONITOR_PID=$!
    log "Monitor PID: ${MONITOR_PID} (tracking script tree $$)"
    sleep 1
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
# Remove checkpoint after each phase to free disk space
# --------------------------------------------------------------------------- #
cleanup_model_output() {
    local phase_name="$1"
    local model_dir="${LOG_DIR}/${phase_name}/model_output"
    if [[ -d "${model_dir}" ]]; then
        log "[cleanup] Removing ${model_dir}..."
        rm -rf "${model_dir}"
        ok "[cleanup] ${phase_name}/model_output deleted"
    fi
}

# --------------------------------------------------------------------------- #
# Build per-phase training config by patching the base template.
# For AMXBF16, model_name_or_path == kt_weight_path (single BF16 model path).
# Also patches cutoff_len for this run.
# --------------------------------------------------------------------------- #
make_phase_config() {
    local phase_name="$1"; shift
    local phase_dir="${LOG_DIR}/${phase_name}"
    mkdir -p "${phase_dir}"
    local cfg="${phase_dir}/train_config.yaml"

    cp "${TRAIN_CONFIG_BASE}" "${cfg}"

    # AMXBF16: both model_name_or_path and kt_weight_path point to BF16 model
    sed -i "s|output_dir: .*|output_dir: ${phase_dir}/model_output|g" "${cfg}"
    sed -i "s|model_name_or_path: .*|model_name_or_path: ${MODEL_PATH}|g" "${cfg}"
    sed -i "s|kt_weight_path: .*|kt_weight_path: ${MODEL_PATH}|g" "${cfg}"
    sed -i "s|cutoff_len: .*|cutoff_len: ${CUTOFF_LEN}|g" "${cfg}"
    sed -i "s|overwrite_output_dir: .*|overwrite_output_dir: true|g" "${cfg}" || \
        echo "overwrite_output_dir: true" >> "${cfg}"

    # Per-phase key=value overrides
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
# Execute a training phase via accelerate launch (4 GPUs, fixed)
# --------------------------------------------------------------------------- #
run_train() {
    local phase_name="$1"
    local desc="$2"
    local train_cfg="$3"
    local phase_log="${LOG_DIR}/${phase_name}/train.log"
    local exit_code=0

    log "Starting [${phase_name}]: ${desc}"
    log "  config   : $(basename ${ACCEL_CONFIG})"
    log "  GPUs     : ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=0,1,2,3)"
    log "  backend  : AMXBF16  |  model: ${MODEL_PATH}"
    send_event "phase:${phase_name}"
    send_event "event:train_start"

    local gpus_str
    gpus_str=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',')

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
        log "[DRY-RUN] ${cmd[*]}"
        return 0
    fi

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
# Post-phase log analysis
# --------------------------------------------------------------------------- #
analyze_log() {
    local phase_name="$1"
    local log_file="${LOG_DIR}/${phase_name}/train.log"
    local result_file="${LOG_DIR}/${phase_name}/log_analysis.txt"

    [[ ! -f "${log_file}" ]] && return

    {
        echo "=== [${phase_name}] Log Analysis (AMXBF16 4-GPU) ==="
        echo "--- Loss (last 20) ---"
        grep -i "loss" "${log_file}" | tail -20 || echo "(none)"
        echo ""
        echo "--- Gradient Norm ---"
        grep -i "grad_norm\|gradient_norm" "${log_file}" | tail -20 || echo "(none)"
        echo ""
        echo "--- NaN / Inf count ---"
        local nn ni
        nn=$(grep -ci "nan" "${log_file}" 2>/dev/null || echo 0)
        ni=$(grep -ci "inf" "${log_file}" 2>/dev/null || echo 0)
        echo "NaN lines: ${nn};  Inf lines: ${ni}"
        echo ""
        echo "--- KT / AMX / MoE alerts ---"
        grep -i "ktransformer\|kt_kernel\|amx\|moe\|expert\|backward" \
             "${log_file}" 2>/dev/null \
             | grep -i "warn\|error\|fail\|bug\|overflow\|sigsegv" | tail -20 \
             || echo "(none)"
        echo ""
        echo "--- Critical errors ---"
        grep -qi "out of memory\|cuda error" "${log_file}" && echo "⚠ CUDA OOM" || true
        grep -qi "segmentation fault\|sigsegv\|core dumped" "${log_file}" && echo "⚠ SIGSEGV" || true
        grep -qi "ddp_timeout\|timeout expired" "${log_file}" && echo "⚠ DDP timeout (P2: CPU backward slow?)" || true
    } | tee "${result_file}"
}

# --------------------------------------------------------------------------- #
# Phase 1: Basic validation — 3 steps, no checkpoint
# --------------------------------------------------------------------------- #
run_phase1() {
    phase_banner "Phase 1: Basic Validation (3 steps, AMXBF16 4-GPU)"

    local cfg
    cfg=$(make_phase_config "phase1" \
        "max_steps=3" \
        "save_strategy='no'" \
        "gradient_accumulation_steps=1" \
        "logging_steps=1")

    log "Goal: verify AMXBF16 FFT pipeline init/forward/backward (3 steps, no crash)"
    log "Config: ${cfg}"

    local exit_code=0
    run_train "phase1" "3-step basic validation (AMXBF16 4GPU)" "${cfg}" || exit_code=$?

    if [[ "${exit_code}" -eq 0 ]]; then
        ok "Phase 1 passed"
    else
        error "Phase 1 failed (exit ${exit_code})"
        local log_f="${LOG_DIR}/phase1/train.log"
        if [[ -f "${log_f}" ]]; then
            grep -qi "segmentation fault\|sigsegv" "${log_f}" && \
                error "  → SIGSEGV: C++ buffer OOB or gradient index bug"
            grep -qi "nan\|inf" "${log_f}" && \
                error "  → NaN/Inf: BF16 gradient instability or router collapse"
            grep -qi "out of memory" "${log_f}" && \
                error "  → CUDA OOM: check non-expert GPU allocation"
            grep -qi "ddp_timeout\|timeout" "${log_f}" && \
                error "  → DDP timeout: AMXBF16 backward may be slow at first step"
        fi
    fi
    analyze_log "phase1"
    echo "${exit_code}" > "${LOG_DIR}/phase1/exit_code.txt"
    cleanup_model_output "phase1"
    return "${exit_code}"
}

# --------------------------------------------------------------------------- #
# Phase 4: Stability + accurate TPS benchmark
# --------------------------------------------------------------------------- #
run_phase4() {
    phase_banner "Phase 4: Stability + TPS Benchmark (${PHASE4_STEPS} steps, AMXBF16 4-GPU)"

    local cfg
    cfg=$(make_phase_config "phase4" \
        "max_steps=${PHASE4_STEPS}" \
        "gradient_accumulation_steps=1" \
        "save_strategy='no'" \
        "logging_steps=1")

    log "Steps    : ${PHASE4_STEPS}  |  GPUs: ${NUM_GPUS}  |  cutoff_len: ${CUTOFF_LEN}"
    log "TPS      : skip first ${WARMUP_SKIP} warmup steps  |  stable_tok/step = $((NUM_GPUS * CUTOFF_LEN))"
    log "Backend  : AMXBF16  |  model: ${MODEL_PATH}"

    local t_start
    t_start=$(date +%s)
    send_event "phase:phase4"

    local exit_code=0
    run_train "phase4" "stability + TPS (${PHASE4_STEPS} steps, 4GPU AMXBF16)" "${cfg}" \
        || exit_code=$?

    local t_end t_sec
    t_end=$(date +%s)
    t_sec=$((t_end - t_start))

    {
        echo "=== Phase 4 Performance Analysis (AMXBF16 4-GPU) ==="
        echo "Wall time: ${t_sec} s  ($(( t_sec/60 ))m $(( t_sec%60 ))s)"

        local actual_steps
        actual_steps=$(grep -c "{'loss'" "${LOG_DIR}/phase4/train.log" 2>/dev/null || \
                       grep -c '"loss"' "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "steps_completed: ${actual_steps} / ${PHASE4_STEPS}"

        echo ""
        echo "--- Accurate TPS (warmup excluded) ---"
        "${PYTHON}" - \
            "${LOG_DIR}/phase4/train.log" \
            "${CUTOFF_LEN}" \
            "${NUM_GPUS}" \
            "${WARMUP_SKIP}" <<'PYEOF'
import re, sys, statistics

log_path   = sys.argv[1]
cutoff_len = int(sys.argv[2])
num_gpus   = int(sys.argv[3])
warmup_n   = int(sys.argv[4])

log_text = open(log_path, errors="replace").read()
pattern = re.compile(r'(\d+)/\d+\s+\[[\d:]+<[\d:]+,\s*([\d.]+)\s*s/it\]')
seen = {}
for m in pattern.finditer(log_text):
    seen[int(m.group(1))] = float(m.group(2))
steps = sorted(seen.items())

tokens_per_step = num_gpus * cutoff_len
stable = [(s, t) for s, t in steps if s > warmup_n]

print(f"  total_steps_logged : {len(steps)}")
print(f"  warmup_steps_skip  : {warmup_n}")
print(f"  stable_steps       : {len(stable)}")
print(f"  tokens_per_step    : {tokens_per_step}  ({num_gpus} GPUs × {cutoff_len} tok)")

if not stable:
    print("  (not enough steps for stable TPS — run more steps)")
    sys.exit(0)

times = [t for _, t in stable]
avg_t = sum(times) / len(times)
med_t = statistics.median(times)

print(f"\n  step_time_avg      : {avg_t:.2f} s")
print(f"  step_time_median   : {med_t:.2f} s")
print(f"  step_time_min      : {min(times):.2f} s")
print(f"  step_time_max      : {max(times):.2f} s")
print(f"\n  TPS (avg)          : {tokens_per_step / avg_t:.1f} tok/s")
print(f"  TPS (median)       : {tokens_per_step / med_t:.1f} tok/s")
print(f"  TPS (peak)         : {tokens_per_step / min(times):.1f} tok/s")
PYEOF

        echo ""
        echo "--- Loss trend ---"
        grep -i "'loss'" "${LOG_DIR}/phase4/train.log" 2>/dev/null | tail -10 || \
            grep -i '"loss"' "${LOG_DIR}/phase4/train.log" 2>/dev/null | tail -10 || \
            echo "  (no loss records)"

        echo ""
        echo "--- NaN/Inf stats ---"
        local nc ic
        nc=$(grep -ci " nan" "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        ic=$(grep -ci " inf" "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "  NaN lines: ${nc};  Inf lines: ${ic}"
        if [[ "${nc}" -gt 0 ]] || [[ "${ic}" -gt 0 ]]; then
            warn "Numerical anomaly detected — inspect train.log"
        fi

        echo ""
        echo "--- CPU backward speed (P2 diagnosis) ---"
        echo "  avg_sec > 120s  => backward_base_weight_grad bottleneck"
        echo "  avg_sec > 300s  => DDP timeout risk (BF16 backward is slower than INT8)"

        echo ""
        echo "--- update_base_weights call count ---"
        local uc
        uc=$(grep -ci "update_base_weights\|re-quantize\|TP_MOE_SFT" \
             "${LOG_DIR}/phase4/train.log" 2>/dev/null || echo 0)
        echo "  update_base_weights triggered: ${uc} times"

        echo ""
        echo "--- MoE aux/balance loss ---"
        grep -i "aux_loss\|router_loss\|balance_loss" \
             "${LOG_DIR}/phase4/train.log" 2>/dev/null | tail -5 || \
             echo "  (no aux_loss found)"

    } | tee "${LOG_DIR}/phase4_analysis.txt"

    analyze_log "phase4"
    echo "${exit_code}" > "${LOG_DIR}/phase4/exit_code.txt"
    cleanup_model_output "phase4"
}

# --------------------------------------------------------------------------- #
# Post-training: parse monitor.csv, report peak VRAM/RAM by training phase
# --------------------------------------------------------------------------- #
analyze_monitor_memory() {
    local csv_file="${LOG_DIR}/monitor.csv"
    [[ ! -f "${csv_file}" ]] && return

    phase_banner "Real-Time Memory Peak Analysis (from monitor.csv)"

    "${PYTHON}" - "${csv_file}" "${NUM_GPUS}" <<'PYEOF'
import csv, sys

csv_path = sys.argv[1]
num_gpus = int(sys.argv[2])

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("  (no monitor data)")
    sys.exit(0)

phase_stats = {}
for row in rows:
    ph = row.get("phase", "init")
    if ph not in phase_stats:
        phase_stats[ph] = {
            "gpu_peak_mb": [0] * num_gpus,
            "ram_peak_gb": 0.0,
            "ram_avail_min": 9999.0,
        }
    ps = phase_stats[ph]
    for i in range(num_gpus):
        key = f"gpu{i}_mem_used_mb"
        if key in row:
            try:
                v = int(float(row[key]))
                if v > ps["gpu_peak_mb"][i]:
                    ps["gpu_peak_mb"][i] = v
            except ValueError:
                pass
    try:
        ru = float(row.get("ram_used_gb", 0))
        ra = float(row.get("ram_avail_gb", 9999))
        if ru > ps["ram_peak_gb"]:    ps["ram_peak_gb"] = ru
        if ra < ps["ram_avail_min"]:  ps["ram_avail_min"] = ra
    except ValueError:
        pass

gpu_hdr = "  ".join(f"GPU{i}_peak" for i in range(num_gpus))
print(f"  {'Phase':<18}  {'RAM peak':>10}  {'RAM avail':>10}  {gpu_hdr}")
print(f"  {'─'*80}")
for ph, ps in sorted(phase_stats.items()):
    gpu_cols = "  ".join(f"{ps['gpu_peak_mb'][i]/1024:>8.1f}G" for i in range(num_gpus))
    avail = f"{ps['ram_avail_min']:.1f}G" if ps['ram_avail_min'] < 9999 else "N/A"
    print(f"  {ph:<18}  {ps['ram_peak_gb']:>8.1f} GB  {avail:>9}  {gpu_cols}")

all_gpu = [0] * num_gpus
all_ram = 0.0
for ps in phase_stats.values():
    for i in range(num_gpus):
        if ps["gpu_peak_mb"][i] > all_gpu[i]: all_gpu[i] = ps["gpu_peak_mb"][i]
    if ps["ram_peak_gb"] > all_ram: all_ram = ps["ram_peak_gb"]

print()
print("  Overall peaks:")
for i in range(num_gpus):
    print(f"    GPU{i}: {all_gpu[i]/1024:.1f} GB VRAM")
print(f"    CPU RAM: {all_ram:.1f} GB")
PYEOF
}

# --------------------------------------------------------------------------- #
# Summary report
# --------------------------------------------------------------------------- #
generate_summary() {
    phase_banner "Generating Summary Report"
    local summary="${LOG_DIR}/SUMMARY.md"
    {
        echo "# Qwen3-30B-A3B FFT Test — AMXBF16 4-GPU"
        echo ""
        echo "**Run time**   : ${RUN_TIME}"
        echo "**Run label**  : ${RUN_LABEL}"
        echo "**Log dir**    : ${LOG_DIR}"
        echo "**Config**     : $(basename ${ACCEL_CONFIG})"
        echo "**GPUs**       : ${NUM_GPUS}× RTX 4090 (FSDP-v2)"
        echo "**Backend**    : AMXBF16 (native BF16, no quantization)"
        echo "**Model**      : ${MODEL_PATH}"
        echo "**cutoff_len** : ${CUTOFF_LEN} tokens (all fft_stress_100 samples truncated)"
        echo "**Dataset**    : fft_stress_100 (100 samples)"
        echo ""
        if [[ -f "${LOG_DIR}/unsupported.txt" ]]; then
            echo "## Capability Result"
            echo ""
            sed 's/^/> /' "${LOG_DIR}/unsupported.txt"
            echo ""
        fi
        echo "## Phase Results"
        echo ""
        echo "| Phase | Exit Code | Status |"
        echo "|-------|-----------|--------|"
        for ph in phase1 phase4; do
            local ec_file="${LOG_DIR}/${ph}/exit_code.txt"
            if [[ -f "${LOG_DIR}/unsupported.txt" ]]; then
                echo "| ${ph} | - | Unsupported |"
            elif [[ -f "${ec_file}" ]]; then
                local ec
                ec=$(cat "${ec_file}")
                local status="Pass"
                [[ "${ec}" -ne 0 ]] && status="Fail(${ec})"
                echo "| ${ph} | ${ec} | ${status} |"
            else
                echo "| ${ph} | - | Skipped |"
            fi
        done
        echo ""
        echo "## TPS (AMXBF16, 4 GPUs, cutoff_len=${CUTOFF_LEN})"
        echo ""
        echo "\`\`\`"
        if [[ -f "${LOG_DIR}/phase4_analysis.txt" ]]; then
            grep -E "TPS|tokens_per_step|step_time|stable_steps" \
                "${LOG_DIR}/phase4_analysis.txt" 2>/dev/null | sed 's/^/  /' \
                || echo "  (run Phase 4 to get TPS data)"
        else
            echo "  (Phase 4 not run)"
        fi
        echo "\`\`\`"
        echo ""
        echo "## Next Steps"
        echo ""
        echo "\`\`\`bash"
        echo "python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
        echo "\`\`\`"
    } > "${summary}"

    log "Summary: ${summary}"
    cat "${summary}"
}

# --------------------------------------------------------------------------- #
# Cleanup on exit
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
    echo -e "${BOLD}Qwen3-30B-A3B KTransformers FFT — AMXBF16 4-GPU${NC}"
    echo -e "Time      : $(run_time_display)"
    echo -e "Run label : ${RUN_LABEL}"
    echo -e "GPUs      : ${NUM_GPUS}  |  cutoff_len: ${CUTOFF_LEN}  |  warmup_skip: ${WARMUP_SKIP}"
    echo -e "Backend   : AMXBF16 native BF16  |  ${MODEL_PATH}"
    echo ""

    check_env
    estimate_resources

    # Phase 0: pre-flight analysis — always runs
    analyze_weight_memory
    analyze_dataset_truncation

    setup_run_dir

    # Save pre-flight analysis to log dir
    analyze_weight_memory    > "${LOG_DIR}/weight_memory_analysis.txt" 2>&1
    analyze_dataset_truncation > "${LOG_DIR}/dataset_truncation.txt"   2>&1

    if record_unsupported_capability; then
        generate_summary
        ok "Skipped unsupported capability combination — log dir: ${LOG_DIR}"
        return 0
    fi

    start_monitor

    local overall_exit=0

    if [[ "${SKIP_PHASE1}" -eq 0 ]]; then
        if ! run_phase1; then
            warn "Phase 1 failed — continuing to Phase 4 to gather more data"
            overall_exit=1
        fi
    else
        log "Phase 1 skipped"
    fi

    if [[ "${SKIP_PHASE4}" -eq 0 ]]; then
        run_phase4 || overall_exit=1
    else
        log "Phase 4 skipped"
    fi

    stop_monitor

    analyze_monitor_memory

    generate_summary

    if [[ -f "${ANALYZE_SCRIPT}" ]]; then
        log "Running visualization..."
        "${PYTHON}" "${ANALYZE_SCRIPT}" --log-dir "${LOG_DIR}" \
            >> "${LOG_DIR}/analyze.log" 2>&1 && \
            ok "Charts: ${LOG_DIR}/plots/" || \
            warn "analyze.py failed; run manually: python3 ${ANALYZE_SCRIPT} --log-dir ${LOG_DIR}"
    fi

    echo ""
    if [[ "${overall_exit}" -eq 0 ]]; then
        ok "Done — log dir: ${LOG_DIR}"
    else
        warn "Some phases had issues — check: ${LOG_DIR}/SUMMARY.md"
    fi

    return "${overall_exit}"
}

main "$@"

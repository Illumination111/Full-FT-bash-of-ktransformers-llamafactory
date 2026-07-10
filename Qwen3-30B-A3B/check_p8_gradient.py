#!/usr/bin/env python3
"""
check_p8_gradient.py — Numerical gradient check for P8 bug.

Bug description:
  cache.m_local_pos_cache is indexed by token ID, but the C++ backward
  implementation accesses it with expert_idx.  When multiple tokens are
  routed to the same expert (expert_idx > 0), input_row and grad_out_row
  are fetched from the WRONG position in the global activation buffer.
  This produces silently incorrect gradients without any crash.

Detection strategy (finite-difference gradient check):
  1. Load the model with KTransformers (AMXBF16 full-FFT mode).
  2. Run one forward pass on a fixed mini-batch; compute scalar loss.
  3. Record the analytical gradient dL/dW for a target expert weight W
     via autograd.
  4. For each element W_ij in a small slice of W:
       perturb W_ij by +eps, recompute loss → L_plus
       restore W_ij, perturb by -eps → L_minus
       numerical gradient ≈ (L_plus - L_minus) / (2*eps)
  5. Compute cosine similarity between the analytical gradient vector
     and the numerical gradient vector over the sampled slice.

Expected result (correct implementation): cosine_sim > 0.99
P8 bug present:                           cosine_sim < 0.90

Usage:
  # Basic (uses first sample from the dataset):
  python3 check_p8_gradient.py \\
      --model-path  /mnt/data3/models/Qwen3-30B-A3B \\
      --dataset-file ./data/fft_stress_100.json \\
      --output-file  /tmp/p8_gradcheck.txt

  # With a specific checkpoint (after some training steps):
  python3 check_p8_gradient.py \\
      --model-path  /mnt/data3/models/Qwen3-30B-A3B \\
      --checkpoint-dir ./test_log/LATEST/phase5_p8/model_output \\
      --dataset-file ./data/fft_stress_100.json

Notes:
  - Requires the KTransformers environment (conda activate Kllama).
  - Needs USE_KT=1 in environment or KTransformers installed as a package.
  - The check targets layer 0, expert 3 gate_proj weight (first 4×4 slice)
    because expert 3 is consistently heavily loaded in Qwen3-30B-A3B.
  - Approximate runtime: 3–5 minutes (model loading ~2 min, 8 FD probes).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="P8 numerical gradient check")
    p.add_argument("--model-path", required=True,
                   help="Path to base HuggingFace model (BF16 safetensors)")
    p.add_argument("--kt-weight-path", default=None,
                   help="KTransformers weight path (defaults to --model-path)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Optional fine-tuned checkpoint dir (overrides --model-path for weights)")
    p.add_argument("--dataset-file", required=True,
                   help="Path to fft_stress_100.json (Alpaca format)")
    p.add_argument("--max-length", type=int, default=128,
                   help="Tokenise to this length (short = fast; bug still manifests)")
    p.add_argument("--layer-idx", type=int, default=0,
                   help="MoE layer index to check")
    p.add_argument("--expert-idx", type=int, default=3,
                   help="Expert index within that layer")
    p.add_argument("--probe-rows", type=int, default=4,
                   help="Number of weight rows to probe (finite-diff cost = 2 × rows)")
    p.add_argument("--probe-cols", type=int, default=4,
                   help="Number of weight columns to probe")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="Finite difference epsilon (1e-3 is safe for BF16)")
    p.add_argument("--output-file", default=None,
                   help="Write results to this file (stdout otherwise)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, kt_weight_path: str,
                              checkpoint_dir: str | None):
    """Load the model via transformers + KTransformers plugin if available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    os.environ.setdefault("USE_KT", "1")
    if kt_weight_path:
        os.environ.setdefault("KT_WEIGHT_PATH", kt_weight_path)

    load_path = checkpoint_dir if checkpoint_dir else model_path

    print(f"[check_p8] Loading tokenizer from {model_path} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[check_p8] Loading model from {load_path} …", flush=True)
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[check_p8] Model loaded in {time.time() - t0:.1f}s", flush=True)
    model.train()
    return model, tokenizer

# ---------------------------------------------------------------------------
# Prepare input batch
# ---------------------------------------------------------------------------

def make_batch(tokenizer, dataset_file: str, max_length: int):
    """Tokenise the first sample from the Alpaca dataset."""
    with open(dataset_file) as f:
        data = json.load(f)

    sample = data[0]
    text = (
        f"### Instruction:\n{sample.get('instruction', '')}\n\n"
        f"### Input:\n{sample.get('input', '')}\n\n"
        f"### Response:\n{sample.get('output', '')}"
    )
    enc = tokenizer(text, return_tensors="pt", max_length=max_length,
                    truncation=True, padding=False)
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    # Move to the device where the model's embedding lives
    device = next(iter(model.parameters())).device if False else "cpu"
    return input_ids, labels

# ---------------------------------------------------------------------------
# Core gradient check
# ---------------------------------------------------------------------------

def gradcheck_expert_weight(model, input_ids, labels,
                             layer_idx: int, expert_idx: int,
                             probe_rows: int, probe_cols: int,
                             eps: float):
    """
    Compare autograd gradient vs finite-difference gradient for the gate_proj
    weight of a specific expert.

    Returns:
        cosine_sim  : float  ∈ [-1, 1]
        rel_error   : float  relative L2 error
        analytical  : list[float]  flattened analytical grad slice
        numerical   : list[float]  flattened numerical grad slice
    """
    # Resolve target parameter
    # Standard LLaMA-Factory / KTransformers naming:
    #   model.model.layers[layer_idx].mlp.experts[expert_idx].gate_proj.weight
    try:
        target_param = (
            model.model.layers[layer_idx]
                 .mlp.experts[expert_idx]
                 .gate_proj.weight
        )
    except AttributeError:
        # Some architectures use a different path
        raise RuntimeError(
            f"Cannot find gate_proj at layers[{layer_idx}].mlp.experts[{expert_idx}]; "
            "adjust --layer-idx / --expert-idx"
        )

    device = target_param.device
    input_ids_dev = input_ids.to(device)
    labels_dev    = labels.to(device)

    # ---- Analytical gradient via autograd ----
    model.zero_grad()
    out = model(input_ids=input_ids_dev, labels=labels_dev)
    loss = out.loss
    loss.backward()

    if target_param.grad is None:
        raise RuntimeError(
            f"No gradient for layers[{layer_idx}].mlp.experts[{expert_idx}].gate_proj; "
            "make sure expert is actually activated by the router for this batch"
        )

    # Slice: first [probe_rows × probe_cols] elements of the weight matrix
    grad_analytical = (
        target_param.grad[:probe_rows, :probe_cols].detach().float().cpu()
    )
    loss_base = loss.item()
    print(f"  base loss       = {loss_base:.6f}", flush=True)
    print(f"  analytical grad slice (first {probe_rows}×{probe_cols}):", flush=True)
    print(f"    {grad_analytical.tolist()}", flush=True)

    # ---- Numerical gradient via central finite differences ----
    model.zero_grad()
    W = target_param.data  # reference to live parameter tensor
    grad_numerical = torch.zeros(probe_rows, probe_cols, dtype=torch.float32)

    for i in range(probe_rows):
        for j in range(probe_cols):
            orig = W[i, j].item()

            # L_plus
            W[i, j] = orig + eps
            with torch.no_grad():
                out_plus = model(input_ids=input_ids_dev, labels=labels_dev)
            l_plus = out_plus.loss.item()

            # L_minus
            W[i, j] = orig - eps
            with torch.no_grad():
                out_minus = model(input_ids=input_ids_dev, labels=labels_dev)
            l_minus = out_minus.loss.item()

            W[i, j] = orig  # restore
            grad_numerical[i, j] = (l_plus - l_minus) / (2.0 * eps)
            print(f"  probe [{i},{j}]: L+={l_plus:.6f}  L-={l_minus:.6f} "
                  f"  num_grad={grad_numerical[i,j]:.4e}", flush=True)

    # ---- Compare ----
    a_flat = grad_analytical.flatten()
    n_flat = grad_numerical.flatten()

    cos_sim = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), n_flat.unsqueeze(0)
    ).item()

    rel_error = (a_flat - n_flat).norm() / (n_flat.norm() + 1e-12)

    return cos_sim, rel_error.item(), a_flat.tolist(), n_flat.tolist()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_lines = []

    def emit(msg: str = ""):
        print(msg, flush=True)
        out_lines.append(msg)

    emit("=" * 60)
    emit("  P8 Numerical Gradient Check")
    emit("  Bug: cache.m_local_pos_cache indexed by expert_idx")
    emit("       instead of original token ID in C++ backward")
    emit("=" * 60)
    emit()
    emit(f"  model path      : {args.model_path}")
    emit(f"  checkpoint dir  : {args.checkpoint_dir or '(none — using base model)'}")
    emit(f"  target          : layer {args.layer_idx}, expert {args.expert_idx}, gate_proj")
    emit(f"  probe slice     : {args.probe_rows}×{args.probe_cols} elements")
    emit(f"  eps             : {args.eps}")
    emit(f"  max_length      : {args.max_length} tokens")
    emit()

    kt_weight_path = args.kt_weight_path or args.model_path

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, kt_weight_path, args.checkpoint_dir
        )
    except Exception as e:
        emit(f"ERROR loading model: {e}")
        sys.exit(2)

    try:
        input_ids, labels = make_batch(tokenizer, args.dataset_file, args.max_length)
    except Exception as e:
        emit(f"ERROR preparing batch: {e}")
        sys.exit(2)

    emit(f"  input_ids shape : {list(input_ids.shape)}")
    emit(f"  token count     : {input_ids.numel()}")
    emit()

    try:
        cos_sim, rel_err, analytical, numerical = gradcheck_expert_weight(
            model, input_ids, labels,
            layer_idx=args.layer_idx,
            expert_idx=args.expert_idx,
            probe_rows=args.probe_rows,
            probe_cols=args.probe_cols,
            eps=args.eps,
        )
    except RuntimeError as e:
        emit(f"ERROR during gradient check: {e}")
        emit()
        emit("Possible reasons:")
        emit("  1. Router did not activate this expert for the given batch.")
        emit("     Try a different --expert-idx or increase --max-length.")
        emit("  2. KTransformers C++ extension not loaded (USE_KT=0).")
        sys.exit(2)

    emit()
    emit("=" * 60)
    emit("  RESULTS")
    emit("=" * 60)
    emit(f"  cosine_similarity : {cos_sim:.6f}")
    emit(f"  relative_L2_error : {rel_err:.6f}")
    emit()

    if cos_sim > 0.95:
        verdict = "PASS — gradients match finite differences (P8 NOT triggered)"
        emit(f"  ✓ {verdict}")
    elif cos_sim > 0.80:
        verdict = ("WARN — partial mismatch (cosine_sim ∈ (0.80, 0.95]); "
                   "possible P8 or floating-point noise from large eps")
        emit(f"  ⚠ {verdict}")
        emit("    Recommendation: re-run with --eps 1e-4 or increase --max-length")
    else:
        verdict = (f"FAIL — cosine_sim={cos_sim:.4f} << 0.95: "
                   "P8 cache indexing bug CONFIRMED")
        emit(f"  ✗ {verdict}")
        emit()
        emit("  Root cause: m_local_pos_cache[expert_idx] reads from the wrong")
        emit("  token's activation.  Fix: use original_token_id as the index.")
        emit()
        emit("  analytical (autograd)  : " + str([f"{v:.3e}" for v in analytical]))
        emit("  numerical  (fd approx) : " + str([f"{v:.3e}" for v in numerical]))

    emit()
    emit("=" * 60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write("\n".join(out_lines))
        print(f"[check_p8] Results written to {args.output_file}", flush=True)

    sys.exit(0 if cos_sim > 0.95 else 1)


if __name__ == "__main__":
    main()

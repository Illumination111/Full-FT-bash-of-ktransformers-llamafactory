#!/usr/bin/env python3
"""Generate deterministic skewed route traces for APTMoE smoke tests only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aptmoe_proxy.storage import (
    DEFAULT_SIMULATION_ROOT,
    require_within_simulation_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=DEFAULT_SIMULATION_ROOT,
    )
    args = parser.parse_args()
    if min(
        args.sequence_length,
        args.global_batch_size,
        args.layers,
        args.experts,
        args.top_k,
    ) <= 0:
        raise SystemExit("shape arguments must be positive")
    if args.top_k > args.experts:
        raise SystemExit("top-k cannot exceed expert count")
    if args.zipf_exponent < 0:
        raise SystemExit("zipf exponent must be non-negative")

    output = require_within_simulation_root(
        args.output,
        args.simulation_root,
    )
    tokens = args.sequence_length * args.global_batch_size
    rng = np.random.default_rng(args.seed)
    expert_rank = np.arange(1, args.experts + 1, dtype=np.float64)
    base = expert_rank ** (-args.zipf_exponent)
    base /= base.sum()
    routes = np.empty(
        (args.layers, tokens, args.top_k),
        dtype=np.int16,
    )
    for layer_idx in range(args.layers):
        permutation = rng.permutation(args.experts)
        probabilities = np.empty_like(base)
        probabilities[permutation] = base
        for token_idx in range(tokens):
            routes[layer_idx, token_idx] = rng.choice(
                args.experts,
                size=args.top_k,
                replace=False,
                p=probabilities,
            )

    metadata = {
        "schema_version": 1,
        "source": "synthetic_zipf_smoke_only",
        "valid_for_qwen35_tps_estimate": False,
        "sequence_length": args.sequence_length,
        "global_batch_size": args.global_batch_size,
        "layers": args.layers,
        "tokens": tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "zipf_exponent": args.zipf_exponent,
        "seed": args.seed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        topk_indices=routes,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"[synthetic_routes] {routes.shape} -> {output}")


if __name__ == "__main__":
    main()

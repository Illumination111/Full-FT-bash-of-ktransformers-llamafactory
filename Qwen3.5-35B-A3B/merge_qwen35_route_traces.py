#!/usr/bin/env python3
"""Merge per-rank exact Qwen3.5 routes into one APTMoE replay pattern."""

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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-patterns", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=DEFAULT_SIMULATION_ROOT,
    )
    args = parser.parse_args()
    if min(
        args.expected_ranks,
        args.expected_patterns,
        args.sequence_length,
        args.global_batch_size,
    ) <= 0:
        raise SystemExit("rank, pattern, sequence, and batch sizes must be positive")

    output = require_within_simulation_root(
        args.output,
        args.simulation_root,
    )
    rank_files = [
        args.input_dir / f"rank_{rank:02d}.npz"
        for rank in range(args.expected_ranks)
    ]
    missing = [str(path) for path in rank_files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing route rank files: {missing}")

    arrays: list[np.ndarray] = []
    source_metadata: list[dict] = []
    source_backend: str | None = None
    expected_tokens_per_rank = (
        args.sequence_length * args.global_batch_size // args.expected_ranks
    )
    if args.global_batch_size % args.expected_ranks != 0:
        raise SystemExit("global batch must be divisible by expected ranks")
    for expected_rank, path in enumerate(rank_files):
        with np.load(path, allow_pickle=False) as data:
            array = np.asarray(data["topk_indices"])
            raw_metadata = data["metadata_json"].item()
        if (
            array.ndim != 4
            or array.shape[0] != args.expected_patterns
            or array.shape[1] != 40
            or array.shape[3] != 8
        ):
            raise SystemExit(f"invalid route shape in {path}: {array.shape}")
        metadata = json.loads(str(raw_metadata))
        expected_metadata = {
            "source": "exact_qwen35_router_forward_hook",
            "rank": expected_rank,
            "world_size": args.expected_ranks,
            "sequence_length": args.sequence_length,
            "patterns": args.expected_patterns,
            "layers": 40,
            "tokens_on_rank": expected_tokens_per_rank,
            "top_k": 8,
        }
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise SystemExit(
                    f"{path} metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        backend = metadata.get("backend")
        if backend not in {"kt", "ktransformers", "deepspeed"}:
            raise SystemExit(
                f"{path} metadata backend={backend!r} is not an exact backend"
            )
        if source_backend is None:
            source_backend = backend
        elif backend != source_backend:
            raise SystemExit(
                f"{path} metadata backend={backend!r}, "
                f"expected {source_backend!r}"
            )
        arrays.append(array.astype(np.int16, copy=False))
        source_metadata.append(metadata)

    merged = np.concatenate(arrays, axis=2)
    expected_tokens = args.sequence_length * args.global_batch_size
    if merged.shape != (args.expected_patterns, 40, expected_tokens, 8):
        raise SystemExit(
            f"merged route shape={merged.shape}, "
            f"expected={(args.expected_patterns, 40, expected_tokens, 8)}"
        )
    metadata = {
        "schema_version": 1,
        "source": "merged_exact_qwen35_router_trace",
        "source_backend": source_backend,
        "sequence_length": args.sequence_length,
        "global_batch_size": args.global_batch_size,
        "patterns": args.expected_patterns,
        "layers": 40,
        "tokens": expected_tokens,
        "top_k": 8,
        "rank_files": [str(path.resolve()) for path in rank_files],
        "source_metadata": source_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        topk_indices=merged,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"[merge_routes] {merged.shape} -> {output}")


if __name__ == "__main__":
    main()

"""Replay or synthesize Qwen3.5 top-k routes without changing router FLOPs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RouteController:
    """Serve deterministic per-layer top-k indices to the proxy router.

    A replay file is an NPZ containing ``topk_indices`` with one of these
    shapes:

    - ``[layers, tokens, top_k]``: one legacy/smoke pattern, repeated;
    - ``[patterns, layers, tokens, top_k]``: formal warmup patterns replayed
      across successive accumulation microbatches.

    The router still computes logits, softmax, and top-k. Replayed indices only
    replace the dispatch choice, and selected weights are gathered from the
    differentiable router probabilities.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int,
        sequence_length: int,
        tokens_per_microbatch: int,
        microbatches_per_step: int = 1,
        expected_patterns: int | None = None,
        trace_path: str | Path | None,
        allow_synthetic: bool,
    ) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.sequence_length = sequence_length
        self.tokens_per_microbatch = tokens_per_microbatch
        if microbatches_per_step <= 0:
            raise ValueError("microbatches_per_step must be positive")
        self.microbatches_per_step = microbatches_per_step
        if expected_patterns is not None and expected_patterns <= 0:
            raise ValueError("expected_patterns must be positive")
        self.expected_patterns = expected_patterns
        self.step = 0
        self.microbatch = 0
        self._cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._last_counts: dict[int, list[int]] = {}
        self.trace_path = self._resolve_trace_path(trace_path)
        self.trace: np.ndarray | None = None
        self._trace_counts: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.trace_sha256: str | None = None

        if self.trace_path is not None:
            self._load_trace(self.trace_path)
            source = self.metadata.get("source")
            if source == "merged_exact_qwen35_router_trace":
                assert self.trace is not None
                if self.trace.ndim != 4:
                    raise ValueError(
                        "formal Qwen3.5 route traces must use the "
                        "[patterns, layers, tokens, top_k] schema"
                    )
                if (
                    self.expected_patterns is not None
                    and self.trace.shape[0] != self.expected_patterns
                ):
                    raise ValueError(
                        f"formal route patterns={self.trace.shape[0]}, "
                        f"expected warmup_steps*GAS={self.expected_patterns}"
                    )
                if self.metadata.get("source_backend") not in {
                    "kt",
                    "ktransformers",
                    "deepspeed",
                }:
                    raise ValueError(
                        "formal Qwen3.5 route trace is missing an exact "
                        "source_backend"
                    )
                self.mode = "replayed_qwen35_topk_indices"
            elif allow_synthetic:
                self.mode = "synthetic_trace_smoke_only"
            else:
                raise ValueError(
                    "formal APTMoE proxy runs only accept a trace produced by "
                    "merge_qwen35_route_traces.py from exact Qwen3.5 router "
                    f"hooks; got metadata source={source!r}"
                )
        elif allow_synthetic:
            self.mode = "random_router_synthetic"
        else:
            raise ValueError(
                "formal APTMoE proxy runs require a Qwen3.5 route trace; "
                "use --allow-synthetic-routing only for smoke tests"
            )

    def _resolve_trace_path(self, trace_path: str | Path | None) -> Path | None:
        if trace_path is None:
            return None
        candidate = Path(trace_path).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / f"seq_{self.sequence_length}.npz"
        if not candidate.is_file():
            raise FileNotFoundError(f"route trace was not found: {candidate}")
        return candidate

    def _load_trace(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            if "topk_indices" not in data:
                raise ValueError("route trace is missing topk_indices")
            trace = np.asarray(data["topk_indices"])
            raw_metadata = (
                data["metadata_json"]
                if "metadata_json" in data.files
                else None
            )
            if raw_metadata is not None:
                self.metadata = json.loads(str(raw_metadata.item()))
        if not isinstance(self.metadata, dict):
            raise ValueError("route trace metadata_json must contain an object")

        if trace.ndim == 3:
            expected = (
                self.num_layers,
                self.tokens_per_microbatch,
                self.top_k,
            )
            if trace.shape != expected:
                raise ValueError(
                    f"route trace shape={trace.shape}, expected {expected}"
                )
        elif trace.ndim == 4:
            expected_tail = (
                self.num_layers,
                self.tokens_per_microbatch,
                self.top_k,
            )
            if trace.shape[1:] != expected_tail:
                raise ValueError(
                    f"route trace shape={trace.shape}, expected [steps, {expected_tail}]"
                )
        else:
            raise ValueError("topk_indices must have rank 3 or 4")

        if trace.size == 0:
            raise ValueError("route trace is empty")
        if trace.min() < 0 or trace.max() >= self.num_experts:
            raise ValueError("route trace contains an out-of-range expert id")
        sorted_indices = np.sort(trace, axis=-1)
        if np.any(np.diff(sorted_indices, axis=-1) == 0):
            raise ValueError("route trace selects the same expert twice for one token")
        metadata_checks = {
            "schema_version": 1,
            "sequence_length": self.sequence_length,
            "layers": self.num_layers,
            "tokens": self.tokens_per_microbatch,
            "top_k": self.top_k,
        }
        if self.tokens_per_microbatch % self.sequence_length != 0:
            raise ValueError(
                "route tokens must be divisible by the sequence length"
            )
        metadata_checks["global_batch_size"] = (
            self.tokens_per_microbatch // self.sequence_length
        )
        if trace.ndim == 4:
            metadata_checks["patterns"] = trace.shape[0]
        for key, expected_value in metadata_checks.items():
            value = self.metadata.get(key)
            if value != expected_value:
                raise ValueError(
                    f"route trace metadata {key}={value!r}, "
                    f"expected {expected_value!r}"
                )

        self.trace = trace.astype(np.int16, copy=False)
        patterns = self.trace[None, ...] if self.trace.ndim == 3 else self.trace
        self._trace_counts = np.empty(
            (patterns.shape[0], self.num_layers, self.num_experts),
            dtype=np.int64,
        )
        for pattern_idx, pattern in enumerate(patterns):
            for layer_idx in range(self.num_layers):
                self._trace_counts[pattern_idx, layer_idx] = np.bincount(
                    pattern[layer_idx].reshape(-1),
                    minlength=self.num_experts,
                )
        self.trace_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def set_position(self, step: int, microbatch: int) -> None:
        self.step = step
        self.microbatch = microbatch

    def _pattern_index(self) -> int:
        if self.trace is None or self.trace.ndim == 3:
            return 0
        forward_index = (
            self.step * self.microbatches_per_step + self.microbatch
        )
        return forward_index % self.trace.shape[0]

    def _replayed_indices(
        self,
        layer_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert self.trace is not None
        pattern = self._pattern_index()
        key = (pattern, layer_idx, str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        source = (
            self.trace[layer_idx]
            if self.trace.ndim == 3
            else self.trace[pattern, layer_idx]
        )
        result = torch.as_tensor(
            source.astype(np.int64, copy=False),
            device=device,
            dtype=torch.long,
        )
        self._cache[key] = result
        return result

    def select(
        self,
        *,
        layer_idx: int,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        probabilities = torch.softmax(logits.float(), dim=-1)
        computed_scores, computed_indices = torch.topk(
            probabilities,
            self.top_k,
            dim=-1,
        )
        if self.trace is None:
            selected_indices = computed_indices
            selected_scores = computed_scores
        else:
            selected_indices = self._replayed_indices(layer_idx, logits.device)
            if selected_indices.shape[0] != logits.shape[0]:
                raise RuntimeError(
                    f"layer {layer_idx} route tokens={selected_indices.shape[0]}, "
                    f"runtime tokens={logits.shape[0]}"
                )
            selected_scores = probabilities.gather(1, selected_indices)

        selected_scores = selected_scores / selected_scores.sum(
            dim=-1,
            keepdim=True,
        )
        counts_tensor = torch.bincount(
            selected_indices.reshape(-1),
            minlength=self.num_experts,
        )
        counts = counts_tensor.to("cpu").tolist()
        self._last_counts[layer_idx] = counts
        return (
            selected_scores.to(logits.dtype),
            selected_indices,
            counts,
        )

    def predicted_counts(self, layer_idx: int) -> list[int]:
        next_layer = min(layer_idx + 1, self.num_layers - 1)
        return self.counts_for_layer(next_layer)

    def counts_for_layer(self, layer_idx: int) -> list[int]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer index out of range: {layer_idx}")
        if self.trace is not None:
            assert self._trace_counts is not None
            return self._trace_counts[
                self._pattern_index(),
                layer_idx,
            ].tolist()
        return list(
            self._last_counts.get(
                layer_idx,
                [0] * self.num_experts,
            )
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "trace_sha256": self.trace_sha256,
            "trace_metadata": self.metadata,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "sequence_length": self.sequence_length,
            "tokens_per_microbatch": self.tokens_per_microbatch,
            "microbatches_per_step": self.microbatches_per_step,
            "expected_patterns": self.expected_patterns,
        }

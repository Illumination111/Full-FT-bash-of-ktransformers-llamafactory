"""Capture warmup Qwen3.5 top-k route patterns for APTMoE replay.

The hook buffers selected expert IDs in RAM and writes only at process exit.
Capturing should be enabled only when those forward passes are excluded as
warmup, because every captured router output performs a GPU-to-CPU copy.
"""

from __future__ import annotations

import atexit
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aptmoe_proxy.storage import (
    DEFAULT_SIMULATION_ROOT,
    require_within_simulation_root,
)


LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class RouteTraceCapture:
    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_length: int,
        max_patterns: int,
        expected_layers: int = 40,
        top_k: int = 8,
    ) -> None:
        simulation_root = Path(
            os.environ.get(
                "FFT_APTMOE_SIMULATION_ROOT",
                str(DEFAULT_SIMULATION_ROOT),
            )
        )
        self.output_dir = require_within_simulation_root(
            output_dir,
            simulation_root,
        )
        self.sequence_length = sequence_length
        if max_patterns <= 0:
            raise ValueError("route capture max_patterns must be positive")
        self.max_patterns = max_patterns
        self.expected_layers = expected_layers
        self.top_k = top_k
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.backend = os.environ.get("FFT_TRAINING_BACKEND", "unknown")
        self.patterns: list[dict[int, np.ndarray]] = []
        self.current_pattern: dict[int, np.ndarray] | None = None
        self._hook_handles: list[Any] = []
        self._written = False

    def set_hook_handles(self, handles: list[Any]) -> None:
        self._hook_handles = handles

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def begin_model_forward(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:
        del module, inputs
        if len(self.patterns) >= self.max_patterns:
            return
        if self.current_pattern is not None:
            raise RuntimeError(
                "nested Qwen3.5 top-level forward during route capture"
            )
        self.current_pattern = {}

    def end_model_forward(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs, output
        if self.current_pattern is None:
            return
        missing = sorted(
            set(range(self.expected_layers)) - self.current_pattern.keys()
        )
        if missing:
            raise RuntimeError(
                "Qwen3.5 route capture forward is incomplete; "
                f"missing layers={missing}"
            )
        self.patterns.append(self.current_pattern)
        self.current_pattern = None
        if len(self.patterns) == self.max_patterns:
            self.remove_hooks()

    def hook(self, layer_idx: int):  # type: ignore[no-untyped-def]
        def capture(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            del module, inputs
            current = self.current_pattern
            if current is None or layer_idx in current:
                return
            if not isinstance(output, tuple) or len(output) < 3:
                raise RuntimeError(
                    "Qwen3.5 router output does not contain selected experts"
                )
            selected = output[2]
            if not isinstance(selected, torch.Tensor):
                raise RuntimeError("Qwen3.5 selected experts are not a tensor")
            if selected.ndim != 2 or selected.shape[1] != self.top_k:
                raise RuntimeError(
                    f"layer {layer_idx} route shape={tuple(selected.shape)}, "
                    f"expected [tokens, {self.top_k}]"
                )
            current[layer_idx] = (
                selected.detach()
                .to(device="cpu", dtype=torch.int16)
                .numpy()
                .copy()
            )

        return capture

    def write(self) -> None:
        if self._written:
            return
        self._written = True
        if len(self.patterns) != self.max_patterns:
            print(
                f"[qwen35_route_capture] rank={self.rank} incomplete; "
                f"patterns={len(self.patterns)}/{self.max_patterns}",
                flush=True,
            )
            return
        pattern_arrays = [
            np.stack(
                [pattern[index] for index in range(self.expected_layers)],
                axis=0,
            )
            for pattern in self.patterns
        ]
        arrays = [
            array
            for pattern in pattern_arrays
            for array in pattern
        ]
        token_counts = {array.shape[0] for array in arrays}
        if len(token_counts) != 1:
            raise RuntimeError(
                f"route layers have inconsistent token counts: {token_counts}"
            )
        topk_indices = np.stack(pattern_arrays, axis=0)
        metadata = {
            "schema_version": 1,
            "source": "exact_qwen35_router_forward_hook",
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world_size,
            "sequence_length": self.sequence_length,
            "patterns": self.max_patterns,
            "layers": self.expected_layers,
            "tokens_on_rank": int(topk_indices.shape[2]),
            "top_k": self.top_k,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"rank_{self.rank:02d}.npz"
        np.savez_compressed(
            output,
            topk_indices=topk_indices,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        print(
            f"[qwen35_route_capture] rank={self.rank} wrote {output} "
            f"shape={topk_indices.shape}",
            flush=True,
        )


def install_route_capture(
    model: torch.nn.Module,
    output_dir: str | Path,
    sequence_length: int,
) -> RouteTraceCapture:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTopKRouter,
    )

    capture = RouteTraceCapture(
        output_dir=Path(output_dir),
        sequence_length=sequence_length,
        max_patterns=int(os.environ.get("FFT_ROUTE_TRACE_PATTERNS", "1")),
    )
    registered: set[int] = set()
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not isinstance(module, Qwen3_5MoeTopKRouter):
            continue
        match = LAYER_PATTERN.search(name)
        if match is None:
            raise RuntimeError(f"cannot extract layer id from router name {name!r}")
        layer_idx = int(match.group(1))
        if layer_idx in registered:
            raise RuntimeError(f"duplicate Qwen3.5 router for layer {layer_idx}")
        registered.add(layer_idx)
        handles.append(module.register_forward_hook(capture.hook(layer_idx)))
    if registered != set(range(capture.expected_layers)):
        raise RuntimeError(
            "Qwen3.5 route capture expected layers 0..39, "
            f"found {sorted(registered)}"
        )
    handles.append(model.register_forward_pre_hook(capture.begin_model_forward))
    handles.append(model.register_forward_hook(capture.end_model_forward))
    capture.set_hook_handles(handles)
    atexit.register(capture.write)
    print(
        f"[qwen35_route_capture] installed rank={capture.rank} "
        f"patterns={capture.max_patterns} output={capture.output_dir}; "
        "copies are warmup-only",
        flush=True,
    )
    return capture

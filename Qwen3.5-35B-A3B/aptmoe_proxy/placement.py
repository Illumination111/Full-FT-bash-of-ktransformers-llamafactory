"""APTMoE expert-placement solver backed by a host-specific lookup table."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_EXPERT_BF16_BYTES = 6 * (1 << 20)


def _positive_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"lookup field {key!r} must be a positive number")
    return float(value)


class ProxyPlacementSolver:
    """Choose hot GPU experts using the same compute/load comparison as APTMoE."""

    def __init__(
        self,
        num_experts: int,
        num_chunks: int,
        *,
        lookup_path: str | Path | None,
        prefetch_portion: float,
        allow_unprofiled: bool,
        expected_profile: str | None = None,
        required_max_tokens: int | None = None,
    ) -> None:
        if num_experts <= 0 or num_chunks <= 0:
            raise ValueError("num_experts and num_chunks must be positive")
        if required_max_tokens is not None and required_max_tokens <= 0:
            raise ValueError("required_max_tokens must be positive")
        if not 0.0 < prefetch_portion <= 1.0:
            raise ValueError("prefetch_portion must be in (0, 1]")
        self.num_experts = num_experts
        self.num_chunks = num_chunks
        self.prefetch_portion = prefetch_portion
        self.expected_profile = expected_profile
        self.required_max_tokens = required_max_tokens
        self.lookup_path = Path(lookup_path).resolve() if lookup_path else None
        self.lookup: dict[str, Any] | None = None
        self.lookup_sha256: str | None = None
        self.mode = "unprofiled_fraction"

        if self.lookup_path is not None:
            lookup_bytes = self.lookup_path.read_bytes()
            self.lookup_sha256 = hashlib.sha256(lookup_bytes).hexdigest()
            self.lookup = json.loads(lookup_bytes)
            self._validate_lookup(self.lookup)
            self.mode = "profiled_compute_load"
        elif not allow_unprofiled:
            raise ValueError(
                "a host-specific Qwen3.5 proxy lookup table is required; "
                "use --allow-unprofiled-placement only for smoke tests"
            )

    def _validate_lookup(self, lookup: dict[str, Any]) -> None:
        if lookup.get("schema_version") != 1:
            raise ValueError("lookup schema_version must be 1")
        if lookup.get("benchmark_class") != "aptmoe_qwen35_proxy_lookup":
            raise ValueError("lookup benchmark_class is not a Qwen3.5 proxy lookup")
        if (
            self.expected_profile is not None
            and lookup.get("deployment_profile") != self.expected_profile
        ):
            raise ValueError(
                "lookup deployment_profile="
                f"{lookup.get('deployment_profile')!r}, "
                f"expected {self.expected_profile!r}"
            )
        expert = lookup.get("expert") or {}
        if expert.get("bf16_bytes") != EXPECTED_EXPERT_BF16_BYTES:
            raise ValueError(
                "lookup expert size does not match the Qwen3.5 6 MiB BF16 expert"
            )
        if expert.get("num_experts") != self.num_experts:
            raise ValueError(
                f"lookup num_experts={expert.get('num_experts')!r}, "
                f"expected {self.num_experts}"
            )
        _positive_number(expert, "h2d_seconds")
        control = lookup.get("control_plane") or {}
        _positive_number(control, "load_seconds")
        _positive_number(control, "non_mixer_load_seconds")
        token_mixers = lookup.get("token_mixers") or {}
        for layer_type in ("linear_attention", "full_attention"):
            profile = token_mixers.get(layer_type) or {}
            _positive_number(profile, "h2d_seconds")
        extras = lookup.get("extra_modules") or {}
        for key in (
            "embedding_h2d_seconds",
            "final_norm_h2d_seconds",
            "lm_head_h2d_seconds",
        ):
            _positive_number(extras, key)
        cpu = lookup.get("cpu_expert") or {}
        points = cpu.get("forward_seconds_by_tokens")
        max_tokens = cpu.get("max_tokens")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
            or not isinstance(points, list)
            or len(points) != max_tokens + 1
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
                for value in points
            )
        ):
            raise ValueError(
                "cpu_expert.max_tokens must be positive and "
                "forward_seconds_by_tokens must be a matching "
                "non-negative numeric list"
            )
        if (
            self.required_max_tokens is not None
            and max_tokens < self.required_max_tokens
        ):
            raise ValueError(
                f"lookup CPU curve max_tokens={max_tokens}, "
                f"required at least {self.required_max_tokens}"
            )

    def _cpu_seconds(self, tokens: int) -> float:
        assert self.lookup is not None
        curve = self.lookup["cpu_expert"]["forward_seconds_by_tokens"]
        last_index = len(curve) - 1
        # Match APTMoE's R_solver exactly: when a count exceeds the profiled
        # table, first normalize by pipeline chunks and then clamp.  The proxy
        # currently uses one chunk, but keeping the rule here prevents a future
        # num_chunks change from silently altering placement.
        if tokens > last_index:
            tokens //= self.num_chunks
        return float(curve[min(tokens, last_index)])

    def solve(
        self,
        assigned_tokens_list: list[int],
        *,
        layer_type: str | None = None,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
    ) -> list[int]:
        if len(assigned_tokens_list) != self.num_experts:
            raise ValueError(
                f"expected {self.num_experts} expert counts, "
                f"got {len(assigned_tokens_list)}"
            )
        if any(value < 0 for value in assigned_tokens_list):
            raise ValueError("expert token counts must be non-negative")

        sorted_cold_to_hot = sorted(
            range(self.num_experts),
            key=lambda expert_id: assigned_tokens_list[expert_id],
        )
        if self.lookup is None:
            hot_count = max(
                1,
                math.ceil(self.num_experts * self.prefetch_portion),
            )
            return sorted_cold_to_hot[-hot_count:]

        if layer_type not in {"linear_attention", "full_attention"}:
            raise ValueError(
                "profiled placement requires layer_type=linear_attention "
                "or full_attention"
            )
        load_expert = float(self.lookup["expert"]["h2d_seconds"])
        load_seconds = float(
            self.lookup["control_plane"]["non_mixer_load_seconds"]
        ) + float(
            self.lookup["token_mixers"][layer_type]["h2d_seconds"]
        )
        extras = self.lookup["extra_modules"]
        if is_first_stage:
            load_seconds += float(extras["embedding_h2d_seconds"])
        if is_last_stage:
            load_seconds += float(extras["final_norm_h2d_seconds"])
            load_seconds += float(extras["lm_head_h2d_seconds"])
        cpu_seconds = 0.0
        cpu_experts: list[int] = []
        for expert_id in sorted_cold_to_hot:
            token_count = assigned_tokens_list[expert_id]
            cpu_seconds += self._cpu_seconds(token_count)
            load_seconds += load_expert
            if cpu_seconds / load_seconds < 1.0:
                cpu_experts.append(expert_id)
            else:
                break

        cpu_set = set(cpu_experts)
        gpu_experts = [
            expert_id
            for expert_id in sorted_cold_to_hot
            if expert_id not in cpu_set
        ]
        return gpu_experts

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lookup_path": str(self.lookup_path) if self.lookup_path else None,
            "lookup_sha256": self.lookup_sha256,
            "prefetch_portion": self.prefetch_portion,
            "num_experts": self.num_experts,
            "num_chunks": self.num_chunks,
            "deployment_profile": self.expected_profile,
            "required_max_tokens": self.required_max_tokens,
        }

"""Reference components for a Qwen3.5-shaped APTMoE deployment proxy.

The token mixers below reuse the real Transformers Qwen3.5 implementations, so
their trainable tensor shapes and operator family match the target.  Expert
weights are intentionally random: the proxy measures deployment behavior, not
model quality or checkpoint compatibility.

This module is small on purpose.  APTMoE should wrap these components with its
own pipeline/offload scheduler instead of teaching its checkpoint loader every
Qwen3.5 key format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
    is_fast_path_available,
)


def load_text_config(model_path: str | Path) -> Qwen3_5MoeTextConfig:
    raw = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") != "qwen3_5_moe":
        raise ValueError(f"not a Qwen3.5-MoE checkpoint: {model_path}")
    text = raw.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_moe_text":
        raise ValueError("Qwen3.5 checkpoint does not contain a text_config")
    config = Qwen3_5MoeTextConfig(**text)
    # SDPA preserves the target's quadratic full-attention workload without
    # materializing an eager SxS score tensor when the installed backend can
    # provide a fused implementation.
    config._attn_implementation = "sdpa"
    return config


def require_linear_attention_fastpath() -> None:
    """Reject the slow fallback before collecting GPU-attention timings."""
    if not is_fast_path_available:
        raise RuntimeError(
            "Qwen3.5 linear-attention fast path is unavailable. Install compatible "
            "flash-linear-attention and causal-conv1d builds, then record their "
            "versions in the proxy manifest. The fallback is valid only for "
            "parameter/memory audits, not GPU-attention performance."
        )


class Qwen35TokenMixer(nn.Module):
    """APTMoE-facing wrapper around one exact Qwen3.5 token mixer."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.on_GPU = False
        if self.layer_type == "linear_attention":
            self.impl = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
            self.rotary = None
        elif self.layer_type == "full_attention":
            self.impl = Qwen3_5MoeAttention(config, layer_idx)
            self.rotary = Qwen3_5MoeTextRotaryEmbedding(config)
        else:
            raise ValueError(f"unsupported layer type: {self.layer_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.impl(
                hidden_states=hidden_states,
                cache_params=None,
                attention_mask=attention_mask,
            )

        batch, sequence, _ = hidden_states.shape
        position_ids = (
            torch.arange(
                sequence,
                device=hidden_states.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(batch, -1)
        )
        position_embeddings = self.rotary(hidden_states, position_ids)
        output, _ = self.impl(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        return output


class Qwen35RoutedExpert(nn.Module):
    """One movable Qwen3.5 fused-gate/up expert (6 MiB in BF16)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        layer_id: int,
        expert_id: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            **factory_kwargs,
        )
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.on_GPU = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen35SharedExpert(nn.Module):
    """Always-active shared expert including Qwen3.5's scalar sigmoid gate."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            **factory_kwargs,
        )
        self.shared_expert_gate = nn.Linear(
            hidden_size,
            1,
            bias=False,
            **factory_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expert = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return torch.sigmoid(self.shared_expert_gate(hidden_states)) * expert


class Qwen35Router(nn.Module):
    """Qwen3.5 softmax top-k router with the exact target weight shape."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, device=device, dtype=dtype)
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.on_GPU = False

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states, self.weight)
        scores = torch.softmax(logits.float(), dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        return topk_scores.to(hidden_states.dtype), topk_indices


class Qwen35ProxyLayerComponents(nn.Module):
    """Parameter-exact layer components; dispatch remains an APTMoE concern.

    Constructing all 256 real experts allocates 1.5 GiB of BF16 weights per
    layer.  Use ``device='meta'`` for parameter auditing and ``device='cpu'``
    only inside the real APTMoE stage owner.
    """

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_idx: int,
        *,
        device: str | torch.device = "meta",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        with torch.device(device):
            self.token_mixer = Qwen35TokenMixer(config, layer_idx)
            self.input_layernorm = Qwen3_5MoeRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.router = Qwen35Router(
                config.hidden_size,
                config.num_experts,
                config.num_experts_per_tok,
                device=device,
                dtype=dtype,
            )
            self.experts = nn.ModuleList(
                [
                    Qwen35RoutedExpert(
                        config.hidden_size,
                        config.moe_intermediate_size,
                        layer_id=layer_idx,
                        expert_id=expert_id,
                        device=device,
                        dtype=dtype,
                    )
                    for expert_id in range(config.num_experts)
                ]
            )
            self.shared_expert = Qwen35SharedExpert(
                config.hidden_size,
                config.shared_expert_intermediate_size,
                device=device,
                dtype=dtype,
            )
        # Modules created under a device context inherit the device, but dtype
        # still needs an explicit conversion for BF16 proxy construction.
        if str(device) != "meta":
            self.to(**factory_kwargs)


def component_parameter_counts(
    config: Qwen3_5MoeTextConfig,
    layer_idx: int,
) -> dict[str, int]:
    """Audit one layer on the meta device without allocating expert weights."""
    layer = Qwen35ProxyLayerComponents(config, layer_idx, device="meta")
    result = {
        "token_mixer": sum(
            parameter.numel() for parameter in layer.token_mixer.parameters()
        ),
        "router": sum(parameter.numel() for parameter in layer.router.parameters()),
        "routed_experts": sum(
            parameter.numel() for parameter in layer.experts.parameters()
        ),
        "shared_expert_and_gate": sum(
            parameter.numel() for parameter in layer.shared_expert.parameters()
        ),
        "norms": (
            sum(parameter.numel() for parameter in layer.input_layernorm.parameters())
            + sum(
                parameter.numel()
                for parameter in layer.post_attention_layernorm.parameters()
            )
        ),
    }
    result["total"] = sum(result.values())
    return result

#!/usr/bin/env python3
"""Build and validate an APTMoE deployment-proxy contract for Qwen3.5-MoE.

This tool deliberately does not instantiate the 35B model or read checkpoint
shards.  It derives the exact text-model parameter counts from ``config.json``
and emits the component/placement contract that an APTMoE proxy must satisfy.

The generated proxy is a systems benchmark.  It is not a checkpoint-compatible
Qwen3.5 model and its TPS must not be reported as Qwen3.5 training TPS.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GIB = 1 << 30
EXPECTED_SOURCE_MODEL_TYPE = "qwen3_5_moe"
EXPECTED_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
QWEN3_MODEL_TYPE = "qwen3_moe"


@dataclass(frozen=True)
class Component:
    parameters: int
    bf16_bytes: int

    @classmethod
    def from_parameters(cls, parameters: int) -> "Component":
        return cls(parameters=parameters, bf16_bytes=parameters * 2)


def _load_text_config(model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = model_path / "config.json"
    try:
        source = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"config.json was not found: {config_path}") from error

    if source.get("model_type") != EXPECTED_SOURCE_MODEL_TYPE:
        raise SystemExit(
            f"expected source model_type={EXPECTED_SOURCE_MODEL_TYPE!r}, "
            f"got {source.get('model_type')!r}"
        )

    text = source.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != EXPECTED_TEXT_MODEL_TYPE:
        raise SystemExit(
            f"expected text_config.model_type={EXPECTED_TEXT_MODEL_TYPE!r}"
        )
    return source, text


def _positive_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SystemExit(
            f"text_config.{name} must be a positive integer, got {value!r}"
        )
    return value


def _full_attention_parameters(text: dict[str, Any]) -> int:
    hidden = _positive_int(text, "hidden_size")
    heads = _positive_int(text, "num_attention_heads")
    kv_heads = _positive_int(text, "num_key_value_heads")
    head_dim = _positive_int(text, "head_dim")
    q_multiplier = 2 if bool(text.get("attn_output_gate", False)) else 1

    # q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
    return (
        hidden * heads * head_dim * q_multiplier
        + hidden * kv_heads * head_dim * 2
        + hidden * heads * head_dim
        + head_dim * 2
    )


def _linear_attention_parameters(text: dict[str, Any]) -> int:
    hidden = _positive_int(text, "hidden_size")
    key_heads = _positive_int(text, "linear_num_key_heads")
    value_heads = _positive_int(text, "linear_num_value_heads")
    key_head_dim = _positive_int(text, "linear_key_head_dim")
    value_head_dim = _positive_int(text, "linear_value_head_dim")
    kernel = _positive_int(text, "linear_conv_kernel_dim")

    key_dim = key_heads * key_head_dim
    value_dim = value_heads * value_head_dim
    conv_dim = key_dim * 2 + value_dim

    # in_proj_qkv, in_proj_z, in_proj_a/b, depthwise conv, gated RMSNorm,
    # out_proj, dt_bias and A_log.
    return (
        hidden * conv_dim
        + hidden * value_dim
        + hidden * value_heads * 2
        + conv_dim * kernel
        + value_head_dim
        + hidden * value_dim
        + value_heads * 2
    )


def _target_components(text: dict[str, Any]) -> dict[str, Component]:
    hidden = _positive_int(text, "hidden_size")
    layers = _positive_int(text, "num_hidden_layers")
    experts = _positive_int(text, "num_experts")
    expert_intermediate = _positive_int(text, "moe_intermediate_size")
    shared_intermediate = _positive_int(text, "shared_expert_intermediate_size")
    vocab = _positive_int(text, "vocab_size")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise SystemExit(
            "text_config.layer_types must contain one entry per decoder layer"
        )
    unknown = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unknown:
        raise SystemExit(f"unsupported layer_types: {unknown}")

    linear_layers = layer_types.count("linear_attention")
    full_layers = layer_types.count("full_attention")
    routed_per_expert = 3 * hidden * expert_intermediate

    counts = {
        "routed_experts": layers * experts * routed_per_expert,
        "router": layers * hidden * experts,
        "shared_expert": layers * 3 * hidden * shared_intermediate,
        "shared_expert_gate": layers * hidden,
        "linear_attention": linear_layers * _linear_attention_parameters(text),
        "full_attention": full_layers * _full_attention_parameters(text),
        "embedding": vocab * hidden,
        "lm_head": vocab * hidden,
        # Two RMSNorms per layer plus the final RMSNorm.
        "norm": (2 * layers + 1) * hidden,
    }
    return {name: Component.from_parameters(value) for name, value in counts.items()}


def _plain_qwen3_fallback(text: dict[str, Any]) -> dict[str, Component]:
    """Count the tempting but invalid all-GQA Qwen3 proxy."""
    hidden = _positive_int(text, "hidden_size")
    layers = _positive_int(text, "num_hidden_layers")
    heads = _positive_int(text, "num_attention_heads")
    kv_heads = _positive_int(text, "num_key_value_heads")
    head_dim = _positive_int(text, "head_dim")
    experts = _positive_int(text, "num_experts")
    expert_intermediate = _positive_int(text, "moe_intermediate_size")
    vocab = _positive_int(text, "vocab_size")

    qwen3_attention_per_layer = (
        hidden * heads * head_dim
        + hidden * kv_heads * head_dim * 2
        + hidden * heads * head_dim
        + head_dim * 2
    )
    counts = {
        "routed_experts": layers * experts * 3 * hidden * expert_intermediate,
        "router": layers * hidden * experts,
        "shared_expert": 0,
        "shared_expert_gate": 0,
        "attention": layers * qwen3_attention_per_layer,
        "embedding": vocab * hidden,
        "lm_head": vocab * hidden,
        "norm": (2 * layers + 1) * hidden,
    }
    return {name: Component.from_parameters(value) for name, value in counts.items()}


def _qwen3_reference_components(
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Component]]:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(
            f"Qwen3 reference config.json was not found: {config_path}"
        ) from error
    if config.get("model_type") != QWEN3_MODEL_TYPE:
        raise SystemExit(
            f"expected Qwen3 reference model_type={QWEN3_MODEL_TYPE!r}, "
            f"got {config.get('model_type')!r}"
        )

    hidden = _positive_int(config, "hidden_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    experts = _positive_int(config, "num_experts")
    expert_intermediate = _positive_int(config, "moe_intermediate_size")
    vocab = _positive_int(config, "vocab_size")
    attention_per_layer = (
        hidden * heads * head_dim
        + hidden * kv_heads * head_dim * 2
        + hidden * heads * head_dim
        + head_dim * 2
    )
    counts = {
        "routed_experts": layers * experts * 3 * hidden * expert_intermediate,
        "router": layers * hidden * experts,
        "shared_expert": 0,
        "shared_expert_gate": 0,
        "attention": layers * attention_per_layer,
        "embedding": vocab * hidden,
        "lm_head": vocab * hidden,
        "norm": (2 * layers + 1) * hidden,
    }
    return config, {
        name: Component.from_parameters(value) for name, value in counts.items()
    }


def _sum_parameters(
    components: dict[str, Component], names: tuple[str, ...] | None = None
) -> int:
    selected = (
        components.values() if names is None else (components[name] for name in names)
    )
    return sum(component.parameters for component in selected)


def _ratio(actual: int, expected: int) -> float:
    return actual / expected if expected else math.nan


def _storage_item(byte_count: int) -> dict[str, int | float]:
    return {"bytes": byte_count, "gib": byte_count / GIB}


def _training_state_projection(parameters: int) -> dict[str, Any]:
    """Project tensor payloads; allocator, activations, and temporary buffers are extra."""
    bf16_weights = parameters * 2
    bf16_gradients = parameters * 2
    bf16_adam_moments = parameters * 4
    fp32_adam_moments = parameters * 8
    fp32_master_weights = parameters * 4

    return {
        "assumptions": {
            "model_dtype": "bf16",
            "gradient_dtype": "bf16",
            "adam_moment_count": 2,
            "projection_assumes_all_sparse_gradients_and_states_materialized": True,
            "checkpoint_gradients_by_default": False,
            "random_initialization_requires_checkpoint_copy": False,
            "actual_peak_ram_includes_unprojected_activations_and_temporary_buffers": True,
        },
        "components": {
            "bf16_model_weights": _storage_item(bf16_weights),
            "bf16_gradients": _storage_item(bf16_gradients),
            "two_bf16_adam_moments": _storage_item(bf16_adam_moments),
            "two_fp32_adam_moments": _storage_item(fp32_adam_moments),
            "optional_fp32_master_weights": _storage_item(fp32_master_weights),
        },
        "aggregate_runtime_tensor_payload": {
            "current_aptmoe_bf16_moments": _storage_item(
                bf16_weights + bf16_gradients + bf16_adam_moments
            ),
            "fp32_moments_without_master_weights": _storage_item(
                bf16_weights + bf16_gradients + fp32_adam_moments
            ),
            "fp32_moments_with_master_weights": _storage_item(
                bf16_weights
                + bf16_gradients
                + fp32_adam_moments
                + fp32_master_weights
            ),
        },
        "checkpoint_payload_without_gradients": {
            "bf16_model_only": _storage_item(bf16_weights),
            "model_plus_bf16_moments": _storage_item(
                bf16_weights + bf16_adam_moments
            ),
            "model_plus_fp32_moments": _storage_item(
                bf16_weights + fp32_adam_moments
            ),
            "model_plus_fp32_moments_and_master_weights": _storage_item(
                bf16_weights + fp32_adam_moments + fp32_master_weights
            ),
        },
        "model_only_checkpoint_sweep": {
            "eight_sequence_lengths_one_profile": _storage_item(bf16_weights * 8),
            "eight_sequence_lengths_two_profiles": _storage_item(bf16_weights * 16),
        },
    }


def build_manifest(
    model_path: Path,
    qwen3_reference_model_path: Path | None = None,
) -> dict[str, Any]:
    source, text = _load_text_config(model_path)
    target = _target_components(text)
    fallback = _plain_qwen3_fallback(text)

    layer_types = list(text["layer_types"])
    attention_names = ("linear_attention", "full_attention")
    target_attention = _sum_parameters(target, attention_names)
    fallback_attention = fallback["attention"].parameters
    target_total = _sum_parameters(target)
    fallback_total = _sum_parameters(fallback)
    routed_per_expert = (
        3
        * _positive_int(text, "hidden_size")
        * _positive_int(text, "moe_intermediate_size")
    )

    target_json = {name: asdict(component) for name, component in target.items()}
    target_json["attention_total"] = asdict(Component.from_parameters(target_attention))
    target_json["model_total"] = asdict(Component.from_parameters(target_total))

    fallback_json = {name: asdict(component) for name, component in fallback.items()}
    fallback_json["model_total"] = asdict(Component.from_parameters(fallback_total))
    fallback_json["alignment"] = {
        "routed_expert_parameters_ratio": _ratio(
            fallback["routed_experts"].parameters,
            target["routed_experts"].parameters,
        ),
        "attention_parameters_ratio": _ratio(fallback_attention, target_attention),
        "shared_expert_parameters_ratio": 0.0,
        "total_parameters_ratio": _ratio(fallback_total, target_total),
        "sequence_complexity_matches": False,
        "valid_proxy": False,
        "reason": (
            "All 40 layers use quadratic GQA, while the target has 30 linear-attention "
            "and 10 full-attention layers. Similar total parameters do not make this "
            "a valid attention or long-sequence performance proxy."
        ),
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_class": "aptmoe_qwen35_component_isomorphic_deployment_proxy",
        "target": {
            "model_path": str(model_path.resolve()),
            "source_model_type": source["model_type"],
            "text_model_type": text["model_type"],
            "architecture": "Qwen3_5MoeForCausalLM",
            "precision": "bf16",
            "text_only": True,
            "vision_included": False,
            "mtp_included": False,
            "hidden_size": text["hidden_size"],
            "num_hidden_layers": text["num_hidden_layers"],
            "layer_types": layer_types,
            "linear_attention_layers": layer_types.count("linear_attention"),
            "full_attention_layers": layer_types.count("full_attention"),
            "num_experts": text["num_experts"],
            "num_experts_per_tok": text["num_experts_per_tok"],
            "moe_intermediate_size": text["moe_intermediate_size"],
            "shared_expert_intermediate_size": text["shared_expert_intermediate_size"],
            "vocab_size": text["vocab_size"],
            "components": target_json,
            "training_state_projection": _training_state_projection(target_total),
        },
        "required_proxy_contract": {
            "weight_source": "random_initialization",
            "checkpoint_compatible": False,
            "training_quality_claims_allowed": False,
            "real_forward_backward_optimizer_update_required": True,
            "numerical_weight_change_required": True,
            "checkpoint_save_required": False,
            "llamafactory_backend_claim_allowed": False,
            "end_to_end_qwen35_tps_claim_allowed": False,
            "allowed_claims": [
                "GPU token-mixer component time and memory",
                "CPU routed-expert compute and memory",
                "expert CPU/GPU transfer volume and scheduling overhead",
                "proxy-only full-parameter forward/backward/update stability",
            ],
            "gpu_resident": [
                "exact Qwen3.5 GatedDeltaNet/full-attention token mixers",
                "input/post-attention RMSNorm",
                "router",
                "shared expert and shared-expert gate",
            ],
            "cpu_resident_default": ["routed experts"],
            "routed_expert": {
                "count_per_layer": text["num_experts"],
                "total_count": text["num_hidden_layers"] * text["num_experts"],
                "parameters_each": routed_per_expert,
                "bf16_bytes_each": routed_per_expert * 2,
                "bf16_mib_each": routed_per_expert * 2 / (1 << 20),
            },
            "attention_implementation": (
                "Reuse Transformers Qwen3_5MoeGatedDeltaNet and "
                "Qwen3_5MoeAttention, or prove identical tensor shapes and kernels."
            ),
            "attention_runtime_preflight": {
                "require_linear_attention_fastpath": True,
                "record_package_and_kernel_versions": True,
                "fallback_performance_claim_allowed": False,
            },
            "routing": {
                "top_k": text["num_experts_per_tok"],
                "preferred_input": (
                    "replayed per-layer, per-token top-k expert IDs from an "
                    "exact Qwen3.5 run"
                ),
                "fallback": "fixed deterministic routing histogram, reported as synthetic",
            },
            "optimizer": {
                "scope": "all proxy parameters",
                "required_state_precision": "must match the ideal APTMoE reference run",
                "current_aptmoe_default": (
                    "torch.optim.AdamW on BF16 parameters; exp_avg and exp_avg_sq "
                    "are BF16 in the locally installed environment"
                ),
                "require_runtime_state_dtype_audit": True,
                "note": (
                    "Report model weights, gradients, actual Adam moment dtype, and "
                    "optional FP32 master weights separately. Changing BF16 moments "
                    "to FP32 for only one side invalidates the TPS comparison."
                ),
            },
        },
        "plain_qwen3_all_gqa_fallback": fallback_json,
    }

    if qwen3_reference_model_path is not None:
        reference_config, reference = _qwen3_reference_components(
            qwen3_reference_model_path
        )
        reference_total = _sum_parameters(reference)
        reference_expert_parameters = (
            3
            * _positive_int(reference_config, "hidden_size")
            * _positive_int(reference_config, "moe_intermediate_size")
        )
        target_active_expert_parameters = _positive_int(
            text, "num_experts_per_tok"
        ) * routed_per_expert + 3 * _positive_int(text, "hidden_size") * _positive_int(
            text, "shared_expert_intermediate_size"
        )
        reference_active_expert_parameters = (
            _positive_int(reference_config, "num_experts_per_tok")
            * reference_expert_parameters
        )
        reference_json = {
            name: asdict(component) for name, component in reference.items()
        }
        reference_json["model_total"] = asdict(
            Component.from_parameters(reference_total)
        )
        reference_json["model_path"] = str(qwen3_reference_model_path.resolve())
        reference_json["shape"] = {
            "hidden_size": reference_config["hidden_size"],
            "num_hidden_layers": reference_config["num_hidden_layers"],
            "num_experts": reference_config["num_experts"],
            "num_experts_per_tok": reference_config["num_experts_per_tok"],
            "moe_intermediate_size": reference_config["moe_intermediate_size"],
            "expert_count_total": (
                reference_config["num_hidden_layers"] * reference_config["num_experts"]
            ),
            "parameters_per_expert": reference_expert_parameters,
            "bf16_mib_per_expert": reference_expert_parameters * 2 / (1 << 20),
            "attention_kind": "full_attention_on_every_layer",
        }
        reference_json["alignment"] = {
            "routed_expert_storage_ratio": _ratio(
                reference["routed_experts"].parameters,
                target["routed_experts"].parameters,
            ),
            "active_expert_parameters_per_token_ratio": _ratio(
                reference_active_expert_parameters,
                target_active_expert_parameters,
            ),
            "attention_parameters_ratio": _ratio(
                reference["attention"].parameters,
                target_attention,
            ),
            "total_parameters_ratio": _ratio(reference_total, target_total),
            "expert_transfer_granularity_ratio": _ratio(
                reference_expert_parameters,
                routed_per_expert,
            ),
            "expert_count_ratio": _ratio(
                (
                    reference_config["num_hidden_layers"]
                    * reference_config["num_experts"]
                ),
                text["num_hidden_layers"] * text["num_experts"],
            ),
            "sequence_complexity_matches": False,
            "valid_as_aptmoe_operational_baseline": True,
            "valid_as_qwen35_equivalent_proxy": False,
            "reason": (
                "Qwen3-30B is useful for APTMoE pipeline bring-up, but it has 48 "
                "quadratic-attention layers, 6,144 experts of 9 MiB each, and no "
                "shared expert. Qwen3.5 has a 30-linear/10-full attention mix and "
                "10,240 experts of 6 MiB each."
            ),
        }
        manifest["qwen3_30b_reference"] = reference_json

    return manifest


def _print_summary(manifest: dict[str, Any]) -> None:
    components = manifest["target"]["components"]
    storage = manifest["target"]["training_state_projection"]
    fallback = manifest["plain_qwen3_all_gqa_fallback"]
    expert = manifest["required_proxy_contract"]["routed_expert"]

    print("Qwen3.5 text-only target")
    print(f"  parameters: {components['model_total']['parameters']:,}")
    print(f"  BF16 weights: {components['model_total']['bf16_bytes'] / GIB:.3f} GiB")
    print(
        "  attention: "
        f"{components['attention_total']['parameters']:,} params "
        f"({components['attention_total']['bf16_bytes'] / GIB:.3f} GiB)"
    )
    print(
        "  routed experts: "
        f"{components['routed_experts']['parameters']:,} params "
        f"({components['routed_experts']['bf16_bytes'] / GIB:.3f} GiB)"
    )
    print(
        "  expert transfer granularity: "
        f"{expert['parameters_each']:,} params / {expert['bf16_mib_each']:.3f} MiB BF16"
    )
    print(
        "  aggregate model+gradient+current APTMoE BF16 moments: "
        f"{storage['aggregate_runtime_tensor_payload']['current_aptmoe_bf16_moments']['gib']:.3f} GiB"
    )
    print(
        "  model-only / model+BF16-moment checkpoint payload: "
        f"{storage['checkpoint_payload_without_gradients']['bf16_model_only']['gib']:.3f} / "
        f"{storage['checkpoint_payload_without_gradients']['model_plus_bf16_moments']['gib']:.3f} GiB"
    )
    print("Plain Qwen3 all-GQA fallback (rejected)")
    print(
        "  total parameter ratio: "
        f"{fallback['alignment']['total_parameters_ratio']:.4%}"
    )
    print(
        "  attention parameter ratio: "
        f"{fallback['alignment']['attention_parameters_ratio']:.4%}"
    )
    print("  sequence-complexity match: no")
    reference = manifest.get("qwen3_30b_reference")
    if reference:
        print("Unmodified Qwen3-30B-A3B reference")
        print(
            "  total / routed-expert / attention ratios: "
            f"{reference['alignment']['total_parameters_ratio']:.4%} / "
            f"{reference['alignment']['routed_expert_storage_ratio']:.4%} / "
            f"{reference['alignment']['attention_parameters_ratio']:.4%}"
        )
        print(
            "  expert count / transfer-granularity ratios: "
            f"{reference['alignment']['expert_count_ratio']:.4%} / "
            f"{reference['alignment']['expert_transfer_granularity_ratio']:.4%}"
        )
        print(
            "  active expert parameters/token ratio (shared expert included): "
            f"{reference['alignment']['active_expert_parameters_per_token_ratio']:.4%}"
        )
        print("  verdict: operational baseline only; not an equivalent proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/data3/models/Qwen3.5-35B-A3B"),
        help="Qwen3.5 multimodal checkpoint directory (only config.json is read)",
    )
    parser.add_argument(
        "--qwen3-reference-model-path",
        type=Path,
        default=Path("/mnt/data3/models/Qwen3-30B-A3B"),
        help=(
            "optional unmodified Qwen3-30B-A3B directory used for a quantitative "
            "baseline comparison"
        ),
    )
    parser.add_argument(
        "--no-qwen3-reference",
        action="store_true",
        help="do not include the Qwen3-30B-A3B baseline comparison",
    )
    parser.add_argument(
        "--output", type=Path, help="write the proxy manifest to this JSON file"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a concise human-readable summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = (
        None if args.no_qwen3_reference else args.qwen3_reference_model_path
    )
    manifest = build_manifest(args.model_path, reference_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.summary or not args.output:
        _print_summary(manifest)


if __name__ == "__main__":
    main()

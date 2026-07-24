"""Force Qwen3.5-MoE multimodal checkpoints into a text-only CausalLM.

The local Qwen3.5-35B-A3B checkpoint is packaged as
``Qwen3_5MoeForConditionalGeneration``.  For this TPS benchmark we deliberately
load only its ``text_config`` as ``Qwen3_5MoeForCausalLM``.  Transformers 5.x
then remaps ``model.language_model.*`` checkpoint keys to ``model.*`` and
ignores the visual and MTP checkpoint keys.

This module patches only the benchmark process.  It does not modify the model
checkpoint or the shared LLaMA-Factory checkout.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedModel


SOURCE_MODEL_TYPE = "qwen3_5_moe"
SOURCE_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
TEXT_MODEL_TYPE = "qwen3_5_moe_text"
TEXT_ARCHITECTURE = "Qwen3_5MoeForCausalLM"
TEXT_FSDP_LAYER_CLASS = "Qwen3_5MoeDecoderLayer"
_INSTALLED = False


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _rank0_print(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(message, flush=True)


def _extract_text_config(source_config: Any) -> Any:
    source_type = getattr(source_config, "model_type", None)
    source_architectures = list(getattr(source_config, "architectures", None) or [])
    if source_type != SOURCE_MODEL_TYPE or SOURCE_ARCHITECTURE not in source_architectures:
        raise RuntimeError(
            "The text-only benchmark requires a Qwen3.5-MoE multimodal source checkpoint; "
            f"got model_type={source_type!r}, architectures={source_architectures!r}."
        )

    source_text_config = getattr(source_config, "text_config", None)
    if source_text_config is None or getattr(source_text_config, "model_type", None) != TEXT_MODEL_TYPE:
        raise RuntimeError("The source checkpoint does not expose a Qwen3.5-MoE text_config.")

    text_config = copy.deepcopy(source_text_config)
    text_config.architectures = [TEXT_ARCHITECTURE]
    text_config._name_or_path = getattr(source_config, "_name_or_path", "")
    text_config.tie_word_embeddings = bool(getattr(source_config, "tie_word_embeddings", False))
    return text_config


def assert_text_only_model(model: "PreTrainedModel", finetuning_type: str) -> None:
    """Fail before optimizer construction if any multimodal component survived."""
    import torch

    model_type = getattr(model.config, "model_type", None)
    architectures = list(getattr(model.config, "architectures", None) or [])
    if model_type != TEXT_MODEL_TYPE or architectures != [TEXT_ARCHITECTURE]:
        raise RuntimeError(
            "Text-only model contract failed: "
            f"model_type={model_type!r}, architectures={architectures!r}."
        )
    if type(model).__name__ != TEXT_ARCHITECTURE:
        raise RuntimeError(
            f"Expected {TEXT_ARCHITECTURE}, but Transformers constructed {type(model).__name__}."
        )
    if finetuning_type != "full":
        raise RuntimeError(f"This benchmark is full fine-tuning only, got {finetuning_type!r}.")

    conv3d_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv3d)]
    if conv3d_modules:
        raise RuntimeError(f"Text-only model unexpectedly contains Conv3d modules: {conv3d_modules[:5]}")

    multimodal_markers = (
        "visual",
        "vision_tower",
        "multi_modal_projector",
        "image_tower",
        "video_tower",
    )
    multimodal_parameters = [
        name for name, _ in model.named_parameters() if any(marker in name.lower() for marker in multimodal_markers)
    ]
    if multimodal_parameters:
        raise RuntimeError(
            "Text-only model unexpectedly contains multimodal parameters: "
            f"{multimodal_parameters[:5]}"
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    _rank0_print(
        "[qwen35_text_only] contract=OK "
        f"class={type(model).__name__} model_type={model_type} "
        f"conv3d=0 multimodal_params=0 trainable={trainable} total={total}"
    )


def _normalize_text_only_distributed_metadata(model: "PreTrainedModel") -> None:
    """Remove the absent vision block from Transformers' inherited FSDP metadata."""
    decoder_layer_count = sum(
        module.__class__.__name__ == TEXT_FSDP_LAYER_CLASS for module in model.modules()
    )
    if decoder_layer_count == 0:
        raise RuntimeError(
            f"Text-only model does not contain the FSDP wrap class {TEXT_FSDP_LAYER_CLASS}."
        )

    metadata_owners = [model, getattr(model, "model", None)]
    for owner in metadata_owners:
        if owner is not None:
            owner._no_split_modules = [TEXT_FSDP_LAYER_CLASS]

    _rank0_print(
        "[qwen35_text_only] fsdp_wrap="
        f"{TEXT_FSDP_LAYER_CLASS} decoder_layers={decoder_layer_count}; vision block excluded"
    )


def _install_deepspeed_leaf_support() -> None:
    """Teach this LLaMA-Factory version about the text config's MoE leaf."""
    from llamafactory.model import patcher
    from llamafactory.model.model_utils import moe as moe_utils

    original_add_z3_leaf_module = patcher.add_z3_leaf_module

    def add_z3_leaf_module(model: "PreTrainedModel") -> None:
        if getattr(model.config, "model_type", None) != TEXT_MODEL_TYPE:
            original_add_z3_leaf_module(model)
            return

        from transformers.integrations import is_deepspeed_zero3_enabled

        if not is_deepspeed_zero3_enabled():
            return
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

        moe_utils._set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    patcher.add_z3_leaf_module = add_z3_leaf_module


def install_text_only_loading() -> None:
    """Install the strict text-only loader before LLaMA-Factory imports workflows."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not _enabled(os.environ.get("FFT_TEXT_ONLY")):
        raise RuntimeError("FFT_TEXT_ONLY=1 is required for the Qwen3.5 TPS benchmark.")

    import llamafactory.model as model_api
    from llamafactory.model import loader
    from transformers import AutoConfig, AutoTokenizer

    original_load_model = loader.load_model

    def load_text_config(model_args: Any) -> Any:
        init_kwargs = loader._get_init_kwargs(model_args)
        source_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        text_config = _extract_text_config(source_config)
        _rank0_print(
            "[qwen35_text_only] source="
            f"{SOURCE_ARCHITECTURE} -> load={TEXT_ARCHITECTURE}; visual and MTP weights excluded"
        )
        return text_config

    def load_text_tokenizer(model_args: Any) -> dict[str, Any]:
        init_kwargs = loader._get_init_kwargs(model_args)
        attempts = [model_args.use_fast_tokenizer, not model_args.use_fast_tokenizer]
        tokenizer = None
        last_error: Exception | None = None
        for use_fast in dict.fromkeys(attempts):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=use_fast,
                    split_special_tokens=model_args.split_special_tokens,
                    padding_side="right",
                    **init_kwargs,
                )
                break
            except Exception as error:  # match LLaMA-Factory's fast/slow fallback
                last_error = error
        if tokenizer is None:
            raise OSError("Failed to load the text tokenizer.") from last_error

        loader.patch_tokenizer(tokenizer, model_args)
        _rank0_print("[qwen35_text_only] tokenizer loaded; AutoProcessor disabled")
        return {"tokenizer": tokenizer, "processor": None}

    def load_text_model(
        tokenizer: Any,
        model_args: Any,
        finetuning_args: Any,
        is_trainable: bool = False,
        add_valuehead: bool = False,
    ) -> "PreTrainedModel":
        model = original_load_model(
            tokenizer,
            model_args,
            finetuning_args,
            is_trainable=is_trainable,
            add_valuehead=add_valuehead,
        )
        assert_text_only_model(model, str(finetuning_args.finetuning_type))
        _normalize_text_only_distributed_metadata(model)
        route_output = os.environ.get("FFT_ROUTE_TRACE_DIR")
        if route_output:
            from qwen35_route_capture import install_route_capture

            sequence_length = int(
                os.environ.get("FFT_ROUTE_TRACE_SEQUENCE_LENGTH", "0")
            )
            if sequence_length <= 0:
                raise RuntimeError(
                    "FFT_ROUTE_TRACE_SEQUENCE_LENGTH must be positive when "
                    "FFT_ROUTE_TRACE_DIR is set"
                )
            install_route_capture(model, route_output, sequence_length)
        return model

    loader.load_config = load_text_config
    loader.load_tokenizer = load_text_tokenizer
    loader.load_model = load_text_model
    model_api.load_config = load_text_config
    model_api.load_tokenizer = load_text_tokenizer
    model_api.load_model = load_text_model
    _install_deepspeed_leaf_support()
    _INSTALLED = True

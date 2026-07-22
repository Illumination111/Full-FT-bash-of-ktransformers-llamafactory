#!/usr/bin/env python3
"""Validate the local BF16 Qwen3.5 benchmark model and long-text dataset."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def resolve_dataset_file(dataset_dir: Path, dataset_name: str) -> Path:
    info_path = dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    entry = info.get(dataset_name)
    if not isinstance(entry, dict) or not entry.get("file_name"):
        raise ValueError(f"Dataset {dataset_name!r} is not registered in {info_path}")
    path = dataset_dir / str(entry["file_name"])
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {path}")
    return path


def validate_model(model_path: Path) -> dict[str, object]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    source_architecture = "Qwen3_5MoeForConditionalGeneration"
    text_architecture = "Qwen3_5MoeForCausalLM"
    if source_architecture not in architectures:
        raise ValueError(f"Expected Qwen3.5-35B-A3B MoE architecture, got {architectures}")
    text_config = config.get("text_config") or {}
    if text_config.get("model_type") != "qwen3_5_moe_text":
        raise ValueError(
            "Expected qwen3_5_moe_text in the multimodal checkpoint's text_config, "
            f"got {text_config.get('model_type')!r}"
        )
    dtype = str(text_config.get("dtype", text_config.get("torch_dtype", ""))).lower()
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError(f"Model is not declared BF16: text_config dtype={dtype!r}")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Safetensors index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} model shards: {missing[:3]}")
    text_weight_keys = sum(
        key.startswith("model.language_model.") or key.startswith("lm_head.") for key in weight_map
    )
    visual_weight_keys = sum(key.startswith("model.visual.") for key in weight_map)
    mtp_weight_keys = sum(key.startswith("mtp.") for key in weight_map)
    if text_weight_keys == 0 or visual_weight_keys == 0:
        raise ValueError(
            "Expected a multimodal checkpoint containing both language and visual weights; "
            f"found language={text_weight_keys}, visual={visual_weight_keys}"
        )

    from transformers import AutoConfig, AutoModelForCausalLM

    source_config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    runtime_text_config = source_config.text_config
    runtime_text_config.architectures = [text_architecture]
    mapped_class = AutoModelForCausalLM._model_mapping[type(runtime_text_config)].__name__
    if mapped_class != text_architecture:
        raise ValueError(
            f"Transformers cannot map the text_config to {text_architecture}; got {mapped_class}"
        )
    return {
        "source_architecture": architectures[0],
        "source_model_type": config.get("model_type"),
        "load_architecture": text_architecture,
        "load_model_type": text_config.get("model_type"),
        "modality": "text_only",
        "dtype": "bfloat16",
        "model_shards": len(shards),
        "language_weight_keys": text_weight_keys,
        "excluded_visual_weight_keys": visual_weight_keys,
        "excluded_mtp_weight_keys": mtp_weight_keys,
    }


def validate_text_only_rows(rows: list[dict[str, object]]) -> None:
    multimodal_fields = {
        "image",
        "images",
        "video",
        "videos",
        "audio",
        "audios",
    }
    invalid: list[tuple[int, list[str]]] = []
    for index, row in enumerate(rows):
        fields = sorted(multimodal_fields.intersection(row))
        if fields:
            invalid.append((index, fields))
    if invalid:
        raise ValueError(
            "Text-only benchmark dataset contains multimodal fields; "
            f"first rows: {invalid[:5]}"
        )


def token_lengths(model_path: Path, rows: list[dict[str, object]]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    texts = [
        "\n".join(
            str(row.get(key, "")) for key in ("instruction", "input", "output")
        )
        for row in rows
    ]
    lengths: list[int] = []
    for start in range(0, len(texts), 8):
        encoded = tokenizer(
            texts[start : start + 8], add_special_tokens=True, truncation=False
        )["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--required-length", type=int, default=4096)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    model_info = validate_model(args.model_path)
    dataset_file = resolve_dataset_file(args.dataset_dir, args.dataset_name)
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset must be a non-empty JSON list: {dataset_file}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Every dataset sample must be an object: {dataset_file}")
    validate_text_only_rows(rows)
    lengths = token_lengths(args.model_path, rows)
    too_short = [i for i, length in enumerate(lengths) if length <= args.required_length]
    if too_short:
        raise ValueError(
            f"{len(too_short)} samples are <= {args.required_length} tokens; "
            f"first indices: {too_short[:8]}"
        )

    result = {
        **model_info,
        "model_path": str(args.model_path.resolve()),
        "dataset_name": args.dataset_name,
        "dataset_file": str(dataset_file.resolve()),
        "samples": len(rows),
        "required_length_exclusive": args.required_length,
        "token_length_min": min(lengths),
        "token_length_max": max(lengths),
        "token_length_mean": statistics.fmean(lengths),
        "status": "OK",
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

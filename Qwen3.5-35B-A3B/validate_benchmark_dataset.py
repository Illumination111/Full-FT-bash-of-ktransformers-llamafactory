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
    if "Qwen3_5MoeForConditionalGeneration" not in architectures:
        raise ValueError(f"Expected Qwen3.5-35B-A3B MoE architecture, got {architectures}")
    text_config = config.get("text_config") or {}
    dtype = str(text_config.get("dtype", text_config.get("torch_dtype", ""))).lower()
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError(f"Model is not declared BF16: text_config dtype={dtype!r}")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Safetensors index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set((index.get("weight_map") or {}).values()))
    missing = [name for name in shards if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} model shards: {missing[:3]}")
    return {
        "architecture": architectures[0],
        "dtype": "bfloat16",
        "model_shards": len(shards),
    }


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

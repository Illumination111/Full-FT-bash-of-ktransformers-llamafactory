#!/usr/bin/env python3
"""Aggregate probe-free post-warmup phase timing and TPS for a BF16 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMING_MODE = "coarse_host_wall_no_cuda_sync"
RESULT_COLUMNS = [
    "backend",
    "profile",
    "precision",
    "modality",
    "model_load_architecture",
    "sequence_length",
    "num_gpus",
    "global_batch_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "tokens_per_step",
    "cpu_threads_per_rank",
    "cpu_thread_budget_total",
    "warmup_steps",
    "stable_steps",
    "mean_step_sec",
    "stable_tps",
    "forward_sec",
    "backward_sec",
    "optimizer_sec",
    "memory_limit",
    "numa_policy",
    "timing_mode",
    "status",
    "exit_code",
    "run_dir",
]


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def aggregate_run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = config_path.parent
    exit_path = run_dir / "exit_code.txt"
    exit_code = exit_path.read_text(encoding="utf-8").strip() if exit_path.is_file() else "MISSING"
    timing_path = run_dir / "step_timing" / "step_timing.json"
    row: dict[str, Any] = {
        **{key: config.get(key) for key in RESULT_COLUMNS},
        "stable_steps": None,
        "mean_step_sec": None,
        "stable_tps": None,
        "forward_sec": None,
        "backward_sec": None,
        "optimizer_sec": None,
        "timing_mode": None,
        "exit_code": exit_code,
        "run_dir": str(run_dir),
    }
    if exit_code == "DRY_RUN":
        row["status"] = "DRY_RUN"
        return row
    if exit_code != "0":
        row["status"] = "FAILED"
        return row
    if not timing_path.is_file():
        row["status"] = "TIMING_MISSING"
        return row

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    stable = timing.get("aggregate_stable") or {}
    attribution = timing.get("tps_attribution") or {}
    instrumentation = timing.get("instrumentation") or {}
    row["timing_mode"] = timing.get("timing_mode")
    row["stable_steps"] = timing.get("num_stable_steps")
    row["mean_step_sec"] = (stable.get("step_total_sec") or {}).get("mean_sec")
    row["stable_tps"] = attribution.get("stable_tps")
    for key in ("forward_sec", "backward_sec", "optimizer_sec"):
        row[key] = (stable.get(key) or {}).get("mean_sec")

    expected_stable = int(config["steps"]) - int(config["warmup_steps"])
    required_values = (
        row["mean_step_sec"],
        row["stable_tps"],
        row["forward_sec"],
        row["backward_sec"],
        row["optimizer_sec"],
    )
    if str(config.get("precision", "")).lower() != "bf16":
        row["status"] = "PRECISION_MISMATCH"
    elif config.get("modality") != "text_only" or config.get("model_load_architecture") != (
        "Qwen3_5MoeForCausalLM"
    ):
        row["status"] = "MODEL_CONTRACT_MISMATCH"
    elif row["timing_mode"] != TIMING_MODE:
        row["status"] = "TIMING_MODE_MISMATCH"
    elif any(
        instrumentation.get(key) is not False
        for key in (
            "forced_cuda_synchronize",
            "backend_internal_probes",
            "system_resource_monitor",
            "per_step_file_io",
        )
    ):
        row["status"] = "FORBIDDEN_INSTRUMENTATION"
    elif any(as_float(value) is None for value in required_values):
        row["status"] = "TIMING_FIELDS_MISSING"
    elif int(row["stable_steps"] or 0) != expected_stable:
        row["status"] = "INCOMPLETE_STABLE_WINDOW"
    else:
        row["status"] = "OK"
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Qwen3.5-35B-A3B 文本-only BF16 全量微调 TPS Sweep",
        "",
        f"- 结果根目录：`{root}`",
        "- 仅记录每个 optimizer step 的 forward、backward、optimizer 和 total host wall time。",
        "- 不强制 CUDA 同步，不启用后端内部性能探针，不运行系统资源采样器，也不在 step 内写文件。",
        "- TPS 仅使用 `global_step > warmup_steps` 的稳定窗口。",
        "- 多模态源 checkpoint 仅加载 `Qwen3_5MoeForCausalLM`；视觉塔和 processor 不参与。",
        "- 公式：`TPS = GPUs × per-device batch × sequence length × GAS / mean stable step seconds`。",
        "",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("backend")), str(row.get("profile")))].append(row)
    for (backend, profile), group in sorted(groups.items()):
        first = group[0]
        lines.extend(
            [
                f"## {backend} / {profile}",
                "",
                f"- GPU：{first.get('num_gpus')}；全局 batch：{first.get('global_batch_size')}；精度：{first.get('precision')}",
                f"- 模态：{first.get('modality')}；加载架构：{first.get('model_load_architecture')}",
                f"- CPU 线程：{first.get('cpu_threads_per_rank')}/rank，合计预算 {first.get('cpu_thread_budget_total')}",
                f"- 内存策略：{first.get('memory_limit')}；NUMA：{first.get('numa_policy')}",
                "",
                "| Seq | Stable steps | Mean step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | Status |",
                "|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(group, key=lambda item: int(item["sequence_length"])):
            lines.append(
                "| {seq} | {steps} | {step} | {tps} | {forward} | {backward} | {optimizer} | {status} |".format(
                    seq=row["sequence_length"],
                    steps=row.get("stable_steps") or "-",
                    step=fmt(row.get("mean_step_sec")),
                    tps=fmt(row.get("stable_tps"), 2),
                    forward=fmt(row.get("forward_sec")),
                    backward=fmt(row.get("backward_sec")),
                    optimizer=fmt(row.get("optimizer_sec")),
                    status=row.get("status"),
                )
            )
        lines.append("")
    if not rows:
        lines.append("尚无 run_config.json。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = sorted(
        root.glob("**/seq_*/run_config.json"),
        key=lambda path: (
            str(path.parent.parent),
            int(re.search(r"seq_(\d+)", path.parent.name).group(1)),
        ),
    )
    rows = [aggregate_run(path) for path in configs]
    write_csv(output / "sweep_results.csv", rows)
    write_markdown(output / "summary.md", root, rows)
    print(f"[aggregate] runs={len(rows)} -> {output / 'sweep_results.csv'}")
    print(f"[aggregate] summary -> {output / 'summary.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate post-warmup TPS and whole-run host metrics for a BF16 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "backend",
    "profile",
    "precision",
    "sequence_length",
    "num_gpus",
    "global_batch_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "tokens_per_step",
    "warmup_steps",
    "stable_steps",
    "mean_step_sec",
    "stable_tps",
    "dataloader_sec",
    "data_prep_sec",
    "forward_sec",
    "backward_sec",
    "optimizer_sec",
    "step_other_sec",
    "top_bottleneck",
    "cpu_mean_pct_whole_run",
    "cpu_max_pct_whole_run",
    "disk_read_mean_mbps_whole_run",
    "disk_write_mean_mbps_whole_run",
    "gpu_sm_mean_pct_whole_run",
    "gpu_sm_max_pct_whole_run",
    "cgroup_memory_peak_gib",
    "numa0_anon_peak_gib",
    "numa1_anon_peak_gib",
    "memory_limit",
    "numa_policy",
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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def load_monitor(profile_dir: Path) -> dict[str, list[dict[str, str]]]:
    path = profile_dir / "monitor.csv"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not path.is_file():
        return grouped
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row.get("phase", "")].append(row)
    return grouped


def monitor_stats(rows: list[dict[str, str]], devices: str) -> dict[str, float | None]:
    cpu = [v for row in rows if (v := as_float(row.get("cpu_util_pct"))) is not None]
    reads = [v for row in rows if (v := as_float(row.get("disk_read_mbps"))) is not None]
    writes = [v for row in rows if (v := as_float(row.get("disk_write_mbps"))) is not None]
    gpu_ids = [part for part in devices.split(",") if part.isdigit()]
    sm_values: list[float] = []
    for row in rows:
        per_row = [
            value
            for gpu_id in gpu_ids
            if (value := as_float(row.get(f"gpu{gpu_id}_sm_util_pct"))) is not None
        ]
        if per_row:
            sm_values.append(sum(per_row) / len(per_row))
    return {
        "cpu_mean_pct_whole_run": mean(cpu),
        "cpu_max_pct_whole_run": max(cpu) if cpu else None,
        "disk_read_mean_mbps_whole_run": mean(reads),
        "disk_write_mean_mbps_whole_run": mean(writes),
        "gpu_sm_mean_pct_whole_run": mean(sm_values),
        "gpu_sm_max_pct_whole_run": max(sm_values) if sm_values else None,
    }


def aggregate_run(config_path: Path, monitor_cache: dict[Path, dict[str, list[dict[str, str]]]]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = config_path.parent
    profile_dir = run_dir.parent
    monitor = monitor_cache.setdefault(profile_dir, load_monitor(profile_dir))
    system_metrics = monitor_stats(monitor.get(f"seq_{config['sequence_length']}", []), config.get("devices", ""))

    exit_path = run_dir / "exit_code.txt"
    exit_code = exit_path.read_text(encoding="utf-8").strip() if exit_path.is_file() else "MISSING"
    timing_path = run_dir / "step_timing" / "step_timing.json"
    row: dict[str, Any] = {
        **{key: config.get(key) for key in RESULT_COLUMNS},
        **system_metrics,
        "stable_steps": None,
        "mean_step_sec": None,
        "stable_tps": None,
        "dataloader_sec": None,
        "data_prep_sec": None,
        "forward_sec": None,
        "backward_sec": None,
        "optimizer_sec": None,
        "step_other_sec": None,
        "top_bottleneck": None,
        "cgroup_memory_peak_gib": None,
        "numa0_anon_peak_gib": None,
        "numa1_anon_peak_gib": None,
        "exit_code": exit_code,
        "run_dir": str(run_dir),
    }
    resource_summary_path = run_dir / "resource_summary.json"
    if resource_summary_path.is_file():
        resource = json.loads(resource_summary_path.read_text(encoding="utf-8"))
        row["cgroup_memory_peak_gib"] = resource.get("memory_peak_gib")
        row["numa0_anon_peak_gib"] = resource.get("anon_n0_peak_gib")
        row["numa1_anon_peak_gib"] = resource.get("anon_n1_peak_gib")
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
    row["stable_steps"] = timing.get("num_stable_steps")
    row["mean_step_sec"] = (stable.get("step_total_sec") or {}).get("mean_sec")
    row["stable_tps"] = attribution.get("stable_tps")
    for key in (
        "dataloader_sec",
        "data_prep_sec",
        "forward_sec",
        "backward_sec",
        "optimizer_sec",
        "step_other_sec",
    ):
        row[key] = (stable.get(key) or {}).get("mean_sec")
    top = attribution.get("top_bottleneck") or {}
    row["top_bottleneck"] = top.get("phase")
    expected_stable = int(config["steps"]) - int(config["warmup_steps"])
    if int(row["stable_steps"] or 0) != expected_stable:
        row["status"] = "INCOMPLETE_STABLE_WINDOW"
    elif str(config.get("precision", "")).lower() != "bf16":
        row["status"] = "PRECISION_MISMATCH"
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
        "# Qwen3.5-35B-A3B BF16 全量微调 TPS Sweep",
        "",
        f"- 结果根目录：`{root}`",
        "- TPS 仅使用 `global_step > warmup_steps` 的稳定窗口。",
        "- 公式：`TPS = GPUs × per-device batch × sequence length × GAS / mean stable step seconds`。",
        "- CPU、磁盘与 GPU 利用率是对应 sequence 整段运行（含预处理/加载）的监控均值，不冒充稳定步指标。",
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
                f"- 内存策略：{first.get('memory_limit')}；NUMA：{first.get('numa_policy')}",
                "",
                "| Seq | Stable steps | Mean step (s) | TPS | CPU mean/max % | GPU SM mean/max % | Disk R/W MB/s | Cgroup peak / anon N0:N1 GiB | Bottleneck | Status |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in sorted(group, key=lambda item: int(item["sequence_length"])):
            lines.append(
                "| {seq} | {steps} | {step} | {tps} | {cpu_mean}/{cpu_max} | "
                "{gpu_mean}/{gpu_max} | {read}/{write} | {memory}/{numa0}:{numa1} | {bottleneck} | {status} |".format(
                    seq=row["sequence_length"],
                    steps=row.get("stable_steps") or "-",
                    step=fmt(row.get("mean_step_sec")),
                    tps=fmt(row.get("stable_tps"), 2),
                    cpu_mean=fmt(row.get("cpu_mean_pct_whole_run"), 1),
                    cpu_max=fmt(row.get("cpu_max_pct_whole_run"), 1),
                    gpu_mean=fmt(row.get("gpu_sm_mean_pct_whole_run"), 1),
                    gpu_max=fmt(row.get("gpu_sm_max_pct_whole_run"), 1),
                    read=fmt(row.get("disk_read_mean_mbps_whole_run"), 1),
                    write=fmt(row.get("disk_write_mean_mbps_whole_run"), 1),
                    memory=fmt(row.get("cgroup_memory_peak_gib"), 1),
                    numa0=fmt(row.get("numa0_anon_peak_gib"), 1),
                    numa1=fmt(row.get("numa1_anon_peak_gib"), 1),
                    bottleneck=row.get("top_bottleneck") or "-",
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
    monitor_cache: dict[Path, dict[str, list[dict[str, str]]]] = {}
    rows = [aggregate_run(path, monitor_cache) for path in configs]
    write_csv(output / "sweep_results.csv", rows)
    write_markdown(output / "summary.md", root, rows)
    print(f"[aggregate] runs={len(rows)} -> {output / 'sweep_results.csv'}")
    print(f"[aggregate] summary -> {output / 'summary.md'}")


if __name__ == "__main__":
    main()

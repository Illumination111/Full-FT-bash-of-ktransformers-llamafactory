#!/usr/bin/env python3
"""Run a benchmark command while verifying and sampling its own cgroup.

This helper is deliberately launched *inside* the transient consumer scope, so
``/proc/self/cgroup`` describes the same resource domain as every training
rank and dataloader worker. It adds no hooks to the measured optimizer steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CGROUP_ROOT = Path("/sys/fs/cgroup")


def self_cgroup() -> Path:
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, _, relative = line.partition(":")
        if hierarchy == "0":
            relative = relative.split(":", 1)[-1]
            return CGROUP_ROOT / relative.lstrip("/")
    raise RuntimeError("Unable to resolve the unified cgroup v2 path")


def read_text(path: Path, default: str = "unknown") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


def numeric_limit(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def effective_limit(cgroup: Path, filename: str) -> tuple[int | None, list[dict[str, str]]]:
    observations: list[dict[str, str]] = []
    current = cgroup
    numeric: list[int] = []
    while True:
        raw = read_text(current / filename)
        observations.append({"cgroup": str(current), "value": raw})
        parsed = numeric_limit(raw)
        if parsed is not None:
            numeric.append(parsed)
        if current == CGROUP_ROOT:
            break
        current = current.parent
        if CGROUP_ROOT not in (current, *current.parents):
            break
    return (min(numeric) if numeric else None), observations


def parse_numa_stat(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        metric = fields[0]
        for field in fields[1:]:
            node, sep, value = field.partition("=")
            if sep and node.startswith("N"):
                try:
                    result[f"{metric}_{node.lower()}"] = int(value)
                except ValueError:
                    pass
    return result


def numa_policy() -> str:
    try:
        result = subprocess.run(
            ["numactl", "--show"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in result.stdout.splitlines():
        if line.lower().startswith("policy:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def gib(value: int | None) -> float | None:
    return None if value is None else value / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("server", "consumer"), required=True)
    parser.add_argument("--expected-memory-max", type=int)
    parser.add_argument("--require-swap-zero", action="store_true")
    parser.add_argument("--numa-nodes", default="0,1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cgroup = self_cgroup()
    memory_max, memory_chain = effective_limit(cgroup, "memory.max")
    swap_max, swap_chain = effective_limit(cgroup, "memory.swap.max")
    policy = numa_policy()
    contract: dict[str, Any] = {
        "profile": args.profile,
        "cgroup": str(cgroup),
        "memory_max_effective_bytes": memory_max,
        "memory_max_effective_gib": gib(memory_max),
        "memory_max_chain": memory_chain,
        "memory_swap_max_effective_bytes": swap_max,
        "memory_swap_max_chain": swap_chain,
        "numa_nodes": args.numa_nodes,
        "numa_policy": policy,
        "pid": os.getpid(),
        "status": "OK",
    }
    errors: list[str] = []
    if args.profile == "consumer":
        if args.expected_memory_max is None:
            errors.append("consumer expected-memory-max was not supplied")
        elif memory_max != args.expected_memory_max:
            errors.append(
                f"effective memory.max is {memory_max}, expected exactly {args.expected_memory_max}"
            )
        if args.require_swap_zero and swap_max != 0:
            errors.append(f"effective memory.swap.max is {swap_max}, expected 0")
        if policy not in {"interleave", "weighted interleave"}:
            errors.append(f"NUMA policy is {policy!r}, expected interleave")
    if errors:
        contract["status"] = "FAILED"
        contract["errors"] = errors
    contract_path = args.output_dir / "resource_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    if errors:
        for error in errors:
            print(f"[resource_scope_probe] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(91)

    print(
        f"[resource_scope_probe] profile={args.profile} cgroup={cgroup} "
        f"memory_max={memory_max} swap_max={swap_max} numa_policy={policy}",
        flush=True,
    )

    csv_path = args.output_dir / "resource_memory.csv"
    fields = [
        "timestamp",
        "elapsed_sec",
        "memory_current_bytes",
        "memory_current_gib",
        "anon_n0_bytes",
        "anon_n0_gib",
        "anon_n1_bytes",
        "anon_n1_gib",
        "file_n0_bytes",
        "file_n1_bytes",
    ]
    samples: list[dict[str, float | int]] = []
    child: subprocess.Popen[Any] | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    start = time.monotonic()
    with csv_path.open("w", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        child = subprocess.Popen(command)
        while True:
            current = numeric_limit(read_text(cgroup / "memory.current")) or 0
            numa = parse_numa_stat(cgroup / "memory.numa_stat")
            row: dict[str, float | int] = {
                "timestamp": time.time(),
                "elapsed_sec": time.monotonic() - start,
                "memory_current_bytes": current,
                "memory_current_gib": current / (1024**3),
                "anon_n0_bytes": numa.get("anon_n0", 0),
                "anon_n0_gib": numa.get("anon_n0", 0) / (1024**3),
                "anon_n1_bytes": numa.get("anon_n1", 0),
                "anon_n1_gib": numa.get("anon_n1", 0) / (1024**3),
                "file_n0_bytes": numa.get("file_n0", 0),
                "file_n1_bytes": numa.get("file_n1", 0),
            }
            writer.writerow(row)
            samples.append(row)
            try:
                exit_code = child.wait(timeout=max(0.1, args.interval))
                break
            except subprocess.TimeoutExpired:
                continue

    peak = max(samples, key=lambda row: int(row["memory_current_bytes"]))
    summary = {
        "profile": args.profile,
        "cgroup": str(cgroup),
        "memory_max_effective_bytes": memory_max,
        "memory_max_effective_gib": gib(memory_max),
        "memory_peak_bytes": peak["memory_current_bytes"],
        "memory_peak_gib": peak["memory_current_gib"],
        "anon_n0_at_memory_peak_gib": peak["anon_n0_gib"],
        "anon_n1_at_memory_peak_gib": peak["anon_n1_gib"],
        "anon_n0_peak_gib": max(float(row["anon_n0_gib"]) for row in samples),
        "anon_n1_peak_gib": max(float(row["anon_n1_gib"]) for row in samples),
        "samples": len(samples),
        "child_exit_code": exit_code,
    }
    (args.output_dir / "resource_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

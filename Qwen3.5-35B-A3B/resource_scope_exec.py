#!/usr/bin/env python3
"""Validate the benchmark resource scope once, then replace this process.

The validation happens before model loading.  ``exec`` then replaces this
helper with the training command, so no sampler or wrapper process remains
during measured optimizer steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    numeric: list[int] = []
    current = cgroup
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


def numa_policy() -> str:
    try:
        result = subprocess.run(
            ["numactl", "--show"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
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
        "validation": "one_shot_before_exec",
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
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if errors:
        for error in errors:
            print(f"[resource_scope_exec] ERROR: {error}", flush=True)
        raise SystemExit(91)

    print(
        f"[resource_scope_exec] validated profile={args.profile} cgroup={cgroup} "
        f"memory_max={memory_max} swap_max={swap_max} numa_policy={policy}; exec training",
        flush=True,
    )
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()

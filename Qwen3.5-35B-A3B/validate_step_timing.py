#!/usr/bin/env python3
"""Reject timing output that violates the benchmark's probe-free contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TIMING_MODE = "coarse_host_wall_no_cuda_sync"
INSTRUMENTATION_FLAGS = (
    "forced_cuda_synchronize",
    "backend_internal_probes",
    "system_resource_monitor",
    "per_step_file_io",
)
STEP_FIELDS = {
    "global_step",
    "microbatches",
    "forward_sec",
    "backward_sec",
    "optimizer_sec",
    "step_total_sec",
    "step_tps",
}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    args = parser.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("timing_mode") != TIMING_MODE:
        errors.append(f"timing_mode must be {TIMING_MODE!r}")
    instrumentation = data.get("instrumentation") or {}
    for key in INSTRUMENTATION_FLAGS:
        if instrumentation.get(key) is not False:
            errors.append(f"instrumentation.{key} must be false")

    steps = data.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be a list")
        steps = []
    if len(steps) != args.expected_steps:
        errors.append(f"steps has {len(steps)} rows, expected {args.expected_steps}")
    for index, row in enumerate(steps, start=1):
        if not isinstance(row, dict):
            errors.append(f"steps[{index}] must be an object")
            continue
        missing = STEP_FIELDS - row.keys()
        extra = row.keys() - STEP_FIELDS
        if missing:
            errors.append(f"steps[{index}] missing fields: {sorted(missing)}")
        if extra:
            errors.append(f"steps[{index}] contains forbidden fields: {sorted(extra)}")
        for key in STEP_FIELDS - {"global_step", "microbatches"}:
            if key in row and not finite_number(row[key]):
                errors.append(f"steps[{index}].{key} must be a finite number")

    expected_stable = args.expected_steps - args.warmup_steps
    if data.get("num_stable_steps") != expected_stable:
        errors.append(
            f"num_stable_steps={data.get('num_stable_steps')!r}, expected {expected_stable}"
        )
    stable_tps = (data.get("tps_attribution") or {}).get("stable_tps")
    if not finite_number(stable_tps) or float(stable_tps) <= 0:
        errors.append("tps_attribution.stable_tps must be a positive finite number")

    if errors:
        for error in errors:
            print(f"[validate_step_timing] ERROR: {error}")
        raise SystemExit(1)
    print(
        f"[validate_step_timing] OK: {len(steps)} steps, "
        f"{expected_stable} stable, mode={TIMING_MODE}"
    )


if __name__ == "__main__":
    main()

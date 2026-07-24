"""Filesystem safety helpers for large local-only APTMoE proxy artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path


DEFAULT_SIMULATION_ROOT = Path("/mnt/data2/wbw/FFTtest/APTMoE-simulate")


def resolve_simulation_root(path: str | Path | None) -> Path:
    root = Path(path) if path is not None else DEFAULT_SIMULATION_ROOT
    return root.expanduser().resolve()


def require_within_simulation_root(
    path: str | Path,
    simulation_root: str | Path,
) -> Path:
    root = resolve_simulation_root(simulation_root)
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"large proxy artifact path must be inside {root}, got {target}"
        ) from error
    return target


def require_free_space(
    target: str | Path,
    required_bytes: int,
    *,
    reserve_bytes: int = 10 * (1 << 30),
) -> None:
    target_path = Path(target)
    probe = target_path if target_path.exists() else target_path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = shutil.disk_usage(probe).free
    required_with_reserve = required_bytes + reserve_bytes
    if available < required_with_reserve:
        raise OSError(
            "insufficient free space for random proxy weights: "
            f"available={available} required={required_bytes} "
            f"reserve={reserve_bytes} path={target_path}"
        )

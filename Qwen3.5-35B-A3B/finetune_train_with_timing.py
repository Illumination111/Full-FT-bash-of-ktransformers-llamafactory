"""LLaMA-Factory entrypoint with low-overhead, rank-safe step timing.

The timing implementation is shared with the Qwen3 benchmark. Rank 0 writes
the canonical ``step_timing`` files used for TPS; other ranks write sibling
``.rankN`` directories so concurrent callbacks cannot overwrite rank 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _install_timing() -> None:
    precision = os.environ.get("FFT_PRECISION", "bf16").strip().lower()
    if precision not in {"bf16", "bfloat16"}:
        raise RuntimeError(f"This benchmark is BF16-only, got FFT_PRECISION={precision!r}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1 and rank != 0:
        base = os.environ.get("KT_STEP_TIMING_OUT_DIR", "step_timing_out")
        os.environ["KT_STEP_TIMING_OUT_DIR"] = f"{base}.rank{rank}"

    fft_root = Path(__file__).resolve().parent.parent
    timing_module_dir = fft_root / "Qwen3-30B-A3B"
    if not (timing_module_dir / "step_timing_probe.py").is_file():
        raise FileNotFoundError(
            f"Shared timing probe is missing: {timing_module_dir / 'step_timing_probe.py'}"
        )
    sys.path.insert(0, str(timing_module_dir))

    import step_timing_probe

    step_timing_probe.install_step_timing()
    print(
        f"[qwen35_bf16_timing] rank={rank}/{world_size} "
        f"out={os.environ.get('KT_STEP_TIMING_OUT_DIR')}",
        flush=True,
    )


def _disable_benchmark_saves() -> None:
    if os.environ.get("FFT_SKIP_FINAL_SAVE", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    from transformers import Trainer

    def skip_save_model(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.is_world_process_zero():
            print("[qwen35_bf16] final model save skipped for TPS benchmark", flush=True)

    def skip_save_state(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.is_world_process_zero():
            print("[qwen35_bf16] trainer state save skipped for TPS benchmark", flush=True)

    Trainer.save_model = skip_save_model
    Trainer.save_state = skip_save_state


def main() -> None:
    _install_timing()
    _disable_benchmark_saves()
    from llamafactory.cli import main as llamafactory_main

    llamafactory_main()


if __name__ == "__main__":
    main()

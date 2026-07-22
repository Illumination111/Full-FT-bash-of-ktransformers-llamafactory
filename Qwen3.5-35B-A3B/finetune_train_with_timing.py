"""LLaMA-Factory entrypoint with minimal, rank-safe step phase timing."""

from __future__ import annotations

import os


def _install_timing() -> None:
    precision = os.environ.get("FFT_PRECISION", "bf16").strip().lower()
    if precision not in {"bf16", "bfloat16"}:
        raise RuntimeError(f"This benchmark is BF16-only, got FFT_PRECISION={precision!r}")
    if os.environ.get("FFT_DISABLE_PERF_PROBES", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("FFT_DISABLE_PERF_PROBES=1 is required for this benchmark")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1 and rank != 0:
        base = os.environ.get("FFT_STEP_TIMING_OUT_DIR", "step_timing_out")
        os.environ["FFT_STEP_TIMING_OUT_DIR"] = f"{base}.rank{rank}"

    from step_phase_timer import install_step_phase_timing

    install_step_phase_timing()
    print(
        f"[qwen35_bf16_timing] rank={rank}/{world_size} "
        f"out={os.environ.get('FFT_STEP_TIMING_OUT_DIR')}",
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
    from qwen35_text_only import install_text_only_loading

    install_text_only_loading()
    from llamafactory.cli import main as llamafactory_main

    llamafactory_main()


if __name__ == "__main__":
    main()

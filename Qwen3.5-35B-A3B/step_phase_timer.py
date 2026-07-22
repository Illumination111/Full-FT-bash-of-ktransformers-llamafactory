"""Minimal per-step phase timing for the Qwen3.5 BF16 TPS benchmark.

Only coarse forward, backward, and optimizer API boundaries are timed.  The
timer deliberately does not call ``torch.cuda.synchronize()``, install a
PyTorch profiler, sample system resources, or instrument backend internals.
Per-step records are buffered in memory and written after training so file I/O
does not perturb the measured optimizer-step window.
"""

from __future__ import annotations

import atexit
import csv
import functools
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainerCallback


TIMING_MODE = "coarse_host_wall_no_cuda_sync"
PHASE_KEYS = ("forward_sec", "backward_sec", "optimizer_sec")
STEP_KEYS = (
    "global_step",
    "microbatches",
    *PHASE_KEYS,
    "step_total_sec",
    "step_tps",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _stats(rows: list[dict[str, float | int]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"count": 0, "mean_sec": None, "min_sec": None, "max_sec": None}
    return {
        "count": len(values),
        "mean_sec": statistics.fmean(values),
        "min_sec": min(values),
        "max_sec": max(values),
    }


class StepPhaseRecorder:
    """Accumulate three coarse phase timers for each optimizer step."""

    def __init__(
        self,
        out_dir: Path,
        warmup_steps: int,
        tokens_per_step: int,
        backend: str,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.warmup_steps = warmup_steps
        self.tokens_per_step = tokens_per_step
        self.backend = backend
        self.steps: list[dict[str, float | int]] = []
        self._active = False
        self._step_started = 0.0
        self._phase_totals = {key: 0.0 for key in PHASE_KEYS}
        self._optimizer_started: float | None = None
        self._microbatches = 0
        self._written = False

    def begin_step(self) -> None:
        if self._active:
            return
        self._active = True
        self._step_started = time.perf_counter()
        self._phase_totals = {key: 0.0 for key in PHASE_KEYS}
        self._optimizer_started = None
        self._microbatches = 0

    def cancel_step(self) -> None:
        self._active = False
        self._optimizer_started = None

    def add_phase(self, key: str, elapsed: float) -> None:
        if self._active and key in self._phase_totals:
            self._phase_totals[key] += max(0.0, float(elapsed))

    def phase_total(self, key: str) -> float:
        return float(self._phase_totals.get(key, 0.0))

    def add_microbatch(self) -> None:
        if not self._active:
            self.begin_step()
        self._microbatches += 1

    def begin_optimizer(self) -> None:
        if self._active and self._optimizer_started is None:
            self._optimizer_started = time.perf_counter()

    def end_optimizer(self) -> None:
        if self._optimizer_started is None:
            return
        started = self._optimizer_started
        self._optimizer_started = None
        self.add_phase("optimizer_sec", time.perf_counter() - started)

    def finish_step(self, global_step: int) -> None:
        if not self._active:
            return
        self.end_optimizer()
        total = max(0.0, time.perf_counter() - self._step_started)
        row: dict[str, float | int] = {
            "global_step": int(global_step),
            "microbatches": self._microbatches,
            **self._phase_totals,
            "step_total_sec": total,
            "step_tps": self.tokens_per_step / total if total > 0 else 0.0,
        }
        self.steps.append(row)
        self._active = False

    def _summary(self) -> dict[str, Any]:
        stable = [
            row for row in self.steps if int(row["global_step"]) > self.warmup_steps
        ]
        aggregate_all = {key: _stats(self.steps, key) for key in (*PHASE_KEYS, "step_total_sec")}
        aggregate_stable = {key: _stats(stable, key) for key in (*PHASE_KEYS, "step_total_sec")}
        stable_step = aggregate_stable["step_total_sec"]["mean_sec"]
        stable_tps = (
            self.tokens_per_step / float(stable_step)
            if stable_step is not None and float(stable_step) > 0
            else None
        )
        return {
            "schema_version": 1,
            "timing_mode": TIMING_MODE,
            "backend": self.backend,
            "precision": "bf16",
            "instrumentation": {
                "forced_cuda_synchronize": False,
                "backend_internal_probes": False,
                "system_resource_monitor": False,
                "per_step_file_io": False,
            },
            "warmup_steps": self.warmup_steps,
            "tokens_per_step": self.tokens_per_step,
            "num_steps": len(self.steps),
            "num_stable_steps": len(stable),
            "steps": self.steps,
            "aggregate_all": aggregate_all,
            "aggregate_stable": aggregate_stable,
            "tps_attribution": {
                "tokens_per_step": self.tokens_per_step,
                "mean_stable_step_sec": stable_step,
                "stable_tps": stable_tps,
            },
        }

    def write(self) -> None:
        if self._written or not self.steps:
            return
        self._written = True
        self.out_dir.mkdir(parents=True, exist_ok=True)
        summary = self._summary()
        (self.out_dir / "step_timing.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with (self.out_dir / "step_timing.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=STEP_KEYS)
            writer.writeheader()
            writer.writerows(self.steps)

        stable = summary["aggregate_stable"]
        tps = summary["tps_attribution"]["stable_tps"]
        lines = [
            "# Step phase timing",
            "",
            f"- Mode: `{TIMING_MODE}`",
            "- No forced CUDA synchronization, backend-internal probes, system monitor, or per-step file I/O.",
            f"- Stable steps: {summary['num_stable_steps']} (excluded warmup: {self.warmup_steps})",
            f"- Stable TPS: {tps:.3f}" if tps is not None else "- Stable TPS: unavailable",
            "",
            "| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in self.steps:
            lines.append(
                "| {global_step} | {microbatches} | {forward_sec:.6f} | "
                "{backward_sec:.6f} | {optimizer_sec:.6f} | "
                "{step_total_sec:.6f} | {step_tps:.3f} |".format(**row)
            )
        lines.extend(
            [
                "",
                "## Stable means",
                "",
                "| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |",
                "|---:|---:|---:|---:|",
                "| {forward:.6f} | {backward:.6f} | {optimizer:.6f} | {total:.6f} |".format(
                    forward=float(stable["forward_sec"]["mean_sec"] or 0.0),
                    backward=float(stable["backward_sec"]["mean_sec"] or 0.0),
                    optimizer=float(stable["optimizer_sec"]["mean_sec"] or 0.0),
                    total=float(stable["step_total_sec"]["mean_sec"] or 0.0),
                ),
                "",
            ]
        )
        (self.out_dir / "step_timing.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def _wrap_deepspeed_step(trainer: Any, recorder: StepPhaseRecorder) -> None:
    wrapped_engine = getattr(trainer.accelerator, "deepspeed_engine_wrapped", None)
    engine = getattr(wrapped_engine, "engine", None)
    if engine is None or getattr(engine, "_fft_coarse_step_timed", False):
        return
    original_step = engine.step

    @functools.wraps(original_step)
    def timed_step(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_step(*args, **kwargs)
        finally:
            recorder.add_phase("optimizer_sec", time.perf_counter() - started)

    engine.step = timed_step
    engine._fft_coarse_step_timed = True


def _patch_get_batch_samples(recorder: StepPhaseRecorder) -> None:
    if getattr(Trainer, "_fft_coarse_get_batch_patched", False):
        return
    original = Trainer.get_batch_samples

    @functools.wraps(original)
    def timed_get_batch_samples(self: Trainer, *args: Any, **kwargs: Any) -> Any:
        recorder.begin_step()
        try:
            return original(self, *args, **kwargs)
        except BaseException:
            recorder.cancel_step()
            raise

    Trainer.get_batch_samples = timed_get_batch_samples
    Trainer._fft_coarse_get_batch_patched = True


def _patch_training_step(recorder: StepPhaseRecorder) -> None:
    if getattr(Trainer, "_fft_coarse_training_step_patched", False):
        return
    original = Trainer.training_step

    @functools.wraps(original)
    def timed_training_step(self: Trainer, *args: Any, **kwargs: Any) -> Any:
        recorder.add_microbatch()
        if recorder.backend == "deepspeed":
            _wrap_deepspeed_step(self, recorder)

        original_compute_loss = self.compute_loss
        original_backward = self.accelerator.backward

        @functools.wraps(original_compute_loss)
        def timed_compute_loss(*phase_args: Any, **phase_kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_compute_loss(*phase_args, **phase_kwargs)
            finally:
                recorder.add_phase("forward_sec", time.perf_counter() - started)

        @functools.wraps(original_backward)
        def timed_backward(*phase_args: Any, **phase_kwargs: Any) -> Any:
            optimizer_before = recorder.phase_total("optimizer_sec")
            started = time.perf_counter()
            try:
                return original_backward(*phase_args, **phase_kwargs)
            finally:
                elapsed = time.perf_counter() - started
                optimizer_inside = max(
                    0.0,
                    recorder.phase_total("optimizer_sec") - optimizer_before,
                )
                recorder.add_phase("backward_sec", max(0.0, elapsed - optimizer_inside))

        self.compute_loss = timed_compute_loss
        self.accelerator.backward = timed_backward
        try:
            return original(self, *args, **kwargs)
        finally:
            self.compute_loss = original_compute_loss
            self.accelerator.backward = original_backward

    Trainer.training_step = timed_training_step
    Trainer._fft_coarse_training_step_patched = True


class StepPhaseTimingCallback(TrainerCallback):
    def __init__(self, recorder: StepPhaseRecorder) -> None:
        self.recorder = recorder

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.recorder.begin_step()
        return control

    def on_pre_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> Any:
        if self.recorder.backend != "deepspeed":
            self.recorder.begin_optimizer()
        return control

    def on_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> Any:
        if self.recorder.backend != "deepspeed":
            self.recorder.end_optimizer()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.recorder.finish_step(int(state.global_step))
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.recorder.write()
        return control


def install_step_phase_timing() -> StepPhaseRecorder:
    """Install the minimal timer before LLaMA-Factory constructs its Trainer."""

    out_dir = Path(os.environ.get("FFT_STEP_TIMING_OUT_DIR", "step_timing"))
    recorder = StepPhaseRecorder(
        out_dir=out_dir,
        warmup_steps=_env_int("FFT_STEP_TIMING_WARMUP_STEPS", 0),
        tokens_per_step=_env_int("FFT_STEP_TIMING_TOKENS_PER_STEP", 0),
        backend=os.environ.get("FFT_TRAINING_BACKEND", "unknown").strip().lower(),
    )
    _patch_get_batch_samples(recorder)
    _patch_training_step(recorder)

    if getattr(Trainer, "_fft_coarse_callback_installed", False):
        return recorder
    original_init = Trainer.__init__

    @functools.wraps(original_init)
    def patched_init(self: Trainer, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.add_callback(StepPhaseTimingCallback(recorder))

    Trainer.__init__ = patched_init
    Trainer._fft_coarse_callback_installed = True
    atexit.register(recorder.write)
    print(
        f"[step_phase_timer] mode={TIMING_MODE} backend={recorder.backend} out={out_dir}",
        flush=True,
    )
    return recorder

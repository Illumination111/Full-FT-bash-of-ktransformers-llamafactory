"""Per-step Full/LoRA timing probe for KTransformers or DeepSpeed/HF Trainer.

Aligns with tqdm s/it (TPS denominator) by timing one optimizer-step cycle as:
  get_batch_samples (dataloader)
  → training_step (data_prep / forward / backward)
  → clip / optimizer / post_optim
  → update_base_weights (requant; may nest inside next forward)
  → _maybe_log_save_evaluate (logging / checkpoint / eval)

Enabled by KT_STEP_TIMING=1.  DeepSpeed nested timing is additionally selected
with DS_PROBE_MODE=off|low_overhead|exact.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainerCallback


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _sync_cuda_exact() -> None:
    """Synchronize without hiding CUDA failures when exact timing is requested."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Phases that attribute into the TPS step cycle (exclusive of step_total).
ATTRIBUTED_PHASES = (
    "dataloader_sec",
    "data_prep_sec",
    "forward_sec",
    "backward_sec",
    "clip_grad_sec",
    "optimizer_sec",
    "post_optim_sec",
    "update_base_weights_sec",
    "log_save_eval_sec",
    "step_other_sec",
)

PHASE_DISPLAY_ORDER = (
    "backward_sec",
    "update_base_weights_sec",
    "forward_sec",
    "optimizer_sec",
    "dataloader_sec",
    "log_save_eval_sec",
    "clip_grad_sec",
    "post_optim_sec",
    "data_prep_sec",
    "step_other_sec",
    "step_total_sec",
)

PHASE_LABELS = {
    "dataloader_sec": "dataloader/get_batch",
    "data_prep_sec": "data_prep",
    "forward_sec": "forward",
    "backward_sec": "backward",
    "clip_grad_sec": "grad_clip",
    "optimizer_sec": "optimizer",
    "post_optim_sec": "post_optim",
    "update_base_weights_sec": "requant/update_base_weights",
    "log_save_eval_sec": "log/save/eval",
    "step_other_sec": "other",
    "step_total_sec": "TOTAL (TPS cycle)",
}

# These are nested diagnostics, not additional TPS phases.  In DeepSpeed mode
# Accelerate invokes engine.backward() and engine.step() from inside
# accelerator.backward(), so adding these values to ATTRIBUTED_PHASES would
# double-count the step.  Residuals make the nesting explicit:
#
# backward_sec
#   = engine backward + engine step + Accelerate wrapper overhead
# engine step
#   = ZeRO optimizer step + other engine work
# ZeRO optimizer step
#   = CPUAdam step(s) + ZeRO/offload orchestration
DEEPSPEED_DIAGNOSTIC_TIME_KEYS = (
    "ds_engine_backward_sec",
    "ds_engine_step_sec",
    "ds_zero_optimizer_step_sec",
    "ds_cpu_adam_step_sec",
    "ds_wrapper_overhead_sec",
    "ds_engine_step_other_sec",
    "ds_zero_optimizer_overhead_sec",
)

DEEPSPEED_DIAGNOSTIC_CALL_KEYS = (
    "ds_engine_backward_calls",
    "ds_engine_step_calls",
    "ds_zero_optimizer_step_calls",
    "ds_cpu_adam_step_calls",
)

DEEPSPEED_DIAGNOSTIC_LABELS = {
    "ds_engine_backward_sec": "DeepSpeedEngine.backward",
    "ds_engine_step_sec": "DeepSpeedEngine.step (inclusive)",
    "ds_zero_optimizer_step_sec": "ZeRO optimizer.step (inclusive)",
    "ds_cpu_adam_step_sec": "DeepSpeedCPUAdam.step (sum of calls)",
    "ds_wrapper_overhead_sec": "Accelerate wrapper residual",
    "ds_engine_step_other_sec": "engine.step residual outside ZeRO optimizer",
    "ds_zero_optimizer_overhead_sec": "ZeRO/offload residual outside CPUAdam",
}

DEEPSPEED_PROBE_MODES = ("off", "low_overhead", "exact")


class StepTimingRecorder:
    PHASE_KEYS = (*ATTRIBUTED_PHASES, "step_total_sec")
    CSV_KEYS = (
        *PHASE_KEYS,
        *DEEPSPEED_DIAGNOSTIC_TIME_KEYS,
        *DEEPSPEED_DIAGNOSTIC_CALL_KEYS,
    )

    def __init__(
        self,
        out_dir: Path,
        warmup_skip: int = 0,
        tokens_per_step: int = 0,
        finetune_mode: str = "unknown",
        backend: str = "kt",
        deepspeed_probe_mode: str = "off",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.warmup_skip = warmup_skip
        self.tokens_per_step = tokens_per_step
        self.finetune_mode = finetune_mode
        self.backend = backend
        self.deepspeed_probe_mode = deepspeed_probe_mode
        self.deepspeed_probe: dict[str, Any] = {
            "mode": deepspeed_probe_mode,
            "installed": False,
            "components": {},
        }
        self.steps: list[dict[str, Any]] = []
        self._cur: dict[str, float] | None = None
        self._step_t0: float | None = None
        self._accum: dict[str, float] = defaultdict(float)
        self._diagnostic_accum: dict[str, float] = defaultdict(float)
        self._diagnostic_calls: dict[str, int] = defaultdict(int)
        self._micro_count = 0
        self._pending_step: int | None = None
        self.enabled = True
        # Nested phase stack: child time is excluded from parent (e.g. requant in forward).
        self._stack: list[tuple[str, float, float]] = []  # (key, t0, child_sec)
        self.backward_timing = None

    def begin_cycle(self) -> None:
        """Start a TPS-aligned step cycle (usually at get_batch_samples)."""
        if self._cur is not None:
            # Previous cycle never finalized — close it.
            self.finalize_step(self._pending_step or -1)
        _sync_cuda()
        self._step_t0 = time.perf_counter()
        self._cur = defaultdict(float)
        self._accum = defaultdict(float)
        self._diagnostic_accum = defaultdict(float)
        self._diagnostic_calls = defaultdict(int)
        self._micro_count = 0
        self._stack = []
        self._pending_step = None

    def on_step_begin(self) -> None:
        # get_batch_samples usually already opened the cycle; only start if missing.
        if self._cur is None:
            self.begin_cycle()

    def add(self, key: str, dt: float) -> None:
        """Legacy additive API (used by synthetic tests). Prefer begin/end_phase."""
        if self._cur is None:
            if key == "update_base_weights_sec" and self.steps:
                self.steps[-1][key] = float(self.steps[-1].get(key, 0.0)) + dt
                self.steps[-1]["step_total_sec"] = float(self.steps[-1].get("step_total_sec", 0.0)) + dt
            return
        self._accum[key] += dt
        if self._stack:
            key_p, t0, child = self._stack[-1]
            self._stack[-1] = (key_p, t0, child + dt)

    def begin_phase(self, key: str) -> None:
        _sync_cuda()
        self._stack.append((key, time.perf_counter(), 0.0))

    def add_diagnostic(self, key: str, dt: float, calls: int = 1) -> None:
        """Accumulate nested diagnostics without changing exclusive phase timing."""
        if self._cur is None:
            return
        self._diagnostic_accum[key] += max(0.0, float(dt))
        self._diagnostic_calls[f"{key.removesuffix('_sec')}_calls"] += int(calls)

    def end_phase(self, key: str) -> None:
        if not self._stack:
            return
        top_key, t0, child_sec = self._stack.pop()
        if top_key != key:
            key = top_key
        _sync_cuda()
        exclusive = max(0.0, (time.perf_counter() - t0) - child_sec)
        if self._cur is None:
            if key == "update_base_weights_sec" and self.steps:
                self.steps[-1][key] = float(self.steps[-1].get(key, 0.0)) + exclusive
                self.steps[-1]["step_total_sec"] = float(self.steps[-1].get("step_total_sec", 0.0)) + exclusive
            return
        self._accum[key] += exclusive
        if self._stack:
            p_key, p_t0, p_child = self._stack[-1]
            self._stack[-1] = (p_key, p_t0, p_child + exclusive + child_sec)

    def on_step_end(self, global_step: int) -> None:
        """Mark optimizer step finished; finalize after log/save (or immediately if no open cycle wait)."""
        self._pending_step = int(global_step)

    def finalize_step(self, global_step: int | None = None) -> None:
        if self._cur is None or self._step_t0 is None:
            return
        while self._stack:
            self.end_phase(self._stack[-1][0])
        _sync_cuda()
        wall = time.perf_counter() - self._step_t0
        known = sum(float(self._accum[k]) for k in ATTRIBUTED_PHASES if k != "step_other_sec")
        total = max(wall, known)
        other = max(0.0, total - known)
        gs = int(global_step if global_step is not None else (self._pending_step or -1))
        row = {
            "global_step": gs,
            "microbatches": int(self._micro_count),
            "dataloader_sec": float(self._accum["dataloader_sec"]),
            "data_prep_sec": float(self._accum["data_prep_sec"]),
            "forward_sec": float(self._accum["forward_sec"]),
            "backward_sec": float(self._accum["backward_sec"]),
            "clip_grad_sec": float(self._accum["clip_grad_sec"]),
            "optimizer_sec": float(self._accum["optimizer_sec"]),
            "post_optim_sec": float(self._accum["post_optim_sec"]),
            "update_base_weights_sec": float(self._accum["update_base_weights_sec"]),
            "log_save_eval_sec": float(self._accum["log_save_eval_sec"]),
            "step_other_sec": float(other),
            "step_total_sec": float(total),
        }
        for key in DEEPSPEED_DIAGNOSTIC_TIME_KEYS[:4]:
            row[key] = float(self._diagnostic_accum[key])
        row["ds_wrapper_overhead_sec"] = max(
            0.0,
            row["backward_sec"]
            - row["ds_engine_backward_sec"]
            - row["ds_engine_step_sec"],
        )
        row["ds_engine_step_other_sec"] = max(
            0.0,
            row["ds_engine_step_sec"] - row["ds_zero_optimizer_step_sec"],
        )
        row["ds_zero_optimizer_overhead_sec"] = max(
            0.0,
            row["ds_zero_optimizer_step_sec"] - row["ds_cpu_adam_step_sec"],
        )
        for key in DEEPSPEED_DIAGNOSTIC_CALL_KEYS:
            row[key] = int(self._diagnostic_calls[key])
        self.steps.append(row)
        self._cur = None
        self._step_t0 = None
        self._pending_step = None

        print(
            "[step_timing] "
            f"step={row['global_step']} total={row['step_total_sec']:.3f}s "
            f"data={row['dataloader_sec']:.3f}s "
            f"fwd={row['forward_sec']:.3f}s bwd={row['backward_sec']:.3f}s "
            f"optim={row['optimizer_sec']:.3f}s post={row['post_optim_sec']:.3f}s "
            f"requant={row['update_base_weights_sec']:.3f}s "
            f"log={row['log_save_eval_sec']:.3f}s other={row['step_other_sec']:.3f}s",
            flush=True,
        )
        if self.deepspeed_probe_mode != "off":
            print(
                "[deepspeed_probe] "
                f"step={row['global_step']} "
                f"engine_bwd={row['ds_engine_backward_sec']:.3f}s "
                f"engine_step={row['ds_engine_step_sec']:.3f}s "
                f"zero_step={row['ds_zero_optimizer_step_sec']:.3f}s "
                f"cpu_adam={row['ds_cpu_adam_step_sec']:.3f}s "
                f"cpu_adam_calls={row['ds_cpu_adam_step_calls']}",
                flush=True,
            )

    def mark_microbatch(self) -> None:
        self._micro_count += 1

    def summarize(self) -> dict[str, Any]:
        if not self.steps:
            return {
                "status": "EMPTY",
                "reason": "no timed steps recorded",
                "steps": [],
                "aggregate_all": {},
                "aggregate_stable": {},
            }

        def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
            if not rows:
                return {}
            out: dict[str, Any] = {"num_steps": len(rows)}
            for key in (*self.PHASE_KEYS, *DEEPSPEED_DIAGNOSTIC_TIME_KEYS):
                vals = [float(r[key]) for r in rows]
                total = sum(vals)
                out[key] = {
                    "sum_sec": total,
                    "mean_sec": total / len(vals),
                    "median_sec": statistics.median(vals),
                    "min_sec": min(vals),
                    "max_sec": max(vals),
                }
            for key in DEEPSPEED_DIAGNOSTIC_CALL_KEYS:
                vals = [int(r[key]) for r in rows]
                out[key] = {
                    "sum_calls": sum(vals),
                    "mean_calls": sum(vals) / len(vals),
                    "min_calls": min(vals),
                    "max_calls": max(vals),
                }
            total_sum = out["step_total_sec"]["sum_sec"]
            out["fraction_of_total"] = {
                k: (out[k]["sum_sec"] / total_sum if total_sum > 0 else 0.0)
                for k in ATTRIBUTED_PHASES
            }
            return out

        stable = [r for r in self.steps if int(r["global_step"]) > self.warmup_skip]
        agg_stable = _agg(stable)
        tps_attr = build_tps_attribution(agg_stable, self.tokens_per_step)
        if self.backend == "deepspeed":
            phase_legend = {
                "dataloader_sec": "get_batch_samples / DataLoader next (before on_step_begin; in tqdm s/it)",
                "data_prep_sec": "input prepare / device transfer before forward",
                "forward_sec": "GPU model forward + loss, including ZeRO-3 parameter prefetch",
                "backward_sec": "accelerator.backward / GPU autograd + ZeRO-3 gradient partitioning/offload",
                "clip_grad_sec": "gradient clipping",
                "optimizer_sec": (
                    "HF outer optimizer.step (normally a no-op under DeepSpeed; "
                    "use ds_* nested diagnostics for ZeRO/CPUAdam)"
                ),
                "post_optim_sec": "lr_scheduler + zero_grad",
                "update_base_weights_sec": "KT-only metric; expected to remain zero for DeepSpeed",
                "log_save_eval_sec": "_maybe_log_save_evaluate (metric log / checkpoint / eval)",
                "step_other_sec": "unaccounted wall time inside the TPS cycle",
                "step_total_sec": "TPS cycle wall: dataloader → train → log/save (matches tqdm s/it)",
                "deepspeed_nested_diagnostics": (
                    "nested under backward_sec; never add them to TPS phases. "
                    "engine.step includes ZeRO optimizer.step, which includes CPUAdam.step calls"
                ),
            }
        else:
            phase_legend = {
                "dataloader_sec": "get_batch_samples / DataLoader next (before on_step_begin; in tqdm s/it)",
                "data_prep_sec": "input prepare / device transfer before forward",
                "forward_sec": "model forward + loss (compute_loss), exclusive of nested requant",
                "backward_sec": "accelerator.backward / autograd (CPU AMX expert backward plus GPU autograd)",
                "clip_grad_sec": "gradient clipping",
                "optimizer_sec": (
                    "optimizer.step (KT Full: GPU AdamW + CPU DeepSpeedCPUAdam; LoRA: AdamW)"
                ),
                "post_optim_sec": "KT pointer sync + lr_scheduler + GPU/CPU optimizer zero_grad",
                "update_base_weights_sec": "KT update_base_weights / online requant (often nested inside next forward)",
                "log_save_eval_sec": "_maybe_log_save_evaluate (metric log / checkpoint / eval)",
                "step_other_sec": "unaccounted wall time inside the TPS cycle",
                "step_total_sec": "TPS cycle wall: dataloader → train → log/save (matches tqdm s/it)",
            }
        return {
            "status": "OK",
            "warmup_skip": self.warmup_skip,
            "finetune_mode": self.finetune_mode,
            "backend": self.backend,
            "tokens_per_step": self.tokens_per_step,
            "num_steps": len(self.steps),
            "num_stable_steps": len(stable),
            "steps": self.steps,
            "aggregate_all": _agg(self.steps),
            "aggregate_stable": agg_stable,
            "tps_attribution": tps_attr,
            "phase_legend": phase_legend,
            "deepspeed_probe": self.deepspeed_probe,
        }

    def flush(self) -> dict[str, Any]:
        if self._cur is not None:
            self.finalize_step(self._pending_step or -1)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarize()
        json_path = self.out_dir / "step_timing.json"
        csv_path = self.out_dir / "step_timing.csv"
        md_path = self.out_dir / "step_timing.md"

        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["global_step", "microbatches", *self.CSV_KEYS],
            )
            writer.writeheader()
            for row in self.steps:
                writer.writerow(row)

        md_path.write_text(render_timing_markdown(summary))
        print(f"[step_timing] wrote {json_path}", flush=True)
        print(f"[step_timing] wrote {csv_path}", flush=True)
        print(f"[step_timing] wrote {md_path}", flush=True)
        return summary


def build_tps_attribution(agg: dict[str, Any], tokens_per_step: int) -> dict[str, Any]:
    if not agg or "step_total_sec" not in agg:
        return {}
    mean_total = float(agg["step_total_sec"]["mean_sec"])
    fracs = agg.get("fraction_of_total") or {}
    ranked = []
    for key in ATTRIBUTED_PHASES:
        stats = agg.get(key)
        if not stats:
            continue
        ranked.append(
            {
                "phase": key,
                "label": PHASE_LABELS.get(key, key),
                "mean_sec": float(stats["mean_sec"]),
                "share": float(fracs.get(key, 0.0)),
            }
        )
    ranked.sort(key=lambda x: x["mean_sec"], reverse=True)
    tps = (tokens_per_step / mean_total) if tokens_per_step > 0 and mean_total > 0 else None
    # Hypothetical TPS if top phase were free (upper bound intuition).
    hypo = []
    for item in ranked[:3]:
        remain = max(1e-9, mean_total - item["mean_sec"])
        hypo.append(
            {
                "if_remove": item["label"],
                "remaining_sec": remain,
                "tps_if_removed": (tokens_per_step / remain) if tokens_per_step > 0 else None,
            }
        )
    return {
        "tokens_per_step": tokens_per_step,
        "stable_mean_step_sec": mean_total,
        "stable_tps": tps,
        "ranked_phases": ranked,
        "top_bottleneck": ranked[0] if ranked else None,
        "what_if_remove_top": hypo,
    }


def render_timing_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    mode = summary.get("finetune_mode", "unknown")
    lines.append(f"# {mode} Fine-Tuning Step Timing Breakdown")
    lines.append("")
    if summary.get("status") != "OK":
        lines.append(f"status: {summary.get('status')}")
        lines.append(f"reason: {summary.get('reason', '')}")
        return "\n".join(lines) + "\n"

    lines.append(f"- timed_steps: {summary['num_steps']}")
    lines.append(f"- stable_steps (after warmup_skip={summary['warmup_skip']}): {summary['num_stable_steps']}")
    if summary.get("tokens_per_step"):
        lines.append(f"- tokens_per_step: {summary['tokens_per_step']}")
    lines.append("")

    tps = summary.get("tps_attribution") or {}
    if tps:
        lines.append("## TPS bottleneck (stable steps)")
        lines.append("")
        if tps.get("stable_tps") is not None:
            lines.append(
                f"- stable TPS ≈ **{tps['stable_tps']:.2f}** tok/s "
                f"(tokens_per_step={tps['tokens_per_step']} / mean_step={tps['stable_mean_step_sec']:.3f}s)"
            )
        top = tps.get("top_bottleneck")
        if top:
            lines.append(
                f"- top bottleneck: **{top['label']}** "
                f"mean={top['mean_sec']:.3f}s ({top['share'] * 100:.1f}% of step)"
            )
        for w in tps.get("what_if_remove_top") or []:
            if w.get("tps_if_removed") is not None:
                lines.append(
                    f"- if remove `{w['if_remove']}`: ~{w['tps_if_removed']:.1f} tok/s "
                    f"(remaining {w['remaining_sec']:.2f}s/step)"
                )
        lines.append("")

    ds_probe = summary.get("deepspeed_probe") or {}
    if ds_probe.get("mode", "off") != "off":
        agg = summary.get("aggregate_stable") or summary.get("aggregate_all") or {}
        lines.append("## DeepSpeed nested backward / optimizer probe")
        lines.append("")
        lines.append(
            f"- mode: `{ds_probe.get('mode')}`; installed: `{bool(ds_probe.get('installed'))}`"
        )
        lines.append(
            "- nested diagnostics only; do not add them to `step_total_sec` or to each other"
        )
        lines.append("")
        lines.append("| Diagnostic | mean (s/step) | calls/step |")
        lines.append("|---|---:|---:|")
        call_key_by_time = dict(
            zip(DEEPSPEED_DIAGNOSTIC_TIME_KEYS[:4], DEEPSPEED_DIAGNOSTIC_CALL_KEYS)
        )
        for key in DEEPSPEED_DIAGNOSTIC_TIME_KEYS:
            stats = agg.get(key)
            if not stats:
                continue
            call_stats = agg.get(call_key_by_time.get(key, "")) or {}
            calls = call_stats.get("mean_calls")
            calls_s = f"{calls:.2f}" if calls is not None else "-"
            lines.append(
                f"| `{key}` ({DEEPSPEED_DIAGNOSTIC_LABELS[key]}) | "
                f"{stats['mean_sec']:.3f} | {calls_s} |"
            )
        lines.append("")

    for title, key in (("All steps", "aggregate_all"), ("Stable steps only", "aggregate_stable")):
        agg = summary.get(key) or {}
        if not agg:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Phase | mean (s) | median (s) | min (s) | max (s) | sum (s) | share |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        fracs = agg.get("fraction_of_total", {})
        for phase in PHASE_DISPLAY_ORDER:
            stats = agg.get(phase)
            if not stats:
                continue
            share = fracs.get(phase)
            share_s = f"{share * 100:.1f}%" if share is not None else "-"
            lines.append(
                f"| {phase} | {stats['mean_sec']:.3f} | {stats['median_sec']:.3f} | "
                f"{stats['min_sec']:.3f} | {stats['max_sec']:.3f} | {stats['sum_sec']:.3f} | {share_s} |"
            )
        lines.append("")

    lines.append("## Phase legend")
    lines.append("")
    for k, v in (summary.get("phase_legend") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Per-step table")
    lines.append("")
    rows = summary.get("steps") or []
    lines.append(
        "| step | total | data | forward | backward | optim | post-optim | requant | log | other |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if len(rows) <= 24:
        show_rows = rows
        omit_at = None
    else:
        show_rows = rows[:8] + rows[-8:]
        omit_at = 8
    for i, r in enumerate(show_rows):
        if omit_at is not None and i == omit_at:
            lines.append("| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |")
        lines.append(
            f"| {r['global_step']} | {r['step_total_sec']:.3f} | {r['dataloader_sec']:.3f} | "
            f"{r['forward_sec']:.3f} | {r['backward_sec']:.3f} | {r['optimizer_sec']:.3f} | "
            f"{r['post_optim_sec']:.3f} | {r['update_base_weights_sec']:.3f} | "
            f"{r['log_save_eval_sec']:.3f} | {r['step_other_sec']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_job_timing_from_train_log(train_log: Path | str) -> dict[str, Any]:
    """Coarse job-level timeline from train.log timestamps (not in TPS cycle)."""
    path = Path(train_log)
    if not path.exists():
        return {"status": "MISSING", "reason": f"no file: {path}"}

    text = path.read_text(errors="replace")
    backend = "deepspeed" if re.search(r"DeepSpeed|ZeRO[- ]?3|cpu_adam", text, re.I) else "kt"
    ts_pat = re.compile(r"(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    def _near_ts(pos: int) -> datetime | None:
        window = text[max(0, pos - 300) : pos + 80]
        found = ts_pat.findall(window)
        if not found:
            # search a bit earlier
            window = text[max(0, pos - 2000) : pos]
            found = ts_pat.findall(window)
        if not found:
            return None
        return datetime.strptime(found[-1], "%Y-%m-%d %H:%M:%S")

    def _find(pat: str, flags=0) -> tuple[int | None, datetime | None]:
        m = re.search(pat, text, flags)
        if not m:
            return None, None
        return m.start(), _near_ts(m.start())

    all_ts = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in ts_pat.findall(text)]
    markers: dict[str, datetime | None] = {
        "log_first_ts": all_ts[0] if all_ts else None,
        "log_last_ts": all_ts[-1] if all_ts else None,
    }
    for name, pat in (
        ("dataset_load", r"Loading dataset"),
        ("model_config", r"loading configuration file"),
        ("kt_inject", r"Injected \d+ fused expert"),
        ("deepspeed_init", r"DeepSpeed.*info|ZeRO[- ]?3|DeepSpeedCPUAdam|cpu_adam"),
        ("running_training", r"\*+ Running training \*+"),
        ("train_metrics", r"train_runtime"),
    ):
        _, t = _find(pat)
        markers[name] = t

    # online quant occurrences (P7 signal in logs; may be 0 if not printed)
    online_quant_hits = len(re.findall(r"online quant", text, re.I))

    def _sec(a: datetime | None, b: datetime | None) -> float | None:
        if a is None or b is None:
            return None
        return max(0.0, (b - a).total_seconds())

    if backend == "deepspeed":
        segments = [
            ("startup_to_dataset", markers["log_first_ts"], markers["dataset_load"]),
            ("dataset_to_deepspeed_init", markers["dataset_load"], markers["deepspeed_init"]),
            ("model_load_and_deepspeed_init", markers["log_first_ts"], markers["deepspeed_init"]),
            ("deepspeed_init_to_train_loop", markers["deepspeed_init"], markers["running_training"]),
            ("train_loop", markers["running_training"], markers["train_metrics"] or markers["log_last_ts"]),
            ("job_wall_from_log_ts", markers["log_first_ts"], markers["log_last_ts"]),
        ]
    else:
        segments = [
            ("startup_to_dataset", markers["log_first_ts"], markers["dataset_load"]),
            ("dataset_to_kt_inject", markers["dataset_load"], markers["kt_inject"]),
            ("model_load_and_kt_inject", markers["log_first_ts"], markers["kt_inject"]),
            ("kt_inject_to_train_loop", markers["kt_inject"], markers["running_training"]),
            ("train_loop", markers["running_training"], markers["train_metrics"] or markers["log_last_ts"]),
            ("job_wall_from_log_ts", markers["log_first_ts"], markers["log_last_ts"]),
        ]
    out_segments = []
    for name, a, b in segments:
        sec = _sec(a, b)
        out_segments.append(
            {
                "name": name,
                "start": a.isoformat(sep=" ") if a else None,
                "end": b.isoformat(sep=" ") if b else None,
                "sec": sec,
            }
        )

    return {
        "status": "OK",
        "backend": backend,
        "markers": {k: (v.isoformat(sep=" ") if v else None) for k, v in markers.items()},
        "segments": out_segments,
        "online_quant_log_hits": online_quant_hits,
        "note": (
            "Job-level segments are coarse (1s log timestamps). "
            "They explain startup overhead; TPS is limited by per-step phases."
        ),
    }


def render_summary_timing_section(
    step_timing: dict[str, Any] | None,
    job_timing: dict[str, Any] | None = None,
    finetune_mode: str | None = None,
) -> str:
    """Markdown block for Chinese summary.md."""
    lines: list[str] = []
    mode = finetune_mode or (step_timing or {}).get("finetune_mode") or "unknown"
    lines.append(f"## TPS 瓶颈拆解（{mode} Step Timing）")
    lines.append("")

    if not step_timing or step_timing.get("status") != "OK":
        lines.append("- step timing 不可用（需 `KT_STEP_TIMING=1` 重跑训练）")
        lines.append("")
    else:
        agg = step_timing.get("aggregate_stable") or step_timing.get("aggregate_all") or {}
        backend = step_timing.get("backend", "kt")
        tps = step_timing.get("tps_attribution") or build_tps_attribution(
            agg, int(step_timing.get("tokens_per_step") or 0)
        )
        lines.append(
            f"- timed_steps={step_timing.get('num_steps')}  "
            f"stable_steps={step_timing.get('num_stable_steps')}  "
            f"(warmup_skip={step_timing.get('warmup_skip')})"
        )
        if tps.get("stable_tps") is not None:
            lines.append(
                f"- 稳定 TPS ≈ **{tps['stable_tps']:.2f}** tok/s "
                f"= {tps['tokens_per_step']} / {tps['stable_mean_step_sec']:.3f}s"
            )
        top = tps.get("top_bottleneck")
        if top:
            lines.append(
                f"- **当前 TPS 主瓶颈**: `{top['label']}` "
                f"{top['mean_sec']:.3f}s/step ({top['share'] * 100:.1f}%)"
            )
        lines.append("")
        lines.append("| 阶段 | mean (s) | share | 对 TPS 含义 |")
        lines.append("|------|--------:|------:|------------|")
        fracs = agg.get("fraction_of_total") or {}
        for key in PHASE_DISPLAY_ORDER:
            stats = agg.get(key)
            if not stats:
                continue
            label = PHASE_LABELS.get(key, key)
            share = fracs.get(key)
            share_s = f"{share * 100:.1f}%" if share is not None else "-"
            hint = ""
            if key == "backward_sec":
                hint = (
                    "GPU autograd + ZeRO-3 梯度分片/卸载"
                    if backend == "deepspeed"
                    else "CPU AMX expert backward；Full 含基座 dW，LoRA 只含基座 dX + adapter 梯度"
                )
            elif key == "update_base_weights_sec":
                hint = "DeepSpeed 下应为 0" if backend == "deepspeed" else "online requant（P7）"
            elif key == "forward_sec":
                hint = (
                    "GPU 全模型 forward + ZeRO-3 参数预取"
                    if backend == "deepspeed"
                    else "含 GPU attention + CPU expert forward"
                )
            elif key == "optimizer_sec":
                hint = (
                    "HF 外层通常为 0；ZeRO/CPUAdam 见下方 ds_* 嵌套探针"
                    if backend == "deepspeed"
                    else "AdamW；参数规模随 Full/LoRA 模式变化"
                )
            elif key == "dataloader_sec":
                hint = "DataLoader；过大说明输入侧卡住"
            elif key == "log_save_eval_sec":
                hint = "logging/checkpoint/eval"
            elif key == "step_total_sec":
                hint = "与 tqdm s/it 对齐的 TPS 分母"
            lines.append(f"| {label} | {stats['mean_sec']:.3f} | {share_s} | {hint} |")
        lines.append("")
        if tps.get("what_if_remove_top"):
            lines.append("**若去掉最大阶段（粗略上界）:**")
            for w in tps["what_if_remove_top"]:
                if w.get("tps_if_removed") is None:
                    continue
                lines.append(
                    f"- 去掉 `{w['if_remove']}` → ~{w['tps_if_removed']:.1f} tok/s "
                    f"（剩余 {w['remaining_sec']:.2f}s/step）"
                )
            lines.append("")

        ds_probe = step_timing.get("deepspeed_probe") or {}
        if backend == "deepspeed" and ds_probe.get("mode", "off") != "off":
            lines.append("**DeepSpeed backward / optimizer 嵌套探针（不可相加）:**")
            lines.append("")
            lines.append("| 诊断字段 | mean (s/step) | calls/step |")
            lines.append("|---|---:|---:|")
            call_key_by_time = dict(
                zip(DEEPSPEED_DIAGNOSTIC_TIME_KEYS[:4], DEEPSPEED_DIAGNOSTIC_CALL_KEYS)
            )
            for key in DEEPSPEED_DIAGNOSTIC_TIME_KEYS:
                stats = agg.get(key)
                if not stats:
                    continue
                call_stats = agg.get(call_key_by_time.get(key, "")) or {}
                calls = call_stats.get("mean_calls")
                calls_s = f"{calls:.2f}" if calls is not None else "-"
                lines.append(
                    f"| `{key}` ({DEEPSPEED_DIAGNOSTIC_LABELS[key]}) | "
                    f"{stats['mean_sec']:.3f} | {calls_s} |"
                )
            lines.append("")

    if job_timing and job_timing.get("status") == "OK":
        lines.append("## Job 级开销（不计入 TPS，但占墙钟）")
        lines.append("")
        note = job_timing.get("note") or (
            "Job 级分段来自 train.log 时间戳（约 1s 精度），反映启动开销；TPS 仍由逐步阶段决定。"
        )
        # Prefer Chinese note even if older JSON stored English.
        if "Job-level segments" in note or "coarse" in note:
            note = "Job 级分段来自 train.log 时间戳（约 1s 精度），反映启动开销；TPS 仍由逐步阶段决定。"
        lines.append(f"- {note}")
        lines.append("")
        lines.append("| 阶段 | 秒 | 起止 |")
        lines.append("|------|---:|------|")
        seg_labels = {
            "startup_to_dataset": "启动→数据集加载",
            "dataset_to_kt_inject": "数据集→KT inject",
            "model_load_and_kt_inject": "模型加载+KT inject",
            "kt_inject_to_train_loop": "inject→训练循环",
            "dataset_to_deepspeed_init": "数据集→DeepSpeed 初始化",
            "model_load_and_deepspeed_init": "模型加载+DeepSpeed 初始化",
            "deepspeed_init_to_train_loop": "DeepSpeed 初始化→训练循环",
            "train_loop": "训练循环",
            "job_wall_from_log_ts": "整次 job 墙钟（日志）",
        }
        for seg in job_timing.get("segments") or []:
            sec = seg.get("sec")
            sec_s = f"{sec:.0f}" if sec is not None else "N/A"
            name = seg_labels.get(seg["name"], seg["name"])
            lines.append(
                f"| {name} | {sec_s} | {seg.get('start') or '?'} → {seg.get('end') or '?'} |"
            )
        lines.append("")
        if job_timing.get("backend", "kt") != "deepspeed":
            lines.append(f"- online_quant 日志命中次数: {job_timing.get('online_quant_log_hits', 0)}")
        lines.append("")

    return "\n".join(lines)


def _wrap_deepspeed_method(
    target: Any,
    method_name: str,
    recorder: StepTimingRecorder,
    diagnostic_key: str,
    *,
    synchronize: bool,
) -> bool:
    """Wrap one bound DeepSpeed method while preserving its public signature."""
    marker = f"_fft_ds_probe_{method_name}_wrapped"
    if target is None or not hasattr(target, method_name):
        return False
    if getattr(target, marker, False):
        return True

    original = getattr(target, method_name)
    if not callable(original):
        return False

    @functools.wraps(original)
    def timed(*args, **kwargs):
        # exact mode deliberately establishes a completed-work boundary on
        # both sides.  low_overhead mode measures host wall time only.
        if synchronize:
            _sync_cuda_exact()
        t0_ns = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            if synchronize:
                _sync_cuda_exact()
            recorder.add_diagnostic(diagnostic_key, (time.perf_counter_ns() - t0_ns) / 1e9)

    setattr(target, method_name, timed)
    setattr(target, marker, True)
    return True


def _install_deepspeed_timing_probe(trainer: Any, recorder: StepTimingRecorder) -> bool:
    """Attach nested probes after Accelerate has constructed the DS engine."""
    mode = recorder.deepspeed_probe_mode
    if mode == "off":
        return False
    if recorder.deepspeed_probe.get("installed"):
        return True

    accelerator = getattr(trainer, "accelerator", None)
    engine_wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
    engine = getattr(engine_wrapper, "engine", None)
    if engine is None:
        candidate = getattr(trainer, "model_wrapped", None)
        if candidate is not None and "deepspeedengine" in type(candidate).__name__.lower():
            engine = candidate
    if engine is None:
        return False

    zero_optimizer = getattr(engine, "optimizer", None)
    cpu_optimizer = getattr(zero_optimizer, "optimizer", None)
    exact = mode == "exact"

    components = {
        "engine_backward": {
            "class": type(engine).__name__,
            "wrapped": _wrap_deepspeed_method(
                engine,
                "backward",
                recorder,
                "ds_engine_backward_sec",
                synchronize=exact,
            ),
        },
        "engine_step": {
            "class": type(engine).__name__,
            "wrapped": _wrap_deepspeed_method(
                engine,
                "step",
                recorder,
                "ds_engine_step_sec",
                synchronize=exact,
            ),
        },
        "zero_optimizer_step": {
            "class": type(zero_optimizer).__name__ if zero_optimizer is not None else None,
            "wrapped": _wrap_deepspeed_method(
                zero_optimizer,
                "step",
                recorder,
                "ds_zero_optimizer_step_sec",
                synchronize=exact,
            ),
        },
        "cpu_adam_step": {
            "class": type(cpu_optimizer).__name__ if cpu_optimizer is not None else None,
            "wrapped": _wrap_deepspeed_method(
                cpu_optimizer,
                "step",
                recorder,
                "ds_cpu_adam_step_sec",
                # DeepSpeedCPUAdam.step is a synchronous host call.  CUDA
                # synchronization here would only charge unrelated stream work.
                synchronize=False,
            ),
        },
    }
    core_installed = components["engine_backward"]["wrapped"] and components["engine_step"]["wrapped"]
    complete = core_installed and all(item["wrapped"] for item in components.values())
    recorder.deepspeed_probe.update(
        {
            "installed": bool(core_installed),
            "complete": bool(complete),
            "synchronization": "CUDA boundaries" if exact else "host wall only",
            "components": components,
        }
    )
    print(
        "[deepspeed_probe] "
        f"mode={mode} installed={bool(core_installed)} complete={bool(complete)} "
        f"engine={type(engine).__name__} "
        f"zero_optimizer={type(zero_optimizer).__name__ if zero_optimizer is not None else 'missing'} "
        f"cpu_optimizer={type(cpu_optimizer).__name__ if cpu_optimizer is not None else 'missing'}",
        flush=True,
    )
    return bool(core_installed)


class StepTimingCallback(TrainerCallback):
    def __init__(self, recorder: StepTimingRecorder) -> None:
        super().__init__()
        self.recorder = recorder
        self._clip_wrapped = False
        # Trainer can replace the optimizer after create_optimizer(), notably
        # when KT Full FT installs KTHybridOptimizer after accelerator.prepare().
        # Track identities so both the original optimizer and the final facade
        # can be wrapped exactly once.
        self._wrapped_optimizer_ids: set[int] = set()
        self._sched_wrapped = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._wrap_update_base_weights(model)
        trainer_like = kwargs.get("trainer")
        if trainer_like is not None:
            self._wrap_trainer_optim_path(trainer_like)
        else:
            class _Tmp:
                pass

            tmp = _Tmp()
            tmp.optimizer = kwargs.get("optimizer")
            tmp.lr_scheduler = kwargs.get("lr_scheduler")
            if tmp.optimizer is not None or tmp.lr_scheduler is not None:
                self._wrap_trainer_optim_path(tmp)
        print("[step_timing] callback active", flush=True)

    def on_step_begin(self, args, state, control, **kwargs):
        self.recorder.on_step_begin()
        if self.recorder.backward_timing is not None:
            self.recorder.backward_timing.begin_step(int(getattr(state, "global_step", 0) or 0) + 1)

    def on_step_end(self, args, state, control, **kwargs):
        global_step = int(getattr(state, "global_step", 0) or 0)
        if self.recorder.backward_timing is not None:
            self.recorder.backward_timing.end_step(
                global_step,
                float(self.recorder._accum.get("backward_sec", 0.0)),
            )
        self.recorder.on_step_end(global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.recorder.flush()
        if self.recorder.backward_timing is not None:
            self.recorder.backward_timing.flush()

    def _wrap_trainer_optim_path(self, trainer: Any) -> None:
        rec = self.recorder

        if (not self._clip_wrapped) and hasattr(trainer, "_clip_grad_norm"):
            orig_clip = trainer._clip_grad_norm

            def timed_clip(*a, **k):
                rec.begin_phase("clip_grad_sec")
                try:
                    return orig_clip(*a, **k)
                finally:
                    rec.end_phase("clip_grad_sec")

            trainer._clip_grad_norm = timed_clip
            self._clip_wrapped = True

        opt = getattr(trainer, "optimizer", None)
        if opt is not None and id(opt) not in self._wrapped_optimizer_ids:
            # IMPORTANT: preserve original signatures. AcceleratedOptimizer.zero_grad
            # uses inspect.signature(...).parameters to decide whether set_to_none is
            # supported; a bare (*a, **k) wrapper strips set_to_none and crashes KT
            # training with: ValueError: `set_to_none` ... is not supported.
            orig_step = opt.step

            @functools.wraps(orig_step)
            def timed_opt_step(*a, **k):
                rec.begin_phase("optimizer_sec")
                try:
                    return orig_step(*a, **k)
                finally:
                    rec.end_phase("optimizer_sec")

            opt.step = timed_opt_step

            orig_zero = opt.zero_grad

            @functools.wraps(orig_zero)
            def timed_zero(*a, **k):
                rec.begin_phase("post_optim_sec")
                try:
                    return orig_zero(*a, **k)
                finally:
                    rec.end_phase("post_optim_sec")

            opt.zero_grad = timed_zero
            self._wrapped_optimizer_ids.add(id(opt))
            print(f"[step_timing] optimizer wrapped: {type(opt).__name__}", flush=True)

        sched = getattr(trainer, "lr_scheduler", None)
        if (not self._sched_wrapped) and sched is not None:
            orig_lr = sched.step

            @functools.wraps(orig_lr)
            def timed_lr(*a, **k):
                rec.begin_phase("post_optim_sec")
                try:
                    return orig_lr(*a, **k)
                finally:
                    rec.end_phase("post_optim_sec")

            sched.step = timed_lr
            self._sched_wrapped = True

    def _wrap_update_base_weights(self, model) -> None:
        if model is None:
            return
        try:
            from kt_kernel.sft.lora import _find_kt_wrappers
        except Exception:
            return
        wrappers = _find_kt_wrappers(model) or []
        rec = self.recorder
        for w in wrappers:
            inner = getattr(w, "wrapper", None)
            if inner is None or not hasattr(inner, "update_base_weights"):
                continue
            if getattr(inner, "_kt_step_timing_wrapped", False):
                continue
            orig = inner.update_base_weights

            def make_timed(orig_fn):
                def timed_update(*a, **k):
                    rec.begin_phase("update_base_weights_sec")
                    try:
                        return orig_fn(*a, **k)
                    finally:
                        rec.end_phase("update_base_weights_sec")

                return timed_update

            inner.update_base_weights = make_timed(orig)
            inner._kt_step_timing_wrapped = True


def _patch_training_step(recorder: StepTimingRecorder) -> None:
    if getattr(Trainer, "_kt_step_timing_training_step_patched", False):
        return

    orig_training_step = Trainer.training_step

    def timed_training_step(self, model, inputs, num_items_in_batch=None):
        recorder.mark_microbatch()
        if recorder.backend == "deepspeed" and recorder.deepspeed_probe_mode != "off":
            _install_deepspeed_timing_probe(self, recorder)

        orig_prepare = self._prepare_inputs
        orig_compute_loss = self.compute_loss
        orig_backward = self.accelerator.backward

        def timed_prepare(x):
            recorder.begin_phase("data_prep_sec")
            try:
                return orig_prepare(x)
            finally:
                recorder.end_phase("data_prep_sec")

        def timed_compute_loss(*a, **k):
            recorder.begin_phase("forward_sec")
            try:
                return orig_compute_loss(*a, **k)
            finally:
                recorder.end_phase("forward_sec")

        def timed_backward(*a, **k):
            backward_timing = recorder.backward_timing
            if backward_timing is not None:
                backward_timing.begin_microbatch()
            recorder.begin_phase("backward_sec")
            try:
                return orig_backward(*a, **k)
            finally:
                recorder.end_phase("backward_sec")
                if backward_timing is not None:
                    backward_timing.end_microbatch()

        self._prepare_inputs = timed_prepare
        self.compute_loss = timed_compute_loss
        self.accelerator.backward = timed_backward
        try:
            return orig_training_step(self, model, inputs, num_items_in_batch=num_items_in_batch)
        finally:
            self._prepare_inputs = orig_prepare
            self.compute_loss = orig_compute_loss
            self.accelerator.backward = orig_backward

    Trainer.training_step = timed_training_step
    Trainer._kt_step_timing_training_step_patched = True


def _patch_get_batch_samples(recorder: StepTimingRecorder) -> None:
    if getattr(Trainer, "_kt_step_timing_get_batch_patched", False):
        return
    orig = Trainer.get_batch_samples

    def timed_get_batch_samples(self, epoch_iterator, num_batches, device):
        recorder.begin_cycle()
        recorder.begin_phase("dataloader_sec")
        try:
            return orig(self, epoch_iterator, num_batches, device)
        finally:
            recorder.end_phase("dataloader_sec")

    Trainer.get_batch_samples = timed_get_batch_samples
    Trainer._kt_step_timing_get_batch_patched = True


def _patch_maybe_log_save_evaluate(recorder: StepTimingRecorder) -> None:
    if getattr(Trainer, "_kt_step_timing_log_save_patched", False):
        return
    orig = Trainer._maybe_log_save_evaluate

    def timed_maybe_log(self, *args, **kwargs):
        # Only charge/finalize when we are inside an open TPS cycle.
        if recorder._cur is None:
            return orig(self, *args, **kwargs)
        recorder.begin_phase("log_save_eval_sec")
        try:
            return orig(self, *args, **kwargs)
        finally:
            recorder.end_phase("log_save_eval_sec")
            if recorder._pending_step is not None or recorder._cur is not None:
                recorder.finalize_step()

    Trainer._maybe_log_save_evaluate = timed_maybe_log
    Trainer._kt_step_timing_log_save_patched = True


def _patch_update_kt_lora_pointers(recorder: StepTimingRecorder) -> None:
    try:
        import transformers.trainer as trainer_mod
    except Exception:
        return
    fn = getattr(trainer_mod, "update_kt_lora_pointers", None)
    if fn is None or getattr(fn, "_kt_step_timing_wrapped", False):
        return

    def timed_update(model):
        recorder.begin_phase("post_optim_sec")
        try:
            return fn(model)
        finally:
            recorder.end_phase("post_optim_sec")

    timed_update._kt_step_timing_wrapped = True
    trainer_mod.update_kt_lora_pointers = timed_update


class _TrainerAwareTimingCallback(StepTimingCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model=model, **kwargs)


def install_step_timing() -> StepTimingRecorder | None:
    if not _env_flag("KT_STEP_TIMING", "0"):
        return None

    out_dir = Path(os.environ.get("KT_STEP_TIMING_OUT_DIR", "step_timing_out"))
    warmup = _env_int("KT_STEP_TIMING_WARMUP_SKIP", 0)
    tokens = _env_int("KT_STEP_TIMING_TOKENS_PER_STEP", 0)
    finetune_mode = os.environ.get("KT_FINETUNE_MODE", "unknown")
    backend = os.environ.get("FFT_TRAINING_BACKEND", "kt").lower()
    deepspeed_probe_mode = os.environ.get("DS_PROBE_MODE", "off").strip().lower()
    if deepspeed_probe_mode not in DEEPSPEED_PROBE_MODES:
        raise ValueError(
            f"Invalid DS_PROBE_MODE={deepspeed_probe_mode!r}; "
            f"expected one of {', '.join(DEEPSPEED_PROBE_MODES)}"
        )
    if backend != "deepspeed" and deepspeed_probe_mode != "off":
        print(
            f"[deepspeed_probe] ignoring mode={deepspeed_probe_mode}: backend={backend}",
            flush=True,
        )
        deepspeed_probe_mode = "off"
    recorder = StepTimingRecorder(
        out_dir=out_dir,
        warmup_skip=warmup,
        tokens_per_step=tokens,
        finetune_mode=finetune_mode,
        backend=backend,
        deepspeed_probe_mode=deepspeed_probe_mode,
    )
    if os.environ.get("KT_BACKWARD_TIMING", "off").strip().lower() not in ("", "0", "off"):
        try:
            from kt_kernel.sft.backward_timing import get_backward_timing_recorder

            recorder.backward_timing = get_backward_timing_recorder()
            print(
                f"[backward_timing] mode={recorder.backward_timing.mode} "
                f"-> {recorder.backward_timing.out_dir}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError("Failed to initialize KT backward internal timing") from exc

    _patch_get_batch_samples(recorder)
    _patch_training_step(recorder)
    _patch_maybe_log_save_evaluate(recorder)
    _patch_update_kt_lora_pointers(recorder)

    if getattr(Trainer, "_kt_step_timing_callback_installed", False):
        return recorder

    original_init = Trainer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            cb = _TrainerAwareTimingCallback(recorder)
            self.add_callback(cb)
            self._kt_step_timing_recorder = recorder
            self._kt_step_timing_callback = cb
            print("[step_timing] callback installed on Trainer", flush=True)
        except Exception as exc:
            print(f"[step_timing] failed to install callback: {exc}", flush=True)

    if hasattr(Trainer, "create_optimizer") and not getattr(Trainer, "_kt_step_timing_create_opt_patched", False):
        orig_create_opt = Trainer.create_optimizer

        def timed_create_optimizer(self):
            out = orig_create_opt(self)
            cb = getattr(self, "_kt_step_timing_callback", None)
            if cb is not None:
                cb._wrap_trainer_optim_path(self)
            return out

        Trainer.create_optimizer = timed_create_optimizer
        Trainer._kt_step_timing_create_opt_patched = True

    if hasattr(Trainer, "create_scheduler") and not getattr(Trainer, "_kt_step_timing_create_sched_patched", False):
        orig_create_sched = Trainer.create_scheduler

        def timed_create_scheduler(self, *args, **kwargs):
            out = orig_create_sched(self, *args, **kwargs)
            cb = getattr(self, "_kt_step_timing_callback", None)
            if cb is not None:
                cb._wrap_trainer_optim_path(self)
            return out

        Trainer.create_scheduler = timed_create_scheduler
        Trainer._kt_step_timing_create_sched_patched = True

    Trainer.__init__ = patched_init
    Trainer._kt_step_timing_callback_installed = True
    print(
        f"[step_timing] enabled -> {out_dir}; deepspeed_probe={deepspeed_probe_mode}",
        flush=True,
    )
    return recorder

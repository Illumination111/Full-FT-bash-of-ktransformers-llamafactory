"""Runtime stall detector for Full-FT (no KT / LLaMA-Factory code changes).

Watches wall-clock vs process CPU time during hot phases (especially
optimizer.step). When the process is effectively idle for longer than a
threshold while still inside the timed phase, it snapshots:

  - /proc/self/stat, status, io
  - /proc/self/task/*/stack (kernel wait stacks; often needs CAP_SYS_PTRACE
    or same-uid; best-effort)
  - /proc/vmstat (pgfault / pgmajfault / numa_* deltas)
  - optional: brief `perf stat` is NOT used (needs privileges / overhead)

Enable with KT_STALL_WATCH=1. Output goes under KT_STEP_TIMING_OUT_DIR or
KT_STALL_WATCH_OUT_DIR.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _read_text(path: str, max_bytes: int = 65536) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _proc_cpu_seconds(pid: int = 0) -> float | None:
    """User+sys CPU seconds for this process from /proc/<pid>/stat."""
    path = f"/proc/{pid or 'self'}/stat"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        # comm may contain spaces/parens: split after last ')'
        rparen = raw.rfind(")")
        if rparen < 0:
            return None
        fields = raw[rparen + 2 :].split()
        # fields[11]=utime, fields[12]=stime (0-based after state)
        utime = int(fields[11])
        stime = int(fields[12])
        hz = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
        return (utime + stime) / float(hz)
    except Exception:
        return None


def _parse_vmstat() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/vmstat", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in (
                    "pgfault",
                    "pgmajfault",
                    "pgpgin",
                    "pgpgout",
                    "pswpin",
                    "pswpout",
                    "numa_hit",
                    "numa_miss",
                    "numa_foreign",
                    "numa_interleave",
                    "numa_local",
                    "numa_other",
                    "compact_stall",
                    "compact_fail",
                    "compact_success",
                    "kswapd_low_wmark_hit_quickly",
                    "allocstall_dma",
                    "allocstall_dma32",
                    "allocstall_normal",
                    "allocstall_movable",
                ):
                    out[parts[0]] = int(parts[1])
    except Exception:
        pass
    return out


def _status_fields() -> dict[str, str]:
    wanted = (
        "VmRSS",
        "VmSize",
        "voluntary_ctxt_switches",
        "nonvoluntary_ctxt_switches",
        "Threads",
        "Cpus_allowed_list",
        "Mems_allowed_list",
    )
    text = _read_text("/proc/self/status")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k in wanted:
            out[k] = v.strip()
    return out


def _io_fields() -> dict[str, str]:
    text = _read_text("/proc/self/io")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _sample_task_stacks(max_tasks: int = 8, max_lines: int = 12) -> list[dict[str, Any]]:
    """Best-effort kernel stacks for a few threads (often empty without ptrace)."""
    samples: list[dict[str, Any]] = []
    task_dir = Path("/proc/self/task")
    try:
        tids = sorted(int(p.name) for p in task_dir.iterdir() if p.name.isdigit())
    except Exception:
        return samples

    # Prefer main tid first, then a few others.
    for tid in tids[:max_tasks]:
        stack = _read_text(f"/proc/self/task/{tid}/stack", max_bytes=8192)
        wchan = _read_text(f"/proc/self/task/{tid}/wchan", max_bytes=256).strip()
        if stack.startswith("<unavailable") and not wchan:
            continue
        lines = [ln for ln in stack.splitlines() if ln.strip()][:max_lines]
        if not lines and not wchan:
            continue
        samples.append({"tid": tid, "wchan": wchan, "stack_top": lines})
    return samples


@dataclass
class StallEvent:
    phase: str
    global_step_hint: int | None
    wall_sec: float
    cpu_delta_sec: float | None
    idle_ratio: float | None
    detected_at_wall: float
    status: dict[str, str] = field(default_factory=dict)
    io: dict[str, str] = field(default_factory=dict)
    vmstat_delta: dict[str, int] = field(default_factory=dict)
    task_stacks: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""


class StallWatchdog:
    """Background sampler that fires when a phase stalls."""

    def __init__(
        self,
        out_dir: Path,
        idle_sec: float = 5.0,
        poll_sec: float = 1.0,
        cpu_idle_ratio: float = 0.15,
        phases: tuple[str, ...] = ("optimizer", "backward", "post_optim"),
    ) -> None:
        self.out_dir = Path(out_dir)
        self.idle_sec = idle_sec
        self.poll_sec = poll_sec
        self.cpu_idle_ratio = cpu_idle_ratio
        self.phases = set(phases)
        self.events: list[StallEvent] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active_phase: str | None = None
        self._phase_t0: float | None = None
        self._phase_cpu0: float | None = None
        self._phase_vm0: dict[str, int] = {}
        self._last_progress_wall: float | None = None
        self._last_progress_cpu: float | None = None
        self._fired_this_phase = False
        self._step_hint: int | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="kt-stall-watch", daemon=True)
        self._thread.start()
        print(
            f"[stall_watch] enabled idle_sec={self.idle_sec} poll={self.poll_sec}s "
            f"phases={sorted(self.phases)} -> {self.out_dir}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.flush()

    def set_step_hint(self, step: int | None) -> None:
        self._step_hint = step

    def enter(self, phase: str) -> None:
        if phase not in self.phases:
            return
        with self._lock:
            now = time.perf_counter()
            cpu = _proc_cpu_seconds()
            self._active_phase = phase
            self._phase_t0 = now
            self._phase_cpu0 = cpu
            self._phase_vm0 = _parse_vmstat()
            self._last_progress_wall = now
            self._last_progress_cpu = cpu
            self._fired_this_phase = False

    def leave(self, phase: str) -> None:
        with self._lock:
            if self._active_phase != phase:
                return
            # If phase was long and mostly idle but never crossed threshold mid-way,
            # still record a summary when wall >> cpu.
            if (
                not self._fired_this_phase
                and self._phase_t0 is not None
                and self._phase_cpu0 is not None
            ):
                wall = time.perf_counter() - self._phase_t0
                cpu_now = _proc_cpu_seconds()
                if cpu_now is not None and wall >= self.idle_sec:
                    cpu_delta = max(0.0, cpu_now - self._phase_cpu0)
                    idle_ratio = 1.0 - min(1.0, cpu_delta / wall) if wall > 0 else 0.0
                    if idle_ratio >= (1.0 - self.cpu_idle_ratio) and wall >= self.idle_sec * 2:
                        self._record_event(phase, wall, cpu_delta, idle_ratio, end_of_phase=True)
            self._active_phase = None
            self._phase_t0 = None
            self._phase_cpu0 = None

    def wrap(self, phase: str, fn: Callable) -> Callable:
        """Return a signature-preserving wrapper that watches `phase`."""

        @functools.wraps(fn)
        def wrapped(*a, **k):
            self.enter(phase)
            try:
                return fn(*a, **k)
            finally:
                self.leave(phase)

        return wrapped

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_sec):
            with self._lock:
                phase = self._active_phase
                if phase is None or self._fired_this_phase:
                    continue
                now = time.perf_counter()
                cpu = _proc_cpu_seconds()
                if self._last_progress_wall is None or self._last_progress_cpu is None:
                    self._last_progress_wall = now
                    self._last_progress_cpu = cpu
                    continue
                if cpu is not None and self._last_progress_cpu is not None:
                    cpu_gain = cpu - self._last_progress_cpu
                    # Any meaningful CPU progress resets the idle timer.
                    if cpu_gain >= 0.05:
                        self._last_progress_wall = now
                        self._last_progress_cpu = cpu
                        continue
                idle_for = now - self._last_progress_wall
                if idle_for < self.idle_sec:
                    continue
                wall = now - (self._phase_t0 or now)
                cpu_delta = None
                idle_ratio = None
                if cpu is not None and self._phase_cpu0 is not None:
                    cpu_delta = max(0.0, cpu - self._phase_cpu0)
                    idle_ratio = 1.0 - min(1.0, cpu_delta / wall) if wall > 0 else 1.0
                self._record_event(phase, wall, cpu_delta, idle_ratio, end_of_phase=False)
                self._fired_this_phase = True

    def _record_event(
        self,
        phase: str,
        wall: float,
        cpu_delta: float | None,
        idle_ratio: float | None,
        end_of_phase: bool,
    ) -> None:
        vm1 = _parse_vmstat()
        vm_delta = {k: vm1.get(k, 0) - self._phase_vm0.get(k, 0) for k in set(vm1) | set(self._phase_vm0)}
        # Keep only interesting counters
        vm_delta = {k: v for k, v in vm_delta.items() if v != 0}
        ev = StallEvent(
            phase=phase,
            global_step_hint=self._step_hint,
            wall_sec=float(wall),
            cpu_delta_sec=None if cpu_delta is None else float(cpu_delta),
            idle_ratio=None if idle_ratio is None else float(idle_ratio),
            detected_at_wall=time.time(),
            status=_status_fields(),
            io=_io_fields(),
            vmstat_delta=vm_delta,
            task_stacks=_sample_task_stacks(),
            note=(
                "end-of-phase mostly-idle summary"
                if end_of_phase
                else f"no CPU progress for >= {self.idle_sec:.1f}s inside {phase}"
            ),
        )
        self.events.append(ev)
        print(
            "[stall_watch] "
            f"STALL phase={phase} step_hint={self._step_hint} "
            f"wall={wall:.1f}s cpu_delta={cpu_delta if cpu_delta is not None else 'NA'} "
            f"idle_ratio={idle_ratio if idle_ratio is not None else 'NA'} "
            f"vmΔ={ {k: vm_delta[k] for k in list(vm_delta)[:6]} }",
            flush=True,
        )
        # Persist incrementally so a later crash still keeps evidence.
        self.flush()

    def flush(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "stall_events.json"
        payload = {
            "status": "OK" if self.events else "EMPTY",
            "num_events": len(self.events),
            "config": {
                "idle_sec": self.idle_sec,
                "poll_sec": self.poll_sec,
                "cpu_idle_ratio": self.cpu_idle_ratio,
                "phases": sorted(self.phases),
            },
            "events": [asdict(e) for e in self.events],
            "interpretation_hints": [
                "wall>>cpu + pgmajfault↑ => page faults / reclaim",
                "wall>>cpu + numa_foreign/numa_miss↑ => NUMA remote / migration",
                "wall>>cpu + compact_stall↑ => memory compaction stall",
                "wall>>cpu + stacks show D state / waiting on mutex => lock or kernel wait",
                "wall>>cpu + stacks unavailable => rerun with CAP_SYS_PTRACE or echo 0 > /proc/sys/kernel/yama/ptrace_scope (admin)",
            ],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        md = self.out_dir / "stall_events.md"
        md.write_text(render_stall_markdown(payload))
        return path


def render_stall_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Stall Watch Report", ""]
    lines.append(f"- status: {payload.get('status')}")
    lines.append(f"- num_events: {payload.get('num_events')}")
    cfg = payload.get("config") or {}
    lines.append(
        f"- config: idle_sec={cfg.get('idle_sec')} poll={cfg.get('poll_sec')} "
        f"phases={cfg.get('phases')}"
    )
    lines.append("")
    events = payload.get("events") or []
    if not events:
        lines.append("No stalls detected.")
        lines.append("")
        return "\n".join(lines)
    for i, ev in enumerate(events, 1):
        lines.append(f"## Event {i}: {ev.get('phase')} (step_hint={ev.get('global_step_hint')})")
        lines.append("")
        lines.append(f"- wall_sec: {ev.get('wall_sec'):.2f}")
        lines.append(f"- cpu_delta_sec: {ev.get('cpu_delta_sec')}")
        lines.append(f"- idle_ratio: {ev.get('idle_ratio')}")
        lines.append(f"- note: {ev.get('note')}")
        vm = ev.get("vmstat_delta") or {}
        if vm:
            top = sorted(vm.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
            lines.append("- vmstat Δ: " + ", ".join(f"{k}={v}" for k, v in top))
        st = ev.get("status") or {}
        if st:
            lines.append(
                "- status: "
                + ", ".join(f"{k}={st[k]}" for k in ("VmRSS", "Threads", "Mems_allowed_list") if k in st)
            )
        stacks = ev.get("task_stacks") or []
        if stacks:
            lines.append("- task stacks (top):")
            for s in stacks[:4]:
                lines.append(f"  - tid={s.get('tid')} wchan={s.get('wchan')}")
                for ln in (s.get("stack_top") or [])[:5]:
                    lines.append(f"    {ln}")
        else:
            lines.append("- task stacks: unavailable (permissions / yama ptrace_scope)")
        lines.append("")
    lines.append("## Hints")
    lines.append("")
    for h in payload.get("interpretation_hints") or []:
        lines.append(f"- {h}")
    lines.append("")
    return "\n".join(lines)


_WATCHDOG: StallWatchdog | None = None


def get_watchdog() -> StallWatchdog | None:
    return _WATCHDOG


def install_stall_watch() -> StallWatchdog | None:
    """Install watchdog and patch Trainer optimizer path via step_timing hooks if present.

    Safe to call with or without KT_STEP_TIMING. If step timing is enabled, we
    attach to the same optimizer wrappers; otherwise we patch Trainer.create_optimizer.
    """
    global _WATCHDOG
    if not _env_flag("KT_STALL_WATCH", "0"):
        return None
    if _WATCHDOG is not None:
        return _WATCHDOG

    out = Path(
        os.environ.get("KT_STALL_WATCH_OUT_DIR")
        or os.environ.get("KT_STEP_TIMING_OUT_DIR")
        or "stall_watch_out"
    )
    phases_raw = os.environ.get("KT_STALL_WATCH_PHASES", "optimizer,backward,post_optim")
    phases = tuple(p.strip() for p in phases_raw.split(",") if p.strip())
    wd = StallWatchdog(
        out_dir=out,
        idle_sec=_env_float("KT_STALL_WATCH_IDLE_SEC", 5.0),
        poll_sec=_env_float("KT_STALL_WATCH_POLL_SEC", 1.0),
        cpu_idle_ratio=_env_float("KT_STALL_WATCH_CPU_BUSY_RATIO", 0.15),
        phases=phases,
    )
    wd.start()
    _WATCHDOG = wd
    _patch_trainer_for_stall_watch(wd)
    return wd


def _patch_trainer_for_stall_watch(wd: StallWatchdog) -> None:
    from transformers import Trainer

    # Prefer attaching when create_optimizer finishes (same timing as step_timing).
    if getattr(Trainer, "_kt_stall_watch_create_opt_patched", False):
        return

    orig_create_opt = Trainer.create_optimizer

    def create_optimizer_with_watch(self):
        out = orig_create_opt(self)
        _attach_to_trainer(self, wd)
        return out

    Trainer.create_optimizer = create_optimizer_with_watch
    Trainer._kt_stall_watch_create_opt_patched = True

    # Also wrap on_step_begin via a tiny callback to set step hint.
    original_init = Trainer.__init__
    if not getattr(Trainer, "_kt_stall_watch_callback_installed", False):

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:
                from transformers import TrainerCallback

                class _Hint(TrainerCallback):
                    def on_step_begin(self, args, state, control, **kw):
                        wd.set_step_hint(int(getattr(state, "global_step", 0) or 0) + 1)

                    def on_train_end(self, args, state, control, **kw):
                        wd.stop()

                self.add_callback(_Hint())
            except Exception as exc:
                print(f"[stall_watch] callback install failed: {exc}", flush=True)

        Trainer.__init__ = patched_init
        Trainer._kt_stall_watch_callback_installed = True


def _attach_to_trainer(trainer: Any, wd: StallWatchdog) -> None:
    if getattr(trainer, "_kt_stall_watch_attached", False):
        return
    opt = getattr(trainer, "optimizer", None)
    if opt is None:
        return

    # Wrap whatever is currently installed (may already be step_timing wrapper).
    if "optimizer" in wd.phases and not getattr(opt.step, "_kt_stall_wrapped", False):
        opt.step = wd.wrap("optimizer", opt.step)
        opt.step._kt_stall_wrapped = True

    if "post_optim" in wd.phases and hasattr(opt, "zero_grad"):
        if not getattr(opt.zero_grad, "_kt_stall_wrapped", False):
            opt.zero_grad = wd.wrap("post_optim", opt.zero_grad)
            opt.zero_grad._kt_stall_wrapped = True

    # backward: patch accelerator.backward once per training_step via Trainer.training_step
    from transformers import Trainer as Tr

    if "backward" in wd.phases and not getattr(Tr, "_kt_stall_bwd_patched", False):
        orig_ts = Tr.training_step

        def timed_ts(self, model, inputs, num_items_in_batch=None):
            if getattr(self, "accelerator", None) is None:
                return orig_ts(self, model, inputs, num_items_in_batch=num_items_in_batch)
            orig_bwd = self.accelerator.backward

            def bwd(*a, **k):
                wd.enter("backward")
                try:
                    return orig_bwd(*a, **k)
                finally:
                    wd.leave("backward")

            self.accelerator.backward = bwd
            try:
                return orig_ts(self, model, inputs, num_items_in_batch=num_items_in_batch)
            finally:
                self.accelerator.backward = orig_bwd

        Tr.training_step = timed_ts
        Tr._kt_stall_bwd_patched = True

    trainer._kt_stall_watch_attached = True
    print("[stall_watch] attached to trainer optimizer path", flush=True)

#!/usr/bin/env python3
"""
Qwen3-30B-A3B FFT 测试结果分析与可视化

功能：
  1. 解析 monitor.csv（GPU 显存、CPU RAM、CPU 利用率）
  2. 解析 LLaMA-Factory 训练日志（loss、grad_norm、学习率）
  3. 生成 7 张可视化图（保存到 <log_dir>/plots/）
  4. 生成中文 summary.md（合并原英文 SUMMARY 内容；不再保留 SUMMARY.md）

用法：
    python3 analyze.py --log-dir /path/to/test_log/YYYYMMDD_HHMMSS

图表列表：
    01_gpu_memory.png      - 各 GPU 显存占用时序
    02_cpu_ram.png         - CPU RAM 时序
    03_training_loss.png   - 训练 loss 曲线（各 Phase）
    04_grad_norm.png       - 梯度范数曲线（各 Phase）
    05_nan_inf_timeline.png- NaN/Inf/崩溃事件时间线
    06_phase_summary.png   - 各 Phase 性能对比表（步均耗时、峰值显存等）
    07_tps.png             - 逐步 TPS 曲线（phase4 为主，含 warmup 区间标注）
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[analyze] matplotlib not installed, skipping plots. pip install matplotlib", file=sys.stderr)

try:
    import pandas as pd
    _HAS_PD = True
except ImportError:
    _HAS_PD = False

# --------------------------------------------------------------------------- #
# 颜色定义（各 Phase 使用不同颜色）
# --------------------------------------------------------------------------- #
PHASE_COLORS = {
    "init":    "#888888",
    "phase1":  "#4e9af1",
    "phase2a": "#f1a94e",
    "phase2b": "#e05252",
    "phase4":  "#9b59b6",
    "phase5_p9": "#16a085",
    "phase5_p8": "#2c3e50",
}
GPU_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]

WARMUP_SKIP = 5  # 与 run_fft_test.sh 一致，计算稳定 TPS 时跳过前 N 步
STEP_TIME_PATTERN = re.compile(
    r"(\d+)/\d+\s+\[[\d:]+<[\d:]+,\s*([\d.]+)\s*s/it\]"
)


def find_train_log(phase_dir: Path) -> Path:
    """Return legacy train.log or a mode-labelled train_*.log."""
    legacy = phase_dir / "train.log"
    if legacy.exists():
        return legacy
    candidates = sorted(phase_dir.glob("train_*.log"))
    return candidates[0] if candidates else legacy


def infer_warmup_skip(phase_dir: Path) -> int:
    timing_path = phase_dir / "step_timing" / "step_timing.json"
    if timing_path.exists():
        try:
            return int(json.loads(timing_path.read_text()).get("warmup_skip", WARMUP_SKIP))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return WARMUP_SKIP if phase_dir.name == "phase4" else 0

# --------------------------------------------------------------------------- #
# 解析 monitor.csv
# --------------------------------------------------------------------------- #
def load_monitor_csv(csv_path: Path) -> dict:
    """
    返回 dict：
      {
        "elapsed": [float, ...],
        "phase": [str, ...],
        "event": [str, ...],
        "cpu_util": [float, ...],
        "ram_used_gb": [float, ...],          # 整机
        "proc_ram_gb": [float, ...],          # 进程树（若无则空）
        "ram_plot_gb": [float, ...],          # 出图/峰值优先用进程树
        "ram_total_gb": float,
        "has_proc_metrics": bool,
        "gpus": {
          0: {
            "mem_used": [...],       # 整卡 GB
            "proc_mem": [...],       # 进程树 GB（若无则空）
            "plot_mem": [...],       # 出图优先用进程树
            "mem_total": float,
            "sm_util": [...],
            "mem_util": [...],
          }, ...
        }
      }
    """
    if not csv_path.exists():
        return {}

    data: dict = {
        "elapsed": [],
        "phase": [],
        "event": [],
        "cpu_util": [],
        "ram_used_gb": [],
        "proc_ram_gb": [],
        "ram_plot_gb": [],
        "ram_total_gb": 0.0,
        "has_proc_metrics": False,
        "gpus": {},
    }

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return data

    has_proc_ram = "proc_ram_gb" in (rows[0] or {})
    data["has_proc_metrics"] = has_proc_ram

    for r in rows:
        def _f(k, default=0.0):
            try:
                return float(r.get(k, default) or default)
            except (ValueError, TypeError):
                return default

        data["elapsed"].append(_f("elapsed_sec"))
        data["phase"].append(r.get("phase", ""))
        data["event"].append(r.get("event", ""))
        data["cpu_util"].append(_f("cpu_util_pct"))
        ram_host = _f("ram_used_gb")
        data["ram_used_gb"].append(ram_host)
        if has_proc_ram:
            proc_ram = _f("proc_ram_gb")
            data["proc_ram_gb"].append(proc_ram)
            data["ram_plot_gb"].append(proc_ram)
        else:
            data["ram_plot_gb"].append(ram_host)

        rt = _f("ram_total_gb")
        if rt > data["ram_total_gb"]:
            data["ram_total_gb"] = rt

        # GPU 列（动态探测）
        for key in r:
            m = re.match(r"^gpu(\d+)_mem_used_mb$", key)
            if m:
                gi = int(m.group(1))
                if gi not in data["gpus"]:
                    total_key = f"gpu{gi}_mem_total_mb"
                    data["gpus"][gi] = {
                        "mem_used": [],
                        "proc_mem": [],
                        "plot_mem": [],
                        "mem_total": _f(total_key) / 1024,
                        "sm_util": [],
                        "mem_util": [],
                    }
                host_gb = _f(key) / 1024
                data["gpus"][gi]["mem_used"].append(host_gb)
                data["gpus"][gi]["sm_util"].append(_f(f"gpu{gi}_sm_util_pct"))
                data["gpus"][gi]["mem_util"].append(_f(f"gpu{gi}_mem_util_pct"))
                proc_key = f"proc_gpu{gi}_mem_mb"
                if proc_key in r and r.get(proc_key) not in (None, ""):
                    proc_gb = _f(proc_key) / 1024
                    data["gpus"][gi]["proc_mem"].append(proc_gb)
                    data["gpus"][gi]["plot_mem"].append(proc_gb)
                    data["has_proc_metrics"] = True
                else:
                    data["gpus"][gi]["plot_mem"].append(host_gb)

    return data


# --------------------------------------------------------------------------- #
# 解析 LLaMA-Factory 训练日志
# --------------------------------------------------------------------------- #
def load_trainer_logs(log_dir: Path) -> dict:
    """
    扫描 <log_dir>/phase*/train*.log 和 phase*/model_output/trainer_log.jsonl
    返回 {phase_name: {"steps": [], "loss": [], "grad_norm": [], "lr": [], "nan_lines": []}}
    """
    result = {}

    for phase_dir in sorted(log_dir.iterdir()):
        if not phase_dir.is_dir() or not phase_dir.name.startswith("phase"):
            continue
        phase = phase_dir.name

        steps, losses, grad_norms, lrs, nan_lines = [], [], [], [], []

        # 方式1：trainer_log.jsonl（LLaMA-Factory 标准输出）
        for jsonl_path in phase_dir.rglob("trainer_log.jsonl"):
            try:
                with jsonl_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        step = obj.get("current_steps") or obj.get("step")
                        if step is not None:
                            steps.append(int(step))
                            losses.append(float(obj.get("loss", 0) or 0))
                            grad_norms.append(float(obj.get("grad_norm", 0) or 0))
                            lrs.append(float(obj.get("learning_rate", 0) or 0))
            except Exception:
                pass

        # 方式2：从 train*.log 正则提取（fallback）
        train_log = find_train_log(phase_dir)
        if train_log.exists() and not steps:
            loss_pattern = re.compile(
                r"['\"]?loss['\"]?\s*[=:]\s*([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)",
                re.IGNORECASE,
            )
            gn_pattern = re.compile(
                r"['\"]?grad_norm['\"]?\s*[=:]\s*([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)",
                re.IGNORECASE,
            )
            lr_pattern = re.compile(
                r"['\"]?learning_rate['\"]?\s*[=:]\s*([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)",
                re.IGNORECASE,
            )
            step_pattern = re.compile(r"\bstep\s*[=:]\s*(\d+)\b", re.IGNORECASE)
            with train_log.open(errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if re.search(r"\bnan\b|\binf\b", line, re.IGNORECASE):
                        nan_lines.append(lineno)
                    lm = loss_pattern.search(line)
                    gm = gn_pattern.search(line)
                    sm = step_pattern.search(line)
                    lrm = lr_pattern.search(line)
                    if lm:
                        losses.append(float(lm.group(1)))
                        steps.append(int(sm.group(1)) if sm else len(steps))
                        grad_norms.append(float(gm.group(1)) if gm else float("nan"))
                        lrs.append(float(lrm.group(1)) if lrm else float("nan"))
        elif train_log.exists():
            with train_log.open(errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if re.search(r"\bnan\b|\binf\b", line, re.IGNORECASE):
                        nan_lines.append(lineno)

        result[phase] = {
            "steps": steps,
            "loss": losses,
            "grad_norm": grad_norms,
            "lr": lrs,
            "nan_lines": nan_lines,
        }

    return result


# --------------------------------------------------------------------------- #
# 读取各 Phase 退出码
# --------------------------------------------------------------------------- #
def load_exit_codes(log_dir: Path) -> dict[str, int]:
    codes = {}
    for ec_file in log_dir.glob("phase*/exit_code.txt"):
        phase = ec_file.parent.name
        try:
            codes[phase] = int(ec_file.read_text().strip())
        except Exception:
            codes[phase] = -1
    return codes


# --------------------------------------------------------------------------- #
# TPS 解析（从 tqdm 日志提取逐步耗时）
# --------------------------------------------------------------------------- #
def _parse_yaml_int(yaml_path: Path, key: str, default: int) -> int:
    if not yaml_path.exists():
        return default
    m = re.search(rf"^{re.escape(key)}:\s*(\d+)", yaml_path.read_text(), re.MULTILINE)
    return int(m.group(1)) if m else default


def infer_num_gpus(log_dir: Path) -> int:
    for name in ("summary.md", "SUMMARY.md"):
        summary = log_dir / name
        if summary.exists():
            text = summary.read_text()
            m = re.search(r"(?:GPU count|GPU 数量)\*\*:\s*(\d+)", text)
            if m:
                return int(m.group(1))
    for part in reversed(log_dir.parts):
        m = re.search(r"(?:^|_)(\d+)gpu(?:_|$)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 1


def parse_step_times_from_log(log_path: Path) -> list[tuple[int, float]]:
    """从 train.log 的 tqdm 行解析 (step, seconds_per_it)。"""
    text = log_path.read_text(errors="replace")
    seen: dict[int, float] = {}
    for m in STEP_TIME_PATTERN.finditer(text):
        seen[int(m.group(1))] = float(m.group(2))
    return sorted(seen.items())


def load_tps_data(log_dir: Path) -> dict:
    """
    返回 {phase: {steps, step_times, tps, tokens_per_step, stable_avg_tps, warmup_skip}}
    TPS = tokens_per_step / step_time
    tokens_per_step = num_gpus * cutoff_len * batch * grad_accum
    """
    num_gpus = infer_num_gpus(log_dir)
    result: dict = {}

    for phase_dir in sorted(log_dir.iterdir()):
        if not phase_dir.is_dir() or not phase_dir.name.startswith("phase"):
            continue
        train_log = find_train_log(phase_dir)
        if not train_log.exists():
            continue

        steps_times = parse_step_times_from_log(train_log)
        if not steps_times:
            continue

        cfg_path = phase_dir / "train_config.yaml"
        cutoff = _parse_yaml_int(cfg_path, "cutoff_len", 2048)
        batch = _parse_yaml_int(cfg_path, "per_device_train_batch_size", 1)
        accum = _parse_yaml_int(cfg_path, "gradient_accumulation_steps", 1)
        tokens_per_step = num_gpus * cutoff * batch * accum

        steps = [s for s, _ in steps_times]
        times = [t for _, t in steps_times]
        tps_vals = [tokens_per_step / t if t > 0 else 0.0 for t in times]

        warmup_skip = infer_warmup_skip(phase_dir)
        stable = [v for s, v in zip(steps, tps_vals) if s > warmup_skip]
        stable_avg = sum(stable) / len(stable) if stable else None

        result[phase_dir.name] = {
            "steps": steps,
            "step_times": times,
            "tps": tps_vals,
            "tokens_per_step": tokens_per_step,
            "stable_avg_tps": stable_avg,
            "warmup_skip": warmup_skip,
            "num_gpus": num_gpus,
            "cutoff_len": cutoff,
        }

    return result


# --------------------------------------------------------------------------- #
# 绘图工具
# --------------------------------------------------------------------------- #
def _add_phase_bands(ax, elapsed, phases, alpha=0.07):
    """在时间轴上添加 Phase 背景色带。"""
    if not elapsed or not phases:
        return
    phase_segs: list[tuple[float, float, str]] = []
    cur_phase = phases[0]
    seg_start = elapsed[0]
    for t, p in zip(elapsed, phases):
        if p != cur_phase:
            phase_segs.append((seg_start, t, cur_phase))
            cur_phase = p
            seg_start = t
    phase_segs.append((seg_start, elapsed[-1], cur_phase))

    for x0, x1, ph in phase_segs:
        color = PHASE_COLORS.get(ph, "#aaaaaa")
        ax.axvspan(x0, x1, alpha=alpha, color=color, linewidth=0)


def _add_event_vlines(ax, elapsed, events, color="red", alpha=0.7, label_prefix=""):
    """在时间轴上标注事件竖线。"""
    seen: set[str] = set()
    for t, ev in zip(elapsed, events):
        if not ev:
            continue
        if t not in seen:
            seen.add(t)
            ax.axvline(x=t, color=color, alpha=alpha, linewidth=1.2, linestyle="--")
            ax.text(t, ax.get_ylim()[1] * 0.95, label_prefix + ev,
                    fontsize=6, rotation=90, ha="right", va="top", color=color, alpha=0.8)


def _save_fig(fig, path: Path, title: str):
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path.name}")


# --------------------------------------------------------------------------- #
# 图1：GPU 显存时序
# --------------------------------------------------------------------------- #
def plot_gpu_memory(monitor: dict, plots_dir: Path):
    if not _HAS_MPL or not monitor.get("elapsed"):
        return
    elapsed = monitor["elapsed"]
    gpus = monitor.get("gpus", {})
    if not gpus:
        print("  -> no GPU data, skipping plot 1")
        return

    use_proc = monitor.get("has_proc_metrics", False) and any(
        gdata.get("proc_mem") for gdata in gpus.values()
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Sub-plot 1: VRAM usage (GB) — prefer process-tree when available
    ax = axes[0]
    for gi, gdata in sorted(gpus.items()):
        series = gdata.get("plot_mem") or gdata["mem_used"]
        label = f"GPU{gi}" + (" (proc)" if use_proc else "")
        ax.plot(elapsed, series, label=label, color=GPU_COLORS[gi % len(GPU_COLORS)], linewidth=1.2)
        if use_proc and gdata.get("mem_used"):
            ax.plot(
                elapsed, gdata["mem_used"],
                color=GPU_COLORS[gi % len(GPU_COLORS)], linewidth=0.7,
                alpha=0.25, linestyle="--",
            )
    if gpus:
        total = list(gpus.values())[0]["mem_total"]
        if total > 0:
            ax.axhline(y=total, color="gray", linestyle=":", linewidth=0.8, label=f"Total VRAM {total:.0f} GB")
    ax.set_ylabel("Memory Used (GB)")
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    _add_phase_bands(ax, elapsed, monitor.get("phase", []))

    # Sub-plot 2: SM utilization
    ax2 = axes[1]
    for gi, gdata in sorted(gpus.items()):
        ax2.plot(elapsed, gdata["sm_util"], label=f"GPU{gi} SM%",
                 color=GPU_COLORS[gi % len(GPU_COLORS)], linewidth=1.0, alpha=0.8)
    ax2.set_ylabel("SM Utilization (%)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", fontsize=7, ncol=4)
    ax2.grid(True, alpha=0.3)
    _add_phase_bands(ax2, elapsed, monitor.get("phase", []))

    title = "GPU VRAM Usage & SM Utilization\n(colored bands = phase intervals)"
    if use_proc:
        title = (
            "GPU VRAM Usage & SM Utilization\n"
            "solid = this job process tree; dashed = whole-card (neighbor noise)"
        )
    _save_fig(fig, plots_dir / "01_gpu_memory.png", title)


# --------------------------------------------------------------------------- #
# 图2：CPU RAM 时序
# --------------------------------------------------------------------------- #
def plot_cpu_ram(monitor: dict, plots_dir: Path):
    if not _HAS_MPL or not monitor.get("elapsed"):
        return
    elapsed = monitor["elapsed"]
    use_proc = bool(monitor.get("proc_ram_gb"))
    ram_plot = monitor.get("ram_plot_gb") or monitor.get("ram_used_gb", [])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    label = "Job process-tree RSS" if use_proc else "Used RAM (host)"
    ax.plot(elapsed, ram_plot, color="#e74c3c", linewidth=1.2, label=label)
    if use_proc and monitor.get("ram_used_gb"):
        ax.plot(
            elapsed, monitor["ram_used_gb"],
            color="#e74c3c", linewidth=0.8, alpha=0.3, linestyle="--",
            label="Host used RAM",
        )
    total = monitor.get("ram_total_gb", 0)
    if total > 0:
        ax.axhline(y=total, color="gray", linestyle=":", linewidth=0.8, label=f"Total RAM {total:.0f} GB")
    ax.set_ylabel("Memory (GB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_phase_bands(ax, elapsed, monitor.get("phase", []))

    ax2 = axes[1]
    ax2.plot(elapsed, monitor["cpu_util"], color="#3498db", linewidth=1.0, label="CPU Utilization")
    ax2.set_ylabel("CPU Utilization (%)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    _add_phase_bands(ax2, elapsed, monitor.get("phase", []))

    title = "CPU RAM & CPU Utilization\n(watch CPU spike during backward_base_weight_grad in full mode)"
    if use_proc:
        title = (
            "CPU RAM & CPU Utilization\n"
            "solid = this job process-tree RSS; dashed = whole-host used RAM"
        )
    _save_fig(fig, plots_dir / "02_cpu_ram.png", title)


# --------------------------------------------------------------------------- #
# 图3：训练 Loss 曲线
# --------------------------------------------------------------------------- #
def plot_training_loss(trainer_logs: dict, plots_dir: Path):
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    has_data = False
    for phase, data in sorted(trainer_logs.items()):
        if data["steps"] and data["loss"]:
            color = PHASE_COLORS.get(phase, "#888888")
            ax.plot(data["steps"], data["loss"], label=phase, color=color, linewidth=1.5, marker=".", markersize=3)
            has_data = True
    if not has_data:
        ax.text(0.5, 0.5, "No loss data found\n(training may not have run, or log format not matched)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Training Step")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    _save_fig(fig, plots_dir / "03_training_loss.png",
              "Training Loss Curve (per Phase)\n(log scale)")


# --------------------------------------------------------------------------- #
# 图4：梯度范数曲线
# --------------------------------------------------------------------------- #
def plot_grad_norm(trainer_logs: dict, plots_dir: Path):
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    has_data = False
    for phase, data in sorted(trainer_logs.items()):
        gn = [g for g in data.get("grad_norm", []) if g == g]  # 过滤 NaN
        if data["steps"] and gn:
            steps = data["steps"][:len(gn)]
            color = PHASE_COLORS.get(phase, "#888888")
            ax.plot(steps, gn, label=phase, color=color, linewidth=1.5, marker=".", markersize=3)
            has_data = True
    if not has_data:
        ax.text(0.5, 0.5, "No grad_norm data found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_ylabel("Gradient Norm")
    ax.set_xlabel("Training Step")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # P1 diagnostic annotation
    ax.text(0.02, 0.97,
            "P1 check: if phase2b(accum=4) norm ~= phase2a(accum=1) / 4  =>  grad overwrite bug confirmed",
            transform=ax.transAxes, fontsize=7, va="top", style="italic", color="#666666")

    _save_fig(fig, plots_dir / "04_grad_norm.png",
              "Gradient Norm Curve (per Phase)\n(P1: compare phase2a vs phase2b to detect grad overwrite bug)")


# --------------------------------------------------------------------------- #
# 图5：NaN/Inf/崩溃事件时间线
# --------------------------------------------------------------------------- #
def plot_nan_timeline(trainer_logs: dict, exit_codes: dict, plots_dir: Path):
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 4))

    phases = sorted(trainer_logs.keys())
    y_labels = []
    has_event = False

    for yi, phase in enumerate(phases):
        data = trainer_logs[phase]
        nan_lines = data.get("nan_lines", [])
        ec = exit_codes.get(phase, None)

        color = PHASE_COLORS.get(phase, "#888888")
        # 画基线
        ax.hlines(yi, 0, max(data.get("steps", [1]) or [1]),
                  colors=color, linewidth=2, alpha=0.5)

        # NaN 事件（用行号当 x）
        if nan_lines:
            ax.scatter(nan_lines, [yi] * len(nan_lines),
                       marker="x", color="red", s=60, zorder=5,
                       label=f"{phase}: NaN/Inf" if not has_event else "")
            has_event = True

        # 崩溃标记
        if ec is not None and ec != 0:
            max_step = max(data.get("steps", [10]) or [10])
            ax.scatter([max_step], [yi], marker="*", color="darkred", s=120, zorder=6)

        y_labels.append(f"{phase} (ec={ec})" if ec is not None else phase)

    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Step / Log Line Number")
    ax.set_title("NaN/Inf Locations (red x) & Crashes (red star)")
    ax.grid(True, alpha=0.3, axis="x")

    if not has_event:
        ax.text(0.5, 0.5, "No NaN/Inf events detected (numerically stable)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="green")

    _save_fig(fig, plots_dir / "05_nan_inf_timeline.png",
              "NaN/Inf/Crash Event Timeline\n(P5: C++ grad index bug | P6: Router grad explosion)")


# --------------------------------------------------------------------------- #
# 图6：各 Phase 性能对比
# --------------------------------------------------------------------------- #
def plot_phase_summary(monitor: dict, trainer_logs: dict, exit_codes: dict, plots_dir: Path):
    if not _HAS_MPL:
        return

    phases = ["phase1", "phase2a", "phase2b", "phase4", "phase5_p9", "phase5_p8"]
    # 收集指标
    peak_gpu_mem = {}
    peak_ram_gb = {}

    elapsed = monitor.get("elapsed", [])
    phase_list = monitor.get("phase", [])
    gpus = monitor.get("gpus", {})
    ram_used = monitor.get("ram_plot_gb") or monitor.get("ram_used_gb", [])

    for ph in phases:
        indices = [i for i, p in enumerate(phase_list) if p == ph]
        if not indices or not gpus:
            peak_gpu_mem[ph] = 0
        else:
            peak_gpu_mem[ph] = max(
                max(
                    (gdata.get("plot_mem") or gdata["mem_used"])[i]
                    for i in indices
                    if i < len(gdata.get("plot_mem") or gdata["mem_used"])
                )
                for gi, gdata in gpus.items()
            ) if gpus else 0

        if indices and ram_used:
            peak_ram_gb[ph] = max(ram_used[i] for i in indices if i < len(ram_used))
        else:
            peak_ram_gb[ph] = 0

    # 每步样本数
    step_counts = {ph: len(trainer_logs.get(ph, {}).get("steps", [])) for ph in phases}
    avg_loss = {}
    for ph in phases:
        losses = [l for l in trainer_logs.get(ph, {}).get("loss", []) if l == l and l > 0]
        avg_loss[ph] = sum(losses) / len(losses) if losses else 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x = range(len(phases))
    bar_colors = [PHASE_COLORS.get(p, "#888888") for p in phases]

    def _bar(ax, vals, title, ylabel, note=""):
        bars = ax.bar(phases, vals, color=bar_colors, alpha=0.8, edgecolor="white")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        if note:
            ax.text(0.01, 0.99, note, transform=ax.transAxes, fontsize=6,
                    va="top", style="italic", color="#555555")

    _bar(axes[0, 0], [peak_gpu_mem.get(p, 0) for p in phases],
         "Peak GPU VRAM (GB)", "GB",
         "Expert BF16 buffers reside on CPU\nGPU holds attention/embedding/norm only")

    _bar(axes[0, 1], [peak_ram_gb.get(p, 0) for p in phases],
         "Peak CPU RAM (GB)", "GB",
         "MoE expert weights/grads/optimizer on CPU")

    _bar(axes[1, 0], [step_counts.get(p, 0) for p in phases],
         "Steps Completed", "steps",
         "steps < expected => crash / DDP timeout")

    _bar(axes[1, 1], [avg_loss.get(p, 0) for p in phases],
         "Mean Loss", "Loss",
         "phase1-2 only 3-8 steps; high loss is expected")

    # Exit code annotation
    for i, ph in enumerate(phases):
        ec = exit_codes.get(ph)
        if ec is not None and ec != 0:
            for ax_row in axes:
                for ax in ax_row:
                    bars = ax.containers
                    ax.text(i, 0, f"ec={ec}", ha="center", va="bottom",
                            fontsize=6, color="red", fontweight="bold")

    _save_fig(fig, plots_dir / "06_phase_summary.png",
              "Phase Performance Summary\n(Peak VRAM | Peak RAM | Steps Completed | Mean Loss)")


# --------------------------------------------------------------------------- #
# 图7：逐步 TPS 曲线
# --------------------------------------------------------------------------- #
def plot_tps(tps_data: dict, plots_dir: Path):
    if not _HAS_MPL or not tps_data:
        print("  -> no TPS data (tqdm step times not found), skipping plot 7")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Sub-plot 1: 逐步耗时 (s/it)
    ax1 = axes[0]
    for phase, data in sorted(tps_data.items()):
        color = PHASE_COLORS.get(phase, "#888888")
        is_benchmark = phase == "phase4"
        ax1.plot(
            data["steps"], data["step_times"],
            label=f"{phase} ({data['tokens_per_step']} tok/step)",
            color=color,
            linewidth=2.0 if is_benchmark else 1.0,
            alpha=1.0 if is_benchmark else 0.65,
            marker=".", markersize=4,
        )
    ax1.set_ylabel("Step Time (s/it)")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Per-step training time from tqdm (lower = faster)", fontsize=9, loc="left")

    # Sub-plot 2: 逐步 TPS
    ax2 = axes[1]
    if "phase4" in tps_data:
        d4 = tps_data["phase4"]
        ws = d4["warmup_skip"]
        if ws > 0 and d4["steps"]:
            ax2.axvspan(0.5, ws + 0.5, alpha=0.12, color="#f39c12",
                        label=f"warmup (steps 1–{ws}, excluded from avg)")
        if d4["stable_avg_tps"] is not None:
            ax2.axhline(
                d4["stable_avg_tps"], color="#27ae60", linestyle="--", linewidth=1.5,
                label=f"phase4 stable avg: {d4['stable_avg_tps']:.1f} tok/s",
            )

    for phase, data in sorted(tps_data.items()):
        color = PHASE_COLORS.get(phase, "#888888")
        is_benchmark = phase == "phase4"
        ax2.plot(
            data["steps"], data["tps"],
            label=phase,
            color=color,
            linewidth=2.2 if is_benchmark else 1.0,
            alpha=1.0 if is_benchmark else 0.65,
            marker=".", markersize=4,
        )

    ax2.set_ylabel("TPS (tokens/sec)")
    ax2.set_xlabel("Training Step")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # 标注 phase4 峰值 TPS
    if "phase4" in tps_data:
        d4 = tps_data["phase4"]
        stable_steps = [(s, t) for s, t in zip(d4["steps"], d4["tps"]) if s > d4["warmup_skip"]]
        if stable_steps:
            peak_step, peak_tps = max(stable_steps, key=lambda x: x[1])
            ax2.annotate(
                f"peak {peak_tps:.0f} tok/s @ step {peak_step}",
                xy=(peak_step, peak_tps),
                xytext=(peak_step + 2, peak_tps + 15),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
            )

    _save_fig(
        fig, plots_dir / "07_tps.png",
        "Training Throughput (TPS)\n"
        f"TPS = num_gpus × cutoff_len × batch × accum / step_time  "
        f"(warmup skip={WARMUP_SKIP} for phase4 avg)",
    )


# --------------------------------------------------------------------------- #
# 生成 summary.md（仅中文；合并原 SUMMARY.md 内容）
# --------------------------------------------------------------------------- #
def generate_summary_md(
    log_dir: Path,
    monitor: dict,
    trainer_logs: dict,
    exit_codes: dict,
    tps_data: dict | None = None,
):
    summary_path = log_dir / "summary.md"
    legacy = log_dir / "SUMMARY.md"
    if legacy.exists():
        try:
            legacy.unlink()
        except Exception:
            pass

    lines = [
        "# Qwen3-30B-A3B FFT 测试报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**日志目录**: `{log_dir}`",
        f"**运行标签**: `{log_dir.name}`",
        "",
        "---",
        "",
        "## 1. Phase 执行结果",
        "",
        "| Phase | 描述 | 完成步数 | 最终 Loss | 退出码 | 状态 |",
        "|-------|------|----------|-----------|--------|------|",
    ]

    phase_descs = {
        "phase1": "基础验证（3步）",
        "phase2a": "梯度累积基准 accum=1（4步）",
        "phase2b": "梯度累积压力 accum=4（8步）",
        "phase4": "稳定性延伸 + TPS（phase4）",
        "phase5_p9": "P9 buffer overflow 压力（batch=2）",
        "phase5_p8": "P8 梯度方向验证（10步）",
    }

    for ph, desc in phase_descs.items():
        ec = exit_codes.get(ph, None)
        tl = trainer_logs.get(ph, {})
        steps_done = len(tl.get("steps", []))
        losses = [l for l in tl.get("loss", []) if l == l and l > 0]
        last_loss = f"{losses[-1]:.4f}" if losses else "N/A"
        ec_str = str(ec) if ec is not None else "跳过"
        status = "✓" if ec == 0 else ("⚠ 跳过" if ec is None else f"✗ 失败({ec})")
        lines.append(f"| {ph} | {desc} | {steps_done} | {last_loss} | {ec_str} | {status} |")

    lines += ["", "---", "", "## 2. 关键资源峰值", ""]
    if monitor.get("elapsed"):
        use_proc = monitor.get("has_proc_metrics", False)
        scope = "本任务进程树" if use_proc else "整机"
        gpus = monitor.get("gpus", {})
        if gpus:
            lines.append(f"**GPU 显存峰值（{scope}）:**")
            for gi, gd in sorted(gpus.items()):
                series = gd.get("plot_mem") or gd["mem_used"]
                peak = max(series) if series else 0
                total = gd["mem_total"]
                if total:
                    lines.append(f"  - GPU{gi}: {peak:.1f} / {total:.0f} GB ({peak/total*100:.0f}%)")
                else:
                    lines.append(f"  - GPU{gi}: {peak:.1f} GB")
                if use_proc and gd.get("mem_used"):
                    host_peak = max(gd["mem_used"])
                    lines.append(f"    （整卡对照峰值: {host_peak:.1f} GB）")
        ram = monitor.get("ram_plot_gb") or monitor.get("ram_used_gb", [])
        if ram:
            lines.append(
                f"\n**CPU RAM 峰值（{scope}）**: {max(ram):.1f} GB"
                + (f" / {monitor.get('ram_total_gb', 0):.0f} GB host total" if monitor.get("ram_total_gb") else "")
            )
            if use_proc and monitor.get("ram_used_gb"):
                lines.append(f"  （整机 used 对照峰值: {max(monitor['ram_used_gb']):.1f} GB）")
        cpu = monitor.get("cpu_util", [])
        if cpu:
            lines.append(f"\n**CPU 利用率峰值**: {max(cpu):.0f}%")

    lines += ["", "**训练吞吐 (TPS):**"]
    if tps_data and "phase4" in tps_data:
        import statistics as _stats

        d4 = tps_data["phase4"]
        lines.append(
            f"  - tokens/step: {d4['tokens_per_step']} "
            f"({d4['num_gpus']} GPU × {d4['cutoff_len']} tokens)"
        )
        stable_n = len([s for s in d4["steps"] if s > d4["warmup_skip"]])
        lines.append(f"  - 稳定步数: {stable_n}（跳过 warmup {d4['warmup_skip']}）")
        stable_times = [t for s, t in zip(d4["steps"], d4["step_times"]) if s > d4["warmup_skip"]]
        if stable_times:
            avg_t = sum(stable_times) / len(stable_times)
            med_t = _stats.median(stable_times)
            tok = d4["tokens_per_step"]
            lines.append(
                f"  - 步时 mean/median/min/max: "
                f"{avg_t:.2f} / {med_t:.2f} / {min(stable_times):.2f} / {max(stable_times):.2f} s"
            )
            lines.append(
                f"  - TPS mean/median/peak: "
                f"**{tok/avg_t:.1f}** / {tok/med_t:.1f} / {tok/min(stable_times):.1f} tok/s"
            )
        elif d4.get("stable_avg_tps") is not None:
            lines.append(f"  - 稳定 TPS: **{d4['stable_avg_tps']:.1f}** tok/s")
    else:
        p4a = log_dir / "phase4_analysis.txt"
        if p4a.exists():
            for ln in p4a.read_text(errors="replace").splitlines():
                if any(k in ln for k in ("TPS", "tokens_per_step", "stable_steps", "step_time")):
                    lines.append(f"  {ln.strip()}")
        else:
            lines.append("  （无 TPS 数据）")

    lines += ["", "---", "", "## 3. 内存归因", ""]
    lines.append(f"- 静态分量估计: `{log_dir / 'memory_component_estimate.txt'}`")
    lines.append(f"- 运行时时间线: `{log_dir / 'memory_component_timeline.csv'}`")
    lines.append(f"- 观测摘要: `{log_dir / 'memory_component_observed.txt'}`")
    lines.append("")
    obs = log_dir / "memory_component_observed.txt"
    if obs.exists():
        for ln in obs.read_text(errors="replace").splitlines():
            if re.search(r"peak_|baseline_|timeline_csv", ln):
                lines.append(f"  {ln.strip()}")

    lines += [
        "",
        "---",
        "",
        "## 4. 关键问题检测",
        "",
        "| 编号 | 问题 | 检测结论 |",
        "|------|------|---------|",
    ]

    p2a_gn = [g for g in trainer_logs.get("phase2a", {}).get("grad_norm", []) if g == g]
    p2b_gn = [g for g in trainer_logs.get("phase2b", {}).get("grad_norm", []) if g == g]
    if p2a_gn and p2b_gn:
        avg_2a = sum(p2a_gn) / len(p2a_gn)
        avg_2b = sum(p2b_gn) / len(p2b_gn)
        ratio = avg_2b / avg_2a if avg_2a > 0 else 0
        if ratio < 0.4:
            p1_result = (
                f"⚠ **确认 P1**: accum=4 梯度均值 {avg_2b:.3f} ≈ accum=1 的 {ratio:.2f}× "
                f"(期望 ≈1.0x)，梯度被覆盖"
            )
        elif 0.4 <= ratio < 0.8:
            p1_result = f"⚠ 疑似 P1: 梯度比值 {ratio:.2f}，低于理论值 1.0"
        else:
            p1_result = f"✓ P1 未触发: 梯度比值 {ratio:.2f} ≈ 正常"
    else:
        p1_result = "数据不足（Phase 2 未运行或无 grad_norm）"
    lines.append(f"| P1 | 梯度累积覆盖 bug | {p1_result} |")

    p4_analysis = log_dir / "phase4_analysis.txt"
    p2_result = "见 phase4_analysis.txt"
    if p4_analysis.exists():
        text = p4_analysis.read_text()
        m = re.search(r"avg_sec_per_step[:\s]+([\d.]+)", text)
        if m:
            sec = float(m.group(1))
            if sec > 300:
                p2_result = f"**严重**: 平均 {sec:.0f}s/step，DDP 超时风险高"
            elif sec > 120:
                p2_result = f"⚠ 平均 {sec:.0f}s/step，疑似 CPU backward 瓶颈"
            else:
                p2_result = f"平均 {sec:.0f}s/step（可接受）"
        else:
            m2 = re.search(r"step_time_avg\s*:\s*([\d.]+)", text)
            if m2:
                p2_result = f"稳定步均 {float(m2.group(1)):.1f}s（见 phase4_analysis）"
    lines.append(f"| P2 | CPU backward 速度瓶颈 | {p2_result} |")

    p4_log = find_train_log(log_dir / "phase4")
    p4_moe = "无专项数据（需路由统计 hook）"
    if p4_log.exists():
        text = p4_log.read_text(errors="replace")
        if re.search(r"aux_loss|balance_loss|router_loss", text, re.IGNORECASE):
            p4_moe = f"检测到路由辅助损失，见 phase4/{p4_log.name}"
        elif re.search(r"expert.*token|token.*expert", text, re.IGNORECASE):
            p4_moe = f"检测到 expert 路由日志，见 phase4/{p4_log.name}"
    lines.append(f"| P4 | MoE 路由负载均衡 | {p4_moe} |")

    p5_results = []
    for ph in phase_descs:
        tl = trainer_logs.get(ph, {})
        if tl.get("nan_lines"):
            p5_results.append(f"{ph}(行:{tl['nan_lines'][:3]})")
        # The phase exit-code file contains the outer accelerate launcher status
        # (usually 1), while torch elastic records the crashed child as -11 in
        # train.log.  Detect SIGSEGV from the raw log regardless of launcher code.
        log_f = find_train_log(log_dir / ph)
        if log_f.exists() and re.search(
            r"sigsegv|segfault|segmentation fault|signal\s+11|core dump(?:ed)?",
            log_f.read_text(errors="replace"),
            re.IGNORECASE,
        ):
            p5_results.append(f"{ph}:SIGSEGV")
    p5_result = "、".join(p5_results) if p5_results else "✓ 未检测到 NaN/Inf/崩溃"
    if p5_results:
        p5_result = "⚠ " + p5_result
    lines.append(f"| P5 | C++ 梯度索引 bug / NaN | {p5_result} |")
    lines.append("| P6 | Router 梯度稳定性 | 见 `plots/04_grad_norm.png`（full 模式 Router 参与梯度） |")

    p7_count = 0
    if p4_log.exists():
        text = p4_log.read_text(errors="replace")
        p7_count = sum(
            1
            for line in text.splitlines()
            if re.search(r"update_base_weights|re-quantize|syncing updated|online quant", line, re.IGNORECASE)
        )
    lines.append(f"| P7 | update_base_weights / online requant | phase4 相关日志命中 {p7_count} 次 |")

    lines += ["", "---", ""]
    out_dir = log_dir / "phase4" / "step_timing"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from step_timing_probe import (
            parse_job_timing_from_train_log,
            render_summary_timing_section,
        )

        step_path = out_dir / "step_timing.json"
        step_timing = json.loads(step_path.read_text()) if step_path.exists() else None
        job_path = out_dir / "job_timing.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        if job_path.exists():
            job_timing = json.loads(job_path.read_text())
        else:
            job_timing = parse_job_timing_from_train_log(find_train_log(log_dir / "phase4"))
            job_path.write_text(json.dumps(job_timing, indent=2, ensure_ascii=False))

        lines.append("## 5. TPS 瓶颈拆解与计时文件")
        lines.append("")
        lines.append(f"- 逐步 JSON: `{out_dir / 'step_timing.json'}`")
        lines.append(f"- 逐步 CSV: `{out_dir / 'step_timing.csv'}`")
        lines.append(f"- 逐步 Markdown: `{out_dir / 'step_timing.md'}`")
        lines.append(f"- Job 计时 JSON: `{out_dir / 'job_timing.json'}`")
        lines.append(f"- 停顿检测: `{out_dir / 'stall_events.md'}`")
        lines.append("")
        lines.append(render_summary_timing_section(step_timing, job_timing).rstrip())
    except Exception as exc:
        lines += [
            "## 5. TPS 瓶颈拆解与计时文件",
            "",
            f"- 生成失败: {exc}",
            "- 需 `KT_STEP_TIMING=1` 重跑后查看 `phase4/step_timing/`",
        ]

    stall_path = out_dir / "stall_events.json"
    lines += ["", "### 停顿检测（Stall Watch）", ""]
    if stall_path.exists():
        try:
            stall = json.loads(stall_path.read_text())
            lines.append(f"- 状态: {stall.get('status')}，事件数: {stall.get('num_events', 0)}")
            for i, ev in enumerate(stall.get("events") or [], 1):
                vm = ev.get("vmstat_delta") or {}
                top = sorted(vm.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                top_s = "，".join(f"{k}={v}" for k, v in top) if top else "无 vmstat 增量"
                lines.append(
                    f"- 事件{i}: 阶段=`{ev.get('phase')}` step≈{ev.get('global_step_hint')} "
                    f"墙钟={ev.get('wall_sec'):.1f}s CPU增量={ev.get('cpu_delta_sec')} "
                    f"空闲比={ev.get('idle_ratio')}｜{top_s}"
                )
            if not stall.get("events"):
                lines.append("- 本轮未检测到停顿事件")
        except Exception as exc:
            lines.append(f"- 解析失败: {exc}")
    else:
        lines.append("- 无 `stall_events.json`（需 `KT_STALL_WATCH=1`）")

    lines += ["", "---", "", "## 6. Expert 梯度与基座权重检查", ""]
    lines += ["### 6.1 optimizer 前专家梯度", ""]
    lines.append("- 方法: 完整扫描每层 `grad_gate/up/down_proj_buf` 及对应 `Parameter.grad`")
    lines.append(f"- 文本报告: `{log_dir / 'expert_gradient_check.txt'}`")
    lines.append(f"- JSON 报告: `{log_dir / 'expert_gradient_check.json'}`")
    lines.append(f"- 原始 probe: `{log_dir / 'phase4' / 'expert_grad_probe.json'}`")
    grad_report = log_dir / "expert_gradient_check.json"
    if grad_report.exists():
        try:
            grad_data = json.loads(grad_report.read_text())
            lines.append(f"- 状态: **{grad_data.get('status', 'UNKNOWN')}**")
            if grad_data.get("reason"):
                lines.append(f"- 原因: {grad_data['reason']}")
            for step in grad_data.get("steps") or []:
                grad_summary = step.get("summary") or {}
                lines.append(
                    f"- Step {step.get('optimizer_step')}: LR={step.get('learning_rates', [])}，"
                    f"非零 C++ buffer={grad_summary.get('nonzero_grad_buffers', 0)}/"
                    f"{grad_summary.get('ok_records', 0)}，"
                    f"非零 Parameter.grad={grad_summary.get('nonzero_parameter_grads', 0)}/"
                    f"{grad_summary.get('ok_records', 0)}，"
                    f"非有限={grad_summary.get('nonfinite_grad_buffers', 0) + grad_summary.get('nonfinite_parameter_grads', 0)}"
                )
        except Exception as exc:
            lines.append(f"- 解析失败: {exc}")
    else:
        lines.append("- （未运行 expert 梯度检查）")

    lines += ["", "### 6.2 Expert 基座权重变化", ""]
    lines.append("- 方法: 训练中采样 KT `gate/up/down_proj_buf`（非 HF checkpoint）")
    lines.append(f"- 文本报告: `{log_dir / 'expert_weight_change_check.txt'}`")
    lines.append(f"- JSON 报告: `{log_dir / 'expert_weight_change_check.json'}`")
    lines.append(f"- 原始 probe: `{log_dir / 'phase4' / 'expert_buf_probe.json'}`")
    lines.append("")
    ew = log_dir / "expert_weight_change_check.json"
    if ew.exists():
        try:
            data = json.loads(ew.read_text())
            agg = data.get("aggregate") or {}
            status = data.get("status", "UNKNOWN")
            status_cn = {"OK": "通过", "FAIL": "未通过", "ERROR": "错误", "SKIPPED": "跳过"}.get(status, status)
            lines.append(f"- 状态: **{status_cn}**（{status}）")
            reason = data.get("reason", "")
            if reason:
                reason_cn = reason
                if "none of the sampled" in reason:
                    reason_cn = "抽样的 expert 基座 buffer 均未超过 atol 变化"
                elif "probe JSON missing" in reason:
                    reason_cn = "缺少 in-training probe JSON（callback 未安装或训练未到 train_end）"
                lines.append(f"- 原因: {reason_cn}")
            lines.append(f"- 方法: {data.get('method', 'unknown')}")
            lines.append(f"- 抽样张量: {data.get('sampled_tensors', 0)}")
            lines.append(f"- 发生变化: {data.get('changed_tensors', 0)}")
            lines.append(f"- 非有限值: {data.get('nonfinite_tensors', 0)}")
            if agg:
                lines.append(f"- changed_tensor_fraction: {agg.get('changed_tensor_fraction', 0.0):.6e}")
                lines.append(f"- mean_rel_l2_delta: {agg.get('mean_rel_l2_delta', 0.0):.6e}")
                lines.append(f"- max_abs_delta: {agg.get('max_abs_delta', 0.0):.6e}")
        except Exception as exc:
            lines.append(f"- 解析失败: {exc}")
    else:
        lines.append("- （未运行 expert 权重检查）")

    lines += [
        "",
        "---",
        "",
        "## 7. 可视化图表",
        "",
        "| 文件 | 内容 |",
        "|------|------|",
        "| `plots/01_gpu_memory.png` | GPU 显存占用 & SM 利用率时序（优先进程树） |",
        "| `plots/02_cpu_ram.png` | CPU RAM & CPU 利用率时序（优先进程树 RSS） |",
        "| `plots/03_training_loss.png` | 训练 Loss 曲线（各 Phase） |",
        "| `plots/04_grad_norm.png` | 梯度范数曲线（含 P1 诊断说明） |",
        "| `plots/05_nan_inf_timeline.png` | NaN/Inf/崩溃事件时间线 |",
        "| `plots/06_phase_summary.png` | 各 Phase 性能对比 |",
        "| `plots/07_tps.png` | 逐步 TPS 曲线（phase4 稳定吞吐 + warmup 标注） |",
        "",
        "---",
        "",
        "## 8. 重新运行分析",
        "",
        "```bash",
        f"python3 {Path(__file__).resolve()} --log-dir {log_dir}",
        "```",
        "",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → 已保存 summary: {summary_path}")
    return summary_path


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="FFT 测试结果分析与可视化")
    parser.add_argument("--log-dir", required=True, help="测试日志目录（含 monitor.csv 和 phase* 子目录）")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.exists():
        print(f"[analyze] 目录不存在: {log_dir}", file=sys.stderr)
        sys.exit(1)

    plots_dir = log_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"[analyze] log_dir : {log_dir}")
    print(f"[analyze] plots_dir: {plots_dir}")

    # Load data
    print("[analyze] loading monitor.csv ...")
    monitor = load_monitor_csv(log_dir / "monitor.csv")

    print("[analyze] loading trainer logs ...")
    trainer_logs = load_trainer_logs(log_dir)

    print("[analyze] loading exit codes ...")
    exit_codes = load_exit_codes(log_dir)

    print("[analyze] parsing TPS from tqdm logs ...")
    tps_data = load_tps_data(log_dir)

    print(f"[analyze] phases found   : {list(trainer_logs.keys())}")
    print(f"[analyze] exit codes     : {exit_codes}")
    print(f"[analyze] TPS phases     : {list(tps_data.keys())}")
    print(f"[analyze] monitor samples: {len(monitor.get('elapsed', []))}")

    # Generate plots
    if _HAS_MPL:
        print("[analyze] plot 1: GPU VRAM ...")
        plot_gpu_memory(monitor, plots_dir)

        print("[analyze] plot 2: CPU RAM ...")
        plot_cpu_ram(monitor, plots_dir)

        print("[analyze] plot 3: training loss ...")
        plot_training_loss(trainer_logs, plots_dir)

        print("[analyze] plot 4: grad norm ...")
        plot_grad_norm(trainer_logs, plots_dir)

        print("[analyze] plot 5: NaN/Inf timeline ...")
        plot_nan_timeline(trainer_logs, exit_codes, plots_dir)

        print("[analyze] plot 6: phase summary ...")
        plot_phase_summary(monitor, trainer_logs, exit_codes, plots_dir)

        print("[analyze] plot 7: TPS ...")
        plot_tps(tps_data, plots_dir)
    else:
        print("[analyze] matplotlib not installed, skipping plots")

    # Generate summary.md
    print("[analyze] generating summary.md ...")
    summary_path = generate_summary_md(log_dir, monitor, trainer_logs, exit_codes, tps_data)

    print(f"\n[analyze] done.")
    print(f"  plots  : {plots_dir}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()

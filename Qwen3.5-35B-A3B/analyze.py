#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B FFT 测试结果分析与可视化

功能：
  1. 解析 monitor.csv（GPU 显存、CPU RAM、磁盘 I/O、CPU 利用率）
  2. 解析 LLaMA-Factory 训练日志（loss、grad_norm、学习率）
  3. 生成 7 张可视化图（保存到 <log_dir>/plots/）
  4. 生成 summary.md（关键结论 + 问题检测报告）

用法：
    python3 analyze.py --log-dir /path/to/test_log/YYYYMMDD_HHMMSS

图表列表：
    01_gpu_memory.png      - 各 GPU 显存占用时序
    02_cpu_ram.png         - CPU RAM 时序
    03_disk_throughput.png - 磁盘读写吞吐时序（含 checkpoint 事件标注）
    04_training_loss.png   - 训练 loss 曲线（各 Phase）
    05_grad_norm.png       - 梯度范数曲线（各 Phase）
    06_nan_inf_timeline.png- NaN/Inf/崩溃事件时间线
    07_phase_summary.png   - 各 Phase 性能对比表（步均耗时、峰值显存等）
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
    from matplotlib.patches import Patch
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
    "phase3":  "#7db87d",
    "phase4":  "#9b59b6",
}
GPU_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]

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
        "ram_used_gb": [float, ...],
        "ram_total_gb": float,
        "disk_read_mbps": [float, ...],
        "disk_write_mbps": [float, ...],
        "gpus": {0: {"mem_used": [...], "mem_total": float, "sm_util": [...], "mem_util": [...]}, ...}
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
        "ram_total_gb": 0.0,
        "disk_read_mbps": [],
        "disk_write_mbps": [],
        "gpus": {},
    }

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return data

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
        data["ram_used_gb"].append(_f("ram_used_gb"))
        data["disk_read_mbps"].append(_f("disk_read_mbps"))
        data["disk_write_mbps"].append(_f("disk_write_mbps"))

        rt = _f("ram_total_gb")
        if rt > data["ram_total_gb"]:
            data["ram_total_gb"] = rt

        # GPU 列（动态探测）
        for key in r:
            m = re.match(r"gpu(\d+)_mem_used_mb", key)
            if m:
                gi = int(m.group(1))
                if gi not in data["gpus"]:
                    total_key = f"gpu{gi}_mem_total_mb"
                    data["gpus"][gi] = {
                        "mem_used": [],
                        "mem_total": _f(total_key) / 1024,
                        "sm_util": [],
                        "mem_util": [],
                    }
                data["gpus"][gi]["mem_used"].append(_f(key) / 1024)
                data["gpus"][gi]["sm_util"].append(_f(f"gpu{gi}_sm_util_pct"))
                data["gpus"][gi]["mem_util"].append(_f(f"gpu{gi}_mem_util_pct"))

    return data


# --------------------------------------------------------------------------- #
# 解析 LLaMA-Factory 训练日志
# --------------------------------------------------------------------------- #
def load_trainer_logs(log_dir: Path) -> dict:
    """
    扫描 <log_dir>/phase*/train.log 和 phase*/model_output/trainer_log.jsonl
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

        # 方式2：从 train.log 正则提取（fallback）
        train_log = phase_dir / "train.log"
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

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Sub-plot 1: VRAM usage (GB)
    ax = axes[0]
    for gi, gdata in sorted(gpus.items()):
        ax.plot(elapsed, gdata["mem_used"], label=f"GPU{gi}", color=GPU_COLORS[gi % len(GPU_COLORS)], linewidth=1.2)
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

    _save_fig(fig, plots_dir / "01_gpu_memory.png",
              "GPU VRAM Usage & SM Utilization\n(colored bands = phase intervals)")


# --------------------------------------------------------------------------- #
# 图2：CPU RAM 时序
# --------------------------------------------------------------------------- #
def plot_cpu_ram(monitor: dict, plots_dir: Path):
    if not _HAS_MPL or not monitor.get("elapsed"):
        return
    elapsed = monitor["elapsed"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    ax.plot(elapsed, monitor["ram_used_gb"], color="#e74c3c", linewidth=1.2, label="Used RAM")
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

    _save_fig(fig, plots_dir / "02_cpu_ram.png",
              "CPU RAM & CPU Utilization\n(watch CPU spike during backward_base_weight_grad in full mode)")


# --------------------------------------------------------------------------- #
# 图3：磁盘吞吐时序（含 checkpoint 事件标注）
# --------------------------------------------------------------------------- #
def plot_disk_throughput(monitor: dict, plots_dir: Path):
    if not _HAS_MPL or not monitor.get("elapsed"):
        return
    elapsed = monitor["elapsed"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(elapsed, monitor["disk_write_mbps"], alpha=0.4, color="#e74c3c", label="Write (MB/s)")
    ax.fill_between(elapsed, monitor["disk_read_mbps"], alpha=0.4, color="#3498db", label="Read (MB/s)")
    ax.plot(elapsed, monitor["disk_write_mbps"], color="#e74c3c", linewidth=1.0)
    ax.plot(elapsed, monitor["disk_read_mbps"], color="#3498db", linewidth=1.0)

    # Mark checkpoint events
    events = monitor.get("event", [])
    for t, ev in zip(elapsed, events):
        if "checkpoint" in ev.lower():
            ax.axvline(x=t, color="orange", linewidth=1.5, linestyle="--", alpha=0.8)
            ax.text(t, ax.get_ylim()[1] * 0.85, "ckpt", fontsize=7,
                    rotation=90, ha="right", color="orange")

    ax.set_ylabel("Disk I/O Rate (MB/s)")
    ax.set_xlabel("Time (seconds)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _add_phase_bands(ax, elapsed, monitor.get("phase", []))

    # Phase legend patches
    legend_patches = [Patch(color=c, alpha=0.3, label=ph)
                      for ph, c in PHASE_COLORS.items() if ph != "init"]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_patches,
              labels=ax.get_legend_handles_labels()[1] + list(PHASE_COLORS.keys())[1:],
              fontsize=7, ncol=4, loc="upper left")

    _save_fig(fig, plots_dir / "03_disk_throughput.png",
              "Disk I/O Throughput\n(P3: write peak during checkpoint save; orange lines = ckpt events)")


# --------------------------------------------------------------------------- #
# 图4：训练 Loss 曲线
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

    _save_fig(fig, plots_dir / "04_training_loss.png",
              "Training Loss Curve (per Phase)\n(log scale)")


# --------------------------------------------------------------------------- #
# 图5：梯度范数曲线
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

    _save_fig(fig, plots_dir / "05_grad_norm.png",
              "Gradient Norm Curve (per Phase)\n(P1: compare phase2a vs phase2b to detect grad overwrite bug)")


# --------------------------------------------------------------------------- #
# 图6：NaN/Inf/崩溃事件时间线
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

    _save_fig(fig, plots_dir / "06_nan_inf_timeline.png",
              "NaN/Inf/Crash Event Timeline\n(P5: C++ grad index bug | P6: Router grad explosion)")


# --------------------------------------------------------------------------- #
# 图7：各 Phase 性能对比
# --------------------------------------------------------------------------- #
def plot_phase_summary(monitor: dict, trainer_logs: dict, exit_codes: dict, plots_dir: Path):
    if not _HAS_MPL:
        return

    phases = ["phase1", "phase2a", "phase2b", "phase3", "phase4"]
    # 收集指标
    peak_gpu_mem = {}
    peak_disk_write = {}

    elapsed = monitor.get("elapsed", [])
    phase_list = monitor.get("phase", [])
    gpus = monitor.get("gpus", {})
    disk_write = monitor.get("disk_write_mbps", [])

    for ph in phases:
        indices = [i for i, p in enumerate(phase_list) if p == ph]
        if not indices or not gpus:
            peak_gpu_mem[ph] = 0
        else:
            peak_gpu_mem[ph] = max(
                max(gpus[gi]["mem_used"][i] for i in indices if i < len(gpus[gi]["mem_used"]))
                for gi in gpus
            ) if gpus else 0

        if indices and disk_write:
            peak_disk_write[ph] = max(disk_write[i] for i in indices if i < len(disk_write))
        else:
            peak_disk_write[ph] = 0

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

    _bar(axes[0, 1], [peak_disk_write.get(p, 0) for p in phases],
         "Peak Disk Write (MB/s)", "MB/s",
         "P3: phase3 should spike (checkpoint save)\nphase3 >> others => P3 confirmed")

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

    _save_fig(fig, plots_dir / "07_phase_summary.png",
              "Phase Performance Summary\n(Peak VRAM | Disk Write | Steps Completed | Mean Loss)")


# --------------------------------------------------------------------------- #
# 生成 summary.md
# --------------------------------------------------------------------------- #
def generate_summary_md(
    log_dir: Path,
    monitor: dict,
    trainer_logs: dict,
    exit_codes: dict,
):
    summary_path = log_dir / "summary.md"

    lines = [
        "# Qwen3.5-35B-A3B FFT 测试 - 自动分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**日志目录**: `{log_dir}`",
        "",
        "---",
        "",
        "## 1. Phase 执行结果",
        "",
        "| Phase | 描述 | 完成步数 | 最终 Loss | 退出码 | 状态 |",
        "|-------|------|----------|-----------|--------|------|",
    ]

    phase_descs = {
        "phase1":  "基础验证（3步）",
        "phase2a": "梯度累积基准 accum=1（4步）",
        "phase2b": "梯度累积压力 accum=4（8步）",
        "phase3":  "高频保存 I/O（6步，save_steps=2）",
        "phase4":  "稳定性延伸（50步）",
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

    # 峰值资源
    lines += [
        "",
        "---",
        "",
        "## 2. 关键资源峰值",
        "",
    ]
    if monitor.get("elapsed"):
        gpus = monitor.get("gpus", {})
        if gpus:
            lines.append("**GPU 显存峰值:**")
            for gi, gd in sorted(gpus.items()):
                peak = max(gd["mem_used"]) if gd["mem_used"] else 0
                total = gd["mem_total"]
                lines.append(f"  - GPU{gi}: {peak:.1f} / {total:.0f} GB ({peak/total*100:.0f}%)" if total else f"  - GPU{gi}: {peak:.1f} GB")
        ram = monitor.get("ram_used_gb", [])
        if ram:
            lines.append(f"\n**CPU RAM 峰值**: {max(ram):.1f} GB / {monitor.get('ram_total_gb', 0):.0f} GB")
        disk_w = monitor.get("disk_write_mbps", [])
        if disk_w:
            lines.append(f"\n**磁盘写入峰值**: {max(disk_w):.0f} MB/s")
        cpu = monitor.get("cpu_util", [])
        if cpu:
            lines.append(f"\n**CPU 利用率峰值**: {max(cpu):.0f}%")

    # 问题检测报告
    lines += [
        "",
        "---",
        "",
        "## 3. 关键问题检测报告",
        "",
        "| 编号 | 问题 | 检测结论 |",
        "|------|------|---------|",
    ]

    # P1：梯度覆盖
    p2a_gn = [g for g in trainer_logs.get("phase2a", {}).get("grad_norm", []) if g == g]
    p2b_gn = [g for g in trainer_logs.get("phase2b", {}).get("grad_norm", []) if g == g]
    if p2a_gn and p2b_gn:
        avg_2a = sum(p2a_gn) / len(p2a_gn)
        avg_2b = sum(p2b_gn) / len(p2b_gn)
        ratio = avg_2b / avg_2a if avg_2a > 0 else 0
        if ratio < 0.4:
            p1_result = f"⚠ **确认 P1**: accum=4 梯度均值 {avg_2b:.3f} ≈ accum=1 的 {ratio:.2f}× (期望 ≈1.0x)，梯度被覆盖"
        elif 0.4 <= ratio < 0.8:
            p1_result = f"⚠ 疑似 P1: 梯度比值 {ratio:.2f}，低于理论值 1.0"
        else:
            p1_result = f"✓ P1 未触发: 梯度比值 {ratio:.2f} ≈ 正常"
    else:
        p1_result = "数据不足（Phase 2 未运行或无 grad_norm）"
    lines.append(f"| P1 | 梯度累积覆盖 bug | {p1_result} |")

    # P2：CPU 慢（从 phase4 步数 + 耗时估算）
    p4_analysis = log_dir / "phase4_analysis.txt"
    p2_result = "see phase4_analysis.txt"
    if p4_analysis.exists():
        text = p4_analysis.read_text()
        m = re.search(r"avg_sec_per_step[:\s]+([\d.]+)", text)
        if m:
            sec = float(m.group(1))
            if sec > 300:
                p2_result = f"**CRITICAL**: avg {sec:.0f}s/step, high DDP timeout risk"
            elif sec > 120:
                p2_result = f"WARNING: avg {sec:.0f}s/step, backward_base_weight_grad bottleneck"
            else:
                p2_result = f"avg {sec:.0f}s/step (acceptable)"
    lines.append(f"| P2 | CPU backward 速度瓶颈 | {p2_result} |")

    # P3：磁盘 I/O
    p3_summary = log_dir / "phase3_io_summary.txt"
    p3_result = "see phase3_io_summary.txt"
    if p3_summary.exists():
        text = p3_summary.read_text()
        m = re.search(r"peak[:\s]+([\d.]+)\s*MB/s", text, re.IGNORECASE)
        if m:
            peak = float(m.group(1))
            if peak < 100:
                p3_result = f"**LOW**: peak {peak:.0f} MB/s, saving ~190 GB will take {190*1024/max(peak,1):.0f} s"
            elif peak < 500:
                p3_result = f"peak {peak:.0f} MB/s, estimated save time {190*1024/peak:.0f} s"
            else:
                p3_result = f"peak {peak:.0f} MB/s, disk I/O sufficient"
    lines.append(f"| P3 | 磁盘吞吐极限 | {p3_result} |")

    # P4：MoE 负载
    p4_log = log_dir / "phase4" / "train.log"
    p4_moe = "无专项数据（需路由统计 hook）"
    if p4_log.exists():
        with p4_log.open(errors="replace") as f:
            text = f.read()
        if re.search(r"aux_loss|balance_loss|router_loss", text, re.IGNORECASE):
            p4_moe = "检测到路由辅助损失，见 phase4/train.log"
        elif re.search(r"expert.*token|token.*expert", text, re.IGNORECASE):
            p4_moe = "检测到 expert 路由日志，见 phase4/train.log"
    lines.append(f"| P4 | MoE 路由负载均衡 | {p4_moe} |")

    # P5：NaN/Inf/崩溃
    p5_results = []
    for ph in phase_descs:
        tl = trainer_logs.get(ph, {})
        if tl.get("nan_lines"):
            p5_results.append(f"{ph}(行:{tl['nan_lines'][:3]})")
        ec = exit_codes.get(ph)
        if ec and ec < 0:
            log_f = log_dir / ph / "train.log"
            if log_f.exists() and re.search(r"sigsegv|segfault|core dump", log_f.read_text(errors="replace"), re.IGNORECASE):
                p5_results.append(f"{ph}:SIGSEGV")
    p5_result = "、".join(p5_results) if p5_results else "✓ 未检测到 NaN/Inf/崩溃"
    if p5_results:
        p5_result = "⚠ " + p5_result
    lines.append(f"| P5 | C++ 梯度索引 bug / NaN | {p5_result} |")

    # P6：Router 梯度
    p6_result = "见 05_grad_norm.png（full 模式 Router 参与梯度）"
    lines.append(f"| P6 | Router 梯度稳定性 | {p6_result} |")

    # P7：re-quantize
    p7_count = 0
    if p4_log.exists():
        with p4_log.open(errors="replace") as f:
            p7_count = sum(1 for line in f if re.search(r"update_base_weights|re-quantize|syncing updated", line, re.IGNORECASE))
    p7_result = f"phase4 中触发 {p7_count} 次（每步一次为正常）"
    lines.append(f"| P7 | update_base_weights 耗时 | {p7_result} |")

    lines += [
        "",
        "---",
        "",
        "## 4. 可视化图表",
        "",
        "| 文件 | 内容 |",
        "|------|------|",
        "| `plots/01_gpu_memory.png` | GPU 显存占用 & SM 利用率时序 |",
        "| `plots/02_cpu_ram.png` | CPU RAM & CPU 利用率时序 |",
        "| `plots/03_disk_throughput.png` | 磁盘读写吞吐（含 checkpoint 标注） |",
        "| `plots/04_training_loss.png` | 训练 Loss 曲线（各 Phase） |",
        "| `plots/05_grad_norm.png` | 梯度范数曲线（含 P1 诊断说明） |",
        "| `plots/06_nan_inf_timeline.png` | NaN/Inf/崩溃事件时间线 |",
        "| `plots/07_phase_summary.png` | 各 Phase 性能对比 |",
        "",
        "---",
        "",
        "## 5. 重新运行分析",
        "",
        "```bash",
        f"python3 {log_dir.parent.parent}/analyze.py --log-dir {log_dir}",
        "```",
    ]

    summary_path.write_text("\n".join(lines))
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

    print(f"[analyze] phases found   : {list(trainer_logs.keys())}")
    print(f"[analyze] exit codes     : {exit_codes}")
    print(f"[analyze] monitor samples: {len(monitor.get('elapsed', []))}")

    # Generate plots
    if _HAS_MPL:
        print("[analyze] plot 1: GPU VRAM ...")
        plot_gpu_memory(monitor, plots_dir)

        print("[analyze] plot 2: CPU RAM ...")
        plot_cpu_ram(monitor, plots_dir)

        print("[analyze] plot 3: disk throughput ...")
        plot_disk_throughput(monitor, plots_dir)

        print("[analyze] plot 4: training loss ...")
        plot_training_loss(trainer_logs, plots_dir)

        print("[analyze] plot 5: grad norm ...")
        plot_grad_norm(trainer_logs, plots_dir)

        print("[analyze] plot 6: NaN/Inf timeline ...")
        plot_nan_timeline(trainer_logs, exit_codes, plots_dir)

        print("[analyze] plot 7: phase summary ...")
        plot_phase_summary(monitor, trainer_logs, exit_codes, plots_dir)
    else:
        print("[analyze] matplotlib not installed, skipping plots")

    # Generate summary.md
    print("[analyze] generating summary.md ...")
    summary_path = generate_summary_md(log_dir, monitor, trainer_logs, exit_codes)

    print(f"\n[analyze] done.")
    print(f"  plots  : {plots_dir}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()

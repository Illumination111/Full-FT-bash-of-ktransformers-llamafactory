#!/usr/bin/env python3
"""
后台系统指标采集脚本 —— Qwen3-30B-A3B FFT 测试监控器

采集指标（每 INTERVAL 秒一次）：
  - 每张 GPU 的已用显存、总显存、SM 利用率、显存利用率（整机）
  - 系统 RAM（总量、已用、可用）
  - CPU 利用率（总体）
  - 可选：按 --pid 进程树过滤的 proc_ram_gb / proc_gpu*_mem_mb
    （不需要 root；用 psutil RSS + NVML/nvidia-smi 进程级显存）

事件标记：
  监听命名管道 FIFO（--fifo 参数），训练脚本向其写入如下格式的事件：
    phase:<name>
    event:<step_start|step_end|backward_start>
    pid:<root_pid>          # 动态更新进程树根（可选）

用法：
  python monitor.py --out /path/to/monitor.csv [--fifo /tmp/monitor_events.fifo] \
                    [--interval 2] [--pid $$]
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# 依赖检查
# --------------------------------------------------------------------------- #
try:
    import psutil
except ImportError:
    print("[monitor] psutil 未安装，运行: pip install psutil", file=sys.stderr)
    sys.exit(1)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _NVML_AVAILABLE = False
    _GPU_COUNT = 0


def _nvidia_smi_query() -> list[dict]:
    """通过 nvidia-smi 获取 GPU 指标（pynvml 不可用时的备用方案）。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "mem_used_mb": int(float(parts[1])),
                    "mem_total_mb": int(float(parts[2])),
                    "sm_util_pct": int(float(parts[3])),
                    "mem_util_pct": int(float(parts[4])),
                })
        return gpus
    except Exception:
        return []


# 自动探测 GPU 数量（用于 CSV 列定义）
if not _NVML_AVAILABLE:
    _smi_result = _nvidia_smi_query()
    _GPU_COUNT = len(_smi_result)
    if _GPU_COUNT > 0:
        print(f"[monitor] pynvml 不可用，将使用 nvidia-smi 获取 GPU 信息（{_GPU_COUNT} 张）")


# --------------------------------------------------------------------------- #
# CSV 列定义
# --------------------------------------------------------------------------- #
def _gpu_columns(n: int) -> list[str]:
    cols = []
    for i in range(n):
        cols += [
            f"gpu{i}_mem_used_mb",
            f"gpu{i}_mem_total_mb",
            f"gpu{i}_mem_util_pct",
            f"gpu{i}_sm_util_pct",
            f"proc_gpu{i}_mem_mb",
        ]
    return cols


BASE_COLUMNS = [
    "timestamp",
    "elapsed_sec",
    "phase",
    "event",
    "cpu_util_pct",
    "ram_used_gb",
    "ram_total_gb",
    "ram_avail_gb",
    "root_pid",
    "proc_count",
    "proc_ram_gb",
    "proc_cpu_pct",
]


# --------------------------------------------------------------------------- #
# GPU 指标（整机）
# --------------------------------------------------------------------------- #
def _sample_gpu() -> list[dict]:
    """采集所有 GPU 指标，返回列表（每卡一个 dict）。"""
    if not _NVML_AVAILABLE:
        return _nvidia_smi_query()
    results = []
    for i in range(_GPU_COUNT):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            results.append({
                "mem_used_mb": mem.used // (1024 * 1024),
                "mem_total_mb": mem.total // (1024 * 1024),
                "mem_util_pct": util.memory,
                "sm_util_pct": util.gpu,
            })
        except Exception:
            results.append({
                "mem_used_mb": 0,
                "mem_total_mb": 0,
                "mem_util_pct": 0,
                "sm_util_pct": 0,
            })
    return results


def _safe_used_gpu_memory_bytes(proc_info) -> int:
    """NVML usedGpuMemory 在部分驱动上可能为 -1 / None。"""
    try:
        mem = int(getattr(proc_info, "usedGpuMemory", 0) or 0)
    except Exception:
        return 0
    return mem if mem > 0 else 0


def _sample_gpu_proc_mem_nvml(pid_set: set[int]) -> list[int]:
    """按 PID 过滤，返回每卡进程显存 MB。"""
    out = [0] * _GPU_COUNT
    if not pid_set or not _NVML_AVAILABLE:
        return out

    getters = [
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
        getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v3", None),
        getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
    ]

    for i in range(_GPU_COUNT):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue
        seen: set[int] = set()
        total = 0
        for getter in getters:
            if getter is None:
                continue
            try:
                procs = getter(handle) or []
            except Exception:
                continue
            for p in procs:
                try:
                    pid = int(p.pid)
                except Exception:
                    continue
                if pid not in pid_set or pid in seen:
                    continue
                seen.add(pid)
                total += _safe_used_gpu_memory_bytes(p)
        out[i] = total // (1024 * 1024)
    return out


def _sample_gpu_proc_mem_smi(pid_set: set[int]) -> list[int]:
    """nvidia-smi 进程级显存备用方案。"""
    out = [0] * _GPU_COUNT
    if not pid_set or _GPU_COUNT <= 0:
        return out
    try:
        uuid_map: dict[str, int] = {}
        meta = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if meta.returncode == 0:
            for line in meta.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    uuid_map[parts[1]] = int(parts[0])

        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if apps.returncode != 0:
            return out
        for line in apps.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
            try:
                pid = int(pid_s)
                mem_mb = int(float(mem_s))
            except ValueError:
                continue
            if pid not in pid_set or mem_mb <= 0:
                continue
            gi = uuid_map.get(uuid)
            if gi is None or gi < 0 or gi >= _GPU_COUNT:
                continue
            out[gi] += mem_mb
    except Exception:
        pass
    return out


def _sample_gpu_proc_mem(pid_set: set[int]) -> list[int]:
    if _NVML_AVAILABLE:
        return _sample_gpu_proc_mem_nvml(pid_set)
    return _sample_gpu_proc_mem_smi(pid_set)


# --------------------------------------------------------------------------- #
# 进程树采样（无需 root）
# --------------------------------------------------------------------------- #
def _collect_process_tree(root_pid: int | None) -> tuple[set[int], float, float]:
    """
    返回 (pid_set, rss_gb, cpu_pct)。
    cpu_pct 为进程树内各进程 cpu_percent 之和（可 >100，多核）。
    自动排除 monitor 自身及其子进程，避免把监控开销算进训练。
    """
    if root_pid is None or root_pid <= 0:
        return set(), 0.0, 0.0
    try:
        root = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set(), 0.0, 0.0

    try:
        procs = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        procs = [root]

    # 排除 monitor 自身子树（root 常为启动脚本 $$，monitor 是其子进程）
    exclude: set[int] = {os.getpid()}
    try:
        exclude.update(c.pid for c in psutil.Process(os.getpid()).children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    pid_set: set[int] = set()
    rss = 0
    cpu = 0.0
    for p in procs:
        try:
            if p.pid in exclude:
                continue
            pid_set.add(p.pid)
            rss += int(p.memory_info().rss)
            # interval=None：相对上次调用的增量；首次多为 0
            cpu += float(p.cpu_percent(interval=None))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return pid_set, rss / 1e9, cpu


# --------------------------------------------------------------------------- #
# 主监控循环
# --------------------------------------------------------------------------- #
class Monitor:
    def __init__(
        self,
        out_path: str,
        fifo_path: str | None,
        interval: float,
        root_pid: int | None = None,
    ):
        self.out_path = out_path
        self.fifo_path = fifo_path
        self.interval = interval
        self.root_pid = root_pid
        self._running = True
        self._phase = "init"
        self._event = ""
        self._start_time = time.time()
        self._numa_cpu_map = self._build_numa_map()

        print(f"[monitor] 输出文件: {out_path}", flush=True)
        print(f"[monitor] GPU 数量: {_GPU_COUNT}", flush=True)
        print(f"[monitor] FIFO 路径: {fifo_path}", flush=True)
        print(f"[monitor] 进程树根 PID: {root_pid}", flush=True)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cols = BASE_COLUMNS + _gpu_columns(_GPU_COUNT)
        self._f = open(out_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._f, fieldnames=cols, extrasaction="ignore")
        self._writer.writeheader()
        self._f.flush()

        if fifo_path:
            self._ensure_fifo(fifo_path)
            self._fifo_thread = threading.Thread(
                target=self._fifo_reader, daemon=True
            )
            self._fifo_thread.start()

    @staticmethod
    def _build_numa_map() -> dict[int, list[int]]:
        numa_map: dict[int, list[int]] = {}
        base = Path("/sys/devices/system/node")
        if not base.exists():
            return numa_map
        for node_dir in sorted(base.glob("node[0-9]*")):
            try:
                node_id = int(node_dir.name[4:])
                cpulist = (node_dir / "cpulist").read_text().strip()
                cpus: list[int] = []
                for part in cpulist.split(","):
                    if "-" in part:
                        lo, hi = part.split("-")
                        cpus.extend(range(int(lo), int(hi) + 1))
                    else:
                        cpus.append(int(part))
                numa_map[node_id] = cpus
            except Exception:
                pass
        return numa_map

    @staticmethod
    def _ensure_fifo(path: str) -> None:
        p = Path(path)
        if p.exists():
            if not stat.S_ISFIFO(p.stat().st_mode):
                p.unlink()
                os.mkfifo(path)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            os.mkfifo(path)

    def _fifo_reader(self) -> None:
        while self._running:
            try:
                # Keep a writer endpoint open for the monitor lifetime.  A
                # read-only FIFO sees EOF whenever a short-lived event writer
                # closes, creating a reopen race that can SIGPIPE the runner.
                fd = os.open(self.fifo_path, os.O_RDWR)
                with os.fdopen(fd, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("phase:"):
                            self._phase = line[6:]
                            print(f"[monitor] Phase 切换 → {self._phase}", flush=True)
                        elif line.startswith("event:"):
                            self._event = line[6:]
                            print(f"[monitor] 事件: {self._event}", flush=True)
                        elif line.startswith("pid:"):
                            try:
                                self.root_pid = int(line[4:].strip())
                                print(
                                    f"[monitor] 进程树根 PID 更新 → {self.root_pid}",
                                    flush=True,
                                )
                            except ValueError:
                                print(f"[monitor] 忽略非法 pid 事件: {line}", flush=True)
            except Exception:
                if self._running:
                    time.sleep(0.5)

    def _sample_once(self, dt: float) -> dict:
        del dt  # 保留签名兼容
        now = datetime.now().astimezone().isoformat(timespec="milliseconds")
        elapsed = time.time() - self._start_time

        cpu_util = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()

        pid_set, proc_ram_gb, proc_cpu_pct = _collect_process_tree(self.root_pid)
        proc_gpu = _sample_gpu_proc_mem(pid_set)

        row: dict = {
            "timestamp": now,
            "elapsed_sec": f"{elapsed:.1f}",
            "phase": self._phase,
            "event": self._event,
            "cpu_util_pct": f"{cpu_util:.1f}",
            "ram_used_gb": f"{vm.used / 1e9:.2f}",
            "ram_total_gb": f"{vm.total / 1e9:.2f}",
            "ram_avail_gb": f"{vm.available / 1e9:.2f}",
            "root_pid": self.root_pid if self.root_pid else "",
            "proc_count": len(pid_set),
            "proc_ram_gb": f"{proc_ram_gb:.2f}",
            "proc_cpu_pct": f"{proc_cpu_pct:.1f}",
        }

        for i, g in enumerate(_sample_gpu()):
            row[f"gpu{i}_mem_used_mb"] = g["mem_used_mb"]
            row[f"gpu{i}_mem_total_mb"] = g["mem_total_mb"]
            row[f"gpu{i}_mem_util_pct"] = g["mem_util_pct"]
            row[f"gpu{i}_sm_util_pct"] = g["sm_util_pct"]
            row[f"proc_gpu{i}_mem_mb"] = proc_gpu[i] if i < len(proc_gpu) else 0

        self._event = ""
        return row

    def run(self) -> None:
        psutil.cpu_percent(interval=None)
        # 预热进程树 cpu_percent
        _collect_process_tree(self.root_pid)
        prev_t = time.time()

        print("[monitor] 开始采样...", flush=True)
        while self._running:
            time.sleep(self.interval)
            now_t = time.time()
            dt = max(now_t - prev_t, 0.01)
            row = self._sample_once(dt)
            self._writer.writerow(row)
            prev_t = now_t

        self._f.close()
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("[monitor] 已停止。", flush=True)

    def stop(self) -> None:
        self._running = False


# --------------------------------------------------------------------------- #
# 信号处理
# --------------------------------------------------------------------------- #
_monitor_instance: Monitor | None = None


def _sig_handler(signum, frame):
    print(f"\n[monitor] 收到信号 {signum}，正在停止...", flush=True)
    if _monitor_instance:
        _monitor_instance.stop()
    sys.exit(0)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main():
    global _monitor_instance

    parser = argparse.ArgumentParser(description="FFT 系统指标监控器")
    parser.add_argument("--out", required=True, help="输出 CSV 文件路径")
    parser.add_argument(
        "--fifo",
        default="/tmp/fft_monitor_events.fifo",
        help="命名管道路径（训练脚本向此写入事件）",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="采样间隔（秒）")
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="进程树根 PID（通常传启动脚本的 $$）；只统计该树的 RAM/显存",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    _monitor_instance = Monitor(
        out_path=args.out,
        fifo_path=args.fifo,
        interval=args.interval,
        root_pid=args.pid,
    )
    _monitor_instance.run()


if __name__ == "__main__":
    main()

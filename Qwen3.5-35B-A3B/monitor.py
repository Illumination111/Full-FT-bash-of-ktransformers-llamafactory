#!/usr/bin/env python3
"""
后台系统指标采集脚本 —— Qwen3.5-35B-A3B FFT 测试监控器

采集指标（每 INTERVAL 秒一次）：
  - 每张 GPU 的已用显存、总显存、SM 利用率、显存利用率
  - 系统 RAM（总量、已用、可用）
  - 磁盘 I/O 速率（读写 MB/s），优先采 /mnt/data2 所在设备
  - CPU 利用率（总体 + 每 NUMA 节点估算）

事件标记：
  监听命名管道 FIFO（--fifo 参数），训练脚本向其写入如下格式的事件：
    phase:<name>
    event:<checkpoint_start|checkpoint_end|step_start|step_end|backward_start>

用法：
  python monitor.py --out /path/to/monitor.csv [--fifo /tmp/monitor_events.fifo] \
                    [--interval 2] [--disk-mount /mnt/data2]
"""

import argparse
import csv
import os
import select
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

# 若 pynvml 不可用，尝试通过 nvidia-smi 子进程获取 GPU 信息
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
    "disk_read_mbps",
    "disk_write_mbps",
    "disk_read_iops",
    "disk_write_iops",
]

# --------------------------------------------------------------------------- #
# 磁盘设备解析
# --------------------------------------------------------------------------- #
def _resolve_disk_device(mount_point: str) -> str | None:
    """返回挂载点所在的磁盘设备名（如 'sda' / 'nvme0n1'）。"""
    try:
        for part in psutil.disk_partitions(all=True):
            if part.mountpoint == mount_point:
                dev = Path(part.device).name
                # 去掉分区号后缀，取磁盘整体（如 nvme0n1p1 → nvme0n1）
                for suffix in ["p1","p2","p3","p4","1","2","3","4"]:
                    if dev.endswith(suffix):
                        candidate = dev[: -len(suffix)]
                        if Path(f"/sys/block/{candidate}").exists():
                            return candidate
                if Path(f"/sys/block/{dev}").exists():
                    return dev
    except Exception:
        pass
    return None


def _get_disk_io(device: str | None) -> dict:
    """返回磁盘 I/O 计数器快照（字节数、IOPS）。"""
    try:
        counters = psutil.disk_io_counters(perdisk=True)
        if device and device in counters:
            c = counters[device]
        else:
            # 汇总所有设备
            c = psutil.disk_io_counters(perdisk=False)
        if c is None:
            return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}
        return {
            "read_bytes": c.read_bytes,
            "write_bytes": c.write_bytes,
            "read_count": c.read_count,
            "write_count": c.write_count,
        }
    except Exception:
        return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}


# --------------------------------------------------------------------------- #
# GPU 指标
# --------------------------------------------------------------------------- #
def _sample_gpu() -> list[dict]:
    """采集所有 GPU 指标，返回列表（每卡一个 dict）。"""
    # 优先使用 pynvml（低开销）；不可用时 fallback 到 nvidia-smi
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


# --------------------------------------------------------------------------- #
# 主监控循环
# --------------------------------------------------------------------------- #
class Monitor:
    def __init__(
        self,
        out_path: str,
        fifo_path: str | None,
        interval: float,
        disk_mount: str,
    ):
        self.out_path = out_path
        self.fifo_path = fifo_path
        self.interval = interval
        self.disk_mount = disk_mount
        self.disk_device = _resolve_disk_device(disk_mount)
        self._running = True
        self._phase = "init"
        self._event = ""
        self._start_time = time.time()

        # NUMA CPU 利用率（用 /proc/stat 近似，每插槽 CPU 列表）
        self._numa_cpu_map = self._build_numa_map()

        print(f"[monitor] 输出文件: {out_path}", flush=True)
        print(f"[monitor] 磁盘设备: {self.disk_device or '(all devices)'}", flush=True)
        print(f"[monitor] GPU 数量: {_GPU_COUNT}", flush=True)
        print(f"[monitor] FIFO 路径: {fifo_path}", flush=True)

        # 打开 CSV 写入
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cols = BASE_COLUMNS + _gpu_columns(_GPU_COUNT)
        self._f = open(out_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._f, fieldnames=cols, extrasaction="ignore")
        self._writer.writeheader()
        self._f.flush()

        # FIFO 监听线程
        if fifo_path:
            self._ensure_fifo(fifo_path)
            self._fifo_thread = threading.Thread(
                target=self._fifo_reader, daemon=True
            )
            self._fifo_thread.start()

    @staticmethod
    def _build_numa_map() -> dict[int, list[int]]:
        """解析 /sys/devices/system/node/ 得到 NUMA 节点→CPU 列表映射。"""
        numa_map: dict[int, list[int]] = {}
        base = Path("/sys/devices/system/node")
        if not base.exists():
            return numa_map
        for node_dir in sorted(base.glob("node[0-9]*")):
            try:
                node_id = int(node_dir.name[4:])
                cpulist_file = node_dir / "cpulist"
                cpulist = cpulist_file.read_text().strip()
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
        """持续从 FIFO 读取事件行（阻塞式）。"""
        while self._running:
            try:
                with open(self.fifo_path, "r") as f:
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
            except Exception as e:
                if self._running:
                    time.sleep(0.5)

    def _sample_once(self, prev_disk: dict, dt: float) -> dict:
        now = datetime.utcnow().isoformat(timespec="milliseconds")
        elapsed = time.time() - self._start_time

        # CPU
        cpu_util = psutil.cpu_percent(interval=None)

        # RAM
        vm = psutil.virtual_memory()
        ram_used_gb = vm.used / 1e9
        ram_total_gb = vm.total / 1e9
        ram_avail_gb = vm.available / 1e9

        # Disk
        curr_disk = _get_disk_io(self.disk_device)
        read_mbps = (curr_disk["read_bytes"] - prev_disk["read_bytes"]) / 1e6 / dt
        write_mbps = (curr_disk["write_bytes"] - prev_disk["write_bytes"]) / 1e6 / dt
        read_iops = (curr_disk["read_count"] - prev_disk["read_count"]) / dt
        write_iops = (curr_disk["write_count"] - prev_disk["write_count"]) / dt

        row: dict = {
            "timestamp": now,
            "elapsed_sec": f"{elapsed:.1f}",
            "phase": self._phase,
            "event": self._event,
            "cpu_util_pct": f"{cpu_util:.1f}",
            "ram_used_gb": f"{ram_used_gb:.2f}",
            "ram_total_gb": f"{ram_total_gb:.2f}",
            "ram_avail_gb": f"{ram_avail_gb:.2f}",
            "disk_read_mbps": f"{read_mbps:.1f}",
            "disk_write_mbps": f"{write_mbps:.1f}",
            "disk_read_iops": f"{read_iops:.0f}",
            "disk_write_iops": f"{write_iops:.0f}",
        }

        # GPU
        for i, g in enumerate(_sample_gpu()):
            row[f"gpu{i}_mem_used_mb"] = g["mem_used_mb"]
            row[f"gpu{i}_mem_total_mb"] = g["mem_total_mb"]
            row[f"gpu{i}_mem_util_pct"] = g["mem_util_pct"]
            row[f"gpu{i}_sm_util_pct"] = g["sm_util_pct"]

        # 重置事件（单次触发）
        self._event = ""
        return row, curr_disk

    def run(self) -> None:
        # 初始化 CPU percent（第一次调用返回 0）
        psutil.cpu_percent(interval=None)
        prev_disk = _get_disk_io(self.disk_device)
        prev_t = time.time()

        print("[monitor] 开始采样...", flush=True)
        while self._running:
            time.sleep(self.interval)
            now_t = time.time()
            dt = max(now_t - prev_t, 0.01)
            row, prev_disk = self._sample_once(prev_disk, dt)
            self._writer.writerow(row)
            prev_t = now_t

        self._f.close()
        if _NVML_AVAILABLE:
            pynvml.nvmlShutdown()
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
        "--disk-mount",
        default="/mnt/data2",
        help="重点监控的磁盘挂载点",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    _monitor_instance = Monitor(
        out_path=args.out,
        fifo_path=args.fifo,
        interval=args.interval,
        disk_mount=args.disk_mount,
    )
    _monitor_instance.run()


if __name__ == "__main__":
    main()

"""GPU monitor — pynvml with nvidia-smi fallback, cached snapshots.

Async equivalent of forge_gui/api/gpu_monitor.py: a background task
polls the GPU every ``interval_s`` and publishes ``gpu`` events; API
consumers read the cached snapshot so no request ever blocks on NVML.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any
import contextlib

logger = logging.getLogger(__name__)


def _nvml_snapshot() -> dict[str, Any] | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            limit = (pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
                     or None)
        except Exception:
            power = limit = None
        try:
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
        except Exception:
            name = "GPU"
        return {
            "available": True, "name": name,
            "vram_used_mb": round(mem.used / 1e6),
            "vram_total_mb": round(mem.total / 1e6),
            "vram_pct": round(mem.used / max(mem.total, 1) * 100, 1),
            "util_pct": util.gpu, "temp_c": temp,
            "power_w": round(power, 1) if power else None,
            "power_limit_w": round(limit, 1) if limit else None,
        }
    except Exception:
        return None


def _smi_snapshot() -> dict[str, Any] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        used, total = float(parts[1]), float(parts[2])
        return {
            "available": True, "name": parts[0],
            "vram_used_mb": round(used), "vram_total_mb": round(total),
            "vram_pct": round(used / max(total, 1) * 100, 1),
            "util_pct": float(parts[3]),
            "temp_c": float(parts[4]) if parts[4] else None,
            "power_w": float(parts[5]) if parts[5] else None,
            "power_limit_w": float(parts[6]) if len(parts) > 6 else None,
        }
    except Exception:
        return None


class GpuMonitor:
    """Cached GPU snapshot + background publisher."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self._snapshot: dict[str, Any] = {
            "available": False, "name": "", "vram_used_mb": 0,
            "vram_total_mb": 0, "vram_pct": 0.0, "util_pct": 0,
            "temp_c": None, "power_w": None, "power_limit_w": None,
        }
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._snapshot)

    def poll_once(self) -> dict[str, Any]:
        snap = _nvml_snapshot() or _smi_snapshot()
        if snap is not None:
            self._snapshot = snap
        else:
            self._snapshot["available"] = False
        return self._snapshot

    async def run(self, hub) -> None:
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            while not self._stop.is_set():
                try:
                    snap = await loop.run_in_executor(pool, self.poll_once)
                    hub.publish("gpu", snap)
                except Exception as e:
                    logger.debug("gpu poll failed: %s", e)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(),
                                           timeout=self.interval_s)

    def start(self, hub) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(hub))

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=3)


monitor = GpuMonitor()

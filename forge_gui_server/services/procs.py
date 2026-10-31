"""ProcessService — asyncio port of forge_gui.api.process_manager.

Spawns subprocesses, streams stdout+stderr line-by-line into the event
hub, persists logs to research/tasks/<task_id>/log.txt. Presets are
reused verbatim from the Qt-free parts of the original module.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from forge_gui.api.status_reader import project_root
import contextlib

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    id: str
    name: str
    command: list[str]
    pid: int = 0
    status: str = "starting"   # starting | running | done | crashed | killed
    started_at: float = 0.0
    ended_at: float = 0.0
    exit_code: int | None = None
    log_path: str | None = None
    lines: list[str] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        end = self.ended_at if self.ended_at else time.time()
        return max(0.0, end - self.started_at)

    @property
    def is_live(self) -> bool:
        return self.status in ("starting", "running")

    def to_dict(self, tail: int = 0) -> dict:
        # asdict() deep-copied the whole 5000-line buffer per call — build
        # the summary lazily; `lines` only ships when a tail is requested.
        d = {"id": self.id, "name": self.name, "command": self.command,
             "pid": self.pid, "status": self.status,
             "started_at": self.started_at, "ended_at": self.ended_at,
             "exit_code": self.exit_code, "log_path": self.log_path,
             "n_lines": len(self.lines),
             "elapsed_s": round(self.elapsed_s, 1),
             "is_live": self.is_live}
        if tail:
            d["lines"] = self.lines[-tail:]
        return d


class ProcessService:
    def __init__(self, hub) -> None:
        self._hub = hub
        self.tasks: dict[str, TaskInfo] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        # task_id -> [file handle, unflushed-write-count] buffered log writers
        self._log_fhs: dict[str, list] = {}
        self._counter = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        # asyncio tasks launched without a running loop — kept alive here
        # so the GC can't collect them mid-flight
        self._bg_tasks: set[asyncio.Task] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def launch(self, name: str, cmd: list[str],
               cwd: str | None = None) -> str:
        """Spawn a subprocess — sync so worker threads (LoRA training
        trigger, routes) can call it without an event loop."""
        root = project_root()
        # The counter restarts every process — skip ids that collide with
        # a live task or a stale research/tasks/ dir (log.txt would append
        # into the old run's log otherwise).
        while True:
            self._counter += 1
            task_id = f"task_{self._counter:04d}"
            task_dir = root / "research" / "tasks" / task_id
            if task_id not in self.tasks and not task_dir.exists():
                break
        task_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_dir / "log.txt"

        info = TaskInfo(id=task_id, name=name, command=cmd,
                        started_at=time.time(), log_path=str(log_path))
        self.tasks[task_id] = info
        self._hub.publish("task_added", info.to_dict())
        coro = self._run(task_id, cmd, cwd or str(root))
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            try:
                bg = asyncio.ensure_future(coro)
                self._bg_tasks.add(bg)
                bg.add_done_callback(self._bg_tasks.discard)
            except RuntimeError:
                threading.Thread(
                    target=lambda: asyncio.run(coro), daemon=True).start()
        return task_id

    async def _run(self, task_id: str, cmd: list[str], cwd: str) -> None:
        info = self.tasks[task_id]
        # Open log.txt once for the task's lifetime — the old per-line
        # open/write/close hammered the disk every output line.
        try:
            self._log_fhs[task_id] = [
                open(info.log_path, "a", encoding="utf-8"), 0]
        except (OSError, TypeError):
            self._log_fhs.pop(task_id, None)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=self._no_window())
        except Exception as e:
            self._emit_line(task_id, f"[PROCESS ERROR] {type(e).__name__}: {e}")
            self._set_status(task_id, "crashed", -1)
            return
        self._procs[task_id] = proc
        info.pid = proc.pid
        self._set_status(task_id, "running")
        try:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line:
                    self._emit_line(task_id, line)
            code = await proc.wait()
        except Exception as e:
            self._emit_line(task_id, f"[PROCESS ERROR] {type(e).__name__}: {e}")
            code = -1
        self._procs.pop(task_id, None)
        info.exit_code = code
        if info.status == "killed":
            pass
        elif code == 0:
            self._set_status(task_id, "done", code)
        else:
            self._set_status(task_id, "crashed", code)
        self._flush_log(task_id, force=True)
        ent = self._log_fhs.pop(task_id, None)
        if ent is not None:
            with contextlib.suppress(Exception):
                ent[0].close()
        self._hub.publish("task_finished", info.to_dict())

    @staticmethod
    def _no_window() -> int:
        import subprocess
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def _flush_log(self, task_id: str, force: bool = False) -> None:
        ent = self._log_fhs.get(task_id)
        if ent is None:
            return
        ent[1] += 1
        if force or ent[1] >= 32:
            with contextlib.suppress(Exception):
                ent[0].flush()
            ent[1] = 0

    def _emit_line(self, task_id: str, line: str) -> None:
        info = self.tasks.get(task_id)
        if info:
            info.lines.append(line)
            if len(info.lines) > 5000:
                del info.lines[: len(info.lines) - 5000]
            ent = self._log_fhs.get(task_id)
            if ent is not None:
                try:
                    ent[0].write(line + "\n")
                    self._flush_log(task_id)
                except Exception:
                    logger.debug("task log write failed", exc_info=True)
            elif info.log_path:
                try:
                    with open(info.log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    logger.debug("task log write failed", exc_info=True)
        self._hub.publish("task_line", {"task_id": task_id, "line": line})

    def _set_status(self, task_id: str, status: str,
                    exit_code: int | None = None) -> None:
        info = self.tasks.get(task_id)
        if not info:
            return
        info.status = status
        if exit_code is not None:
            info.exit_code = exit_code
        if status in ("done", "crashed", "killed"):
            info.ended_at = time.time()
        self._hub.publish("task_status",
                          {"task_id": task_id, "status": status,
                           "exit_code": info.exit_code})

    def kill(self, task_id: str) -> bool:
        proc = self._procs.get(task_id)
        info = self.tasks.get(task_id)
        # Only stamp "killed" on live tasks — killing an already-finished
        # task used to clobber its done/crashed status + end time.
        if info and info.is_live:
            info.status = "killed"
            info.ended_at = time.time()
        if proc and proc.returncode is None:
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception as e:
                    logger.warning("kill failed: %s", e)
        self._hub.publish("task_status",
                          {"task_id": task_id, "status": "killed"})
        return proc is not None

    def remove(self, task_id: str) -> bool:
        info = self.tasks.get(task_id)
        if info and not info.is_live:
            del self.tasks[task_id]
            self._hub.publish("task_removed", {"task_id": task_id})
            return True
        return False

    def all_tasks(self) -> list[dict]:
        return [t.to_dict() for t in self.tasks.values()]

    async def shutdown(self, timeout_s: float = 5.0) -> None:
        for tid in list(self._procs):
            self.kill(tid)
        deadline = time.time() + timeout_s
        while self._procs and time.time() < deadline:
            await asyncio.sleep(0.1)


proc_service: ProcessService | None = None

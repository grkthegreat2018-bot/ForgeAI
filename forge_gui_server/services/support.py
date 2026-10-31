"""Support services — Qt-free ports of the QObject managers that
ToolHarness consumes: SubAgentManager, TimeManager, BackupManager,
LibraryInstallManager.

Interfaces match the originals 1:1 (the harness calls them synchronously
from the agent worker thread). Approval flows that used Qt modal dialogs
now publish an ``approval`` hub event and block on a threading.Event
resolved by a REST call — same pattern as agent tool approvals.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import threading
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

from forge_gui.api.status_reader import project_root
import contextlib

logger = logging.getLogger(__name__)


# ── approvals ───────────────────────────────────────────────────────────

class ApprovalChannel:
    """Generic approve/deny gate: publish event → block → REST resolves."""

    def __init__(self, hub) -> None:
        self._hub = hub
        self._waiters: dict[str, tuple[threading.Event, dict, dict]] = {}
        # itertools.count.next() is atomic — the old `+= 1` could mint a
        # duplicate req_id, orphaning a waiter until the 300s timeout.
        self._counter = itertools.count(1)

    def request(self, kind: str, detail: dict,
                timeout_s: float = 300.0) -> bool:
        req_id = f"appr_{next(self._counter):04d}"
        evt = threading.Event()
        box: dict = {"granted": False}
        self._waiters[req_id] = (evt, box,
                                 {"id": req_id, "kind": kind,
                                  "detail": detail})
        self._hub.publish("approval", {
            "id": req_id, "kind": kind, "detail": detail})
        got = evt.wait(timeout_s)
        self._waiters.pop(req_id, None)
        return got and box["granted"]

    def respond(self, req_id: str, granted: bool) -> bool:
        entry = self._waiters.get(req_id)
        if not entry:
            return False
        evt, box, _meta = entry
        box["granted"] = granted
        evt.set()
        return True

    def pending(self) -> list[dict]:
        return [dict(meta) for _evt, _box, meta in self._waiters.values()]


# ── sub-agents ──────────────────────────────────────────────────────────

@dataclass
class SubAgentTask:
    task_id: str
    prompt: str
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    top_k: int = 80
    status: str = "pending"
    result: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    started_at: float = 0.0


class SubAgentService:
    """Port of SubAgentManager — ThreadPoolExecutor + engine lease."""

    def __init__(self, hub, engine_runtime, max_concurrent: int = 3) -> None:
        self._hub = hub
        self.engine_runtime = engine_runtime
        self.max_concurrent = max_concurrent
        self._tasks: dict[str, SubAgentTask] = {}
        self._futures: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"sub_{self._counter:03d}"

    def spawn(self, prompt: str, system_prompt: str = "",
              temperature: float = 0.7, max_tokens: int = 512,
              top_p: float = 0.95, top_k: int = 80) -> str:
        task_id = self._next_id()
        task = SubAgentTask(
            task_id=task_id, prompt=prompt, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, top_k=top_k, started_at=time.time())
        self._tasks[task_id] = task
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent,
                thread_name_prefix="subagent")
        self._futures[task_id] = self._executor.submit(
            self._run_sub_agent, task)
        self._hub.publish("sub_agent", {"kind": "started",
                                        "task_id": task_id})
        return task_id

    def spawn_batch(self, tasks: list[dict]) -> list[str]:
        return [self.spawn(
            prompt=t.get("prompt", ""),
            system_prompt=t.get("system_prompt", ""),
            temperature=t.get("temperature", 0.7),
            max_tokens=t.get("max_tokens", 512),
            top_p=t.get("top_p", 0.95),
            top_k=t.get("top_k", 80)) for t in tasks]

    def _run_sub_agent(self, task: SubAgentTask) -> None:
        task.status = "running"
        try:
            messages = []
            if task.system_prompt:
                messages.append({"role": "system",
                                 "content": task.system_prompt})
            messages.append({"role": "user", "content": task.prompt})
            from forge.self_play.discovery.qwen_adapter import render_messages_for_config
            cfg_name = self.engine_runtime.info.get("config_name", "")
            rendered = render_messages_for_config(
                messages, config_name=cfg_name, add_generation_prompt=True)
            parts: list[str] = []
            with self.engine_runtime.acquire(timeout_s=60.0) as engine:
                for tok in engine.generate_stream(
                        rendered, max_new_tokens=task.max_tokens,
                        temperature=task.temperature, top_p=task.top_p,
                        top_k=task.top_k):
                    parts.append(tok)
            task.result = "".join(parts)
            task.status = "done"
            task.elapsed_s = time.time() - task.started_at
            self._hub.publish("sub_agent", {"kind": "done",
                                            "task_id": task.task_id})
        except Exception as e:
            task.error = f"{type(e).__name__}: {e}"
            task.status = "error"
            task.elapsed_s = time.time() - task.started_at
            self._hub.publish("sub_agent", {
                "kind": "error", "task_id": task.task_id,
                "error": task.error})

    def get_result(self, task_id: str) -> SubAgentTask | None:
        return self._tasks.get(task_id)

    def wait_all(self, timeout_s: float = 120) -> dict[str, SubAgentTask]:
        for _tid, fut in list(self._futures.items()):
            with contextlib.suppress(Exception):
                fut.result(timeout=timeout_s)
        self._futures.clear()
        return dict(self._tasks)

    def list_tasks(self) -> list[dict]:
        return [{"task_id": t.task_id, "status": t.status,
                 "elapsed_s": round(t.elapsed_s, 2),
                 "result_preview": t.result[:200] if t.result else "",
                 "error": t.error}
                for t in self._tasks.values()]

    def clear(self) -> None:
        self._tasks = {k: v for k, v in self._tasks.items()
                       if v.status in ("pending", "running")}

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


# ── timers ──────────────────────────────────────────────────────────────

@dataclass
class TimerEntry:
    timer_id: str
    kind: str = "timer"
    label: str = ""
    fire_at: float = 0.0
    interval_s: float = 0.0
    repeat: bool = False
    on_process_exit: str = ""
    on_user_prompt: bool = False
    status: str = "active"
    fired_count: int = 0
    created_at: float = 0.0
    message: str = ""


class TimeService:
    """Port of TimeManager — threading.Timer instead of QTimer."""

    def __init__(self, hub) -> None:
        self._hub = hub
        self._timers: dict[str, TimerEntry] = {}
        self._handles: dict[str, threading.Timer] = {}
        self._counter = 0
        self._start_time = time.time()

    @property
    def uptime_s(self) -> float:
        return time.time() - self._start_time

    def _next_id(self) -> str:
        self._counter += 1
        return f"timer_{self._counter:03d}"

    def get_time(self) -> dict:
        now = datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": time.tzname[0] if time.tzname else "unknown",
            "unix_timestamp": time.time(),
            "uptime_seconds": round(self.uptime_s, 1),
            "uptime_human": _human_duration(self.uptime_s),
        }

    def set_timer(self, seconds: float, label: str = "",
                  message: str = "", repeat: bool = False,
                  on_process_exit: str = "",
                  on_user_prompt: bool = False) -> str:
        timer_id = self._next_id()
        entry = TimerEntry(
            timer_id=timer_id, kind="timer",
            label=label or f"Timer {seconds:.0f}s",
            fire_at=time.time() + seconds, interval_s=seconds,
            repeat=repeat, on_process_exit=on_process_exit,
            on_user_prompt=on_user_prompt, created_at=time.time(),
            message=message or f"Timer '{label}' fired")
        self._timers[timer_id] = entry
        self._arm(timer_id, seconds)
        return timer_id

    def set_alarm(self, time_str: str, label: str = "",
                  message: str = "", repeat: bool = False,
                  on_process_exit: str = "",
                  on_user_prompt: bool = False) -> str:
        target = _parse_time(time_str)
        if target is None:
            return ""
        now = datetime.now()
        fire_dt = now.replace(hour=target.hour, minute=target.minute,
                              second=0, microsecond=0)
        if fire_dt <= now:
            fire_dt += timedelta(days=1)
        delay_s = (fire_dt - now).total_seconds()
        timer_id = self._next_id()
        entry = TimerEntry(
            timer_id=timer_id, kind="alarm",
            label=label or f"Alarm {time_str}",
            fire_at=fire_dt.timestamp(), interval_s=delay_s,
            repeat=repeat, on_process_exit=on_process_exit,
            on_user_prompt=on_user_prompt, created_at=time.time(),
            message=message or f"Alarm '{label}' fired at {time_str}")
        self._timers[timer_id] = entry
        self._arm(timer_id, delay_s)
        return timer_id

    def _arm(self, timer_id: str, delay_s: float) -> None:
        t = threading.Timer(max(delay_s, 0.05), self._on_fire,
                            args=(timer_id,))
        t.daemon = True
        t.start()
        self._handles[timer_id] = t

    def _on_fire(self, timer_id: str) -> None:
        entry = self._timers.get(timer_id)
        if entry is None or entry.status != "active":
            return
        entry.fired_count += 1
        entry.status = "fired"
        self._hub.publish("timer", {"kind": "fired",
                                    "timer_id": entry.timer_id,
                                    "label": entry.label,
                                    "message": entry.message})
        if entry.repeat:
            entry.status = "active"
            entry.fire_at = time.time() + entry.interval_s
            self._arm(timer_id, entry.interval_s)

    def check_timer(self, timer_id: str) -> dict | None:
        entry = self._timers.get(timer_id)
        if entry is None:
            return None
        remaining = (max(0, entry.fire_at - time.time())
                     if entry.status == "active" else 0)
        return {
            "timer_id": entry.timer_id, "kind": entry.kind,
            "label": entry.label, "status": entry.status,
            "fired_count": entry.fired_count,
            "remaining_seconds": round(remaining, 1),
            "fire_at": datetime.fromtimestamp(entry.fire_at).strftime(
                "%Y-%m-%d %H:%M:%S") if entry.fire_at else "",
            "repeat": entry.repeat,
            "on_process_exit": entry.on_process_exit,
            "on_user_prompt": entry.on_user_prompt,
        }

    def cancel_timer(self, timer_id: str) -> bool:
        entry = self._timers.get(timer_id)
        if entry is None:
            return False
        entry.status = "cancelled"
        h = self._handles.pop(timer_id, None)
        if h:
            h.cancel()
        self._hub.publish("timer", {"kind": "cancelled",
                                    "timer_id": timer_id})
        return True

    def list_timers(self) -> list[dict]:
        return [self.check_timer(tid) for tid in self._timers
                if self.check_timer(tid) is not None]

    def check_conditions(self, active_processes: list[str] | None = None,
                         user_prompted: bool = False) -> list[str]:
        cancelled = []
        for tid, entry in list(self._timers.items()):
            if entry.status != "active":
                continue
            if entry.on_process_exit and active_processes is not None:
                if entry.on_process_exit not in active_processes:
                    self.cancel_timer(tid)
                    cancelled.append(tid)
                    continue
            if entry.on_user_prompt and user_prompted:
                self.cancel_timer(tid)
                cancelled.append(tid)
        return cancelled

    def get_fired_timers(self) -> list[dict]:
        fired = []
        for _tid, entry in list(self._timers.items()):
            if entry.status == "fired" and not entry.repeat:
                fired.append({"timer_id": entry.timer_id,
                              "label": entry.label,
                              "message": entry.message,
                              "kind": entry.kind})
                entry.status = "expired"
        return fired

    def shutdown(self) -> None:
        for h in self._handles.values():
            h.cancel()
        self._handles.clear()
        self._timers.clear()


def _parse_time(time_str: str) -> dtime | None:
    import re
    s = time_str.strip().upper()
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3)
        if ampm == "PM" and hour < 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            return dtime(hour, minute)
    m = re.match(r"(\d{1,2}):(\d{2})$", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return dtime(hour, minute)
    return None


def _human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    return f"{minutes // 60}h {minutes % 60}m"


# ── backups ─────────────────────────────────────────────────────────────

CHECK_INTERVAL_S = 60
MIN_CHANGED_FILES = 3
MAX_BACKUPS = 20
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".pytest_cache", ".ruff_cache", "data", ".devin",
             "research/checkpoints", "data/backups"}
SKIP_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
             ".safetensors", ".pt", ".bin", ".gguf", ".zip"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class BackupService:
    """Port of BackupManager — asyncio periodic check + REST approval."""

    def __init__(self, hub, approvals: ApprovalChannel,
                 root: Path | None = None) -> None:
        self._hub = hub
        self._approvals = approvals
        self.project_root = Path(root or project_root()).resolve()
        self.backup_dir = self.project_root / "data" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._file_hashes: dict[str, str] = {}
        self._active = False
        self._frozen = False
        self._task: asyncio.Task | None = None
        self._project_name = self.project_root.name

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def start(self) -> None:
        self._active = True
        self._snapshot_hashes()
        self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while self._active:
            await asyncio.sleep(CHECK_INTERVAL_S)
            if self._active and not self._frozen:
                try:
                    loop = asyncio.get_running_loop()
                    changes = await loop.run_in_executor(
                        None, self._count_changes)
                    if changes >= MIN_CHANGED_FILES:
                        # Whole-project zip on the loop stalled every
                        # endpoint — offload like the REST route does.
                        await loop.run_in_executor(
                            None, self.create_backup)
                except Exception as e:
                    logger.warning("backup check failed: %s", e)

    def _snapshot_hashes(self) -> None:
        self._file_hashes = {}
        for path in self._walk_files():
            try:
                self._file_hashes[str(path)] = self._hash_file(path)
            except OSError:
                continue

    def _walk_files(self) -> list[Path]:
        results = []
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in SKIP_EXTS:
                    continue
                try:
                    if p.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                results.append(p)
        return results

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Change fingerprint: mtime + size + sampled head/mid/tail blocks.
        The old md5(size + first 4KB) missed edits past the 4KB mark."""
        st = path.stat()
        size = st.st_size
        h = hashlib.md5()
        h.update(str(size).encode())
        h.update(str(st.st_mtime_ns).encode())
        with open(path, "rb") as f:
            h.update(f.read(4096))
            if size > 8192:
                f.seek(size // 2)
                h.update(f.read(4096))
                f.seek(max(0, size - 4096))
                h.update(f.read(4096))
        return h.hexdigest()

    def _count_changes(self) -> int:
        current = {}
        for path in self._walk_files():
            try:
                current[str(path)] = self._hash_file(path)
            except OSError:
                continue
        changes = sum(1 for k, v in current.items()
                      if self._file_hashes.get(k) != v)
        changes += len(set(self._file_hashes) - set(current))
        self._file_hashes = current
        return changes

    def create_backup(self) -> str | None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.backup_dir / f"{self._project_name}_{timestamp}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                for path in self._walk_files():
                    try:
                        zf.write(path, path.relative_to(self.project_root))
                    except OSError:
                        continue
            backups = sorted(self.backup_dir.glob("*.zip"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            for old in backups[MAX_BACKUPS:]:
                with contextlib.suppress(OSError):
                    old.unlink()
            self._hub.publish("backup", {"kind": "created",
                                         "path": str(zip_path)})
            return str(zip_path)
        except Exception as e:
            logger.error("backup creation failed: %s", e)
            return None

    def list_backups(self) -> list[dict]:
        out = []
        for p in sorted(self.backup_dir.glob("*.zip"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            st = p.stat()
            out.append({"name": p.name, "path": str(p),
                        "size_mb": round(st.st_size / (1024 * 1024), 1),
                        "date": datetime.fromtimestamp(st.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S")})
        return out

    def request_restore(self, backup_path: str) -> bool:
        """Freeze agent → publish approval → restore if granted."""
        self._frozen = True
        try:
            granted = self._approvals.request(
                "restore_backup",
                {"backup": backup_path,
                 "warning": "Wipes current project files and replaces "
                            "them with the backup. Cannot be undone."})
            if not granted:
                return False
            self._restore_backup(backup_path)
            self._hub.publish("backup", {"kind": "restored",
                                         "path": backup_path})
            return True
        finally:
            self._frozen = False

    _BAD_MEMBER_RE = re.compile(r"(^|/|\\)\.\.(/|\\|$)|^[a-zA-Z]:")

    @classmethod
    def _validate_zip_member(cls, name: str) -> None:
        """Reject zip-slip members: absolute paths, drive letters, '..'
        segments (forward- or back-slash)."""
        if not name or name.startswith(("/", "\\")) \
                or cls._BAD_MEMBER_RE.search(name):
            raise ValueError(f"unsafe zip member: {name!r}")

    def _restore_backup(self, zip_path: str) -> None:
        zip_p = Path(zip_path).resolve()
        # Only backups we made are restorable — a caller-supplied zip
        # anywhere else is rejected before the project is touched.
        backup_root = self.backup_dir.resolve()
        if not (zip_p == backup_root or backup_root in zip_p.parents):
            raise ValueError(
                f"backup path outside backups dir: {zip_path}")
        if not zip_p.is_file():
            raise FileNotFoundError(f"backup not found: {zip_path}")
        # Validate every member + extract to a staging dir FIRST — the
        # project tree is only wiped once the zip is known-good, so a
        # corrupt/evil archive can't wipe-then-fail.
        staging = self.backup_dir / f".restore_staging_{int(time.time())}"
        try:
            with zipfile.ZipFile(zip_p, "r") as zf:
                for m in zf.namelist():
                    self._validate_zip_member(m)
                staging.mkdir(parents=True, exist_ok=True)
                zf.extractall(staging)
        except Exception:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
            raise
        wipe_skip = SKIP_DIRS | {"data/backups"}
        # dirnames pruning only works TOP-DOWN (with topdown=False the
        # subdirs are walked before the parent's tuple is yielded — the
        # old prune was dead code and the wipe recursed into .git/venv/
        # data anyway). Do the file pass top-down, then rmdir bottom-up.
        dirs_seen: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(
                self.project_root, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in wipe_skip]
            dirs_seen.append(Path(dirpath))
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in SKIP_EXTS:
                    continue
                try:
                    p.unlink()
                except OSError:
                    continue
        # deepest-first so parents become empty after children are gone
        for d in sorted(dirs_seen, key=lambda x: len(x.parts),
                        reverse=True):
            if d == self.project_root:
                continue
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                continue
        import shutil

        def _merge_move(src: Path, dst: Path) -> None:
            # shutil.move(src→existing-dir) NESTS (src lands inside dst);
            # merge directories recursively instead.
            if src.is_dir() and dst.is_dir():
                for child in src.iterdir():
                    _merge_move(child, dst / child.name)
                src.rmdir()
            elif dst.exists():
                dst.unlink() if dst.is_file() else shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        try:
            for item in staging.iterdir():
                _merge_move(item, self.project_root / item.name)
            staging.rmdir()
        except Exception:
            import shutil as _sh
            _sh.rmtree(staging, ignore_errors=True)
            raise


# ── library install ─────────────────────────────────────────────────────

class LibraryService:
    """Port of LibraryInstallManager — allowlist + approval-gated pip."""

    def __init__(self, hub, approvals: ApprovalChannel) -> None:
        self._hub = hub
        self._approvals = approvals
        self._allowlist_path = project_root() / "data" / "library_allowlist.json"
        self._allowlist: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            if self._allowlist_path.is_file():
                self._allowlist = json.loads(
                    self._allowlist_path.read_text(encoding="utf-8"))
        except Exception:
            self._allowlist = []

    def _save(self) -> None:
        try:
            self._allowlist_path.parent.mkdir(parents=True, exist_ok=True)
            self._allowlist_path.write_text(
                json.dumps(self._allowlist, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("allowlist save failed", exc_info=True)

    def get_allowlist(self) -> list[str]:
        return list(self._allowlist)

    def is_allowed(self, package: str) -> bool:
        pkg = package.split("=")[0].split(">")[0].split("<")[0].strip()
        return pkg.lower() in {p.lower() for p in self._allowlist}

    def request_install(self, package: str) -> dict:
        if not package:
            return {"error": "package required"}
        if not self.is_allowed(package):
            granted = self._approvals.request(
                "install_library",
                {"package": package,
                 "warning": "Runs `pip install` inside the project venv. "
                            "Only approve packages you trust."})
            if not granted:
                return {"error": "install declined by user"}
            self._allowlist.append(package.split("=")[0])
            self._save()
        return self._do_install(package)

    def _do_install(self, package: str) -> dict:
        venv_py = project_root() / "venv" / "Scripts" / "python.exe"
        exe = str(venv_py) if venv_py.is_file() else "python"
        try:
            proc = subprocess.run(
                [exe, "-m", "pip", "install", package],
                capture_output=True, text=True, timeout=300,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            ok = proc.returncode == 0
            self._hub.publish("library", {"kind": "installed" if ok else
                                          "failed", "package": package})
            return {"ok": ok, "package": package,
                    "output": (proc.stdout or "")[-2000:],
                    "error": (proc.stderr or "")[-1000:] if not ok else ""}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


# ── LoRA harness adapter (sync facade over async LoraService) ──────────

class LoraHarnessAdapter:
    """Gives ToolHarness the synchronous interface it expects from
    LoraHarness: adapters_by_category() → LoRAEntry lists, _load(),
    _mgr.unload_from_engine(), _current, _runtime, current_adapter."""

    def __init__(self, lora_service, runtime, loop) -> None:
        self._svc = lora_service
        self._runtime = runtime
        self._loop = loop
        self._mgr = self  # harness calls ._mgr.unload_from_engine()

    @property
    def current_adapter(self) -> str | None:
        return self._svc._current

    @property
    def _current(self) -> str | None:
        return self._svc._current

    def adapters_by_category(self):
        from forge_gui.api.lora_store import scan_lora_adapters
        out: dict[str, list] = {}
        for e in scan_lora_adapters():
            out.setdefault(e.category, []).append(e)
        return out

    def _run_sync(self, coro):
        loop = self._loop
        if loop is None or not loop.is_running():
            # No server loop (make_harness before startup) — run inline.
            return asyncio.run(coro)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=180)

    def _load(self, path: str) -> None:
        self._run_sync(self._svc._load_auto(path))

    def unload_from_engine(self, runtime=None) -> None:
        self._run_sync(self._svc.unload_from_engine())

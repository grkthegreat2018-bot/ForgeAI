"""Reads training status.json files written by research.training_utils.write_status_json.

Discovers all known status files under research/checkpoints/ and research/tasks/,
tails them, and exposes a normalized snapshot for the GUI.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STOP_SENTINEL = "STOP_REQUESTED"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_search_dirs() -> list[Path]:
    root = project_root()
    return [
        root / "research" / "checkpoints",
        root / "research" / "tasks",
        root / "logs",
    ]


@dataclass
class RunSnapshot:
    """Normalized view of one training/processing run."""

    id: str
    name: str
    status_file: str
    status: str = "idle"            # idle | running | stopped | crashed | done
    step: int = 0
    max_steps: int = 0
    loss: float = 0.0
    lr: float = 0.0
    vram_gb: float = 0.0
    method: str = ""
    updated_at: float = 0.0
    heartbeat_age_s: float = 0.0
    extra: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        if self.max_steps <= 0:
            return 0.0
        return min(100.0, 100.0 * self.step / self.max_steps)

    @property
    def eta_s(self) -> float | None:
        if self.step <= 0 or self.max_steps <= 0 or self.updated_at <= 0:
            return None
        return None  # filled by reader if start_ts present

    @property
    def is_live(self) -> bool:
        return self.status == "running" and self.heartbeat_age_s < 60.0


class StatusReader:
    """Polls status.json files and returns RunSnapshots."""

    def __init__(self, search_dirs: list[Path] | None = None) -> None:
        self.search_dirs = search_dirs or _default_search_dirs()
        self._known: dict[str, float] = {}  # path -> mtime last seen
        self._data_cache: dict[str, dict] = {}  # path -> last parsed JSON

    def _iter_status_files(self):
        for d in self.search_dirs:
            if not d.is_dir():
                continue
            for p in d.rglob("*.json"):
                name = p.name.lower()
                if "status" in name or "heartbeat" in name:
                    yield p

    def _classify(self, data: dict, mtime: float) -> str:
        explicit = str(data.get("status", "")).lower()
        if explicit in ("running", "stopped", "crashed", "done", "paused"):
            return explicit
        # Heuristic: if heartbeat updated within 60s, treat as running.
        age = time.time() - mtime
        if age < 60:
            return "running"
        if data.get("step", 0) >= data.get("max_steps", 1) > 0:
            return "done"
        return "idle"

    def snapshot(self) -> list[RunSnapshot]:
        out: list[RunSnapshot] = []
        now = time.time()
        for path in self._iter_status_files():
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except OSError as e:
                logger.warning("stat failed for status file %s: %s", path, e)
                continue
            if self._known.get(key) == mtime and key in self._data_cache:
                # unchanged on disk — skip the JSON re-parse
                data = self._data_cache[key]
            else:
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning("failed to read status file %s: %s", path, e)
                    continue
                if isinstance(data, dict):
                    self._known[key] = mtime
                    self._data_cache[key] = data
            if not isinstance(data, dict):
                continue
            hb = data.get("ts") or data.get("heartbeat") or mtime
            try:
                hb_f = float(hb)
            except (TypeError, ValueError):
                hb_f = mtime
            status = self._classify(data, mtime)
            run_id = path.stem
            name = data.get("name") or data.get("method") or run_id
            snap = RunSnapshot(
                id=run_id,
                name=str(name),
                status_file=str(path),
                status=status,
                step=int(data.get("step", 0) or 0),
                max_steps=int(data.get("max_steps", 0) or 0),
                loss=float(data.get("loss", 0.0) or 0.0),
                lr=float(data.get("lr", 0.0) or 0.0),
                vram_gb=float(data.get("vram_gb", data.get("vram", 0.0)) or 0.0),
                method=str(data.get("method", "")),
                updated_at=hb_f,
                heartbeat_age_s=now - hb_f,
                extra={k: v for k, v in data.items()
                       if k not in {"step", "max_steps", "loss", "lr", "vram",
                                    "vram_gb", "method", "status", "ts",
                                    "heartbeat", "name"}},
                raw=data,
            )
            out.append(snap)
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    def _confined_run_file(self, path: Path) -> Path | None:
        """Resolve ``path`` and require it under a runs search dir — a
        caller-supplied status_file used to let /runs/log_tail read
        log.txt out of ANY directory on the host."""
        try:
            p = path.resolve()
        except OSError:
            return None
        for d in self.search_dirs:
            d = d.resolve()
            if p == d or d in p.parents:
                return p
        return None

    def read_log_tail(self, status_file: str, lines: int = 200) -> list[str]:
        """Tail the sibling log.txt for a given status file's directory.

        Seeks from EOF — the old readlines() loaded the entire (possibly
        multi-GB) log on every 3s poll."""
        sf = self._confined_run_file(Path(status_file))
        if sf is None:
            logger.warning("log_tail rejected out-of-bounds status_file: %s",
                           status_file)
            return []
        log_path = sf.parent / "log.txt"
        if not log_path.is_file():
            # Fall back to project logs dir
            lp = project_root() / "logs"
            candidates = sorted(lp.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = candidates[0] if candidates else None
        if not log_path or not log_path.is_file():
            return []
        try:
            size = log_path.stat().st_size
            # ~200B/line is generous; cap the read at 1MB.
            read_back = min(size, max(lines * 200, 65536), 1 << 20)
            with open(log_path, "rb") as f:
                f.seek(size - read_back)
                chunk = f.read(read_back)
            tail = chunk.decode("utf-8", errors="ignore").splitlines()
            return tail[-lines:]
        except Exception as e:
            logger.warning("failed to tail log %s: %s", log_path, e)
            return []

    def request_stop(self, run: RunSnapshot) -> bool:
        """Write a STOP_REQUESTED sentinel next to the run's status file.

        Trainers poll for this sentinel in their status-file directory and
        shut down cooperatively. Returns True if the sentinel was written.
        """
        if not run or not run.status_file:
            logger.warning("request_stop called with no status file")
            return False
        sentinel = Path(run.status_file).parent / STOP_SENTINEL
        try:
            sentinel.write_text(
                f"stop requested at {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8")
            logger.info("stop requested for run %s via %s", run.id, sentinel)
            return True
        except OSError as e:
            logger.warning("failed to write stop sentinel %s: %s", sentinel, e)
            return False

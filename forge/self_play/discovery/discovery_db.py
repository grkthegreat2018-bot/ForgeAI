"""Discovery database — SQLite memory the LLM reads, writes, and can refactor.

The schema is intentionally permissive (TEXT-heavy) so the LLM can evolve it
via `migrate_schema`. Every LLM-initiated schema change is recorded in
`schema_migrations` so the user can audit what the model changed and why.

Core tables (created on first open):
  sessions          — one row per discovery session
  thoughts          — train-of-thought entries (theorizing / sudo-thinking)
  scripts           — Python the LLM wrote + execution results
  research          — internet research findings
  theories          — hypotheses with status + evidence tallies
  discoveries       — confirmed findings the LLM chose to record
  events            — append-only event log (every tool call + result)
  schema_migrations — audit log of LLM-initiated DDL

All writes go through this class so events are emitted consistently and the
user-facing monitor can tail a single `events` table.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started REAL NOT NULL,
        ended REAL,
        summary TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS thoughts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        parent_id INTEGER,
        kind TEXT NOT NULL,           -- 'think' | 'sudo_think' | 'musing'
        content TEXT NOT NULL,
        confidence REAL,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS scripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        thought_id INTEGER,
        code TEXT NOT NULL,
        language TEXT DEFAULT 'python',
        stdout TEXT,
        stderr TEXT,
        returncode INTEGER,
        exec_ms REAL,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS research (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        url TEXT,
        title TEXT,
        summary TEXT,
        raw_snippet TEXT,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS theories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        statement TEXT NOT NULL,
        status TEXT DEFAULT 'open',   -- 'open' | 'supported' | 'refuted' | 'abandoned'
        evidence_for INTEGER DEFAULT 0,
        evidence_against INTEGER DEFAULT 0,
        notes TEXT,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS discoveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        theory_id INTEGER,
        summary TEXT NOT NULL,
        confidence REAL,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        kind TEXT NOT NULL,
        payload TEXT,                 -- JSON blob
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS schema_migrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        sql TEXT NOT NULL,
        reason TEXT,
        success INTEGER NOT NULL,
        error TEXT,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS epochs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch_num INTEGER NOT NULL,
        checkpoint_path TEXT NOT NULL,
        parent_epoch INTEGER,
        quality REAL,
        skill REAL,
        compute REAL,
        composite REAL,
        status TEXT DEFAULT 'candidate',  -- 'best' | 'archived' | 'candidate'
        kind TEXT DEFAULT 'finetune',      -- 'finetune' | 'distill'
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS distill_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch_num INTEGER NOT NULL,
        from_epoch INTEGER NOT NULL,
        to_epoch INTEGER NOT NULL,
        filtered_items INTEGER,
        bloat_removed INTEGER,
        notes TEXT,
        ts REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS tool_trajectories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        task TEXT NOT NULL,              -- the user query/task
        messages TEXT NOT NULL,          -- JSON: full Qwen-format conversation
        tool_calls TEXT NOT NULL,        -- JSON: list of {name, args, result, success}
        n_tool_calls INTEGER NOT NULL,
        n_successful INTEGER NOT NULL,
        reward REAL NOT NULL,            -- composite reward 0..1
        format_ok INTEGER NOT NULL,      -- 1 if all tool calls parsed correctly
        final_answer TEXT,               -- model's final answer
        ts REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_session ON thoughts(session_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_scripts_session ON scripts(session_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_traj_session ON tool_trajectories(session_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_traj_reward ON tool_trajectories(reward)",
]


# DDL statements the LLM is allowed to run via migrate_schema.
_ALLOWED_DDL = ("CREATE TABLE", "ALTER TABLE", "CREATE INDEX", "DROP INDEX",
                "CREATE VIEW", "DROP VIEW")
# Hard-blocked — never let the LLM run these even inside a multi-statement string.
_BLOCKED = ("DROP TABLE", "DELETE FROM", "UPDATE ", "INSERT INTO",
            "PRAGMA ", "ATTACH ", "DETACH ")


class DiscoveryDB:
    """SQLite-backed memory for the discovery self-play loop.

    Thread-safe via a per-call connection (sqlite3 with check_same_thread=False
    is avoided; we open a fresh connection per operation under a lock).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = __import__("threading").Lock()
        self._init_schema()

    # ── connection helpers ───────────────────────────────────────────
    @contextmanager
    def _conn(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            for stmt in _SCHEMA:
                c.execute(stmt)

    # ── events ───────────────────────────────────────────────────────
    def emit(self, kind: str, payload: dict | None = None,
             session_id: str | None = None) -> int:
        """Append an event and return its row id."""
        blob = json.dumps(payload or {}, ensure_ascii=False)
        ts = time.time()
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO events(session_id, kind, payload, ts) VALUES(?,?,?,?)",
                (session_id, kind, blob, ts))
            return cur.lastrowid

    # ── sessions ─────────────────────────────────────────────────────
    def start_session(self, session_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO sessions(id, started) VALUES(?,?)",
                      (session_id, time.time()))

    def end_session(self, session_id: str, summary: str = "") -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE sessions SET ended=?, summary=? WHERE id=?",
                      (time.time(), summary, session_id))

    def last_session(self) -> str | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT id FROM sessions ORDER BY started DESC LIMIT 1").fetchone()
            return row["id"] if row else None

    # ── generic insert helpers ───────────────────────────────────────
    def add_thought(self, session_id: str, kind: str, content: str,
                    parent_id: int | None = None, confidence: float | None = None) -> int:
        ts = time.time()
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO thoughts(session_id, parent_id, kind, content, confidence, ts) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, parent_id, kind, content, confidence, ts))
            return cur.lastrowid

    def add_script(self, session_id: str, code: str, **fields) -> int:
        cols = ["session_id", "code", "ts"]
        vals: list[Any] = [session_id, code, time.time()]
        for k in ("thought_id", "language", "stdout", "stderr",
                  "returncode", "exec_ms"):
            if k in fields and fields[k] is not None:
                cols.append(k); vals.append(fields[k])
        placeholders = ",".join("?" * len(cols))
        with self._lock, self._conn() as c:
            cur = c.execute(
                f"INSERT INTO scripts({','.join(cols)}) VALUES({placeholders})", vals)
            return cur.lastrowid

    def add_research(self, session_id: str, query: str, **fields) -> int:
        cols = ["session_id", "query", "ts"]
        vals: list[Any] = [session_id, query, time.time()]
        for k in ("url", "title", "summary", "raw_snippet"):
            if k in fields and fields[k] is not None:
                cols.append(k); vals.append(fields[k])
        placeholders = ",".join("?" * len(cols))
        with self._lock, self._conn() as c:
            cur = c.execute(
                f"INSERT INTO research({','.join(cols)}) VALUES({placeholders})", vals)
            return cur.lastrowid

    def add_theory(self, session_id: str, statement: str, notes: str = "") -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO theories(session_id, statement, status, notes, ts) "
                "VALUES(?,?, 'open', ?, ?)",
                (session_id, statement, notes, time.time()))
            return cur.lastrowid

    def update_theory(self, theory_id: int, status: str | None = None,
                      delta_for: int = 0, delta_against: int = 0,
                      notes: str | None = None) -> None:
        sets, vals = [], []
        if status is not None:
            sets.append("status=?"); vals.append(status)
        if delta_for:
            sets.append("evidence_for=evidence_for+?"); vals.append(int(delta_for))
        if delta_against:
            sets.append("evidence_against=evidence_against+?"); vals.append(int(delta_against))
        if notes is not None:
            sets.append("notes=?"); vals.append(notes)
        if not sets:
            return
        vals.append(theory_id)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE theories SET {','.join(sets)} WHERE id=?", vals)

    def add_discovery(self, session_id: str, summary: str,
                      theory_id: int | None = None, confidence: float | None = None) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO discoveries(session_id, theory_id, summary, confidence, ts) "
                "VALUES(?,?,?,?,?)",
                (session_id, theory_id, summary, confidence, time.time()))
            return cur.lastrowid

    # ── tool-use trajectories ────────────────────────────────────────
    def add_tool_trajectory(self, session_id: str, task: str,
                            messages: list[dict], tool_calls: list[dict],
                            reward: float, final_answer: str | None = None) -> int:
        """Save a tool-use trajectory for SFT/RL training.

        Args:
            session_id: discovery session.
            task: the user query/task string.
            messages: full Qwen-format conversation (list of message dicts).
            tool_calls: list of {name, args, result, success} per tool call.
            reward: composite reward 0..1.
            final_answer: model's final answer text (if any).
        """
        n_total = len(tool_calls)
        n_ok = sum(1 for tc in tool_calls if tc.get("success"))
        format_ok = 1 if all(tc.get("success") for tc in tool_calls) else 0
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO tool_trajectories"
                "(session_id, task, messages, tool_calls, n_tool_calls, "
                "n_successful, reward, format_ok, final_answer, ts) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (session_id, task, json.dumps(messages, ensure_ascii=False),
                 json.dumps(tool_calls, ensure_ascii=False),
                 n_total, n_ok, reward, format_ok, final_answer, time.time()))
            return cur.lastrowid

    def get_trajectories(self, min_reward: float = 0.5,
                         limit: int = 200) -> list[dict]:
        """Get high-quality tool-use trajectories for SFT training."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT * FROM tool_trajectories WHERE reward >= ? "
                "ORDER BY reward DESC, ts DESC LIMIT ?",
                (min_reward, limit)).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["messages"] = json.loads(d["messages"])
                d["tool_calls"] = json.loads(d["tool_calls"])
                results.append(d)
            return results

    def trajectory_stats(self) -> dict:
        """Aggregate stats for trajectory quality monitoring."""
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) as n, "
                "AVG(reward) as avg_reward, "
                "AVG(n_successful) as avg_success, "
                "SUM(format_ok) as n_format_ok "
                "FROM tool_trajectories").fetchone()
            return dict(row) if row else {"n": 0}

    # ── LLM-facing read/query ────────────────────────────────────────
    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Read-only SELECT for the LLM. Returns list of dicts."""
        stripped = sql.lstrip().upper()
        if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
            raise ValueError("query() is read-only — only SELECT/WITH allowed")
        with self._lock, self._conn() as c:
            rows = c.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def migrate_schema(self, sql: str, reason: str = "",
                       session_id: str | None = None) -> dict:
        """Run LLM-requested DDL, audited into schema_migrations.

        Only additive/safe DDL is allowed (CREATE/ALTER/INDEX/VIEW). DROP TABLE
        and any DML are hard-blocked to stop the model from nuking its own
        memory. Multi-statement strings are split on ';' and checked per-stmt.
        """
        result = {"applied": [], "errors": []}
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            up = stmt.lstrip().upper()
            if any(b in up for b in _BLOCKED):
                err = f"blocked statement: {stmt[:80]}"
                result["errors"].append(err)
                self._log_migration(session_id, stmt, reason, 0, err)
                continue
            if not up.startswith(_ALLOWED_DDL):
                err = f"not allowed DDL: {stmt[:80]}"
                result["errors"].append(err)
                self._log_migration(session_id, stmt, reason, 0, err)
                continue
            try:
                with self._lock, self._conn() as c:
                    c.execute(stmt)
                result["applied"].append(stmt[:120])
                self._log_migration(session_id, stmt, reason, 1, None)
            except sqlite3.Error as e:
                result["errors"].append(str(e))
                self._log_migration(session_id, stmt, reason, 0, str(e))
        return result

    def _log_migration(self, session_id, sql, reason, success, error):
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO schema_migrations(session_id, sql, reason, success, error, ts) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, sql, reason, success, error, time.time()))

    # ── user-facing introspection ────────────────────────────────────
    def table_counts(self) -> dict[str, int]:
        tables = ["sessions", "thoughts", "scripts", "research",
                  "theories", "discoveries", "events", "schema_migrations",
                  "epochs", "distill_runs", "tool_trajectories"]
        out = {}
        with self._lock, self._conn() as c:
            for t in tables:
                out[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return out

    def recent(self, table: str, n: int = 20) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
            return [dict(r) for r in rows]

    # ── savepoint / rollback (stuck-loop recovery) ───────────────────
    # SQLite SAVEPOINTs are connection-scoped, but this DB opens a fresh
    # connection per call (for thread safety). So we use a high-water-mark
    # scheme instead: at checkpoint, record the max rowid of each content
    # table; on rollback, DELETE rows with id > the snapshot. This reverts
    # all writes since the checkpoint, portably, across connections.
    _SP_COUNTER = 0
    _CONTENT_TABLES = ("thoughts", "scripts", "research", "theories",
                       "discoveries", "events", "schema_migrations")

    def savepoint(self) -> str:
        """Snapshot current max rowids. Returns a token for later rollback."""
        with self._lock, self._conn() as c:
            self._SP_COUNTER += 1
            name = f"sp_{self._SP_COUNTER}"
            marks = {}
            for t in self._CONTENT_TABLES:
                row = c.execute(f"SELECT MAX(id) AS m FROM {t}").fetchone()
                marks[t] = row["m"] if row["m"] is not None else 0
            # Also snapshot any LLM-added tables (best-effort).
            try:
                tables = [r[0] for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT IN ('sqlite_sequence')").fetchall()]
                for t in tables:
                    if t not in marks and t not in ("sessions", "epochs", "distill_runs"):
                        try:
                            row = c.execute(f"SELECT MAX(id) AS m FROM {t}").fetchone()
                            marks[t] = row["m"] if row["m"] is not None else 0
                        except sqlite3.Error:
                            pass
            except sqlite3.Error:
                pass
            self._savepoints[name] = marks
            return name

    _savepoints: dict = {}

    def rollback_to(self, name: str) -> None:
        """Delete all rows added since the savepoint (revert the burst)."""
        marks = self._savepoints.get(name)
        if marks is None:
            return
        with self._lock, self._conn() as c:
            for t, max_id in marks.items():
                if max_id <= 0:
                    # Table was empty at checkpoint — wipe any new rows.
                    with suppress(sqlite3.Error):
                        c.execute(f"DELETE FROM {t} WHERE id > 0")
                else:
                    with suppress(sqlite3.Error):
                        c.execute(f"DELETE FROM {t} WHERE id > ?", (max_id,))
        del self._savepoints[name]

    def release(self, name: str) -> None:
        """Release (commit) a savepoint — just drop the snapshot."""
        self._savepoints.pop(name, None)

    # ── fingerprints (anti-regression: block exact repeats) ──────────
    def all_content_rows(self) -> list[dict]:
        """Return normalized content rows for fingerprinting across epochs.

        Used by anti_regression to block the LLM from repeating exact prior
        steps. Covers thoughts, scripts, theories, discoveries, research.
        """
        rows = []
        with self._lock, self._conn() as c:
            for table, col in [("thoughts", "content"), ("scripts", "code"),
                               ("theories", "statement"),
                               ("discoveries", "summary"),
                               ("research", "query")]:
                try:
                    for r in c.execute(f"SELECT {col} FROM {table}").fetchall():
                        rows.append({"table": table, "content": r[col] or ""})
                except sqlite3.Error:
                    pass
        return rows

    # ── epoch / distill bookkeeping ──────────────────────────────────
    def add_epoch(self, epoch_num: int, checkpoint_path: str,
                  parent_epoch: int | None = None, kind: str = "finetune",
                  quality: float | None = None, skill: float | None = None,
                  compute: float | None = None, composite: float | None = None,
                  status: str = "candidate") -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO epochs(epoch_num, checkpoint_path, parent_epoch, kind, "
                "quality, skill, compute, composite, status, ts) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (epoch_num, checkpoint_path, parent_epoch, kind, quality, skill,
                 compute, composite, status, time.time()))
            return cur.lastrowid

    def set_epoch_status(self, epoch_id: int, status: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE epochs SET status=? WHERE id=?", (status, epoch_id))

    def best_epoch(self) -> dict | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM epochs WHERE status='best' ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def last_epoch_num(self) -> int:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT MAX(epoch_num) AS m FROM epochs").fetchone()
            return int(row["m"] or 0)

    def add_distill_run(self, epoch_num: int, from_epoch: int, to_epoch: int,
                        filtered_items: int = 0, bloat_removed: int = 0,
                        notes: str = "") -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO distill_runs(epoch_num, from_epoch, to_epoch, "
                "filtered_items, bloat_removed, notes, ts) VALUES(?,?,?,?,?,?,?)",
                (epoch_num, from_epoch, to_epoch, filtered_items,
                 bloat_removed, notes, time.time()))
            return cur.lastrowid

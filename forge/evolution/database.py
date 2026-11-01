"""Persistent storage for ForgeEvolve findings using SQLite.

Tables:
  - runs: metadata about each search run (domain, config, timestamps)
  - discoveries: configs that entered the MAP-Elites archive
  - evaluations: every (config, score) pair ever evaluated
  - generators: saved generator weights (for resuming/cross-run transfer)
  - surrogate: saved surrogate weights (for warm-starting)

Usage:
  db = FindingsDB("forge_evolve.db")
  db.save_run(run_id, config, results)
  db.save_discoveries(run_id, discoveries_list)
  db.save_generators(run_id, batched_gen)
  db.load_generators(run_id, batched_gen)  # warm-start
  past = db.query_discoveries(domain="quant", min_score=-2.0)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import torch


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalars (avoids default=str)."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _dumps(obj) -> str:
    """JSON dumps with numpy support."""
    return json.dumps(obj, default=_NumpyEncoder().default)


# ── Provenance blob cap (A21) ────────────────────────────────────────────
# forge_evolve_meta_inner.db reached 40GB because discoveries rows stored
# ~1MB script_text/input_text/output_text payloads each — the ledger only
# needs enough text to identify the discovery. FORGE_META_BLOB_CAP caps
# each provenance TEXT column at write time: over-cap values are cut and
# suffixed with "...[truncated]", and the sha256 of the ORIGINAL
# concatenated payloads is recorded in content_sha256 so truncation is
# auditable (ledger integrity, A13). 0 disables the cap entirely.
# scripts/truncate_meta_blobs.py applies the same policy retroactively.
_BLOB_CAP_ENV = "FORGE_META_BLOB_CAP"
_BLOB_CAP_DEFAULT = 2048
TRUNCATED_SUFFIX = "...[truncated]"
_PROVENANCE_COLS = ("script_text", "input_text", "output_text",
                    "expected_text")


def blob_cap_bytes() -> int:
    """Active provenance-blob cap in chars; 0 means 'store full text'."""
    try:
        return int(os.environ.get(_BLOB_CAP_ENV, _BLOB_CAP_DEFAULT))
    except (TypeError, ValueError):
        return _BLOB_CAP_DEFAULT


def truncate_blob(value, cap: int | None = None):
    """Cap a provenance text value. Non-str or under-cap passes through."""
    if not isinstance(value, str):
        return value
    if cap is None:
        cap = blob_cap_bytes()
    if cap <= 0 or len(value) <= cap:
        return value
    return value[:cap] + TRUNCATED_SUFFIX


def provenance_sha256(*texts) -> str:
    """sha256 over the ORIGINAL (pre-truncation) provenance payloads."""
    h = hashlib.sha256()
    for t in texts:
        if isinstance(t, str):
            h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


class FindingsDB:
    """SQLite-backed persistent storage for ForgeEvolve findings."""

    def __init__(self, db_path: str | Path = "forge_evolve.db"):
        if db_path is None:
            # None = "no persistence" — test/frontier callers pass it
            # deliberately. Before, str(None) connected to a file
            # literally named "None" (a 119MB orphan at the repo root);
            # an in-memory db gives the same throwaway semantics
            # without the artifact.
            db_path = ":memory:"
        if str(db_path).strip() in ("", "None"):
            raise ValueError(
                f"FindingsDB needs a real db_path, got {db_path!r}")
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._connect()
        self._init_tables()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # busy_timeout FIRST — parallel campaign runs on a FRESH db race
        # the WAL switch below: the second connect used to fail instantly
        # with "database is locked" before the handler was registered.
        self._conn.execute("PRAGMA busy_timeout=30000")   # wait 30s on lock contention
        # WAL mode + tuned PRAGMAs: 10-100x faster commits, concurrent reads
        for _ in range(50):
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError:
                time.sleep(0.2)
        else:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-65536")    # 64MB cache (was 10MB)
        self._conn.execute("PRAGMA temp_store=MEMORY")    # temp tables in RAM
        self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O

    @property
    def conn(self) -> sqlite3.Connection:
        """Live connection — lazily reconnects after release(). Callers
        (CodeEvolve campaign rounds, Mutator.note) treat FindingsDB as
        long-lived; an open WAL conn per run pins frames, so run()
        releases between rounds and the next access transparently
        reopens. close() is terminal — it raises here."""
        if self._closed:
            raise sqlite3.ProgrammingError(
                "Cannot operate on a closed database.")
        if self._conn is None:
            with self._lock:
                if self._conn is None and not self._closed:
                    self._connect()
        return self._conn

    # Current schema version. Bump when adding columns/tables; migration
    # logic below auto-upgrades existing DBs idempotently.
    SCHEMA_VERSION = "3"

    def _init_tables(self):
        c = self.conn.cursor()

        # ── schema_meta: versioning + scoring-hash registry ──────────────
        # Created first so we can read the schema_version before migrating.
        c.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                config_json TEXT,
                start_time REAL,
                end_time REAL,
                generations INTEGER,
                total_evals INTEGER,
                discoveries INTEGER,
                best_score REAL,
                best_config_json TEXT,
                device TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                generation INTEGER,
                config_json TEXT NOT NULL,
                score REAL NOT NULL,
                behavioral_json TEXT,
                metadata_json TEXT,
                timestamp REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_domain
            ON discoveries(domain)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_score
            ON discoveries(domain, score DESC)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_dedup
            ON discoveries(domain, config_json)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                generation INTEGER,
                config_json TEXT NOT NULL,
                score REAL NOT NULL,
                metadata_json TEXT,
                timestamp REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS generators (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                weights_blob BLOB NOT NULL,
                fitness_blob BLOB NOT NULL,
                config_json TEXT,
                generation INTEGER,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS surrogate (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                weights_blob BLOB NOT NULL,
                n_trained INTEGER,
                config_json TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Canonical generators/surrogate: the best-ever weights per domain,
        # persisted across ALL runs and profiles. Not keyed by run_id —
        # keyed by domain only. Updated only when a run beats the stored
        # best_score. This is the "permanent knowledge" layer.
        c.execute("""
            CREATE TABLE IF NOT EXISTS canonical_generators (
                domain TEXT PRIMARY KEY,
                weights_blob BLOB NOT NULL,
                fitness_blob BLOB NOT NULL,
                config_json TEXT,
                best_score REAL NOT NULL,
                run_id TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS canonical_surrogate (
                domain TEXT PRIMARY KEY,
                weights_blob BLOB NOT NULL,
                n_trained INTEGER,
                config_json TEXT,
                best_score REAL NOT NULL,
                run_id TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # ── Gen model storage (LLM checkpoints for curriculum fine-tuning) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS gen_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                config_json TEXT,
                weights_blob BLOB,
                param_count INTEGER,
                performance_score REAL,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_gen_models_version
            ON gen_models(version, timestamp DESC)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS gen_model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                domain TEXT NOT NULL,
                round INTEGER,
                score REAL,
                param_count INTEGER,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_gen_model_perf_version
            ON gen_model_performance(version, timestamp DESC)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_gen_model_perf_domain
            ON gen_model_performance(domain, timestamp DESC)
        """)

        self.conn.commit()

        # ── Auto-migrate: add provenance columns to existing DBs ──────────
        # Idempotent: checks PRAGMA table_info and only adds missing columns.
        self._migrate_schema()

    # ── Schema migration ────────────────────────────────────────────────

    def _table_columns(self, table: str) -> set[str]:
        """Return the set of column names currently on a table."""
        c = self.conn.cursor()
        c.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in c.fetchall()}

    def _add_column_if_missing(self, table: str, col_def: str) -> None:
        """Add a column (col_def = 'name TYPE ...') if it doesn't exist.

        SQLite's ALTER TABLE ADD COLUMN is idempotent-safe when guarded by
        a column-existence check, so this can be called on every init.
        """
        cols = self._table_columns(table)
        col_name = col_def.split()[0]
        if col_name not in cols:
            c = self.conn.cursor()
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
            self.conn.commit()

    def _migrate_schema(self) -> None:
        """Idempotent migration: add provenance columns to discoveries and
        record the schema_version in schema_meta. Safe to run repeatedly."""
        c = self.conn.cursor()

        # ── v2: provenance columns on discoveries ────────────────────────
        provenance_cols = [
            "script_text TEXT",
            "input_text TEXT",
            "output_text TEXT",
            "expected_text TEXT",
            "gen_model_size INTEGER",
            "gen_model_version TEXT",
            "scoring_hash TEXT",
            "applied INTEGER DEFAULT 0",
        ]
        for col_def in provenance_cols:
            self._add_column_if_missing("discoveries", col_def)

        # Index for curriculum queries (score + provenance availability)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_curriculum
            ON discoveries(domain, score DESC)
            WHERE input_text IS NOT NULL AND output_text IS NOT NULL
        """)

        # ── v3: content_sha256 on discoveries (A21) ─────────────────────
        # Hex sha256 of the ORIGINAL concatenated provenance texts, set
        # only when a payload was truncated under FORGE_META_BLOB_CAP —
        # NULL means 'stored verbatim'. Lets the 40GB-meta_inner cleanup
        # keep ledger integrity after payloads are cut.
        self._add_column_if_missing("discoveries", "content_sha256 TEXT")

        # Record schema_version (idempotent upsert)
        c.execute("""
            INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (self.SCHEMA_VERSION,))
        self.conn.commit()

    def save_run(self, run_id: str, domain: str, config: dict,
                 results: dict, start_time: float):
        """Save or update a run record."""
        with self._lock:
            c = self.conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO runs
                (run_id, domain, config_json, start_time, end_time,
                 generations, total_evals, discoveries, best_score,
                 best_config_json, device)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, domain, _dumps(config),
                start_time, time.time(),
                results.get("generations", 0),
                results.get("total_evaluations", 0),
                results.get("discoveries", 0),
                results.get("best_score", 0),
                _dumps(results.get("best_config")),
                results.get("device", "unknown"),
            ))
            self.conn.commit()

    def save_discoveries(self, run_id: str, domain: str,
                         discoveries: list[dict]):
        """Save discoveries from a run. One row per unique (domain, config).
        If a config already exists, UPDATE its score if the new one is higher
        (don't insert a duplicate). This keeps the DB strictly unique.
        Thread-safe: acquires write lock to prevent concurrent write conflicts.

        Each discovery dict may optionally include provenance fields:
            script_text, input_text, output_text, expected_text,
            gen_model_size, gen_model_version, scoring_hash
        These are stored in the new provenance columns. Existing callers
        that omit them continue to work (NULL is stored).
        Provenance text over FORGE_META_BLOB_CAP chars is truncated with a
        "...[truncated]" marker; the original payload's sha256 lands in
        content_sha256 (A21 — see module header for the 40GB rationale).
        """
        with self._lock:
            c = self.conn.cursor()
            n_saved = 0
            n_updated = 0
            n_skipped = 0
            cap = blob_cap_bytes()
            for d in discoveries:
                config_json = _dumps(d.get("config"))
                score = d.get("score", 0)
                # Provenance fields (optional — default to None)
                script_text = d.get("script_text")
                input_text = d.get("input_text")
                output_text = d.get("output_text")
                expected_text = d.get("expected_text")
                gen_model_size = d.get("gen_model_size")
                gen_model_version = d.get("gen_model_version")
                scoring_hash = d.get("scoring_hash")
                # A21 payload cap: over-cap provenance text is truncated
                # and the sha of the ORIGINAL concatenation is recorded —
                # NULL content_sha256 means 'stored verbatim'.
                content_sha256 = None
                if cap > 0:
                    originals = (script_text, input_text,
                                 output_text, expected_text)
                    if any(isinstance(t, str) and len(t) > cap
                           for t in originals):
                        content_sha256 = provenance_sha256(*originals)
                        script_text = truncate_blob(script_text, cap)
                        input_text = truncate_blob(input_text, cap)
                        output_text = truncate_blob(output_text, cap)
                        expected_text = truncate_blob(expected_text, cap)
                # Check if this exact config already exists for this domain
                c.execute(
                    "SELECT id, score FROM discoveries WHERE domain=? AND config_json=? "
                    "ORDER BY score DESC LIMIT 1",
                    (domain, config_json))
                row = c.fetchone()
                if row is not None:
                    existing_id, existing_score = row[0], row[1]
                    if score > existing_score:
                        # Update the existing row with the better score + metadata
                        # + provenance (only overwrite provenance if new values
                        # are provided, to avoid clobbering existing data).
                        c.execute("""
                            UPDATE discoveries
                            SET run_id=?, generation=?, score=?,
                                behavioral_json=?, metadata_json=?,
                                script_text=COALESCE(?, script_text),
                                input_text=COALESCE(?, input_text),
                                output_text=COALESCE(?, output_text),
                                expected_text=COALESCE(?, expected_text),
                                gen_model_size=COALESCE(?, gen_model_size),
                                gen_model_version=COALESCE(?, gen_model_version),
                                scoring_hash=COALESCE(?, scoring_hash),
                                content_sha256=COALESCE(?, content_sha256)
                            WHERE id=?
                        """, (
                            run_id, d.get("generation", 0), score,
                            _dumps(d.get("behavioral")),
                            _dumps(d.get("metadata", {})),
                            script_text, input_text, output_text, expected_text,
                            gen_model_size, gen_model_version, scoring_hash,
                            content_sha256,
                            existing_id,
                        ))
                        n_updated += 1
                    else:
                        n_skipped += 1
                    continue
                c.execute("""
                    INSERT INTO discoveries
                    (run_id, domain, generation, config_json, score,
                     behavioral_json, metadata_json,
                     script_text, input_text, output_text, expected_text,
                     gen_model_size, gen_model_version, scoring_hash,
                     content_sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, domain, d.get("generation", 0),
                    config_json,
                    score,
                    _dumps(d.get("behavioral")),
                    _dumps(d.get("metadata", {})),
                    script_text, input_text, output_text, expected_text,
                    gen_model_size, gen_model_version, scoring_hash,
                    content_sha256,
                ))
                n_saved += 1
            self.conn.commit()
        return n_saved, n_skipped + n_updated

    def save_evaluation(self, run_id: str, domain: str, generation: int,
                        config: dict, score: float, metadata: dict):
        """Save a single evaluation (for incremental saves during run)."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO evaluations
            (run_id, domain, generation, config_json, score, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id, domain, generation,
            _dumps(config), score,
            _dumps(metadata),
        ))
        self.conn.commit()

    def save_generators(self, run_id: str, domain: str, batched_gen):
        """Save generator weights for resuming or cross-run transfer."""
        weights = {}
        for name, p in batched_gen.named_parameters():
            weights[name] = p.detach().cpu().numpy()
        fitness = batched_gen.fitness_ema.detach().cpu().numpy()

        config = {
            "noise_dim": batched_gen.cfg.noise_dim,
            "context_dim": batched_gen.cfg.context_dim,
            "hidden_dim": batched_gen.cfg.hidden_dim,
            "output_dim": batched_gen.cfg.output_dim,
            "n_generators": batched_gen.cfg.n_generators,
        }

        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO generators
            (run_id, domain, weights_blob, fitness_blob, config_json, generation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id, domain,
            pickle.dumps(weights),
            pickle.dumps(fitness),
            json.dumps(config),
            0,  # TODO: track actual generation
        ))
        self.conn.commit()

    def load_generators(self, run_id: str, batched_gen) -> bool:
        """Load generator weights from a past run. Returns True if loaded."""
        c = self.conn.cursor()
        c.execute("SELECT weights_blob, fitness_blob FROM generators WHERE run_id=?",
                  (run_id,))
        row = c.fetchone()
        if row is None:
            return False

        weights = pickle.loads(row[0])
        fitness = pickle.loads(row[1])

        with torch.no_grad():
            for name, p in batched_gen.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape != p.shape:
                        continue  # skip mismatched shapes (e.g. different output_dim)
                    p.copy_(saved)
            if fitness.shape == batched_gen.fitness_ema.shape:
                batched_gen.fitness_ema.copy_(
                    torch.from_numpy(fitness).to(batched_gen.fitness_ema.device)
                )
        return True

    def save_surrogate(self, run_id: str, domain: str, surrogate):
        """Save surrogate network weights for warm-starting."""
        if surrogate.mode != "mlp":
            return  # only MLP supported

        weights = {}
        for name, p in surrogate.net.named_parameters():
            weights[name] = p.detach().cpu().numpy()

        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO surrogate
            (run_id, domain, weights_blob, n_trained, config_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            run_id, domain,
            pickle.dumps(weights),
            surrogate.n_trained,
            json.dumps({"mode": surrogate.mode, "input_dim": surrogate.input_dim}),
        ))
        self.conn.commit()

    def load_surrogate(self, run_id: str, surrogate) -> bool:
        """Load surrogate weights. Returns True if loaded."""
        if surrogate.mode != "mlp":
            return False

        c = self.conn.cursor()
        c.execute("SELECT weights_blob, n_trained FROM surrogate WHERE run_id=?",
                  (run_id,))
        row = c.fetchone()
        if row is None:
            return False

        weights = pickle.loads(row[0])
        surrogate.n_trained = row[1]

        with torch.no_grad():
            for name, p in surrogate.net.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape != p.shape:
                        continue  # skip mismatched shapes (e.g. different input_dim)
                    p.copy_(saved)
        return True

    # ── Canonical generators/surrogate (permanent knowledge layer) ──────

    def save_canonical_generators(self, domain: str, batched_gen,
                                  best_score: float, run_id: str) -> bool:
        """Save generators as canonical for this domain if best_score beats
        the stored value. Returns True if updated."""
        c = self.conn.cursor()
        c.execute("SELECT best_score FROM canonical_generators WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is not None and row[0] >= best_score:
            return False  # stored is better, don't overwrite

        weights = {}
        for name, p in batched_gen.named_parameters():
            weights[name] = p.detach().cpu().numpy()
        fitness = batched_gen.fitness_ema.detach().cpu().numpy()
        config = {
            "noise_dim": batched_gen.cfg.noise_dim,
            "context_dim": batched_gen.cfg.context_dim,
            "hidden_dim": batched_gen.cfg.hidden_dim,
            "output_dim": batched_gen.cfg.output_dim,
            "n_generators": batched_gen.cfg.n_generators,
        }
        c.execute("""
            INSERT OR REPLACE INTO canonical_generators
            (domain, weights_blob, fitness_blob, config_json, best_score, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (domain, pickle.dumps(weights), pickle.dumps(fitness),
              json.dumps(config), best_score, run_id))
        self.conn.commit()
        return True

    def load_canonical_generators(self, domain: str, batched_gen) -> bool:
        """Load canonical generators for a domain. Returns True if loaded."""
        c = self.conn.cursor()
        c.execute("SELECT weights_blob, fitness_blob FROM canonical_generators WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is None:
            return False
        weights = pickle.loads(row[0])
        fitness = pickle.loads(row[1])
        with torch.no_grad():
            for name, p in batched_gen.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape == p.shape:
                        p.copy_(saved)
            if fitness.shape == batched_gen.fitness_ema.shape:
                batched_gen.fitness_ema.copy_(
                    torch.from_numpy(fitness).to(batched_gen.fitness_ema.device))
        return True

    def save_canonical_surrogate(self, domain: str, surrogate,
                                 best_score: float, run_id: str) -> bool:
        """Save surrogate as canonical for this domain if best_score beats
        the stored value. Returns True if updated."""
        if surrogate.mode != "mlp":
            return False
        c = self.conn.cursor()
        c.execute("SELECT best_score FROM canonical_surrogate WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is not None and row[0] >= best_score:
            return False

        weights = {}
        for name, p in surrogate.net.named_parameters():
            weights[name] = p.detach().cpu().numpy()
        c.execute("""
            INSERT OR REPLACE INTO canonical_surrogate
            (domain, weights_blob, n_trained, config_json, best_score, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (domain, pickle.dumps(weights), surrogate.n_trained,
              json.dumps({"mode": surrogate.mode, "input_dim": surrogate.input_dim}),
              best_score, run_id))
        self.conn.commit()
        return True

    def load_canonical_surrogate(self, domain: str, surrogate) -> bool:
        """Load canonical surrogate for a domain. Returns True if loaded."""
        if surrogate.mode != "mlp":
            return False
        c = self.conn.cursor()
        c.execute("SELECT weights_blob, n_trained FROM canonical_surrogate WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is None:
            return False
        weights = pickle.loads(row[0])
        surrogate.n_trained = row[1]
        with torch.no_grad():
            for name, p in surrogate.net.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape == p.shape:
                        p.copy_(saved)
        return True

    def get_canonical_best_score(self, domain: str) -> float | None:
        """Get the best score ever stored canonically for a domain."""
        c = self.conn.cursor()
        c.execute("SELECT best_score FROM canonical_generators WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        return row[0] if row else None

    def query_discoveries(self, domain: str, min_score: float = -1e9,
                          limit: int = 100) -> list[dict]:
        """Query past discoveries for a domain, sorted by score.

        Includes provenance fields (script_text, input_text, output_text,
        expected_text, gen_model_size, gen_model_version, scoring_hash)
        when present.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT config_json, score, behavioral_json, metadata_json,
                   run_id, generation,
                   script_text, input_text, output_text, expected_text,
                   gen_model_size, gen_model_version, scoring_hash,
                   content_sha256
            FROM discoveries
            WHERE domain=? AND score >= ?
            ORDER BY score DESC
            LIMIT ?
        """, (domain, min_score, limit))
        rows = c.fetchall()

        return [{
            "config": json.loads(r[0]),
            "score": r[1],
            "behavioral": json.loads(r[2]) if r[2] else None,
            "metadata": json.loads(r[3]) if r[3] else {},
            "run_id": r[4],
            "generation": r[5],
            "script_text": r[6],
            "input_text": r[7],
            "output_text": r[8],
            "expected_text": r[9],
            "gen_model_size": r[10],
            "gen_model_version": r[11],
            "scoring_hash": r[12],
            "content_sha256": r[13],
        } for r in rows]

    def query_best_configs(self, domain: str, limit: int = 10) -> list[dict]:
        """Get the best configs ever found for a domain."""
        return self.query_discoveries(domain, min_score=-1e9, limit=limit)

    # ── Applied-flag tracking (which discoveries have been promoted) ─────

    def get_unapplied_discoveries(self, domain: str | None = None,
                                  limit: int = 100) -> list[dict]:
        """Return discoveries that haven't been applied yet (applied = 0 or
        NULL), sorted by score DESC.

        If ``domain`` is None, query across all domains. Each dict has keys:
        id, domain, config (parsed), score, behavioral (parsed),
        metadata (parsed).
        """
        c = self.conn.cursor()
        if domain is not None:
            c.execute("""
                SELECT id, domain, config_json, score,
                       behavioral_json, metadata_json
                FROM discoveries
                WHERE domain=? AND (applied = 0 OR applied IS NULL)
                ORDER BY score DESC
                LIMIT ?
            """, (domain, limit))
        else:
            c.execute("""
                SELECT id, domain, config_json, score,
                       behavioral_json, metadata_json
                FROM discoveries
                WHERE applied = 0 OR applied IS NULL
                ORDER BY score DESC
                LIMIT ?
            """, (limit,))
        rows = c.fetchall()
        results = []
        for r in rows:
            try:
                config = json.loads(r[2])
            except (json.JSONDecodeError, TypeError):
                config = None
            try:
                behavioral = json.loads(r[4]) if r[4] else None
            except (json.JSONDecodeError, TypeError):
                behavioral = None
            try:
                metadata = json.loads(r[5]) if r[5] else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            results.append({
                "id": r[0],
                "domain": r[1],
                "config": config,
                "score": r[3],
                "behavioral": behavioral,
                "metadata": metadata,
            })
        return results

    def mark_applied(self, discovery_ids: list[int]) -> None:
        """Mark the given discovery IDs as applied (applied = 1).

        Uses a single UPDATE with WHERE id IN (...). No-op if the list is
        empty.
        """
        if not discovery_ids:
            return
        with self._lock:
            c = self.conn.cursor()
            placeholders = ",".join("?" for _ in discovery_ids)
            c.execute(
                f"UPDATE discoveries SET applied = 1 WHERE id IN ({placeholders})",
                tuple(discovery_ids),
            )
            self.conn.commit()

    def mark_all_applied(self, domain: str | None = None) -> int:
        """Mark all unapplied discoveries as applied, optionally filtered by
        domain. Returns the count of updated rows."""
        with self._lock:
            c = self.conn.cursor()
            if domain is not None:
                c.execute("""
                    UPDATE discoveries SET applied = 1
                    WHERE domain=? AND (applied = 0 OR applied IS NULL)
                """, (domain,))
            else:
                c.execute("""
                    UPDATE discoveries SET applied = 1
                    WHERE applied = 0 OR applied IS NULL
                """)
            n = c.rowcount
            self.conn.commit()
        return n

    def count_unapplied(self, domain: str | None = None) -> int:
        """Return the count of discoveries where applied = 0 or NULL."""
        c = self.conn.cursor()
        if domain is not None:
            c.execute("""
                SELECT COUNT(*) FROM discoveries
                WHERE domain=? AND (applied = 0 OR applied IS NULL)
            """, (domain,))
        else:
            c.execute("""
                SELECT COUNT(*) FROM discoveries
                WHERE applied = 0 OR applied IS NULL
            """)
        row = c.fetchone()
        return row[0] if row else 0

    def seed_from_past(self, domain: str, batched_gen, surrogate,
                       n_seed: int = 5) -> int:
        """Warm-start generators + surrogate from best past findings.

        Loads the best past configs, evaluates them to set context,
        and loads surrogate weights if available.

        Returns number of seeds loaded.
        """
        # Load best past discoveries
        past = self.query_best_configs(domain, limit=n_seed)
        if not past:
            return 0

        # Load surrogate weights from most recent run
        c = self.conn.cursor()
        c.execute("""
            SELECT run_id FROM surrogate WHERE domain=?
            ORDER BY timestamp DESC LIMIT 1
        """, (domain,))
        surr_row = c.fetchone()
        if surr_row:
            self.load_surrogate(surr_row[0], surrogate)

        # Load generator weights from most recent run
        c.execute("""
            SELECT run_id FROM generators WHERE domain=?
            ORDER BY timestamp DESC LIMIT 1
        """, (domain,))
        gen_row = c.fetchone()
        if gen_row:
            self.load_generators(gen_row[0], batched_gen)

        return len(past)

    def seed_cross_domain(self, domain: str, batched_gen, surrogate,
                          output_dim: int) -> int:
        """Cross-domain transfer: load generators from a different domain
        with the same output_dim. The generator learns the parameter
        distribution, which transfers across domains with similar dim.

        Returns number of source domains transferred from, or 0 if none.
        """
        c = self.conn.cursor()

        # Find the best-scoring run from ANY domain with matching output_dim
        # by checking the generator weights shape
        c.execute("""
            SELECT g.run_id, g.domain, g.weights_blob, g.fitness_blob,
                   r.best_score
            FROM generators g
            JOIN runs r ON g.run_id = r.run_id
            WHERE g.domain != ?
            ORDER BY r.best_score DESC
            LIMIT 10
        """, (domain,))
        rows = c.fetchall()

        transferred = 0
        for row in rows:
            run_id, src_domain, weights_blob, fitness_blob, best_score = row
            try:
                weights = pickle.loads(weights_blob)
                # Check if output_dim matches (W2 last dim = output_dim)
                w2_key = [k for k in weights if k.endswith("W2") or "W2" in k]
                if not w2_key:
                    continue
                src_out_dim = weights[w2_key[0]].shape[-1]
                if src_out_dim != output_dim:
                    continue

                # Load weights — this transfers learned parameter distribution
                with torch.no_grad():
                    for name, p in batched_gen.named_parameters():
                        if name in weights:
                            src = torch.from_numpy(weights[name])
                            if src.shape == p.shape:
                                p.copy_(src.to(p.device))
                    fitness = pickle.loads(fitness_blob)
                    if fitness.shape == batched_gen.fitness_ema.shape:
                        batched_gen.fitness_ema.copy_(
                            torch.from_numpy(fitness).to(batched_gen.fitness_ema.device))

                # Add small noise to break symmetry (domain-specific adaptation)
                    for p in batched_gen.parameters():
                        p.add_(torch.randn_like(p) * 0.05)

                transferred += 1
                # Only transfer from the best source (first match)
                break
            except Exception as e:
                print(f"  [ForgeEvolve] Cross-domain transfer failed: {e}")
                continue

        return transferred

    # ── Gen model storage (LLM checkpoints for curriculum fine-tuning) ───

    def save_gen_model(self, version: str, config: dict, weights,
                       param_count: int, performance_score: float) -> bool:
        """Save a gen model checkpoint (LLM weights) to the DB.

        ``weights`` may be:
          - a state_dict (dict[str, Tensor]) — pickled directly
          - an nn.Module — named_parameters() are extracted
          - bytes/BLOB — stored as-is
        Returns True on success.
        """
        c = self.conn.cursor()
        if isinstance(weights, dict) and not all(
            isinstance(v, (bytes, bytearray)) for v in weights.values()
        ):
            # state_dict of tensors → pickle
            blob = pickle.dumps(weights)
        elif hasattr(weights, "state_dict"):
            blob = pickle.dumps(weights.state_dict())
        elif hasattr(weights, "named_parameters"):
            blob = pickle.dumps({
                name: p.detach().cpu().numpy()
                for name, p in weights.named_parameters()
            })
        elif isinstance(weights, (bytes, bytearray)):
            blob = bytes(weights)
        else:
            blob = pickle.dumps(weights)

        c.execute("""
            INSERT INTO gen_models
            (version, config_json, weights_blob, param_count,
             performance_score)
            VALUES (?, ?, ?, ?, ?)
        """, (
            version, _dumps(config), blob,
            int(param_count), float(performance_score),
        ))
        self.conn.commit()
        return True

    def load_gen_model(self, version: str) -> dict | None:
        """Load the most recent gen model checkpoint for a version.

        Returns {"version", "config", "weights", "param_count",
                 "performance_score", "timestamp"} or None if not found.
        ``weights`` is the pickled state_dict (caller loads into model).
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT version, config_json, weights_blob, param_count,
                   performance_score, timestamp
            FROM gen_models WHERE version=?
            ORDER BY timestamp DESC LIMIT 1
        """, (version,))
        row = c.fetchone()
        if row is None:
            return None
        return {
            "version": row[0],
            "config": json.loads(row[1]) if row[1] else {},
            "weights": pickle.loads(row[2]),
            "param_count": row[3],
            "performance_score": row[4],
            "timestamp": row[5],
        }

    def get_latest_gen_model(self) -> dict | None:
        """Get the most recently saved gen model checkpoint (any version)."""
        c = self.conn.cursor()
        c.execute("""
            SELECT version, config_json, weights_blob, param_count,
                   performance_score, timestamp
            FROM gen_models
            ORDER BY timestamp DESC LIMIT 1
        """)
        row = c.fetchone()
        if row is None:
            return None
        return {
            "version": row[0],
            "config": json.loads(row[1]) if row[1] else {},
            "weights": pickle.loads(row[2]),
            "param_count": row[3],
            "performance_score": row[4],
            "timestamp": row[5],
        }

    def record_gen_model_performance(self, version: str, domain: str,
                                     round: int, score: float,
                                     param_count: int) -> None:
        """Record a gen model's performance on a domain at a given round."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO gen_model_performance
            (version, domain, round, score, param_count)
            VALUES (?, ?, ?, ?, ?)
        """, (version, domain, int(round), float(score), int(param_count)))
        self.conn.commit()

    def get_gen_model_performance_history(self,
                                          version: str | None = None
                                          ) -> list[dict]:
        """Get performance history for a gen model version (or all)."""
        c = self.conn.cursor()
        if version is not None:
            c.execute("""
                SELECT version, domain, round, score, param_count, timestamp
                FROM gen_model_performance WHERE version=?
                ORDER BY round ASC, timestamp ASC
            """, (version,))
        else:
            c.execute("""
                SELECT version, domain, round, score, param_count, timestamp
                FROM gen_model_performance
                ORDER BY timestamp ASC
            """)
        rows = c.fetchall()
        return [{
            "version": r[0], "domain": r[1], "round": r[2],
            "score": r[3], "param_count": r[4], "timestamp": r[5],
        } for r in rows]

    def get_curriculum_data(self, min_score: float = 0.0,
                            limit: int = 1000) -> list[dict]:
        """Return successful discoveries with input_text + output_text for
        fine-tuning. Sorted by score ASC (easiest first for curriculum).

        Each dict has: domain, score, input_text, output_text, expected_text,
        script_text, gen_model_version, run_id, generation.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT domain, score, input_text, output_text, expected_text,
                   script_text, gen_model_version, run_id, generation
            FROM discoveries
            WHERE score >= ? AND input_text IS NOT NULL
                  AND output_text IS NOT NULL
            ORDER BY score ASC
            LIMIT ?
        """, (min_score, limit))
        rows = c.fetchall()
        return [{
            "domain": r[0], "score": r[1],
            "input_text": r[2], "output_text": r[3],
            "expected_text": r[4], "script_text": r[5],
            "gen_model_version": r[6], "run_id": r[7],
            "generation": r[8],
        } for r in rows]

    # ── Scoring hash regression guard ───────────────────────────────────

    def save_scoring_hash(self, domain: str, hash: str) -> None:
        """Store the scoring-spec hash for a domain in schema_meta."""
        c = self.conn.cursor()
        key = f"scoring_hash_{domain}"
        c.execute("""
            INSERT INTO schema_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, hash))
        self.conn.commit()

    def get_scoring_hash(self, domain: str) -> str | None:
        """Retrieve the stored scoring-spec hash for a domain."""
        c = self.conn.cursor()
        c.execute(
            "SELECT value FROM schema_meta WHERE key=?",
            (f"scoring_hash_{domain}",))
        row = c.fetchone()
        return row[0] if row else None

    def check_scoring_compatibility(self, domain: str,
                                    current_hash: str) -> bool:
        """Return True if the stored scoring hash matches the current one.

        Returns False if the hash changed (scoring spec was modified since
        the last run → discoveries need rescoring) or if no hash is stored
        yet (first run for this domain — caller should save it).
        """
        stored = self.get_scoring_hash(domain)
        if stored is None:
            return False
        return stored == current_hash

    def list_runs(self, domain: str | None = None) -> list[dict]:
        """List all runs, optionally filtered by domain."""
        c = self.conn.cursor()
        if domain:
            c.execute("""
                SELECT run_id, domain, generations, total_evals,
                       discoveries, best_score, device, end_time
                FROM runs WHERE domain=? ORDER BY end_time DESC
            """, (domain,))
        else:
            c.execute("""
                SELECT run_id, domain, generations, total_evals,
                       discoveries, best_score, device, end_time
                FROM runs ORDER BY end_time DESC
            """)
        rows = c.fetchall()
        return [{
            "run_id": r[0], "domain": r[1], "generations": r[2],
            "total_evals": r[3], "discoveries": r[4], "best_score": r[5],
            "device": r[6], "end_time": r[7],
        } for r in rows]

    def release(self) -> None:
        """Transient disconnect — drops the WAL handle between campaign
        rounds; the next conn access lazily reconnects. Idempotent."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def close(self):
        """Terminal close — conn access raises after this."""
        with self._lock:
            self._closed = True
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        # Check _conn directly — hasattr(self, 'conn') would run the
        # property and OPEN a fresh connection just to close it.
        if getattr(self, "_conn", None) is not None:
            self.close()


"""One-shot migration: copy findings from old per-profile DBs into the
shared forge_evolve.db and populate canonical_generators/surrogate tables."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RESULTS = "D:/windsurf/ForgeAI/research/results"
SHARED_DB = f"{RESULTS}/forge_evolve.db"
SOURCE_DBS = [
    f"{RESULTS}/forge_evolve_boot.db",
    f"{RESULTS}/forge_evolve_deep.db",
    f"{RESULTS}/forge_evolve_all.db",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_generators (
    domain TEXT PRIMARY KEY, weights_blob BLOB NOT NULL, fitness_blob BLOB NOT NULL,
    config_json TEXT, best_score REAL NOT NULL, run_id TEXT,
    timestamp REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS canonical_surrogate (
    domain TEXT PRIMARY KEY, weights_blob BLOB NOT NULL, n_trained INTEGER,
    config_json TEXT, best_score REAL NOT NULL, run_id TEXT,
    timestamp REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS generators (
    run_id TEXT PRIMARY KEY, domain TEXT NOT NULL, weights_blob BLOB NOT NULL,
    fitness_blob BLOB NOT NULL, config_json TEXT, generation INTEGER,
    timestamp REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS surrogate (
    run_id TEXT PRIMARY KEY, domain TEXT NOT NULL, weights_blob BLOB NOT NULL,
    n_trained INTEGER, config_json TEXT, timestamp REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, domain TEXT NOT NULL, config_json TEXT,
    start_time REAL, end_time REAL, generations INTEGER, total_evals INTEGER,
    discoveries INTEGER, best_score REAL, best_config_json TEXT, device TEXT);
CREATE TABLE IF NOT EXISTS discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, domain TEXT NOT NULL,
    generation INTEGER, config_json TEXT NOT NULL, score REAL NOT NULL,
    behavioral_json TEXT, metadata_json TEXT, timestamp REAL DEFAULT (strftime('%s','now')));
"""

dst = sqlite3.connect(SHARED_DB)
dst.executescript(SCHEMA)
dst.commit()

for src_path in SOURCE_DBS:
    if not os.path.exists(src_path):
        print(f"Skip (not found): {src_path}")
        continue
    print(f"\nMigrating from: {src_path}")
    src = sqlite3.connect(src_path)

    for table in ["runs", "generators", "surrogate", "discoveries"]:
        try:
            cursor = src.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            if not rows:
                continue
            cols = [d[0] for d in cursor.description]
            col_list = ",".join(cols)
            placeholders = ",".join(["?"] * len(cols))
            dst.executemany(
                f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})",
                rows
            )
            print(f"  Copied {len(rows)} rows from {table}")
        except Exception as e:
            print(f"  Skip {table}: {e}")

    # Populate canonical from best runs per domain
    sc = src.cursor()
    domains_with_gens = sc.execute("SELECT DISTINCT domain FROM generators").fetchall()
    migrated = 0
    for (domain,) in domains_with_gens:
        best = sc.execute(
            "SELECT g.run_id, g.weights_blob, g.fitness_blob, r.best_score "
            "FROM generators g JOIN runs r ON g.run_id = r.run_id "
            "WHERE g.domain=? ORDER BY r.best_score DESC LIMIT 1",
            (domain,)
        ).fetchone()
        if not best or best[3] is None:
            continue
        run_id, wblob, fblob, best_score = best
        # Only update canonical if this beats what's already there
        existing = dst.execute(
            "SELECT best_score FROM canonical_generators WHERE domain=?", (domain,)
        ).fetchone()
        if existing and existing[0] >= best_score:
            continue  # stored is better
        dst.execute(
            "INSERT OR REPLACE INTO canonical_generators "
            "(domain, weights_blob, fitness_blob, best_score, run_id) "
            "VALUES (?,?,?,?,?)",
            (domain, wblob, fblob, best_score, run_id)
        )
        surr = sc.execute(
            "SELECT weights_blob, n_trained FROM surrogate WHERE run_id=?",
            (run_id,)
        ).fetchone()
        if surr:
            dst.execute(
                "INSERT OR REPLACE INTO canonical_surrogate "
                "(domain, weights_blob, n_trained, best_score, run_id) "
                "VALUES (?,?,?,?,?)",
                (domain, surr[0], surr[1], best_score, run_id)
            )
        migrated += 1
    print(f"  Migrated canonical for {migrated} domains")
    src.close()

dst.commit()

# Summary
count_gen = dst.execute("SELECT COUNT(*) FROM canonical_generators").fetchone()[0]
count_surr = dst.execute("SELECT COUNT(*) FROM canonical_surrogate").fetchone()[0]
count_disc = dst.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
print(f"\nShared DB ready: {count_gen} canonical generators, "
      f"{count_surr} canonical surrogates, {count_disc} discoveries")
dst.close()

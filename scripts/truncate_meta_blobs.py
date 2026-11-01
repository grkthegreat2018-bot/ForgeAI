"""One-shot payload truncation for findings DBs (audit A21).

``forge_evolve_meta_inner.db`` grew to ~40GB because ``discoveries`` rows
store ~1MB ``script_text``/``input_text``/``output_text``/``expected_text``
payloads each (41K rows, all synthetic-domain, applied=0). This script
applies the same policy the writer now enforces at INSERT time
(FORGE_META_BLOB_CAP, default 2048 chars) retroactively, in place:

  - over-cap TEXT columns are cut to ``cap`` chars + "...[truncated]"
  - the sha256 of the ORIGINAL concatenated payloads is stored in
    ``content_sha256`` (schema v3 column; added here if missing) so the
    ledger keeps a verifiable fingerprint of what was removed
  - NO VACUUM: freelist is zero in this DB and the file would stay ~40GB
    while holding an exclusive lock; freed pages land on the freelist and
    get reused by future writes instead.

Usage:
    python scripts/truncate_meta_blobs.py                      # meta_inner db
    python scripts/truncate_meta_blobs.py --db forge_evolve.db --dry-run
    python scripts/truncate_meta_blobs.py --cap 4096 --chunk 500 --no-sha
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.evolution.database import (  # noqa: E402
    _PROVENANCE_COLS, TRUNCATED_SUFFIX, truncate_blob, blob_cap_bytes)

_UPDATE_COLS = list(_PROVENANCE_COLS)


def _sha256_texts(*texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        if isinstance(t, str):
            h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f}GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="forge_evolve_meta_inner.db",
                    help="Findings DB to truncate in place")
    ap.add_argument("--cap", type=int, default=None,
                    help="Max chars kept per provenance column "
                         "(default: FORGE_META_BLOB_CAP env or 2048)")
    ap.add_argument("--chunk", type=int, default=256,
                    help="Rows per transaction (default 256)")
    ap.add_argument("--no-sha", action="store_true",
                    help="Skip content_sha256 backfill (faster, SQL-only — "
                         "no blob transfer through Python)")
    ap.add_argument("--purge-generator-weights", action="store_true",
                    help="DELETE all rows from the generators table — the "
                         "actual ~40GB payload in meta_inner dbs (682 "
                         "inner-run weights_blob snapshots ~58MB avg; "
                         "inner runs are throwaway, see meta_domains.py "
                         "'Inner runs need zero persistence'). "
                         "DESTRUCTIVE: those runs lose resume-ability.")
    ap.add_argument("--purge-surrogate-weights", action="store_true",
                    help="Also DELETE surrogate table weight blobs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report how many rows would be truncated")
    args = ap.parse_args()

    cap = args.cap if args.cap is not None else blob_cap_bytes()
    if cap <= 0:
        print("cap <= 0 means 'store full'; nothing to do. "
              "Pass --cap N > 0.")
        return 2
    if not os.path.exists(args.db):
        print(f"db not found: {args.db}")
        return 2

    size_before = os.path.getsize(args.db)
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM discoveries").fetchone()[0]
        print(f"db={args.db} size={_fmt_gb(size_before)} "
              f"pages={page_count} freelist={freelist} rows={total_rows}")

        # Ensure the v3 column exists (idempotent — FindingsDB does the
        # same on open, but this script may run against a never-reopened
        # db directly).
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(discoveries)")}
        missing = [c for c in ("content_sha256",) if c not in cols]
        for col in missing:
            conn.execute(
                f"ALTER TABLE discoveries ADD COLUMN {col} TEXT")
        if missing:
            conn.commit()
            print(f"added columns: {missing}")

        # length() on TEXT reads the record serial-type header only —
        # overflow pages are NOT touched, so this scan is cheap even on
        # a 40GB db.
        cond = " OR ".join(f"length({c}) > ?" for c in _UPDATE_COLS)
        ids = [r[0] for r in conn.execute(
            f"SELECT id FROM discoveries WHERE {cond}",
            (cap,) * len(_UPDATE_COLS))]
        print(f"rows over cap ({cap} chars): {len(ids)}")

        # ── optional: the REAL payload in meta_inner DBs ────────────────
        # Measured: discoveries' text columns total ~14MB; the 40GB lives
        # in generators.weights_blob (682 inner-run snapshots). Opt-in
        # purge — destructive to weight data, keeps runs/discoveries.
        for table, flag in (("generators", args.purge_generator_weights),
                            ("surrogate", args.purge_surrogate_weights)):
            if not flag:
                continue
            try:
                est = conn.execute(
                    f"SELECT COUNT(*), SUM(length(weights_blob)) "
                    f"FROM {table}").fetchone()
                print(f"{table}: {est[0]} rows, "
                      f"{(est[1] or 0)/1e9:.2f}GB of weights_blob")
                if not args.dry_run:
                    conn.execute(f"DELETE FROM {table}")
                    conn.commit()
                    print(f"  deleted {est[0]} rows from {table}")
            except sqlite3.OperationalError as e:
                print(f"  {table}: skipped ({e})")

        if args.dry_run or (not ids and not (
                args.purge_generator_weights or args.purge_surrogate_weights)):
            return 0
        if not ids:
            return 0

        t0 = time.time()
        done = 0
        col_set = ", ".join(
            f"{c} = CASE WHEN length({c}) > ? "
            f"THEN substr({c}, 1, ?) || ? ELSE {c} END"
            for c in _UPDATE_COLS)
        if args.no_sha:
            # SQL-only path: no blob transfer through Python at all.
            params = []
            for _c in _UPDATE_COLS:
                params += [cap, cap, TRUNCATED_SUFFIX]
            for i in range(0, len(ids), args.chunk):
                chunk = ids[i:i + args.chunk]
                ph = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE discoveries SET {col_set} "
                    f"WHERE id IN ({ph})", (*params, *chunk))
                conn.commit()
                done += len(chunk)
                print(f"  {done}/{len(ids)} rows "
                      f"({done / max(time.time() - t0, 1e-9):.0f}/s)",
                      flush=True)
        else:
            # Hash path: read originals, compute content_sha256, then
            # truncate — the ledger keeps a fingerprint of what was cut.
            upd = (f"UPDATE discoveries SET {col_set}, "
                   f"content_sha256 = COALESCE(?, content_sha256) "
                   f"WHERE id = ?")
            for i in range(0, len(ids), args.chunk):
                chunk = ids[i:i + args.chunk]
                sel = ("SELECT id, " + ", ".join(_UPDATE_COLS) +
                       f" FROM discoveries WHERE id IN "
                       f"({','.join('?' for _ in chunk)})")
                rows = conn.execute(sel, chunk).fetchall()
                for row in rows:
                    rid, texts = row[0], row[1:]
                    sha = _sha256_texts(*texts)
                    params = []
                    for _c in _UPDATE_COLS:
                        params += [cap, cap, TRUNCATED_SUFFIX]
                    conn.execute(upd, (*params, sha, rid))
                conn.commit()
                done += len(chunk)
                if done % (args.chunk * 4) < args.chunk or done == len(ids):
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"  {done}/{len(ids)} rows ({rate:.0f}/s)",
                          flush=True)

        # Checkpoint the WAL so the -wal sidecar doesn't keep the churn.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size_after = os.path.getsize(args.db)
        freelist_after = conn.execute(
            "PRAGMA freelist_count").fetchone()[0]
        print(f"done in {time.time() - t0:.0f}s — "
              f"size {_fmt_gb(size_before)} -> {_fmt_gb(size_after)} "
              f"(freelist {freelist} -> {freelist_after} pages; "
              f"no VACUUM by design)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

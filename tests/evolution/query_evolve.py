"""ForgeEvolve DB query tool — single entry point for database inspection.

Replaces read_db.py.

Usage:
    # List all runs
    python query_evolve.py --db forge_evolve_deep.db list-runs

    # List runs for a specific domain
    python query_evolve.py --db forge_evolve_deep.db list-runs --domain quant

    # Top discoveries across all domains
    python query_evolve.py --db forge_evolve_deep.db top --n 20

    # Top discoveries for a specific domain
    python query_evolve.py --db forge_evolve_deep.db top --domain aaac_quant --n 10

    # Score progression by generation
    python query_evolve.py --db forge_evolve_deep.db progression --domain aaac_quant

    # Export all discoveries to JSON
    python query_evolve.py --db forge_evolve_deep.db export --out discoveries.json

    # Stats summary
    python query_evolve.py --db forge_evolve_deep.db stats
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "research" / "results"


def get_db_path(db_arg: str) -> str:
    p = Path(db_arg)
    if not p.is_absolute():
        p = RESULTS_DIR / db_arg
    return str(p)


def cmd_list_runs(args):
    db = get_db_path(args.db)
    conn = sqlite3.connect(db)
    c = conn.cursor()

    if args.domain:
        c.execute(
            """SELECT run_id, domain, generations, total_evals,
                      discoveries, best_score, start_time, end_time
                   FROM runs WHERE domain LIKE ? ORDER BY start_time DESC""",
            (f"%{args.domain}%",),
        )
    else:
        c.execute("""SELECT run_id, domain, generations, total_evals,
                      discoveries, best_score, start_time, end_time
                   FROM runs ORDER BY start_time DESC""")

    rows = c.fetchall()
    if not rows:
        print("No runs found.")
        return

    print(f"{'Run ID':<35s} {'Domain':<25s} {'Gens':>5s} {'Evals':>6s} {'Disc':>5s} {'Best':>10s} {'Time':>8s}")
    print(f"{'-' * 35} {'-' * 25} {'-' * 5} {'-' * 6} {'-' * 5} {'-' * 10} {'-' * 8}")
    for r in rows:
        dur = (r[7] - r[6]) if r[6] and r[7] else 0
        print(f"{r[0]:<35s} {r[1]:<25s} {r[2]:5d} {r[3]:6d} {r[4]:5d} {r[5]:10.2f} {dur:7.1f}s")
    conn.close()


def cmd_top(args):
    db = get_db_path(args.db)
    conn = sqlite3.connect(db)
    c = conn.cursor()

    if args.domain:
        c.execute(
            """SELECT domain, config_json, score, behavioral_json, metadata_json,
                      run_id, generation
                   FROM discoveries WHERE domain LIKE ? AND score IS NOT NULL
                   ORDER BY score DESC LIMIT ?""",
            (f"%{args.domain}%", args.n),
        )
    else:
        c.execute(
            """SELECT domain, config_json, score, behavioral_json, metadata_json,
                      run_id, generation
                   FROM discoveries WHERE score IS NOT NULL
                   ORDER BY score DESC LIMIT ?""",
            (args.n,),
        )

    rows = c.fetchall()
    if not rows:
        print("No discoveries found.")
        return

    print(f"Top {len(rows)} discoveries:\n")
    for i, r in enumerate(rows):
        config = json.loads(r[1]) if r[1] else {}
        meta = json.loads(r[4]) if r[4] else {}
        print(f"  [{i + 1}] {r[0]:<25s} score={r[2]:.2f}  gen={r[6]}")
        print(f"       config: {json.dumps(config, default=str)}")
        if meta:
            print(f"       meta:   {json.dumps(meta, default=str)}")
        print()
    conn.close()


def cmd_progression(args):
    db = get_db_path(args.db)
    conn = sqlite3.connect(db)
    c = conn.cursor()

    c.execute(
        """SELECT generation, score, config_json
                 FROM discoveries WHERE domain LIKE ? AND score IS NOT NULL
                 ORDER BY generation""",
        (f"%{args.domain}%",),
    )
    rows = c.fetchall()
    if not rows:
        print(f"No discoveries for domain '{args.domain}'.")
        return

    print(f"Score progression for '{args.domain}' ({len(rows)} discoveries):\n")
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r[0]].append(r[1])

    print(f"  {'Gen':>5s} {'Best':>10s} {'Mean':>10s} {'Count':>6s}")
    print(f"  {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 6}")
    for gen in sorted(by_gen.keys()):
        scores = by_gen[gen]
        print(f"  {gen:5d} {max(scores):10.2f} {np.mean(scores):10.2f} {len(scores):6d}")
    conn.close()


def cmd_export(args):
    db = get_db_path(args.db)
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("""SELECT domain, config_json, score, behavioral_json, metadata_json,
                  run_id, generation FROM discoveries WHERE score IS NOT NULL
                  ORDER BY domain, score DESC""")
    rows = c.fetchall()
    data = []
    for r in rows:
        data.append(
            {
                "domain": r[0],
                "config": json.loads(r[1]) if r[1] else {},
                "score": r[2],
                "behavioral": json.loads(r[3]) if r[3] else [],
                "metadata": json.loads(r[4]) if r[4] else {},
                "run_id": r[5],
                "generation": r[6],
            }
        )
    out_path = args.out or str(RESULTS_DIR / "discoveries_export.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Exported {len(data)} discoveries to {out_path}")
    conn.close()


def cmd_stats(args):
    db = get_db_path(args.db)
    conn = sqlite3.connect(db)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM runs")
    n_runs = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM discoveries")
    n_disc = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM evaluations")
    n_evals = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT domain) FROM discoveries")
    n_domains = c.fetchone()[0]
    c.execute("SELECT MAX(score), MIN(score), AVG(score) FROM discoveries")
    s_max, s_min, s_avg = c.fetchone()

    print(f"Database: {db}")
    print(f"  Runs:         {n_runs}")
    print(f"  Discoveries:  {n_disc}")
    print(f"  Evaluations:  {n_evals}")
    print(f"  Domains:      {n_domains}")
    print(f"  Score range:  [{s_min:.2f}, {s_max:.2f}]  avg={s_avg:.2f}")

    # Per-domain breakdown
    c.execute("""SELECT domain, COUNT(*), MAX(score), AVG(score)
                 FROM discoveries GROUP BY domain ORDER BY MAX(score) DESC""")
    rows = c.fetchall()
    if rows:
        print(f"\n  {'Domain':<30s} {'Count':>6s} {'Best':>10s} {'Avg':>10s}")
        print(f"  {'-' * 30} {'-' * 6} {'-' * 10} {'-' * 10}")
        for r in rows:
            print(f"  {r[0]:<30s} {r[1]:6d} {r[2]:10.2f} {r[3]:10.2f}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="ForgeEvolve DB query tool")
    parser.add_argument("--db", default="forge_evolve_deep.db", help="DB filename or path")
    sub = parser.add_subparsers(dest="command")

    p_runs = sub.add_parser("list-runs", help="List all runs")
    p_runs.add_argument("--domain", default=None, help="Filter by domain name")

    p_top = sub.add_parser("top", help="Top discoveries")
    p_top.add_argument("--domain", default=None, help="Filter by domain")
    p_top.add_argument("--n", type=int, default=20, help="Number to show")

    p_prog = sub.add_parser("progression", help="Score progression by generation")
    p_prog.add_argument("--domain", required=True, help="Domain name")

    p_export = sub.add_parser("export", help="Export discoveries to JSON")
    p_export.add_argument("--out", default=None, help="Output file path")

    sub.add_parser("stats", help="Database statistics")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {
        "list-runs": cmd_list_runs,
        "top": cmd_top,
        "progression": cmd_progression,
        "export": cmd_export,
        "stats": cmd_stats,
    }[args.command](args)


if __name__ == "__main__":
    main()

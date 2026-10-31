"""Dump verified winning programs from the evolve DB for human review.

Usage:
    python scripts/show_winners.py <db> <target> [n] [--diff]

Shows the top-N ok=1 programs for a target with score/metrics/island/
gen/adapter. --diff prints a unified diff against the target's seed.
"""
import difflib
import json
import sqlite3
import sys

sys.path.insert(0, ".")


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "forge_evolve.db"
    tgt = sys.argv[2] if len(sys.argv) > 2 else "top_p_filter"
    n = int(sys.argv[3]) if len(sys.argv) > 3 and \
        sys.argv[3].isdigit() else 3
    want_diff = "--diff" in sys.argv

    rows = sqlite3.connect(db).execute(
        "SELECT code, score, metrics_json, island, gen, ts, adapter "
        "FROM code_programs WHERE target=? AND ok=1 "
        "ORDER BY score DESC LIMIT ?", (tgt, n)).fetchall()
    if not rows:
        print(f"no verified rows for target {tgt!r} in {db}")
        return

    seed = None
    if want_diff:
        try:
            from forge.evolution.code_evolve import Target
            seed = Target(Target.resolve(tgt)).seed_code
        except Exception as e:
            print(f"(seed unavailable for diff: {e})")

    for code, score, mj, isl, gen, ts, adapter in rows:
        print(f"\n{'=' * 70}\n"
              f"score={score:.4f} island={isl} gen={gen} "
              f"adapter={adapter} ts={ts}\n"
              f"metrics={mj or '{}'}\n{'=' * 70}")
        if seed is not None:
            for line in difflib.unified_diff(
                    seed.splitlines(), code.splitlines(),
                    "seed", "winner", lineterm=""):
                print(line)
        else:
            print(code)


if __name__ == "__main__":
    main()

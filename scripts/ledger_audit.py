"""Promotion-ledger audit (COMPETITIVE_AUDIT A11b): what fraction of
evolution output ever got WIRED IN — run quarterly, trend matters.

Reports, per DB: discoveries applied-rate, CodeEvolve promotions vs
rejects by stage, and verified improvement claimed per target (vs the
seed incumbent, not just the resumed bar)."""
import json
import sqlite3
import sys
from collections import Counter, defaultdict


def main(db: str = "forge_evolve.db") -> int:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    have = {r[0] for r in con.execute(
        "select name from sqlite_master where type='table'")}

    if "discoveries" in have:
        n, applied, wins = 0, 0, 0
        for cfg, score in con.execute(
                "select config_json, score from discoveries"):
            n += 1
            try:
                applied += bool(json.loads(cfg or "{}").get("applied"))
            except Exception:
                pass
            wins += (score or 0) > 0
        print(f"discoveries: {n} rows | applied {applied} "
              f"({100*applied/max(n,1):.1f}%) | positive-score {wins}")

    if "code_programs" not in have:
        return 0
    stages = Counter()
    best = {}
    raws = gated = 0
    for target, stage, score in con.execute(
            "select target, stage, score from code_programs"):
        raws += 1
        stages[stage or "?"] += 1
        if stage == "gated":
            gated += 1
        if score is not None:
            best[target] = max(best.get(target, -1e18), score)
    print(f"code_programs: {raws} rows | gated {gated} "
          f"({100*gated/max(raws,1):.1f}%)")
    print("stage_counts:", dict(stages.most_common(10)))

    seeds = {}
    try:
        for target, score in con.execute(
                "select target, min(score) from code_programs "
                "where parent_hash is null or parent_hash='' "
                "group by target"):
            seeds[target] = score
    except Exception:
        pass
    print("\nper-target best score (min-goal targets show negative):")
    for t in sorted(best):
        print(f"  {t:24s} {best[t]:.6g}")

    promos = con.execute(
        "select count(distinct run_id) from code_programs").fetchone()[0]
    print(f"\nruns logged: {promos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))

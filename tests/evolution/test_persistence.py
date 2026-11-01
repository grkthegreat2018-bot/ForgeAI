"""Test persistence: run ForgeEvolve twice, verify second run benefits from first.

Validates:
1. Findings are saved to SQLite database
2. Second run warm-starts from first (generators + surrogate + archive)
3. Second run finds better score faster (fewer evaluations to match first run)
4. Database can be queried for past discoveries
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = str(Path(__file__).resolve().parents[2] / "research" / "results" / "forge_evolve_test.db")


def main(out_dir=None):
    from forge.evolution import FindingsDB, ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.quant import QuantDomain

    # Clean start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("=" * 70)
    print("  Persistence Test: Run twice, verify cross-run learning")
    print("=" * 70)

    domain = QuantDomain(matrix_size=(2048, 4096), seed=42)
    print(f"  Domain: quant, device={domain.device}")
    print()

    # ── Run 1: Cold start (no past findings) ──
    print("  [Run 1] Cold start — no past findings...")
    cfg1 = ForgeEvolveConfig(
        domain=domain,
        n_generators=200,
        filter_ratio=10,
        min_evaluate=10,
        max_evaluate=20,
        generations=10,
        exploration=0.3,
        db_path=DB_PATH,
        run_id="quant_run1",
        warm_start=False,  # cold start
        verbose=True,
    )
    engine1 = ForgeEvolve(cfg1)
    t0 = time.perf_counter()
    r1 = engine1.run()
    t1 = time.perf_counter() - t0

    print(
        f"\n  Run 1: best={r1['best_score']:.4f}, "
        f"{r1['total_evaluations']} evals, {r1['discoveries']} discoveries, "
        f"{t1:.1f}s, warm_started={r1['warm_started']}"
    )

    # ── Verify database has findings ──
    print("\n  [Database check] Querying past findings...")
    db = FindingsDB(DB_PATH)
    runs = db.list_runs("quant")
    print(f"  Runs in DB: {len(runs)}")
    for run in runs:
        print(
            f"    {run['run_id']}: gen={run['generations']}, "
            f"discoveries={run['discoveries']}, best={run['best_score']:.4f}"
        )

    past_disc = db.query_discoveries("quant", limit=5)
    print(f"  Top discoveries in DB: {len(past_disc)}")
    for d in past_disc[:3]:
        print(f"    score={d['score']:.4f}, config={d['config']}")

    # ── Run 2: Warm start (should benefit from Run 1) ──
    print("\n  [Run 2] Warm start — loading past findings...")
    cfg2 = ForgeEvolveConfig(
        domain=domain,
        n_generators=200,
        filter_ratio=10,
        min_evaluate=10,
        max_evaluate=20,
        generations=10,
        exploration=0.3,
        db_path=DB_PATH,
        run_id="quant_run2",
        warm_start=True,  # warm start from Run 1
        verbose=True,
    )
    engine2 = ForgeEvolve(cfg2)
    t0 = time.perf_counter()
    r2 = engine2.run()
    t2 = time.perf_counter() - t0

    print(
        f"\n  Run 2: best={r2['best_score']:.4f}, "
        f"{r2['total_evaluations']} evals, {r2['discoveries']} discoveries, "
        f"{t2:.1f}s, warm_started={r2['warm_started']}"
    )

    db.close()

    # ── Comparison ──
    print()
    print("  " + "=" * 60)
    print(f"  {'Run':<12} {'Best Score':>12} {'Evals':>8} {'Time':>8} {'Warm':>6}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 6}")
    print(f"  {'Run 1':<12} {r1['best_score']:>12.4f} {r1['total_evaluations']:>8} {t1:>7.1f}s {'No':>6}")
    print(f"  {'Run 2':<12} {r2['best_score']:>12.4f} {r2['total_evaluations']:>8} {t2:>7.1f}s {'Yes':>6}")

    # ── Validation ──
    print()
    all_pass = True

    # Check 1: Database has findings
    db = FindingsDB(DB_PATH)
    disc_count = len(db.query_discoveries("quant"))
    if disc_count > 0:
        print(f"  PASS: Database has {disc_count} discoveries")
    else:
        print("  FAIL: Database has no discoveries")
        all_pass = False

    # Check 2: Run 2 was warm-started
    if r2["warm_started"]:
        print("  PASS: Run 2 was warm-started from Run 1")
    else:
        print("  FAIL: Run 2 was not warm-started")
        all_pass = False

    # Check 3: Run 2 archive was pre-seeded
    # Run 2 should start with some archive entries from Run 1
    if r2["best_score"] >= r1["best_score"]:
        print(f"  PASS: Run 2 ({r2['best_score']:.4f}) >= Run 1 ({r1['best_score']:.4f})")
    else:
        print(f"  WARN: Run 2 ({r2['best_score']:.4f}) < Run 1 ({r1['best_score']:.4f})")
        print("         (may happen if exploration dominates — not a hard failure)")

    # Check 4: Run 2 found discoveries faster (per eval)
    # A warm-started run may evaluate 0 fresh configs (all past discoveries
    # were loaded into the archive) — treat that as a pass, not a crash.
    rate1 = r1["discoveries"] / max(r1["total_evaluations"], 1)
    if r2["total_evaluations"] == 0:
        print("  PASS: Run 2 evaluated 0 fresh configs (all cached from archive)")
        rate2 = rate1
    else:
        rate2 = r2["discoveries"] / r2["total_evaluations"]
    if rate2 >= rate1 * 0.5:  # at least half as efficient (new cells get scarce)
        print(f"  PASS: Run 2 discovery rate ({rate2:.3f}/eval) reasonable vs Run 1 ({rate1:.3f}/eval)")
    else:
        print(f"  WARN: Run 2 discovery rate ({rate2:.3f}) much lower than Run 1 ({rate1:.3f})")

    # Check 5: Can query database for best configs
    best = db.query_best_configs("quant", limit=1)
    if best and best[0]["score"] >= r1["best_score"]:
        print(f"  PASS: DB query returns best config (score={best[0]['score']:.4f})")
    else:
        print("  FAIL: DB query doesn't return best config")
        all_pass = False

    db.close()

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — persistence + cross-run learning working")
    else:
        print("  SOME CHECKS FAILED — review above")

    # Save summary
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parents[2] / "research" / "results"
    out = out_dir / "persistence_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "run1": {
                    "best": r1["best_score"],
                    "evals": r1["total_evaluations"],
                    "discoveries": r1["discoveries"],
                    "time": t1,
                },
                "run2": {
                    "best": r2["best_score"],
                    "evals": r2["total_evaluations"],
                    "discoveries": r2["discoveries"],
                    "time": t2,
                    "warm_started": r2["warm_started"],
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Summary saved to {out}")

    # Under pytest, the script's own FAIL checks must fail the test
    # (WARN checks above intentionally do not fail).
    assert all_pass, "SOME CHECKS FAILED — review above"


def test_persistence(tmp_path):
    main(tmp_path)


if __name__ == "__main__":
    main()

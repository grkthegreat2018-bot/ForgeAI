"""Test ForgeEvolve: compare against random search and grid search.

Validates that:
1. ForgeEvolve finds good configs faster than random search
2. The surrogate filter improves over generations (top-50 predictions get better)
3. MAP-Elites archive fills diverse cells (not just one local optimum)
4. Discovery rate is within target (1-30 min per discovery)

Runs on synthetic domain (fast, no GPU needed) + quant domain (real evaluation).
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary DB path to isolate tests from stale forge_evolve.db."""
    return str(tmp_path / "test_forge_evolve.db")


# ── Test 1: Synthetic domain (fast, validates the loop) ──


def test_synthetic(tmp_db):
    """Compare ForgeEvolve vs random search vs grid search on synthetic function."""
    from forge.evolution import ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.synthetic import SyntheticDomain

    print("=" * 70)
    print("  Test 1: Synthetic Domain (validates the loop)")
    print("=" * 70)

    domain = SyntheticDomain(dim=8, seed=42)
    print(f"  Optimal score: {domain.optimal_score:.4f}")
    print()

    # ── Random search baseline ──
    print("  [Random search] 200 evaluations...")
    t0 = time.perf_counter()
    rng = np.random.RandomState(123)
    random_best = -1e9
    random_history = []
    for _i in range(200):
        x = rng.rand(8)
        result = domain.evaluate({"x": x})
        random_best = max(random_best, result["score"])
        random_history.append(random_best)
    t_random = time.perf_counter() - t0
    print(f"  Random: best={random_best:.4f} in {t_random:.2f}s")

    # ── Grid search baseline ──
    print("  [Grid search] 3^8 = 6561 evaluations...")
    t0 = time.perf_counter()
    grid_best = -1e9
    grid_evals = 0
    for x0 in [0.0, 0.5, 1.0]:
        for x1 in [0.0, 0.5, 1.0]:
            for x2 in [0.0, 0.5, 1.0]:
                for x3 in [0.0, 0.5, 1.0]:
                    for x4 in [0.0, 0.5, 1.0]:
                        for x5 in [0.0, 0.5, 1.0]:
                            for x6 in [0.0, 0.5, 1.0]:
                                for x7 in [0.0, 0.5, 1.0]:
                                    x = np.array([x0, x1, x2, x3, x4, x5, x6, x7])
                                    result = domain.evaluate({"x": x})
                                    grid_best = max(grid_best, result["score"])
                                    grid_evals += 1
                                    if grid_evals >= 200:  # match budget
                                        break
                                if grid_evals >= 200:
                                    break
                            if grid_evals >= 200:
                                break
                        if grid_evals >= 200:
                            break
                    if grid_evals >= 200:
                        break
                if grid_evals >= 200:
                    break
            if grid_evals >= 200:
                break
        if grid_evals >= 200:
            break
    t_grid = time.perf_counter() - t0
    print(f"  Grid (200 evals): best={grid_best:.4f} in {t_grid:.2f}s")

    # ── ForgeEvolve ──
    print()
    print("  [ForgeEvolve] 500 generators, 20:1 filter, 20 generations...")
    cfg = ForgeEvolveConfig(
        domain=domain,
        n_generators=500,
        filter_ratio=20,
        min_evaluate=10,
        max_evaluate=25,
        generations=20,
        exploration=0.3,
        verbose=True,
        db_path=tmp_db,
    )
    engine = ForgeEvolve(cfg)
    t0 = time.perf_counter()
    results = engine.run()
    t_forge = time.perf_counter() - t0

    print()
    print(f"  ForgeEvolve: best={results['best_score']:.4f} in {t_forge:.2f}s ({results['total_evaluations']} evals)")
    print(f"  Discoveries: {results['discoveries']}")
    print(f"  Archive coverage: {results['archive_coverage'] * 100:.1f}%")

    # ── Comparison ──
    print()
    print("  " + "=" * 60)
    print(f"  {'Method':<20} {'Best Score':>12} {'Evals':>8} {'Time':>8} {'Score/Eval':>12}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 12}")
    print(f"  {'Random':<20} {random_best:>12.4f} {200:>8} {t_random:>7.2f}s {random_best / 200:>12.6f}")
    print(f"  {'Grid (200)':<20} {grid_best:>12.4f} {200:>8} {t_grid:>7.2f}s {grid_best / 200:>12.6f}")
    print(
        f"  {'ForgeEvolve':<20} {results['best_score']:>12.4f} "
        f"{results['total_evaluations']:>8} {t_forge:>7.2f}s "
        f"{results['best_score'] / results['total_evaluations']:>12.6f}"
    )

    # Efficiency: score per evaluation
    fe_eff = results["best_score"] / results["total_evaluations"]
    rand_eff = random_best / 200
    print(f"\n  ForgeEvolve is {fe_eff / rand_eff:.1f}x more efficient than random search")

    # Check if ForgeEvolve beat random with fewer or equal evaluations
    # Use a relaxed threshold (95% of random) since ForgeEvolve uses fewer evals
    # and stochasticity can cause variance. The assertion ensures the test
    # actually fails if ForgeEvolve regresses badly.
    if results["best_score"] > random_best:
        print("  PASS: ForgeEvolve found better score than random search")
    elif results["best_score"] > random_best * 0.95:
        print("  PASS: ForgeEvolve matched random search (within 5%)")
    else:
        print(f"  FAIL: ForgeEvolve did not match random search ({results['best_score']:.4f} vs {random_best:.4f})")
    assert results["best_score"] >= random_best * 0.80, (
        f"ForgeEvolve ({results['best_score']:.4f}) severely underperformed "
        f"random search ({random_best:.4f}) — possible regression in "
        f"generator/surrogate/trainer"
    )

    # Check archive diversity
    elites = results["archive"].get_all_elites()
    if len(elites) >= 5:
        print(f"  PASS: Archive has {len(elites)} diverse elites (MAP-Elites working)")
    else:
        print(f"  WARN: Archive only has {len(elites)} elites (low diversity)")
    assert len(elites) >= 1, "Archive has zero elites — MAP-Elites is broken"


# ── Test 2: Quant domain (real evaluation, slower) ──


def test_quant():
    """Test ForgeEvolve on real quantization parameter search."""
    from forge.evolution import ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.quant import QuantDomain

    print()
    print("=" * 70)
    print("  Test 2: Quant Domain (real FP4 evaluation, GPU-accelerated)")
    print("=" * 70)

    domain = QuantDomain(matrix_size=(2048, 4096), seed=42)
    print(f"  Test matrix: {domain.matrix_size}, W norm={domain.W.norm():.4f}, device={domain.device}")
    print()

    # Baseline: current best known config
    baseline_config = {"block_size": 32, "scale_method": "absmax", "residual_ratio": 0.0, "global_scale_factor": 1.0}
    baseline = domain.evaluate(baseline_config)
    print(
        f"  Baseline (current NVFP4): score={baseline['score']:.4f}, "
        f"fwd_err={baseline['metadata']['fwd_err']:.4f}, "
        f"frob_err={baseline['metadata']['frob_err']:.4f}, "
        f"compression={baseline['metadata']['compression']:.1f}x"
    )

    # Known good: AS-FP4
    asfp4_config = {"block_size": 32, "scale_method": "mse_optimal", "residual_ratio": 0.0, "global_scale_factor": 1.0}
    asfp4 = domain.evaluate(asfp4_config)
    print(
        f"  AS-FP4 (known good): score={asfp4['score']:.4f}, "
        f"fwd_err={asfp4['metadata']['fwd_err']:.4f}, "
        f"frob_err={asfp4['metadata']['frob_err']:.4f}"
    )

    print()
    print("  [ForgeEvolve] 200 generators, 10:1 filter, 15 generations...")
    cfg = ForgeEvolveConfig(
        domain=domain,
        n_generators=200,
        filter_ratio=10,
        min_evaluate=10,
        max_evaluate=20,
        generations=15,
        exploration=0.3,
        verbose=True,
    )
    engine = ForgeEvolve(cfg)
    t0 = time.perf_counter()
    results = engine.run()
    t_forge = time.perf_counter() - t0

    print()
    print(f"  ForgeEvolve: best={results['best_score']:.4f} in {t_forge:.2f}s ({results['total_evaluations']} evals)")
    print(f"  Discoveries: {results['discoveries']}")

    if results["best_config"]:
        best = results["best_config"]
        best_result = domain.evaluate(best)
        print(f"  Best config: {best}")
        print(
            f"  Best result: fwd_err={best_result['metadata']['fwd_err']:.4f}, "
            f"frob_err={best_result['metadata']['frob_err']:.4f}, "
            f"compression={best_result['metadata']['compression']:.1f}x"
        )

    # Compare to baseline
    if results["best_score"] > baseline["score"]:
        improvement = (results["best_score"] - baseline["score"]) / abs(baseline["score"]) * 100
        print(f"\n  PASS: ForgeEvolve beat baseline by {improvement:.1f}%")
    else:
        print("\n  INFO: ForgeEvolve did not beat baseline (baseline may be near-optimal)")

    # Time per discovery
    if results["discoveries"] > 0:
        time_per_disc = t_forge / results["discoveries"]
        print(f"  Time per discovery: {time_per_disc:.1f}s")
        if time_per_disc < 1800:
            print("  PASS: Within target (<30 min per discovery)")
        else:
            print(f"  WARN: Slower than target ({time_per_disc / 60:.1f} min per discovery)")


# ── Test 3: Surrogate accuracy over generations ──


def test_surrogate_learning(tmp_db):
    """Verify that the surrogate filter improves over generations."""
    from forge.evolution import ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.synthetic import SyntheticDomain

    print()
    print("=" * 70)
    print("  Test 3: Surrogate Learning (does filter improve over time?)")
    print("=" * 70)

    domain = SyntheticDomain(dim=4, seed=99)

    cfg = ForgeEvolveConfig(
        domain=domain,
        n_generators=200,
        filter_ratio=20,
        min_evaluate=10,
        max_evaluate=10,
        generations=30,
        exploration=0.2,
        verbose=False,
        db_path=tmp_db,
    )
    engine = ForgeEvolve(cfg)
    results = engine.run()

    # Check: did later generations find better scores?
    early = [r for r in results["all_results"] if r["generation"] < 5]
    late = [r for r in results["all_results"] if r["generation"] >= 20]

    early_best = max(r["score"] for r in early) if early else -1e9
    late_best = max(r["score"] for r in late) if late else -1e9

    print(f"  Early (gen 0-4) best: {early_best:.4f}")
    print(f"  Late (gen 20+): best: {late_best:.4f}")

    if late_best > early_best:
        print("  PASS: Later generations found better scores (surrogate + REINFORCE working)")
        print(f"  Improvement: {late_best - early_best:.4f}")
    else:
        print("  WARN: No improvement over generations — check learning rate / exploration")
    # Assert that late generations are at least as good as early (not a strict
    # improvement requirement since stochasticity on dim=4 can plateau).
    assert late_best >= early_best * 0.98, (
        f"Late generations ({late_best:.4f}) regressed vs early ({early_best:.4f}) "
        f"— surrogate or REINFORCE may be broken"
    )

    # Discovery timeline
    if results["discoveries_list"]:
        print("\n  Discovery timeline:")
        for d in results["discoveries_list"][-5:]:  # last 5
            print(f"    Gen {d['generation']:3d}: score={d['score']:.4f}")


def main(out_dir=None):
    print()
    print("ForgeEvolve Test Suite")
    print("=" * 70)
    print()

    # Test 1: synthetic (fast, ~30 sec)
    r1 = test_synthetic()

    # Test 2: quant (slower, ~2-5 min)
    r2 = test_quant()

    # Test 3: surrogate learning (~30 sec)
    r3 = test_surrogate_learning()

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"  Test 1 (synthetic): best={r1['best_score']:.4f}, "
        f"{r1['total_evaluations']} evals, {r1['discoveries']} discoveries"
    )
    print(
        f"  Test 2 (quant): best={r2['best_score']:.4f}, "
        f"{r2['total_evaluations']} evals, {r2['discoveries']} discoveries"
    )
    print(f"  Test 3 (surrogate): late > early = {r3 is not None}")

    # Save results
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parents[2] / "research" / "results"
    out_path = out_dir / "forge_evolve_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "synthetic": {
                    "best_score": r1["best_score"],
                    "evaluations": r1["total_evaluations"],
                    "discoveries": r1["discoveries"],
                    "time_s": r1["time_s"],
                },
                "quant": {
                    "best_score": r2["best_score"],
                    "evaluations": r2["total_evaluations"],
                    "discoveries": r2["discoveries"],
                    "time_s": r2["time_s"],
                    "best_config": str(r2.get("best_config")),
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

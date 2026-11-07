"""ForgeEvolve DB rescore tool — re-evaluate discoveries with updated scoring.

V2 (2026-08-25): Now uses domain JSON specs instead of the FIXED_DOMAINS dict.
Any domain with a JSON spec in configs/domains/ can be rescored automatically.
No need to maintain a separate dict of fixed domains — the JSON spec IS the
source of truth for scoring.

When a domain's scoring is fixed (e.g. to penalize a synthetic-metric artifact):
1. Update the domain's JSON spec (add penalty entries, change weights, etc.)
2. Run: python rescore_db.py --db forge_evolve.db --prune

The tool re-evaluates every discovery with the current JSON spec + RewardGuard,
updates scores in-place, and optionally prunes fake winners (discoveries whose
score dropped sharply under the fixed scoring).

Usage:
    # Dry-run: see what would change for all JSON-specified domains
    python rescore_db.py --db forge_evolve.db --dry-run

    # Apply rescore + remove bad ideas
    python rescore_db.py --db forge_evolve.db --prune

    # Rescore specific domains only
    python rescore_db.py --db forge_evolve.db --domains moe_routing,rope_config

    # Custom prune threshold (default: drop >50% = artifact)
    python rescore_db.py --db forge_evolve.db --prune --drop-threshold 0.4

    # Rescore ALL domains (including non-JSON ones via Python classes)
    python rescore_db.py --db forge_evolve.db --all
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "research" / "results"


def get_db_path(db_arg: str) -> str:
    p = Path(db_arg)
    if not p.is_absolute():
        p = RESULTS_DIR / db_arg
    return str(p)


def load_domain(domain_name: str):
    """Instantiate the domain for a given domain name.

    Tries JSON spec first (canonical source of truth), falls back to Python
    class lookup for domains without JSON specs.
    """
    from forge.evolution.domain_spec import JSONSpecDomain, list_specs, load_spec
    from forge.evolution.domains import DOMAINS

    # Check if a JSON spec exists for this domain name
    # Domain names in DB may have _refine_dN suffix — strip it
    base_name = domain_name
    for suffix in ["_refine_d1", "_refine_d2", "_refine_d3", "_refine_d4", "_refine_d5"]:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break

    # Try JSON spec match by domain name
    for spec_name in list_specs():
        try:
            spec = load_spec(spec_name)
            if spec.name == domain_name or spec.name == base_name:
                return JSONSpecDomain(spec=spec)
        except Exception:
            continue

    # Fall back to Python class lookup
    # Try CamelCase class name from domain name
    class_name = "".join(w.capitalize() for w in base_name.split("_"))
    if class_name in DOMAINS:
        try:
            return DOMAINS[class_name]()
        except Exception:
            pass

    # Try direct match
    if domain_name in DOMAINS:
        return DOMAINS[domain_name]()

    raise KeyError(f"No domain found for DB domain '{domain_name}'. Checked JSON specs and Python classes.")


def get_all_db_domains(conn) -> list[str]:
    """Get all distinct domain names from the discoveries table."""
    c = conn.cursor()
    c.execute("SELECT DISTINCT domain FROM discoveries")
    return [r[0] for r in c.fetchall()]


def rescore_domain(
    conn,
    domain_name: str,
    seed: int = 42,
    prune: bool = False,
    prune_low: bool = False,
    drop_threshold: float = 0.5,
    dry_run: bool = False,
) -> dict:
    """Re-evaluate all discoveries for one domain. Returns summary stats."""
    c = conn.cursor()
    c.execute(
        "SELECT id, config_json, score, metadata_json, behavioral_json "
        "FROM discoveries WHERE domain = ? ORDER BY score DESC",
        (domain_name,),
    )
    rows = c.fetchall()
    if not rows:
        return {"domain": domain_name, "n": 0, "rescored": 0, "pruned": 0, "old_best": None, "new_best": None}

    try:
        domain = load_domain(domain_name)
    except Exception as e:
        print(f"  Cannot load domain for '{domain_name}': {e}")
        return {
            "domain": domain_name,
            "n": len(rows),
            "rescored": 0,
            "pruned": 0,
            "old_best": max(r[2] for r in rows),
            "new_best": None,
            "error": str(e),
        }

    old_scores = [r[2] for r in rows]
    old_best = max(old_scores)

    updates = []
    prune_ids = []
    new_scores = []

    for row in rows:
        did, config_json, old_score, _meta_json, _behav_json = row
        config = json.loads(config_json)

        # Deterministic re-evaluation
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            result = domain.evaluate(config)
            new_score = float(result["score"])
            if not np.isfinite(new_score):
                new_score = -1e9
                new_meta = json.dumps({"rescore_error": f"non-finite score: {result['score']}"})
                new_behav = json.dumps(())
            else:
                new_meta = json.dumps(result.get("metadata", {}), default=str)
                new_behav = json.dumps(result.get("behavioral", ()), default=str)
        except Exception as e:
            new_score = -1e9
            new_meta = json.dumps({"rescore_error": str(e)})
            new_behav = json.dumps(())

        new_scores.append(new_score)
        updates.append((did, new_score, new_meta, new_behav))

        # Prune fake winners: discoveries that dropped >drop_threshold (relative)
        if prune and old_score > 0:
            relative_drop = (old_score - new_score) / abs(old_score)
            if relative_drop > drop_threshold:
                prune_ids.append(did)

    new_best = max(new_scores) if new_scores else None

    # Optional aggressive cleanup: also prune below 25th percentile
    if prune_low and new_scores:
        q25 = float(np.percentile(new_scores, 25))
        for did, ns, _, _ in updates:
            if ns < q25 and did not in prune_ids:
                prune_ids.append(did)

    if not dry_run:
        c.executemany(
            "UPDATE discoveries SET score = ?, metadata_json = ?, behavioral_json = ? WHERE id = ?",
            [(ns, nm, nb, did) for (did, ns, nm, nb) in updates],
        )
        if prune_ids:
            placeholders = ",".join("?" * len(prune_ids))
            c.execute(f"DELETE FROM discoveries WHERE id IN ({placeholders})", prune_ids)
        # Update runs table best_score / best_config_json
        c.execute("SELECT run_id FROM runs WHERE domain = ?", (domain_name,))
        for (run_id,) in c.fetchall():
            c.execute(
                "SELECT config_json, score FROM discoveries "
                "WHERE run_id = ? AND domain = ? "
                "ORDER BY score DESC LIMIT 1",
                (run_id, domain_name),
            )
            best = c.fetchone()
            if best:
                c.execute(
                    "UPDATE runs SET best_score = ?, best_config_json = ? WHERE run_id = ?",
                    (best[1], best[0], run_id),
                )
        conn.commit()

    n_pruned = len(prune_ids) if prune else 0
    return {
        "domain": domain_name,
        "n": len(rows),
        "rescored": len(updates),
        "pruned": n_pruned,
        "old_best": old_best,
        "new_best": new_best,
        "old_mean": float(np.mean(old_scores)),
        "new_mean": float(np.mean(new_scores)) if new_scores else 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="forge_evolve.db", help="DB filename in research/results/")
    ap.add_argument(
        "--domains", default=None, help="Comma-separated domain names to rescore (default: all JSON-specified)"
    )
    ap.add_argument("--all", action="store_true", help="Rescore ALL domains in the DB (including non-JSON ones)")
    ap.add_argument("--dry-run", action="store_true", help="Show changes without writing to DB")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Delete fake winners: discoveries whose score dropped >threshold (artifacts)",
    )
    ap.add_argument(
        "--prune-low", action="store_true", help="Also delete discoveries below 25th percentile (aggressive cleanup)"
    )
    ap.add_argument(
        "--drop-threshold",
        type=float,
        default=0.5,
        help="Relative score drop threshold for pruning (default: 0.5 = 50%% drop)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic re-eval")
    args = ap.parse_args()

    db_path = get_db_path(args.db)
    if not os.path.exists(db_path):
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    # Determine which domains to rescore
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    elif args.all:
        conn_tmp = sqlite3.connect(db_path)
        domains = get_all_db_domains(conn_tmp)
        conn_tmp.close()
    else:
        # Default: all domains that have JSON specs
        from forge.evolution.domain_spec import list_specs, load_spec

        domains = []
        for spec_name in list_specs():
            try:
                spec = load_spec(spec_name)
                domains.append(spec.name)
            except Exception:
                continue

    print(f"DB: {db_path}")
    print(
        f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}"
        f"{' + PRUNE' if args.prune else ''}"
        f"{' + PRUNE-LOW' if args.prune_low else ''}"
    )
    print(f"Domains: {len(domains)}")
    print(f"Seed: {args.seed}")
    print()

    conn = sqlite3.connect(db_path)
    summaries = []
    for domain_name in domains:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM discoveries WHERE domain = ?", (domain_name,))
        count = c.fetchone()[0]
        if count == 0:
            print(f"  {domain_name:40s}  (0 discoveries, skipping)")
            continue
        print(f"  Rescoring {domain_name} ({count} discoveries)...", end=" ", flush=True)
        try:
            s = rescore_domain(
                conn,
                domain_name,
                seed=args.seed,
                prune=args.prune,
                prune_low=args.prune_low,
                drop_threshold=args.drop_threshold,
                dry_run=args.dry_run,
            )
            summaries.append(s)
            if "error" in s:
                print(f"ERROR: {s['error']}")
            else:
                delta_best = (s["new_best"] - s["old_best"]) if s["old_best"] else 0
                delta_mean = s["new_mean"] - s["old_mean"]
                prune_str = f", pruned={s['pruned']}" if args.prune else ""
                print(
                    f"old_best={s['old_best']:.2f} -> new_best={s['new_best']:.2f}"
                    f" (d={delta_best:+.2f}), mean d={delta_mean:+.2f}{prune_str}"
                )
        except Exception as e:
            print(f"ERROR: {e}")

    conn.close()

    # Summary table
    if summaries:
        print("\n" + "=" * 90)
        print(f"{'Domain':40s} {'N':>6s} {'Old Best':>10s} {'New Best':>10s} {'d Best':>8s} {'Pruned':>7s}")
        print("-" * 90)
        total_pruned = 0
        for s in summaries:
            if "error" in s:
                continue
            delta = (s["new_best"] - s["old_best"]) if s["old_best"] else 0
            total_pruned += s["pruned"]
            print(
                f"{s['domain']:40s} {s['n']:>6d} {s['old_best']:>10.2f}"
                f" {s['new_best']:>10.2f} {delta:>+8.2f} {s['pruned']:>7d}"
            )
        print("-" * 90)
        print(f"Total pruned: {total_pruned}")
        if args.dry_run:
            print("\n(DRY-RUN: no changes written. Re-run without --dry-run to apply.)")


if __name__ == "__main__":
    main()

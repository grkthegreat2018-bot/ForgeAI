"""ForgeEvolve profiler — single entry point for domain profiling.

Replaces profile_all.py, profile_domains.py, timing_test.py.

Usage:
    # Profile all domains (timing + VRAM)
    python profile_evolve.py

    # Specific category
    python profile_evolve.py --category quantization

    # Specific domains
    python profile_evolve.py --domains QuantDomain,AaacQuant

    # More evals for accuracy
    python profile_evolve.py --evals 50

    # Only show slow domains (>5ms)
    python profile_evolve.py --threshold 5.0

    # Seed configs only (quick check)
    python profile_evolve.py --seeds-only
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import argparse
import json
import time
from pathlib import Path

import torch

from forge.evolution.domains import DOMAINS, list_domains

CONFIG_DIR = Path(__file__).resolve().parents[2] / "forge" / "evolution" / "configs"


def load_json(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


def resolve_domains(category: str | None, domain_list: str | None) -> list[str]:
    if domain_list:
        names = [d.strip() for d in domain_list.split(",")]
        for n in names:
            if n not in DOMAINS:
                raise KeyError(f"Unknown domain '{n}'. Available: {list_domains()}")
        return names
    if category is None or category == "all":
        return list_domains()
    cats = load_json("domain_categories.json")["categories"]
    if category not in cats:
        raise KeyError(f"Unknown category '{category}'. Available: {list(cats.keys())}")
    return [d for d in cats[category]["domains"] if d in DOMAINS]


def profile_domain(name: str, n_evals: int = 20, seeds_only: bool = False, track_vram: bool = True) -> dict:
    """Profile a single domain. Returns {name, ms, peak_mb, error}."""
    cls = DOMAINS[name]
    try:
        d = cls()
        od = d.output_dim()

        if seeds_only:
            seeds = d.seed_configs()
            if not seeds:
                return {"name": name, "ms": 0, "peak_mb": 0, "error": None, "n_seeds": 0}
            # Warmup
            for s in seeds[:1]:
                d.evaluate(s)
            torch.cuda.synchronize()
            if track_vram:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(n_evals):
                for s in seeds:
                    d.evaluate(s)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / (n_evals * len(seeds)) * 1000
        else:
            # Warmup (triggers CUDA JIT)
            p = torch.rand(od)
            c = d.decode(p)
            d.evaluate(c)
            torch.cuda.synchronize()
            if track_vram:
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            for _ in range(n_evals):
                p = torch.rand(od)
                c = d.decode(p)
                d.evaluate(c)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / n_evals * 1000

        peak_mb = 0
        if track_vram and torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        return {"name": name, "ms": ms, "peak_mb": peak_mb, "error": None}

    except Exception as e:
        torch.cuda.empty_cache()
        return {"name": name, "ms": -1, "peak_mb": 0, "error": str(e)[:80]}


def main():
    parser = argparse.ArgumentParser(description="ForgeEvolve domain profiler")
    parser.add_argument("--category", default=None, help="Domain category")
    parser.add_argument("--domains", default=None, help="Comma-separated domain names")
    parser.add_argument("--evals", type=int, default=20, help="Evals per domain")
    parser.add_argument("--threshold", type=float, default=0, help="Only show domains above this ms/eval")
    parser.add_argument("--seeds-only", action="store_true", help="Only profile seed configs")
    parser.add_argument("--no-vram", action="store_true", help="Skip VRAM tracking")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    names = resolve_domains(args.category, args.domains)
    print(f"Profiling {len(names)} domains ({args.evals} evals each)...\n")

    results = []
    for name in names:
        r = profile_domain(name, n_evals=args.evals, seeds_only=args.seeds_only, track_vram=not args.no_vram)
        if r["error"]:
            print(f"  {name:<30s} FAIL: {r['error']}")
        elif args.threshold == 0 or r["ms"] > args.threshold:
            flag = " <<< FREEZE" if r["ms"] > 50 else (" << slow" if r["ms"] > 10 else "")
            if args.json:
                results.append(r)
            else:
                print(f"  {name:<30s} {r['ms']:8.1f} ms  {r['peak_mb']:6.0f} MB{flag}")
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Summary
    valid = [r for r in results if r["error"] is None and r["ms"] >= 0]
    if valid:
        total_ms = sum(r["ms"] for r in valid)
        slow = [r for r in valid if r["ms"] > 10]
        print(f"\n  {len(valid)} domains, avg {total_ms / len(valid):.1f} ms/eval")
        if slow:
            print(f"  {len(slow)} slow domains (>10ms):")
            for r in sorted(slow, key=lambda x: x["ms"], reverse=True):
                print(f"    {r['name']:<30s} {r['ms']:.1f} ms")


if __name__ == "__main__":
    main()

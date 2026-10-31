"""Scout — lightweight goal-driven discovery loop (OpenEvolve-lite).

Searches the existing ForgeEvolve domain specs with a tiny TPE proposer
instead of the 500-generator MLP stack. Only verified wins (replicated,
noise-aware margin) are recorded as discoveries. Runs until stagnation.

Examples:
    # Chase a goal: minimize memory footprint across quant/KV domains
    python scripts/scout_loop.py --focus memory --cycles 300

    # Chase speed (decoding + kv domains)
    python scripts/scout_loop.py --focus speed

    # Specific domains, with a required end-result gate: a promotion only
    # counts if err_reduction stays >= 0.1
    python scripts/scout_loop.py --domains hadamard_kv kvzip_kv \
        --require "err_reduction>=0.1"

    # Random-search baseline (for proving the ML proposer helps)
    python scripts/scout_loop.py --mode random --cycles 100

    # Inspect afterwards
    python scripts/scout_loop.py --report --db forge_evolve.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.evolution.scout import Scout, ScoutConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--focus", choices=["memory", "speed", "quality",
                                        "training", "all"], default="all",
                    help="goal filter: routes to domain categories")
    ap.add_argument("--domains", nargs="+", default=None,
                    help="explicit domain spec names (overrides --focus)")
    ap.add_argument("--require", default=None, metavar="METRIC>=V",
                    help="promotion gate, e.g. 'sqnr>=30' or 'compute_cost<=8'")
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mode", choices=["tpe", "random"], default="tpe")
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-reps", type=int, default=3)
    ap.add_argument("--patience", type=int, default=30,
                    help="stop after N cycles with no verified discovery")
    ap.add_argument("--patience-dom", type=int, default=40,
                    help="retire a domain after N dry cycles")
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="print discoveries + cross-domain insights, no run")
    args = ap.parse_args()

    if args.report:
        sc = Scout(ScoutConfig(db_path=args.db, verbose=False))
        print(sc.report())
        return 0

    cfg = ScoutConfig(domains=args.domains, focus=args.focus,
                      require=args.require, cycles=args.cycles,
                      batch=args.batch, mode=args.mode, db_path=args.db,
                      device=args.device, seed=args.seed,
                      verify_reps=args.verify_reps,
                      patience=args.patience, patience_dom=args.patience_dom,
                      verbose=not args.quiet)
    res = Scout(cfg).run()
    print("=" * 60)
    print(f"stop: {res['stop_reason']}")
    print(f"evals={res['evals']} cycles={res['cycles']} "
          f"discoveries={len(res['discoveries'])} "
          f"({res['elapsed_s']}s)")
    if res["discoveries"]:
        print("\ntop verified discoveries:")
        for d in sorted(res["discoveries"], key=lambda x: -x["score"])[:10]:
            print(f"  {d['domain']:<22} score={d['score']:.4f} "
                  f"+{d['delta']:.4f}  {d['config']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

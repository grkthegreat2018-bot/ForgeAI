"""Read EXP-XTAL curve files — extract crystallization transitions.

Usage: python scripts/exp_xtal_read.py results/xtal/wd001.jsonl [--full]
Prints per-run: step where each rule crosses excess-loss thresholds,
heldout generalization onset, embedding gap + alignment trajectories.
"""
from __future__ import annotations

import json
import sys


def load(path):
    import math
    evals, grads = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (evals if r["kind"] == "eval" else grads).append(r)
    # backfill xs/ keys for curves written before the driver fix:
    # floors — A/AH/B: log(6 train members); C: 0
    floor6 = math.log(6)
    for r in evals:
        for evkey, xskey, fl in (
                ("eval/A/ruleA", "xs/A/ruleA", floor6),
                ("eval/AH/ruleA_heldout", "xs/AH/ruleA_heldout", floor6),
                ("eval/B/ruleB", "xs/B/ruleB", floor6),
                ("eval/C/ruleC", "xs/C/ruleC", 0.0)):
            if xskey not in r and evkey in r:
                r[xskey] = round(r[evkey] - fl, 4)
    return evals, grads


def first_cross(evals, key, thresh, below=True):
    for r in evals:
        v = r.get(key)
        if v is None:
            continue
        if (below and v < thresh) or (not below and v > thresh):
            return r["step"]
    return None


def series(recs, key):
    return [(r["step"], r[key]) for r in recs if key in r]


def summarize(path, full=False):
    evals, grads = load(path)
    if not evals:
        print("no eval records")
        return
    print(f"== {path} — {len(evals)} eval pts, {len(grads)} grad pts, "
          f"{evals[-1]['step']} steps")

    # crystallization thresholds (excess nats above floor)
    for key, label in (("xs/A/ruleA", "ruleA"),
                       ("xs/AH/ruleA_heldout", "heldoutA"),
                       ("xs/B/ruleB", "ruleB"),
                       ("xs/C/ruleC", "ruleC")):
        for thr in (1.0, 0.5, 0.2):
            s = first_cross(evals, key, thr)
            if s is not None:
                print(f"  {label:9s} xs<{thr}: step {s}")
                break
        else:
            print(f"  {label:9s} never below 1.0 excess"
                  f" (final {evals[-1].get(key, float('nan')):.3f})")

    # generalization gap trajectory (train A vs heldout A)
    gap = [(r["step"], round(r.get("eval/AH/ruleA_heldout", 0) -
                             r.get("eval/A/ruleA", 0), 3))
           for r in evals if "eval/AH/ruleA_heldout" in r]
    if gap:
        print(f"  gen-gap (AH-A): start {gap[1][1] if len(gap)>1 else '?'} "
              f"final {gap[-1][1]}")

    for key in ("emb/class_gap", "align/A_C", "align/A_N"):
        s = series(grads, key)
        if s:
            traj = " ".join(f"{st}:{v:+.2f}" for st, v in s[::max(1, len(s)//8)])
            print(f"  {key:18s} {traj} -> final {s[-1][1]:+.3f}")

    if full:
        print("  -- eval trajectory (step: A AH B C N)")
        for r in evals[::max(1, len(evals)//15)]:
            print(f"  {r['step']:6d}: "
                  f"{r.get('eval/A/ruleA', 0):.2f} "
                  f"{r.get('eval/AH/ruleA_heldout', 0):.2f} "
                  f"{r.get('eval/B/ruleB', 0):.2f} "
                  f"{r.get('eval/C/ruleC', 0):.2f} "
                  f"{r.get('eval/N', 0):.2f}")


if __name__ == "__main__":
    full = "--full" in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        summarize(p, full=full)
        print()

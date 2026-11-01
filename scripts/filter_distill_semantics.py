"""Semantic filter for teacher distill rows.

Compile-valid is NOT enough — the proposal contract requires identical
semantics, and a fenced block that silently renames or drifts teaches the
proposer exactly the wrong lesson. Each row is matched to its evolution
target (by the 'keep public names (...)' clause) and run through the
target's own test(ns) — the same oracle eval_pool scores against.

Rows are classified: ok / semantic-fail / compile-fail / unverifiable
(no matching target). ok + unverifiable are kept; the rest dropped —
unverifiable rows still teach format even if semantics can't be checked.

Usage:
    venv\\Scripts\\python.exe scripts/filter_distill_semantics.py \
        --in results/qwen_distill.jsonl --out results/qwen_distill_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from forge.evolution.code_evolve import _extract_code  # noqa: E402


def _load_targets(targets_dir: Path):
    """{frozenset(REQUIRED): (name, test_fn)} — exec each target module."""
    import types
    table = {}
    for f in sorted(targets_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod = types.ModuleType(f.stem)
        try:
            exec(compile(f.read_text(encoding="utf-8", errors="replace"),
                         f.name, "exec"), mod.__dict__)
        except Exception as e:
            print(f"[filter] target {f.stem} exec failed: {e}", flush=True)
            continue
        req = tuple(sorted(getattr(mod, "REQUIRED", []) or ()))
        test = getattr(mod, "test", None)
        if req and callable(test):
            table[frozenset(req)] = (f.stem, test)
    return table


def _required_from_prompt(prompt: str) -> frozenset:
    m = re.search(r"keep public names \(([^)]*)\)", prompt)
    if not m:
        return frozenset()
    return frozenset(s.strip() for s in m.group(1).split(",")
                     if s.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    table = _load_targets(REPO / "forge" / "evolution" / "targets")
    print(f"[filter] {len(table)} target oracles loaded")

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")
            if l.strip()]
    kept, dropped = [], []
    stats = {"ok": 0, "semantic_fail": 0, "compile_fail": 0,
             "exec_fail": 0, "unverifiable": 0, "no_target": 0}
    for i, r in enumerate(rows):
        code = _extract_code(r.get("response", ""))
        if not code.strip():
            stats["compile_fail"] += 1
            dropped.append({"i": i, "why": "no_code"})
            continue
        try:
            compile(code, "<c>", "exec")
        except SyntaxError:
            stats["compile_fail"] += 1
            dropped.append({"i": i, "why": "syntax"})
            continue
        req = _required_from_prompt(r["prompt"])
        entry = table.get(req)
        if entry is None:
            # partial match: prompt names ⊇ target REQUIRED (proposer
            # sometimes keeps a superset)
            cands = [(k, v) for k, v in table.items() if k <= req]
            entry = cands[0][1] if len(cands) == 1 else None
        if entry is None:
            stats["no_target"] += 1
            kept.append(r)
            continue
        tname, test = entry
        ns = {}
        try:
            exec(compile(code, "<c>", "exec"), ns)
        except Exception as e:
            stats["exec_fail"] += 1
            dropped.append({"i": i, "why": f"exec:{type(e).__name__}",
                            "target": tname})
            continue
        missing = [n for n in req if n not in ns]
        if missing:
            stats["semantic_fail"] += 1
            dropped.append({"i": i, "why": f"missing:{missing}",
                            "target": tname})
            continue
        try:
            res = test(ns)
        except Exception as e:
            stats["exec_fail"] += 1
            dropped.append({"i": i, "why": f"test_raise:{type(e).__name__}",
                            "target": tname})
            continue
        if res.get("ok"):
            stats["ok"] += 1
            kept.append(r)
        else:
            stats["semantic_fail"] += 1
            dropped.append({"i": i, "why": res.get("detail", "?")[:120],
                            "target": tname})
        if (i + 1) % 40 == 0:
            print(f"[filter] {i+1}/{len(rows)} {stats}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    rep = {"stats": stats, "kept": len(kept), "dropped": dropped}
    rp = args.report or str(Path(args.out).with_suffix(".report.json"))
    Path(rp).write_text(json.dumps(rep, indent=2))
    print(f"[filter] {stats} -> kept {len(kept)}/{len(rows)} "
          f"({len(dropped)} dropped) report={rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

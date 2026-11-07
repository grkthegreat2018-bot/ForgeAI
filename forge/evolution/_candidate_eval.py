"""Subprocess evaluator for code evolution candidates.

Usage (internal — invoked by code_evolve's runner):
    python -m forge.evolution._candidate_eval <target_path> <cand_path>

Loads the target spec module, execs the candidate program, then runs
target.test(ns) (correctness gate) and target.bench(ns) (metrics).
Prints one line:  __RESULT__{json}

All failures are converted to a structured result so a crashing candidate
never kills the parent loop. Timeout is enforced by the caller.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _emit(out: dict) -> None:
    out["_nonce"] = _NONCE
    print("__RESULT__" + json.dumps(out, default=str), flush=True)


def _clearly_worse(metrics: dict, cheap: dict) -> bool:
    """Eval-cascade decision: the reduced-rep bench metric lands more
    than `margin` fraction worse than the incumbent -> the full bench
    can't change the outcome, skip it (~30-50% eval savings)."""
    v = metrics.get(cheap.get("metric", ""))
    inc = cheap.get("incumbent")
    if v is None or inc is None:
        return False
    try:
        v, inc = float(v), float(inc)
    except (TypeError, ValueError):
        return False
    if v != v or inc != inc:                      # NaN guard
        return False
    margin = float(cheap.get("margin", 1.0))
    denom = abs(inc) + 1e-12
    rel = ((v - inc) if cheap.get("goal", "min") == "min"
           else (inc - v)) / denom
    return rel > margin


def _cheap_gate(target, ns: dict, cheap: dict) -> dict | None:
    """Run bench() with a reduced BENCH_REPS (specs that define it) and
    return a result dict when the candidate is clearly worse than the
    incumbent, else None (proceed to the full bench). Specs without
    BENCH_REPS get no cascade — test() already gates them for free."""
    try:
        orig = int(getattr(target, "BENCH_REPS", 0) or 0)
    except (TypeError, ValueError):
        orig = 0
    want = int(cheap.get("reps", 1) or 1)
    if orig <= want:
        return None
    try:
        target.BENCH_REPS = want
        m = target.bench(ns)
    except Exception:
        m = None                    # fall through: the full bench
                                    # re-raises with a real traceback
    finally:
        target.BENCH_REPS = orig
    if isinstance(m, dict) and _clearly_worse(m, cheap):
        return {"ok": True, "stage": "cheap_gate", "metrics": m}
    return None


def _scrub_secrets() -> None:
    """Drop secret-looking env vars before exec'ing candidate code.
    Deny-wins: a secret-pattern match survives ONLY via the exact-match
    safe list — the old prefix allowlist passed FORGE_API_KEY,
    NVIDIA_API_KEY, NODE_AUTH_TOKEN, CARGO_REGISTRY_TOKEN, OS_PASSWORD…"""
    import re
    pat = re.compile(
        r'(token|secret|pass|credential|private|api[_-]?key|auth|bearer'
        r'|jwt|cookie|session|netrc|(^|_)(key|pwd|url|dsn|cert)($|_))',
        re.IGNORECASE)
    safe = {'TOKENIZERS_PARALLELISM'}
    for k in list(os.environ):
        ku = k.upper()
        if pat.search(ku) and ku not in safe:
            del os.environ[k]


_NONCE = ""


def main() -> int:
    global _NONCE
    # Pop the parent's per-eval nonce BEFORE candidate exec — after this
    # os.environ no longer carries it, so forged __RESULT__ lines can't
    # learn the token (PEB/env-block snapshot reads remain a residual).
    _NONCE = os.environ.pop("_FORGE_EVAL_NONCE", "")
    target_path, cand_path = sys.argv[1], sys.argv[2]
    cheap = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    _scrub_secrets()
    try:
        target = _load_module(target_path, "_evolve_target")
        code = open(cand_path, encoding="utf-8").read()
    except Exception as e:
        _emit({"ok": False, "stage": "load", "detail": str(e)})
        return 2

    # Execute candidate in an isolated namespace; pre-seed only the deps
    # the code actually references (proposer contract: stdlib + numpy +
    # torch) — a torch import costs seconds, skip it when unused.
    ns: dict = {"__name__": "_candidate"}
    pre = []
    if "math." in code:
        pre.append("import math")
    if "np." in code or "numpy" in code:
        pre.append("import numpy as np")
    if "torch" in code:
        pre.append("import torch\nimport torch.nn.functional as F")
    try:
        if pre:
            exec("\n".join(pre), ns)
        exec(compile(code, cand_path, "exec"), ns)
    except Exception:
        _emit({"ok": False, "stage": "exec",
               "detail": traceback.format_exc(limit=3)})
        return 2

    # Correctness gate.
    try:
        t = target.test(ns)
        if isinstance(t, dict):
            ok = bool(t.get("ok"))
            detail = t.get("detail", "")
        else:
            ok, detail = bool(t), ""
        if not ok:
            _emit({"ok": False, "stage": "test", "detail": detail or "failed"})
            return 3
    except Exception:
        _emit({"ok": False, "stage": "test",
               "detail": traceback.format_exc(limit=3)})
        return 3

    # Eval cascade: cheap reduced-rep bench first — a candidate that is
    # clearly worse than the incumbent never pays for the full bench.
    if cheap:
        gated = _cheap_gate(target, ns, cheap)
        if gated is not None:
            _emit(gated)
            return 0

    # Benchmark.
    try:
        metrics = target.bench(ns)
        if not isinstance(metrics, dict):
            _emit({"ok": False, "stage": "bench",
                   "detail": "bench() must return a metrics dict"})
            return 4
        _emit({"ok": True, "stage": "done", "metrics": metrics})
        return 0
    except Exception:
        _emit({"ok": False, "stage": "bench",
               "detail": traceback.format_exc(limit=3)})
        return 4


if __name__ == "__main__":
    # os._exit, not sys.exit: candidate atexit handlers must NEVER run —
    # a registered handler could print a forged __RESULT__ line after the
    # real one and win last-line parsing. Flush first; os._exit skips it.
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)

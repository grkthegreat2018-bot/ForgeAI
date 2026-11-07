"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'fsm_transition_matrix'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['fsm_transition_matrix']
PROMPT = 'Optimize the function to compute the transition matrix for a finite state machine with N states and K symbols. The function should return a dense (N x N) matrix where entry (i,j) is 1 if state i transitions to state j on any symbol, else 0. The current implementation uses nested loops and checks all symbols for each state pair, which is inefficient. Your goal is to vectorize this computation using PyTorch operations to reduce time complexity.'

SEED_CODE = 'import torch\n\ndef fsm_transition_matrix(states, symbols, transitions):\n    """\n    Args:\n        states: List of state names (length N)\n        symbols: List of input symbols (length K)\n        transitions: List of (state, symbol, next_state) tuples\n    Returns:\n        torch.Tensor of shape (N, N) with 1s for valid transitions, 0s otherwise\n    """\n    N = len(states)\n    state_to_idx = {s: i for i, s in enumerate(states)}\n    matrix = torch.zeros((N, N))\n    for (s, _, t) in transitions:\n        i, j = state_to_idx[s], state_to_idx[t]\n        matrix[i, j] = 1\n    return matrix'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import random

def _gen_cases(seed):
    cases = []
    random.seed(seed)
    for _ in range(5):
        N = random.randint(3, 10)
        K = random.randint(2, 5)
        states = [f"s{i}" for i in range(N)]
        symbols = [f"a{i}" for i in range(K)]
        transitions = []
        for s in states:
            for _ in range(random.randint(1, 3)):
                t = random.choice(states)
                sym = random.choice(symbols)
                transitions.append((s, sym, t))
        cases.append((states, symbols, transitions))
    return cases


def _ref():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns[REQUIRED[0]]


def _same(a, b, atol=1e-5, rtol=1e-4):
    import math as _m
    ta, tb = torch.is_tensor(a), torch.is_tensor(b)
    if ta or tb:
        if not (ta and tb) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if torch.equal(a, b):
            return True
        if a.is_complex() or b.is_complex():
            fa = torch.isfinite(a.real) & torch.isfinite(a.imag)
            fb = torch.isfinite(b.real) & torch.isfinite(b.imag)
        else:
            fa, fb = torch.isfinite(a), torch.isfinite(b)
        if not torch.equal(fa, fb):
            return False
        return bool(torch.allclose(a[fa], b[fb], atol=atol, rtol=rtol))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _same(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _same(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return _m.isclose(float(a), float(b),
                              abs_tol=atol, rel_tol=rtol)
        except (TypeError, ValueError):
            return a == b
    return a == b


def _call(fn, case):
    return fn(*case) if isinstance(case, tuple) else fn(case)


def _any_cuda(obj, depth=0):
    if depth > 4:
        return False
    if torch.is_tensor(obj):
        return bool(obj.is_cuda)
    if isinstance(obj, (list, tuple)):
        return any(_any_cuda(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return any(_any_cuda(x, depth + 1) for x in obj.values())
    return False


def test(ns):
    fn = ns.get(REQUIRED[0])
    if not callable(fn):
        return {"ok": False, "detail": f"no {REQUIRED[0]}"}
    ref = _ref()
    _s = random.Random().getrandbits(63)
    cases_a = _gen_cases(_s)
    cases_b = _gen_cases(_s)          # fresh inputs — mutation-safe
    for i, (ca, cb) in enumerate(zip(cases_a, cases_b)):
        try:
            y = _call(fn, ca)
            y0 = _call(ref, cb)
        except Exception as e:
            return {"ok": False,
                    "detail": f"case {i} raised {type(e).__name__}: {e}"}
        try:
            if not _same(y, y0):
                return {"ok": False, "detail": f"case {i} mismatch"}
        except Exception as e:
            return {"ok": False, "detail": f"case {i} compare {e}"}
    return {"ok": True}


def bench(ns):
    fn = ns[REQUIRED[0]]
    cases = _gen_cases(random.Random().getrandbits(63))
    for c in cases:
        _call(fn, c)
    on_cuda = _any_cuda(cases)
    if on_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(BENCH_REPS):
        for c in cases:
            _call(fn, c)
            n += 1
    if on_cuda:
        torch.cuda.synchronize()
    return {"ms_per_call": (time.perf_counter() - t0) / max(n, 1) * 1000}

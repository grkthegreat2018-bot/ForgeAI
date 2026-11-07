"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'filter_special_tokens'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['filter_special_tokens']
PROMPT = 'Optimize a function that filters out special tokens from a vocabulary, excluding a predefined set of legitimate terminators. The function must scan the vocabulary once, match tokens against a regex pattern, and return the IDs of tokens that match the pattern but are not in the exclusion set. The correctness constraint is that the output must exactly match the set of IDs derived from the regex and exclusion criteria.'

SEED_CODE = 'import re\nimport torch\n\ndef filter_special_tokens(vocab: dict[str, int], keep_ids: set[int], pattern: str) -> list[int]:\n    """Filter tokens from vocab that match pattern but are not in keep_ids."""\n    regex = re.compile(pattern)\n    banned_ids = []\n    for token, tid in vocab.items():\n        if tid not in keep_ids and regex.match(token):\n            banned_ids.append(tid)\n    return banned_ids'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import re
import random

def _gen_cases(seed):
    random.seed(seed)
    cases = []
    
    # Small vocab
    vocab1 = {f"<special_{i}>": i for i in range(10)}
    vocab1.update({f"<keep_{i}>": i+10 for i in range(5)})
    keep1 = {i+10 for i in range(5)}
    cases.append((vocab1, keep1, r"^<special_\d+>$"))
    
    # Large vocab with mixed tokens
    vocab2 = {f"token_{i}": i for i in range(1000)}
    vocab2.update({f"<special_{i}>": i+1000 for i in range(500)})
    vocab2.update({f"<keep_{i}>": i+1500 for i in range(200)})
    keep2 = {i+1500 for i in range(200)}
    cases.append((vocab2, keep2, r"^<special_\d+>$"))
    
    # Edge case: empty vocab
    cases.append(({}, {0}, r"^<special_\d+>$"))
    
    # Edge case: all tokens match
    vocab3 = {f"<special_{i}>": i for i in range(100)}
    cases.append((vocab3, set(), r"^<special_\d+>$"))
    
    # Edge case: no tokens match
    vocab4 = {f"token_{i}": i for i in range(100)}
    cases.append((vocab4, set(), r"^<special_\d+>$"))
    
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

"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'compute_normalized_features'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['compute_normalized_features']
PROMPT = "Optimize a function that computes normalized features from a configuration dictionary, where each feature is derived by scaling and clipping a parameter value. The function must return a tensor of normalized values, and the correctness constraint is that the output must match the original implementation's behavior exactly, including the same scaling factors and clipping bounds."

SEED_CODE = "import numpy as np\r\nimport torch\r\n\r\ndef compute_normalized_features(config):\r\n    # Feature 1: Normalized top_k value (scaled and clipped between 0 and 1)\r\n    top_k = config.get('top_k', 256)\r\n    feature1 = np.clip((top_k - 64) / 960, 0, 1)\r\n    \r\n    # Feature 2: Pattern type mapping (standard=0, csa=0.5, csa_hca_hybrid=1.0)\r\n    pt_map = {'standard': 0, 'csa': 0.5, 'csa_hca_hybrid': 1.0}\r\n    feature2 = pt_map.get(config.get('pattern_type', 'standard'), 0)\r\n    \r\n    # Feature 3: Normalized block_size value (scaled and clipped between 0 and 1)\r\n    block_size = config.get('block_size', 32)\r\n    feature3 = np.clip((block_size - 8) / 120, 0, 1)\r\n    \r\n    # Combine into a tensor\r\n    return torch.tensor([feature1, feature2, feature3], dtype=torch.float32)"

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

def _gen_cases(seed):
    cases = [
        {'top_k': 128, 'pattern_type': 'csa', 'block_size': 16},
        {'top_k': 512, 'pattern_type': 'standard', 'block_size': 64},
        {'top_k': 32, 'pattern_type': 'csa_hca_hybrid', 'block_size': 8},
        {'top_k': 1024, 'pattern_type': 'csa', 'block_size': 128},
        {'top_k': 64, 'pattern_type': 'standard', 'block_size': 8},
        {'top_k': 2000, 'pattern_type': 'csa_hca_hybrid', 'block_size': 248},
        {'top_k': 0, 'pattern_type': 'unknown', 'block_size': 0},
        {'top_k': 1000, 'pattern_type': 'csa', 'block_size': 150},
        {'top_k': 192, 'pattern_type': 'standard', 'block_size': 40},
    ]
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

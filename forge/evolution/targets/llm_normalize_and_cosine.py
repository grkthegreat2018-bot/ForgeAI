"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'normalize_and_cosine'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['normalize_and_cosine']
PROMPT = 'Optimize this function to compute the cosine similarity between two normalized text vectors. The function must first normalize the input vectors (convert to lowercase, remove non-alphabetic characters, and split into words), then compute their cosine similarity using PyTorch. The correctness constraint is that the cosine similarity must match the original implementation within a small epsilon (1e-6).'

SEED_CODE = "import torch\nimport re\n\ndef normalize_and_cosine(text1: str, text2: str) -> float:\n    # Normalize text1\n    text1 = text1.lower()\n    text1 = re.sub(r'[^a-z ]', '', text1)\n    words1 = text1.split()\n    \n    # Normalize text2\n    text2 = text2.lower()\n    text2 = re.sub(r'[^a-z ]', '', text2)\n    words2 = text2.split()\n    \n    # Create word sets and intersection\n    set1 = set(words1)\n    set2 = set(words2)\n    intersection = set1 & set2\n    \n    # Compute cosine similarity (simplified version)\n    if not intersection:\n        return 0.0\n    \n    # Convert to word counts\n    count1 = {word: words1.count(word) for word in set1}\n    count2 = {word: words2.count(word) for word in set2}\n    \n    # Compute dot product and magnitudes\n    dot_product = sum(count1[word] * count2[word] for word in intersection)\n    magnitude1 = sum(v ** 2 for v in count1.values()) ** 0.5\n    magnitude2 = sum(v ** 2 for v in count2.values()) ** 0.5\n    \n    return dot_product / (magnitude1 * magnitude2)\n"

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

def _gen_cases():
    random.seed(42)
    cases = [
        # Simple cases
        ("Hello world", "Hello world"),
        ("Hello world", "hello world"),
        ("Hello world", "world hello"),
        ("Hello, world!", "Hello world"),
        # Medium cases
        ("The quick brown fox jumps over the lazy dog", "The lazy dog jumps over the quick brown fox"),
        ("Python is great", "Python is awesome"),
        ("Optimize this code", "Optimize this code efficiently"),
        # Edge cases
        ("", ""),
        ("No common words", "Different words here"),
        ("Repeated words words words", "Repeated words words"),
        # Long cases
        (" " * 1000 + "test", " " * 1000 + "test"),
        ("This is a very long sentence with many words and some repetition repetition", "This is a long sentence with many words and repetition repetition"),
        # Mixed cases
        ("Mixed CASE and punctuation!", "mixed case and punctuation"),
        ("Numbers 123 should be ignored", "Numbers should be ignored")
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
    cases_a = _gen_cases()
    cases_b = _gen_cases()          # fresh inputs — mutation-safe
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
    cases = _gen_cases()
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

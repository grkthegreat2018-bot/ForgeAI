"""Evolution target: faster softmax for decode-path logits.

Goal: minimize ms_per_call on the hot shape while staying numerically
equivalent to torch.softmax (atol 1e-5). Evolve region = the whole
`softmax_fast` program — generators may restructure freely (chunking,
fused ops, dtype tricks, caching) as long as test() passes.
"""

NAME = "fast_softmax"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["kernel", "speed", "softmax", "attention"]
BUDGETS = {"ms_per_call": ("<=", 5.0)}   # reject absurd slowdowns outright
REQUIRED = ["softmax_fast"]
PROMPT = ("Used on decode logits: many calls on float32 rows of 4096 "
          "elements. Try chunked/online softmax, fused reductions, "
          "contiguity, or fewer kernel launches.")

SEED_CODE = '''
import torch


def softmax_fast(x: torch.Tensor) -> torch.Tensor:
    """Baseline: library softmax, dim=-1."""
    return torch.softmax(x.float(), dim=-1)
'''

# ── correctness gate ────────────────────────────────────────────────────
def _inputs():
    import torch
    g = torch.Generator().manual_seed(1234)
    return [
        torch.randn(1, 4096, generator=g),
        torch.randn(64, 4096, generator=g),
        torch.randn(256, 256, generator=g) * 30.0,   # overflow-prone scale
        torch.randn(8, 1, generator=g),
    ]


def test(ns) -> dict:
    import torch
    fn = ns.get("softmax_fast")
    if not callable(fn):
        return {"ok": False, "detail": "no softmax_fast"}
    ref = torch.softmax
    for i, x in enumerate(_inputs()):
        try:
            y = fn(x.clone())
        except Exception as e:
            return {"ok": False, "detail": f"input {i} raised {e}"}
        if not isinstance(y, torch.Tensor) or y.shape != x.shape:
            return {"ok": False, "detail": f"input {i} bad shape"}
        if not torch.allclose(y.float(), ref(x.float(), -1),
                              atol=1e-5, rtol=1e-4):
            return {"ok": False, "detail": f"input {i} wrong values"}
    return {"ok": True}


def bench(ns) -> dict:
    import time
    import torch
    fn = ns["softmax_fast"]
    x = torch.randn(64, 4096, generator=torch.Generator().manual_seed(7))
    for _ in range(10):  # warmup
        fn(x)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        fn(x)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"ms_per_call": ms}

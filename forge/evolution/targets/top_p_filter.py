"""Evolution target: research.sampling_utils.top_p_filter_logits.

Nucleus filter on the decode path. The seed topk-shortcuts then does a
redundant double scatter-map-back (sorted_remove -> topk_remove) and an
in-place clone shift. Goal: identical kept-set, lower ms_per_call.
Gate is torch.equal — the nucleus boundary is semantics, not noise.
"""

import random
NAME = "top_p_filter"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["decoding", "sampler", "hot-path", "kernel"]
BUDGETS = {}
REQUIRED = ["top_p_filter_logits"]
PROMPT = (
    "Faster top-p nucleus filter on (B,V) logits, V~50k-150k. Seed does "
    "topk(k=100) -> sort k -> cumsum -> shift-right via clone -> TWO "
    "scatter map-backs -> scatter into -inf canvas. Ideas: fuse the "
    "remove-shift with a comparison on cumsum, single scatter back to "
    "vocab, avoid building both sorted_remove and topk_remove, reuse "
    "sorted order from topk directly. Output must equal seed EXACTLY "
    "(same -inf positions) for all inputs."
)

SEED_CODE = '''
import torch
import torch.nn.functional as F

_TOP_P_CANDIDATES = 100


def top_p_filter_logits(logits: torch.Tensor, top_p: float,
                        candidates: int = _TOP_P_CANDIDATES) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    k = min(candidates, logits.shape[-1])
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)
    if topk_probs.sum() < top_p:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        indices = remove.scatter(1, sorted_idx, remove)
        return logits.masked_fill(indices, float("-inf"))
    sorted_probs, sorted_idx = torch.sort(topk_probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    remove = cumsum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_remove = torch.zeros_like(sorted_probs)
    sorted_remove.scatter_(-1, sorted_idx, remove.to(sorted_remove.dtype))
    topk_remove = torch.zeros_like(topk_probs)
    topk_remove.scatter_(-1, sorted_idx, sorted_remove)
    topk_logits = topk_logits.masked_fill(topk_remove.bool(), float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, topk_idx, topk_logits)
    return out
'''


def _inputs():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(random.SystemRandom().getrandbits(63))
    cases = []
    for b, v, scale in ((1, 50257, 4.0), (4, 128256, 4.0), (8, 32000, 6.0),
                        (2, 50257, 1.5)):
        lg = torch.randn(b, v, generator=g).to(dev) * scale
        for p in (0.5, 0.9, 0.95, 0.99, 1.0):
            cases.append((lg, p))
    # flat distribution forces the full-sort fallback path
    flat = torch.ones(1, 40000, device=dev) + \
        torch.randn(1, 40000, generator=g).to(dev) * 0.001
    cases += [(flat, 0.9), (flat, 0.5)]
    return cases


def _ref():
    import torch
    import torch.nn.functional as F

    def top_p_filter_logits_ref(logits, top_p, candidates=100):
        if top_p >= 1.0:
            return logits
        k = min(candidates, logits.shape[-1])
        topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        if topk_probs.sum() < top_p:
            sorted_logits, sorted_idx = torch.sort(
                logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            indices = remove.scatter(1, sorted_idx, remove)
            return logits.masked_fill(indices, float("-inf"))
        sorted_probs, sorted_idx = torch.sort(
            topk_probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        remove = cumsum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_remove = torch.zeros_like(sorted_probs)
        sorted_remove.scatter_(-1, sorted_idx, remove.to(sorted_remove.dtype))
        topk_remove = torch.zeros_like(topk_probs)
        topk_remove.scatter_(-1, sorted_idx, sorted_remove)
        topk_logits = topk_logits.masked_fill(
            topk_remove.bool(), float("-inf"))
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, topk_idx, topk_logits)
        return out

    return top_p_filter_logits_ref


def test(ns):
    import torch
    ref = _ref()
    fn = ns.get("top_p_filter_logits")
    if not callable(fn):
        return {"ok": False, "detail": "no top_p_filter_logits"}
    for i, (lg, p) in enumerate(_inputs()):
        try:
            y, y0 = fn(lg, p), ref(lg, p)
        except Exception as e:
            return {"ok": False,
                    "detail": f"input {i} raised {type(e).__name__}: {e}"}
        try:
            ok = torch.equal(y, y0)
        except Exception:
            ok = y == y0
        if not ok:
            return {"ok": False, "detail": f"input {i} mismatch"}
    return {"ok": True}


def bench(ns):
    import time, torch
    fn = ns["top_p_filter_logits"]
    cases = _inputs()
    for lg, p in cases:
        fn(lg, p)
    dev = cases[0][0].device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        for lg, p in cases:
            fn(lg, p)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return {"ms_per_call": (time.perf_counter() - t0) / (reps * len(cases)) * 1000}

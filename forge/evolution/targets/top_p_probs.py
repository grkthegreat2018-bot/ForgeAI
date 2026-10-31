"""Evolution target: research.sampling_utils.top_p_sample_probs.

Nucleus filter on PROBABILITIES (not logits) — renormalizes after
filtering. Seed double-scatters the cutoff mask (sorted -> topk ->
vocab) and sorts 100 topk candidates. Gate: torch.equal on the
renormalized output — the nucleus boundary is semantics.
"""

NAME = "top_p_probs"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["decoding", "sampler", "hot-path", "kernel"]
BUDGETS = {}
REQUIRED = ["top_p_sample_probs"]
PROMPT = (
    "Faster top-p filter on probability tensors (…,V), renormalized "
    "output. Seed: topk(100) -> sum check (full-sort fallback) -> sort "
    "k -> cumsum -> shift -> scatter sorted->topk -> scatter topk->V "
    "-> renormalize. Ideas: single scatter back to vocab, sort once "
    "and reuse order, fuse renormalization into the filtered probs. "
    "Output must equal seed EXACTLY (same support, same renorm)."
)

SEED_CODE = '''
import torch
import torch.nn.functional as F

_TOP_P_CANDIDATES = 100


def top_p_sample_probs(probs: torch.Tensor, top_p: float,
                       candidates: int = _TOP_P_CANDIDATES) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    k = min(candidates, probs.shape[-1])
    topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
    if topk_probs.sum() < top_p:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        full_probs = torch.zeros_like(probs)
        full_probs.scatter_(-1, sorted_idx, sorted_probs)
        return full_probs / full_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    sorted_probs, sorted_idx = torch.sort(topk_probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumsum > top_p
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    topk_filtered = torch.zeros_like(topk_probs)
    topk_filtered.scatter_(-1, sorted_idx, sorted_probs)
    full_probs = torch.zeros_like(probs)
    full_probs.scatter_(-1, topk_idx, topk_filtered)
    return full_probs / full_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
'''


def _inputs():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    cases = []
    for b, v, sharp in ((1, 50257, 4.0), (4, 128256, 2.0), (8, 32000, 6.0),
                        (2, 50257, 1.0), (1, 4096, 12.0)):
        lg = torch.randn(b, v, generator=g).to(dev) * sharp
        p = torch.softmax(lg, dim=-1)
        for tp in (0.5, 0.9, 0.95, 0.99, 1.0):
            cases.append((p, tp))
    flat = torch.full((1, 40000), 1.0 / 40000, device=dev)
    flat += torch.randn(1, 40000, generator=g).to(dev) * 1e-7
    flat = flat.clamp(min=0) / flat.sum()
    cases += [(flat, 0.9), (flat, 0.5)]
    return cases


def _ref():
    import torch

    def ref(probs, top_p, candidates=100):
        if top_p >= 1.0:
            return probs
        k = min(candidates, probs.shape[-1])
        topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
        if topk_probs.sum() < top_p:
            sorted_probs, sorted_idx = torch.sort(
                probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum > top_p
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            full_probs = torch.zeros_like(probs)
            full_probs.scatter_(-1, sorted_idx, sorted_probs)
            return full_probs / full_probs.sum(
                dim=-1, keepdim=True).clamp(min=1e-10)
        sorted_probs, sorted_idx = torch.sort(
            topk_probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        topk_filtered = torch.zeros_like(topk_probs)
        topk_filtered.scatter_(-1, sorted_idx, sorted_probs)
        full_probs = torch.zeros_like(probs)
        full_probs.scatter_(-1, topk_idx, topk_filtered)
        return full_probs / full_probs.sum(
            dim=-1, keepdim=True).clamp(min=1e-10)

    return ref


def test(ns):
    import torch
    ref = _ref()
    fn = ns.get("top_p_sample_probs")
    if not callable(fn):
        return {"ok": False, "detail": "no top_p_sample_probs"}
    for i, (p, tp) in enumerate(_inputs()):
        try:
            y, y0 = fn(p, tp), ref(p, tp)
        except Exception as e:
            return {"ok": False,
                    "detail": f"input {i} raised {type(e).__name__}: {e}"}
        try:
            ok = torch.equal(y, y0) or torch.allclose(
                y, y0, atol=1e-6, rtol=1e-5, equal_nan=True)
        except Exception:
            ok = bool((y == y0).all())
        if not ok:
            return {"ok": False, "detail": f"input {i} mismatch"}
    return {"ok": True}


def bench(ns):
    import time, torch
    fn = ns["top_p_sample_probs"]
    cases = _inputs()
    for p, tp in cases:
        fn(p, tp)
    dev = cases[0][0].device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        for p, tp in cases:
            fn(p, tp)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return {"ms_per_call": (time.perf_counter() - t0) / (reps * len(cases)) * 1000}

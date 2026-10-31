"""Evolution target: forge.engine.engine_common._min_k_filter (decode hot path).

The seed does TWO full torch.sort over the vocab per call plus several
(B, V) temporaries — it runs every generated token. Goal: same output,
lower ms_per_call. Correctness must match the reference bit-for-bit on
the sampled inputs (allclose atol=1e-5 to allow benign fp assoc order).
"""

NAME = "min_k_filter"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["decoding", "sampling", "kernel", "hot-path"]
BUDGETS = {}
REQUIRED = ["_min_k_filter"]
PROMPT = (
    "Rewrite _min_k_filter to be faster on (B, V) logits with V~50k-150k. "
    "The seed sorts the vocab twice, builds linspace/arange every call, and "
    "scatters a full mask. Ideas: reuse one sort's indices, cache the weight "
    "vector per (n, device, dtype), fuse the cliff search, avoid full-vocab "
    "temporaries, or use topk. Output must equal the seed's semantics: keep "
    "tokens [0..cliff_pos] in sorted order, mask the rest to -inf; "
    "sensitivity<=0 returns logits unchanged."
)

SEED_CODE = '''
import torch


def _min_k_filter(logits: torch.Tensor, sensitivity: float) -> torch.Tensor:
    if sensitivity <= 0:
        return logits
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    eps = 1e-8
    diffs = sorted_logits[..., :-1] - sorted_logits[..., 1:]
    n = diffs.shape[-1]
    weights = torch.linspace(1.0, 0.1, n, device=logits.device, dtype=logits.dtype)
    weighted_diffs = diffs * weights
    max_decay = weighted_diffs.max(dim=-1, keepdim=True).values.clamp(min=eps)
    cliff_mask = weighted_diffs > sensitivity * max_decay
    idx = torch.arange(n, device=logits.device).expand_as(cliff_mask)
    last_cliff = torch.where(
        cliff_mask, idx, torch.full_like(idx, -1)).amax(dim=-1, keepdim=True)
    cliff_pos = torch.where(
        last_cliff >= 0, last_cliff,
        weighted_diffs.argmax(dim=-1, keepdim=True))
    positions = torch.arange(sorted_logits.shape[-1], device=logits.device)
    keep = positions <= cliff_pos
    sorted_indices = torch.sort(logits, descending=True, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, ~keep)
    return logits.masked_fill(mask, float("-inf"))
'''


def _inputs():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)
    cases = []
    # Small vocabularies and extreme logit scales are where a weight-
    # profile drift actually flips the cliff argmax — the gate must cover
    # them (a 0.1->0.05 linspace perturbation passed the old 16-case
    # large-vocab-only gate while diverging on 42/280 wider cases).
    for b, v, scale in ((1, 16, 1.0), (1, 128, 4.0), (2, 1024, 8.0),
                        (1, 50257, 4.0), (4, 128256, 4.0),
                        (8, 50257, 8.0), (2, 32000, 2.0),
                        (1, 4096, 0.1), (1, 4096, 20.0)):
        lg = torch.randn(b, v, generator=g).to(dev) * scale
        for s in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            cases.append((lg, s))
    # near-flat logits — all-tie cliffs stress the fallback path
    flat = torch.ones(1, 512, device=dev) + \
        torch.randn(1, 512, generator=g).to(dev) * 1e-4
    for s in (0.5, 0.9):
        cases.append((flat, s))
    return cases


def test(ns):
    import torch
    ref = _ref()   # module-local — candidates cannot spoof it via ns
    fn = ns.get("_min_k_filter")
    if not callable(fn):
        return {"ok": False, "detail": "no _min_k_filter"}
    for i, (lg, s) in enumerate(_inputs()):
        try:
            y, y0 = fn(lg, s), ref(lg, s)
        except Exception as e:
            return {"ok": False, "detail": f"input {i} raised {type(e).__name__}: {e}"}
        try:
            ok = torch.equal(y, y0) or torch.allclose(
                torch.nan_to_num(y, neginf=-1e30),
                torch.nan_to_num(y0, neginf=-1e30), atol=1e-5, rtol=1e-4)
        except Exception:
            ok = y == y0
        if not ok:
            return {"ok": False, "detail": f"input {i} mismatch"}
    return {"ok": True}


def bench(ns):
    import time, torch
    fn = ns["_min_k_filter"]
    cases = _inputs()
    for lg, s in cases:
        fn(lg, s)
    dev = cases[0][0].device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        for lg, s in cases:
            fn(lg, s)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return {"ms_per_call": (time.perf_counter() - t0) / (reps * len(cases)) * 1000}


def _ref():
    import torch

    def _min_k_filter_ref(logits: torch.Tensor, sensitivity: float) -> torch.Tensor:
        if sensitivity <= 0:
            return logits
        sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
        eps = 1e-8
        diffs = sorted_logits[..., :-1] - sorted_logits[..., 1:]
        n = diffs.shape[-1]
        weights = torch.linspace(1.0, 0.1, n, device=logits.device, dtype=logits.dtype)
        weighted_diffs = diffs * weights
        max_decay = weighted_diffs.max(dim=-1, keepdim=True).values.clamp(min=eps)
        cliff_mask = weighted_diffs > sensitivity * max_decay
        idx = torch.arange(n, device=logits.device).expand_as(cliff_mask)
        last_cliff = torch.where(
            cliff_mask, idx, torch.full_like(idx, -1)).amax(dim=-1, keepdim=True)
        cliff_pos = torch.where(
            last_cliff >= 0, last_cliff,
            weighted_diffs.argmax(dim=-1, keepdim=True))
        positions = torch.arange(sorted_logits.shape[-1], device=logits.device)
        keep = positions <= cliff_pos
        sorted_indices = torch.sort(logits, descending=True, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, ~keep)
        return logits.masked_fill(mask, float("-inf"))

    return _min_k_filter_ref

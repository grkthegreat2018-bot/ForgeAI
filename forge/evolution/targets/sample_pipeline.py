"""Evolution target: engine_generation._sample_next_token's filter chain.

The deterministic part of per-token sampling: repetition penalty ->
temperature -> DRY penalty -> min_p -> min_k -> top_k -> top_p ->
masked logits in vocab order (the multinomial draw on top is trivial).
Runs on EVERY generated token of EVERY generate() call — the engine's
hottest pure-torch path after the forward itself. Seed softmaxes twice
(top-p cumsum, then the caller re-softmaxes for multinomial) and makes
several (B, V) temporaries per step.

Gate: masked logits must match the reference on a flag matrix — the
-inf support IS the semantics (a dropped token must stay dropped).
"""

import random
NAME = "sample_pipeline"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["decoding", "sampling", "kernel", "hot-path", "engine"]
BUDGETS = {"ms_per_call": ("<=", 20.0)}
REQUIRED = ["filter_logits"]
PROMPT = (
    "Faster sampling filter chain on (B,V) logits, V~65k-152k. Seed: "
    "rep-penalty index_copy -> /temp -> DRY index_copy_ -> min_p where "
    "-> min_k sort -> topk -> top_p sort+cumsum+masked_fill, scattered "
    "back to vocab order. Ideas: fuse filters into fewer passes, skip "
    "disabled branches cheaply, one sort for min_k+top_p when both on, "
    "avoid full-vocab temporaries via topk-first masking, reuse cached "
    "arange/positions. Output masked logits in ORIGINAL vocab order, "
    "-inf outside support — must match reference exactly (support + "
    "finite values)."
)

SEED_CODE = '''
import torch


_MIN_K_CACHE: dict = {}


def _mk_ctor(fn, *a, **k):
    key = (fn.__name__, repr(a), repr(sorted(k.items())))
    v = _MIN_K_CACHE.get(key)
    if v is None:
        v = fn(*a, **k)
        _MIN_K_CACHE[key] = v
    return v


def _min_k_filter(logits: torch.Tensor, sensitivity: float) -> torch.Tensor:
    if sensitivity <= 0:
        return logits
    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1)
    eps = 1e-8
    diffs = sorted_logits[..., :-1] - sorted_logits[..., 1:]
    n = diffs.shape[-1]
    weights = _mk_ctor(torch.linspace, 1.0, 0.1, n,
                       device=logits.device, dtype=logits.dtype)
    weighted_diffs = diffs * weights
    max_decay = weighted_diffs.max(dim=-1, keepdim=True).values.clamp(min=eps)
    cliff_mask = weighted_diffs > sensitivity * max_decay
    idx = _mk_ctor(torch.arange, n, device=logits.device).expand_as(cliff_mask)
    last_cliff = torch.where(
        cliff_mask, idx, torch.full_like(idx, -1)).amax(dim=-1, keepdim=True)
    cliff_pos = torch.where(
        last_cliff >= 0, last_cliff,
        weighted_diffs.argmax(dim=-1, keepdim=True))
    positions = _mk_ctor(torch.arange, sorted_logits.shape[-1],
                         device=logits.device)
    keep = positions <= cliff_pos
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, ~keep)
    return logits.masked_fill(mask, float("-inf"))


def _apply_dry(next_logits, penalties):
    if not penalties:
        return next_logits
    toks = torch.tensor(list(penalties.keys()), device=next_logits.device)
    pens = torch.tensor(list(penalties.values()),
                        device=next_logits.device, dtype=next_logits.dtype)
    next_logits.index_copy_(
        -1, toks, next_logits.index_select(-1, toks) - pens)
    return next_logits


def filter_logits(logits, temperature, top_k, top_p, repetition_penalty,
                  generated_ids, min_p=0.0, min_k=0.0, dry_penalties=None):
    """Deterministic pre-multinomial filter chain. Returns (B, V)
    masked logits in vocab order: temperature-scaled, rep/DRY penalties
    applied, filtered support set to -inf."""
    next_logits = logits
    if repetition_penalty != 1.0 and generated_ids:
        next_logits = logits.clone()
        repeated = torch.tensor(
            tuple(set(generated_ids[-64:])), device=logits.device)
        next_logits.index_copy_(
            -1, repeated,
            next_logits.index_select(-1, repeated) / repetition_penalty)
    if temperature <= 0:
        return next_logits
    next_logits = next_logits / temperature
    if dry_penalties:
        next_logits = _apply_dry(next_logits, dry_penalties)
    if min_p > 0.0:
        probs = torch.softmax(next_logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = min_p * max_prob
        next_logits = torch.where(
            probs < threshold,
            torch.full_like(next_logits, float("-inf")),
            next_logits)
    if min_k > 0.0:
        next_logits = _min_k_filter(next_logits, min_k)
    if top_k > 0:
        candidate_logits, candidate_ids = torch.topk(
            next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
    else:
        candidate_logits = next_logits
        candidate_ids = torch.arange(
            next_logits.shape[-1], device=logits.device).expand_as(
            next_logits)
    if top_p < 1.0:
        candidate_logits, order = torch.sort(
            candidate_logits, descending=True, dim=-1)
        candidate_ids = candidate_ids.gather(-1, order)
        cumulative_probs = torch.cumsum(
            torch.softmax(candidate_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        candidate_logits = candidate_logits.masked_fill(
            remove, float("-inf"))
    out = torch.full_like(next_logits, float("-inf"))
    out.scatter_(-1, candidate_ids, candidate_logits)
    return out
'''


# ── reference implementation (same math as the seed) ─────────────────────
def _ref():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns["filter_logits"]


def _cases():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(random.SystemRandom().getrandbits(63))
    logits = torch.randn(4, 8192, generator=g).to(dev)
    sharp = (torch.randn(2, 8192, generator=g) * 8).to(dev)
    dry = {11: 0.4, 77: 0.9, 1234: 0.2}
    gen = [int(x) for x in torch.randint(0, 8000, (80,), generator=g)]
    out = []
    for lg in (logits, sharp):
        out += [
            (lg, 0.0, 0, 1.0, 1.0, [], 0.0, 0.0, None),
            (lg, 0.8, 80, 1.0, 1.0, [], 0.0, 0.0, None),
            (lg, 0.8, 80, 0.95, 1.05, gen, 0.0, 0.0, None),
            (lg, 0.7, 0, 0.9, 1.1, gen, 0.05, 0.0, dry),
            (lg, 1.2, 200, 0.98, 1.0, gen, 0.0, 0.4, dry),
            (lg, 0.5, 50, 0.7, 1.2, gen[-20:], 0.1, 0.6, dry),
            (lg, 0.9, 8192, 0.5, 1.05, gen, 0.02, 0.2, dry),
        ]
    return out


def _same(a, b) -> bool:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    if not torch.equal(fa, fb):
        return False
    return torch.allclose(a[fa], b[fb], atol=1e-5, rtol=1e-4)


def test(ns):
    fn = ns.get("filter_logits")
    if not callable(fn):
        return {"ok": False, "detail": "no filter_logits"}
    ref = _ref()
    for i, args in enumerate(_cases()):
        try:
            y = fn(*[a.clone() if hasattr(a, "clone") else a
                     for a in args[:1]] + list(args[1:]))
            y0 = ref(*args)
        except Exception as e:
            return {"ok": False, "detail": f"case {i} raised {e}"}
        try:
            if not _same(y, y0):
                return {"ok": False, "detail": f"case {i} mismatch"}
        except Exception as e:
            return {"ok": False, "detail": f"case {i} compare {e}"}
    return {"ok": True}


def bench(ns):
    import time
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fn = ns["filter_logits"]
    g = torch.Generator().manual_seed(random.SystemRandom().getrandbits(63))
    logits = torch.randn(8, 65536, generator=g).to(dev)
    gen = [int(x) for x in torch.randint(0, 65500, (64,), generator=g)]
    dry = {5: 0.5, 91: 0.7, 404: 0.3, 2077: 1.1, 9000: 0.2, 45000: 0.6}
    args = (logits, 0.8, 80, 0.95, 1.05, gen, 0.05, 0.0, dry)

    def _sync():
        if dev == "cuda":
            torch.cuda.synchronize()
    for _ in range(10):
        fn(*args)
    _sync()
    t0 = time.perf_counter()
    n = 60
    for _ in range(n):
        fn(*args)
    _sync()
    return {"ms_per_call": (time.perf_counter() - t0) / n * 1000}

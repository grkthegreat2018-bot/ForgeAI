"""Evolution target: engine_common._dry_penalties.

Pure-python DRY repeat-penalty matcher (llama.cpp sampler_dry port).
Seed scans last_n positions for the last token, then extends each hit
backward one token at a time — O(last_n * match_len) worst case with
python-level comparisons. Headroom: suffix/KMP-style matching, early
exits, dict/slice tricks, or vectorized numpy compare. Gate is dict
equality — the penalty values are semantics.
"""

import random
NAME = "dry_penalties"
GOAL = "min"
METRIC = "us_per_call"
TAGS = ["decoding", "penalty", "algo", "hot-path"]
BUDGETS = {}
REQUIRED = ["_dry_penalties"]
PROMPT = (
    "Faster DRY repeat-penalty matcher on token lists (~500-2000 ids, "
    "last_n window ~256). Seed: for each i in window where ctx[i]=="
    "ctx[last], walk backward counting matches; penalty = multiplier * "
    "base ** (r - allowed). Ideas: find candidate positions once (set/"
    "dict of positions per token, reversed view, str/bytes tricks if "
    "vocab fits), skip positions that can't beat allowed_length, "
    "vectorize with numpy, or use a suffix-match structure. Output dict "
    "must equal seed EXACTLY."
)

SEED_CODE = '''
def _dry_penalties(
    context_ids: list[int],
    last_n: int,
    allowed_length: int,
    multiplier: float,
    base: float,
) -> dict[int, float]:
    n = len(context_ids)
    if multiplier <= 0 or n < 2 or last_n <= 0:
        return {}
    last_idx = n - 1
    start = max(0, last_idx - last_n)
    match_len: dict[int, int] = {}
    for i in range(start, last_idx):
        if context_ids[i] != context_ids[last_idx]:
            continue
        r = 1
        while i - r >= 0 and context_ids[i - r] == context_ids[last_idx - r]:
            r += 1
        if r > allowed_length:
            nxt = context_ids[i + 1]
            if r > match_len.get(nxt, 0):
                match_len[nxt] = r
    return {
        tok: multiplier * (base ** (r - allowed_length))
        for tok, r in match_len.items()
    }
'''


def _mk_ctx(kind, n, g):
    import random
    if kind == "repeats":
        # long context with planted repeating phrases (the DRY use-case)
        vocab = 2000
        ctx = [g.randrange(vocab) for _ in range(n)]
        # plant a 30-token phrase twice near the tail + a 60-token once
        ph = [g.randrange(vocab) for _ in range(30)]
        ctx[n - 40:n - 10] = ph
        ctx[n - 200:n - 170] = ph
        ctx[n - 5:] = ph[:5]
        ph2 = [g.randrange(vocab) for _ in range(60)]
        ctx[n - 300:n - 240] = ph2
        ctx[n - 120:n - 60] = ph2
        ctx[-1] = ph2[-1]
        return ctx
    if kind == "uniform":
        return [g.randrange(50000) for _ in range(n)]
    if kind == "lowvocab":
        # repetitive text (code-like): many accidental repeats
        return [g.randrange(64) for _ in range(n)]
    raise ValueError(kind)


def _inputs():
    import random
    g = random.Random()
    cases = []
    for kind in ("repeats", "uniform", "lowvocab"):
        for n in (600, 1500):
            ctx = _mk_ctx(kind, n, g)
            cases.append((ctx, 256, 2, 0.8, 1.75))
            cases.append((ctx, 96, 0, 1.0, 1.3))
    cases.append(([1, 2, 3, 4, 5, 6, 7, 8, 9, 5], 256, 2, 0.8, 1.75))
    cases.append(([7] * 400, 128, 2, 0.8, 1.75))
    cases.append(([1, 2], 256, 2, 0.8, 1.75))
    cases.append(([], 256, 2, 0.8, 1.75))
    cases.append((list(range(500)) + list(range(300, 500)), 256, 3, 1.2, 2.0))
    return cases


def _ref():
    def ref(context_ids, last_n, allowed_length, multiplier, base):
        n = len(context_ids)
        if multiplier <= 0 or n < 2 or last_n <= 0:
            return {}
        last_idx = n - 1
        start = max(0, last_idx - last_n)
        match_len = {}
        for i in range(start, last_idx):
            if context_ids[i] != context_ids[last_idx]:
                continue
            r = 1
            while (i - r >= 0
                   and context_ids[i - r] == context_ids[last_idx - r]):
                r += 1
            if r > allowed_length:
                nxt = context_ids[i + 1]
                if r > match_len.get(nxt, 0):
                    match_len[nxt] = r
        return {tok: multiplier * (base ** (r - allowed_length))
                for tok, r in match_len.items()}
    return ref


def test(ns):
    ref = _ref()
    fn = ns.get("_dry_penalties")
    if not callable(fn):
        return {"ok": False, "detail": "no _dry_penalties"}
    for i, args in enumerate(_inputs()):
        try:
            y, y0 = fn(*args), ref(*args)
        except Exception as e:
            return {"ok": False,
                    "detail": f"input {i} raised {type(e).__name__}: {e}"}
        if not isinstance(y, dict) or y != y0:
            return {"ok": False, "detail": f"input {i} mismatch"}
    return {"ok": True}


def bench(ns):
    import time
    fn = ns["_dry_penalties"]
    cases = _inputs()
    for a in cases:
        fn(*a)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        for a in cases:
            fn(*a)
    return {"us_per_call": (time.perf_counter() - t0) / (reps * len(cases)) * 1e6}

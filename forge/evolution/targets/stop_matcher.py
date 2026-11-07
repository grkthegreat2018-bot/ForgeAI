"""Evolution target: engine_common._StopMatcher — incremental stop-string
matching in the decode loop.

Every generated token pushes one id through decode(tail) + substring
scan over each stop string. Seed re-decodes the tail window from token
ids on every push — tokenizer.decode dominates (list comp + join +
special-token filter per step).

Gate: a scripted push trace must produce the same bool sequence as the
reference, over multiple stop sets / decode stubs / skip flags — a
missed stop string means runaway generation, a false positive means
truncated output.
"""

import random
NAME = "stop_matcher"
GOAL = "min"
METRIC = "us_per_push"
TAGS = ["decoding", "string", "hot-path", "engine"]
BUDGETS = {"us_per_push": ("<=", 50.0)}
REQUIRED = ["StopMatcher"]
PROMPT = (
    "Faster incremental stop matcher. Seed: each push() re-decodes the "
    "last _tail_tokens ids via tok.decode + `any(s in text)`. Ideas: "
    "incremental decode (new suffix only) + rolling text buffer, Aho-"
    "Corasick/KMP over decoded chars, skip decode when no stop can "
    "overlap the new token, memoize decoded tails. Semantics EXACT: "
    "same True/False per push, honoring skip_special_tokens and the "
    "_tail_tokens window (a stop spanning older tokens must still hit)."
)

SEED_CODE = '''
class StopMatcher:
    """Incremental stop-string matcher for token-by-token decode."""

    def __init__(self, tokenizer, stop, skip_special_tokens=True):
        self.stop = [s for s in (stop or []) if s]
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self._ids = []
        self._tail_tokens = (
            2 * max(len(s) for s in self.stop) + 16 if self.stop else 0)

    def __bool__(self):
        return bool(self.stop)

    def push(self, token_id: int) -> bool:
        if not self.stop:
            return False
        self._ids.append(token_id)
        tail = self._ids[-self._tail_tokens:]
        text = self.tokenizer.decode(
            tail, skip_special_tokens=self.skip_special_tokens)
        return any(s in text for s in self.stop)
'''


IM_END = "<" + "|im_end|" + ">"
IM_START = "<" + "|im_start|" + ">"
_SPECIALS = {151645: IM_END, 151644: IM_START, 43: "###"}


class _StubTok:
    """Deterministic decode stub: specials map to their strings (dropped
    when skip_special_tokens), regular ids to fixed 2-char pairs so
    multi-token stop spans actually exercise."""

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            s = _SPECIALS.get(i)
            if s is not None:
                if not skip_special_tokens:
                    out.append(s)
            else:
                out.append(chr(97 + (i % 26)) + chr(97 + ((i * 7) % 26)))
        return "".join(out)


def _drive(cls, stop, skip, seed=17, n=4000):
    import random
    rng = random.Random(seed)
    m = cls(_StubTok(), stop, skip_special_tokens=skip)
    res = []
    specials = list(_SPECIALS)
    for i in range(n):
        tok = specials[i % len(specials)] if i % 97 == 0 else (
            rng.randrange(0, 200) if rng.random() < 0.9 else 151644)
        res.append(bool(m.push(tok)))
    m2 = cls(_StubTok(), stop, skip_special_tokens=skip)
    hit = []
    if stop:
        s = next((x for x in stop if x), None)
        if s == IM_END:
            for t in [1, 2, 3] + ([] if skip else [151645]):
                hit.append(bool(m2.push(t)))
        elif s == "###":
            for t in [1, 2, 43, 4]:
                hit.append(bool(m2.push(t)))
        else:
            for t in range(10):
                hit.append(bool(m2.push(t)))
    res.append(hit)
    return res


def test(ns):
    cls = ns.get("StopMatcher")
    if cls is None:
        return {"ok": False, "detail": "no StopMatcher"}
    ref: dict = {}
    exec(SEED_CODE, ref)
    ref_cls = ref["StopMatcher"]
    stops = [
        [IM_END], ["###"], ["ab", "yz", IM_END],
        ["", "###", "x"], [], None,
        ["a"], [IM_START + IM_END],
    ]
    for i, stop in enumerate(stops):
        for skip in (True, False):
            try:
                got = _drive(cls, stop, skip)
                want = _drive(ref_cls, stop, skip)
            except Exception as e:
                return {"ok": False,
                        "detail": f"stops[{i}] skip={skip} raised {e}"}
            if got != want:
                return {"ok": False,
                        "detail": f"stops[{i}] skip={skip} diverged"}
    return {"ok": True}


def bench(ns):
    import random
    import time
    cls = ns["StopMatcher"]
    rng = random.Random()
    m = cls(_StubTok(), [IM_END, "###", "qw"],
            skip_special_tokens=True)
    toks = [rng.randrange(0, 500) for _ in range(60000)]
    for t in toks[:2000]:
        m.push(t)
    t0 = time.perf_counter()
    for t in toks:
        m.push(t)
    dt = time.perf_counter() - t0
    return {"us_per_push": dt / len(toks) * 1e6}

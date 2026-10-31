"""Evolution target: ngram_mem.SuffixMemory (R65) — the streaming
exact-suffix sidecar attached to every generate() call.

Per-token work: feed_one writes the tail->successor pair at every
order (O(orders) hashes + dict lookups), lift_last turns the deepest
matching rows into a dense (V,) logit lift, draft() rolls a virtual
tail for speculative prompts. On a 128k-vocab model the dense numpy
pass and the per-order python dict walk are the hot spots — both run
every single decoded token.

Gate: an op trace (feed/feed_one/lift_last/draft/snapshot/restore)
must match the reference exactly — lift arrays equal to fp32 precision
and draft lists identical, including across decay events.
"""

NAME = "suffix_mem"
GOAL = "min"
METRIC = "us_per_token"
TAGS = ["decoding", "memory", "sidecar", "hot-path", "engine"]
BUDGETS = {"us_per_token": ("<=", 800.0)}
REQUIRED = ["SuffixMem"]
PROMPT = (
    "Faster SuffixMem: per-order rolling suffix->successor tables. "
    "feed_one cost = O(orders) hash+dict writes; lift_last = per-order "
    "row lookups + a dense (V,) numpy lift; draft = virtual-tail argmax "
    "walk. Ideas: rolling-hash reuse across orders (each order's key "
    "is a *suffix* of the longest — recompute is O(sum orders) but a "
    "length-indexed prefix-hash ring makes it O(orders)), vectorize "
    "lift assembly, cache the unigram/floor arrays, sparse lift output, "
    "faster decay (count halving pass). Semantics EXACT: same lift "
    "values (fp32), same drafts, same decay survivors."
)

SEED_CODE = '''
import numpy as np
import torch


def _mix64(x: int) -> int:
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    return x


class SuffixMem:
    def __init__(self, vocab_size: int,
                 orders=(4, 8, 16, 32, 64),
                 gate: float = 1.0,
                 order_weight: float = 0.7,
                 max_cells_per_order: int = 200_000,
                 unigram_mix: float = 0.1,
                 uniform_floor: float = 0.01):
        self.vocab = int(vocab_size)
        self.orders = tuple(sorted(int(o) for o in orders))
        self.max_order = self.orders[-1]
        self.gate = float(gate)
        self.order_weight = float(order_weight)
        self.max_cells = int(max_cells_per_order)
        self.unigram_mix = float(unigram_mix)
        self.uniform_floor = float(uniform_floor)

        self.tables = {o: {} for o in self.orders}
        self.unigram = np.zeros(self.vocab, dtype=np.float64)
        self.total = 0
        self._decays = 0
        self._tail = []

    def _suffix_key(self, toks: list, order: int) -> int:
        h = 0x9E3779B97F4A7C15
        for tok in toks[-order:]:
            h = _mix64(h ^ (tok + 1))
        return h or 1

    def _write_pair(self, succ_tok: int) -> None:
        for o in self.orders:
            if len(self._tail) >= o:
                key = self._suffix_key(self._tail, o)
                tab = self.tables[o]
                row = tab.get(key)
                if row is None:
                    if len(tab) >= self.max_cells:
                        self._decay(o)
                        row = tab.get(key)
                        if row is None:
                            row = {}
                            tab[key] = row
                    else:
                        row = {}
                        tab[key] = row
                row[succ_tok] = row.get(succ_tok, 0) + 1

    def _decay(self, order: int) -> None:
        tab = self.tables[order]
        for key in list(tab.keys()):
            row = tab[key]
            for tok in list(row.keys()):
                row[tok] //= 2
                if row[tok] == 0:
                    del row[tok]
            if not row:
                del tab[key]
        self._decays += 1

    def feed(self, ids) -> int:
        for tok in ids:
            self.feed_one(int(tok))
        return len(self._tail)

    def feed_one(self, tok: int) -> None:
        self._write_pair(tok)
        if 0 <= tok < self.vocab:
            self.unigram[tok] += 1
        self.total += 1
        self._tail.append(tok)
        if len(self._tail) > self.max_order:
            del self._tail[: len(self._tail) - self.max_order]

    def _row_probs(self, order: int):
        if len(self._tail) < order:
            return None, 0.0
        row = self.tables[order].get(self._suffix_key(self._tail, order))
        if not row:
            return None, 0.0
        total = sum(row.values())
        return {t: c / total for t, c in row.items()}, float(total)

    def lift_last(self, device=None, dtype=torch.float32,
                  gate=None) -> torch.Tensor:
        uni = self.unigram / max(self.total, 1)
        floor = np.full(self.vocab, np.log(self.uniform_floor),
                        dtype=np.float32)
        lift = np.zeros(self.vocab, dtype=np.float32)
        max_o = self.orders[-1]
        matched = 0
        for o in self.orders:
            probs, _cnt = self._row_probs(o)
            if probs is None:
                continue
            matched += 1
            w = self.order_weight ** (max_o - o)
            for tok, p in probs.items():
                p_s = p + self.unigram_mix * uni[tok] + self.uniform_floor
                lift[tok] += w * np.log(p_s)
        if matched == 0:
            return torch.zeros(self.vocab, dtype=dtype, device=device)
        g = self.gate if gate is None else float(gate)
        out = g * (lift + 0.25 * floor)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    def draft(self, n: int = 4) -> list:
        tail = list(self._tail)
        out = []
        for _ in range(n):
            best = None
            for o in reversed(self.orders):
                if len(tail) < o:
                    continue
                row = self.tables[o].get(self._suffix_key(tail, o))
                if row:
                    best = max(row.items(), key=lambda kv: kv[1])[0]
                    break
            if best is None:
                break
            out.append(best)
            tail.append(best)
            if len(tail) > self.max_order:
                tail = tail[-self.max_order:]
        return out

    def snapshot_state(self) -> dict:
        return {
            "tables": {o: {k: dict(r) for k, r in t.items()}
                       for o, t in self.tables.items()},
            "unigram": self.unigram.copy(),
            "total": self.total,
            "tail": list(self._tail),
            "decays": self._decays,
        }

    def restore_state(self, snap: dict) -> None:
        self.tables = {o: {k: dict(r) for k, r in t.items()}
                       for o, t in snap["tables"].items()}
        self.unigram = snap["unigram"].copy()
        self.total = snap["total"]
        self._tail = list(snap["tail"])
        self._decays = snap.get("decays", 0)
'''


def _ref_cls():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns["SuffixMem"]


def _stream(seed: int, n: int, vocab: int):
    import random
    rng = random.Random(seed)
    # skewed + repeating runs so suffix hits actually occur
    hot = [rng.randrange(vocab // 4) for _ in range(64)]
    out = []
    while len(out) < n:
        if rng.random() < 0.35 and len(out) > 32:
            out.extend(hot[: rng.randrange(4, 48)])
        else:
            out.append(rng.randrange(vocab))
    return out[:n]


def _drive(cls, vocab: int, small: bool):
    """Deterministic op trace -> list of comparable results."""
    m = cls(vocab,
            max_cells_per_order=48 if small else 200_000)
    trace = _stream(11, 2600, vocab)
    res = []
    for i, tok in enumerate(trace):
        m.feed_one(tok)
        if i % 37 == 0:
            lift = m.lift_last()
            res.append(float(lift.abs().sum()))
            res.append(float(lift.max()))
        if i % 211 == 0:
            res.append(list(m.draft(5)))
        if i == 1300:
            snap = m.snapshot_state()
        if i == 1500:
            m.restore_state(snap)
    res.append(list(m.draft(6)))
    res.append(m.total)
    return res


def _cmp(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_cmp(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= 1e-4 * max(1.0, abs(b))
    return a == b


def test(ns):
    cls = ns.get("SuffixMem")
    if cls is None:
        return {"ok": False, "detail": "no SuffixMem"}
    ref = _ref_cls()
    for tag, vocab, small in (("v512", 512, True), ("v4096", 4096, False)):
        try:
            got = _drive(cls, vocab, small)
            want = _drive(ref, vocab, small)
        except Exception as e:
            return {"ok": False, "detail": f"{tag} raised {e}"}
        if not _cmp(got, want):
            return {"ok": False, "detail": f"{tag} trace diverged"}
    return {"ok": True}


def bench(ns):
    import time
    cls = ns["SuffixMem"]
    vocab = 4096
    m = cls(vocab)
    stream = _stream(5, 30000, vocab)
    # warm the tables so reads hit
    m.feed(stream[:8000])
    t0 = time.perf_counter()
    for i, tok in enumerate(stream):
        m.feed_one(tok)
        if i % 4 == 0:
            m.lift_last()
        if i % 1024 == 0:
            m.draft(4)
    dt = time.perf_counter() - t0
    return {"us_per_token": dt / len(stream) * 1e6}

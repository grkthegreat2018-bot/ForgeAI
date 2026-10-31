"""Evolution target: prefix_cache.RadixPrefixCache — SGLang-style radix
tree for cross-request prefix sharing.

Every generate() with the radix cache does lookup_longest_prefix (tree
walk + a second re-walk to find the deepest KV-bearing node) then
insert (walk again + possible edge split + BFS LRU eviction scan). All
pure-Python pointer chasing over {first_token -> node} dicts — the
match loop and the double-walk are the headroom.

Gate: an op trace (insert / lookup / LRU-evict) must return identical
(matched_len, kv_id) results and identical stats to the reference —
the tree structure is observable through lookup lengths.

NOTE: the production wrapper takes a torch tensor and calls
ids[0].tolist(); this target uses plain lists so the search surface is
the tree logic, not the tensor glue.
"""

NAME = "radix_prefix"
GOAL = "min"
METRIC = "us_per_lookup"
TAGS = ["cache", "radix", "tree", "hot-path", "engine"]
BUDGETS = {"us_per_lookup": ("<=", 200.0)}
REQUIRED = ["RadixPrefixCache"]
PROMPT = (
    "Faster radix prefix cache. Seed: _match walks nodes matching token "
    "ids; lookup_longest_prefix then RE-WALKS from root to find the "
    "deepest kv_state-bearing node; insert walks a third time; "
    "_evict_lru BFS-scans the whole tree. Ideas: carry best-kv state "
    "through the single match walk (no re-walk), track node depth/acc-"
    "len so inserts skip recomputation, maintain an LRU heap/dll of "
    "kv nodes instead of BFS, children keyed without per-step dict.get "
    "overhead. Semantics EXACT: same matched_len, same kv_state, same "
    "hit/miss counts, same eviction victims."
)

SEED_CODE = '''
class RadixPrefixCache:
    """Radix-tree cross-request prefix sharing (SGLang-style)."""

    class _Node:
        __slots__ = ("children", "kv_state", "last_used",
                     "ref_count", "token_ids")

        def __init__(self):
            self.token_ids = []
            self.kv_state = None
            self.ref_count = 0
            self.children = {}
            self.last_used = 0.0

    def __init__(self, max_entries: int = 256):
        self.root = self._Node()
        self.max_entries = max_entries
        self._n_entries = 0
        self._hits = 0
        self._misses = 0
        self._clock = 0

    def _match(self, node, token_ids, start: int):
        matched = 0
        cur = node
        parent = None
        i = start
        while i < len(token_ids):
            if matched < len(cur.token_ids):
                if cur.token_ids[matched] == token_ids[i]:
                    matched += 1
                    i += 1
                    continue
                else:
                    break
            child = cur.children.get(token_ids[i])
            if child is None:
                break
            parent, cur = cur, child
            matched = 0
            if matched < len(cur.token_ids) and \
                    cur.token_ids[matched] == token_ids[i]:
                matched += 1
                i += 1
        return i - start, cur, matched, parent

    def _tick(self):
        self._clock += 1
        return float(self._clock)

    def _split_edge(self, node, parent, pos: int):
        mid = self._Node()
        mid.token_ids = node.token_ids[:pos]
        mid.children[node.token_ids[pos]] = node
        mid.last_used = self._tick()
        node.token_ids = node.token_ids[pos:]
        if parent is not None:
            parent.children[mid.token_ids[0]] = mid
        return mid

    def lookup_longest_prefix(self, token_ids):
        if len(token_ids) <= 16:
            return None
        matched, node, _edge_pos, _parent = self._match(
            self.root, token_ids, 0)
        if matched <= 0:
            self._misses += 1
            return None
        best_len = 0
        best_kv = None
        best_node = None
        cur = self.root
        acc = 0
        i = 0
        while i < matched:
            edge_len = min(len(cur.token_ids), matched - i) \
                if cur.token_ids else 0
            if cur.token_ids:
                acc += edge_len
                i += edge_len
                if cur.kv_state is not None and acc > best_len:
                    best_len = acc
                    best_kv = cur.kv_state
                    best_node = cur
                if i >= matched:
                    break
            if i < len(token_ids):
                child = cur.children.get(token_ids[i])
                if child is None:
                    break
                cur = child
            else:
                break
        if best_kv is not None:
            self._hits += 1
            best_node.last_used = self._tick()
            return best_len, best_kv
        self._misses += 1
        return None

    def insert(self, token_ids, kv_state) -> None:
        if not token_ids:
            return
        matched, node, edge_pos, parent = self._match(
            self.root, token_ids, 0)
        if 0 < edge_pos < len(node.token_ids):
            node = self._split_edge(node, parent, edge_pos)
        remaining = token_ids[matched:]
        if not remaining:
            node.kv_state = kv_state
            node.last_used = self._tick()
            return
        new_node = self._Node()
        new_node.token_ids = remaining[:]
        new_node.kv_state = kv_state
        new_node.last_used = self._tick()
        node.children[remaining[0]] = new_node
        self._n_entries += 1
        if self._n_entries > self.max_entries:
            self._evict_lru()

    def _evict_lru(self) -> None:
        best = None
        best_time = float("inf")
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.kv_state is not None and node.ref_count == 0:
                if node.last_used < best_time:
                    best = node
                    best_time = node.last_used
            for child in node.children.values():
                queue.append(child)
        if best is not None:
            best.kv_state = None
            self._n_entries -= 1

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": self._n_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }
'''


def _ref_cls():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns["RadixPrefixCache"]


def _streams(seed: int, n: int, length: int, shared: int):
    """n token streams sharing a `shared`-token common prefix, then
    diverging — the realistic cache workload (system prompt reuse)."""
    import random
    rng = random.Random(seed)
    prefix = [rng.randrange(0, 150000) for _ in range(shared)]
    out = []
    for _ in range(n):
        s = list(prefix)
        s += [rng.randrange(0, 150000)
              for _ in range(length - shared)]
        out.append(s)
    return out


def _drive(cls, max_entries):
    cache = cls(max_entries=max_entries)
    streams = _streams(21, 40, 220, 64)
    res = []
    kvid = 0
    # insert half, interleave lookups, insert the rest
    for i, s in enumerate(streams):
        if i < 24 or i % 3 == 0:
            cache.insert(s, f"kv{kvid}")
            kvid += 1
        q = streams[(i * 7 + 3) % len(streams)]
        res.append(cache.lookup_longest_prefix(q))
        # also a miss-shaped query (fresh divergence)
        res.append(cache.lookup_longest_prefix(
            q[:100] + [999999 + i, i]))
        res.append(cache.stats())
    res.append(len(cache._n_entries and cache.stats()["entries"]
                     and [1]) and cache.stats())
    return res


def test(ns):
    cls = ns.get("RadixPrefixCache")
    if cls is None:
        return {"ok": False, "detail": "no RadixPrefixCache"}
    ref = _ref_cls()
    for tag, cap in (("cap256", 256), ("cap12", 12)):
        try:
            got = _drive(cls, cap)
            want = _drive(ref, cap)
        except Exception as e:
            return {"ok": False, "detail": f"{tag} raised {e}"}
        if got != want:
            return {"ok": False, "detail": f"{tag} trace diverged"}
    return {"ok": True}


def bench(ns):
    import time
    cls = ns["RadixPrefixCache"]
    cache = cls(max_entries=256)
    inserts = _streams(31, 500, 128, 48)
    for i, s in enumerate(inserts):
        cache.insert(s, f"kv{i}")
    queries = _streams(32, 2000, 128, 48)
    # warm
    for q in queries[:100]:
        cache.lookup_longest_prefix(q)
    t0 = time.perf_counter()
    for q in queries:
        cache.lookup_longest_prefix(q)
    dt = time.perf_counter() - t0
    return {"us_per_lookup": dt / len(queries) * 1e6}

"""Evolution target: prefix_cache._chunk_hash/_prefix_hash/prefix_hashes.

ChunkedPrefixCache keys every stored prompt by a blake2b rolling chain:
one _chunk_hash per 256-token chunk plus a _prefix_hash combine per
boundary — struct.pack of the whole chunk each call. Runs on every
generate() with caching enabled; a 4k prompt is 16 packs + 32 digests
through Python hashlib each time.

Gate: exact int equality of the full hash chain — the chain IS a
storage key, so any deviation is a silent cache miss.
"""

NAME = "prefix_hash"
GOAL = "min"
METRIC = "us_per_chain"
TAGS = ["cache", "hash", "hot-path", "engine"]
BUDGETS = {"us_per_chain": ("<=", 2000.0)}
REQUIRED = ["_chunk_hash", "_prefix_hash", "prefix_hashes"]
PROMPT = (
    "Faster rolling prefix-hash chain over token-id chunks. Seed does "
    "hashlib.blake2b(struct.pack('<Ni', *chunk)) per chunk then another "
    "blake2b over two <Q words per boundary. Ideas: reuse a single "
    "hashlib object via copy(), preallocate the struct format, pack "
    "via array/memoryview instead of *args splat, fewer intermediate "
    "bytes objects. MUST produce bit-identical 64-bit ints — the chain "
    "is a storage key; a wrong hash silently breaks the cache."
)

SEED_CODE = '''
import hashlib
import struct


def _chunk_hash(token_ids: list) -> int:
    h = hashlib.blake2b(
        struct.pack(f"<{len(token_ids)}i", *token_ids),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")


def _prefix_hash(prev: int, chunk_h: int) -> int:
    h = hashlib.blake2b(
        struct.pack("<Q", prev & 0xFFFFFFFFFFFFFFFF) +
        struct.pack("<Q", chunk_h & 0xFFFFFFFFFFFFFFFF),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")


def prefix_hashes(token_ids: list, chunk_size: int = 256) -> list:
    hashes = []
    prev = 0
    for i in range(0, len(token_ids), chunk_size):
        ch = _chunk_hash(token_ids[i:i + chunk_size])
        prev = _prefix_hash(prev, ch)
        hashes.append(prev)
    return hashes
'''


def _ref():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns


def _cases():
    import random
    rng = random.Random(3)
    out = []
    for n in (0, 1, 17, 255, 256, 257, 511, 2048, 4097):
        out.append([rng.randrange(-2**31, 2**31) for _ in range(n)])
    return out


def test(ns):
    for name in ("_chunk_hash", "_prefix_hash", "prefix_hashes"):
        if not callable(ns.get(name)):
            return {"ok": False, "detail": f"missing {name}"}
    ref = _ref()
    ch, ph, chain = ns["_chunk_hash"], ns["_prefix_hash"], ns["prefix_hashes"]
    rch, rph, rchain = ref["_chunk_hash"], ref["_prefix_hash"], ref["prefix_hashes"]
    for i, ids in enumerate(_cases()):
        try:
            for cs in (16, 64, 256):
                if chain(ids, cs) != rchain(ids, cs):
                    return {"ok": False, "detail": f"chain {i} cs={cs}"}
        except Exception as e:
            return {"ok": False, "detail": f"chain {i} raised {e}"}
    try:
        if ch([1, -2, 3]) != rch([1, -2, 3]):
            return {"ok": False, "detail": "_chunk_hash"}
        if ph(123456789, -7) != rph(123456789, -7):
            return {"ok": False, "detail": "_prefix_hash"}
        if ph(0, 0) != rph(0, 0):
            return {"ok": False, "detail": "_prefix_hash zero"}
    except Exception as e:
        return {"ok": False, "detail": f"direct raised {e}"}
    return {"ok": True}


def bench(ns):
    import random
    import time
    chain = ns["prefix_hashes"]
    rng = random.Random(9)
    prompts = [[rng.randrange(0, 152000) for _ in range(4096)]
               for _ in range(48)]
    for p in prompts[:4]:
        chain(p, 256)
    t0 = time.perf_counter()
    for p in prompts:
        chain(p, 256)
    dt = time.perf_counter() - t0
    return {"us_per_chain": dt / len(prompts) * 1e6}

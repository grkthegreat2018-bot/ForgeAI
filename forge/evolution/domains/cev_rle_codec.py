"""Non-code evolution domain: a pattern/token-efficiency codec.

This is a code_evolve TARGET SPEC (SEED_CODE/test/bench contract — see
forge/evolution/targets/fast_softmax.py), NOT a BaseDomain subclass; it
lives in domains/ to mark it as a non-ML-kernel problem class.

The searched program is a byte codec: compress(data) -> blob and
decompress(blob) -> data. test() gates roundtrip fidelity on a
generated corpus (run-heavy + structured + text-like + noise bytes);
bench scores `ratio` = compressed_bytes / input_bytes plus speed. The
SEED is a bare PackBits-style RLE — anything with literals+dictionary
or stdlib zlib/lzma compresses much better, so `ratio` has real
headroom while roundtrip keeps candidates honest.

Run standalone:  python forge/evolution/domains/cev_rle_codec.py
Evolve it:       python scripts/rsi_evolve.py --targets cev_rle_codec \
                    --evals 60 --no-train --rounds 1
"""

NAME = "cev_rle_codec"
GOAL = "min"
METRIC = "ratio"
TAGS = ["codec", "compression", "tokens", "noncode"]
BUDGETS = {"ratio": ("<=", 1.05), "ms_per_call": ("<=", 200.0)}
REQUIRED = ["compress", "decompress"]
PROMPT = ("Non-code target: a byte codec. Roundtrip must be EXACT on a "
          "mixed corpus (long runs, repeated patterns, text, random "
          "noise). The seed is plain run-length encoding that expands "
          "on non-run data. Stdlib is allowed — dictionary/entropy "
          "coders or smarter hybrid RLE usually win big on ratio.")

SEED_CODE = '''
_MARK = 0xFF


def compress(data: bytes) -> bytes:
    """PackBits-style RLE: runs >= 4 -> [MARK, count, byte]; literal
    0xFF bytes -> [MARK, 0, byte]; everything else raw."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        run = 1
        while i + run < n and data[i + run] == data[i] and run < 255:
            run += 1
        b = data[i]
        if run >= 4:
            out += bytes((_MARK, run, b))
        elif b == _MARK:
            out += bytes((_MARK, 0, b))
        else:
            out += data[i:i + run]
        i += run
    return bytes(out)


def decompress(blob: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(blob)
    while i < n:
        b = blob[i]
        if b == _MARK and i + 2 < n + 1 and i + 1 < n:
            cnt = blob[i + 1]
            if cnt == 0 and i + 2 < n:
                out.append(blob[i + 2])
                i += 3
                continue
            if i + 2 < n:
                out += bytes((blob[i + 2],)) * cnt
                i += 3
                continue
        out.append(b)
        i += 1
    return bytes(out)
'''


# ── generated corpus ──────────────────────────────────────────────────
def _gen_corpus(seed: int) -> list[bytes]:
    """Deterministic mixed corpus: runs, repeated patterns, text-like
    bytes, and pure noise (which no codec can shrink)."""
    import random
    rng = random.Random(seed)
    corpus: list[bytes] = []
    # long runs
    corpus.append(bytes(300) + b"\x01" * 150 + bytes(range(50)))
    corpus.append(b"\xAA" * 512)
    # repeated short pattern
    pat = bytes(rng.randint(0, 200) for _ in range(7))
    corpus.append(pat * 90)
    corpus.append(b"the quick brown fox jumps over the lazy dog. " * 12)
    # text-like with embedded runs
    words = [b"alpha", b"beta", b"gamma", b"delta"]
    t = b" ".join(rng.choice(words) for _ in range(80))
    corpus.append(t + b"\x00" * 64 + t[:100])
    # sparse: mostly zeros with rare spikes
    sparse = bytearray(400)
    for _ in range(15):
        sparse[rng.randrange(400)] = rng.randint(1, 255)
    corpus.append(bytes(sparse))
    # pure noise — worst case, must not blow up >5%
    corpus.append(bytes(rng.randrange(256) for _ in range(300)))
    return corpus


def test(ns) -> dict:
    comp, decomp = ns.get("compress"), ns.get("decompress")
    if not (callable(comp) and callable(decomp)):
        return {"ok": False, "detail": "need compress+decompress"}
    for i, data in enumerate(_gen_corpus(11)):
        try:
            blob = comp(data)
            back = decomp(blob)
        except Exception as e:
            return {"ok": False, "detail": f"case {i} raised {e}"}
        if not isinstance(back, (bytes, bytearray)):
            return {"ok": False, "detail": f"case {i} non-bytes output"}
        if bytes(back) != data:
            return {"ok": False, "detail": f"case {i} roundtrip failed"}
    return {"ok": True}


def bench(ns) -> dict:
    import time
    comp = ns["compress"]
    corpus = _gen_corpus(42)            # held-out: test() uses seed 11
    for d in corpus[:3]:
        comp(d)
    t0 = time.perf_counter()
    total_in = total_out = 0
    for d in corpus:
        total_in += len(d)
        total_out += len(comp(d))
    ms = (time.perf_counter() - t0) / len(corpus) * 1000
    return {"ratio": total_out / total_in, "ms_per_call": ms}


if __name__ == "__main__":
    _ns: dict = {}
    exec(SEED_CODE, _ns)
    print("seed test:", test(_ns))
    print("seed bench:", bench(_ns))

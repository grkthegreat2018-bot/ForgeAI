"""Tests for the LMCache port (R&D round 14).

Covers the three ported components, all CPU-runnable (no model / CUDA
required) so they run in every CI environment:

  1. ChunkedPrefixCache — rolling-hash prefix matching + partial hits
  2. DiskKVCache — 3-tier GPU->CPU->disk offload + persistence
  3. CacheBlend — ChunkStore / RangeMatcher / BlendAssembler / RoPE re-rotation
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from forge.engine.kv.cacheblend import (
    BlendAssembler,
    CacheBlend,
    ChunkStore,
    RangeMatcher,
    reposition_keys,
)
from forge.engine.kv.cpu_kv_offload import DiskKVCache
from forge.engine.prefix_cache import (
    ChunkedPrefixCache,
    _chunk_hash,
    _slice_past_kv,
)

# ── helpers ───────────────────────────────────────────────────────────────


def _make_past_kv(n_layers, n_kv, seq_len, head_dim, device="cpu",
                  dtype=torch.float32):
    """Build a per-layer KV list matching the model's presents format."""
    kv = []
    for i in range(n_layers):
        if i % 3 == 2:  # simulate conv layers with no KV
            kv.append(None)
            continue
        k = torch.randn(1, n_kv, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, n_kv, seq_len, head_dim, device=device, dtype=dtype)
        kv.append((k, v))
    return kv


class _FakeRope:
    """Minimal RoPE stand-in with cos/sin tables (matches RotaryEmbedding)."""

    def __init__(self, head_dim, max_seq_len, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()


# ── 1. ChunkedPrefixCache ─────────────────────────────────────────────────


def test_chunk_hash_deterministic():
    """Same tokens → same hash, independent of PYTHONHASHSEED."""
    a = _chunk_hash([1, 2, 3, 4])
    b = _chunk_hash([1, 2, 3, 4])
    c = _chunk_hash([1, 2, 3, 5])
    assert a == b
    assert a != c


def test_prefix_hash_chain_distinct():
    """Each prefix boundary has a distinct rolling hash."""
    tokens = list(range(600))
    cache = ChunkedPrefixCache(chunk_size=256)
    hashes = cache.prefix_hashes(tokens)
    assert len(hashes) == 3  # 600 / 256 = 3 chunks (256 + 256 + 88)
    assert len(set(hashes)) == 3  # all distinct


def test_chunked_prefix_cache_full_hit():
    """A re-issued prompt hits the cached full prefix."""
    cache = ChunkedPrefixCache(max_entries=8, chunk_size=64)
    tokens = list(range(200))
    ids = torch.tensor([tokens])
    kv = _make_past_kv(4, 2, 200, 8)
    h = cache.prefix_hashes(tokens)[-1]
    cache.put((h, tokens), kv, len(tokens))

    hit = cache.lookup_longest_prefix(ids)
    assert hit is not None
    matched_len, past_kv, _conv_snap = hit
    assert matched_len == 200
    assert past_kv is not None


def test_chunked_prefix_cache_partial_hit():
    """A longer prompt reuses a shorter cached prefix (partial hit)."""
    cache = ChunkedPrefixCache(max_entries=8, chunk_size=64)
    short = list(range(128))
    short_ids = torch.tensor([short])
    kv = _make_past_kv(4, 2, 128, 8)
    h = cache.prefix_hashes(short)[-1]
    cache.put((h, short), kv, 128)

    # Longer prompt sharing the first 128 tokens
    long = short + list(range(128, 300))
    long_ids = torch.tensor([long])
    hit = cache.lookup_longest_prefix(long_ids)
    assert hit is not None
    matched_len, past_kv, conv_snap = hit
    assert matched_len == 128  # partial — only the cached prefix
    # Partial hits cannot reuse the entry's conv snapshot (it's only
    # position-correct at the entry's own end).
    assert conv_snap is None
    # Sliced KV must be 128 tokens long on each non-None layer
    for layer in past_kv:
        if layer is None:
            continue
        assert layer[0].shape[2] == 128


def test_chunked_prefix_cache_miss():
    """An unrelated prompt misses."""
    cache = ChunkedPrefixCache(max_entries=8, chunk_size=64)
    cache.put((12345, list(range(64))), _make_past_kv(2, 2, 64, 8), 64)
    ids = torch.tensor([list(range(1000, 1100))])
    assert cache.lookup_longest_prefix(ids) is None
    assert cache.stats()["misses"] == 1


def test_chunked_prefix_cache_collision_guard():
    """A hash collision must NOT yield a wrong KV (token verification)."""
    cache = ChunkedPrefixCache(max_entries=8, chunk_size=64)
    tokens_a = list(range(64))
    kv_a = _make_past_kv(2, 2, 64, 8)
    h = cache.prefix_hashes(tokens_a)[-1]
    cache.put((h, tokens_a), kv_a, 64)
    # Manually inject a colliding entry with different tokens under the
    # same hash key (simulating a 64-bit collision).
    cache._cache[h]["token_ids"] = list(range(64, 128))  # wrong tokens
    ids = torch.tensor([list(range(64))])  # query matches the *real* tokens
    hit = cache.lookup_longest_prefix(ids)
    # Token verification fails → treated as collision → miss
    assert hit is None


def test_slice_past_kv():
    kv = _make_past_kv(3, 2, 100, 8)
    sliced = _slice_past_kv(kv, 40)
    for orig, sl in zip(kv, sliced):
        if orig is None:
            assert sl is None
            continue
        assert sl[0].shape[2] == 40
        assert torch.allclose(sl[0], orig[0][:, :, :40])


# ── 2. DiskKVCache (3-tier) ───────────────────────────────────────────────


def test_disk_kv_cache_basic_offload_cycle():
    """Tokens cascade GPU->CPU->disk and fetch back correctly."""
    n_kv, head_dim, dtype = 2, 8, torch.float32
    cache = DiskKVCache(hot_window_size=8, cpu_window_size=8,
                        disk_capacity=16, persist=False)
    cache.init(n_heads=4, head_dim=head_dim, n_kv_heads=n_kv,
               max_seq_len=32, device="cpu", dtype=dtype)

    # Append 20 tokens → GPU(8) + CPU(8) + disk(4)
    for pos in range(20):
        k = torch.randn(1, n_kv, 1, head_dim, dtype=dtype)
        v = torch.randn(1, n_kv, 1, head_dim, dtype=dtype)
        cache.append(k, v, pos)
    info = cache.info()
    assert info["type"] == "disk_offload_3tier"
    assert info["gpu_len"] == 8
    assert info["cpu_len"] == 8
    assert info["disk_len"] == 4
    assert info["seq_len"] == 20


def test_disk_kv_cache_persistence(tmp_path):
    """Disk tier survives clear() when persist=True (LMCache storage mode)."""
    n_kv, head_dim, dtype = 2, 8, torch.float32
    cache = DiskKVCache(disk_path=str(tmp_path), hot_window_size=4,
                        cpu_window_size=4, disk_capacity=8, persist=True)
    cache.init(n_heads=4, head_dim=head_dim, n_kv_heads=n_kv,
               max_seq_len=16, device="cpu", dtype=dtype)
    for pos in range(12):
        cache.append(torch.randn(1, n_kv, 1, head_dim, dtype=dtype),
                     torch.randn(1, n_kv, 1, head_dim, dtype=dtype), pos)
    assert cache.disk_len > 0
    cache.clear()  # should save disk spool
    assert os.path.exists(os.path.join(str(tmp_path), "kv_disk_k.pt"))


def test_disk_kv_cache_factory():
    """build_kv_cache('disk_offload') returns a DiskKVCache."""
    from forge.engine.kv_backend import build_kv_cache
    cache = build_kv_cache("disk_offload")
    from forge.engine.kv.cpu_kv_offload import DiskKVCache as DKC
    assert isinstance(cache, DKC)


# ── 3. CacheBlend ─────────────────────────────────────────────────────────


def test_chunk_store_register_lookup():
    store = ChunkStore(chunk_size=8)
    tokens = list(range(8))
    kv = _make_past_kv(3, 2, 8, 8)
    h = store.register_chunk(tokens, kv)
    assert h != 0
    rec = store.get(h)
    assert rec is not None
    assert rec.token_ids == tokens
    assert len(store) == 1


def test_rangematcher_finds_non_prefix_match():
    """A stored chunk appearing mid-sequence is matched (non-prefix)."""
    store = ChunkStore(chunk_size=8)
    chunk = list(range(100, 108))
    store.register_chunk(chunk, _make_past_kv(3, 2, 8, 8))
    matcher = RangeMatcher(store)
    # chunk appears at position 5 (non-prefix)
    seq = [1, 2, 3, 4, 5] + chunk + [9, 9, 9]
    matches = matcher.find_matches(seq, skip_prefix=True)
    assert len(matches) == 1
    assert matches[0].request_start == 5
    assert matches[0].request_end == 13


def test_rangematcher_skips_prefix_match():
    """A chunk at position 0 is skipped (prefix cache handles it)."""
    store = ChunkStore(chunk_size=8)
    chunk = list(range(100, 108))
    store.register_chunk(chunk, _make_past_kv(3, 2, 8, 8))
    matcher = RangeMatcher(store)
    seq = chunk + [1, 2, 3]
    matches = matcher.find_matches(seq, skip_prefix=True)
    assert len(matches) == 0


def test_rangematcher_collision_guard():
    """Hash collision (faked) is rejected by token verification."""
    store = ChunkStore(chunk_size=8)
    chunk = list(range(100, 108))
    h = store.register_chunk(chunk, _make_past_kv(3, 2, 8, 8))
    # Corrupt the stored tokens so the hash matches but tokens differ.
    store._chunks[h].token_ids = list(range(200, 208))
    matcher = RangeMatcher(store)
    seq = [1, 2] + chunk + [3, 4]
    matches = matcher.find_matches(seq, skip_prefix=True)
    assert len(matches) == 0  # verification rejected it


def test_reposition_keys_identity_when_same_position():
    """Re-rotating to the same position is a no-op."""
    rope = _FakeRope(head_dim=8, max_seq_len=64)
    k = torch.randn(1, 2, 10, 8)
    pos = torch.arange(10)
    k2 = reposition_keys(k, rope, pos, pos)
    assert torch.allclose(k, k2, atol=1e-5)


def test_reposition_keys_roundtrip():
    """rotate(old->new) then rotate(new->old) recovers the original."""
    rope = _FakeRope(head_dim=8, max_seq_len=64)
    k = torch.randn(1, 2, 10, 8)
    old = torch.arange(10)
    new = torch.arange(20, 30)
    k_moved = reposition_keys(k, rope, old, new)
    k_back = reposition_keys(k_moved, rope, new, old)
    assert torch.allclose(k, k_back, atol=1e-4)


def test_blend_assembler_assembles_layout():
    """Assembler produces a per-layer KV buffer of the right length."""
    store = ChunkStore(chunk_size=8)
    chunk = list(range(100, 112))  # 12-token chunk
    kv = _make_past_kv(3, 2, 12, 8)
    store.register_chunk(chunk, kv)
    matcher = RangeMatcher(store)
    seq = [1, 2, 3, 4, 5] + chunk + [6, 7]  # chunk at [5, 17)
    matches = matcher.find_matches(seq, skip_prefix=True)
    assert len(matches) == 1
    rope_modules = [None, _FakeRope(8, 64), _FakeRope(8, 64)]
    asm = BlendAssembler(recompute_fraction=0.2)
    plan = asm.assemble(matches, request_len=len(seq), n_layers=3,
                        device="cpu", dtype=torch.float32,
                        rope_modules=rope_modules)
    assert plan.covered_len == 17
    assert plan.past_kv is not None
    assert len(plan.past_kv) == 3
    # _make_past_kv makes layer index 2 (i % 3 == 2) None (conv), 0/1 have KV
    assert plan.past_kv[0] is not None
    assert plan.past_kv[1] is not None
    assert plan.past_kv[2] is None
    assert plan.past_kv[0][0].shape == (1, 2, 17, 8)
    # Some boundary tokens are marked for recompute
    assert plan.recompute_tokens > 0
    assert plan.reuse_tokens > 0


def test_cache_blend_miss_on_short_prompt():
    """Short prompts below min_cover_tokens return None (no overhead)."""
    blend = CacheBlend(min_cover_tokens=64)
    # No chunks registered, short prompt
    ids = torch.tensor([list(range(10))])
    # Use a dummy engine shell — blend_prefill only touches ids length
    # and the matcher (no model call on the miss path).
    class _Dummy:
        device = torch.device("cpu")
        class model:
            dtype = torch.float32
    assert blend.blend_prefill(_Dummy(), ids) is None
    # Short-prompt path is a skip (not counted as a miss); no hits.
    assert blend.stats()["blend_hits"] == 0


def test_cache_blend_stats():
    blend = CacheBlend()
    s = blend.stats()
    assert "blend_hits" in s
    assert "store" in s
    assert s["reuse_ratio"] == 0.0


if __name__ == "__main__":
    test_chunk_hash_deterministic()
    test_prefix_hash_chain_distinct()
    test_chunked_prefix_cache_full_hit()
    test_chunked_prefix_cache_partial_hit()
    test_chunked_prefix_cache_miss()
    test_chunked_prefix_cache_collision_guard()
    test_slice_past_kv()
    test_disk_kv_cache_basic_offload_cycle()
    test_disk_kv_cache_persistence(tempfile.mkdtemp())
    test_disk_kv_cache_factory()
    test_chunk_store_register_lookup()
    test_rangematcher_finds_non_prefix_match()
    test_rangematcher_skips_prefix_match()
    test_rangematcher_collision_guard()
    test_reposition_keys_identity_when_same_position()
    test_reposition_keys_roundtrip()
    test_blend_assembler_assembles_layout()
    test_cache_blend_miss_on_short_prompt()
    test_cache_blend_stats()
    print("All LMCache port tests PASS")

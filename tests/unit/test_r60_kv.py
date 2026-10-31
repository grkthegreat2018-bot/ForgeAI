"""Tests for the R60 KV-cache trio and the live decode-time bridge:

  - BeaconKVCache  (arXiv 2609.04971) — beacon-centroid key eviction
  - AnchorKVCache  (arXiv 2608.02901) — farthest-point anchors +
    residual-coverage eviction
  - MetaKVCache    (arXiv 2609.07966) — feature-driven dispatcher
  - ForgeEngine._maybe_compact_past_kv — the decode-loop bridge that
    actually invokes compact_past_kv on the model's per-layer
    (k, v) tuple entries.

All CPU-runnable; tensors are small and deterministic.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from forge.engine.kv.anchor_kv import AnchorKVCache
from forge.engine.kv.beacon_kv import BeaconKVCache
from forge.engine.kv.metakv import MetaKVCache
from forge.engine.kv_backend import KVCacheStrategy, build_kv_cache


def _kv(B=1, n_kv=2, T=1, d=4, start=0.0):
    k = torch.arange(start, start + B * n_kv * T * d,
                     dtype=torch.float32).reshape(B, n_kv, T, d)
    return k, k.clone()


def _init(cache, max_seq=64, n_kv=2, d=4):
    cache.init(n_heads=4, head_dim=d, n_kv_heads=n_kv,
               max_seq_len=max_seq, device="cpu", dtype=torch.float32)
    return cache


# ═══════════════════════════════════════════════════════════════════════
# BeaconKV
# ═══════════════════════════════════════════════════════════════════════

def test_beacon_eviction_respects_budget_and_keeps_sinks_recent():
    cache = _init(BeaconKVCache(budget=16, sink=2, window=4,
                                n_beacons=4, slack=8))
    k, v = _kv(T=40)
    cache.append(k, v, 0)  # 40 > compact_over=24 → evict to ~budget
    assert cache.seq_len == 16
    kept_k, kept_v = cache.get()
    # sinks (first 2) and the recent window (last 4) are protected
    assert torch.equal(kept_k[0, 0, :2], k[0, 0, :2])
    assert torch.equal(kept_k[0, 0, -4:], k[0, 0, -4:])
    assert torch.equal(kept_v[0, 0, -4:], v[0, 0, -4:])
    info = cache.info()
    assert info["type"] == "beacon" and info["evictions"] == 1


def test_beacon_compact_past_kv_slices_only_tuple_entries():
    cache = _init(BeaconKVCache(budget=16, sink=2, window=4, n_beacons=4))
    k40, v40 = _kv(T=40)
    k10, v10 = _kv(T=10, start=10000.0)
    ssm_state = {"conv": torch.zeros(3), "ssm": None}
    bare_tensor = torch.zeros(5)
    past = [(k40, v40), None, ssm_state, (k10, v10), bare_tensor]
    out = cache.compact_past_kv(past)
    assert isinstance(out, list)
    # over-budget tuple entry is sliced to the budget
    assert out[0][0].shape[-2] <= 16
    assert out[0][1].shape[-2] == out[0][0].shape[-2]
    # non-tuple entries pass through by identity
    assert out[1] is None
    assert out[2] is ssm_state
    assert out[4] is bare_tensor
    # under-budget tuple entry is the same object (stats updated only)
    assert out[3] is past[3]


def test_beacon_compact_preserves_order_and_tuple_input():
    cache = _init(BeaconKVCache(budget=16, sink=2, window=4, n_beacons=4))
    k, v = _kv(T=40)
    out = cache.compact_past_kv(((k, v),))
    assert isinstance(out, tuple)
    kept = out[0][0]
    # position order preserved: kept entries are increasing in the
    # original arange layout (first dim values strictly increase)
    vals = kept[0, 0, :, 0]
    assert (vals[1:] > vals[:-1]).all()


def test_beacon_beacons_update_per_layer():
    cache = _init(BeaconKVCache(budget=8, sink=2, window=2, n_beacons=4))
    k, v = _kv(T=20)
    cache.compact_past_kv([(k, v), (k.clone(), v.clone()), None])
    # two attention-layer entries → two beacon sets tracked
    assert len(cache._beacons) == 2
    assert cache._beacons[0].shape == (4, 4)


# ═══════════════════════════════════════════════════════════════════════
# AnchorKV
# ═══════════════════════════════════════════════════════════════════════

def _clustered_keys(T=24, d=8, seed=0):
    """near-dup clusters on e1/e2 + one unique vector at position 12."""
    g = torch.Generator().manual_seed(seed)
    e1 = torch.zeros(d); e1[0] = 1.0
    e2 = torch.zeros(d); e2[1] = 1.0
    uni = torch.zeros(d); uni[3] = 1.0
    keys = []
    for t in range(T):
        if t < 12:
            keys.append(e1 + 0.001 * torch.randn(d, generator=g))
        elif t == 12:
            keys.append(uni)
        elif t < 20:
            keys.append(e2 + 0.001 * torch.randn(d, generator=g))
        else:
            keys.append(torch.randn(d, generator=g))
    k = torch.stack(keys).view(1, 1, T, d)
    return k, k.clone()


def test_anchor_low_error_evicted_first():
    """Positions well covered by anchors (low residual error) are
    evicted before unique keys — the paper's core ordering."""
    T, d = 24, 8
    k, v = _clustered_keys(T=T, d=d)
    cache = _init(AnchorKVCache(budget=12, sink=2, window=4,
                                n_anchors=4), max_seq=64, n_kv=1, d=d)
    cache.append(k, v, 0)
    assert cache.seq_len <= 12
    kept = cache._kept_orig.tolist()
    # the unique direction survives (as an anchor or top-error fill)
    assert 12 in kept
    # sinks + recent window always survive
    assert all(p in kept for p in (0, 1))
    assert all(p in kept for p in (20, 21, 22, 23))
    # near-dup e1 cluster (positions 2..11) is mostly evicted — kept
    # members are its anchor(s) plus any top-error fill slots that landed
    # there. Budget arithmetic: 12 = 2 sinks + 4 window + ≤4 anchors +
    # error-fill; a dup can occupy at most ~anchor+fill slots.
    n_dup_kept = sum(1 for p in kept if 2 <= p < 12)
    assert n_dup_kept <= 5
    # the complementary e2 cluster is under the same pressure
    n_dup2_kept = sum(1 for p in kept if 13 <= p < 20)
    assert n_dup_kept + n_dup2_kept <= 6


def test_anchor_get_full_reconstructs_length():
    T, d = 24, 8
    k, v = _clustered_keys(T=T, d=d)
    cache = _init(AnchorKVCache(budget=12, sink=2, window=4,
                                n_anchors=4), max_seq=64, n_kv=1, d=d)
    cache.append(k, v, 0)
    k_full, _v_full = cache.get_full()
    assert k_full.shape == (1, 1, cache.orig_len, d)
    assert cache.orig_len == T
    # retained positions are bit-exact
    kept_idx = cache._kept_orig
    k_ret, _ = cache.get()
    assert torch.equal(k_full[:, :, kept_idx], k_ret)
    # evicted positions are rebuilt as s * anchor (finite, nonzero where
    # the source key was nonzero)
    assert torch.isfinite(k_full).all()


def test_anchor_compact_past_kv_slices_only_tuples():
    cache = _init(AnchorKVCache(budget=12, sink=2, window=4,
                                n_anchors=4), max_seq=64)
    k30, v30 = _kv(T=30)
    k10, v10 = _kv(T=10, start=10000.0)
    past = [(k30, v30), None, {"ssm": torch.zeros(2)}, (k10, v10)]
    out = cache.compact_past_kv(past)
    assert out[0][0].shape[-2] <= 12
    assert out[1] is None and out[2] is past[2]
    assert out[3] is past[3]


# ═══════════════════════════════════════════════════════════════════════
# MetaKV
# ═══════════════════════════════════════════════════════════════════════

def test_metakv_select_table():
    mk = MetaKVCache()
    # long context + memory pressure → anchor
    assert mk.select({"seq_len": 8192,
                      "free_vram_bytes": 2 * 1024**3}) == "anchor"
    assert mk.select({"max_seq_len": 8192,
                      "free_vram_bytes": 2 * 1024**3}) == "anchor"
    # short context under pressure → beacon (default row)
    assert mk.select({"seq_len": 2048,
                      "free_vram_bytes": 2 * 1024**3}) == "beacon"
    # long context with plenty of VRAM → beacon
    assert mk.select({"seq_len": 8192,
                      "free_vram_bytes": 32 * 1024**3}) == "beacon"
    # CPU (no VRAM probe) → beacon regardless of length
    assert mk.select({"seq_len": 8192, "free_vram_bytes": None}) == "beacon"


def test_metakv_init_picks_inner_and_delegates():
    mk = _init(MetaKVCache(), max_seq=64)
    # CPU: no free-VRAM feature → default row → beacon
    assert mk.selected == "beacon"
    assert isinstance(mk._inner, BeaconKVCache)
    info = mk.info()
    assert info["type"] == "metakv" and info["inner"] == "beacon"
    assert info["inner_type"] == "beacon"
    # delegation: append/get run on the inner cache
    k, v = _kv(T=8)
    mk.append(k, v, 0)
    kk, _vv = mk.get(None)
    assert kk.shape[-2] == 8


def test_metakv_compact_past_kv_passthrough():
    # shrink the inner budget so the passthrough provably compacts
    mk = _init(MetaKVCache(inner_overrides={"budget": 8, "window": 2,
                                          "sink": 2}), max_seq=64)
    k30, v30 = _kv(T=30)
    out = mk.compact_past_kv([(k30, v30)])
    assert 0 < out[0][0].shape[-2] <= 8


def test_metakv_force_inner():
    mk = _init(MetaKVCache(force_inner="anchor"), max_seq=64, n_kv=1, d=8)
    assert isinstance(mk._inner, AnchorKVCache)
    assert mk.selected == "anchor"


# ═══════════════════════════════════════════════════════════════════════
# Registry + fallback chain
# ═══════════════════════════════════════════════════════════════════════

def test_registry_builds_all_three():
    for name, cls in (("beacon", BeaconKVCache),
                      ("anchor", AnchorKVCache),
                      ("metakv", MetaKVCache)):
        cache = build_kv_cache(name)
        assert isinstance(cache, cls), name
        assert isinstance(cache, KVCacheStrategy)
        _init(cache)
        assert cache.info()["type"] == name


def test_fallback_chain_has_new_strategies():
    from forge.engine.engine_activation import _ActivationMixin
    chain = _ActivationMixin._KV_FALLBACK_CHAIN
    for name in ("beacon", "anchor", "metakv"):
        assert name in chain, f"{name} missing from _KV_FALLBACK_CHAIN"
        assert chain[name] == ["s4r", "standard", "cpu_offload"]


# ═══════════════════════════════════════════════════════════════════════
# Live decode-time bridge
# ═══════════════════════════════════════════════════════════════════════

def _stub_engine(kv_cache):
    """Bare _GenerationMixin instance — just the attrs the bridge needs."""
    from forge.engine.engine_generation import _GenerationMixin
    eng = _GenerationMixin.__new__(_GenerationMixin)
    eng.kv_cache = kv_cache
    eng._log = lambda *a, **k: None
    return eng


def test_bridge_noop_without_compact_hook():
    eng = _stub_engine(kv_cache=None)
    k, v = _kv(T=40)
    past = [(k, v)]
    assert eng._maybe_compact_past_kv(past) is past
    # strategy object without the hook → same
    eng.kv_cache = object()
    assert eng._maybe_compact_past_kv(past) is past
    # None past_kv → None
    assert eng._maybe_compact_past_kv(None) is None


def test_bridge_watermark_trigger_and_throttle():
    cache = _init(BeaconKVCache(budget=8, sink=2, window=2,
                                n_beacons=4, slack=4), max_seq=32)
    assert cache.compact_over == 12
    eng = _stub_engine(cache)
    k10, v10 = _kv(T=10)
    past_small = [(k10, v10)]
    # under watermark, counter=1 → not due → unchanged object
    assert eng._maybe_compact_past_kv(past_small) is past_small
    k14, v14 = _kv(T=14)
    out = eng._maybe_compact_past_kv([(k14, v14)])
    # over watermark → compacts on the first eligible call
    assert out[0][0].shape[-2] <= 8


def test_bridge_periodic_due_and_force():
    from forge.engine.engine_generation import _KV_COMPACT_EVERY
    cache = _init(BeaconKVCache(budget=8, sink=2, window=2,
                                n_beacons=4, slack=4), max_seq=32)
    eng = _stub_engine(cache)
    k6, v6 = _kv(T=6)
    past = [(k6, v6)]
    # run the counter up to the periodic boundary — the hook runs even
    # under the watermark (T=6 < budget=8 so no slicing, but beacons
    # are refreshed by the call).
    eng._kv_compact_ctr = _KV_COMPACT_EVERY - 1
    out = eng._maybe_compact_past_kv(past)
    assert out[0][0].shape[-2] == 6  # under budget: no slicing
    assert cache._beacons  # the due-run still updated beacon stats
    # force=True: one-shot post-prefill compaction
    k20, v20 = _kv(T=20)
    eng._kv_compact_ctr = 0
    out = eng._maybe_compact_past_kv([(k20, v20)], force=True)
    assert out[0][0].shape[-2] <= 8


def test_bridge_exception_guard_returns_input():
    class _Boom:
        compact_over = 4
        def compact_past_kv(self, past_kv, **kw):
            raise RuntimeError("boom")
    eng = _stub_engine(_Boom())
    k, v = _kv(T=10)
    past = [(k, v)]
    logged = []
    eng._log = lambda msg, **kw: logged.append(msg)
    out = eng._maybe_compact_past_kv(past, force=True)
    assert out is past  # failure → unmodified cache
    assert logged and "KV compaction" in logged[0]


def test_decode_loop_invokes_bridge():
    """The per-token decode path calls _maybe_compact_past_kv after
    unpack_output_with_kv — source-level regression guard so the hook
    can't silently go dead again."""
    import inspect

    from forge.engine import engine_generation as gen
    src = inspect.getsource(gen._GenerationMixin._decode_tokens)
    assert "_maybe_compact_past_kv(past_kv)" in src
    src_prefill = inspect.getsource(gen._GenerationMixin._prefill)
    assert "_maybe_compact_past_kv(past_kv, force=True)" in src_prefill

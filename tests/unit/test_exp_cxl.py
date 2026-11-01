"""CPU tests for forge/exp/cxlmem.py — tier logic, round-trip
correctness, stats counters, capacity math, streamed attention, and the
GQA/FFN wrappers. All on CPU (pinned memory is skipped off-CUDA)."""

import torch
import torch.nn.functional as F

from forge.exp.cxlmem import (
    CxLLink,
    TieredFFN,
    TieredKV,
    attach_tiered_kv,
    detach_tiered_kv,
    kv_bytes_per_token,
    unwrap_ffn_tiered,
    wrap_ffn_tiered,
)


def _mk(seq_cap=4096, window=64, page=16, n_layers=3, burst_pages=2):
    tier = TieredKV(
        n_layers=n_layers,
        gpu_window_tokens=window,
        page_tokens=page,
        cxl_capacity_tokens=seq_cap,
        burst_pages=burst_pages,
        device="cpu",
    )
    return tier


def _gen_kv(B=1, H=2, T=8, hd=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, T, hd, generator=g), torch.randn(B, H, T, hd, generator=g))


# ── spill / materialize round-trip ────────────────────────────────────


def test_append_materialize_roundtrip_order():
    """Values recoverable in exact token order after spilling to 'CXL'."""
    tier = _mk(window=8, page=4, n_layers=2)
    T = 30
    for li in range(2):
        # token index encoded in k[..., 0] so ordering is verifiable
        k = torch.zeros(1, 2, T, 8)
        k[:, :, torch.arange(T), 0] = torch.arange(T).float()
        v = torch.randn(1, 2, T, 8)
        # feed in uneven chunks
        tier.append(li, k[:, :, :7], v[:, :, :7])
        tier.append(li, k[:, :, 7:19], v[:, :, 7:19])
        tier.append(li, k[:, :, 19:], v[:, :, 19:])
        assert tier.seq_len(li) == T
        kk, vv = tier.materialize(li)
        assert kk.shape == (1, 2, T, 8)
        assert torch.equal(kk, k)
        assert torch.equal(vv, v)


def test_bulk_append_gt_window():
    """Prefill-style bulk append (T >> window) spills the prefix."""
    tier = _mk(window=8, page=4, n_layers=1)
    k, v = _gen_kv(T=50)
    tier.append(0, k, v)
    st = tier._st[0]
    assert st.hot_len == 8 and st.cold_len == 42
    kk, vv = tier.materialize(0)
    assert torch.equal(kk, k) and torch.equal(vv, v)


def test_spill_page_granular():
    """Hot window evicts whole pages, keeping hot_len bounded by W."""
    tier = _mk(window=16, page=4, n_layers=1)
    for t in range(40):
        k, v = _gen_kv(T=1, seed=t)
        tier.append(0, k, v)
        st = tier._st[0]
        assert st.hot_len <= 16
        assert st.cold_len + st.hot_len == t + 1
        assert st.cold_len % 4 == 0  # cold grows page-wise
    kk, _vv = tier.materialize(0)
    assert kk.shape[2] == 40


def test_cxl_capacity_drop_oldest():
    """Over-capacity pool drops the OLDEST tokens and counts them."""
    tier = _mk(seq_cap=20, window=8, page=4, n_layers=1)
    T = 30
    k = torch.zeros(1, 2, T, 8)
    k[:, :, torch.arange(T), 0] = torch.arange(T).float()
    v = torch.zeros_like(k)
    tier.append(0, k, v)
    st = tier._st[0]
    # bulk: 22 tokens go cold but cap is 20 -> oldest 2 lost
    assert st.cold_len == 20 and st.hot_len == 8
    assert tier.link.stats["dropped_tokens"] == 2
    kk, _ = tier.materialize(0)
    # surviving sequence = tokens [2, 30)
    assert kk.shape[2] == 28
    assert torch.equal(kk[0, 0, :, 0], torch.arange(2, 30).float())


def test_reset_clears_state():
    tier = _mk(n_layers=1)
    k, v = _gen_kv(T=20)
    tier.append(0, k, v)
    tier.reset()
    assert tier.total_len == 0
    assert tier.persistent_gpu_bytes() == 0


# ── stats counters ────────────────────────────────────────────────────


def test_stats_counters():
    tier = _mk(window=8, page=4, n_layers=2)
    k, v = _gen_kv(T=8)
    k2, v2 = _gen_kv(T=8, seed=1)
    for li in range(2):
        tier.append(li, k, v)  # hot=8
        tier.append(li, k2, v2)  # ovf=8 -> spill 8 (one event)
        tier.append(li, k, v)  # spill 8 again
    st = tier.link.stats
    tok_bytes = 2 * 2 * 8 * 4  # (k,v) * heads * hd * fp32
    # per layer: 2 spills * 8 tok; 2 layers total
    assert st["d2h_bytes"] == 2 * 2 * 8 * tok_bytes
    assert st["spill_events"] == 4
    # materialize layer 0: 16 cold tokens = 4 pages
    _kk, _ = tier.materialize(0)
    assert st["materialize_calls"] == 1
    assert st["misses"] == 4
    assert st["h2d_bytes"] == 16 * tok_bytes
    # all-hot read = hit, no misses
    tier2 = _mk(window=64, n_layers=1)
    tier2.append(0, *_gen_kv(T=10))
    tier2.materialize(0)
    assert tier2.link.stats["hits"] == 1
    assert tier2.link.stats["misses"] == 0


def test_capacity_math():
    assert kv_bytes_per_token(24, 2, 64) == 24 * 2 * 2 * 64 * 2
    tier = _mk(window=32, page=8, n_layers=2)
    k, v = _gen_kv(T=80)
    for li in range(2):
        tier.append(li, k, v)
    # persistent GPU = 2 layers * 2 (k+v) * 2 heads * 32 tok * 8 hd * 4B
    assert tier.persistent_gpu_bytes() == 2 * 2 * 2 * 32 * 8 * 4
    assert tier.logical_kv_bytes() == 2 * 2 * 2 * 8 * 4 * 80
    # capacity ratio: logical/persistent
    assert tier.logical_kv_bytes() / tier.persistent_gpu_bytes() == 2.5


# ── streamed attention vs SDPA reference ──────────────────────────────


def _ref_attn(q, k, v, causal):
    _B, H, T, _hd = q.shape
    G = k.shape[1]
    kf = k.repeat_interleave(H // G, dim=1)
    vf = v.repeat_interleave(H // G, dim=1)
    if causal:
        mask = torch.full((T, T), float("-inf")).triu(1)
        return F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask)
    return F.scaled_dot_product_attention(q, kf, vf)


def test_attend_stream_matches_sdpa_decode():
    torch.manual_seed(0)
    B, H, G, hd = 1, 4, 2, 8
    tier = _mk(window=16, page=4, n_layers=1)
    k, v = _gen_kv(B, G, 37, hd)
    tier.append(0, k, v)  # 24 cold + 13 hot... wait W=16
    q = torch.randn(B, H, 1, hd)
    kn, vn = _gen_kv(B, G, 1, hd, seed=7)
    out = tier.attend(0, q, kn, vn)
    ref = _ref_attn(q, torch.cat([k, kn], 2), torch.cat([v, vn], 2), causal=False)
    assert torch.allclose(out, ref, atol=1e-5)
    assert tier.seq_len(0) == 38  # appended inside attend


def test_attend_stream_matches_sdpa_chunk():
    """T>1 continuation: causal within the new block, full view of past."""
    torch.manual_seed(0)
    B, H, G, hd = 1, 4, 2, 8
    tier = _mk(window=8, page=4, n_layers=1)
    k, v = _gen_kv(B, G, 20, hd)
    tier.append(0, k, v)
    T = 3
    q = torch.randn(B, H, T, hd)
    kn, vn = _gen_kv(B, G, T, hd, seed=3)
    out = tier.attend(0, q, kn, vn)
    k_all = torch.cat([k, kn], 2)
    v_all = torch.cat([v, vn], 2)
    kf = k_all.repeat_interleave(H // G, dim=1)
    vf = v_all.repeat_interleave(H // G, dim=1)
    # query row i (abs pos 20+i) may see all 20 past + new j<=i
    j = torch.arange(20 + T)
    i = torch.arange(T).unsqueeze(1)
    mask = torch.zeros(T, 20 + T).masked_fill(j.unsqueeze(0) > 20 + i, float("-inf"))
    ref = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attend_no_past():
    tier = _mk(n_layers=1)
    B, H, G, hd = 1, 4, 2, 8
    q = torch.randn(B, H, 1, hd)
    kn, vn = _gen_kv(B, G, 1, hd)
    out = tier.attend(0, q, kn, vn)
    # single key -> output equals v broadcast across query heads
    assert torch.allclose(out, vn.repeat_interleave(2, dim=1), atol=1e-6)


# ── GQA wrapper ───────────────────────────────────────────────────────


def _mk_gqa(d_model=32, n_heads=4, n_kv=2, hd=8):
    from forge.model.layers import GroupedQueryAttention

    attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv, max_seq_len=512, head_dim=hd)
    attn.eval()
    return attn


def _ref_gqa_decode(attn, x_seq, xs):
    """Reference: plain GQA forward with growing cat'd past."""
    with torch.no_grad():
        _out0, past = attn(x_seq, use_cache=True)
        outs = []
        for xt in xs:
            o, past = attn(xt, past_key_value=past, use_cache=True)
            outs.append(o)
    return outs


class _FakeBlocks:
    def __init__(self, attns):
        self.blocks = [type("B", (), {"attn": a})() for a in attns]


def test_attach_stream_matches_plain_decode():
    torch.manual_seed(0)
    attn = _mk_gqa()
    ref_attn = _mk_gqa()
    ref_attn.load_state_dict(attn.state_dict())
    x_seq = torch.randn(1, 20, 32)
    xs = [torch.randn(1, 1, 32) for _ in range(3)]
    outs_ref = _ref_gqa_decode(ref_attn, x_seq, xs)
    tier = _mk(window=8, page=4, n_layers=1)
    model = _FakeBlocks([attn])
    attach_tiered_kv(model, tier, mode="stream")
    with torch.no_grad():
        attn(x_seq, use_cache=True)  # prefill -> tier
        outs = []
        for xt in xs:
            o, present = attn(xt, use_cache=True)
            assert present is None
            outs.append(o)
    for a, b in zip(outs, outs_ref, strict=False):
        assert torch.allclose(a, b, atol=1e-4)
    detach_tiered_kv(model)
    _o2, p2 = attn(torch.randn(1, 1, 32), use_cache=True)
    assert p2 is not None  # orig path restored


def test_attach_fetch_mode_matches():
    torch.manual_seed(1)
    attn = _mk_gqa()
    ref_attn = _mk_gqa()
    ref_attn.load_state_dict(attn.state_dict())
    x_seq = torch.randn(1, 14, 32)
    xs = [torch.randn(1, 1, 32) for _ in range(3)]
    outs_ref = _ref_gqa_decode(ref_attn, x_seq, xs)
    tier = _mk(window=6, page=3, n_layers=1)
    model = _FakeBlocks([attn])
    attach_tiered_kv(model, tier, mode="fetch")
    with torch.no_grad():
        attn(x_seq, use_cache=True)
        outs = [attn(xt, use_cache=True)[0] for xt in xs]
    for a, b in zip(outs, outs_ref, strict=False):
        assert torch.allclose(a, b, atol=1e-4)


def test_tiered_prefill_chunked_continuation():
    """T>1 with existing tier state goes through materialize+append."""
    torch.manual_seed(2)
    attn = _mk_gqa()
    ref_attn = _mk_gqa()
    ref_attn.load_state_dict(attn.state_dict())
    x1, x2 = torch.randn(1, 10, 32), torch.randn(1, 5, 32)
    with torch.no_grad():
        _, past = ref_attn(x1, use_cache=True)
        ref2, _ = ref_attn(x2, past_key_value=past, use_cache=True)
    tier = _mk(window=6, page=4, n_layers=1)
    model = _FakeBlocks([attn])
    attach_tiered_kv(model, tier, mode="stream")
    with torch.no_grad():
        attn(x1, use_cache=True)
        out2, _ = attn(x2, use_cache=True)
    assert torch.allclose(out2, ref2, atol=1e-4)
    assert tier.seq_len(0) == 15


# ── weight tier ───────────────────────────────────────────────────────


def test_tiered_ffn_matches_and_hides_params():
    from forge.model.layers import SwiGLUFFN

    torch.manual_seed(0)
    ffn = SwiGLUFFN(32, 96)
    x = torch.randn(1, 5, 32)
    ref = ffn(x)
    link = CxLLink("cpu")
    tf = TieredFFN(ffn, link)
    out = tf(x)
    assert torch.allclose(out, ref, atol=1e-6)
    assert len(list(tf.parameters())) == 0  # weights not VRAM-resident
    assert link.stats["w_fetch_ops"] == 1
    assert tf.ffn_bytes == (32 * 96 * 3) * 4
    assert link.stats["w_h2d_bytes"] == tf.ffn_bytes


def test_wrap_unwrap_ffn_tiered():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM

    cfg = get_config("forgelm_tiny", layer_types=["attention"] * 2)
    cfg.vocab_size = 128
    cfg.device = "cpu"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    model.eval()
    ids = torch.randint(0, 128, (1, 8))
    ref = model(ids)[0]
    link = CxLLink("cpu")
    info = wrap_ffn_tiered(model, [0, 1], link)
    assert info["blocks"] == [0, 1]
    out = model(ids)[0]
    assert torch.allclose(ref, out, atol=1e-5)
    n = unwrap_ffn_tiered(model)
    assert n == 2
    out2 = model(ids)[0]
    assert torch.allclose(ref, out2, atol=1e-6)

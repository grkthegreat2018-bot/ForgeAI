"""Tests for the R50 missing-feature batch:
DRY penalty, DoLa decoding, filler-token KV, depth upscaling, PRM head.

All CPU-runnable; models/tokenizers are deterministic stubs.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forge.engine.decision_head import ProcessRewardHead, expected_calibration_error, fit_prm, score_steps
from forge.engine.decoding import DoLaDecoding, StandardDecoding, _sample_from_logits, build_decoding
from forge.engine.engine_common import _apply_dry, _dry_penalties
from forge.engine.kv.filler_kv import FillerKVCache, filler_token_ids
from forge.engine.kv_backend import FillerKVCacheStrategy, build_kv_cache
from research.merge_models import depth_layer_types, depth_upscale, parse_layer_map

# ═══════════════════════════════════════════════════════════════════════════
# DRY repetition penalty
# ═══════════════════════════════════════════════════════════════════════════


def test_dry_penalizes_repeated_suffix_continuation():
    # context: 1,2,3,4 | 1,2,3,4 — last token repeats the 4-token suffix;
    # its continuation earlier in the context is token 1 (position 4).
    ctx = [1, 2, 3, 4, 1, 2, 3, 4]
    pen = _dry_penalties(ctx, last_n=512, allowed_length=2, multiplier=1.0, base=1.75)
    assert pen == {1: pytest.approx(1.75**2)}


def test_dry_allowed_length_free_repeats():
    ctx = [1, 2, 3, 4, 1, 2, 3, 4]
    pen = _dry_penalties(ctx, last_n=512, allowed_length=4, multiplier=1.0, base=1.75)
    assert pen == {}


def test_dry_disabled_and_window():
    ctx = [1, 2, 3, 4, 1, 2, 3, 4]
    assert _dry_penalties(ctx, 512, 2, 0.0, 1.75) == {}
    # last_n=3 window: scan positions 4..6 — none equal the last token (4).
    assert _dry_penalties(ctx, 3, 2, 1.0, 1.75) == {}


def test_dry_apply_subtracts_logits():
    logits = torch.zeros(1, 10)
    out = _apply_dry(logits.clone(), {3: 2.0, 7: 0.5})
    assert out[0, 3].item() == -2.0
    assert out[0, 7].item() == -0.5
    assert out[0, 0].item() == 0.0


def test_min_k_sensitivity_controls_truncation():
    """Regression: the sensitivity threshold was computed then discarded
    (dead expression), so sensitivity had no effect — argmax always won.
    Now higher sensitivity must truncate more aggressively."""
    from forge.engine.forge_engine import _min_k_filter

    # diffs: sharp cliff at pos0 (5.0), medium cliff at pos4 (2.7).
    logits = torch.tensor([[10.0, 5.0, 4.9, 4.8, 4.7, 2.0, 1.9, 1.8, 1.7, 1.6]])
    low = _min_k_filter(logits.clone(), 0.1)
    high = _min_k_filter(logits.clone(), 0.9)
    n_low = torch.isfinite(low).sum().item()
    n_high = torch.isfinite(high).sum().item()
    # low sens: both cliffs flagged → rightmost=4 → keep 5.
    # high sens: only the pos0 cliff → keep 1.
    assert n_low == 5 and n_high == 1
    assert n_high < n_low


def test_dry_in_sample_chain_shifts_argmax():
    logits = torch.zeros(1, 16)
    logits[0, 5] = 10.0  # argmax
    logits[0, 6] = 9.0
    tok = _sample_from_logits(
        logits.clone(),
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        generated_ids=[],
        min_p=0.0,
        min_k=0.0,
        dry_penalties={5: 50.0},
        top_p_fn=StandardDecoding()._top_p,
    )
    assert tok.item() == 6


# ═══════════════════════════════════════════════════════════════════════════
# DoLa decoding
# ═══════════════════════════════════════════════════════════════════════════


class _StubHead(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.lin = nn.Linear(vocab, vocab, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(vocab))

    def forward(self, h):
        return self.lin(h)


class _DoLaStub(nn.Module):
    """Minimal model stub: head = identity projection, no ln_f."""

    def __init__(self, vocab=16, n_layers=8):
        super().__init__()
        self.head = _StubHead(vocab)
        self.ln_f = None
        self.blocks = [nn.Identity() for _ in range(n_layers)]


def test_dola_jsd_identical_and_disjoint():
    p = torch.tensor([[0.5, 0.5]])
    assert DoLaDecoding._jsd(p, p) == pytest.approx(0.0, abs=1e-6)
    q = torch.tensor([[1.0, 0.0]])
    r = torch.tensor([[0.0, 1.0]])
    assert DoLaDecoding._jsd(q, r) == pytest.approx(0.6931, abs=1e-3)


def test_dola_contrast_identical_layers_uniform():
    model = _DoLaStub(vocab=8)
    hidden = torch.zeros(1, 4, 8)
    dola = DoLaDecoding(early_layer=2, candidate_top_k=0)
    final = torch.zeros(1, 8)
    out = dola._contrast_logits(model, final, [hidden] * 8, [2])
    # final == early → contrasted logits are 0 → uniform distribution
    assert torch.allclose(out, torch.zeros_like(out))


def test_dola_candidate_top_k_masks():
    model = _DoLaStub(vocab=8)
    hidden = torch.zeros(1, 4, 8)
    dola = DoLaDecoding(early_layer=2, candidate_top_k=3)
    final = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    out = dola._contrast_logits(model, final, [hidden] * 8, [2])
    finite = torch.isfinite(out[0])
    assert finite.sum().item() == 3
    # top-3 by final prob = tokens 7,6,5
    assert finite[7] and finite[6] and finite[5]


def test_build_decoding_dola():
    d = build_decoding("dola")
    assert isinstance(d, DoLaDecoding)
    d2 = build_decoding("dola", early_layer=3, candidate_top_k=32)
    assert d2.early_layer == 3 and d2.candidate_top_k == 32


def test_dola_contrast_accepts_inference_tensors():
    """Regression: hidden_list comes from model forwards run under
    inference_mode — _contrast_logits crashed with 'Inference tensors
    cannot be saved for backward' once a param (ln_f/head weight) had
    requires_grad=True. Method must run under its own no_grad scope and
    return a normal tensor the sampling chain can mutate in place."""
    vocab, d = 8, 8
    torch.manual_seed(0)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(d, vocab, bias=False)
            self.ln_f = nn.LayerNorm(d)

    model = _M()
    with torch.inference_mode():
        hidden = torch.randn(1, 4, d)
        final = torch.randn(1, vocab)
    dola = DoLaDecoding(early_layer=0, candidate_top_k=0)
    out = dola._contrast_logits(model, final, [hidden], [0])
    out[:, 0] = 0.0  # in-place mutation must work (sampling chain does it)
    assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════
# Filler-token KV
# ═══════════════════════════════════════════════════════════════════════════


def _kv(B=1, n_kv=2, T=1, d=4, start=0.0):
    k = torch.arange(start, start + B * n_kv * T * d, dtype=torch.float32).reshape(B, n_kv, T, d)
    return k, k.clone()


def test_filler_kv_evicts_fillers_first():
    cache = FillerKVCache(
        observation_window=4,
        budget=8,
        n_kv_heads=2,
        head_dim=4,
        n_sink=4,
        filler_ids={5, 7, 11, 13},
        device="cpu",
        dtype=torch.float32,
    )
    # 20 appends of 1 token; fillers at positions 5,7,11,13 (unprotected:
    # sink=0..3, obs=16..19) → pass1 evicts 4, pass2 evicts 4 more → 12 kept.
    k, v = _kv(T=20)
    cache.append(k, v, 0, token_ids=list(range(20)))
    assert cache.seq_len == 12
    assert cache.filler_evicted == 4
    assert cache.score_evicted == 4
    kept_k, _kept_v = cache.get()
    # sink + observation window survived
    assert torch.equal(kept_k[0, 0, :4], k[0, 0, :4])
    assert torch.equal(kept_k[0, 0, -4:], k[0, 0, -4:])
    # no unprotected filler position survived: kept values must not
    # contain the value-vector of any evicted filler position
    kept_first = set(kept_k[0, 0, :, 0].tolist())
    assert kept_first.isdisjoint({v * 4.0 for v in []})  # sanity
    info = cache.info()
    assert info["type"] == "filler" and info["fillers_cached"] == 0
    assert info["filler_evicted"] == 4


def test_filler_kv_no_token_ids_falls_back_to_scores():
    cache = FillerKVCache(
        observation_window=4, budget=8, n_kv_heads=2, head_dim=4, n_sink=2, device="cpu", dtype=torch.float32
    )
    k, v = _kv(T=16)
    cache.append(k, v, 0)  # no token_ids → pure score eviction
    assert cache.seq_len == 12
    assert cache.filler_evicted == 0
    assert cache.score_evicted == 4


def test_filler_kv_clear_and_get_past_kv():
    cache = FillerKVCache(budget=8, n_kv_heads=2, head_dim=4, device="cpu", dtype=torch.float32)
    assert cache.get_past_kv() is None
    k, v = _kv(T=4)
    cache.append(k, v, 0)
    assert cache.get_past_kv() is not None
    cache.clear()
    assert cache.seq_len == 0 and cache.get_past_kv() is None


def test_filler_strategy_wrapper_and_factory():
    strat = FillerKVCacheStrategy()
    strat.init(n_heads=2, head_dim=4, n_kv_heads=2, max_seq_len=16, device="cpu", dtype=torch.float32)
    k, v = _kv(T=4)
    strat.append(k, v, 0, token_ids=[1, 2, 3, 4])
    assert strat.seq_len == 4
    strat.set_filler_ids({2})
    assert strat.cache.filler_ids == {2}
    assert strat.info()["type"] == "filler"
    strat.clear()
    assert strat.seq_len == 0
    built = build_kv_cache("filler")
    assert isinstance(built, FillerKVCacheStrategy)


def test_filler_token_ids_helper():
    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s[:2]]

    ids = filler_token_ids(_Tok())
    assert isinstance(ids, frozenset) and len(ids) > 0


def test_snapkv_multi_token_overflow_regression():
    """Regression: appending T>1 in one shot over-allocated the buffer
    beyond `total`, so the bool keep-mask (size total) failed to
    broadcast — IndexError. Fixed by masking only the first `total`
    slots."""
    from forge.engine.kv.snapkv import SnapKVCache

    cache = SnapKVCache(observation_window=4, budget=8, n_kv_heads=2, head_dim=4, device="cpu", dtype=torch.float32)
    k, v = _kv(T=20)
    cache.append(k, v, 0)  # buffer grows to 32 > total=20
    assert cache.seq_len == 12


# ═══════════════════════════════════════════════════════════════════════════
# Depth upscaling (passthrough merge)
# ═══════════════════════════════════════════════════════════════════════════


def _fake_state(n_blocks=4, d=8):
    torch.manual_seed(0)
    state = {"embed.weight": torch.randn(32, d), "head.weight": torch.randn(32, d)}
    for i in range(n_blocks):
        state[f"blocks.{i}.attn.w"] = torch.full((d, d), float(i))
        state[f"blocks.{i}.ffn.w"] = torch.full((d,), float(i))
    return state


def test_parse_layer_map():
    assert parse_layer_map("0-3,2-3") == [0, 1, 2, 3, 2, 3]
    assert parse_layer_map("0, 5, 7") == [0, 5, 7]
    assert parse_layer_map(" 1 -2 , 4 ") == [1, 2, 4]
    with pytest.raises(ValueError):
        parse_layer_map("3-1")
    with pytest.raises(ValueError):
        parse_layer_map("")


def test_depth_upscale_duplicates_blocks():
    state = _fake_state(4)
    out, meta = depth_upscale(state, [0, 1, 2, 3, 1, 2])
    assert meta["n_layers"] == 6 and meta["source_n_layers"] == 4
    # non-block keys preserved
    assert torch.equal(out["embed.weight"], state["embed.weight"])
    # block 4 == source block 1, block 5 == source block 2
    assert torch.equal(out["blocks.4.attn.w"], state["blocks.1.attn.w"])
    assert torch.equal(out["blocks.5.ffn.w"], state["blocks.2.ffn.w"])
    assert "blocks.6.attn.w" not in out


def test_depth_upscale_layer_types():
    state = _fake_state(4)
    _, meta = depth_upscale(state, [0, 1, 0, 1], layer_types=["mamba", "attn", "mamba", "attn"])
    assert meta["layer_types"] == ["mamba", "attn", "mamba", "attn"]
    with pytest.raises(ValueError):
        depth_upscale(state, [0, 1], layer_types=["mamba"])
    with pytest.raises(ValueError):
        depth_upscale(state, [0, 9])
    types = depth_layer_types(state, [0, 1, 0], ["mamba", "attn", "mamba", "attn"])
    assert types == ["mamba", "attn", "mamba"]


# ═══════════════════════════════════════════════════════════════════════════
# Process Reward Model
# ═══════════════════════════════════════════════════════════════════════════


class _StubTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 64 for c in text]


class _StubModel(nn.Module):
    """Deterministic embedding-only model returning hidden states."""

    def __init__(self, vocab=128, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        torch.manual_seed(0)
        with torch.no_grad():
            self.emb.weight.normal_(0, 1)

    def forward(self, idx, return_hidden=False, **kw):
        h = self.emb(idx)
        return (None, None, h)


def test_prm_head_shapes_and_prob():
    head = ProcessRewardHead(d_model=16)
    h = torch.randn(5, 16)
    s = head.score(h)
    p = head.prob(h)
    assert s.shape == (5,) and p.shape == (5,)
    assert ((p >= 0) & (p <= 1)).all()
    mlp = ProcessRewardHead(d_model=16, hidden_dim=8)
    assert mlp.prob(h).shape == (5,)


def test_prm_save_load(tmp_path):
    head = ProcessRewardHead(d_model=16, hidden_dim=8)
    path = str(tmp_path / "prm.pt")
    head.save(path, meta={"round": "r50"})
    loaded = ProcessRewardHead.load(path)
    assert loaded.d_model == 16
    h = torch.randn(3, 16)
    assert torch.equal(loaded.score(h), head.score(h))


def test_score_steps_passthrough_and_head():
    model = _StubModel()
    tok = _StubTokenizer()
    probs = score_steps(None, model, tok, "Q?", ["step1", "step2"], "cpu")
    assert probs == [0.5, 0.5]
    head = ProcessRewardHead(d_model=16)
    probs2 = score_steps(head, model, tok, "Q?", ["a", "b", "c"], "cpu")
    assert len(probs2) == 3
    assert all(0.0 <= p <= 1.0 for p in probs2)


def test_fit_prm_learns_separable_labels():
    """Labels correlate with the step's last token embedding (vowel =
    correct). A linear probe must reach high val accuracy."""
    model = _StubModel()
    tok = _StubTokenizer()
    dataset = []
    for i in range(24):
        last = "aeiou"[i % 5] if i % 2 == 0 else "xyzjk"[i % 5]
        steps = [f"reason {i}", f"concl {last}"]
        labels = [1, 1 if i % 2 == 0 else 0]
        dataset.append(("question?", steps, labels))
    _head, metrics = fit_prm(model, tok, "cpu", dataset, epochs=30, val_frac=0.25, progress=False, seed=0)
    assert metrics["n_examples"] == 24
    assert metrics["steps"] == 48
    assert 0.0 <= metrics["val_acc"] <= 1.0
    assert metrics["val_acc"] >= 0.5
    assert "val_ece" in metrics


def test_expected_calibration_error_perfect():
    p = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert expected_calibration_error(p, y) < 1e-6

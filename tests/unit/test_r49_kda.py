"""Tests for R49-2: KDAKey + KDALayer — Kimi Delta Attention side-path.

Covers the port-first contract:
1. Key properties (name, KeyClass.BI)
2. forward(): adds deterministic zero-init kda.* params, originals untouched
3. reverse(): strips kda.*, round-trip is identity while gate ~ 0
4. convert_model_state(): whole-checkpoint block-prefix port
5. KDALayer: gate=0 -> exact zeros; gate>0 -> correct shapes, finite output
6. Recurrence math: hand-computed delta-rule update
7. Prefill/decode parity + prefix-restore consistency (cache plumbing)
8. Block-level bit-exactness: ModularBlock(use_kda=True) == baseline at init
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forge.config import ModelConfig
from forge.keys.attention.kda_key import KDA_PARAM_SUFFIXES, KDAKey, KDALayer
from forge.keys.misc.base import KeyClass
from forge.model.layers import ModularBlock

D_MODEL = 64
N_HEADS = 4
HEAD_DIM = 16


def _make_layer_weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "attn.q_proj.weight": torch.randn(D_MODEL, D_MODEL),
        "attn.out_proj.weight": torch.randn(D_MODEL, D_MODEL),
        "ln1.weight": torch.ones(D_MODEL),
    }


def _make_kda_layer(**kw) -> KDALayer:
    torch.manual_seed(0)
    return KDALayer(D_MODEL, N_HEADS, head_dim=HEAD_DIM, **kw)


# ── Key contract ──────────────────────────────────────────────────────────────


def test_key_properties():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    assert key.name == "kda"
    assert key.key_class() == KeyClass.BI
    assert "Delta" in key.description


def test_forward_adds_kda_params_and_preserves_originals():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    data = _make_layer_weights()
    res = key.forward(data)
    assert res.success, res.error
    assert res.metadata["lossless"] and res.metadata["gate"] == 0.0
    for k, v in data.items():
        assert torch.equal(res.weights[k], v), f"original {k} modified"
    for suffix in KDA_PARAM_SUFFIXES:
        assert f"kda.{suffix}" in res.weights, f"missing kda.{suffix}"
    assert torch.equal(res.weights["kda.gate"], torch.zeros(1))


def test_forward_deterministic():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    a = key.forward(_make_layer_weights()).weights
    b = key.forward(_make_layer_weights()).weights
    for k in a:
        assert torch.equal(a[k], b[k]), f"nondeterministic init for {k}"


def test_forward_rejects_double_port():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    ported = key.forward(_make_layer_weights()).weights
    res = key.forward(ported)
    assert not res.success and "double-port" in res.error


def test_reverse_roundtrip_identity():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    data = _make_layer_weights()
    ported = key.forward(data).weights
    back = key.reverse(ported)
    assert back.success and back.metadata["lossless"]
    assert set(back.data.keys()) == set(data.keys())
    for k, v in data.items():
        assert torch.equal(back.data[k], v), f"round-trip mismatch on {k}"


def test_reverse_reports_nonzero_gate():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    ported = key.forward(_make_layer_weights()).weights
    ported["kda.gate"].fill_(0.5)  # trained gate -> reverse is lossy
    back = key.reverse(ported)
    assert back.success
    assert not back.metadata["lossless"]
    assert back.metadata["max_gate"] == pytest.approx(0.5)


def test_reverse_no_kda_fails():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    res = key.reverse(_make_layer_weights())
    assert not res.success


def test_convert_model_state():
    key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    state = {}
    for i in range(3):
        state[f"blocks.{i}.attn.q_proj.weight"] = torch.randn(D_MODEL, D_MODEL)
    state["embed.weight"] = torch.randn(128, D_MODEL)
    res = key.convert_model_state(state)
    assert res.success and res.metadata["blocks"] == [0, 1, 2]
    for i in range(3):
        for suffix in KDA_PARAM_SUFFIXES:
            assert f"blocks.{i}._kda.{suffix}" in res.weights
    for k, v in state.items():
        assert torch.equal(res.weights[k], v)
    # subset + double-port guard
    res2 = key.convert_model_state(state, indices=[1])
    assert "blocks.1._kda.gate" in res2.weights
    assert "blocks.0._kda.gate" not in res2.weights
    res3 = key.convert_model_state(res.weights)
    assert not res3.success


# ── KDALayer behaviour ────────────────────────────────────────────────────────


def test_layer_gate_zero_exact_zeros():
    layer = _make_kda_layer()
    x = torch.randn(2, 8, D_MODEL)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.equal(out, torch.zeros_like(out))


def test_layer_open_gate_shape_and_finite():
    layer = _make_kda_layer()
    layer.gate.data.fill_(1.0)
    x = torch.randn(2, 8, D_MODEL)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


def test_scan_math_hand_computed():
    """Verify the delta-rule update against a manual two-step computation."""
    layer = _make_kda_layer()
    # shapes: (B=1, T=2, H=1, d=2) for q/k/v/alpha; (1,2,1) for beta
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 3.0]], [[5.0, 7.0]]]])
    alpha = torch.tensor([[[[1.0, 1.0]], [[0.5, 0.5]]]])
    beta = torch.tensor([[[1.0], [1.0]]])
    S = torch.zeros(1, 1, 2, 2)
    o0, S0 = layer._scan(q[:, :1], k[:, :1], v[:, :1], alpha[:, :1], beta[:, :1], S)
    # t=0: S0 = k0 (v0)^T ; o0 = S0^T q0 = v0 (q0=k0)
    assert torch.allclose(S0[0, 0], torch.tensor([[2.0, 3.0], [0.0, 0.0]]))
    assert torch.allclose(o0[0, 0, 0], torch.tensor([2.0, 3.0]))
    o1, S1 = layer._scan(q[:, 1:], k[:, 1:], v[:, 1:], alpha[:, 1:], beta[:, 1:], S0)
    # t=1: Sa = diag(0.5) S0 = [[1,1.5],[0,0]]; pred = Sa^T k1 = col1 = (0,0)
    #      S1 = Sa + k1 (v1 - 0)^T = [[1,1.5],[5,7]]; o1 = S1^T q1 = row2 = (5,7)
    assert torch.allclose(S1[0, 0], torch.tensor([[1.0, 1.5], [5.0, 7.0]]))
    assert torch.allclose(o1[0, 0, 0], torch.tensor([5.0, 7.0]))


def test_prefill_decode_parity():
    """Full T=9 forward == T=8 prefill + T=1 decode at the last position."""
    layer = _make_kda_layer()
    layer.gate.data.fill_(1.0)
    layer.eval()
    x = torch.randn(1, 9, D_MODEL)
    with torch.no_grad():
        full = layer(x, use_cache=True)[:, -1]
        layer.reset_state()
        layer(x[:, :8], use_cache=True)
        step = layer(x[:, 8:], use_cache=True)[:, -1]
    assert torch.allclose(full, step, atol=1e-5, rtol=1e-5)


def test_reset_state_clears():
    layer = _make_kda_layer()
    layer.gate.data.fill_(1.0)
    x = torch.randn(1, 4, D_MODEL)
    layer(x, use_cache=True)
    assert layer._state is not None and layer.conv_q.state is not None
    layer.reset_state()
    assert layer._state is None
    assert layer.conv_q.state is None and layer.conv_k.state is None and layer.conv_v.state is None


def test_prefix_restore_consistency():
    """Snapshot at position P + set_state_prefix on a clone == uninterrupted
    prefill — the property the prefix-cache restore path relies on."""
    layer = _make_kda_layer()
    layer.gate.data.fill_(1.0)
    layer.eval()
    x = torch.randn(1, 10, D_MODEL)
    with torch.no_grad():
        full = layer(x, use_cache=True)
        layer.reset_state()
        layer(x[:, :6], use_cache=True)
        snap = layer.snapshot_state()
        clone = _make_kda_layer()
        clone.load_state_dict(layer.state_dict())
        clone.set_state_prefix(snap)
        resumed = clone(x[:, 6:], use_cache=True)
    assert torch.allclose(full[:, 6:], resumed, atol=1e-5, rtol=1e-5)


def test_state_reset_flag():
    layer = _make_kda_layer()
    layer.gate.data.fill_(1.0)
    x = torch.randn(1, 4, D_MODEL)
    layer(x, use_cache=True)
    layer._state_reset = True  # what llm.py sets on a new sequence
    layer(x, use_cache=True)
    assert layer._state_reset is False


# ── Block-level bit-exactness ─────────────────────────────────────────────────


def _block_config(use_kda: bool) -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=D_MODEL,
        n_layers=1,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        max_seq_len=64,
        layer_types=["attention"],
        use_kda=use_kda,
    )


def test_block_bit_exact_at_init():
    torch.manual_seed(0)
    blk_kda = ModularBlock(_block_config(True), layer_idx=0)
    blk_ref = ModularBlock(_block_config(False), layer_idx=0)
    ref_sd = blk_ref.state_dict()
    blk_kda.load_state_dict({k: v for k, v in ref_sd.items() if k in blk_kda.state_dict()}, strict=False)
    blk_kda.eval()
    blk_ref.eval()
    x = torch.randn(1, 8, D_MODEL)
    with torch.no_grad():
        out_kda, _ = blk_kda(x)
        out_ref, _ = blk_ref(x)
    assert torch.equal(out_kda, out_ref)


def test_block_kda_opens_with_gate():
    torch.manual_seed(0)
    blk = ModularBlock(_block_config(True), layer_idx=0)
    blk.eval()
    blk._kda.gate.data.fill_(1.0)
    blk._kda_gate_zero = None  # invalidate cached gate check
    x = torch.randn(1, 8, D_MODEL)
    blk2 = ModularBlock(_block_config(False), layer_idx=0)
    blk2.load_state_dict({k: v for k, v in blk.state_dict().items() if k in blk2.state_dict()}, strict=False)
    blk2.eval()
    with torch.no_grad():
        out_kda, _ = blk(x)
        out_ref, _ = blk2(x)
    assert not torch.equal(out_kda, out_ref)
    assert torch.isfinite(out_kda).all()

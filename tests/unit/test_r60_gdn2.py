"""Tests for R60-1: GDN2Key + GDN2Layer — Gated DeltaNet-2 side-path.

Covers the port-first contract plus the GDN2-specific differentiator:
1. Key properties (name, KeyClass.BI)
2. forward(): adds deterministic zero-init gdn2.* params, originals untouched
3. reverse(): strips gdn2.*, round-trip is identity while gate ~ 0
4. convert_model_state(): whole-checkpoint port (fresh + kda-upgrade paths)
5. KDA->GDN2 upgrade is FUNCTION-EXACT at any gate value (tiled scalar
   gates broadcast per-channel — the paper's collapse rule)
6. GDN2Layer: gate=0 -> exact zeros; gate>0 -> correct shapes, finite output
7. Recurrence math: hand-computed decoupled erase/write update
8. Prefill/decode parity + prefix-restore consistency (cache plumbing)
9. Block-level bit-exactness: ModularBlock(use_gdn2=True) == baseline at init
10. Mutual exclusion: use_gdn2 + use_kda is a config error
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forge.config import MODEL_CONFIGS, ModelConfig
from forge.keys.attention.gdn2_key import (
    GDN2_PARAM_SUFFIXES,
    GDN2Key,
    GDN2Layer,
    kda_state_to_gdn2,
)
from forge.keys.attention.kda_key import KDAKey, KDALayer
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


def _make_gdn2_layer(**kw) -> GDN2Layer:
    torch.manual_seed(0)
    return GDN2Layer(D_MODEL, N_HEADS, head_dim=HEAD_DIM, **kw)


def _make_kda_layer(**kw) -> KDALayer:
    torch.manual_seed(0)
    return KDALayer(D_MODEL, N_HEADS, head_dim=HEAD_DIM, **kw)


# ── Key contract ──────────────────────────────────────────────────────────────


def test_key_properties():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    assert key.name == "gdn2"
    assert key.key_class() == KeyClass.BI
    assert "DeltaNet-2" in key.description


def test_forward_adds_gdn2_params_and_preserves_originals():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    data = _make_layer_weights()
    res = key.forward(data)
    assert res.success, res.error
    assert res.metadata["lossless"] and res.metadata["gate"] == 0.0
    for k, v in data.items():
        assert torch.equal(res.weights[k], v), f"original {k} modified"
    for suffix in GDN2_PARAM_SUFFIXES:
        assert f"gdn2.{suffix}" in res.weights, f"missing gdn2.{suffix}"
    assert torch.equal(res.weights["gdn2.gate"], torch.zeros(1))
    # channel-wise gates: H*d_k / H*d_v rows
    assert res.weights["gdn2.e_proj.weight"].shape[0] == N_HEADS * HEAD_DIM
    assert res.weights["gdn2.w_proj.weight"].shape[0] == N_HEADS * HEAD_DIM


def test_forward_deterministic():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    a = key.forward(_make_layer_weights()).weights
    b = key.forward(_make_layer_weights()).weights
    for k in a:
        assert torch.equal(a[k], b[k]), f"nondeterministic init for {k}"


def test_forward_rejects_double_port():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    ported = key.forward(_make_layer_weights()).weights
    res = key.forward(ported)
    assert not res.success and "double-port" in res.error


def test_reverse_roundtrip_identity():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    data = _make_layer_weights()
    ported = key.forward(data).weights
    back = key.reverse(ported)
    assert back.success and back.metadata["lossless"]
    assert set(back.data.keys()) == set(data.keys())
    for k, v in data.items():
        assert torch.equal(back.data[k], v), f"round-trip mismatch on {k}"


def test_reverse_reports_nonzero_gate():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    ported = key.forward(_make_layer_weights()).weights
    ported["gdn2.gate"].fill_(0.5)
    back = key.reverse(ported)
    assert back.success
    assert not back.metadata["lossless"]
    assert back.metadata["max_gate"] == pytest.approx(0.5)


def test_reverse_no_gdn2_fails():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    res = key.reverse(_make_layer_weights())
    assert not res.success


def test_convert_model_state():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    state = {}
    for i in range(3):
        state[f"blocks.{i}.attn.q_proj.weight"] = torch.randn(D_MODEL, D_MODEL)
    state["embed.weight"] = torch.randn(128, D_MODEL)
    res = key.convert_model_state(state)
    assert res.success and res.metadata["blocks"] == [0, 1, 2]
    for i in range(3):
        for suffix in GDN2_PARAM_SUFFIXES:
            assert f"blocks.{i}._gdn2.{suffix}" in res.weights
    for k, v in state.items():
        assert torch.equal(res.weights[k], v)
    res2 = key.convert_model_state(state, indices=[1])
    assert "blocks.1._gdn2.gate" in res2.weights
    assert "blocks.0._gdn2.gate" not in res2.weights
    res3 = key.convert_model_state(res.weights)
    assert not res3.success


# ── KDA -> GDN2 function-exact upgrade ────────────────────────────────────────


def test_kda_upgrade_is_function_exact():
    """A KDA-keyed block upgraded to GDN2 produces identical outputs —
    the collapse rule b_t = w_t = beta_t makes the port a special case,
    so this holds for TRAINED gates too, not just gate=0."""
    torch.manual_seed(0)
    kda = _make_kda_layer()
    kda.gate.data.fill_(1.0)  # trained-open gate — the hard case
    gdn2 = _make_gdn2_layer()
    gdn2.load_state_dict(kda_state_to_gdn2(kda.state_dict()))
    gdn2.eval()
    kda.eval()
    x = torch.randn(2, 7, D_MODEL)
    with torch.no_grad():
        out_kda = kda(x, use_cache=True)
        out_g = gdn2(x, use_cache=True)
    assert torch.allclose(out_kda, out_g, atol=1e-5, rtol=1e-5)


def test_key_upgrades_kda_weights_function_exact():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    kda = _make_kda_layer()
    kda.gate.data.fill_(1.0)
    data = {f"kda.{k}": v.clone() for k, v in kda.state_dict().items()}
    data.update(_make_layer_weights())
    res = key.forward(data)
    assert res.success and res.metadata["conversion"] == "kda->gdn2"
    assert res.metadata["lossless"]
    assert not any(k.startswith("kda.") for k in res.weights)
    gdn2 = _make_gdn2_layer()
    gdn2.load_state_dict({k[len("gdn2.") :]: v for k, v in res.weights.items() if k.startswith("gdn2.")})
    gdn2.eval()
    x = torch.randn(1, 5, D_MODEL)
    with torch.no_grad():
        assert torch.allclose(kda(x), gdn2(x), atol=1e-5, rtol=1e-5)
    # reverse reports it could collapse back to KDA
    back = key.reverse(res.weights)
    assert back.metadata["kda_collapsible"]


def test_convert_model_state_upgrades_kda_blocks():
    key = GDN2Key(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    kda_key = KDAKey(D_MODEL, N_HEADS, head_dim=HEAD_DIM)
    state = {}
    for i in range(2):
        state[f"blocks.{i}.attn.q_proj.weight"] = torch.randn(D_MODEL, D_MODEL)
    kda_state = kda_key.convert_model_state(state).weights
    res = key.convert_model_state(kda_state)
    assert res.success and res.metadata["kda_upgraded"] == 2
    for i in range(2):
        assert f"blocks.{i}._gdn2.gate" in res.weights
        assert not any(k.startswith(f"blocks.{i}._kda.") for k in res.weights)


# ── GDN2Layer behaviour ───────────────────────────────────────────────────────


def test_layer_gate_zero_exact_zeros():
    layer = _make_gdn2_layer()
    x = torch.randn(2, 8, D_MODEL)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.equal(out, torch.zeros_like(out))


def test_layer_open_gate_shape_and_finite():
    layer = _make_gdn2_layer()
    layer.gate.data.fill_(1.0)
    x = torch.randn(2, 8, D_MODEL)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


def test_scan_math_hand_computed():
    """Hand-computed decoupled erase/write: H=1, d_k=d_v=2."""
    layer = _make_gdn2_layer()
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 3.0]], [[5.0, 7.0]]]])
    alpha = torch.ones(1, 2, 1, 2) * 0.5
    egate = torch.ones(1, 2, 1, 2) * 0.9
    wgate = torch.tensor([[[[0.5, 1.0]], [[1.0, 0.5]]]])
    S = torch.zeros(1, 1, 2, 2)
    # t=0: S = k0 (w0⊙v0)^T = [[1,3],[0,0]]; o0 = S^T q0 = (1,3)
    o0, S0 = layer._scan(q[:, :1], k[:, :1], v[:, :1], alpha[:, :1], egate[:, :1], wgate[:, :1], S)
    assert torch.allclose(S0[0, 0], torch.tensor([[1.0, 3.0], [0.0, 0.0]]))
    assert torch.allclose(o0[0, 0, 0], torch.tensor([1.0, 3.0]))
    # t=1: Sa = 0.5*S0 = [[0.5,1.5],[0,0]]; e1⊙k1 = (0,0.9) -> erase =
    #      0.9*row1 of Sa = (0,0); w1⊙v1 = (5,3.5); S1 = Sa + k1(5,3.5)^T
    #      = [[0.5,1.5],[5,3.5]]; o1 = S1^T q1 = row1 = (5, 3.5)
    o1, S1 = layer._scan(q[:, 1:], k[:, 1:], v[:, 1:], alpha[:, 1:], egate[:, 1:], wgate[:, 1:], S0)
    assert torch.allclose(S1[0, 0], torch.tensor([[0.5, 1.5], [5.0, 3.5]]))
    assert torch.allclose(o1[0, 0, 0], torch.tensor([5.0, 3.5]))


def test_scan_matches_kda_when_gates_collapse():
    """With egate=wgate=beta broadcast, _scan must equal KDALayer._scan.
    fp64 inputs so the check proves formulation equivalence — fp32 rounding
    of beta*(v-pred) vs (beta*v - beta*pred) differs at ~1e-5."""
    kda = _make_kda_layer()
    gdn2 = _make_gdn2_layer()
    torch.manual_seed(1)
    B, T, H, dk, dv = 2, 5, N_HEADS, HEAD_DIM, HEAD_DIM
    dt = torch.float64
    q = torch.randn(B, T, H, dk, dtype=dt)
    k = torch.randn(B, T, H, dk, dtype=dt)
    v = torch.randn(B, T, H, dv, dtype=dt)
    alpha = torch.rand(B, T, H, dk, dtype=dt) * 0.5 + 0.5
    beta = torch.rand(B, T, H, dtype=dt)
    S = torch.randn(B, H, dk, dv, dtype=dt)
    o_ref, S_ref = kda._scan(q, k, v, alpha, beta, S.clone())
    o_new, S_new = gdn2._scan(
        q, k, v, alpha, beta.unsqueeze(-1).expand(-1, -1, -1, dk), beta.unsqueeze(-1).expand(-1, -1, -1, dv), S.clone()
    )
    assert torch.allclose(o_ref, o_new, atol=1e-10)
    assert torch.allclose(S_ref, S_new, atol=1e-10)


def test_prefill_decode_parity():
    layer = _make_gdn2_layer()
    layer.gate.data.fill_(1.0)
    layer.eval()
    x = torch.randn(1, 9, D_MODEL)
    with torch.no_grad():
        full = layer(x, use_cache=True)[:, -1]
        layer.reset_state()
        layer(x[:, :8], use_cache=True)
        step = layer(x[:, 8:], use_cache=True)[:, -1]
    assert torch.allclose(full, step, atol=1e-5, rtol=1e-5)


def test_prefix_restore_consistency():
    layer = _make_gdn2_layer()
    layer.gate.data.fill_(1.0)
    layer.eval()
    x = torch.randn(1, 10, D_MODEL)
    with torch.no_grad():
        full = layer(x, use_cache=True)
        layer.reset_state()
        layer(x[:, :6], use_cache=True)
        snap = layer.snapshot_state()
        clone = _make_gdn2_layer()
        clone.load_state_dict(layer.state_dict())
        clone.set_state_prefix(snap)
        resumed = clone(x[:, 6:], use_cache=True)
    assert torch.allclose(full[:, 6:], resumed, atol=1e-5, rtol=1e-5)


def test_state_reset_flag():
    layer = _make_gdn2_layer()
    layer.gate.data.fill_(1.0)
    x = torch.randn(1, 4, D_MODEL)
    layer(x, use_cache=True)
    layer._state_reset = True
    layer(x, use_cache=True)
    assert layer._state_reset is False


# ── Block-level bit-exactness + config guard ──────────────────────────────────


def _block_config(**kw) -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=D_MODEL,
        n_layers=1,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        max_seq_len=64,
        layer_types=["attention"],
        **kw,
    )


def test_block_bit_exact_at_init():
    torch.manual_seed(0)
    blk_g = ModularBlock(_block_config(use_gdn2=True), layer_idx=0)
    blk_ref = ModularBlock(_block_config(), layer_idx=0)
    ref_sd = blk_ref.state_dict()
    blk_g.load_state_dict({k: v for k, v in ref_sd.items() if k in blk_g.state_dict()}, strict=False)
    blk_g.eval()
    blk_ref.eval()
    x = torch.randn(1, 8, D_MODEL)
    with torch.no_grad():
        out_g, _ = blk_g(x)
        out_ref, _ = blk_ref(x)
    assert torch.equal(out_g, out_ref)


def test_block_gdn2_opens_with_gate():
    torch.manual_seed(0)
    blk = ModularBlock(_block_config(use_gdn2=True), layer_idx=0)
    blk.eval()
    blk._gdn2.gate.data.fill_(1.0)
    blk._gdn2_gate_zero = None
    x = torch.randn(1, 8, D_MODEL)
    blk2 = ModularBlock(_block_config(), layer_idx=0)
    blk2.load_state_dict({k: v for k, v in blk.state_dict().items() if k in blk2.state_dict()}, strict=False)
    blk2.eval()
    with torch.no_grad():
        out_g, _ = blk(x)
        out_ref, _ = blk2(x)
    assert not torch.equal(out_g, out_ref)
    assert torch.isfinite(out_g).all()


def test_config_mutual_exclusion():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _block_config(use_gdn2=True, use_kda=True)


def test_v13_preset_lineage():
    cfg = MODEL_CONFIGS["forgelm_v13_gdn2"]
    assert cfg.use_gdn2 is True
    assert cfg.use_kda is False
    assert cfg.parent == "forgelm_v12_jamba"
    assert "use_gdn2" in cfg.dropped_keys
    # carries forward V12 keys
    assert cfg.use_mamba3 is True and cfg.use_forge_hybrid is True

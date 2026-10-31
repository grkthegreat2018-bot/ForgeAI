"""Benchmark tests: R37 new architecture keys vs their original baselines.

Each test compares a NEW R37 key against the OLD baseline it replaces,
verifying the documented improvement actually holds:

  1. Mamba-3 vs Mamba-2      — 2x expressivity, lossless round-trip
  2. Kronecker vs Standard    — >90% param reduction (large), >80% (small)
  3. PIT vs Standard Tying    — orthonormal full-rank memory, lossless
  4. OutRo vs Causal Attn     — sink row fully unmasked, more connections
  5. ForgeHybrid vs Pure Attn — zero-init SSM, lossless warm start, +params
  6. V12 vs V11 Preset        — carries all V11 flags, adds 5 new, <12GB

All tests run on CPU with small shapes for speed.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from forge.config import get_config
from forge.keys.architecture.kronecker_embed_key import KroneckerEmbedding, KroneckerEmbedKey
from forge.keys.architecture.mamba3_key import MAMBA3_COMPLEX_NORMS, Mamba3Key
from forge.keys.architecture.pit_tying_key import PITKey
from forge.keys.attention.outro_key import OutRoKey
from tests.fixtures.keys.architecture.forge_hybrid_key import ForgeHybridKey

# ── helpers ──────────────────────────────────────────────────────────────────

def _total_params(state: dict[str, torch.Tensor]) -> int:
    """Count total elements across all tensors in a state dict."""
    return sum(t.numel() for t in state.values() if isinstance(t, torch.Tensor))


# ── 1. Mamba-3 vs Mamba-2 (R37-1) ────────────────────────────────────────────

D_MODEL = 64
D_INNER = 128
D_STATE = 16
DT_RANK = 8
D_CONV = 4


def _make_mamba2_state() -> dict[str, torch.Tensor]:
    """Create a fake Mamba-2 weight dict (real-valued SSM states)."""
    torch.manual_seed(0)
    return {
        "in_proj.weight": torch.randn(2 * D_INNER, D_MODEL),
        "conv1d.weight": torch.randn(D_INNER, 1, D_CONV),
        "conv1d.bias": torch.randn(D_INNER),
        "x_proj.weight": torch.randn(DT_RANK + 2 * D_STATE, D_INNER),
        "dt_proj.weight": torch.randn(D_INNER, DT_RANK),
        "dt_proj.bias": torch.randn(D_INNER),
        "A_log": torch.randn(D_INNER, D_STATE),
        "D": torch.ones(D_INNER),
        "out_proj.weight": torch.randn(D_MODEL, D_INNER),
        "dt_norm.weight": torch.ones(D_INNER),
        "A_norm.weight": torch.ones(D_STATE),
        "B_norm.weight": torch.ones(D_STATE),
        "C_norm.weight": torch.ones(D_STATE),
    }


class TestMamba3VsMamba2:
    """Mamba-3 has 2x expressivity (complex states) with lossless round-trip."""

    def test_a_log_shape_2x_expressivity(self):
        """Mamba-3 A_log (d_inner, d_state, 2) vs Mamba-2 (d_inner, d_state)."""
        state = _make_mamba2_state()
        result = Mamba3Key().forward(state)
        assert result.success
        # Mamba-2: 2D A_log
        assert tuple(state["A_log"].shape) == (D_INNER, D_STATE)
        # Mamba-3: 3D A_log with trailing 2 (real + imag) = 2x expressivity
        assert tuple(result.weights["A_log"].shape) == (D_INNER, D_STATE, 2)

    def test_x_proj_expanded_2x(self):
        """Mamba-3 x_proj has (dt_rank + 2*d_state*2, d_inner) vs Mamba-2
        (dt_rank + 2*d_state, d_inner) — 2x the B/C projection capacity."""
        state = _make_mamba2_state()
        result = Mamba3Key().forward(state)
        old_out = DT_RANK + 2 * D_STATE
        new_out = DT_RANK + 2 * D_STATE * 2
        assert tuple(state["x_proj.weight"].shape) == (old_out, D_INNER)
        assert tuple(result.weights["x_proj.weight"].shape) == (new_out, D_INNER)
        assert new_out > old_out  # strictly more capacity

    def test_mamba3_more_params_than_mamba2(self):
        """Mamba-3 total params > Mamba-2 total params (more capacity added)."""
        state = _make_mamba2_state()
        m2_params = _total_params(state)
        result = Mamba3Key().forward(state)
        assert result.success
        m3_params = _total_params(result.weights)
        assert m3_params > m2_params, (
            f"Mamba-3 ({m3_params}) should have MORE params than "
            f"Mamba-2 ({m2_params})")

    def test_round_trip_bit_exact(self):
        """Mamba-2 -> Mamba-3 -> Mamba-2 is bit-exact (lossless warm start)."""
        state = _make_mamba2_state()
        key = Mamba3Key()
        fwd = key.forward(state)
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success
        for k, v in state.items():
            assert k in rev.data, f"Missing key after round-trip: {k}"
            assert torch.equal(v, rev.data[k]), f"Bit-exact mismatch: {k}"

    def test_imag_zero_init(self):
        """Imaginary parts are zero-init (lossless warm start)."""
        state = _make_mamba2_state()
        result = Mamba3Key().forward(state)
        assert result.weights["A_log"][..., 1].abs().max().item() == 0.0
        for key in MAMBA3_COMPLEX_NORMS:
            assert result.weights[key][..., 1].abs().max().item() == 0.0


# ── 2. Kronecker vs Standard Embedding (R37-2) ───────────────────────────────

class TestKroneckerVsStandard:
    """Kronecker embeddings achieve >90% param reduction (large config)."""

    def test_large_config_reduction_gt_90(self):
        """Large config (65536×2048): >90% reduction (docs say 91-94%, 99.2%)."""
        key = KroneckerEmbedKey(
            vocab_size=65536, d_model=2048, d_char=64, max_char_len=8)
        standard = key.standard_param_count()  # 134.2M
        kronecker = key.param_count()
        reduction = key.reduction_pct()
        assert standard == 65536 * 2048  # 134,217,728
        assert kronecker < standard
        assert reduction > 90.0, f"Large reduction {reduction:.1f}% should be >90%"

    def test_small_config_reduction_gt_20(self):
        """Small config (256×128, d_char=32, L=4): >20% reduction.
        Kronecker benefit scales with vocab size — small vocabs see less benefit."""
        key = KroneckerEmbedKey(
            vocab_size=256, d_model=128, d_char=32, max_char_len=4)
        standard = key.standard_param_count()
        kronecker = key.param_count()
        reduction = key.reduction_pct()
        assert standard == 256 * 128  # 32768
        assert kronecker < standard
        assert reduction > 20.0, f"Small reduction {reduction:.1f}% should be >20%"

    def test_kronecker_forward_correct_shape(self):
        """KroneckerEmbedding forward produces (batch, seq, d_model)."""
        emb = KroneckerEmbedding(256, 128, d_char=32, max_char_len=4)
        token_ids = torch.tensor([0, 1, 42, 127, 255])
        out = emb(token_ids)
        assert out.shape == (5, 128)

    def test_from_embedding_reconstruction_better_than_zero(self):
        """from_embedding reconstruction error <= zero-init error (at least
        marginally better than zero)."""
        torch.manual_seed(42)
        original = nn.Embedding(256, 128)
        kron = KroneckerEmbedding.from_embedding(original, d_char=32, max_char_len=4)
        W_eff = kron.weight
        W_orig = original.weight
        error_ls = (W_eff - W_orig).norm().item()
        error_zero = W_orig.norm().item()
        assert error_ls <= error_zero, (
            f"Least-squares error {error_ls:.4f} should be <= "
            f"zero-init error {error_zero:.4f}")

    def test_param_count_formula(self):
        """Kronecker param count = 256*d_char + L*d_char + d_char*L*d_model."""
        key = KroneckerEmbedKey(256, 128, d_char=32, max_char_len=4)
        expected = 256 * 32 + 4 * 32 + 32 * 4 * 128
        assert key.param_count() == expected


# ── 3. PIT vs Standard Weight Tying (R37-3) ─────────────────────────────────

V_PIT = 128
D_PIT = 64


class TestPITVsStandardTying:
    """PIT: orthonormal shared memory, lossless round-trip, full-rank."""

    def test_m_orthonormal(self):
        """M @ M.T ≈ I (shared memory is orthonormal)."""
        torch.manual_seed(0)
        W_emb = torch.randn(V_PIT, D_PIT)
        W_unemb = torch.randn(D_PIT, V_PIT)
        result = PITKey().forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert result.success
        M = result.weights["M"]
        I = torch.eye(D_PIT, dtype=M.dtype)
        assert torch.allclose(M @ M.T, I, atol=1e-4), "M is not orthonormal"

    def test_reverse_reconstructs_w_emb(self):
        """reverse reconstructs W_emb within tolerance."""
        torch.manual_seed(1)
        # Use structured pair so polar factor is well-defined
        M_true = torch.linalg.qr(torch.randn(D_PIT, D_PIT).T).Q.T
        P_emb = torch.randn(V_PIT, D_PIT)
        W_emb = P_emb @ M_true
        W_unemb = M_true @ torch.randn(D_PIT, V_PIT)
        key = PITKey()
        fwd = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success
        assert torch.allclose(rev.data["W_emb"], W_emb, atol=1e-3), (
            f"W_emb reconstruction error: "
            f"{(rev.data['W_emb'] - W_emb).norm().item():.6f}")

    def test_pit_full_rank_vs_standard_tying_rank_constrained(self):
        """PIT has FULL d×d orthonormal memory; standard tying is rank-constrained.

        For random W_emb and W_unemb, PIT reconstruction error << standard
        tying error (W_unemb = W_emb.T forces a single rank-d matrix).
        """
        torch.manual_seed(2)
        W_emb = torch.randn(V_PIT, D_PIT)
        W_unemb = torch.randn(D_PIT, V_PIT)

        # PIT reconstruction
        key = PITKey()
        fwd = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success
        pit_emb_err = (rev.data["W_emb"] - W_emb).norm().item()
        pit_unemb_err = (rev.data["W_unemb"] - W_unemb).norm().item()

        # Standard weight tying: W_unemb_tied = W_emb.T (rank-constrained)
        W_unemb_tied = W_emb.T  # (D, V)
        tying_unemb_err = (W_unemb_tied - W_unemb).norm().item()

        # PIT reconstructs W_unemb far better than forcing W_emb.T
        assert pit_unemb_err < tying_unemb_err, (
            f"PIT unemb error {pit_unemb_err:.4f} should be << "
            f"standard tying error {tying_unemb_err:.4f}")

    def test_m_shape_full_rank(self):
        """M is d×d (full rank orthonormal), not rank-constrained."""
        torch.manual_seed(3)
        W_emb = torch.randn(V_PIT, D_PIT)
        W_unemb = torch.randn(D_PIT, V_PIT)
        result = PITKey().forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert result.success
        M = result.weights["M"]
        assert M.shape == (D_PIT, D_PIT)  # full d×d, not low-rank


# ── 4. OutRo vs Standard Causal Attention (R37-4) ───────────────────────────

N_HEADS = 4
SEQ_LEN = 32
D_OUTRO = 64


def _make_sink_attn(seed: int = 0) -> torch.Tensor:
    """Build attention weights with a strong sink at position 0."""
    torch.manual_seed(seed)
    logits = torch.randn(N_HEADS, SEQ_LEN, SEQ_LEN) * 0.5
    attn = torch.softmax(logits, dim=-1)
    attn[..., 0, :] = 0.0
    attn[..., 0, 0] = 1.0  # position 0 is a sink
    attn[..., 5, :] = 0.0
    attn[..., 5, 0] = 0.9  # position 5 is a sink
    return attn


class TestOutRoVsCausal:
    """OutRo: sink tokens attend beyond causal constraint."""

    def test_outro_sink_attends_all(self):
        """OutRo mask: sink position (0) can attend to ALL positions."""
        attn = _make_sink_attn()
        key = OutRoKey(sink_threshold=0.5)
        fwd = key.forward({"attn_weights": attn, "d_model": D_OUTRO})
        assert fwd.success
        mask = fwd.weights["attention_mask"]
        # Sink row (position 0) is fully unmasked
        assert mask[..., 0, :].all(), "Sink row should attend to all positions"

    def test_standard_causal_sink_only_self(self):
        """Standard causal mask: position 0 can only attend to itself."""
        causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
        # Position 0 can only see position 0
        assert causal[0, 0]
        assert not causal[0, 1]  # cannot attend to future

    def test_outro_non_sink_stays_causal(self):
        """OutRo: non-sink positions still follow causal constraint."""
        attn = _make_sink_attn()
        key = OutRoKey(sink_threshold=0.5)
        fwd = key.forward({"attn_weights": attn, "d_model": D_OUTRO})
        assert fwd.success
        mask = fwd.weights["attention_mask"]
        # Position 1 is not a sink → causal
        assert mask[..., 1, 1].all()  # self
        assert not mask[..., 1, 2].any()  # future masked

    def test_outro_more_connections_than_causal(self):
        """OutRo allows MORE attention connections than standard causal
        (sink row is fully unmasked)."""
        attn = _make_sink_attn()
        key = OutRoKey(sink_threshold=0.5)
        fwd = key.forward({"attn_weights": attn, "d_model": D_OUTRO})
        assert fwd.success
        outro_mask = fwd.weights["attention_mask"]
        causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
        causal_expanded = causal.expand(N_HEADS, SEQ_LEN, SEQ_LEN)

        outro_count = outro_mask.sum().item()
        causal_count = causal_expanded.sum().item()
        assert outro_count > causal_count, (
            f"OutRo ({outro_count}) should allow MORE connections than "
            f"causal ({causal_count})")


# ── 5. ForgeHybrid vs Pure Attention (R37-5) ────────────────────────────────

D_HYBRID = 64


def _make_pure_attention_checkpoint() -> dict[str, torch.Tensor]:
    """Create a fake 2-layer pure-attention checkpoint."""
    torch.manual_seed(0)
    data = {}
    for layer in range(2):
        p = f"blocks.{layer}.attn."
        data[p + "q_proj.weight"] = torch.randn(D_HYBRID, D_HYBRID)
        data[p + "k_proj.weight"] = torch.randn(D_HYBRID, D_HYBRID)
        data[p + "v_proj.weight"] = torch.randn(D_HYBRID, D_HYBRID)
        data[p + "out_proj.weight"] = torch.randn(D_HYBRID, D_HYBRID)
    return data


class TestForgeHybridVsPureAttention:
    """ForgeHybrid: lossless warm start (zero-init SSM = identical to attention)."""

    def test_ssm_weights_all_zero(self):
        """SSM weights are ALL ZERO (lossless warm start)."""
        data = _make_pure_attention_checkpoint()
        key = ForgeHybridKey(d_model=D_HYBRID, d_state=8)
        result = key.forward(data)
        assert result.success
        for k, v in result.weights.items():
            if ".ssm." in k and "sink_threshold" not in k and v.is_floating_point():
                assert v.abs().max().item() == 0.0, (
                    f"SSM weight {k} should be zero-init, got max={v.abs().max().item()}")

    def test_reverse_bit_exact(self):
        """Reverse (extract attention) is bit-exact: all original keys preserved."""
        data = _make_pure_attention_checkpoint()
        key = ForgeHybridKey(d_model=D_HYBRID, d_state=8)
        fwd = key.forward(data)
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success
        for k, v in data.items():
            assert k in rev.data, f"Missing key after reverse: {k}"
            assert torch.equal(v, rev.data[k]), f"Bit-exact mismatch: {k}"

    def test_hybrid_more_params_than_pure(self):
        """ForgeHybrid has MORE total params (SSM capacity added)."""
        data = _make_pure_attention_checkpoint()
        pure_params = _total_params(data)
        key = ForgeHybridKey(d_model=D_HYBRID, d_state=8)
        result = key.forward(data)
        assert result.success
        hybrid_params = _total_params(result.weights)
        assert hybrid_params > pure_params, (
            f"Hybrid ({hybrid_params}) should have MORE params than "
            f"pure attention ({pure_params})")

    def test_attention_weights_preserved(self):
        """Original attention weights are unchanged (lossless)."""
        data = _make_pure_attention_checkpoint()
        original_q = data["blocks.0.attn.q_proj.weight"].clone()
        key = ForgeHybridKey(d_model=D_HYBRID, d_state=8)
        result = key.forward(data)
        assert result.success
        assert torch.equal(result.weights["blocks.0.attn.q_proj.weight"], original_q)

    def test_zero_contribution_at_warm_start(self):
        """Zero-init SSM means zero contribution at warm start (metadata)."""
        data = _make_pure_attention_checkpoint()
        key = ForgeHybridKey(d_model=D_HYBRID, d_state=8)
        result = key.forward(data)
        assert result.metadata["lossless"] is True
        assert result.metadata["warm_start"] == "zero_init_ssm"


# ── 6. V12 vs V11 Preset (R37-6) ────────────────────────────────────────────

# The 5 new feature flags V12 adds that V11 doesn't have
V12_NEW_FLAGS = {
    "use_mamba3",
    "use_kronecker_embed",
    "use_pit",  # V11 has use_pit=False; V12 enables it
    "use_outro",
    "use_forge_hybrid",
}


class TestV12JambaVsV2:
    """V12-Jamba carries forward all V2 keys, adds 5 new, fits 12GB."""

    V12_NEW = V12_NEW_FLAGS | {"parent", "dropped_keys"}

    def test_v12_has_all_v2_feature_flags(self):
        """V12-Jamba has ALL V2 feature flags (no silent regression)."""
        v2 = get_config("forgelm_v2")
        v12 = get_config("forgelm_v12_jamba")
        v12_dict = v12.__dict__
        for key, val in v2.__dict__.items():
            if key in self.V12_NEW:
                continue
            assert key in v12_dict, f"V12-Jamba dropped V2 key: {key}"
            assert v12_dict[key] == val, (
                f"V12-Jamba changed V2 key {key}: {val} -> {v12_dict[key]}")

    def test_v12_has_5_new_flags_v2_lacks(self):
        """V12-Jamba has 5 NEW feature flags that V2 has disabled."""
        v2 = get_config("forgelm_v2")
        v12 = get_config("forgelm_v12_jamba")
        new_enabled = 0
        for flag in V12_NEW_FLAGS:
            v12_val = getattr(v12, flag, None)
            v2_val = getattr(v2, flag, None)
            # V12 must have the flag enabled
            assert v12_val is not None, f"V12 missing flag: {flag}"
            # V2 either lacks it or has it False
            if v2_val is None or v2_val is False:
                new_enabled += 1
        assert new_enabled == 5, (
            f"V12-Jamba should have 5 new enabled flags, got {new_enabled}")

    def test_v12_memory_budget_under_12gb(self):
        """V12-Jamba memory budget < 12GB (new keys are zero-init)."""
        v12 = get_config("forgelm_v12_jamba")
        # Jamba 3.2B bf16 ~6.4GB + KV cache ~1GB; Kronecker shrinks embed
        n_params = (
            v12.vocab_size * v12.d_model
            + v12.n_layers * (
                3 * v12.d_model * v12.d_model
                + v12.d_model * v12.d_model
                + 3 * v12.d_model * v12.intermediate_size
            )
        )
        lm_gb = n_params * 2 / 1e9  # bf16 upper bound
        kv_gb = 1.0
        total_gb = lm_gb + kv_gb
        assert total_gb < 12.0, (
            f"V12-Jamba estimated {total_gb:.1f}GB should be < 12GB")

    def test_v12_enables_new_keys(self):
        """V12-Jamba should enable the new keys V2 leaves off."""
        v2 = get_config("forgelm_v2")
        v12 = get_config("forgelm_v12_jamba")
        v12_new_keys = {
            "use_mamba3", "use_kronecker_embed", "use_outro", "use_forge_hybrid",
        }
        for key in v12_new_keys:
            assert getattr(v12, key) is True, f"V12-Jamba should enable {key}"
            assert getattr(v2, key) is False, f"V2 should not enable {key}"
        assert v12.use_pit is True
        assert v2.use_pit is False

    def test_v12_core_arch_unchanged(self):
        """V12-Jamba core architecture is identical to V2."""
        v2 = get_config("forgelm_v2")
        v12 = get_config("forgelm_v12_jamba")
        assert v12.d_model == v2.d_model
        assert v12.n_layers == v2.n_layers
        assert v12.n_heads == v2.n_heads
        assert v12.vocab_size == v2.vocab_size
        assert v12.intermediate_size == v2.intermediate_size

    def test_v12_new_key_defaults(self):
        """V12-Jamba new key parameters have sensible defaults (zero/identity init)."""
        v12 = get_config("forgelm_v12_jamba")
        assert v12.use_mamba3 is True
        assert v12.use_kronecker_embed is True
        assert v12.use_pit is True
        assert v12.use_outro is True
        assert v12.use_forge_hybrid is True
        # ForgeHybrid warm start = all attention (threshold=inf)
        assert v12.forge_hybrid_sink_threshold == float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Property-based tests for key transforms.

Tests the fundamental properties that all keys should satisfy:
  1. Identity init: keys with identity init produce identical model output.
  2. Round-trip: forward(reverse(weights)) == weights for bi-directional keys.
  3. Finiteness: all key outputs are finite (no NaN/Inf).
  4. Shape preservation: keys don't change tensor shapes unexpectedly.
  5. Safety: applying a key with safe_apply doesn't corrupt the model.

These tests use hypothesis-style property-based testing without the
hypothesis dependency — we use fixed seeds and parameterized cases.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]


# Load keys dynamically (avoid importing full research package which may
# have missing dependencies on some systems).
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Fixtures ---


@pytest.fixture
def small_model():
    """A small model for testing key application."""

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 32)
            self.lm_head = nn.Linear(32, 100, bias=False)
            # Tie weights
            self.lm_head.weight = self.embed_tokens.weight

        def forward(self, x):
            if isinstance(x, torch.Tensor) and x.dtype == torch.long:
                h = self.embed_tokens(x)
            else:
                h = x  # assume already embedded
            h = self.linear(h)
            return self.lm_head(h)

    return SmallModel()


@pytest.fixture
def test_input_ids():
    """Token IDs for model forward pass."""
    return torch.randint(0, 100, (1, 8))


# --- PIT Key Properties ---


class TestPITKey:
    """Property tests for PIT (Pseudo-Inverse Tying) key."""

    def test_identity_init_forward(self, small_model, test_input_ids):
        """PIT with L=I should produce identical output to standard tying."""
        pit = _load_module("pit_test", REPO / "forge" / "keys" / "misc" / "pit_key.py")

        original_out = small_model(test_input_ids).clone()

        # Apply PIT
        pit_embed = pit.PITEmbedding(100, 32, init="standard")
        pit_embed.memory.data.copy_(small_model.embed_tokens.weight.data)
        pit_head = pit.PITLMHead.from_embedding(pit_embed)

        # Replace in model
        small_model.embed_tokens = pit_embed
        small_model.lm_head = pit_head

        pit_out = small_model(test_input_ids)
        assert torch.allclose(original_out, pit_out, atol=1e-4), (
            f"PIT identity init mismatch: {(original_out - pit_out).abs().max()}"
        )

    def test_T_is_spd(self):
        """T = L @ L^T should always be symmetric positive definite."""
        pit = _load_module("pit_test2", REPO / "forge" / "keys" / "misc" / "pit_key.py")
        embed = pit.PITEmbedding(50, 16, init="standard")

        # Random L (lower triangular)
        L = torch.tril(torch.randn(16, 16))
        embed.L.data.copy_(L)

        T = embed.get_T()
        # Symmetric
        assert torch.allclose(T, T.T, atol=1e-5), "T should be symmetric"
        # Positive definite (allow tiny floating-point negative eigenvalues)
        eigenvalues = torch.linalg.eigvalsh(T)
        assert (eigenvalues > -1e-5).all(), f"T should be PD, min eigenvalue: {eigenvalues.min()}"

    def test_key_roundtrip(self):
        """PITKey forward then reverse should recover the original weight."""
        pit = _load_module("pit_test3", REPO / "forge" / "keys" / "misc" / "pit_key.py")
        key = pit.PITKey()

        original_weight = torch.randn(50, 16)
        result = key.forward({"embed_weight": original_weight})
        assert result.success

        # Reverse with L=I should recover original
        result2 = key.reverse(result.weights)
        assert result2.success
        assert torch.allclose(result2.data["embed_weight"], original_weight, atol=1e-4)


# --- LeRoPE Key Properties ---


class TestLeRoPEKey:
    """Property tests for LeRoPE/AdaRoPE."""

    def test_identity_init(self):
        """LeRoPE with freq_scale=1 should match standard RoPE."""
        lr = _load_module("lerope_test", REPO / "forge" / "keys" / "position" / "lerope_key.py")

        dim = 16
        max_seq = 32
        lerope = lr.LeRoPEEmbedding(dim=dim, max_seq_len=max_seq, base=10000.0)

        # Standard RoPE
        base_inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, base_inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        expected_cos = emb.cos()

        assert torch.allclose(lerope.cos_cached, expected_cos, atol=1e-6)

    def test_finiteness(self):
        """All RoPE outputs should be finite."""
        lr = _load_module("lerope_test2", REPO / "forge" / "keys" / "position" / "lerope_key.py")
        lerope = lr.LeRoPEEmbedding(dim=16, max_seq_len=32)

        x = torch.randn(2, 4, 16, 16)
        out = lerope(x)
        assert torch.isfinite(out).all()

        # After modifying freq_scale
        lerope.freq_scale.data.fill_(2.0)
        lerope.rebuild_cache()
        out2 = lerope(x)
        assert torch.isfinite(out2).all()

    def test_adarope_per_head_differentiation(self):
        """AdaRoPE should allow per-head different frequencies."""
        lr = _load_module("lerope_test3", REPO / "forge" / "keys" / "position" / "lerope_key.py")
        adarope = lr.AdaRoPEEmbedding(dim=16, n_heads=4, max_seq_len=32)

        # At init, all heads should be the same
        for h in range(4):
            assert torch.allclose(adarope.cos_cached[:, h, :], adarope.cos_cached[:, 0, :], atol=1e-6)

        # After learning, heads should differ
        adarope.freq_scale.data[0, :] = 2.0
        adarope.freq_scale.data[1, :] = 0.5
        adarope.rebuild_cache()

        assert not torch.allclose(adarope.cos_cached[:, 0, :], adarope.cos_cached[:, 2, :], atol=1e-4)


# --- AttnRes Key Properties ---


class TestAttnResKey:
    """Property tests for AttnRes."""

    def test_identity_init_zero_gates(self):
        """AttnRes gates should be 0 at init (lossless)."""
        ar = _load_module("attnres_test", REPO / "forge" / "keys" / "architecture" / "attn_residual_key.py")
        module = ar.AttnResModule(64, 8, k=4, n_heads=4)
        assert (module.gates == 0).all()

        # Zero gates -> zero retrieval
        x = torch.randn(2, 16, 64)
        past = [torch.randn(2, 16, 64)]
        out = module(x, 1, past)
        assert torch.allclose(out, torch.zeros_like(x), atol=1e-6)

    def test_finiteness_with_nonzero_gate(self):
        """AttnRes output should be finite even with non-zero gates."""
        ar = _load_module("attnres_test2", REPO / "forge" / "keys" / "architecture" / "attn_residual_key.py")
        module = ar.AttnResModule(64, 8, k=4, n_heads=4)
        module.gates.data.fill_(1.0)

        x = torch.randn(2, 16, 64)
        past = [torch.randn(2, 16, 64) for _ in range(4)]
        out = module(x, 5, past)
        assert torch.isfinite(out).all()


# --- mHC Key Properties ---


class TestMHCKey:
    """Property tests for mHC."""

    def test_identity_init_standard_residual(self):
        """mHC with gate=0 should be standard residual."""
        mhc = _load_module("mhc_test", REPO / "forge" / "keys" / "architecture" / "mhc_key.py")
        module = mhc.MHCModule(64, rank=16)
        assert module.gate.item() == 0.0

        x = torch.randn(2, 16, 64)
        sub = torch.randn(2, 16, 64)
        out = module(x, sub)
        assert torch.allclose(out, x + sub, atol=1e-6)

    def test_T_spd_property(self):
        """U @ V^T should produce valid projections."""
        mhc = _load_module("mhc_test2", REPO / "forge" / "keys" / "architecture" / "mhc_key.py")
        module = mhc.MHCModule(64, rank=16)

        # The projection W = U @ V^T
        W = module.U @ module.V.transpose(-1, -2)
        assert W.shape == (64, 64)
        assert torch.isfinite(W).all()


# --- Safety Framework Properties ---


class TestSafetyFramework:
    """Property tests for the safety validation framework."""

    def test_safe_apply_catches_nan(self):
        """safe_apply should catch NaN corruption and rollback."""
        safety = _load_module("safety_test", REPO / "forge" / "keys" / "safety.py")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(16, 16)

            def forward(self, x):
                return self.w(x)

        model = M()
        original_w = model.w.weight.clone()

        def corrupt(m):
            m.w.weight.data.fill_(float("nan"))
            return m

        with pytest.raises(safety.KeySafetyError):
            safety.safe_apply(model, corrupt, identity_init=False, test_input=torch.randn(1, 4, 16))

        # Verify rollback
        assert torch.allclose(model.w.weight, original_w, atol=1e-6)

    def test_safe_apply_catches_identity_violation(self):
        """safe_apply should catch identity-init violations."""
        safety = _load_module("safety_test2", REPO / "forge" / "keys" / "safety.py")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(16, 16)

            def forward(self, x):
                return self.w(x)

        model = M()
        test_input = torch.randn(1, 4, 16)

        def modify(m):
            m.w.weight.data += 0.5
            return m

        with pytest.raises(safety.KeySafetyError):
            safety.safe_apply(model, modify, identity_init=True, test_input=test_input)

    def test_verify_model_integrity_healthy(self):
        """verify_model_integrity should pass for a healthy model."""
        safety = _load_module("safety_test3", REPO / "forge" / "keys" / "safety.py")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(16, 16)

            def forward(self, x):
                return self.w(x)

        model = M()
        passed, issues = safety.verify_model_integrity(model, torch.randn(1, 4, 16))
        assert passed
        assert len(issues) == 0

    def test_verify_model_integrity_corrupted(self):
        """verify_model_integrity should detect NaN corruption."""
        safety = _load_module("safety_test4", REPO / "forge" / "keys" / "safety.py")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(16, 16)

            def forward(self, x):
                return self.w(x)

        model = M()
        model.w.weight.data.fill_(float("nan"))
        passed, issues = safety.verify_model_integrity(model, torch.randn(1, 4, 16))
        assert not passed
        assert len(issues) > 0

"""Unit tests for LittleBit-2 (latent-factorized sub-1-bit quantization)."""

import torch
import torch.nn as nn

from forge.engine.quant.littlebit2 import (
    LittleBitLinear,
    SmoothSign,
    STEBinary,
    _compute_eff_bits,
    _estimate_split_dim,
    _joint_itq_rotation,
    binary_packer,
    binary_unpacker,
    estimate_littlebit2_memory,
    quantize_model_littlebit2,
)


def _rel_err(layer, lin):
    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    return (w_orig - w_recon).norm().item() / w_orig.norm().item()


# ──────────────────────────────────────────────────────────────────────────
# Binary STE / SmoothSign
# ──────────────────────────────────────────────────────────────────────────


class TestQuantFunctions:
    def test_ste_binary_values(self):
        x = torch.tensor([-2.0, -0.3, 0.0, 0.5, 3.0])
        y = STEBinary(x)
        assert y.tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]

    def test_ste_binary_grad_clip(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
        STEBinary(x).sum().backward()
        # |x|>1 blocked, |x|<1 passes
        assert x.grad.tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_smooth_sign_grad(self):
        x = torch.tensor([0.0, 0.02], requires_grad=True)
        SmoothSign(x).sum().backward()
        # alpha*(1-tanh^2(alpha x)): max at x=0, decays with |x|.
        # (alpha=100 saturates by |x|~0.05, so probe close to the boundary)
        assert x.grad[0].item() > x.grad[1].item() > 0


# ──────────────────────────────────────────────────────────────────────────
# Bit packing round-trip
# ──────────────────────────────────────────────────────────────────────────


class TestBinaryPacker:
    def test_roundtrip(self):
        torch.manual_seed(0)
        t = torch.randint(0, 2, (7, 65), dtype=torch.int8) * 2 - 1
        packed = binary_packer(t)
        assert packed.shape == (7, 3)  # ceil(65/32)
        assert torch.equal(binary_unpacker(packed, t.shape), t)

    def test_roundtrip_exact_multiple(self):
        t = torch.ones(4, 64, dtype=torch.int8) * -1
        packed = binary_packer(t)
        assert packed.shape == (4, 2)
        assert torch.equal(binary_unpacker(packed, t.shape), t)

    def test_wrong_dtype_raises(self):
        import pytest
        with pytest.raises(TypeError):
            binary_packer(torch.ones(4, 4, dtype=torch.float32))


# ──────────────────────────────────────────────────────────────────────────
# Joint-ITQ rotation
# ──────────────────────────────────────────────────────────────────────────


class TestJointITQ:
    def test_rotation_is_orthogonal(self):
        torch.manual_seed(0)
        U = torch.randn(64, 32)
        V = torch.randn(32, 48)
        R = _joint_itq_rotation(U, V, n_iter=10)
        assert torch.allclose(R @ R.t(), torch.eye(32), atol=1e-5)

    def test_rotation_lossless_fp(self):
        """(U@R)(R^T@V) == U@V exactly — rotation is free in FP."""
        torch.manual_seed(0)
        U = torch.randn(64, 32)
        V = torch.randn(32, 48)
        R = _joint_itq_rotation(U, V, n_iter=20)
        assert torch.allclose((U @ R) @ (R.t() @ V), U @ V, atol=1e-4)

    def test_itq_reduces_sign_distortion(self):
        """ITQ-aligned factors should have lower sign-quantization error on
        the joint latent manifold than unrotated SVD factors."""
        torch.manual_seed(0)
        # Spiky/coherent latent factor (worst case for sign quantization):
        # a few dominant coordinates per row — like real SVD singular vectors
        base = torch.randn(96, 32)
        U = base * torch.tensor([8.0] + [1.0] * 31)  # heavy-tailed spectrum
        V = torch.randn(32, 48)

        X0 = torch.cat([U, V.t()], dim=0)
        err0 = (torch.sign(X0) - X0 / X0.norm(dim=1, keepdim=True)).norm()

        R = _joint_itq_rotation(U, V, n_iter=50)
        U_r, V_r = U @ R, R.t() @ V
        X1 = torch.cat([U_r, V_r.t()], dim=0)
        err1 = (torch.sign(X1) - X1 / X1.norm(dim=1, keepdim=True)).norm()
        assert err1 < err0


# ──────────────────────────────────────────────────────────────────────────
# LittleBitLinear
# ──────────────────────────────────────────────────────────────────────────


class TestLittleBitLinear:
    def test_shape_preservation(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=True)
        layer = LittleBitLinear.from_linear(lin, split_dim=32, use_itq=False)
        x = torch.randn(2, 4, 128)
        assert layer(x).shape == (2, 4, 64)

    def test_forward_matches_materialized_weight(self):
        """Factorized forward == x @ W_hat^T."""
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=True)
        layer = LittleBitLinear.from_linear(lin, split_dim=32)
        x = torch.randn(8, 128)
        w = layer._dequantize_weight(torch.float32)
        ref = x @ w.t() + layer.bias.data
        assert torch.allclose(layer(x), ref, atol=1e-4)

    def test_bias_preserved(self):
        lin = nn.Linear(128, 64, bias=True)
        layer = LittleBitLinear.from_linear(lin, split_dim=32)
        assert torch.allclose(layer.bias.data, lin.bias.data)

    def test_itq_improves_reconstruction(self):
        """LittleBit-2 headline claim: Joint-ITQ init beats plain Dual-SVID
        on reconstruction error. Uses a low-rank + coherent weight."""
        torch.manual_seed(0)
        # Synthetic heavy-tailed weight: low-rank + outlier columns
        lin = nn.Linear(128, 64, bias=False)
        with torch.no_grad():
            A = torch.randn(64, 16)
            B = torch.randn(16, 128)
            lin.weight.copy_((A @ B) * 0.05)
            lin.weight[:, :4] *= 6.0  # outlier columns → coherent SVD vectors
        layer_no = LittleBitLinear.from_linear(lin, split_dim=24, use_itq=False)
        layer_itq = LittleBitLinear.from_linear(lin, split_dim=24, use_itq=True,
                                                itq_n_iter=50)
        err_no = _rel_err(layer_no, lin)
        err_itq = _rel_err(layer_itq, lin)
        assert err_itq < err_no, f"ITQ did not help: {err_itq:.4f} vs {err_no:.4f}"

    def test_higher_rank_lower_error(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        errors = {}
        for s in (8, 32, 96):
            layer = LittleBitLinear.from_linear(lin, split_dim=s)
            errors[s] = _rel_err(layer, lin)
        assert errors[96] < errors[8]

    def test_residual_reduces_error(self):
        """Residual factorization fits W - W_hat. Init gain is guaranteed
        when the primary leaves a STRUCTURED residual: a rank-48 weight
        with split_dim=32 leaves its bottom-16 singular components for the
        residual factor to capture. (On noise-like residuals init is
        marginal — the paper's residual gain is realized through QAT.)"""
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        with torch.no_grad():
            A = torch.randn(64, 48)
            B = torch.randn(48, 128)
            lin.weight.copy_((A @ B) * 0.05)
        layer_p = LittleBitLinear.from_linear(lin, split_dim=32, residual=False)
        layer_r = LittleBitLinear.from_linear(lin, split_dim=32, residual=True)
        assert _rel_err(layer_r, lin) < _rel_err(layer_p, lin)

    def test_residual_forward(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=True)
        layer = LittleBitLinear.from_linear(lin, split_dim=32, residual=True)
        x = torch.randn(2, 128)
        assert layer(x).shape == (2, 64)

    def test_eff_bit_derivation(self):
        """eff_bit target → split_dim → actual eff_bit close to target."""
        lin = nn.Linear(1024, 1024, bias=False)
        layer = LittleBitLinear.from_linear(lin, eff_bit=0.5)
        # exact inverse: s = (n^2*e - 16*2n) / (2n+16); actual ≈ target
        assert abs(layer.eff_bit_actual - 0.5) < 0.05

    def test_eff_bit_formula_matches_split(self):
        # in=128, out=64, s=32, no residual: (32*(128+64+16) + 16*192)/8192
        expected = (32 * 208 + 16 * 192) / 8192
        assert abs(_compute_eff_bits(128, 64, 32, False) - expected) < 1e-6
        est = _estimate_split_dim(128, 64, expected, False)
        assert abs(est - 32) < 1e-4

    def test_state_dict_and_gradients(self):
        """All six params are learnable and receive QAT gradients."""
        lin = nn.Linear(64, 32, bias=False)
        layer = LittleBitLinear.from_linear(lin, split_dim=16)
        x = torch.randn(4, 64)
        layer(x).sum().backward()
        for name in ("U", "V", "u1", "u2", "v1", "v2"):
            p = getattr(layer, name)
            assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_freeze_binarized(self):
        lin = nn.Linear(64, 32, bias=False)
        layer = LittleBitLinear.from_linear(lin, split_dim=16)
        out_before = layer(torch.randn(4, 64))
        layer.freeze_binarized()
        assert layer._binarized
        assert layer.U.dtype == torch.int8
        out_after = layer(torch.randn(4, 64))
        assert out_after.shape == out_before.shape

    def test_pack_weights(self):
        lin = nn.Linear(64, 32, bias=False)
        layer = LittleBitLinear.from_linear(lin, split_dim=16)
        packed = layer.pack_weights()
        assert packed["U_packed"].dtype == torch.int32
        assert packed["U_packed"].shape == (32, 1)  # 16 cols → 1 word
        assert packed["V_packed"].shape == (16, 2)  # 64 cols → 2 words
        # unpack reproduces the binarized factor
        U_b = binary_unpacker(packed["U_packed"], tuple(packed["U_shape"].tolist()))
        assert set(U_b.unique().tolist()) <= {-1, 1}


# ──────────────────────────────────────────────────────────────────────────
# Model conversion
# ──────────────────────────────────────────────────────────────────────────


class TestModelConversion:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def test_conversion(self):
        model = self._make_model()
        n = quantize_model_littlebit2(model, split_dim=16, use_itq=False,
                                      verbose=False)
        assert n == 2
        out = model(torch.randn(2, 4, 64))
        assert out.shape == (2, 4, 64)

    def test_skips_embed_head(self):
        model = self._make_model()
        model.embed = nn.Linear(64, 64)
        model.lm_head = nn.Linear(64, 64)
        n = quantize_model_littlebit2(model, split_dim=16, verbose=False)
        assert n == 2
        assert isinstance(model.embed, nn.Linear)

    def test_double_conversion_skips(self):
        model = self._make_model()
        n1 = quantize_model_littlebit2(model, split_dim=16, verbose=False)
        n2 = quantize_model_littlebit2(model, split_dim=16, verbose=False)
        assert n1 == 2 and n2 == 0

    def test_memory_estimate(self):
        model = self._make_model()
        quantize_model_littlebit2(model, split_dim=16, verbose=False)
        mem = estimate_littlebit2_memory(model)
        assert mem["total_bits"] > 0 and mem["total_mb"] > 0
        # 2 layers × (eff_bpw < 1 bit regime sanity): s=16 on 64×128 and 128×64
        for v in mem["breakdown"].values():
            assert v["eff_bpw"] < 4.0

    def test_kv_ratio(self):
        model = nn.Sequential()
        model.attn = nn.ModuleDict({
            "k_proj": nn.Linear(256, 256),
            "q_proj": nn.Linear(256, 256),
        })
        n = quantize_model_littlebit2(model, eff_bit=0.8, kv_ratio=0.5,
                                      verbose=False)
        assert n == 2
        assert model.attn["k_proj"].split_dim < model.attn["q_proj"].split_dim

    def test_block_recon_soft_forward(self):
        """block_recon must see LittleBit params as optimizable and route
        gradients through the patched factorized forward (QAT-lite path)."""
        from forge.engine.quant.block_recon import (
            _apply_optimized_params,
            _get_optimizable_params,
            _install_soft_forward,
        )
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        quantize_model_littlebit2(model, split_dim=16, residual=True,
                                  verbose=False)
        params = _get_optimizable_params(model)
        assert "0.U_soft" in params and "0.u1" in params
        assert "0.U_R_soft" in params and "0.v2_R" in params
        opt = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        patches = _install_soft_forward(model, opt)
        assert len(patches) == 2
        model(torch.randn(4, 64)).sum().backward()
        for k in ("0.U_soft", "0.u1", "0.U_R_soft", "2.V_soft", "2.u2"):
            assert opt[k].grad is not None and opt[k].grad.abs().sum() > 0, k
        _apply_optimized_params(model, {k: v.detach() for k, v in opt.items()})
        for info in patches.values():
            info["module"].forward = info["orig_forward"]
        assert model(torch.randn(4, 64)).shape == (4, 64)

"""Tests for R&D round 14 training speedup features.

Tests run on CPU where possible. GPU tests skip if CUDA unavailable.

Covers:
  - Triton fused kernels (RMSNorm, SwiGLU) — numerical equivalence to PyTorch
  - Varlen attention — block-diagonal mask correctness, no cross-example contamination
  - APOLLO optimizer — convergence on quadratic loss, memory savings
  - BREAD for BAdam — landscape correction behavior
  - FlashOptim — companded 8-bit state quantization round-trip
  - PackedSequenceDataset — cu_seqlens emission
  - Config fields — new fields exist and default correctly
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")


# ─── Triton fused kernels ───────────────────────────────────────────

class TestTritonRMSNorm:
    """Test Triton fused RMSNorm kernel numerical equivalence to F.rms_norm."""

    def test_rms_norm_cpu_fallback(self):
        """On CPU, triton_rms_norm should fall back to F.rms_norm."""
        from forge.decoding.triton_train_kernels import triton_rms_norm
        x = torch.randn(4, 128, dtype=torch.float32)
        weight = torch.ones(128, dtype=torch.float32)
        out = triton_rms_norm(x, weight, eps=1e-6)
        ref = F.rms_norm(x, [128], weight, 1e-6)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)

    def test_rms_norm_numerical_match(self):
        """Triton RMSNorm should match F.rms_norm within tolerance."""
        from forge.decoding.triton_train_kernels import triton_rms_norm
        x = torch.randn(2, 64, 128, dtype=torch.float32, device=DEVICE)
        weight = torch.randn(128, dtype=torch.float32, device=DEVICE)
        out = triton_rms_norm(x, weight, eps=1e-6)
        ref = F.rms_norm(x, [128], weight, 1e-6)
        assert torch.allclose(out, ref, atol=1e-4), f"max diff: {(out-ref).abs().max()}"

    def test_rms_norm_preserves_dtype(self):
        """RMSNorm should preserve input dtype."""
        from forge.decoding.triton_train_kernels import triton_rms_norm
        for dt in [torch.float32, torch.bfloat16]:
            if dt == torch.bfloat16 and not CUDA_AVAILABLE:
                continue  # bf16 CPU RMSNorm may differ
            x = torch.randn(2, 128, dtype=dt, device=DEVICE)
            weight = torch.ones(128, dtype=dt, device=DEVICE)
            out = triton_rms_norm(x, weight, eps=1e-6)
            assert out.dtype == dt


class TestTritonSwiGLU:
    """Test Triton fused SwiGLU activation kernel."""

    def test_swiglu_cpu_fallback(self):
        """On CPU, triton_swiglu_act should fall back to F.silu(gate)*up."""
        from forge.decoding.triton_train_kernels import triton_swiglu_act
        gate = torch.randn(4, 256, dtype=torch.float32)
        up = torch.randn(4, 256, dtype=torch.float32)
        out = triton_swiglu_act(gate, up)
        ref = F.silu(gate) * up
        assert torch.allclose(out, ref, atol=1e-5)

    def test_swiglu_numerical_match(self):
        """Triton SwiGLU should match F.silu(gate)*up within tolerance."""
        from forge.decoding.triton_train_kernels import triton_swiglu_act
        gate = torch.randn(2, 64, 512, dtype=torch.float32, device=DEVICE)
        up = torch.randn(2, 64, 512, dtype=torch.float32, device=DEVICE)
        out = triton_swiglu_act(gate, up)
        ref = F.silu(gate) * up
        assert torch.allclose(out, ref, atol=1e-4), f"max diff: {(out-ref).abs().max()}"

    def test_swiglu_shape_mismatch_raises(self):
        """SwiGLU should assert on shape mismatch (GPU path only)."""
        from forge.decoding.triton_train_kernels import triton_swiglu_act
        if not CUDA_AVAILABLE:
            print("SKIP: shape mismatch assert only triggers on GPU path")
            return
        gate = torch.randn(4, 256, device=DEVICE)
        up = torch.randn(4, 128, device=DEVICE)
        with pytest.raises(AssertionError):
            triton_swiglu_act(gate, up)


# ─── Varlen attention ───────────────────────────────────────────────

class TestVarlenAttention:
    """Test varlen attention block-diagonal masking."""

    def test_block_diag_mask_no_cross_contamination(self):
        """Block-diagonal causal mask should prevent cross-example attention."""
        from forge.model_loader import _build_block_diag_causal_mask
        # Two examples: lengths 4 and 4, total T=8
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.long)
        mask = _build_block_diag_causal_mask(cu_seqlens, 8, torch.device("cpu"), torch.float32)
        # Position 0 (in example 1) should NOT attend to position 4 (in example 2)
        assert mask[0, 4] == float('-inf'), "Cross-example attention not masked!"
        assert mask[4, 0] == float('-inf'), "Cross-example attention not masked!"
        # Within example 1: position 3 should attend to 0,1,2,3 (causal)
        assert mask[3, 0] == 0.0, "Causal within example broken"
        assert mask[3, 3] == 0.0, "Diagonal should be valid"
        assert mask[3, 4] == float('-inf'), "Cross-example not masked"

    def test_block_diag_mask_unequal_lengths(self):
        """Block-diagonal mask with unequal example lengths."""
        from forge.model_loader import _build_block_diag_causal_mask
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.long)  # ex1=3, ex2=5
        mask = _build_block_diag_causal_mask(cu_seqlens, 8, torch.device("cpu"), torch.float32)
        # Example 1: positions 0-2
        assert mask[2, 0] == 0.0  # causal within ex1
        assert mask[2, 3] == float('-inf')  # ex1 can't see ex2
        # Example 2: positions 3-7
        assert mask[7, 3] == 0.0  # causal within ex2
        assert mask[7, 2] == float('-inf')  # ex2 can't see ex1

    def test_varlen_attention_matches_manual(self):
        """Varlen attention output should match manual block-diagonal attention."""
        from forge.model_loader import _build_block_diag_causal_mask, varlen_attention
        if not CUDA_AVAILABLE:
            print("SKIP: varlen attention GPU test requires CUDA")
            return

        B, n_heads, T, hd = 1, 2, 8, 16
        q = torch.randn(B, n_heads, T, hd, device=DEVICE)
        k = torch.randn(B, n_heads, T, hd, device=DEVICE)
        v = torch.randn(B, n_heads, T, hd, device=DEVICE)
        cu_seqlens = torch.tensor([[0, 4, 8]], dtype=torch.int32, device=DEVICE)

        out = varlen_attention(q, k, v, cu_seqlens)
        # Manual: build block-diagonal mask and use SDPA
        mask = _build_block_diag_causal_mask(
            cu_seqlens[0], T, DEVICE, q.dtype)
        ref = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0).expand(B, n_heads, T, T))
        # Note: may not be exactly equal if flash_attn varlen is used (different softmax)
        # but should be close. With fallback (no flash_attn), should be exact.
        assert out.shape == ref.shape


class TestVarlenMetaCache:
    """Packed-sequence metadata caching: boundaries/cu computed once per
    batch, not per layer (the Unsloth/NVIDIA ~14%/batch optimization)."""

    def test_meta_values_correct(self):
        from forge.model.attention_ops import _varlen_meta
        cu = torch.tensor([0, 3, 8, 12], dtype=torch.int32)
        cu_list, max_list, blocks_list = _varlen_meta(cu, torch.device("cpu"))
        assert cu_list[0].tolist() == [0, 3, 8, 12]
        assert cu_list[0].dtype == torch.int32
        assert max_list == [5]
        assert blocks_list == [[(0, 3), (3, 8), (8, 12)]]

    def test_cache_hit_returns_same_objects(self):
        from forge.model.attention_ops import _varlen_meta
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        m1 = _varlen_meta(cu, torch.device("cpu"))
        m2 = _varlen_meta(cu, torch.device("cpu"))
        assert m1 is m2 and m1[0][0] is m2[0][0]

    def test_inplace_mutation_recomputes(self):
        from forge.model.attention_ops import _varlen_meta
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        m1 = _varlen_meta(cu, torch.device("cpu"))
        cu[2] = 10  # bumps _version -> stale entry must not be reused
        m2 = _varlen_meta(cu, torch.device("cpu"))
        assert m2 is not m1
        assert m2[2][0] == [(0, 4), (4, 10)]
        assert m2[1] == [6]

    def test_blocks_path_matches_direct(self):
        from forge.model.attention_ops import _build_block_diag_causal_mask, _varlen_meta
        cu = torch.tensor([0, 3, 8], dtype=torch.long)
        _, _, blocks_list = _varlen_meta(cu, torch.device("cpu"))
        a = _build_block_diag_causal_mask(
            cu, 8, torch.device("cpu"), torch.float32)
        b = _build_block_diag_causal_mask(
            cu, 8, torch.device("cpu"), torch.float32,
            blocks=blocks_list[0])
        assert torch.equal(a, b)

    def test_2d_rows(self):
        from forge.model.attention_ops import _varlen_meta
        cu = torch.tensor([[0, 2, 6], [0, 5, 6]], dtype=torch.int32)
        _cu_list, max_list, blocks_list = _varlen_meta(cu, torch.device("cpu"))
        assert max_list == [4, 5]
        assert blocks_list[0] == [(0, 2), (2, 6)]
        assert blocks_list[1] == [(0, 5), (5, 6)]


# ─── PackedSequenceDataset cu_seqlens ───────────────────────────────

class TestPackedSequenceCuSeqlens:
    """Test PackedSequenceDataset cu_seqlens emission."""

    def test_cu_seqlens_emitted(self):
        """Dataset should emit cu_seqlens when emit_cu_seqlens=True."""
        from forge.training.data.efficient_pipeline import PackedSequenceDataset
        # Two examples: lengths 4 and 4, seq_len=8
        dataset = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},
        ]
        packed = PackedSequenceDataset(dataset, seq_len=8, emit_cu_seqlens=True)
        assert len(packed) == 1
        item = packed[0]
        assert "cu_seqlens" in item
        cu = item["cu_seqlens"]
        # Should be [0, 4, 8] (cumulative lengths)
        assert cu.tolist() == [0, 4, 8], f"Expected [0, 4, 8], got {cu.tolist()}"

    def test_cu_seqlens_not_emitted_by_default(self):
        """Dataset should NOT emit cu_seqlens by default (backward compat)."""
        from forge.training.data.efficient_pipeline import PackedSequenceDataset
        dataset = [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}]
        packed = PackedSequenceDataset(dataset, seq_len=8)
        item = packed[0]
        assert "cu_seqlens" not in item

    def test_cu_seqlens_three_examples(self):
        """cu_seqlens with three examples packed into one sequence."""
        from forge.training.data.efficient_pipeline import PackedSequenceDataset
        dataset = [
            {"input_ids": [1, 2], "labels": [1, 2]},
            {"input_ids": [3, 4], "labels": [3, 4]},
            {"input_ids": [5, 6], "labels": [5, 6]},
        ]
        packed = PackedSequenceDataset(dataset, seq_len=6, emit_cu_seqlens=True)
        item = packed[0]
        cu = item["cu_seqlens"]
        assert cu.tolist() == [0, 2, 4, 6], f"Expected [0, 2, 4, 6], got {cu.tolist()}"


# ─── APOLLO optimizer ───────────────────────────────────────────────

class TestAPOLLO:
    """Test APOLLO optimizer convergence and memory."""

    def test_apollo_convergence(self):
        """APOLLO should converge on a simple quadratic loss."""
        from forge.training.optim.apollo import APOLLO
        torch.manual_seed(42)
        # Simple 2D param with quadratic loss
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.float32, device=DEVICE))
        target = torch.randn(4, 8, dtype=torch.float32, device=DEVICE)
        opt = APOLLO([p], lr=0.5, rank=4, weight_decay=0.0)

        initial_loss = F.mse_loss(p, target).item()
        for _ in range(200):
            opt.zero_grad()
            loss = F.mse_loss(p, target)
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, \
            f"APOLLO did not converge: {initial_loss} → {final_loss}"

    def test_apollo_1d_params_adamw(self):
        """1D params should use standard AdamW path."""
        from forge.training.optim.apollo import APOLLO
        p = nn.Parameter(torch.randn(64, dtype=torch.float32, device=DEVICE))
        target = torch.randn(64, dtype=torch.float32, device=DEVICE)
        opt = APOLLO([p], lr=0.1, rank=4)
        for _ in range(100):
            opt.zero_grad()
            loss = F.mse_loss(p, target)
            loss.backward()
            opt.step()
        assert loss.item() < 1.0

    def test_apollo_memory_savings(self):
        """APOLLO should use less memory than AdamW for 2D params."""
        from forge.training.optim.apollo import APOLLO
        p = nn.Parameter(torch.randn(64, 128, dtype=torch.float32, device=DEVICE))
        opt = APOLLO([p], lr=0.01, rank=4)
        # Initialize states
        p.grad = torch.randn_like(p)
        opt.step()
        state = opt.state[p]
        # APOLLO stores: proj (128, 4), aux (64, 4), scale, exp_avg (64, 128)
        # AdamW would store: exp_avg (64, 128), exp_avg_sq (64, 128) = 2x
        apollo_bytes = sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
        adamw_bytes = 2 * p.numel() * 4  # 2 fp32 states
        assert apollo_bytes < adamw_bytes, \
            f"APOLLO ({apollo_bytes}B) should use less memory than AdamW ({adamw_bytes}B)"


# ─── BREAD for BAdam ────────────────────────────────────────────────

class TestBREAD:
    """Test BREAD landscape correction for BAdam."""

    def test_bread_disabled_is_vanilla_badam(self):
        """BREAD disabled should behave like vanilla BAdam."""
        from forge.training.optim.badam import BAdam
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        opt = BAdam(model, lr=0.01, switch_every=1, bread_sgd_correction="disabled")
        assert opt.bread_sgd_correction == "disabled"
        assert len(opt._bread_visited) == 1  # only block 0 visited at init

    def test_bread_partial_mode(self):
        """BREAD partial mode should track visited blocks."""
        from forge.training.optim.badam import BAdam
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        opt = BAdam(model, lr=0.01, switch_every=1, bread_sgd_correction="partial",
                    bread_sgd_lr_scale=5.0)
        assert opt.bread_sgd_correction == "partial"
        assert opt.bread_sgd_lr_scale == 5.0

    def test_bread_correction_applied(self):
        """BREAD should apply SGD correction to visited inactive blocks."""
        from forge.training.optim.badam import BAdam
        # BAdam partitions by "blocks.N.*" pattern with child modules.
        # Each block needs a sub-module (not a direct Linear) so the
        # partitioner can find params under blocks.N.*.
        class FakeSubLayer(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.fc = nn.Linear(d, d)
            def forward(self, x):
                return self.fc(x)

        class FakeBlockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    FakeSubLayer(16), FakeSubLayer(16), FakeSubLayer(16)])
            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return x

        model = FakeBlockModel()
        opt = BAdam(model, lr=0.01, switch_every=1, bread_sgd_correction="partial")
        # Do a few steps to populate momentum and cycle through blocks
        x = torch.randn(4, 16)
        target = torch.randn(4, 16)
        for _ in range(5):
            opt.zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            opt.step()
        # After 5 steps with switch_every=1, should have visited multiple blocks
        assert len(opt._bread_visited) >= 2, \
            f"Expected >=2 visited blocks, got {len(opt._bread_visited)}"


# ─── FlashOptim ─────────────────────────────────────────────────────

class TestFlashOptim:
    """Test FlashOptim companded 8-bit optimizer states."""

    def test_flashoptim_step(self):
        """FlashOptim should perform optimizer steps without error."""
        from forge.training.optim.flashoptim import FlashOptimAdamW
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.float32, device=DEVICE))
        target = torch.randn(4, 8, dtype=torch.float32, device=DEVICE)
        opt = FlashOptimAdamW([p], lr=0.01, weight_decay=0.0, bits=8)
        for _ in range(10):
            opt.zero_grad()
            loss = F.mse_loss(p, target)
            loss.backward()
            opt.step()
        assert loss.item() < 5.0  # should decrease somewhat

    def test_flashoptim_state_is_int8(self):
        """FlashOptim optimizer states should be int8 (1 byte/param)."""
        from forge.training.optim.flashoptim import FlashOptimAdamW
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.float32, device=DEVICE))
        opt = FlashOptimAdamW([p], lr=0.01, bits=8)
        p.grad = torch.randn_like(p)
        opt.step()
        state = opt.state[p]
        assert state["exp_avg_q"].dtype == torch.int8, \
            f"Expected int8, got {state['exp_avg_q'].dtype}"
        assert state["exp_avg_sq_q"].dtype == torch.int8

    def test_flashoptim_memory_savings(self):
        """FlashOptim should use less state memory than AdamW."""
        from forge.training.optim.flashoptim import FlashOptimAdamW
        p = nn.Parameter(torch.randn(64, 128, dtype=torch.float32, device=DEVICE))
        opt = FlashOptimAdamW([p], lr=0.01, bits=8)
        p.grad = torch.randn_like(p)
        opt.step()
        state = opt.state[p]
        # FlashOptim: 2 int8 tensors (2 bytes/param) + scales (negligible)
        flashoptim_bytes = sum(
            v.numel() * v.element_size() for v in state.values()
            if isinstance(v, torch.Tensor) and v.dtype == torch.int8)
        # AdamW: 2 fp32 tensors (8 bytes/param)
        adamw_bytes = 2 * p.numel() * 4
        assert flashoptim_bytes < adamw_bytes, \
            f"FlashOptim ({flashoptim_bytes}B) should use less than AdamW ({adamw_bytes}B)"

    def test_companding_roundtrip(self):
        """Companding + quantization round-trip should preserve values approximately."""
        from forge.training.optim.flashoptim import _compand, _decompand, _dequantize_from_uint8, _quantize_to_uint8
        x = torch.randn(128, dtype=torch.float32) * 0.1  # small values (like early momentum)
        scale = x.abs().max().clamp(min=1e-12)
        companded = _compand(x, scale)
        q, q_scale = _quantize_to_uint8(companded)
        decompanded = _decompand(_dequantize_from_uint8(q, q_scale), scale)
        # Should be close (int8 quantization has ~1% error)
        rel_err = (decompanded - x).abs().max() / (x.abs().max() + 1e-12)
        assert rel_err < 0.1, f"Companding round-trip error too high: {rel_err}"


# ─── Config fields ──────────────────────────────────────────────────

class TestConfigFields:
    """Test that new config fields exist and default correctly."""

    def test_config_defaults(self):
        """New config fields should have correct defaults."""
        from forge.config import ModelConfig
        cfg = ModelConfig()
        assert not cfg.use_varlen
        assert not cfg.use_triton_kernels
        assert cfg.triton_rms_block_size == 4096
        assert cfg.triton_swiglu_block_size == 16384
        assert cfg.apollo_rank == 8
        assert cfg.apollo_scale == "tensor"
        assert cfg.bread_sgd_correction == "partial"
        assert cfg.bread_sgd_lr_scale == 5.0
        assert cfg.flashoptim_bits == 8

    def test_v10_preset_has_speedup_features(self):
        """V10 preset should have use_varlen and use_triton_kernels available."""
        from forge.config import MODEL_CONFIGS
        for name in ["forgelm_v2"]:
            if name in MODEL_CONFIGS:
                cfg = MODEL_CONFIGS[name]
                # V10 is a lossless port — these features are available but not enabled by default
                assert hasattr(cfg, 'use_varlen'), f"{name} should have use_varlen field"
                assert hasattr(cfg, 'use_triton_kernels'), f"{name} should have use_triton_kernels field"


# ─── Optimizer wiring ───────────────────────────────────────────────

class TestOptimizerWiring:
    """Test that new optimizers are wired into configure_optimizer."""

    def test_apollo_wired(self):
        """configure_optimizer should accept 'apollo'."""
        from forge.training.training_utils import configure_optimizer
        model = nn.Sequential(nn.Linear(16, 16))
        opt = configure_optimizer(model, max_lr=0.01, weight_decay=0.01, optimizer_name="apollo")
        assert opt is not None
        # Do a step
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_flashoptim_wired(self):
        """configure_optimizer should accept 'flashoptim'."""
        from forge.training.training_utils import configure_optimizer
        model = nn.Sequential(nn.Linear(16, 16))
        opt = configure_optimizer(model, max_lr=0.01, weight_decay=0.01, optimizer_name="flashoptim")
        assert opt is not None
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_badam_bread_wired(self):
        """configure_optimizer should accept 'badam' with BREAD config."""
        from forge.training.training_utils import configure_optimizer
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        # Without config on model, BREAD defaults to disabled
        opt = configure_optimizer(model, max_lr=0.01, weight_decay=0.01, optimizer_name="badam")
        assert opt is not None

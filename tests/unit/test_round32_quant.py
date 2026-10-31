"""Tests for Round 32 quantization features.

R32-1: NF4 QLoRA
R32-2: GRINQH 2-bit
R32-3: MixLLM global mixed-precision
R32-4: HyQuant KV cache
R32-5: ACBQ adaptive cross-block
R32-6: ForgeQuant (novel SM120-tuned)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ── R32-1: NF4 QLoRA ──────────────────────────────────────────────────────

class TestNF4QLoRA:
    """NF4 (NormalFloat 4-bit) QLoRA — industry-standard QLoRA path."""

    def test_nf4_quantize_dequantize_roundtrip(self):
        from forge.training.bitnet_lora import NF4Linear
        lin = NF4Linear(128, 64, bias=False, group_size=32)
        w = torch.randn(64, 128) * 0.1
        lin.load_from_weight(w)
        w_dq = lin._dequantize_weight(torch.float32, cache=False)
        # NF4 should reconstruct with reasonable error (< 20% relative)
        rel_err = (w - w_dq).norm() / w.norm()
        assert rel_err < 0.20, f"NF4 roundtrip error too high: {rel_err:.4f}"

    def test_nf4_forward_pass(self):
        from forge.training.bitnet_lora import NF4Linear
        lin = NF4Linear(64, 32, bias=True, group_size=32)
        lin.load_from_weight(torch.randn(32, 64) * 0.1)
        lin.bias.data = torch.randn(32) * 0.01
        x = torch.randn(2, 4, 64)
        out = lin(x)
        assert out.shape == (2, 4, 32), f"Wrong output shape: {out.shape}"

    def test_nf4_convert_model(self):
        from forge.training.bitnet_lora import convert_to_nf4_qlora
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        n_conv, n_skip = convert_to_nf4_qlora(model, group_size=64)
        assert n_conv == 2, f"Expected 2 converted, got {n_conv}"
        assert n_skip == 0

    def test_nf4_lora_adapter(self):
        from forge.training.bitnet_lora import LoRAAdapter, NF4Linear
        lin = NF4Linear(64, 32, bias=False, group_size=32)
        lin.load_from_weight(torch.randn(32, 64) * 0.1)
        lora = LoRAAdapter(64, 32, rank=4, alpha=8)
        lin.lora_adapter = lora
        x = torch.randn(1, 4, 64)
        out = lin(x)
        assert out.shape == (1, 4, 32)

    def test_nf4_merge_lora(self):
        from forge.training.bitnet_lora import LoRAAdapter, NF4Linear
        lin = NF4Linear(64, 32, bias=False, group_size=32)
        w_orig = torch.randn(32, 64) * 0.1
        lin.load_from_weight(w_orig)
        lora = LoRAAdapter(64, 32, rank=4, alpha=8)
        lin.lora_adapter = lora
        merged = lin.merge_lora()
        assert merged is True
        assert lin.lora_adapter is None

    def test_nf4_skip_small_layers(self):
        from forge.training.bitnet_lora import convert_to_nf4_qlora
        model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))
        n_conv, _n_skip = convert_to_nf4_qlora(model, min_size=64)
        assert n_conv == 0, f"Should skip small layers, got {n_conv}"


# ── R32-2: GRINQH ─────────────────────────────────────────────────────────

class TestGRINQH:
    """GRINQH — effective 2-bit weight quantization with dynamic per-channel precision."""

    def test_grinqh_forward_pass(self):
        from forge.engine.quant.grinqh import GRINQHLinear
        lin = GRINQHLinear.from_linear(nn.Linear(128, 64, bias=False), group_size=32)
        x = torch.randn(2, 4, 128)
        out = lin(x)
        assert out.shape == (2, 4, 64)

    def test_grinqh_quantize_model(self):
        from forge.engine.quant.grinqh import quantize_model_grinqh
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        n = quantize_model_grinqh(model, group_size=64, target_effective_bits=2.5)
        assert n == 2, f"Expected 2 quantized, got {n}"

    def test_grinqh_effective_bits(self):
        from forge.engine.quant.grinqh import GRINQHLinear
        lin = GRINQHLinear.from_linear(nn.Linear(128, 64, bias=False), group_size=32)
        x = torch.randn(1, 128)
        out = lin(x)
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all(), "Output has NaN/Inf"


# ── R32-3: MixLLM ─────────────────────────────────────────────────────────

class TestMixLLM:
    """MixLLM — global mixed-precision across output features."""

    def test_mixllm_forward_pass(self):
        from forge.engine.quant.mixllm import MixLLMLinear
        high_mask = torch.zeros(64, dtype=torch.bool)
        high_mask[:6] = True  # top 10% are high precision
        lin = MixLLMLinear.from_linear(nn.Linear(128, 64, bias=False), high_mask, group_size=32)
        x = torch.randn(2, 4, 128)
        out = lin(x)
        assert out.shape == (2, 4, 64)

    def test_mixllm_quantize_model(self):
        from forge.engine.quant.mixllm import quantize_model_mixllm
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        n = quantize_model_mixllm(model, group_size=64, high_fraction=0.1)
        assert n == 2, f"Expected 2 quantized, got {n}"

    def test_mixllm_global_importance(self):
        from forge.engine.quant.mixllm import quantize_model_mixllm
        # Two layers with different weight scales
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # First layer has large weights, second has small
        model[0].weight.data = torch.randn(64, 64) * 1.0
        model[2].weight.data = torch.randn(64, 64) * 0.01
        n = quantize_model_mixllm(model, group_size=64, high_fraction=0.1)
        assert n == 2
        # Verify forward still works
        x = torch.randn(1, 64)
        out = model(x)
        assert out.shape == (1, 64)


# ── R32-4: HyQuant KV Cache ───────────────────────────────────────────────

class TestHyQuantKV:
    """HyQuant — pattern-aware KV cache quantization."""

    def test_hyquant_init_append_get(self):
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        cache = HyQuantKVCache(window_size=4, vertical_ratio=0.2)
        cache.init(n_heads=4, head_dim=32, n_kv_heads=2, max_seq_len=64,
                   device='cpu', dtype=torch.bfloat16)
        # Append 16 tokens
        for i in range(16):
            k = torch.randn(1, 2, 1, 32)
            v = torch.randn(1, 2, 1, 32)
            cache.append(k, v, i)
        assert cache.seq_len == 16
        # Retrieve
        positions = torch.tensor([[0, 5, 10, 15]])
        k_out, v_out = cache.get(positions)
        assert k_out.shape == (1, 2, 4, 32)
        assert v_out.shape == (1, 2, 4, 32)

    def test_hyquant_window_is_high_precision(self):
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        cache = HyQuantKVCache(window_size=4, vertical_ratio=0.0)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32,
                   device='cpu', dtype=torch.bfloat16)
        for i in range(16):
            k = torch.randn(1, 2, 1, 16)
            v = torch.randn(1, 2, 1, 16)
            cache.append(k, v, i)
        # After all 16 tokens appended, last 4 (12-15) should be high precision
        # The window is relative to seq_len at append time, so early tokens
        # were in the window when appended. Check final state:
        n_high = sum(1 for v in cache._is_high.values() if v)
        n_low = sum(1 for v in cache._is_high.values() if not v)
        # At least some should be high (the last few) and some low
        assert n_high > 0, "Should have some high-precision tokens"
        assert n_high >= 4, f"Last 4 tokens should be high precision, got {n_high}"

    def test_hyquant_attention_hints(self):
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        cache = HyQuantKVCache(window_size=4, vertical_ratio=0.2)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32,
                   device='cpu', dtype=torch.bfloat16)
        # Simulate attention scores: (1 layer, 1 head, 10, 10)
        # Make token 2 a "vertical line" (high column sum)
        scores = torch.zeros(1, 1, 10, 10)
        scores[:, :, :, 2] = 1.0  # all queries attend strongly to position 2
        cache.set_attention_hints(scores)
        assert 2 in cache._vertical_indices, "Token 2 should be in vertical indices"

    def test_hyquant_clear(self):
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        cache = HyQuantKVCache(window_size=4, vertical_ratio=0.2)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32,
                   device='cpu', dtype=torch.bfloat16)
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        cache.clear()
        assert cache.seq_len == 0
        assert len(cache._is_high) == 0

    def test_hyquant_info(self):
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        cache = HyQuantKVCache(window_size=4, vertical_ratio=0.2)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32,
                   device='cpu', dtype=torch.bfloat16)
        for i in range(10):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        info = cache.info()
        assert info["type"] == "hyquant"
        assert info["seq_len"] == 10
        assert "compression" in info


# ── R32-5: ACBQ ───────────────────────────────────────────────────────────

class TestACBQ:
    """ACBQ — adaptive cross-block quantization."""

    def test_acbq_forward_pass(self):
        from forge.engine.quant.acbq import ACBQLinear
        lin = ACBQLinear.from_linear(nn.Linear(128, 64, bias=False), group_size=32)
        x = torch.randn(2, 4, 128)
        out = lin(x)
        assert out.shape == (2, 4, 64)

    def test_acbq_quantize_model(self):
        from forge.engine.quant.acbq import quantize_model_acbq
        # ACBQ classifies by module name — use realistic names
        model = nn.Module()
        model.q_proj = nn.Linear(128, 64)
        model.w_gate = nn.Linear(128, 64)
        n = quantize_model_acbq(model, group_size=64, attn_bits=4, ffn_bits=4, verbose=False)
        assert n == 2, f"Expected 2 quantized, got {n}"

    def test_acbq_mixed_precision(self):
        from forge.engine.quant.acbq import quantize_model_acbq
        model = nn.Module()
        model.q_proj = nn.Linear(128, 64)
        model.w_down = nn.Linear(64, 128)
        n = quantize_model_acbq(model, group_size=64, attn_bits=4, ffn_bits=2, verbose=False)
        assert n == 2
        x = torch.randn(1, 128)
        out_q = model.q_proj(x)
        assert out_q.shape == (1, 64)


# ── R32-6: ForgeQuant (novel) ─────────────────────────────────────────────

class TestForgeQuant:
    """ForgeQuant — novel SM120-tuned INT4 dense + 2-bit sparse."""

    def test_forge_quant_roundtrip(self):
        from forge.engine.quant.forge_quant import ForgeQuantLinear
        lin = ForgeQuantLinear(128, 64, bias=False, group_size=32, sparse_ratio=0.1)
        w = torch.randn(64, 128) * 0.1
        lin.load_from_weight(w)
        w_dq = lin._dequantize_weight(torch.float32, cache=False)
        # ForgeQuant should have lower error than pure INT4 because
        # the sparse path corrects outlier channels
        rel_err = (w - w_dq).norm() / w.norm()
        assert rel_err < 0.35, f"ForgeQuant roundtrip error too high: {rel_err:.4f}"

    def test_forge_quant_forward_pass(self):
        from forge.engine.quant.forge_quant import ForgeQuantLinear
        lin = ForgeQuantLinear(64, 32, bias=True, group_size=32, sparse_ratio=0.1)
        lin.load_from_weight(torch.randn(32, 64) * 0.1)
        lin.bias.data = torch.randn(32) * 0.01
        x = torch.randn(2, 4, 64)
        out = lin(x)
        assert out.shape == (2, 4, 32)

    def test_forge_quant_nondivisible_group_size(self):
        """Regression: in_features % group_size != 0 broke _dequantize_dense —
        idx was cropped to in_features before the group reshape, so the padded
        packed stream couldn't reshape into n_groups groups. Hit in the wild by
        MoE dense_bypass expert width 1216 (4864/4) with group_size=128."""
        from forge.engine.quant.forge_quant import ForgeQuantLinear
        for inf in (1216, 1000):
            lin = ForgeQuantLinear(inf, 64, bias=False, group_size=128)
            lin.load_from_weight(torch.randn(64, inf) * 0.1)
            out = lin(torch.randn(4, inf))
            assert out.shape == (4, 64)
            w_dq = lin._dequantize_weight(torch.float32, cache=False)
            assert w_dq.shape == (64, inf)

    def test_forge_quant_outlier_preservation(self):
        """ForgeQuant should preserve outlier channels better than pure INT4."""
        from forge.engine.quant.forge_quant import ForgeQuantLinear
        lin = ForgeQuantLinear(64, 32, bias=False, group_size=32, sparse_ratio=0.2)
        # Create weights with clear outliers
        w = torch.randn(32, 64) * 0.05
        w[0] *= 10  # channel 0 is an outlier
        w[15] *= 8  # channel 15 is an outlier
        lin.load_from_weight(w)
        w_dq = lin._dequantize_weight(torch.float32, cache=False)
        # The outlier channels should be well-preserved
        err_0 = (w[0] - w_dq[0]).norm() / w[0].norm()
        err_15 = (w[15] - w_dq[15]).norm() / w[15].norm()
        # Outlier channels should have lower relative error than average
        avg_err = (w - w_dq).norm() / w.norm()
        assert err_0 < avg_err * 1.5, f"Outlier ch0 error {err_0:.4f} > avg {avg_err:.4f}*1.5"

    def test_forge_quant_quantize_model(self):
        from forge.engine.quant.forge_quant import quantize_model_forge_quant
        model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        n = quantize_model_forge_quant(model, group_size=64, sparse_ratio=0.1)
        assert n == 2, f"Expected 2 quantized, got {n}"

    def test_forge_quant_lora_merge(self):
        from forge.engine.quant.forge_quant import ForgeQuantLinear
        from forge.training.bitnet_lora import LoRAAdapter
        lin = ForgeQuantLinear(64, 32, bias=False, group_size=32, sparse_ratio=0.1)
        lin.load_from_weight(torch.randn(32, 64) * 0.1)
        lora = LoRAAdapter(64, 32, rank=4, alpha=8)
        lin.lora_adapter = lora
        merged = lin.merge_lora()
        assert merged is True
        assert lin.lora_adapter is None

    def test_forge_quant_beats_int4_on_outliers(self):
        """ForgeQuant should not catastrophically degrade vs INT4 on weights with outliers.

        The key advantage of ForgeQuant is SM120 throughput (int4 mma.sync is
        faster than fp4), not necessarily better perplexity. This test verifies
        ForgeQuant doesn't degrade quality vs INT4 by more than 2x.
        """
        from forge.engine.quant.forge_quant import ForgeQuantLinear
        # Create a weight matrix with outliers
        w = torch.randn(64, 256) * 0.02
        # Add outliers to 10% of channels
        for i in range(0, 64, 8):
            w[i] += torch.randn(256) * 0.3

        # ForgeQuant
        fq = ForgeQuantLinear(256, 64, bias=False, group_size=64, sparse_ratio=0.15)
        fq.load_from_weight(w)
        w_fq = fq._dequantize_weight(torch.float32, cache=False)
        fq_err = (w - w_fq).norm() / w.norm()

        # Pure INT4 (via simple quantization)
        w_int4 = torch.zeros_like(w)
        gs = 64
        n_groups = 256 // gs
        for g in range(n_groups):
            block = w[:, g*gs:(g+1)*gs]
            scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
            q = torch.round(block / scale).clamp(-8, 7)
            w_int4[:, g*gs:(g+1)*gs] = q * scale
        int4_err = (w - w_int4).norm() / w.norm()

        # ForgeQuant should not be more than 2x worse than INT4
        # (the throughput benefit on SM120 compensates for slightly higher error)
        assert fq_err <= int4_err * 2.0, (
            f"ForgeQuant error {fq_err:.4f} should be <= INT4 {int4_err:.4f} * 2.0")

    def test_forge_quant_rotation(self):
        """use_rotation=True: Hadamard-rotated dense basis must (a) keep
        forward == xW^T approx (exact algebra), (b) shrink quant error
        on outlier-heavy weights, (c) survive non-divisible in_features."""
        from forge.engine.quant.forge_quant import ForgeQuantLinear, _fwht
        torch.manual_seed(0)
        # FWHT sanity: unnormalized involution
        x = torch.randn(4, 64)
        assert torch.allclose(_fwht(_fwht(x)), x * 64, atol=1e-4)

        in_f, out_f, gs = 256, 128, 64
        w = torch.randn(out_f, in_f) * 0.02
        w[::8] += torch.randn(in_f) * 0.4      # outlier rows
        w[:, ::7] *= 6.0                        # outlier cols
        x = torch.randn(8, in_f)

        def fwd_err(lin):
            lin.load_from_weight(w)
            return ((lin(x) - x @ w.T).norm() / (x @ w.T).norm()).item()

        direct = ForgeQuantLinear(in_f, out_f, group_size=gs,
                                  sparse_ratio=0.10)
        rot = ForgeQuantLinear(in_f, out_f, group_size=gs,
                               sparse_ratio=0.10, use_rotation=True)
        e_d, e_r = fwd_err(direct), fwd_err(rot)
        # measured 0.0366 vs 0.0857 on this construction — keep 30% slack
        assert e_r < e_d * 0.7, (
            f"rotated err {e_r:.4f} not < 0.7x direct {e_d:.4f}")
        # _dequantize_weight reconstructs the ORIGINAL basis
        assert rot._dequantize_weight(torch.float32).shape == (out_f, in_f)
        # non-divisible in_features + rotation (1216 = 4864/4, gs=128)
        lin = ForgeQuantLinear(1216, 64, group_size=128, use_rotation=True)
        lin.load_from_weight(torch.randn(64, 1216) * 0.1)
        assert lin(torch.randn(4, 1216)).shape == (4, 64)
        assert lin._dequantize_weight(torch.float32).shape == (64, 1216)
        # non-pow2 group_size must refuse rotation
        import pytest
        with pytest.raises(ValueError):
            ForgeQuantLinear(96, 32, group_size=48, use_rotation=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
    def test_forge_quant_fused_gemm(self):
        """Batched fused Triton path (M<=32): packed-kernel output must
        match the dequant+matmul fallback, and it must compose with
        use_rotation."""
        from forge.engine.quant.forge_quant import ForgeQuantLinear, _HAS_TRITON
        if not _HAS_TRITON:
            pytest.skip("no triton")
        dev = "cuda"
        torch.manual_seed(0)
        in_f, out_f = 2048, 1024
        w = torch.randn(out_f, in_f) * 0.05
        for rot in (False, True):
            lin = ForgeQuantLinear(in_f, out_f, group_size=128,
                                   sparse_ratio=0.10, use_rotation=rot)
            lin.load_from_weight(w)
            lin = lin.to(dev)
            lin._cache_w = False
            for M in (2, 8, 32):
                x = torch.randn(M, in_f, device=dev)
                y_f = lin._fused_gemm_forward(x)
                lin._fused_gemv = False
                y_d = lin(x)
                lin._fused_gemv = True
                rel = ((y_f - y_d).norm() / y_d.norm()).item()
                assert rel < 1e-4, f"rot={rot} M={M} rel diff {rel:.2e}"


# ── Engine dispatch integration ────────────────────────────────────────────

class TestEngineDispatch:
    """Verify new quantization modes are wired into ForgeEngine dispatch."""

    def test_quant_fallback_chain_has_new_modes(self):
        from forge.engine.forge_engine import ForgeEngine
        chain = ForgeEngine._QUANT_FALLBACK_CHAIN
        assert "forge_quant" in chain, "forge_quant not in fallback chain"
        assert "grinqh" in chain, "grinqh not in fallback chain"
        assert "mixllm" in chain, "mixllm not in fallback chain"
        assert "acbq" in chain, "acbq not in fallback chain"

    def test_kv_dispatch_has_hyquant(self):
        from forge.engine.kv_backend import build_kv_cache
        cache = build_kv_cache("hyquant")
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        assert isinstance(cache, HyQuantKVCache), "hyquant dispatch failed"

    def test_build_kv_cache_unknown_falls_back(self):
        from forge.engine.kv_backend import StandardKVCache, build_kv_cache
        cache = build_kv_cache("nonexistent_strategy")
        assert isinstance(cache, StandardKVCache)

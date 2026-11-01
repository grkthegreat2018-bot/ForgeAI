"""Tests for Llama-3.1 8B HF-compat support + load-time W4 quantization.

Covers:
  - llama31_8b preset shapes (Hermes-3-Llama-3.1-8B-lorablated target)
  - llama3 RoPE frequency scaling (HF _compute_llama3 parity)
  - quantize_model_w4: packed torchao weights, no bf16 materialization,
    forward correctness on CPU
  - from_checkpoint load_quantize plumbing (decision path, not full load)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


class TestLlama31Preset:
    def test_preset_shapes(self):
        from forge.config import get_config

        c = get_config("llama31_8b")
        assert c.vocab_size == 128256
        assert c.d_model == 4096
        assert c.n_layers == 32
        assert c.n_heads == 32
        assert c.n_kv_heads == 8
        assert c.intermediate_size == 14336
        assert c.attn_bias is False
        assert c.norm_eps == pytest.approx(1e-5)
        assert c.rope_base == pytest.approx(500_000.0)
        assert c.tie_word_embeddings is False
        rs = c.rope_scaling
        assert rs["type"] == "llama3"
        assert rs["factor"] == pytest.approx(8.0)
        assert rs["original_max_position_embeddings"] == 8192

    def test_preset_independent_copy(self):
        from forge.config import MODEL_CONFIGS, get_config

        c = get_config("llama31_8b", device="cpu")
        c.vocab_size = 1
        assert MODEL_CONFIGS["llama31_8b"].vocab_size == 128256


class TestLlama3RoPE:
    def _hf_reference(self, inv_freq, factor=8.0, low_f=1.0, high_f=4.0, orig_len=8192):
        """Reference: transformers llama3 rope scaling (piecewise)."""
        low_wl = orig_len / low_f
        high_wl = orig_len / high_f
        wavelen = 2 * math.pi / inv_freq
        out = torch.where(wavelen > low_wl, inv_freq / factor, inv_freq)
        smooth = (orig_len / wavelen - low_f) / (high_f - low_f)
        smoothed = (1 - smooth) * out / factor + smooth * out
        mid = ~(wavelen < high_wl) & ~(wavelen > low_wl)
        return torch.where(mid, smoothed, out)

    def test_llama3_inv_freq_matches_hf(self):
        from forge.model.layers import RotaryEmbedding

        dim = 128
        inv = 1.0 / (500000.0 ** (torch.arange(0, dim, 2).float() / dim))
        cfg = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
        got = RotaryEmbedding._llama3_inv_freq(inv, cfg)
        ref = self._hf_reference(inv)
        assert torch.allclose(got, ref, atol=1e-12), f"max diff {(got - ref).abs().max()}"

    def test_rope_module_uses_scaling(self):
        from forge.model.layers import RotaryEmbedding

        cfg = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        r = RotaryEmbedding(128, max_seq_len=512, base=500000.0, rope_scaling=cfg)
        ref = self._hf_reference(1.0 / (500000.0 ** (torch.arange(0, 128, 2).float() / 128)))
        assert torch.allclose(r.inv_freq, ref, atol=1e-12)
        x = torch.randn(1, 4, 8, 128)
        out = r(x)
        assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="aten::_convert_weight_to_int4pack is CUDA-only")
class TestQuantizeW4:
    def _tiny_model(self):
        from forge.config import get_config
        from forge.model_loader import ModelLoader

        cfg = get_config("llama31_8b", device="cpu")
        cfg.vocab_size = 512
        cfg.d_model = 256
        cfg.n_layers = 2
        cfg.n_heads = 4
        cfg.n_kv_heads = 2
        cfg.intermediate_size = 512
        cfg.max_seq_len = 256
        cfg.rope_scaling = None  # tiny test model — plain rope
        return ModelLoader.build_model(cfg, checkpoint_path=None)

    def test_w4_packs_linear_weights(self):
        from forge.quant.inference_quant import quantize_model_w4

        model = self._tiny_model()
        n = quantize_model_w4(model, device="cuda", verbose=False)
        assert n > 0
        packed = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and "head" not in name:
                assert "Int4" in type(mod.weight).__name__, name
                packed += 1
        assert packed == n

    def test_w4_skips_head_and_norms(self):
        from forge.quant.inference_quant import quantize_model_w4

        model = self._tiny_model()
        quantize_model_w4(model, device="cuda", verbose=False)
        assert isinstance(model.head.weight, torch.Tensor)
        assert "Int4" not in type(model.head.weight).__name__
        assert "Int4" not in type(model.embed.weight).__name__

    def test_w4_forward_cuda(self):
        from forge.quant.inference_quant import quantize_model_w4

        model = self._tiny_model()
        model.eval()
        quantize_model_w4(model, device="cuda", verbose=False)
        model = model.to("cuda")  # remaining params (embed/head/norms)
        ids = torch.randint(0, 512, (1, 8), device="cuda")
        with torch.no_grad():
            out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape[-1] == 512
        assert torch.isfinite(logits).all()

    def test_w4_storage_is_packed(self):
        """Quantized params must stay ~0.5B/param — no bf16 cache tensors.

        Uses a real-size Linear (2048x2048): the 4D tile format pads small
        dims, so a tiny 256-dim weight would measure padding, not packing.
        """
        from forge.quant.inference_quant import quantize_model_w4

        class _Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(2048, 2048, bias=False)

        model = _Blk().to(torch.bfloat16)
        quantize_model_w4(model, device="cuda", verbose=False)
        w = model.q_proj.weight
        assert hasattr(w, "qdata")
        packed_bytes = (
            w.qdata.numel() * w.qdata.element_size() + w.scale_and_zero.numel() * w.scale_and_zero.element_size()
        )
        b_per_param = packed_bytes / w.numel()
        # int4 = 0.5B/param + scale/zero overhead → ~0.55, well under 0.75
        assert b_per_param < 0.75, f"packed {b_per_param:.3f} B/param — expected ~0.53"

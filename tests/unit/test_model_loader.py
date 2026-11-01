"""Tests for forge.model_loader — core components and forward pass.

These tests run on CPU with the tiny_test config to stay fast and GPU-independent.
"""

import pytest
import torch
import torch.nn as nn

from forge.config import ModelConfig
from forge.model_loader import (
    ConfigurableResearchLLM,
    PreAllocatedKVCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFFN,
    build_attention,
    build_ffn,
    flash_attention,
)

# ── RMSNorm ──────────────────────────────────────────────────────────────────


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_weight_init_is_ones(self):
        norm = RMSNorm(32)
        assert torch.allclose(norm.weight, torch.ones(32))

    def test_normalizes_input(self):
        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(1, 1, 64) * 10  # large variance
        out = norm(x)
        # RMS of output should be close to 1 (before weight scaling)
        rms = out.pow(2).mean().sqrt()
        assert abs(rms.item() - 1.0) < 0.1

    def test_preserves_dtype(self):
        norm = RMSNorm(64)
        x = torch.randn(1, 10, 64, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_zero_input(self):
        norm = RMSNorm(16)
        x = torch.zeros(1, 1, 16)
        out = norm(x)
        # With eps, zero input should produce zero output (0 * rsqrt(eps) * weight = 0)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


# ── RotaryEmbedding ──────────────────────────────────────────────────────────


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        x = torch.randn(1, 4, 10, 64)
        out = rope(x)
        assert out.shape == x.shape

    def test_rotation_preserves_norm(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(1, 1, 5, 32)
        out = rope(x)
        # RoPE is a rotation — norms should be preserved
        x_norm = x.pow(2).sum(-1).sqrt()
        out_norm = out.pow(2).sum(-1).sqrt()
        assert torch.allclose(x_norm, out_norm, atol=1e-5)

    def test_offset_zero(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(1, 1, 3, 32)
        out = rope(x, offset=0)
        assert out.shape == x.shape

    def test_offset_nonzero(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(1, 1, 3, 32)
        out = rope(x, offset=10)
        assert out.shape == x.shape

    def test_yarn_scaling(self):
        scaling = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}
        rope = RotaryEmbedding(dim=32, max_seq_len=128, rope_scaling=scaling)
        x = torch.randn(1, 1, 5, 32)
        out = rope(x)
        assert out.shape == x.shape

    def test_rotate_half(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rotated = RotaryEmbedding._rotate_half(x)
        # _rotate_half([a,b,c,d]) = [-c,-d,a,b]
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(rotated, expected)


# ── SwiGLUFFN ────────────────────────────────────────────────────────────────


class TestSwiGLUFFN:
    def test_output_shape(self):
        ffn = SwiGLUFFN(d_model=128, hidden_dim=256)
        x = torch.randn(2, 10, 128)
        out = ffn(x)
        assert out.shape == (2, 10, 128)

    def test_default_hidden_dim(self):
        ffn = SwiGLUFFN(d_model=768)
        # Default: 8 * d_model / 3
        assert ffn.w_gate.out_features == int(8 * 768 / 3)

    def test_no_bias(self):
        ffn = SwiGLUFFN(d_model=64, hidden_dim=128)
        assert ffn.w_gate.bias is None
        assert ffn.w_up.bias is None
        assert ffn.w_down.bias is None

    def test_zero_input_zero_output(self):
        ffn = SwiGLUFFN(d_model=64, hidden_dim=128)
        x = torch.zeros(1, 5, 64)
        out = ffn(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ── build_attention / build_ffn ──────────────────────────────────────────────


class TestBuildFunctions:
    def test_build_attention_gqa(self):
        cfg = ModelConfig(d_model=128, n_heads=4, attn_type="gqa", max_seq_len=64, n_kv_heads=2)
        attn = build_attention(cfg)
        assert attn is not None

    def test_build_attention_unknown_raises(self):
        cfg = ModelConfig(d_model=128, n_heads=4, attn_type="nonexistent", max_seq_len=64)
        with pytest.raises(ValueError, match="Unknown attention type"):
            build_attention(cfg)

    def test_build_ffn_swiglu(self):
        cfg = ModelConfig(d_model=128, ffn_type="swiglu")
        ffn = build_ffn(cfg)
        assert isinstance(ffn, SwiGLUFFN)

    def test_build_ffn_unknown_raises(self):
        cfg = ModelConfig(d_model=128, ffn_type="nonexistent")
        with pytest.raises(ValueError, match="Unknown FFN type"):
            build_ffn(cfg)


# ── ConfigurableResearchLLM ──────────────────────────────────────────────────


class TestConfigurableResearchLLM:
    @pytest.fixture
    def model(self, tiny_config):
        """Builds model with tiny vocab (256) on CPU — avoids 38M param embedding spike."""
        return ConfigurableResearchLLM(tiny_config)

    def test_model_creation(self, model):
        assert isinstance(model, nn.Module)
        assert model.config.d_model == 128
        assert model.config.n_layers == 4

    def test_weight_tying(self, model):
        assert model.embed.weight is model.head.weight

    def test_forward_returns_logits(self, model):
        idx = torch.randint(0, model.config.vocab_size, (1, 5))
        logits, loss = model(idx)
        assert logits.shape == (1, 5, model.config.vocab_size)
        assert loss is None

    def test_forward_with_targets_returns_loss(self, model):
        idx = torch.randint(0, model.config.vocab_size, (1, 5))
        targets = torch.randint(0, model.config.vocab_size, (1, 5))
        logits, loss = model(idx, targets=targets)
        assert logits is not None
        assert loss is not None
        assert loss.dim() == 0  # scalar loss
        assert loss.item() > 0

    def test_forward_with_cache(self, model):
        idx = torch.randint(0, model.config.vocab_size, (1, 5))
        _logits, _loss, presents = model(idx, use_cache=True)
        assert len(presents) == model.config.n_layers
        # Conv layers return None; attention layers return (k, v)
        for p in presents:
            if p is not None:
                k_cache, v_cache = p
                assert k_cache is not None
                assert v_cache is not None

    def test_forward_return_hidden(self, model):
        idx = torch.randint(0, model.config.vocab_size, (1, 5))
        result = model(idx, return_hidden=True)
        assert len(result) == 3  # logits, loss, hidden
        _logits, _loss, hidden = result
        assert hidden.shape == (1, 5, model.config.d_model)

    def test_forward_cache_then_return_hidden(self, model):
        idx = torch.randint(0, model.config.vocab_size, (1, 5))
        result = model(idx, use_cache=True, return_hidden=True)
        assert len(result) == 4  # logits, loss, presents, hidden

    def test_gradient_checkpointing_toggle(self, model):
        model.enable_gradient_checkpointing()
        for block in model.blocks:
            assert block._gradient_checkpointing is True
        model.disable_gradient_checkpointing()
        for block in model.blocks:
            assert block._gradient_checkpointing is False

    def test_zero_init_residual(self, tiny_config):
        cfg = ModelConfig(**{**tiny_config.__dict__, "zero_init_residual": True})
        model = ConfigurableResearchLLM(cfg)
        # With zero-init, attn.out_proj and ffn.w_down should be zero
        for block in model.blocks:
            if hasattr(block.attn, "out_proj"):
                assert torch.allclose(block.attn.out_proj.weight, torch.zeros_like(block.attn.out_proj.weight))
            if hasattr(block.ffn, "w_down"):
                assert torch.allclose(block.ffn.w_down.weight, torch.zeros_like(block.ffn.w_down.weight))

    def test_draft_head_is_none(self, tiny_config):
        """Draft head is always None (EAGLE removed, MTP is separate module)."""
        cfg = ModelConfig(**{**tiny_config.__dict__, "enable_draft_head": True})
        model = ConfigurableResearchLLM(cfg)
        assert model.draft_head is None

    def test_no_draft_head_by_default(self, model):
        assert model.draft_head is None

    def test_backward_pass(self, model):
        """Ensure gradients flow through the model."""
        idx = torch.randint(0, model.config.vocab_size, (1, 5))
        targets = torch.randint(0, model.config.vocab_size, (1, 5))
        _logits, loss = model(idx, targets=targets)
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


# ── flash_attention ──────────────────────────────────────────────────────────


class TestFlashAttention:
    def test_output_shape(self):
        q = torch.randn(1, 4, 10, 64)
        k = torch.randn(1, 4, 10, 64)
        v = torch.randn(1, 4, 10, 64)
        out = flash_attention(q, k, v, is_causal=True)
        assert out.shape == (1, 4, 10, 64)

    def test_causal_masking(self):
        """Causal attention should not attend to future tokens."""
        q = torch.randn(1, 1, 4, 32)
        k = torch.randn(1, 1, 4, 32)
        v = torch.randn(1, 1, 4, 32)
        out_causal = flash_attention(q, k, v, is_causal=True)
        out_non_causal = flash_attention(q, k, v, is_causal=False)
        # They should differ (causal masks future tokens)
        assert not torch.allclose(out_causal, out_non_causal, atol=1e-5)

    def test_single_token(self):
        q = torch.randn(1, 2, 1, 32)
        k = torch.randn(1, 2, 1, 32)
        v = torch.randn(1, 2, 1, 32)
        out = flash_attention(q, k, v, is_causal=True)
        assert out.shape == (1, 2, 1, 32)


# ── PreAllocatedKVCache ──────────────────────────────────────────────────────


class TestPreAllocatedKVCache:
    """Tests for the O(1) append KV cache."""

    def test_creation(self):
        cache = PreAllocatedKVCache(
            n_layers=2,
            batch=1,
            n_kv_heads=4,
            max_seq_len=128,
            head_dim=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert cache.n_layers == 2
        assert cache.position == 0
        assert cache.filled == 0

    def test_get_layer_empty(self):
        cache = PreAllocatedKVCache(2, 1, 4, 128, 64, torch.float32, torch.device("cpu"))
        result = cache.get_layer(0)
        assert result is None  # empty cache returns None

    def test_append_and_get(self):
        cache = PreAllocatedKVCache(2, 1, 4, 128, 64, torch.float32, torch.device("cpu"))
        k_new = torch.randn(1, 4, 5, 64)
        v_new = torch.randn(1, 4, 5, 64)
        cache.append(0, k_new, v_new)
        cache.advance(5)
        k_view, v_view = cache.get_layer(0)
        assert k_view.shape == (1, 4, 5, 64)
        assert torch.equal(k_view, k_new)
        assert torch.equal(v_view, v_new)

    def test_reset(self):
        cache = PreAllocatedKVCache(1, 1, 4, 128, 64, torch.float32, torch.device("cpu"))
        cache.advance(10)
        assert cache.filled == 10
        cache.reset()
        assert cache.filled == 0
        assert cache.get_layer(0) is None

    def test_per_layer_independent(self):
        cache = PreAllocatedKVCache(3, 1, 4, 128, 64, torch.float32, torch.device("cpu"))
        k0 = torch.randn(1, 4, 3, 64)
        v0 = torch.randn(1, 4, 3, 64)
        cache.append(0, k0, v0)
        cache.advance(3)
        # Layer 1 should still be empty (position shared, but data only in layer 0)
        k1 = torch.randn(1, 4, 2, 64)
        v1 = torch.randn(1, 4, 2, 64)
        cache.append(1, k1, v1)
        # Both layers should have data at the same position
        k_view_0, _ = cache.get_layer(0)
        _k_view_1, _ = cache.get_layer(1)
        assert k_view_0.shape[-2] == 3  # only 3 tokens written to layer 0
        # Layer 1 has zeros for positions 0-2 (from layer 0's write), then data at 3-4
        # This is expected — the position is shared, each layer writes independently

    def test_tensor_pos_append_t(self):
        """pos_t mode: device-indexed writes at pos_t, no python slicing."""
        cache = PreAllocatedKVCache(2, 1, 4, 128, 64, torch.float32, torch.device("cpu"))
        cache.enable_tensor_pos()
        assert cache.pos_t is not None
        # Write 3 prompt tokens via the eager append path (T>1).
        k_pre = torch.randn(1, 4, 3, 64)
        v_pre = torch.randn(1, 4, 3, 64)
        cache.append(0, k_pre, v_pre)
        cache.advance(3)
        cache.sync_tensor_pos()
        assert cache.pos_t.item() == 3
        # Single-token writes land at pos_t regardless of python position.
        k1 = torch.randn(1, 4, 1, 64)
        v1 = torch.randn(1, 4, 1, 64)
        cache.append_t(0, k1, v1)
        assert torch.equal(cache.k_caches[0][:, :, 3:4], k1)
        assert torch.equal(cache.v_caches[0][:, :, 3:4], v1)
        # pos_t itself moves; python position is managed by the caller.
        cache.pos_t.fill_(4)
        k2 = torch.randn(1, 4, 1, 64)
        cache.append_t(0, k2, k2)
        assert torch.equal(cache.k_caches[0][:, :, 4:5], k2)
        # reset() clears pos_t.
        cache.reset()
        assert cache.pos_t.item() == 0

    def test_sync_tensor_pos(self):
        cache = PreAllocatedKVCache(1, 1, 4, 32, 64, torch.float32, torch.device("cpu"))
        cache.enable_tensor_pos()
        cache.position = 17
        cache.sync_tensor_pos()
        assert cache.pos_t.item() == 17


class TestKVCacheEquivalence:
    """Verify pre-allocated cache produces identical output to torch.cat cache."""

    def test_gqa_cache_equivalence(self):
        """GQA attention with pre-allocated cache should match torch.cat cache."""
        cfg = ModelConfig(
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            attn_type="gqa",
            max_seq_len=64,
            vocab_size=256,
            device="cpu",
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()

        idx = torch.randint(0, 256, (1, 5))

        # Path 1: traditional cache
        with torch.no_grad():
            logits1, _, presents1 = model(idx, use_cache=True)
            next_token1 = logits1[:, -1:, :].argmax(dim=-1)
            logits1b, _, _ = model(next_token1, past_key_values=presents1, use_cache=True)

        # Path 2: pre-allocated cache (GQA: n_kv_heads != n_heads)
        cache = PreAllocatedKVCache(
            n_layers=cfg.n_layers,
            batch=1,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            head_dim=cfg.d_model // cfg.n_heads,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            logits2, _ = model(idx, use_cache=False, preallocated_cache=cache)
            next_token2 = logits2[:, -1:, :].argmax(dim=-1)
            logits2b, _ = model(next_token2, use_cache=False, preallocated_cache=cache)

        assert torch.allclose(logits1, logits2, atol=1e-5), "GQA prefill logits differ"
        assert torch.allclose(logits1b, logits2b, atol=1e-5), "GQA decode logits differ"

    def test_gqa_tensor_pos_equivalence(self):
        """Static-KV decode (pos_t + mask + full-buffer read) must produce
        token-for-token identical output to the functional past_kv path —
        the contract CUDA-graph replay depends on."""
        cfg = ModelConfig(
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            attn_type="gqa",
            max_seq_len=64,
            vocab_size=256,
            device="cpu",
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()

        idx = torch.randint(0, 256, (1, 5))

        # Reference: functional past_kv decode, greedy.
        with torch.no_grad():
            logits, _, cur = model(idx, use_cache=True)
            ref_tokens = []
            ref_logits = []
            for _ in range(8):
                ref_logits.append(logits[:, -1, :].clone())
                tok = logits[:, -1:, :].argmax(dim=-1)
                ref_tokens.append(tok.item())
                logits, _, cur = model(tok, past_key_values=cur, use_cache=True)

        # Static path: preallocated cache + pos_t + mask_buf, same protocol
        # as StandardDecoding._generate_static.
        cache = PreAllocatedKVCache(
            n_layers=cfg.n_layers,
            batch=1,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            head_dim=cfg.d_model // cfg.n_heads,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache.enable_tensor_pos()
        mask_buf = torch.zeros(1, cache.max_seq_len, dtype=torch.long)
        with torch.no_grad():
            pmask = torch.ones(1, 5, dtype=torch.long)
            logits2, _ = model(idx, attention_mask=pmask, preallocated_cache=cache, use_cache=False)
            cache.position = 5
            mask_buf[:, :5] = 1
            static_tokens = []
            static_logits = [logits2[:, -1, :].clone()]
            for _ in range(8):
                tok = logits2[:, -1:, :].argmax(dim=-1)
                static_tokens.append(tok.item())
                pos = cache.position
                mask_buf[:, pos] = 1
                cache.pos_t.fill_(pos)
                logits2, _ = model(tok, attention_mask=mask_buf, preallocated_cache=cache, use_cache=False)
                cache.position += 1
                static_logits.append(logits2[:, -1, :].clone())

        assert static_tokens == ref_tokens
        for i, (a, b) in enumerate(zip(static_logits[:-1], ref_logits, strict=False)):
            assert torch.allclose(a, b, atol=1e-4), f"step {i}: static-KV logits diverge from functional KV"

    def test_conv_pad_mask_reads_pos_t_column(self):
        """Conv layers' _pad_mask must read the CURRENT decode position's
        mask column, not the bucket view's last column.

        Bug (2026-10-30): under static decode the attention_mask is a
        fixed bucket view [B, b] whose final column is always 0. The old
        ``attention_mask[:, -T:]`` slice therefore fed every conv layer a
        zero mask on each replay — zeroing the gated conv input (Bx *= 0)
        in all conv blocks and degenerating output to prompt echo. The
        fix indexes the live column via ``index_select(1, pos_t)`` so the
        op stays capturable inside the graph.
        """
        cfg = ModelConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            attn_type="gqa",
            n_layers=3,
            layer_types=["conv", "attention", "conv"],
            conv_kernel_size=3,
            max_seq_len=64,
            vocab_size=256,
            device="cpu",
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()
        conv_blocks = [b for b in model.blocks if getattr(b, "_is_conv", False)]
        assert len(conv_blocks) == 2

        # _pad_mask is consumed-and-cleared at the ModularBlock boundary —
        # capture the pad_mask kwarg as each conv layer receives it.
        seen = []
        for b in conv_blocks:
            orig_fwd = b.attn.forward

            def _spy(*args, _orig=orig_fwd, **kw):
                seen.append(kw.get("pad_mask"))
                return _orig(*args, **kw)

            b.attn.forward = _spy

        cache = PreAllocatedKVCache(
            n_layers=cfg.n_layers,
            batch=1,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            head_dim=cfg.d_model // cfg.n_heads,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache.enable_tensor_pos()
        # Bucket view: current position (5) is 1, the view's LAST column
        # is 0 — the old [:, -1] slice read that zero for every conv.
        mask_buf = torch.zeros(1, 32, dtype=torch.long)
        idx = torch.randint(0, 256, (1, 5))
        with torch.no_grad():
            model(idx, attention_mask=torch.ones(1, 5, dtype=torch.long), preallocated_cache=cache, use_cache=False)
            cache.position = 5
            cache.view_len = 32  # bucket width == mask view width
            mask_buf[:, :5] = 1
            mask_buf[:, 5] = 1
            cache.pos_t.fill_(5)
            tok = torch.randint(0, 256, (1, 1))
            model(tok, attention_mask=mask_buf, preallocated_cache=cache, use_cache=False)

        # [prefill_b0, prefill_b2, decode_b0, decode_b2] — the decode
        # step's masks are the last two.
        assert len(seen) == 4
        for m in seen[2:]:
            assert m is not None, "conv layer received no pad_mask"
            assert int(m.sum()) == 1, (
                "conv _pad_mask read the bucket-view tail (0) instead of "
                "the live pos_t column (1) — the gated conv input would "
                "be zeroed on every graph replay"
            )


# ── GPU tests (skipped if no CUDA) ───────────────────────────────────────────


@pytest.mark.gpu
class TestConfigurableResearchLLMGPU:
    """Tests that exercise the actual CUDA forward/backward path."""

    @pytest.fixture
    def gpu_model(self, tiny_config_gpu, gpu_available):
        model = ConfigurableResearchLLM(tiny_config_gpu).to("cuda")
        model.eval()
        return model

    def test_forward_on_gpu(self, gpu_model):
        idx = torch.randint(0, gpu_model.config.vocab_size, (1, 8), device="cuda")
        with torch.no_grad():
            logits, _loss = gpu_model(idx)
        assert logits.is_cuda
        assert logits.shape == (1, 8, gpu_model.config.vocab_size)

    def test_backward_on_gpu(self, gpu_model):
        idx = torch.randint(0, gpu_model.config.vocab_size, (1, 8), device="cuda")
        targets = torch.randint(0, gpu_model.config.vocab_size, (1, 8), device="cuda")
        gpu_model.train()
        _logits, loss = gpu_model(idx, targets=targets)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gpu_model.parameters())
        assert has_grad

    def test_kv_cache_on_gpu(self, gpu_model):
        idx = torch.randint(0, gpu_model.config.vocab_size, (1, 5), device="cuda")
        with torch.no_grad():
            _logits, _loss, presents = gpu_model(idx, use_cache=True)
        assert len(presents) == gpu_model.config.n_layers
        # Conv layers return None for their cache entry; attention layers return (k, v)
        for present in presents:
            if present is not None:
                k_cache, v_cache = present
                assert k_cache.is_cuda
                assert v_cache.is_cuda

    def test_bf16_forward_gpu(self, tiny_config_gpu, gpu_available):
        cfg = ModelConfig(**{**tiny_config_gpu.__dict__, "dtype": "bfloat16"})
        model = ConfigurableResearchLLM(cfg).to("cuda", dtype=torch.bfloat16)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")
        with torch.no_grad():
            logits, _loss = model(idx)
        assert logits.dtype == torch.bfloat16
        assert logits.is_cuda

    def test_compile_for_inference_gpu(self, tiny_config_gpu, gpu_available):
        """torch.compile should work with the pre-allocated KV cache path."""
        cfg = tiny_config_gpu
        model = ConfigurableResearchLLM(cfg).to("cuda")
        model.eval()

        # Compile with default mode (kernel fusion, no CUDA graphs).
        compiled = model.compile_for_inference(mode="default")

        # Pre-allocated cache — use n_kv_heads (conv layers have 0 KV heads)
        n_kv_heads_per_layer = [
            cfg.n_kv_heads if lt == "attention" else 0 for lt in (cfg.layer_types or ["attention"] * cfg.n_layers)
        ]
        cache = PreAllocatedKVCache(
            n_layers=cfg.n_layers,
            batch=1,
            n_kv_heads=cfg.n_kv_heads,
            n_kv_heads_per_layer=n_kv_heads_per_layer,
            max_seq_len=cfg.max_seq_len,
            head_dim=cfg.d_model // cfg.n_heads,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

        idx = torch.randint(0, cfg.vocab_size, (1, 4), device="cuda")
        with torch.no_grad():
            # First call triggers compilation (may be slow).
            logits, _ = compiled(idx, preallocated_cache=cache)
            # Second call uses compiled kernel.
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            logits2, _ = compiled(next_tok, preallocated_cache=cache)

        assert logits.shape == (1, 4, cfg.vocab_size)
        assert logits2.shape == (1, 1, cfg.vocab_size)
        assert logits.is_cuda

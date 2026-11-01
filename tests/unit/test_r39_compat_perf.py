"""Tests for R39: Model Compatibility & Engine Performance.

R39-1/2/3: Qwen3, Gemma3, Llama4 architecture adapters
R39-7: LASER + METRO MoE load balancing
"""

from __future__ import annotations

import pytest
import torch

from forge.engine.compat.arch_adapters import (
    convert_checkpoint,
    convert_gemma3_checkpoint,
    convert_llama4_checkpoint,
    convert_qwen3_checkpoint,
    detect_architecture,
    detect_n_layers,
    gemma3_layer_types,
)
from forge.moe.routers import LASERRouter, METRORouter

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_qwen3_state(n_layers=2, d_model=64, vocab=256):
    """Create a minimal Qwen3-like state dict."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, d_model),
        "model.norm.weight": torch.randn(d_model),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.q_norm.weight"] = torch.randn(d_model)
        sd[p + "self_attn.k_norm.weight"] = torch.randn(d_model)
        sd[p + "input_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(d_model)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.up_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.down_proj.weight"] = torch.randn(d_model, d_model * 2)
    return sd


def _make_gemma3_state(n_layers=2, d_model=64, vocab=256):
    """Create a minimal Gemma3-like state dict."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, d_model),
        "model.norm.weight": torch.randn(d_model),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.q_norm.weight"] = torch.randn(d_model)
        sd[p + "self_attn.k_norm.weight"] = torch.randn(d_model)
        sd[p + "input_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(d_model)
        sd[p + "pre_feedforward_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_feedforward_layernorm.weight"] = torch.randn(d_model)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.up_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.down_proj.weight"] = torch.randn(d_model, d_model * 2)
    return sd


def _make_llama4_state(n_layers=2, d_model=64, vocab=256, n_experts=4):
    """Create a minimal Llama4-like state dict with MoE."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, d_model),
        "model.norm.weight": torch.randn(d_model),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "input_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(d_model)
        # MoE
        sd[p + "feed_forward.router.weight"] = torch.randn(n_experts, d_model)
        # Shared expert
        sd[p + "feed_forward.shared_expert.gate_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "feed_forward.shared_expert.up_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "feed_forward.shared_expert.down_proj.weight"] = torch.randn(d_model, d_model * 2)
        # Routed experts
        for e in range(n_experts):
            ep = f"feed_forward.experts.{e}."
            sd[p + ep + "w1.weight"] = torch.randn(d_model * 2, d_model)
            sd[p + ep + "w3.weight"] = torch.randn(d_model * 2, d_model)
            sd[p + ep + "w2.weight"] = torch.randn(d_model, d_model * 2)
    return sd


# ── R39-1: Qwen3 ───────────────────────────────────────────────────────────


class TestQwen3Adapter:
    def test_detect_qwen3(self):
        sd = _make_qwen3_state()
        assert detect_architecture(sd) == "qwen3"

    def test_convert_qwen3(self):
        sd = _make_qwen3_state(n_layers=2)
        forge = convert_qwen3_checkpoint(sd, n_layers=2)
        # Embedding
        assert "embed.weight" in forge
        assert "final_norm.weight" in forge
        # Layer 0 attention
        assert "blocks.0.attn.q_proj.weight" in forge
        assert "blocks.0.attn.k_proj.weight" in forge
        assert "blocks.0.attn.v_proj.weight" in forge
        assert "blocks.0.attn.o_proj.weight" in forge
        # QK-norm
        assert "blocks.0.attn.q_norm.weight" in forge
        assert "blocks.0.attn.k_norm.weight" in forge
        # Layer norms
        assert "blocks.0.ln1.weight" in forge
        assert "blocks.0.ln2.weight" in forge
        # FFN
        assert "blocks.0.ffn.w_gate.weight" in forge
        assert "blocks.0.ffn.w_up.weight" in forge
        assert "blocks.0.ffn.w_down.weight" in forge
        # Layer 1
        assert "blocks.1.attn.q_proj.weight" in forge

    def test_qwen3_lossless(self):
        """Conversion is lossless (tensors passed through unchanged)."""
        sd = _make_qwen3_state(n_layers=1)
        forge = convert_qwen3_checkpoint(sd, n_layers=1)
        assert torch.equal(forge["embed.weight"], sd["model.embed_tokens.weight"])
        assert torch.equal(forge["blocks.0.attn.q_proj.weight"], sd["model.layers.0.self_attn.q_proj.weight"])

    def test_qwen3_detect_n_layers(self):
        sd = _make_qwen3_state(n_layers=3)
        assert detect_n_layers(sd) == 3


# ── R39-2: Gemma3 ──────────────────────────────────────────────────────────


class TestGemma3Adapter:
    def test_detect_gemma3(self):
        sd = _make_gemma3_state()
        assert detect_architecture(sd) == "gemma3"

    def test_convert_gemma3(self):
        sd = _make_gemma3_state(n_layers=2)
        forge = convert_gemma3_checkpoint(sd, n_layers=2)
        assert "embed.weight" in forge
        assert "blocks.0.attn.q_proj.weight" in forge
        # o_proj → out_proj mapping
        assert "blocks.0.attn.out_proj.weight" in forge
        assert "blocks.0.attn.o_proj.weight" not in forge
        # Gemma3-specific feedforward layernorms
        assert "blocks.0.ln_ffn_pre.weight" in forge
        assert "blocks.0.ln_ffn_post.weight" in forge
        # FFN
        assert "blocks.0.ffn.w_gate.weight" in forge

    def test_gemma3_lossless(self):
        sd = _make_gemma3_state(n_layers=1)
        forge = convert_gemma3_checkpoint(sd, n_layers=1)
        assert torch.equal(forge["embed.weight"], sd["model.embed_tokens.weight"])
        # o_proj → out_proj (tensor unchanged)
        assert torch.equal(forge["blocks.0.attn.out_proj.weight"], sd["model.layers.0.self_attn.o_proj.weight"])

    def test_gemma3_layer_types(self):
        """Gemma3 alternates SWA and global attention."""
        types = gemma3_layer_types(6)
        assert types[0] == "attention_swa"
        assert types[1] == "attention"
        assert types[2] == "attention_swa"
        assert types[3] == "attention"
        assert len(types) == 6


# ── R39-3: Llama4 ──────────────────────────────────────────────────────────


class TestLlama4Adapter:
    def test_detect_llama4(self):
        sd = _make_llama4_state()
        assert detect_architecture(sd) == "llama4"

    def test_convert_llama4(self):
        sd = _make_llama4_state(n_layers=2, n_experts=4)
        forge = convert_llama4_checkpoint(sd, n_layers=2)
        assert "embed.weight" in forge
        assert "blocks.0.attn.q_proj.weight" in forge
        assert "blocks.0.attn.out_proj.weight" in forge
        # MoE
        assert "blocks.0.moe.router.weight" in forge
        assert "blocks.0.moe.shared.w_gate.weight" in forge
        assert "blocks.0.moe.shared.w_up.weight" in forge
        assert "blocks.0.moe.shared.w_down.weight" in forge
        # Routed experts
        assert "blocks.0.moe.experts.0.w_gate.weight" in forge
        assert "blocks.0.moe.experts.0.w_up.weight" in forge
        assert "blocks.0.moe.experts.0.w_down.weight" in forge
        assert "blocks.0.moe.experts.3.w_gate.weight" in forge

    def test_llama4_lossless(self):
        sd = _make_llama4_state(n_layers=1, n_experts=2)
        forge = convert_llama4_checkpoint(sd, n_layers=1)
        assert torch.equal(forge["embed.weight"], sd["model.embed_tokens.weight"])
        assert torch.equal(forge["blocks.0.moe.router.weight"], sd["model.layers.0.feed_forward.router.weight"])
        assert torch.equal(
            forge["blocks.0.moe.experts.0.w_gate.weight"], sd["model.layers.0.feed_forward.experts.0.w1.weight"]
        )


# ── Auto-detection ─────────────────────────────────────────────────────────


class TestAutoDetection:
    def test_auto_convert_qwen3(self):
        sd = _make_qwen3_state()
        forge = convert_checkpoint(sd)
        assert "embed.weight" in forge

    def test_auto_convert_gemma3(self):
        sd = _make_gemma3_state()
        forge = convert_checkpoint(sd)
        assert "embed.weight" in forge

    def test_auto_convert_llama4(self):
        sd = _make_llama4_state()
        forge = convert_checkpoint(sd)
        assert "embed.weight" in forge

    def test_unknown_raises(self):
        sd = {"random.key": torch.randn(10)}
        with pytest.raises(ValueError, match="Unknown architecture"):
            convert_checkpoint(sd)


# ── R39-7: LASER ───────────────────────────────────────────────────────────


class TestLASER:
    def test_construction(self):
        router = LASERRouter(n_experts=8, n_layers=32, default_top_k=2)
        assert router.n_experts == 8
        assert router.n_layers == 32

    def test_layer_k_overrides_default(self):
        """Early layers get more experts, later layers fewer."""
        router = LASERRouter(n_experts=8, n_layers=30, default_top_k=2)
        early_k = router.get_top_k(0)
        late_k = router.get_top_k(29)
        assert early_k >= 2
        assert late_k <= 2

    def test_route(self):
        router = LASERRouter(n_experts=8, n_layers=10, default_top_k=2)
        logits = torch.randn(4, 8)  # 4 tokens, 8 experts
        indices, weights = router.route(logits, layer_idx=0)
        assert indices.shape == (4, router.get_top_k(0))
        assert weights.shape == indices.shape
        # Weights sum to 1 (normalized)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_custom_overrides(self):
        router = LASERRouter(n_experts=8, n_layers=10, default_top_k=2, layer_k_overrides={0: 4, 5: 1})
        assert router.get_top_k(0) == 4
        assert router.get_top_k(5) == 1
        assert router.get_top_k(3) == 2  # default

    def test_get_config(self):
        router = LASERRouter(n_experts=4, n_layers=6, default_top_k=2)
        cfg = router.get_config()
        assert cfg["algorithm"] == "LASER"
        assert cfg["n_experts"] == 4


# ── R39-7: METRO ───────────────────────────────────────────────────────────


class TestMETRO:
    def test_construction(self):
        router = METRORouter(n_experts=8, top_k=2)
        assert router.n_experts == 8
        assert router.top_k == 2

    def test_route(self):
        router = METRORouter(n_experts=8, top_k=2)
        logits = torch.randn(4, 8)
        indices, weights = router.route(logits)
        assert indices.shape == (4, 2)
        assert weights.shape == (4, 2)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_load_tracking(self):
        router = METRORouter(n_experts=4, top_k=2)
        logits = torch.randn(10, 4)
        router.route(logits)
        assert router.total_tokens == 10
        assert router.expert_counts.sum() == 20  # 10 tokens * 2 experts

    def test_load_balance_initially_perfect(self):
        router = METRORouter(n_experts=4, top_k=2)
        assert router.get_load_balance() == 1.0

    def test_rebalancing(self):
        """When load is imbalanced, METRO should correct."""
        router = METRORouter(n_experts=4, top_k=1, balance_threshold=1.5)
        # Create extreme imbalance: always route to expert 0
        router.expert_counts[0] = 100
        router.expert_counts[1] = 1
        router.expert_counts[2] = 1
        router.expert_counts[3] = 1
        router.total_tokens = 100
        # Logits that would normally route to expert 0
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        indices, weights = router.route(logits)
        # With correction, should not always pick expert 0
        # (correction boosts underutilized experts)
        assert indices[0, 0].item() != 0 or weights[0, 0].item() < 0.99

    def test_reset_stats(self):
        router = METRORouter(n_experts=4, top_k=2)
        router.expert_counts[0] = 100
        router.total_tokens = 100
        router.reset_stats()
        assert router.total_tokens == 0
        assert router.expert_counts.sum() == 0

    def test_get_config(self):
        router = METRORouter(n_experts=4, top_k=2)
        cfg = router.get_config()
        assert cfg["algorithm"] == "METRO"
        assert cfg["n_experts"] == 4
        assert "current_balance" in cfg

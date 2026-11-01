"""Tests for MambaKey — lossless checkpoint conversion and forward pass.

Tests:
1. MambaLayer weight structure matches expected shapes
2. MambaKey forward (HF -> ForgeAI) is lossless
3. MambaKey reverse (ForgeAI -> HF) is lossless
4. Round-trip (HF -> ForgeAI -> HF) is identity
5. MambaLayer forward pass produces correct output shape
6. MambaLayer incremental decode (T=1) works
7. MambaLayer integrated into ModularBlock via layer_types
8. Mamba2Key handles extra norm weights
9. Cross-arch Mamba1To2Key converts A_log correctly
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forge.keys.architecture.mamba_probe import MambaLayer
from forge.keys.misc.base import KeyClass
from tests.fixtures.keys.architecture.mamba_key import (
    Mamba1To2Key,
    Mamba2Key,
    MambaKey,
)


class TestMambaLayerWeights:
    """Test MambaLayer weight structure."""

    def test_weight_shapes(self):
        """All weight shapes match expected Mamba architecture."""
        layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
        state = layer.state_dict()

        assert tuple(state["in_proj.weight"].shape) == (256, 64), f"in_proj: {state['in_proj.weight'].shape}"
        assert tuple(state["conv1d.weight"].shape) == (128, 1, 4), f"conv1d: {state['conv1d.weight'].shape}"
        assert tuple(state["conv1d.bias"].shape) == (128,)
        assert tuple(state["x_proj.weight"].shape) == (36, 128), f"x_proj: {state['x_proj.weight'].shape}"
        assert tuple(state["dt_proj.weight"].shape) == (128, 4)
        assert tuple(state["dt_proj.bias"].shape) == (128,)
        assert tuple(state["A_log"].shape) == (128, 16)
        assert tuple(state["D"].shape) == (128,)
        assert tuple(state["out_proj.weight"].shape) == (64, 128)

    def test_param_count(self):
        """Parameter count matches expected formula (with Jamba norms)."""
        layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
        n_params = sum(p.numel() for p in layer.parameters())
        # in_proj: 64*256=16384, conv1d: 128*4+128=640, x_proj: 128*36=4608,
        # dt_proj: 4*128+128=640, A_log: 128*16=2048, D: 128, out_proj: 128*64=8192
        # Jamba norms (use_jamba_norms=True default):
        #   dt_layernorm: dt_rank=4, b_layernorm: d_state=16, c_layernorm: d_state=16
        expected = 16384 + 640 + 4608 + 640 + 2048 + 128 + 8192 + 4 + 16 + 16
        assert n_params == expected, f"Expected {expected}, got {n_params}"

    def test_dt_rank_auto(self):
        """dt_rank='auto' computes ceil(d_model/16)."""
        layer = MambaLayer(d_model=64, dt_rank="auto")
        assert layer.dt_rank == 4  # ceil(64/16) = 4

        layer128 = MambaLayer(d_model=128, dt_rank="auto")
        assert layer128.dt_rank == 8  # ceil(128/16) = 8

    def test_a_log_init(self):
        """A_log initialized to log(arange(1, d_state+1)) — S4D real init."""
        layer = MambaLayer(d_model=64, d_state=16)
        expected = torch.log(torch.arange(1, 17, dtype=torch.float32).repeat(128, 1))
        assert torch.allclose(layer.A_log.data, expected, atol=1e-6)

    def test_d_init(self):
        """D initialized to ones (skip connection)."""
        layer = MambaLayer(d_model=64)
        assert torch.allclose(layer.D.data, torch.ones(128))


class TestMambaKeyLossless:
    """Test MambaKey lossless conversion."""

    def _make_hf_state(self, n_layers=2, d_model=64):
        """Create a fake HuggingFace Mamba checkpoint (generic mixer. prefix).

        Real HF checkpoints store bare nn.Parameters with a .weight suffix
        (e.g. dt_layernorm.weight), so we rename norm params to match.
        """
        layer = MambaLayer(d_model=d_model, d_state=16, d_conv=4, expand=2)
        state = layer.state_dict()
        # Norm params in state_dict are bare (dt_layernorm); HF adds .weight
        _NORM_PARAMS = {"dt_layernorm", "b_layernorm", "c_layernorm"}
        hf_state = {}
        for i in range(n_layers):
            for name, tensor in state.items():
                hf_name = name + ".weight" if name in _NORM_PARAMS else name
                hf_state[f"model.layers.{i}.mixer.{hf_name}"] = tensor.clone()
            # Layer norms (generic naming)
            hf_state[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(d_model)
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(d_model)
        # Embedding + head + final norm (generic naming)
        hf_state["model.embed_tokens.weight"] = torch.randn(1000, d_model)
        hf_state["lm_head.weight"] = torch.randn(1000, d_model)
        hf_state["model.norm.weight"] = torch.ones(d_model)
        return hf_state

    def test_key_class(self):
        """MambaKey is BI (both directions work)."""
        key = MambaKey(n_layers=2)
        assert key.key_class() == KeyClass.BI

    def test_forward_maps_keys(self):
        """Forward (HF -> ForgeAI) maps all keys correctly."""
        hf_state = self._make_hf_state(n_layers=2)
        key = MambaKey(n_layers=2, mamba_prefix="mixer", ffn_style="zamba")
        result = key.forward(hf_state)

        assert result.success
        assert "blocks.0.attn.in_proj.weight" in result.weights
        assert "blocks.0.attn.A_log" in result.weights
        assert "blocks.0.attn.D" in result.weights
        assert "blocks.1.attn.out_proj.weight" in result.weights
        assert "blocks.0.ln1.weight" in result.weights
        assert "blocks.0.ln2.weight" in result.weights
        assert "embed.weight" in result.weights
        assert "head.weight" in result.weights
        assert "ln_f.weight" in result.weights

    def test_forward_is_lossless(self):
        """Forward preserves tensor values exactly."""
        hf_state = self._make_hf_state(n_layers=1)
        key = MambaKey(n_layers=1, mamba_prefix="mixer", ffn_style="zamba")
        result = key.forward(hf_state)

        for hf_key, forge_key in [
            ("model.layers.0.mixer.in_proj.weight", "blocks.0.attn.in_proj.weight"),
            ("model.layers.0.mixer.A_log", "blocks.0.attn.A_log"),
            ("model.layers.0.mixer.D", "blocks.0.attn.D"),
            ("model.embed_tokens.weight", "embed.weight"),
        ]:
            assert torch.equal(hf_state[hf_key], result.weights[forge_key]), f"Mismatch: {hf_key} -> {forge_key}"

    def test_reverse_maps_keys(self):
        """Reverse (ForgeAI -> HF) maps all keys correctly."""
        forge_state = {
            "blocks.0.attn.in_proj.weight": torch.randn(256, 64),
            "blocks.0.attn.A_log": torch.randn(128, 16),
            "blocks.0.attn.D": torch.ones(128),
            "blocks.0.ln1.weight": torch.ones(64),
            "embed.weight": torch.randn(1000, 64),
            "ln_f.weight": torch.ones(64),
        }
        key = MambaKey(n_layers=1, mamba_prefix="mamba", ffn_style="jamba")
        result = key.reverse(forge_state)

        assert result.success
        assert "model.layers.0.mamba.in_proj.weight" in result.data
        assert "model.layers.0.mamba.A_log" in result.data
        assert "model.embed_tokens.weight" in result.data
        assert "model.final_layernorm.weight" in result.data

    def test_round_trip_identity(self):
        """Round-trip (HF -> ForgeAI -> HF) is exact identity."""
        hf_state = self._make_hf_state(n_layers=2)
        # Use mamba_prefix="mixer" for the generic test state dict
        key = MambaKey(n_layers=2, mamba_prefix="mixer", ffn_style="zamba")

        # Forward: HF -> ForgeAI
        fwd = key.forward(hf_state)
        assert fwd.success

        # Reverse: ForgeAI -> HF
        rev = key.reverse(fwd.weights)
        assert rev.success

        # Check all original keys are preserved
        for hf_key, tensor in hf_state.items():
            assert hf_key in rev.data, f"Missing key: {hf_key}"
            assert torch.equal(tensor, rev.data[hf_key]), f"Value mismatch: {hf_key}"

    def test_attention_layer_mapping(self):
        """MambaKey handles mixed mamba/attention layers."""
        layer_types = ["mamba", "attention"]
        hf_state = {
            "model.layers.0.mamba.A_log": torch.randn(128, 16),
            "model.layers.0.mamba.D": torch.ones(128),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.1.self_attn.k_proj.weight": torch.randn(16, 64),
        }
        key = MambaKey(n_layers=2, layer_types=layer_types, mamba_prefix="mamba", ffn_style="jamba")
        result = key.forward(hf_state)

        assert result.success
        assert "blocks.0.attn.A_log" in result.weights
        assert "blocks.1.attn.q_proj.weight" in result.weights


class TestMamba2Key:
    """Test Mamba2Key with extra norm weights."""

    def test_mamba2_extra_weights(self):
        """Mamba2Key passes through extra norm weights (strips .weight suffix)."""
        hf_state = {
            "model.layers.0.mixer.in_proj.weight": torch.randn(256, 64),
            "model.layers.0.mixer.A_log": torch.randn(2, 1),  # Mamba2 scalar A
            "model.layers.0.mixer.D": torch.ones(128),
            "model.layers.0.mixer.dt_norm.weight": torch.ones(128),
            "model.layers.0.mixer.A_norm.weight": torch.ones(64),
            "model.layers.0.mixer.B_norm.weight": torch.ones(64),
            "model.layers.0.mixer.C_norm.weight": torch.ones(64),
        }
        key = Mamba2Key(n_layers=1)
        result = key.forward(hf_state)

        assert result.success
        # Norm names have .weight stripped in ForgeAI format (bare nn.Parameter)
        assert "blocks.0.attn.dt_norm" in result.weights
        assert "blocks.0.attn.A_norm" in result.weights
        assert "blocks.0.attn.B_norm" in result.weights
        assert "blocks.0.attn.C_norm" in result.weights

    def test_mamba2_round_trip(self):
        """Mamba2Key round-trip is identity (Jamba naming)."""
        hf_state = {
            "model.layers.0.mamba.in_proj.weight": torch.randn(256, 64),
            "model.layers.0.mamba.A_log": torch.randn(2, 1),
            "model.layers.0.mamba.D": torch.ones(128),
            "model.layers.0.mamba.dt_layernorm.weight": torch.ones(128),
            "model.layers.0.mamba.out_proj.weight": torch.randn(64, 128),
        }
        key = Mamba2Key(n_layers=1, mamba_prefix="mamba", ffn_style="jamba")
        fwd = key.forward(hf_state)
        rev = key.reverse(fwd.weights)

        for k in hf_state:
            assert k in rev.data, f"Missing key: {k}"
            assert torch.equal(hf_state[k], rev.data[k]), f"Mismatch: {k}"


class TestMamba1To2Key:
    """Test cross-arch Mamba1 -> Mamba2 conversion."""

    def test_partial_key(self):
        """Mamba1To2Key is PARTIAL (A_log conversion is lossy)."""
        key = Mamba1To2Key()
        assert key.key_class() == KeyClass.PARTIAL

    def test_a_log_conversion(self):
        """A_log (d_inner, d_state) -> (n_heads, 1) via averaging."""
        d_inner, d_state, head_dim = 128, 16, 64
        n_heads = d_inner // head_dim  # = 2

        data = {
            "A_log": torch.randn(d_inner, d_state),
            "D": torch.ones(d_inner),
            "_head_dim": head_dim,
        }
        key = Mamba1To2Key()
        result = key.forward(data)

        assert result.success
        assert tuple(result.weights["A_log"].shape) == (n_heads, 1)
        # D passes through unchanged
        assert torch.equal(result.weights["D"], data["D"])
        # Norms initialized to ones
        assert torch.equal(result.weights["dt_norm.weight"], torch.ones(d_inner))

    def test_reverse_not_supported(self):
        """Mamba2 -> Mamba1 is not supported (scalar A can't expand)."""
        key = Mamba1To2Key()
        result = key.reverse({"A_log": torch.randn(2, 1)})
        assert not result.success


class TestMambaForwardPass:
    """Test MambaLayer forward pass correctness."""

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
        layer.eval()
        x = torch.randn(2, 16, 64)  # (B=2, T=16, d_model=64)
        with torch.no_grad():
            y, _ = layer(x)
        assert y.shape == (2, 16, 64)

    def test_forward_single_token(self):
        """Forward pass works with T=1 (incremental decode)."""
        layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
        layer.eval()
        x = torch.randn(1, 1, 64)
        with torch.no_grad():
            y, _ = layer(x)
        assert y.shape == (1, 1, 64)

    def test_forward_deterministic(self):
        """Forward pass is deterministic in eval mode."""
        layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
        layer.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            y1, _ = layer(x)
            y2, _ = layer(x)
        assert torch.allclose(y1, y2, atol=1e-6)

    def test_forward_different_seq_lens(self):
        """Forward pass works with various sequence lengths."""
        layer = MambaLayer(d_model=32, d_state=8, d_conv=4, expand=2)
        layer.eval()
        for T in [1, 4, 16, 64, 256]:
            x = torch.randn(1, T, 32)
            with torch.no_grad():
                y, _ = layer(x)
            assert y.shape == (1, T, 32), f"Failed for T={T}"

    def test_gradient_flow(self):
        """Gradients flow through the layer."""
        layer = MambaLayer(d_model=32, d_state=8, d_conv=4, expand=2)
        x = torch.randn(1, 8, 32, requires_grad=True)
        y, _ = layer(x)
        loss = y.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
        assert has_grad, "No gradients found"


class TestMambaLayerIntegration:
    """Test MambaLayer integration with ModularBlock via layer_types."""

    def test_layer_type_mamba(self):
        """ModularBlock creates MambaLayer when layer_types=['mamba']."""
        from forge.config import ModelConfig
        from forge.model_loader import ModularBlock

        config = ModelConfig(
            d_model=64,
            n_layers=1,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=1000,
            max_seq_len=512,
            layer_types=["mamba"],
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        )
        block = ModularBlock(config, layer_idx=0)
        assert block.layer_type == "mamba"
        assert type(block.attn).__name__ == "MambaLayer"
        assert block._is_mamba
        assert not block._is_conv
        assert not block._supports_prealloc_cache  # Mamba has no KV cache

    def test_layer_type_attention_still_works(self):
        """ModularBlock still creates attention when layer_types=['attention']."""
        from forge.config import ModelConfig
        from forge.model_loader import ModularBlock

        config = ModelConfig(
            d_model=64,
            n_layers=1,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=1000,
            max_seq_len=512,
            layer_types=["attention"],
        )
        block = ModularBlock(config, layer_idx=0)
        assert block.layer_type == "attention"
        assert not block._is_mamba

    def test_mixed_layer_types(self):
        """ModularBlock handles mixed mamba/attention layers."""
        from forge.config import ModelConfig
        from forge.model_loader import ModularBlock

        config = ModelConfig(
            d_model=64,
            n_layers=4,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=1000,
            max_seq_len=512,
            layer_types=["mamba", "mamba", "attention", "mamba"],
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        )
        blocks = [ModularBlock(config, layer_idx=i) for i in range(4)]
        assert blocks[0]._is_mamba
        assert blocks[1]._is_mamba
        assert not blocks[2]._is_mamba
        assert blocks[3]._is_mamba


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Parable-Granite-4.1-3B port: preset fields + HF key remap coverage.

Full weight-level parity lives in scripts/test_parable_port.py (needs the
6.3GB checkpoint); this unit test guards the cheap invariants — preset
dims, Granite multiplier wiring, and that _remap_hf_keys maps every
GraniteForCausalLM tensor name to a ForgeAI name.
"""
import unittest

import torch

from forge.config import get_config
from forge.model.loader import ModelLoader


class TestParablePreset(unittest.TestCase):
    def test_preset_fields(self):
        cfg = get_config("parable_granite41_3b", device="cpu")
        assert cfg.vocab_size == 100_352
        assert cfg.d_model == 2560
        assert cfg.n_layers == 40
        assert cfg.n_heads == 40
        assert cfg.n_kv_heads == 8
        assert cfg.head_dim == 64
        assert cfg.intermediate_size == 8192
        assert cfg.attn_type == "gqa"
        assert cfg.ffn_type == "swiglu"
        assert cfg.norm_type == "rmsnorm"
        assert cfg.norm_eps == 1e-5
        assert cfg.rope_base == 10_000_000.0
        assert cfg.tie_word_embeddings is True
        assert cfg.attn_bias is False

    def test_granite_multipliers(self):
        cfg = get_config("parable_granite41_3b", device="cpu")
        assert cfg.embedding_multiplier == 12.0
        assert cfg.residual_multiplier == 0.22
        assert cfg.attn_scale == 0.015625
        assert cfg.logits_scaling == 10.0

    def test_hf_key_remap(self):
        cfg = get_config("parable_granite41_3b", device="cpu")
        state = {
            "model.embed_tokens.weight": torch.zeros(2, 2),
            "model.norm.weight": torch.zeros(2),
            "model.layers.0.input_layernorm.weight": torch.zeros(2),
            "model.layers.0.post_attention_layernorm.weight": torch.zeros(2),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 2),
            "model.layers.0.self_attn.v_proj.weight": torch.zeros(2, 2),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.up_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.down_proj.weight": torch.zeros(2, 2),
        }
        out = ModelLoader._remap_hf_keys(state, cfg)
        expected = {
            "model.embed_tokens.weight": "embed.weight",
            "model.norm.weight": "ln_f.weight",
            "model.layers.0.input_layernorm.weight": "blocks.0.ln1.weight",
            "model.layers.0.post_attention_layernorm.weight":
                "blocks.0.ln2.weight",
            "model.layers.0.self_attn.q_proj.weight":
                "blocks.0.attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight":
                "blocks.0.attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight":
                "blocks.0.attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight":
                "blocks.0.attn.out_proj.weight",
            "model.layers.0.mlp.gate_proj.weight":
                "blocks.0.ffn.w_gate.weight",
            "model.layers.0.mlp.up_proj.weight": "blocks.0.ffn.w_up.weight",
            "model.layers.0.mlp.down_proj.weight":
                "blocks.0.ffn.w_down.weight",
        }
        for hf_key, forge_key in expected.items():
            assert out[forge_key] is state[hf_key], f"{hf_key} -> {forge_key}"


if __name__ == "__main__":
    unittest.main()

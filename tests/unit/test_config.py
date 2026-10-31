"""Tests for research.config — ModelConfig dataclass and get_config factory."""

import pytest

from forge.config import MODEL_CONFIGS, ModelConfig, get_config


class TestModelConfigDefaults:
    """Default ModelConfig values and validation."""

    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.vocab_size == 65536
        assert cfg.d_model == 2048
        assert cfg.n_layers == 16
        assert cfg.n_heads == 32
        assert cfg.attn_type == "gqa"
        assert cfg.ffn_type == "swiglu"
        assert cfg.norm_type == "rmsnorm"
        assert cfg.dtype == "bfloat16"

    def test_device_is_string(self):
        cfg = ModelConfig()
        assert isinstance(cfg.device, str)
        assert cfg.device in ("cuda", "cpu")

    def test_n_kv_heads_defaults_none(self):
        cfg = ModelConfig()
        assert cfg.n_kv_heads is None

    def test_intermediate_size_defaults_none(self):
        cfg = ModelConfig()
        assert cfg.intermediate_size is None


class TestModelConfigValidation:
    """__post_init__ validation logic."""

    def test_d_model_must_divide_n_heads(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            ModelConfig(d_model=100, n_heads=3)


class TestModelConfigs:
    """Pre-defined MODEL_CONFIGS registry."""

    def test_configs_exist(self):
        expected = {
            "forgelm_v2",
            "forgelm_tiny",
        }
        assert expected.issubset(set(MODEL_CONFIGS.keys()))

    def test_forgelm_tiny_is_small(self):
        cfg = MODEL_CONFIGS["forgelm_tiny"]
        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4

    def test_forgelm_v2_matches_jamba_architecture(self):
        cfg = MODEL_CONFIGS["forgelm_v2"]
        assert cfg.vocab_size == 65536
        assert cfg.d_model == 2560
        assert cfg.n_layers == 28
        assert cfg.n_heads == 20
        assert cfg.n_kv_heads == 1
        assert cfg.intermediate_size == 8192
        assert cfg.attn_type == "gqa"
        assert cfg.attn_bias is False
        assert cfg.norm_type == "rmsnorm"
        assert cfg.rope_base == 1_000_000.0
        assert cfg.tie_word_embeddings is False
        assert cfg.use_rope is False  # Jamba has no RoPE

    def test_forgelm_v2_has_mamba_layers(self):
        cfg = MODEL_CONFIGS["forgelm_v2"]
        assert cfg.layer_types is not None
        assert cfg.layer_types.count("mamba") == 26
        assert cfg.layer_types.count("attention") == 2
        assert cfg.ssm_type == "mamba2"

    def test_all_configs_pass_validation(self):
        """Every pre-defined config should pass __post_init__ without error."""
        for name, cfg in MODEL_CONFIGS.items():
            assert isinstance(cfg, ModelConfig), f"{name} is not ModelConfig"
            assert cfg.d_model % cfg.n_heads == 0, f"{name}: d_model not divisible by n_heads"


class TestGetConfig:
    """get_config factory function."""

    def test_get_named_config(self):
        cfg = get_config("forgelm_tiny")
        assert cfg.d_model == 128
        assert cfg.n_layers == 4

    def test_get_default_config(self):
        cfg = get_config(None)
        assert cfg.d_model == 2048
        assert cfg.n_layers == 16

    def test_get_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown config"):
            get_config("nonexistent_model")

    def test_get_config_with_overrides(self):
        cfg = get_config("forgelm_tiny", d_model=256, n_layers=8)
        assert cfg.d_model == 256
        assert cfg.n_layers == 8
        # Other fields preserved
        assert cfg.n_heads == 4
        assert cfg.attn_type == "gqa"

    def test_get_config_override_creates_new_instance(self):
        original = get_config("forgelm_tiny")
        modified = get_config("forgelm_tiny", d_model=256)
        assert original.d_model == 128  # original unchanged
        assert modified.d_model == 256

    def test_get_config_override_validation(self):
        """Overrides should still trigger __post_init__ validation."""
        with pytest.raises(ValueError, match="divisible by n_heads"):
            get_config("forgelm_tiny", d_model=100, n_heads=3)


class TestHeadDim:
    """Explicit head_dim field (archs that decouple d_model/n_heads)."""

    def test_head_dim_defaults_none(self):
        assert ModelConfig().head_dim is None

    def test_head_dim_explicit_allowed(self):
        cfg = ModelConfig(d_model=2560, n_heads=32, head_dim=128)
        assert cfg.head_dim == 128

    def test_qwen3_4b_preset(self):
        cfg = MODEL_CONFIGS["qwen3_4b"]
        assert cfg.vocab_size == 151936
        assert cfg.d_model == 2560
        assert cfg.n_layers == 36
        assert cfg.n_heads == 32
        assert cfg.n_kv_heads == 8
        assert cfg.head_dim == 128  # 2560/32=80 would be wrong
        assert cfg.attn_bias is False
        assert cfg.use_qk_norm is True
        assert cfg.tie_word_embeddings is True
        assert cfg.rope_base == 5_000_000.0

    def test_gqa_explicit_head_dim_shapes(self):
        """GQA module shapes follow explicit head_dim, not d_model/n_heads."""
        import torch

        from forge.model.layers import GroupedQueryAttention
        attn = GroupedQueryAttention(d_model=2560, n_heads=32, n_kv_heads=8,
                                     head_dim=128, use_qk_norm=True)
        assert attn.q_proj.out_features == 32 * 128
        assert attn.k_proj.out_features == 8 * 128
        assert attn.out_proj.in_features == 32 * 128
        assert attn.out_proj.out_features == 2560
        assert attn.q_norm.normalized_shape[0] == 128
        x = torch.randn(1, 7, 2560)
        out, _ = attn(x)
        assert out.shape == (1, 7, 2560)

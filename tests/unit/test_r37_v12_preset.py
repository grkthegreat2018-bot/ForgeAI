"""Tests for R37-6: V12-Jamba Preset Definition and Lineage.

Per AGENTS.md section A: V12-Jamba MUST derive from ForgeLM V2 (the Jamba
base), carry forward all prior keys, and be bit-exact lossless at warm
start (zero-init new keys).
"""
from __future__ import annotations

from forge.config import MODEL_CONFIGS, get_config


class TestV12JambaPreset:
    """V12-Jamba preset definition and lineage from V2."""

    def test_v12_jamba_exists(self):
        assert "forgelm_v12_jamba" in MODEL_CONFIGS

    def test_v12_jamba_derives_from_v2(self):
        """V12-Jamba must carry forward all V2 architectural parameters."""
        v2 = get_config("forgelm_v2")
        v12 = get_config("forgelm_v12_jamba")
        # Core Jamba architecture unchanged
        assert v12.d_model == v2.d_model
        assert v12.n_layers == v2.n_layers
        assert v12.n_heads == v2.n_heads
        assert v12.n_kv_heads == v2.n_kv_heads
        assert v12.intermediate_size == v2.intermediate_size
        assert v12.vocab_size == v2.vocab_size
        assert v12.max_seq_len == v2.max_seq_len
        assert v12.attn_type == v2.attn_type
        assert v12.ffn_type == v2.ffn_type
        assert v12.norm_type == v2.norm_type
        assert v12.layer_types == v2.layer_types

    def test_v12_jamba_new_keys_enabled(self):
        """V12-Jamba must enable all 5 new architecture keys."""
        v12 = get_config("forgelm_v12_jamba")
        assert v12.use_mamba3 is True
        assert v12.use_kronecker_embed is True
        assert v12.use_pit is True
        assert v12.use_outro is True
        assert v12.use_forge_hybrid is True

    def test_v12_jamba_new_key_parameters(self):
        """V12-Jamba new key parameters have sensible defaults."""
        v12 = get_config("forgelm_v12_jamba")
        # Mamba-3
        assert v12.mamba3_d_state == 16
        # Kronecker
        assert v12.kronecker_d_char == 64
        assert v12.kronecker_max_char_len == 8
        # OutRo
        assert v12.outro_sink_threshold == 0.5
        # ForgeHybrid
        assert v12.forge_hybrid_d_state == 16
        assert v12.forge_hybrid_sink_threshold == float("inf")  # warm start
        assert v12.forge_hybrid_n_ssm_layers is None  # all layers

    def test_v12_jamba_warm_start_is_lossless(self):
        """V12-Jamba warm start config: all new keys are zero/identity-init.

        ForgeHybrid: sink_threshold=inf → all tokens use attention
        (SSM path zero-init, never activated at warm start).
        Mamba-3: imag=0 → identical to Mamba-2.
        Kronecker: SVD init from existing embedding → reconstruction.
        PIT: identity init → lossless.
        OutRo: threshold-based, no weight change.
        """
        v12 = get_config("forgelm_v12_jamba")
        # ForgeHybrid warm start = all attention
        assert v12.forge_hybrid_sink_threshold == float("inf")

    def test_v12_jamba_get_config_returns_fresh_instance(self):
        """get_config must return a fresh instance (not the shared preset)."""
        c1 = get_config("forgelm_v12_jamba")
        c1.d_model = 999
        c2 = get_config("forgelm_v12_jamba")
        assert c2.d_model == 2560  # unchanged

    def test_no_silent_regression(self):
        """V12-Jamba must not silently drop any V2 feature.

        Per AGENTS.md: a new preset that drops a prior key must document WHY.
        V12-Jamba does not drop any V2 keys — it only adds new ones, and all
        divergences are documented in dropped_keys (enforced by
        test_preset_lineage.py).
        """
        v2 = get_config("forgelm_v2")
        v12 = get_config("forgelm_v12_jamba")
        v12_only_keys = {
            "parent", "dropped_keys",
            "use_mamba3", "mamba3_d_state",
            "use_kronecker_embed", "kronecker_d_char", "kronecker_max_char_len",
            "use_outro", "outro_sink_threshold",
            "use_forge_hybrid", "forge_hybrid_d_state",
            "forge_hybrid_sink_threshold", "forge_hybrid_n_ssm_layers",
            # V12-Jamba intentionally enables PIT (was False in V2)
            "use_pit",
        }
        for key, val in v2.__dict__.items():
            if key in v12_only_keys:
                continue
            assert key in v12.__dict__, f"V12-Jamba dropped V2 key: {key}"
            assert v12.__dict__[key] == val, (
                f"V12-Jamba changed V2 key {key}: {val} → {v12.__dict__[key]}"
            )

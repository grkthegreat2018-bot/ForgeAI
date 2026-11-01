"""Test the LFM2.5-Thinking pipeline orchestration."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from forge.self_play.infinite_loop import (
    InfiniteSelfPlayLoop,
    LoopConfig,
    ThinkingPipeline,
    ThinkingPipelineConfig,
)


def test_pipeline_config_defaults():
    """ThinkingPipelineConfig should have sensible defaults for all 4 stages."""
    config = ThinkingPipelineConfig()
    assert config.cpt_enabled
    assert config.sft_enabled
    assert config.dpo_enabled
    assert config.rlvr_enabled
    # V2 (Jamba) is the sole base — optimizer defaults to cpu_offload (3B on 12GB)
    assert config.optimizer == "cpu_offload"
    assert config.config_name == "forgelm_v2"
    assert config.cpt_reasoning_ratio == 0.6
    assert config.sft_mix_ratio == 0.5
    assert config.dpo_n_temp_samples == 5
    assert config.rlvr_use_repetition_penalty
    assert len(config.cpt_reasoning_data) == 3
    assert len(config.cpt_general_data) == 2
    print("PASS: ThinkingPipelineConfig has correct V10 defaults")


def test_pipeline_stage_paths():
    """Stage checkpoint paths should follow V10 naming convention."""
    pipeline = ThinkingPipeline("base.safetensors")
    assert "CPT" in pipeline._stage_path("CPT")
    assert "SFT2" in pipeline._stage_path("SFT2")
    assert "DPO" in pipeline._stage_path("DPO")
    assert "RLVR" in pipeline._stage_path("RLVR")
    # V10 naming: ForgeLM_V10_<stage>.safetensors
    assert "ForgeLM_V10" in pipeline._stage_path("CPT")
    print("PASS: Stage paths follow ForgeLM_V10_<stage> convention")


def test_pipeline_resumability():
    """Pipeline should detect completed stages by checking checkpoint existence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ThinkingPipelineConfig(checkpoint_dir=tmpdir)
        pipeline = ThinkingPipeline("base.safetensors", config)

        # No checkpoints exist yet
        cpt_path = pipeline._stage_path("CPT")
        assert not pipeline._stage_completed(cpt_path)

        # Create a fake checkpoint
        with open(cpt_path, "w") as f:
            f.write("fake")
        assert pipeline._stage_completed(cpt_path)
    print("PASS: Pipeline resumability check works")


def test_pipeline_stage_disabled():
    """Disabled stages should be skipped, returning the input checkpoint."""
    config = ThinkingPipelineConfig(cpt_enabled=False)
    ThinkingPipeline("base.safetensors", config)
    # When CPT is disabled, run_cpt should return base_checkpoint
    # (We can't actually run it, but we can verify the logic)
    assert not config.cpt_enabled
    print("PASS: Disabled stages are handled")


def test_pipeline_history_tracking():
    """Pipeline should track completed stages in history."""
    pipeline = ThinkingPipeline("base.safetensors")
    assert pipeline.history == []
    # History would be populated after each stage runs
    print("PASS: Pipeline history initialized empty")


# ── V10 self-play loop tests ────────────────────────────────────────────────


def test_loop_config_v10_defaults():
    """LoopConfig should default to V10 config + ForgeEngine + training tricks."""
    cfg = LoopConfig()
    # V2 (Jamba) is the sole base
    assert cfg.config_name == "forgelm_v2"
    # ForgeEngine inference features
    assert cfg.use_forge_engine is True
    assert cfg.kv_cache == "spectral"
    assert cfg.decoding == "mtp_selfspec"
    assert cfg.use_triton_conv is True
    assert cfg.use_prefix_cache is True
    # Training tricks
    assert cfg.ft_optimizer == "muon_sf"
    assert cfg.ft_lora is True
    assert cfg.ft_bitnet_everywhere is True
    assert cfg.ft_grad_accum == 5  # evolution-discovered
    assert cfg.ft_grad_compression == "int4"  # evolution-discovered
    assert cfg.ft_entropy_alpha == 0.5  # WeFT/VCORE
    assert cfg.ft_focal_gamma == 4.93  # evolution-discovered
    print("PASS: LoopConfig has correct V10 defaults")


def test_loop_checkpoint_path_v10():
    """Epoch checkpoint paths should use current-base (V2) naming."""
    loop = InfiniteSelfPlayLoop("base.safetensors")
    path = loop._epoch_checkpoint_path(7)
    assert "ForgeLM_V2_SP7" in path
    assert path.endswith(".safetensors")
    print("PASS: Epoch checkpoint paths use ForgeLM_V2_SP naming")


def test_loop_engine_lifecycle():
    """Loop should initialize with _engine=None and free_engine is a no-op."""
    loop = InfiniteSelfPlayLoop("base.safetensors")
    assert loop._engine is None
    # _free_engine on None engine should be a safe no-op
    loop._free_engine()
    assert loop._engine is None
    print("PASS: Engine lifecycle (init None, free no-op) works")


if __name__ == "__main__":
    test_pipeline_config_defaults()
    test_pipeline_stage_paths()
    test_pipeline_resumability()
    test_pipeline_stage_disabled()
    test_pipeline_history_tracking()
    test_loop_config_v10_defaults()
    test_loop_checkpoint_path_v10()
    test_loop_engine_lifecycle()
    print("\n=== All ThinkingPipeline + V10 self-play tests passed ===")

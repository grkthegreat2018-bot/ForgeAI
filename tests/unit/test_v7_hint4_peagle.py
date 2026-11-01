"""Integration test: config fields for HINT4-NLRQ + Tied PEAGLE."""

import pytest
import torch

from forge.config import get_config


class TestNewTech:
    def test_config_has_new_fields(self):
        """Config should have the HINT4 + PEAGLE tied fields."""
        cfg = get_config("forgelm_v2")
        assert hasattr(cfg, "nlrq_use_hadamard")
        assert hasattr(cfg, "use_peagle_tied")
        assert hasattr(cfg, "peagle_lora_rank")

    def test_hint4_config(self):
        """HINT4 can be enabled via overrides."""
        cfg = get_config("forgelm_v2")
        cfg.nlrq_factor_bits = 4
        cfg.nlrq_use_hadamard = True
        assert cfg.nlrq_factor_bits == 4
        assert cfg.nlrq_use_hadamard is True

    def test_peagle_tied_config(self):
        """Tied PEAGLE can be enabled via overrides."""
        cfg = get_config("forgelm_v2")
        cfg.use_peagle_tied = True
        cfg.peagle_lora_rank = 32
        assert cfg.use_peagle_tied is True
        assert cfg.peagle_lora_rank == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_builds_with_new_config(self):
        """Model should build without errors with new config fields."""
        from forge.model_loader import ConfigurableResearchLLM

        cfg = get_config("forgelm_v2")
        cfg.n_layers = 12  # need > hyperloop_begin + hyperloop_end (4+4=8)
        cfg.d_model = 256  # small for fast test
        cfg.n_heads = 8
        cfg.n_kv_heads = 2
        cfg.intermediate_size = 512
        cfg.nlrq_rank = 64
        cfg.vocab_size = 256
        cfg.layer_types = ["attention"] * 12
        # Keep HINT4 off for build test (INT8 is the safe path)
        cfg.nlrq_use_hadamard = False
        cfg.nlrq_factor_bits = 8
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                model = ConfigurableResearchLLM(cfg)
        finally:
            torch.set_default_dtype(old_dtype)
        # Just verify it builds
        assert model is not None
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_hint4_builds(self):
        """HINT4 (INT4 + Hadamard) should build on GPU."""
        from forge.model_loader import ConfigurableResearchLLM

        cfg = get_config("forgelm_v2")
        cfg.n_layers = 12
        cfg.d_model = 256
        cfg.n_heads = 8
        cfg.n_kv_heads = 2
        cfg.intermediate_size = 512
        cfg.nlrq_rank = 64
        cfg.vocab_size = 256
        cfg.layer_types = ["attention"] * 12
        cfg.nlrq_factor_bits = 4
        cfg.nlrq_use_hadamard = True
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                model = ConfigurableResearchLLM(cfg)
        finally:
            torch.set_default_dtype(old_dtype)
        assert model is not None
        del model
        torch.cuda.empty_cache()

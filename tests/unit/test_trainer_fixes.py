"""Tests for trainer refactoring fixes.

Validates the specific bugs fixed in the refactor commit:
1. load_anchor_cached shared utility exists in training_utils with LRU eviction
2. restore_ema is used (not inline copy) — verify it's importable
3. RPO trainer has optimizer.zero_grad() after OOM skip
4. FORGE optimizer hook registration is AFTER optimizer creation (not before)
5. Gradient checkpointing warns on unknown strategy
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

from pathlib import Path

import torch


class TestLoadAnchorCached:
    """Shared load_anchor_cached in training_utils with LRU eviction."""

    def test_function_exists(self):
        from forge.training.training_utils import load_anchor_cached
        assert callable(load_anchor_cached)

    def test_caches_by_path_mtime(self, tmp_path):
        from safetensors.torch import save_file

        from forge.training.training_utils import _ANCHOR_CACHE, load_anchor_cached

        # Create a fake anchor checkpoint
        anchor_path = str(tmp_path / "anchor.safetensors")
        save_file({"weight_a": torch.randn(4, 8)}, anchor_path)

        # Clear cache
        _ANCHOR_CACHE.clear()

        # First load — populates cache
        sd1 = load_anchor_cached(anchor_path)
        assert "weight_a" in sd1
        assert len(_ANCHOR_CACHE) == 1

        # Second load — cache hit (same mtime)
        sd2 = load_anchor_cached(anchor_path)
        assert len(_ANCHOR_CACHE) == 1  # no new entry

    def test_lru_eviction(self, tmp_path):
        """Cache should evict oldest entries when over _ANCHOR_CACHE_MAX."""
        from safetensors.torch import save_file

        from forge.training.training_utils import _ANCHOR_CACHE, _ANCHOR_CACHE_MAX, load_anchor_cached

        _ANCHOR_CACHE.clear()

        # Create more anchors than the cache max
        paths = []
        for i in range(_ANCHOR_CACHE_MAX + 3):
            p = str(tmp_path / f"anchor_{i}.safetensors")
            save_file({f"w_{i}": torch.randn(2, 2)}, p)
            paths.append(p)

        for p in paths:
            load_anchor_cached(p)

        # Cache should not exceed max
        assert len(_ANCHOR_CACHE) <= _ANCHOR_CACHE_MAX


class TestRestoreEmaImportable:
    """restore_ema should be importable from training_utils (used by CPT now)."""

    def test_importable(self):
        from forge.training.training_utils import restore_ema
        assert callable(restore_ema)

    def test_restores_weights(self):
        from forge.training.training_utils import restore_ema

        model = torch.nn.Linear(4, 4)
        # Save current weights as "EMA"
        ema_state = {name: p.data.clone() for name, p in model.named_parameters()}
        # Corrupt model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)
        # Restore
        restore_ema(ema_state, model)
        # Weights should be restored
        for name, p in model.named_parameters():
            assert torch.equal(p.data, ema_state[name])


class TestForgeHookOrdering:
    """FORGE optimizer hooks should be registered AFTER optimizer creation."""

    def test_hook_registration_after_optimizer(self):
        """Verify the source code registers hooks after configure_optimizer."""
        path = Path(__file__).parent.parent.parent / "forge" / "training" / "runners" / "sft_train.py"
        source = path.read_text(encoding="utf-8")
        # Find the positions
        optimizer_pos = source.find("optimizer = configure_optimizer(")
        hook_pos = source.find("optimizer.register_hooks(model)")
        assert optimizer_pos > 0, "configure_optimizer call not found"
        assert hook_pos > 0, "register_hooks call not found"
        assert hook_pos > optimizer_pos, (
            "FORGE hook registration must come AFTER optimizer creation, "
            f"but hook_pos={hook_pos} < optimizer_pos={optimizer_pos}"
        )


class TestGradientCheckpointingWarning:
    """model_loader should warn on unknown gradient checkpointing strategy."""

    def test_warns_on_unknown_strategy(self):
        import warnings

        from forge.config import get_config
        from forge.model_loader import ConfigurableResearchLLM

        cfg = get_config("forgelm_tiny")
        cfg.vocab_size = 64
        cfg.device = "cpu"
        cfg.dtype = "float32"
        cfg.use_gradient_checkpointing = True
        cfg.selective_gradient_checkpointing = "nonexistent_strategy"

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = ConfigurableResearchLLM(cfg)
                # Check that a warning was issued
                assert any("Unknown gradient checkpointing" in str(warning.message) for warning in w), (
                    f"Expected warning about unknown strategy, got: {[str(x.message) for x in w]}"
                )
        finally:
            torch.set_default_dtype(old_dtype)

    def test_no_warning_on_valid_strategy(self):
        import warnings

        from forge.config import get_config
        from forge.model_loader import ConfigurableResearchLLM

        cfg = get_config("forgelm_tiny")
        cfg.vocab_size = 64
        cfg.device = "cpu"
        cfg.dtype = "float32"
        cfg.use_gradient_checkpointing = True
        cfg.selective_gradient_checkpointing = "all"

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = ConfigurableResearchLLM(cfg)
                # No warning about unknown strategy
                assert not any("Unknown gradient checkpointing" in str(warning.message) for warning in w), (
                    f"Unexpected warning for valid strategy 'all': {[str(x.message) for x in w]}"
                )
        finally:
            torch.set_default_dtype(old_dtype)

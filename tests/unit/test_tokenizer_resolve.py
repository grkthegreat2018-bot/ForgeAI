"""Regression test: _resolve_path must not silently replace HF repo-ids
with the canonical ForgeLM tokenizer (the "Qwen gets 65k vocab" bug)."""
import os

from research.tokenizer_cache import CANONICAL_TOKENIZER_PATH, _resolve_path


def test_hf_repo_id_passes_through():
    # One-slash org/name that is NOT a dir on disk must stay as-is so
    # _load_hf_tokenizer resolves the real HF tokenizer.
    assert _resolve_path("Qwen/Qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B"
    assert _resolve_path("LiquidAI/LFM2-1.2B") == "LiquidAI/LFM2-1.2B"


def test_legacy_redirect_still_works():
    # Local legacy paths that don't exist redirect to canonical.
    assert (_resolve_path("research/checkpoints/lfm2_5_tokenizer")
            == CANONICAL_TOKENIZER_PATH)


def test_missing_local_dir_falls_back():
    # Repo-local missing dirs keep the canonical fallback behavior.
    assert (_resolve_path("research/checkpoints/definitely_missing_tok")
            == CANONICAL_TOKENIZER_PATH)


def test_existing_dir_unchanged():
    if os.path.isdir(CANONICAL_TOKENIZER_PATH):
        assert _resolve_path(CANONICAL_TOKENIZER_PATH) == \
            CANONICAL_TOKENIZER_PATH

"""Centralized tokenizer cache — loads once, wraps with gigatoken for ~35x speedup.

All modules should use `get_tokenizer()` instead of `AutoTokenizer.from_pretrained()`.
This avoids the 4.3s import + load overhead repeated across 15+ call sites, and
wraps with gigatoken's HFCompat for ~35x faster encode on bulk operations.

Fast path: uses the `tokenizers` Rust library directly (190ms total) instead of
going through `transformers` (4254ms). Falls back to `transformers` if the fast
path fails (e.g. missing tokenizer.json or special token config).

Usage:
    from research.tokenizer_cache import get_tokenizer

    tokenizer = get_tokenizer()  # uses the canonical tokenizer dir
    tokens = tokenizer("print('hello')", return_tensors="pt")
    decoded = tokenizer.decode(tokens.input_ids[0])
"""
import json
import os
from functools import lru_cache
from typing import Optional


# The canonical tokenizer directory — the only one that exists on disk.
# Old code may pass "research/checkpoints/forgelm_v2_tokenizer" (the pre-V10
# name); get_tokenizer() auto-redirects those to this path.
# Absolute: callers/GUI launch with an arbitrary cwd, and os.path.isdir
# checks below must not depend on it (EC-129).
from pathlib import Path as _Path
CANONICAL_TOKENIZER_PATH = str(
    _Path(__file__).resolve().parent / "checkpoints" / "forgelm_v2_tokenizer")

# Legacy paths that should be silently redirected to the canonical one.
# This avoids touching 20+ call sites that still reference the old name.
_LEGACY_REDIRECTS = {
    "research/checkpoints/forgelm_v2_tokenizer",
    "research/checkpoints/lfm2_5_tokenizer",
}


def _resolve_path(path: str) -> str:
    """Redirect legacy/missing tokenizer paths to the canonical directory.

    If the requested path doesn't exist on disk but the canonical tokenizer
    dir does, use the canonical one. This fixes the bug where old hardcoded
    paths (forgelm_v2_tokenizer) caused FileNotFoundError → HF repo-id validation
    errors.
    """
    if path in _LEGACY_REDIRECTS and not os.path.isdir(path):
        return CANONICAL_TOKENIZER_PATH
    # HF repo-ids ("org/name") are not local paths — never redirect them to
    # the canonical tokenizer. A repo-id has exactly one "/" separating two
    # path segments and doesn't live under a repo-root dir. Without this
    # guard, e.g. "Qwen/Qwen2.5-0.5B" was silently replaced by the 65k-vocab
    # ForgeLM tokenizer while loading a 151936-vocab model.
    p = path.replace("\\", "/")
    first = p.split("/")[0]
    if (p.count("/") == 1 and ":" not in p
            and first not in {"research", "forge", "scripts", "tests",
                              ".", ".."}):
        return path
    # Generic fallback: if the requested dir doesn't exist, try canonical.
    if not os.path.isdir(path) and os.path.isdir(CANONICAL_TOKENIZER_PATH):
        return CANONICAL_TOKENIZER_PATH
    return path


def _load_fast_tokenizer(path: str):
    """Fast path: load via `tokenizers` Rust library + gigatoken wrap.

    Bypasses `transformers` entirely (saves ~4s import time).
    Reads special token IDs from tokenizer_config.json.
    """
    from tokenizers import Tokenizer as RustTokenizer

    tokenizer_json = os.path.join(path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(f"tokenizer.json not found in {path}")

    rust_tok = RustTokenizer.from_file(tokenizer_json)

    # Read special token strings from tokenizer_config.json.
    # gigatoken's eos_token_id property reads from self.eos_token (the string),
    # so we set the strings and the ID properties resolve automatically.
    config_path = os.path.join(path, "tokenizer_config.json")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

    # Wrap with gigatoken for fast encode
    import gigatoken as gt
    fast_tok = gt.Tokenizer(rust_tok).as_hf()

    # Set special token strings — gigatoken's eos_token_id / bos_token_id /
    # pad_token_id properties read from these attributes. Without setting
    # them, the properties return None (the raw tokenizers.Tokenizer doesn't
    # carry HF special token metadata).
    if cfg.get("eos_token"):
        fast_tok.eos_token = cfg["eos_token"]
    if cfg.get("bos_token"):
        fast_tok.bos_token = cfg["bos_token"]
    if cfg.get("pad_token"):
        fast_tok.pad_token = cfg["pad_token"]

    return fast_tok


def _load_hf_tokenizer(path: str):
    """Slow fallback: load via transformers AutoTokenizer.

    Takes ~4s due to the transformers import overhead.
    """
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(path)
    try:
        import gigatoken as gt
        return gt.Tokenizer(hf_tok).as_hf()
    except ImportError:
        return hf_tok


@lru_cache(maxsize=8)
def get_tokenizer(path: str = CANONICAL_TOKENIZER_PATH):
    """Load and cache a tokenizer, wrapped with gigatoken if available.

    Fast path: uses `tokenizers` Rust library directly (190ms).
    Fallback: uses `transformers` AutoTokenizer (4254ms).

    Legacy paths (e.g. "research/checkpoints/forgelm_v2_tokenizer") are
    automatically redirected to the canonical tokenizer directory if
    they don't exist on disk.

    Args:
        path: tokenizer directory path (legacy paths auto-redirected).

    Returns:
        A tokenizer object with HF-compatible API (encode, decode, eos_token_id, etc.).
    """
    resolved = _resolve_path(path)
    try:
        return _load_fast_tokenizer(resolved)
    except Exception:
        return _load_hf_tokenizer(resolved)


@lru_cache(maxsize=8)
def get_tokenizer_no_wrap(path: str = CANONICAL_TOKENIZER_PATH):
    """Load and cache a raw HF tokenizer (no gigatoken wrapping).

    Use this when you need the exact HF tokenizer object (e.g. for save_pretrained).
    """
    from transformers import AutoTokenizer
    resolved = _resolve_path(path)
    return AutoTokenizer.from_pretrained(resolved)

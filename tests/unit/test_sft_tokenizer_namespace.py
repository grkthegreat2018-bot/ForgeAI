"""Regression: disk token cache must namespace by tokenizer identity.

Root cause of the lfm3 distill device-side assert: _tokenize_cached keyed
DiskTokenCache entries under the bare "default" hash, so tokenizations
produced by a 151936-vocab tokenizer were served to a 128000-vocab model —
ids like 151645 reached F.embedding -> CUDA "vectorized gather" assert.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge.training.data.efficient_pipeline import DiskTokenCache
import forge.training.runners.sft_train as sft


class _FakeTokenizer:
    """Minimal stand-in exposing the attrs _tokenize_cached hashes on."""
    def __init__(self, vocab_size, ids_out):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = 0
        self._ids_out = ids_out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": list(self._ids_out)}


def test_disk_cache_entries_are_tokenizer_namespaced(tmp_path, monkeypatch):
    cache = DiskTokenCache(cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(sft, "_disk_tok_cache", cache)
    sft._tokenize_cache.clear()

    tok_qwen = _FakeTokenizer(151936, [151644, 151645, 100])
    tok_lfm3 = _FakeTokenizer(124893, [518, 519, 100])
    text = "some rendered chat text"

    # Qwen-vocab tokenizer writes its ids under its own namespace.
    ids_q = sft._tokenize_cached(tok_qwen, text, 2048)
    assert ids_q == [151644, 151645, 100]

    # The LFM3 tokenizer must NOT be served the Qwen ids from disk —
    # it gets its own miss and tokenizes fresh.
    ids_l = sft._tokenize_cached(tok_lfm3, text, 2048)
    assert ids_l == [518, 519, 100]
    assert max(ids_l) < tok_lfm3.vocab_size

    # And the Qwen ids are still retrievable under the Qwen namespace.
    assert sft._tokenize_cached(tok_qwen, text, 2048) == ids_q


def test_uniform_sample_weights_fall_back_to_chunked_path(monkeypatch):
    """All-1.0 sample_weights must be dropped so compute_loss uses the
    chunked-CE path (no [B,T,V] logits, teacher_logp honored)."""
    import torch

    calls = {}

    class _Cfg:
        entropy_alpha = 0.0
        teacher_logp = None
        use_chunked_ce = True
        ce_chunk_size = 128

    class _M:
        config = _Cfg()

        def __call__(self, input_ids, attention_mask=None, targets=None,
                     return_hidden_states=False):
            calls["targets_given"] = targets is not None
            return None, torch.tensor(1.23)

    ii = torch.zeros(2, 8, dtype=torch.long)
    lab = torch.zeros(2, 8, dtype=torch.long)
    am = torch.ones(2, 8, dtype=torch.long)
    ones = torch.ones(2)
    loss = sft.compute_loss(_M(), ii, lab, am, sample_weights=ones)
    assert float(loss) == pytest.approx(1.23)
    assert calls.get("targets_given") is True

    # Non-uniform weights still take the manual path (targets=None).
    calls.clear()

    class _M2:
        config = _Cfg()

        def __call__(self, input_ids, attention_mask=None, targets=None,
                     return_hidden_states=False):
            calls["targets_given"] = targets is not None
            return (torch.zeros(2, 8, 16), None)

    try:
        sft.compute_loss(_M2(), ii, lab, am,
                         sample_weights=torch.tensor([1.0, 2.0]))
    except Exception:
        pass  # manual path needs real logits; we only care it was entered
    assert calls.get("targets_given") is False


def test_meta_sidecar_tokenizer_resolution(tmp_path):
    """Regression: a bare saved .safetensors must not silently load the
    canonical tokenizer — the distill60 checkpoint generated whitespace
    until the engine read tokenizer_path from the .meta.json sidecar."""
    import json

    from forge.engine.engine_common import _tokenizer_from_meta

    # A checkpoint file with no sidecar -> None (vocab dispatch kicks in).
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x")
    assert _tokenizer_from_meta(ckpt) is None

    # Sidecar naming a dir without tokenizer.json -> None.
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    (tmp_path / "model.safetensors.meta.json").write_text(
        json.dumps({"tokenizer_path": str(bare_dir)}))
    assert _tokenizer_from_meta(ckpt) is None

    # Sidecar naming a real tokenizer dir -> that dir wins.
    tok_dir = tmp_path / "ckpt_tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors.meta.json").write_text(
        json.dumps({"tokenizer_path": str(tok_dir)}))
    assert _tokenizer_from_meta(ckpt) == str(tok_dir)


def test_sft_tokenizer_source_resolution(tmp_path):
    """Regression (spark sft3): a bare .safetensors --checkpoint whose parent
    dir has no tokenizer.json must resolve via the .meta.json sidecar, not
    the canonical default — the Qwen-vocab fallback mis-tokenized every row
    (train loss ~10 while true eval CE was 0.64)."""
    import json

    resolve = sft._resolve_tokenizer_source

    # Explicit --tokenizer always wins.
    assert resolve("some/tok", None) == "some/tok"
    # No checkpoint -> canonical default (None).
    assert resolve(None, None) is None
    assert resolve(None, "scratch") is None

    # Checkpoint dir holding tokenizer.json -> that dir.
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "tokenizer.json").write_text("{}")
    assert resolve(None, str(ckpt_dir)) == str(ckpt_dir)

    # Bare .safetensors in a parent WITHOUT tokenizer.json + sidecar naming
    # a real tokenizer dir -> sidecar path (the spark_sft3 case).
    bare = tmp_path / "saves"
    bare.mkdir()
    ckpt = bare / "model.safetensors"
    ckpt.write_bytes(b"x")
    tok_dir = tmp_path / "spark_tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}")
    (bare / "model.safetensors.meta.json").write_text(
        json.dumps({"tokenizer_path": str(tok_dir)}))
    assert resolve(None, str(ckpt)) == str(tok_dir)

    # Bare file with no sidecar -> None (canonical default, legacy path).
    other = bare / "other.safetensors"
    other.write_bytes(b"x")
    assert resolve(None, str(other)) is None

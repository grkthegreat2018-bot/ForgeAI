"""Lookahead decode tests — fake deterministic model exercises draft,
accept, KV-truncate and EOS paths (CPU, no CUDA needed)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from forge.engine.lookahead_decode import (
    _find_draft, lookahead_config, lookahead_generate)


class _PosModel(nn.Module):
    """Deterministic model: next token = pattern[global_pos % len(pat)].

    pasts = [(k,v)] with k/v shaped (1,1,seq,1); global position of input
    token t is past_len + t. Emulates repeat-heavy output (code echoes)."""

    def __init__(self, pattern, vocab=128):
        super().__init__()
        self.pattern = list(pattern)
        self.vocab = vocab
        self.d = nn.Parameter(torch.zeros(4))

    def forward(self, ids, past_key_values=None, use_cache=False, **_kw):
        past_len = 0 if past_key_values is None else int(
            past_key_values[0][0].shape[2])
        T = ids.shape[1]
        logits = torch.zeros(1, T, self.vocab)
        for t in range(T):
            nxt = self.pattern[(past_len + t + 1) % len(self.pattern)]
            logits[0, t, nxt] = 10.0
        new_k = torch.zeros(1, 1, past_len + T, 1)
        new_v = torch.zeros(1, 1, past_len + T, 1)
        return logits, [(new_k, new_v)]


class _Engine:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")

    def _eos_token_ids(self):
        return {99}


def _std(model, ids, n):
    from forge.engine.engine_generation import unpack_output_with_kv
    logits, past = unpack_output_with_kv(model(ids, use_cache=True))
    out = []
    for _ in range(n):
        tok = logits[:, -1, :].argmax(-1, keepdim=True)
        out.append(tok.item())
        logits, past = unpack_output_with_kv(
            model(tok, past_key_values=past, use_cache=True))
    return out


def test_find_draft():
    # context "a b c d a b" -> last-2 "a b" last seen at 0 -> draft = its
    # continuation "c d a b" (all history; k=4 caps at 4 tokens)
    assert _find_draft([1, 2, 3, 4, 1, 2], 2, 4) == [3, 4, 1, 2]
    assert _find_draft([1, 2, 3, 4], 2, 4) is None


def test_lookahead_matches_sequential_repetitive():
    # period-8 cycle: drafts hit constantly, but 8 distinct tokens stay
    # under the degenerate guard (<8 unique in last-20 => 8 is not <8)
    m = _PosModel([3, 4, 5, 6, 7, 8, 9, 10])
    eng = _Engine(m)
    ids = torch.tensor([[1, 3, 4, 5]])
    ref = _std(m, ids, 18)
    la = lookahead_generate(eng, ids, 18, k=6, w=2)
    assert la.shape[1] - ids.shape[1] == 18
    assert la[0, ids.shape[1]:].tolist() == ref


def test_lookahead_degenerate_guard():
    # period-3 cycle IS degenerate (3 unique < 8): must stop like std
    m = _PosModel([7, 9, 11])
    eng = _Engine(m)
    ids = torch.tensor([[1, 7, 9, 11]])
    out = lookahead_generate(eng, ids, 30, k=6, w=2)
    assert out.shape[1] - ids.shape[1] == 20


def test_lookahead_matches_sequential_novel():
    # all-distinct tokens: drafts never match, still must equal sequential
    m = _PosModel(list(range(3, 30)))
    eng = _Engine(m)
    ids = torch.tensor([[1, 3, 4, 5]])
    ref = _std(m, ids, 20)
    la = lookahead_generate(eng, ids, 20, k=4, w=2)
    assert la[0, ids.shape[1]:].tolist() == ref


def test_lookahead_respects_eos():
    m = _PosModel([99, 7, 7])
    eng = _Engine(m)
    ids = torch.tensor([[1, 99, 7, 7]])
    out = lookahead_generate(eng, ids, 24, k=6, w=2)
    # EOS is appended then generation stops (std-decoder semantics):
    # last emitted token must be 99, and output must be shorter than max.
    emitted = out[0, ids.shape[1]:].tolist()
    assert emitted[-1] == 99 and len(emitted) < 24


def test_config_parse(monkeypatch):
    import forge.engine.lookahead_decode as L
    monkeypatch.setattr(L, "_spec", "8,3")
    assert lookahead_config() == (8, 3)
    monkeypatch.setattr(L, "_spec", "4")
    assert lookahead_config() == (4, 3)
    monkeypatch.setattr(L, "_spec", "")
    assert lookahead_config() is None
    monkeypatch.setattr(L, "_spec", "junk")
    assert lookahead_config() is None

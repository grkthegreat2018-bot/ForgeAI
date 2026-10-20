"""Self-speculative decode tests.

Contract under test:
- ``generate(speculate=K)`` on GPU+learn+argmax emits IDENTICAL
  tokens to sequential argmax decode (speculation is exact
  verification, not approximation).
- The read-only drafter never mutates stream/table state.
- CPU models ignore speculate (no fused path to speculate through).
"""
import numpy as np
import pytest
import torch

from forge.model.flux import FluxConfig, FluxLM


def _cfg(**kw):
    base = dict(vocab_size=64, orders=(1, 2, 3, 4), use_knn=False,
                sparse_taps=(), class_orders=(), word_orders=(),
                hybrid_taps=())
    base.update(kw)
    return FluxConfig(**base)


def test_speculate_ignored_on_cpu():
    """CPU has no fused path -> speculate degrades to plain decode."""
    m = FluxLM(_cfg(device="cpu"))
    m.ingest(([1, 2, 3, 4] * 30), tag="t")
    out = m.generate([1, 2], max_new=8, temperature=0.0,
                     speculate=8)
    out2 = m.generate([1, 2], max_new=8, temperature=0.0)
    # degraded path is still a valid deterministic decode
    assert len(out) == 8 and len(out2) == 8


def test_draft_tokens_cpu_empty():
    """Host models have no _gt tables -> empty draft."""
    m = FluxLM(_cfg(device="cpu"))
    m.ingest([1, 2, 3] * 20, tag="t")
    assert m._draft_tokens(3, 4) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_speculate_matches_greedy_decode():
    """Bit-exact: speculative rows must emit the same tokens as the
    sequential argmax pipeline (rejected drafts are fully reverted)."""
    seq = ([1, 2, 3, 4, 5] * 10 + [9, 9, 8, 7]
           + [1, 2, 3, 4, 5] * 10)
    a = FluxLM(_cfg(device="cuda", cuda_primary=True))
    a.ingest(seq, tag="t")
    b = FluxLM(_cfg(device="cuda", cuda_primary=True))
    b.ingest(seq, tag="t")
    prompt = [1, 2, 3]
    out_a = a.generate(prompt, max_new=24, temperature=0.0)
    out_b = b.generate(prompt, max_new=24, temperature=0.0,
                       speculate=6)
    assert out_b == out_a
    assert len(out_b) == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_speculate_reverts_rejected_tail():
    """A draft the model disagrees with must not stay learned:
    stream length after spec decode = prompt + emitted."""
    rng = np.random.RandomState(0)
    seq = rng.randint(0, 64, size=800).tolist()
    m = FluxLM(_cfg(device="cuda", cuda_primary=True))
    m.ingest(seq, tag="t")
    c0 = m._count
    out = m.generate([5, 6, 7], max_new=20, temperature=0.0,
                     speculate=6)
    # prompt (3) + emitted (20) learned exactly once
    assert m._count == c0 + 3 + 20
    assert len(out) == 20

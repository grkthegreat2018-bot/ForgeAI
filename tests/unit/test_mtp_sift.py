"""MTP + SIFT interaction tests (CPU).

Covers the 2026-11 fixes:
- mtp.py _resolve_tied_weight: `tensor or w` was an ambiguous-bool crash
- post-load MTPModule construction + tied-head forward produces finite loss
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from forge.decoding.mtp import MTPModule


def _tiny(n_heads=2, adapter=True):
    torch.manual_seed(0)
    parent = nn.Module()
    parent.head = nn.Linear(16, 32, bias=False)
    m = MTPModule(d_model=16, vocab_size=32, n_heads=n_heads,
                  loss_weight=0.1, adapter_heads=adapter)
    m.tie_head_from_module(parent, "head")
    return m, parent


def test_resolve_tied_weight_no_ambiguous_bool():
    m, parent = _tiny()
    w = m._resolve_tied_weight(torch.device("cpu"))
    assert w is parent.head.weight  # live reference, no `or` crash


def test_mtp_forward_loss_finite_and_grads_flow():
    m, parent = _tiny(n_heads=1)
    hidden = torch.randn(2, 8, 16)
    embeds = torch.randn(2, 8, 16)
    targets = torch.randint(0, 32, (2, 8))
    loss, logits = m(hidden, embeds, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert m.trunk[0].weight.grad is not None
    if m.adapter_heads:
        assert m.heads[0].weight.grad is not None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

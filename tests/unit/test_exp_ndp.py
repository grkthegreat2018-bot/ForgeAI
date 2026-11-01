"""CPU tests for forge/exp/ndp.py."""

import math

import torch

from forge.exp.ndp import attn_partial, ndp_attention, transfer_bytes


def _ref_attention(q, K, V):
    scores = torch.einsum("hd,htd->ht", q, K) / math.sqrt(q.shape[-1])
    p = scores.softmax(-1)
    return torch.einsum("ht,htd->hd", p, V)


def test_partial_combine_exact():
    torch.manual_seed(0)
    H, D, Tl, Tf = 4, 16, 50, 200
    q = torch.randn(H, D)
    Kl, Vl = torch.randn(H, Tl, D), torch.randn(H, Tl, D)
    Kf, Vf = torch.randn(H, Tf, D), torch.randn(H, Tf, D)
    ref = _ref_attention(q, torch.cat([Kl, Kf], 1), torch.cat([Vl, Vf], 1))
    out = ndp_attention(q, (Kl, Vl), (Kf, Vf))
    assert torch.allclose(ref, out, atol=1e-5)


def test_partial_single_tier_matches_softmax():
    torch.manual_seed(1)
    q = torch.randn(4, 16)
    K, V = torch.randn(4, 128, 16), torch.randn(4, 128, 16)
    _m, s, acc = attn_partial(q, K, V)
    assert torch.allclose(acc / s.unsqueeze(-1), _ref_attention(q, K, V), atol=1e-5)


def test_transfer_bytes_ratio():
    ndp_b = transfer_bytes(14, 64, 16384, 2, migrate=False)
    mig_b = transfer_bytes(14, 64, 16384, 2, migrate=True)
    assert ndp_b < 10_000
    assert mig_b > 4_000_000
    assert mig_b / ndp_b > 500

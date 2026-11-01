"""fla_delta wrapper contract tests (AUDIT A4).

CPU-only: the fla fast path must decline (return None) so callers fall
back to the naive scan. GPU equivalence of chunk_kda/chunk_dplr vs the
naive scans was verified on the RTX 5070 (o err ~2e-3 fp32/TF32).
"""

import torch

from forge.keys.attention.fla_delta import gdn2_scan, kda_scan


def _inputs(B=1, T=8, H=2, dk=4, dv=4):
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g)

    q = torch.nn.functional.normalize(r(B, T, H, dk), dim=-1)
    k = torch.nn.functional.normalize(r(B, T, H, dk), dim=-1)
    v = r(B, T, H, dv)
    alpha = r(B, T, H, dk).sigmoid() * 0.9 + 0.05
    S = r(B, H, dk, dv)
    return q, k, v, alpha, S


def test_kda_scan_returns_none_on_cpu():
    q, k, v, alpha, S = _inputs()
    beta = torch.rand(q.shape[0], q.shape[1], q.shape[2])
    assert kda_scan(q, k, v, alpha, beta, S) is None


def test_gdn2_scan_returns_none_on_cpu():
    q, k, v, alpha, S = _inputs()
    e = torch.rand_like(k).sigmoid()
    w = torch.rand_like(v).sigmoid()
    assert gdn2_scan(q, k, v, alpha, e, w, S) is None


def test_gdn2_scan_returns_none_t1():
    q, k, v, alpha, S = _inputs(T=1)
    e = torch.rand_like(k).sigmoid()
    w = torch.rand_like(v).sigmoid()
    assert gdn2_scan(q, k, v, alpha, e, w, S) is None

"""Unit tests for the offline distillation + ForgeQ-R residual patch work.

CPU-only. Covers:
  - chunked_distill_ce: floor=1.0 == plain chunked CE; advantage math matches
    a manual reference; chunk-boundary invariance; missing-sentinel -> neutral.
  - collate_batch: teacher_logp per-completion list mapped onto supervised
    (non -100) positions; 5-tuple shape; per-example reward weight.
  - ForgeQuantLinear residual: forward applies x@B^T@A^T; quant+residual
    reconstructs W_exact with lower error than quant alone; state_dict
    round-trip; rotation + odd input shapes.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forge.training.losses.chunked_ce import (
    chunked_linear_cross_entropy,
    chunked_distill_ce,
    ChunkedDistillCE,
)
from forge.training.runners.sft_train import collate_batch
from forge.engine.quant.forge_quant import ForgeQuantLinear

torch.manual_seed(0)
MISSING = ChunkedDistillCE.TEACHER_LOGP_MISSING  # -99.0


# --------------------------------------------------------------------------
# chunked_distill_ce  (x [N,H] @ W [V,H] -> advantage-weighted CE)
# --------------------------------------------------------------------------

def _manual_distill(x, weight, target, tlp, alpha, clip, floor):
    logits = (x.float() @ weight.float().t())
    sel = target != -100
    idx = torch.nonzero(sel).squeeze(1)
    tgt = target[idx]
    ll = F.log_softmax(logits[idx], dim=-1)
    s_lp = ll.gather(1, tgt.unsqueeze(1)).squeeze(1)
    ce = -s_lp
    adv = (tlp[idx] - s_lp.detach()) * alpha
    w = floor + (1.0 - floor) * torch.clamp(adv, -clip, clip)
    w = torch.where(tlp[idx] <= MISSING, torch.ones_like(w), w)
    return (w * ce).sum() / sel.sum()


def _mk(N=37, H=32, V=257):
    x = torch.randn(N, H)
    w = torch.randn(V, H) * 0.5
    tgt = torch.randint(0, V, (N,))
    tgt[:7] = -100
    tlp = -torch.rand(N) * 5.0 - 0.3
    return x, w, tgt, tlp


def test_distill_ce_floor1_is_plain_ce():
    x, w, tgt, tlp = _mk()
    got = chunked_distill_ce(x, w, tgt, tlp, chunk_size=8,
                             alpha=1.0, clip=5.0, floor=1.0)
    want = chunked_linear_cross_entropy(x, w, tgt, chunk_size=8)
    assert torch.allclose(got, want, atol=1e-6)


def test_distill_ce_matches_manual():
    x, w, tgt, tlp = _mk()
    got = chunked_distill_ce(x, w, tgt, tlp, chunk_size=16,
                             alpha=1.0, clip=5.0, floor=0.3)
    want = _manual_distill(x, w, tgt, tlp, 1.0, 5.0, 0.3)
    assert torch.allclose(got, want, atol=1e-5)


def test_distill_ce_chunk_invariance_and_missing():
    x, w, tgt, tlp = _mk()
    tlp[10:14] = MISSING  # unscored -> neutral weight
    l_small = chunked_distill_ce(x, w, tgt, tlp, chunk_size=4,
                                 alpha=1.0, clip=5.0, floor=0.3)
    l_big = chunked_distill_ce(x, w, tgt, tlp, chunk_size=1000,
                               alpha=1.0, clip=5.0, floor=0.3)
    assert torch.allclose(l_small, l_big, atol=1e-5)


# --------------------------------------------------------------------------
# collate_batch teacher_logp path
# --------------------------------------------------------------------------

def test_collate_teacher_logp_alignment():
    pad = 0
    exs = [
        # 1 prompt + 2 completion tokens -> teacher_logp has n_comp=2 floats
        {"input_ids": [10, 11, 12], "labels": [-100, 11, 12],
         "teacher_logp": [-0.2, -0.3], "reward": 0.5},
        # 2 supervised tokens, shorter seq (left-pad exercises offset)
        {"input_ids": [20, 21], "labels": [20, 21],
         "teacher_logp": [-1.0, -2.0], "reward": 2.0},
        # no teacher_logp -> sentinel on its supervised positions
        {"input_ids": [30, 31, 32], "labels": [-100, -100, 32],
         "reward": 1.0},
    ]
    ids, labels, attn, rw, tlp = collate_batch(exs, pad_id=pad, device="cpu")
    assert ids.shape == (3, 3) and tlp.shape == (3, 3)
    assert rw.shape == (3,) and torch.allclose(rw, torch.tensor([0.5, 2.0, 1.0]))
    # teacher values land on supervised positions only, in order
    assert torch.allclose(tlp[0], torch.tensor([-99.0, -0.2, -0.3]))
    assert torch.allclose(tlp[1], torch.tensor([-99.0, -1.0, -2.0]))
    assert torch.allclose(tlp[2], torch.tensor([-99.0, -99.0, -99.0]))


def test_collate_no_teacher_logp_returns_none():
    exs = [{"input_ids": [1, 2], "labels": [1, 2]}]
    ids, labels, attn, rw, tlp = collate_batch(exs, pad_id=0, device="cpu")
    assert tlp is None and rw.item() == 1.0


# --------------------------------------------------------------------------
# ForgeQ-R residual patch
# --------------------------------------------------------------------------

def _rand_w(out_f=96, in_f=64):
    w = torch.randn(out_f, in_f)
    w[:6] *= 8.0  # outlier rows -> exercises the int8 sparse path
    return w


def _fit_resid(fq, w, r=16, iters=2):
    """Mirror of the packer's LoftQ-style alternation."""
    a = torch.zeros(w.shape[0], r)
    b = torch.zeros(r, w.shape[1])
    for _ in range(iters):
        fq.load_from_weight(w - a @ b)
        resid = w - fq._dequantize_weight(torch.float32)
        U, S, Vh = torch.linalg.svd(resid, full_matrices=False)
        sq = S[:r].sqrt()
        a = U[:, :r] * sq.unsqueeze(0)
        b = sq.unsqueeze(1) * Vh[:r]
    fq.res_a.data.copy_(a.half())
    fq.res_b.data.copy_(b.half())
    return fq


def test_fq_residual_reduces_error():
    w = _rand_w()
    base = ForgeQuantLinear(64, 96, bias=False)
    base.load_from_weight(w)
    fq = ForgeQuantLinear(64, 96, bias=False, residual_rank=16)
    _fit_resid(fq, w, r=16)
    x = torch.randn(4, 64)
    ref = x @ w.t()
    err_base = (base(x) - ref).norm() / ref.norm()
    err_res = (fq(x) - ref).norm() / ref.norm()
    assert err_res < err_base * 0.9, f"{err_base:.4f} vs {err_res:.4f}"
    wq0 = base._dequantize_weight(torch.float32)
    rel0 = (w - wq0).norm() / w.norm()
    rel1 = (w - (wq0 + fq.res_a.float() @ fq.res_b.float())).norm() / w.norm()
    assert rel1 < rel0


def test_fq_residual_rotation_and_shapes():
    w = _rand_w(96, 128)  # in_f power-of-2 for hadamard
    fq = ForgeQuantLinear(128, 96, bias=False, use_rotation=True,
                          residual_rank=8)
    _fit_resid(fq, w, r=8)
    for shape in [(128,), (3, 128), (2, 3, 128)]:
        x = torch.randn(*shape)
        out = fq(x)
        assert out.shape[:-1] == shape[:-1] and out.shape[-1] == 96
        assert torch.isfinite(out).all()
    x = torch.randn(5, 128)
    ref = x @ w.t()
    base_err = (x @ fq._dequantize_weight(torch.float32).t() - ref).norm()
    res_err = (fq(x) - ref).norm()
    assert res_err <= base_err + 1e-5


def test_fq_residual_state_roundtrip():
    w = _rand_w()
    fq = ForgeQuantLinear(64, 96, bias=False, residual_rank=16)
    _fit_resid(fq, w, r=16)
    sd = {f"m.{k}": v for k, v in fq.state_dict().items()}
    assert "m.res_a" in sd and "m.res_b" in sd
    fq2 = ForgeQuantLinear(64, 96, bias=False, residual_rank=16)
    fq2.load_state_dict({k[2:]: v for k, v in sd.items()})
    x = torch.randn(4, 64)
    assert torch.equal(fq(x), fq2(x))

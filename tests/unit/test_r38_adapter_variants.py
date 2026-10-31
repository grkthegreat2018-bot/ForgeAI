"""Tests for R38-3/4/5: PiSSA, AdaLoRA, rsLoRA adapter variants.

Covers:
  R38-3 PiSSA  — SVD init correctness, merge identity at full rank,
                  top-rank singular value selection.
  R38-4 AdaLoRA — budget allocation, prune/grow logic, importance scoring,
                  budget reallocation from gradient norms.
  R38-5 rsLoRA  — 1/sqrt(r) scaling (vs 1/r), forward pass, merge
                  correctness, stability at high rank.

All tests run on CPU with small (64x128) matrices.
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import math

import torch
import torch.nn as nn

from forge.training.adapter_variants import (
    AdaLoRAClass,
    MiCAInitializer,
    MiCALinear,
    PiSSAInitializer,
    rsLoRALinear,
)

# ── R38-3: PiSSA ─────────────────────────────────────────────────────────

def _random_low_rank_weight(out_f: int, in_f: int, rank: int,
                            seed: int = 0) -> torch.Tensor:
    """Build a weight that is exactly rank-`rank` (so top-r SVD is exact)."""
    torch.manual_seed(seed)
    A = torch.randn(out_f, rank)
    B = torch.randn(rank, in_f)
    return A @ B


def test_pissa_shapes():
    W = torch.randn(64, 128)
    A, B = PiSSAInitializer.initialize(W, rank=8)
    assert A.shape == (8, 128)
    assert B.shape == (64, 8)


def test_pissa_top_rank_reconstruction():
    """For a rank-r matrix, PiSSA top-r init reconstructs W exactly."""
    W = _random_low_rank_weight(64, 128, rank=4, seed=3)
    A, B = PiSSAInitializer.initialize(W, rank=4)
    recon = B @ A  # (out, in) — top-r component
    assert torch.allclose(recon, W, atol=1e-4), \
        f"top-r reconstruction failed: max err {(recon - W).abs().max()}"


def test_pissa_selects_top_singular_values():
    """PiSSA must keep the LARGEST singular values, not arbitrary ones."""
    torch.manual_seed(7)
    W = torch.randn(64, 128)
    _U, S, _Vh = torch.linalg.svd(W, full_matrices=False)
    A, B = PiSSAInitializer.initialize(W, rank=4)
    # energy kept by PiSSA == sum of top-4 squared singular values
    kept = float((B @ A).pow(2).sum())
    top4_energy = float((S[:4] ** 2).sum())
    assert abs(kept - top4_energy) < 1e-2, \
        f"PiSSA kept {kept}, top-4 energy {top4_energy}"
    # and strictly more than the bottom-4 energy
    bottom4_energy = float((S[-4:] ** 2).sum())
    assert kept > bottom4_energy


def test_pissa_merge_identity_at_full_rank():
    """At full rank, merge(A, B, W_residual) == original W.

    PiSSA splits W into top-r (adapter) + residual (base). When r == full
    rank the residual is zero, so merging the adapter back into the
    (zeroed) residual recovers W exactly.
    """
    W = torch.randn(64, 128)
    full_r = min(64, 128)
    A, B = PiSSAInitializer.initialize(W, rank=full_r)
    # residual base = W - top-r component (zero at full rank)
    residual = W - (B @ A)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)
    merged = PiSSAInitializer.merge(A, B, residual)
    assert torch.allclose(merged, W, atol=1e-4)


def test_pissa_merge_adds_delta():
    """merge returns base + scale * (B @ A)."""
    W = torch.randn(64, 128)
    A, B = PiSSAInitializer.initialize(W, rank=8)
    base = torch.randn(64, 128)
    merged = PiSSAInitializer.merge(A, B, base, scale=2.0)
    expected = base + 2.0 * (B @ A)
    assert torch.allclose(merged, expected, atol=1e-5)


def test_pissa_rank_clamp():
    """rank > min(dims) is clamped, not an error, by default."""
    W = torch.randn(64, 128)
    A, _B = PiSSAInitializer.initialize(W, rank=999)
    assert A.shape[0] == 64  # clamped to min(64,128)


def test_pissa_rank_clamp_disabled_raises():
    W = torch.randn(64, 128)
    try:
        PiSSAInitializer.initialize(W, rank=999, clamp_rank=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError when clamp_rank=False")


# ── MiCA (Minor-Component Adaptation) ────────────────────────────────────

def test_mica_shapes_and_zero_a():
    W = torch.randn(64, 128)
    A, B = MiCAInitializer.initialize(W, rank=8)
    assert A.shape == (8, 128)
    assert B.shape == (64, 8)
    assert torch.all(A == 0)          # A starts at zero — exact no-op


def test_mica_selects_minor_subspace():
    """B must span the BOTTOM-r singular directions (inverse of PiSSA)."""
    torch.manual_seed(11)
    W = torch.randn(64, 128)
    U, _S, _Vh = torch.linalg.svd(W, full_matrices=False)
    _A, B = MiCAInitializer.initialize(W, rank=4)
    # B columns must equal the last-4 left singular vectors (up to sign)
    minor = U[:, -4:]
    # |B^T minor| should be ~identity → B lies in the minor subspace
    overlap = (B.T @ minor).abs()
    assert torch.allclose(overlap, torch.eye(4), atol=1e-4), \
        f"MiCA B is not in the minor subspace: {overlap}"
    # and orthogonal to the TOP-r subspace
    top = U[:, :4]
    top_overlap = (B.T @ top).abs().max().item()
    assert top_overlap < 1e-4, f"MiCA B leaks into top subspace: {top_overlap}"


def test_mica_b_is_orthonormal():
    """B = U[:, -r:] columns are orthonormal (SVD property)."""
    W = torch.randn(64, 128)
    _A, B = MiCAInitializer.initialize(W, rank=8)
    gram = B.T @ B
    assert torch.allclose(gram, torch.eye(8), atol=1e-5)


def test_mica_noop_at_init_linear():
    """MiCALinear is an exact no-op at init (A=0)."""
    layer = MiCALinear(128, 64, rank=8)
    x = torch.randn(4, 128)
    assert torch.allclose(layer(x), layer.base(x), atol=1e-6)


def test_mica_b_frozen_a_trainable():
    layer = MiCALinear(128, 64, rank=8)
    assert layer.lora_A.requires_grad
    # B is a buffer — can never collect grads or enter optimizer groups
    assert not layer.lora_B.requires_grad
    param_ids = {id(p) for p in layer.parameters()}
    assert id(layer.lora_B) not in param_ids


def test_mica_forward_and_merge():
    """After A moves, forward = base + scale*(x @ A.T @ B.T), merge folds."""
    layer = MiCALinear(128, 64, rank=4, alpha=1.0)
    with torch.no_grad():
        layer.lora_A.normal_(0, 0.1)
    x = torch.randn(3, 128)
    expected = layer.base(x) + layer.scale * (x @ layer.lora_A.T @ layer.lora_B.T)
    assert torch.allclose(layer(x), expected, atol=1e-5)
    merged = layer.merge()
    assert torch.allclose(merged(x), layer(x), atol=1e-5)


def test_mica_rank_clamp_and_raise():
    W = torch.randn(64, 128)
    _A, B = MiCAInitializer.initialize(W, rank=999)
    assert B.shape[1] == 64  # clamped to min(64,128)
    try:
        MiCAInitializer.initialize(W, rank=999, clamp_rank=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError when clamp_rank=False")


# ── R38-4: AdaLoRA ───────────────────────────────────────────────────────

def test_adalora_budget_allocation_even():
    ada = AdaLoRAClass(total_budget=32, n_layers=4)
    assert sum(ada.ranks) == 32
    assert ada.ranks == [8, 8, 8, 8]


def test_adalora_budget_allocation_remainder():
    ada = AdaLoRAClass(total_budget=34, n_layers=4)
    assert sum(ada.ranks) == 34
    # remainder 2 → first two layers get +1
    assert ada.ranks == [9, 9, 8, 8]


def test_adalora_budget_min_rank_floor():
    ada = AdaLoRAClass(total_budget=8, n_layers=4, min_rank=3)
    # 8/4 = 2 each, but floored at min_rank=3 → all 3, sum 12 > 8 (clamped)
    assert all(r >= 3 for r in ada.ranks)


def test_adalora_budget_max_rank_ceil():
    ada = AdaLoRAClass(total_budget=32, n_layers=4, max_rank=5)
    assert all(r <= 5 for r in ada.ranks)


def test_adalora_prune_zeros_small_svs():
    ada = AdaLoRAClass(total_budget=16, n_layers=2, prune_threshold=0.5)
    sv = torch.tensor([1.0, 0.2, 0.8, 0.05])
    adapter = {"singular_values": sv}
    ada.prune_singular_values(adapter)
    out = adapter["singular_values"]
    # 0.2 and 0.05 are < 0.5*1.0 → pruned to 0
    assert out[1].item() == 0.0
    assert out[3].item() == 0.0
    # 1.0 and 0.8 survive
    assert out[0].item() == 1.0
    assert abs(out[2].item() - 0.8) < 1e-6
    assert "mask" in adapter


def test_adalora_grow_revives_pruned():
    ada = AdaLoRAClass(total_budget=16, n_layers=2, grow_step=0.1)
    sv = torch.tensor([1.0, 0.0, 0.8, 0.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adapter = {"singular_values": sv, "mask": mask}
    ada.grow_singular_values(adapter)
    out = adapter["singular_values"]
    # pruned (mask==0) dirs revived to grow_step * max_sv = 0.1*1.0 = 0.1
    assert abs(out[1].item() - 0.1) < 1e-6
    assert abs(out[3].item() - 0.1) < 1e-6
    # active dirs unchanged
    assert out[0].item() == 1.0
    assert abs(out[2].item() - 0.8) < 1e-6
    # mask updated
    assert adapter["mask"][1].item() == 1.0


def test_adalora_importance_score_grad_weight():
    """Sensitivity = mean(|grad| * |weight|)."""
    A = torch.randn(4, 8, requires_grad=True)
    B = torch.randn(6, 4, requires_grad=True)
    A.grad = torch.full_like(A, 0.5)
    B.grad = torch.full_like(B, 2.0)
    adapter = {"A": A, "B": B}
    score = AdaLoRAClass._importance(adapter)
    # A: mean(|0.5|*|A|), B: mean(|2.0|*|B|) → average of the two
    expected_a = float((0.5 * A.detach().abs()).mean())
    expected_b = float((2.0 * B.detach().abs()).mean())
    assert abs(score - (expected_a + expected_b) / 2) < 1e-6


def test_adalora_importance_no_grad_falls_back_to_weight():
    A = torch.randn(4, 8)
    B = torch.randn(6, 4)
    adapter = {"A": A, "B": B}
    score = AdaLoRAClass._importance(adapter)
    # no grad → uses |w|*|w| = w^2
    expected_a = float((A.abs() * A.abs()).mean())
    expected_b = float((B.abs() * B.abs()).mean())
    assert abs(score - (expected_a + expected_b) / 2) < 1e-6


def test_adalora_update_budget_proportional():
    ada = AdaLoRAClass(total_budget=40, n_layers=4)
    # layer 0 has 10x the gradient norm → should get ~10x budget share
    norms = [10.0, 1.0, 1.0, 1.0]
    new_ranks = ada.update_budget(norms)
    assert sum(new_ranks) == 40  # budget conserved
    assert new_ranks[0] > new_ranks[1]
    assert new_ranks[0] == max(new_ranks)


def test_adalora_update_budget_dict_input():
    ada = AdaLoRAClass(total_budget=20, n_layers=4)
    norms = {0: 4.0, 1: 1.0, 2: 1.0, 3: 0.0}
    new_ranks = ada.update_budget(norms)
    assert sum(new_ranks) == 20
    assert new_ranks[0] >= new_ranks[1]


def test_adalora_update_budget_zero_norms_even_split():
    ada = AdaLoRAClass(total_budget=16, n_layers=4)
    new_ranks = ada.update_budget([0.0, 0.0, 0.0, 0.0])
    assert sum(new_ranks) == 16
    assert new_ranks == [4, 4, 4, 4]


def test_adalora_update_budget_respects_clamps():
    ada = AdaLoRAClass(total_budget=40, n_layers=4, min_rank=2, max_rank=12)
    norms = [100.0, 0.0, 0.0, 0.0]
    new_ranks = ada.update_budget(norms)
    assert sum(new_ranks) == 40
    assert all(2 <= r <= 12 for r in new_ranks)


def test_adalora_invalid_n_layers_raises():
    """n_layers=0 should raise ValueError."""
    try:
        AdaLoRAClass(total_budget=16, n_layers=0, min_rank=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n_layers=0")


# ── R38-5: rsLoRA ────────────────────────────────────────────────────────

def test_rslora_scale_is_one_over_sqrt_r():
    layer = rsLoRALinear(128, 64, rank=16, alpha=1.0)
    expected = 1.0 / math.sqrt(16)
    assert abs(layer.scale - expected) < 1e-6
    # and crucially NOT 1/r
    assert abs(layer.scale - 1.0 / 16) > 1e-3


def test_rslora_scale_with_alpha():
    layer = rsLoRALinear(128, 64, rank=16, alpha=8.0)
    expected = 8.0 / math.sqrt(16)
    assert abs(layer.scale - expected) < 1e-6


def test_rslora_shapes():
    layer = rsLoRALinear(128, 64, rank=8)
    assert layer.lora_A.shape == (8, 128)
    assert layer.lora_B.shape == (64, 8)
    assert layer.base.weight.shape == (64, 128)


def test_rslora_zero_init_is_noop():
    """B is zero-init → adapter contributes nothing initially."""
    layer = rsLoRALinear(128, 64, rank=8)
    x = torch.randn(4, 128)
    base_out = layer.base(x)
    full_out = layer(x)
    assert torch.allclose(base_out, full_out, atol=1e-6)


def test_rslora_forward_matches_formula():
    """forward = base(x) + scale * (x @ A.T @ B.T)."""
    layer = rsLoRALinear(128, 64, rank=4, alpha=1.0)
    with torch.no_grad():
        layer.lora_A.normal_(0, 0.1)
        layer.lora_B.normal_(0, 0.1)
    x = torch.randn(3, 128)
    expected = layer.base(x) + layer.scale * (x @ layer.lora_A.T @ layer.lora_B.T)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_rslora_merge_correctness():
    """merge() returns nn.Linear with weight = base + scale*(B@A)."""
    layer = rsLoRALinear(128, 64, rank=4, alpha=2.0)
    with torch.no_grad():
        layer.lora_A.normal_(0, 0.1)
        layer.lora_B.normal_(0, 0.1)
    merged = layer.merge()
    assert isinstance(merged, nn.Linear)
    delta = layer.scale * (layer.lora_B @ layer.lora_A)
    expected_w = layer.base.weight + delta
    assert torch.allclose(merged.weight, expected_w, atol=1e-5)
    # forward must match the adapter forward
    x = torch.randn(5, 128)
    assert torch.allclose(merged(x), layer(x), atol=1e-5)


def test_rslora_merge_preserves_bias():
    layer = rsLoRALinear(128, 64, rank=4, bias=True)
    with torch.no_grad():
        layer.base.bias.fill_(0.7)
    merged = layer.merge()
    assert merged.bias is not None
    assert torch.allclose(merged.bias, torch.full((64,), 0.7))


def test_rslora_base_frozen():
    layer = rsLoRALinear(128, 64, rank=4)
    assert not layer.base.weight.requires_grad
    # adapter params trainable
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_rslora_stability_high_rank():
    """At high rank, rsLoRA update magnitude stays ~constant vs rank.

    Standard LoRA (1/r) would shrink the update as r grows; rsLoRA (1/sqrt r)
    keeps it bounded. We verify the adapter output std at r=4 vs r=64 is
    within a small factor (not orders of magnitude apart).
    """
    torch.manual_seed(42)
    x = torch.randn(8, 128)

    def adapter_out_std(rank):
        torch.manual_seed(42)
        layer = rsLoRALinear(128, 64, rank=rank, alpha=1.0)
        with torch.no_grad():
            layer.lora_A.normal_(0, 0.02)
            layer.lora_B.normal_(0, 0.02)
        delta = layer(x) - layer.base(x)
        return float(delta.std())

    std_low = adapter_out_std(4)
    std_high = adapter_out_std(64)
    # rsLoRA: ratio should be modest (sqrt scaling), not 16x (which 1/r would give)
    ratio = std_high / max(std_low, 1e-8)
    assert ratio < 8.0, f"rsLoRA not scale-free: ratio {ratio:.2f} (low={std_low:.4e}, high={std_high:.4e})"


def test_rslora_vs_standard_lora_scaling_diverges_at_high_rank():
    """Explicitly show rsLoRA scale != standard LoRA scale at high rank."""
    r = 256
    rs = rsLoRALinear(128, 64, rank=r, alpha=1.0)
    standard_scale = 1.0 / r
    assert abs(rs.scale - standard_scale) > 0.01  # clearly different
    assert abs(rs.scale - 1.0 / math.sqrt(r)) < 1e-6

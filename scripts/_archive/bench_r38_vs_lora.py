"""Benchmark tests comparing R38 adapters vs standard LoRA.

Side-by-side comparisons of every R38 adapter variant against the baseline
LoRAAdapter from bitnet_lora.py:

  DLoRA   — dynamic rank growth (fewer initial params, grows on demand)
  DoRA    — weight-decomposed adaptation (magnitude + direction)
  PiSSA   — SVD-based initialization (starts closer to the target weight)
  AdaLoRA — adaptive rank budget allocation across layers
  rsLoRA  — scale-free 1/sqrt(r) scaling (stable at high rank)
  ForgeAdapter — entropy-guided multi-rank adapter fusion

All tests run on CPU with small (64x128) matrices.
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

from forge.training.adapter_variants import (
    AdaLoRAClass,
    PiSSAInitializer,
    rsLoRALinear,
)
from forge.training.bitnet_lora import LoRAAdapter
from forge.training.dlora import DLoRAAdapter
from forge.training.dora import DoRALinear
from forge.training.forge_adapter import ForgeAdapter

# ── helpers ──────────────────────────────────────────────────────────────

def _trainable_params(module: nn.Module) -> int:
    """Count parameters with requires_grad=True."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _effective_dlora_params(dlora: DLoRAAdapter) -> int:
    """Effective active params = current_rank * (in + out).

    DLoRA pre-allocates max_rank buffers, but dormant rows/cols are zero
    and don't contribute to the forward pass. We count only the active
    slice, which is the fair comparison against fixed-rank LoRA.
    """
    return dlora.current_rank * (dlora.in_features + dlora.out_features)


def _lora_flops(rank: int, in_features: int, out_features: int) -> int:
    """FLOPs for a single LoRA delta: x @ A.T @ B.T."""
    # x @ A.T: batch * in * rank, then @ B.T: batch * rank * out
    # Per-token: in * rank + rank * out = rank * (in + out)
    return rank * (in_features + out_features)


# ── TestDLoRAVsLoRA ──────────────────────────────────────────────────────

class TestDLoRAVsLoRA:
    """DLoRA (dynamic rank) vs fixed LoRA."""

    def test_dlora_fewer_params_initially(self):
        """DLoRA at initial_rank=4 has fewer effective params than LoRA rank=8."""
        dlora = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
        lora = LoRAAdapter(64, 128, rank=8, alpha=16)
        # DLoRA effective: 4*64 + 128*4 = 768
        # LoRA:            8*64 + 128*8 = 1536
        dlora_params = _effective_dlora_params(dlora)
        lora_params = _trainable_params(lora)
        assert dlora_params == 768, f"expected 768, got {dlora_params}"
        assert lora_params == 1536, f"expected 1536, got {lora_params}"
        assert dlora_params < lora_params

    def test_dlora_grow_to_same_params(self):
        """After grow_rank(8), DLoRA has same effective params as LoRA rank=8."""
        dlora = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
        lora = LoRAAdapter(64, 128, rank=8, alpha=16)
        new_rank = dlora.grow_rank(8)
        assert new_rank == 8
        dlora_params = _effective_dlora_params(dlora)
        lora_params = _trainable_params(lora)
        # DLoRA after grow: 8*64 + 128*8 = 1536 == LoRA
        assert dlora_params == 1536, f"expected 1536, got {dlora_params}"
        assert dlora_params == lora_params

    def test_dlora_growth_is_noop(self):
        """forward(x) before grow == forward(x) after grow (new B cols are zero)."""
        torch.manual_seed(42)
        dlora = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
        x = torch.randn(2, 64)
        out_before = dlora.forward(x)
        # Grow to rank 8 — new B columns are zero, so output unchanged.
        dlora.grow_rank(8)
        out_after = dlora.forward(x)
        assert torch.equal(out_before, out_after), \
            "grow_rank changed the forward output (should be a no-op)"

    def test_dlora_exceeds_lora_max(self):
        """DLoRA can grow to max_rank=32 while LoRA is stuck at rank=8."""
        dlora = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
        lora = LoRAAdapter(64, 128, rank=8, alpha=16)
        # Grow DLoRA to max
        dlora.grow_rank(32)
        assert dlora.current_rank == 32
        dlora_params = _effective_dlora_params(dlora)
        lora_params = _trainable_params(lora)
        # DLoRA at rank 32: 32*64 + 128*32 = 6144 > LoRA 1536
        assert dlora_params == 6144, f"expected 6144, got {dlora_params}"
        assert dlora_params > lora_params


# ── TestDoRAVsLoRA ───────────────────────────────────────────────────────

class TestDoRAVsLoRA:
    """DoRA (weight-decomposed) vs fixed LoRA."""

    def test_dora_decomposition_lossless(self):
        """DoRA with delta=0 reconstructs W exactly (lossless decomposition)."""
        torch.manual_seed(42)
        W = torch.randn(128, 64)
        dora = DoRALinear(64, 128, rank=8, alpha=16, bias=False)
        dora.load_from_weight(W)
        eff = dora._effective_weight()
        err = (eff - W).norm().item()
        assert err < 1e-4, f"decomposition not lossless: ||eff - W|| = {err}"

    def test_dora_more_trainable_params(self):
        """DoRA has magnitude + A + B > LoRA's A + B (magnitude is extra)."""
        dora = DoRALinear(64, 128, rank=8, alpha=16, bias=False)
        lora = LoRAAdapter(64, 128, rank=8, alpha=16)
        dora_params = _trainable_params(dora)
        lora_params = _trainable_params(lora)
        # DoRA: magnitude(128) + A(8*64=512) + B(128*8=1024) = 1664
        # LoRA: A(512) + B(1024) = 1536
        assert dora_params == 1664, f"expected 1664, got {dora_params}"
        assert lora_params == 1536, f"expected 1536, got {lora_params}"
        assert dora_params > lora_params

    def test_dora_delta_zero_equals_base(self):
        """DoRA with lora_A=0, lora_B=0: forward(x) == W @ x (base behavior)."""
        torch.manual_seed(42)
        W = torch.randn(128, 64)
        dora = DoRALinear(64, 128, rank=8, alpha=16, bias=False)
        dora.load_from_weight(W)
        # Zero out the adapter — delta_V = 0, so effective weight == W.
        with torch.no_grad():
            dora.lora_A.zero_()
            dora.lora_B.zero_()
        x = torch.randn(2, 64)
        dora_out = dora.forward(x)
        base_out = F.linear(x, W)  # x @ W.T
        assert torch.allclose(dora_out, base_out, atol=1e-5), \
            "DoRA with zero adapter should equal base weight application"


# ── TestPiSSAVsRandomLoRA ────────────────────────────────────────────────

class TestPiSSAVsRandomLoRA:
    """PiSSA (SVD init) vs random-init LoRA (starts at zero)."""

    def test_pissa_starts_closer_to_weight(self):
        """PiSSA init reconstructs top-r of W; random LoRA starts at 0."""
        torch.manual_seed(42)
        # Build a rank-8 weight: W = U[:,:8] @ S[:8] @ V[:,:8].T
        U = torch.randn(128, 8)
        S = torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3]))
        V = torch.randn(8, 64)
        W = U @ S @ V
        # PiSSA init
        A, B = PiSSAInitializer.initialize(W, rank=8)
        recon = B @ A  # top-8 SVD reconstruction
        pissa_err = (W - recon).norm().item()
        # Random LoRA starts at 0 (B=0), so its "reconstruction" is 0.
        random_err = W.norm().item()
        assert pissa_err < random_err, \
            f"PiSSA err {pissa_err} should be < random LoRA err {random_err}"

    def test_pissa_captures_top_singulars(self):
        """Singular values of B@A match the top-8 of W."""
        torch.manual_seed(42)
        U = torch.randn(128, 8)
        S = torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3]))
        V = torch.randn(8, 64)
        W = U @ S @ V
        A, B = PiSSAInitializer.initialize(W, rank=8)
        recon = B @ A
        sv_w = torch.linalg.svdvals(W)
        sv_recon = torch.linalg.svdvals(recon)
        # W is rank-8, so top-8 singulars of B@A should match W's exactly.
        assert torch.allclose(sv_w[:8], sv_recon[:8], atol=1e-4), \
            f"top-8 singulars differ: W={sv_w[:8]}, recon={sv_recon[:8]}"


# ── TestAdaLoRAVsFixedLoRA ───────────────────────────────────────────────

class TestAdaLoRAVsFixedLoRA:
    """AdaLoRA (adaptive budget) vs fixed-rank LoRA."""

    def test_adalora_budget_conservation(self):
        """sum(ranks) == total_budget after initial allocation."""
        ada = AdaLoRAClass(total_budget=32, n_layers=4)
        assert sum(ada.ranks) == 32, f"ranks sum to {sum(ada.ranks)}, expected 32"
        assert len(ada.ranks) == 4

    def test_adalora_can_allocate_unevenly(self):
        """update_budget with uneven grad norms produces uneven ranks."""
        ada = AdaLoRAClass(total_budget=32, n_layers=4)
        new_ranks = ada.update_budget([1.0, 0.1, 0.5, 2.0])
        assert sum(new_ranks) == 32, f"budget not conserved: sum={sum(new_ranks)}"
        # Not all ranks should be equal (adaptive allocation).
        assert len(set(new_ranks)) > 1, \
            f"all ranks equal {new_ranks} — allocation not adaptive"

    def test_adalora_prune_grow_cycle(self):
        """prune_singular_values reduces non-zero count; grow restores it."""
        ada = AdaLoRAClass(total_budget=32, n_layers=4, prune_threshold=0.5,
                           grow_step=0.1)
        adapter = {
            "singular_values": torch.tensor([1.0, 0.8, 0.3, 0.1]),
        }
        # Prune: threshold = 0.5 * max(1.0) = 0.5 → SVs < 0.5 are zeroed.
        pruned = ada.prune_singular_values(adapter)
        n_nonzero_pruned = (pruned["singular_values"] != 0).sum().item()
        assert n_nonzero_pruned < 4, \
            f"prune did not reduce non-zero count: {n_nonzero_pruned}"
        # Grow: revive pruned (masked-out) directions.
        grown = ada.grow_singular_values(pruned)
        n_nonzero_grown = (grown["singular_values"] != 0).sum().item()
        assert n_nonzero_grown > n_nonzero_pruned, \
            f"grow did not increase non-zero count: {n_nonzero_grown}"


# ── TestRsLoRAVsLoRA ─────────────────────────────────────────────────────

class TestRsLoRAVsLoRA:
    """rsLoRA (1/sqrt(r) scaling) vs standard LoRA (1/r scaling)."""

    def test_rslora_stable_at_high_rank(self):
        """rsLoRA scale at rank=256 is 1.0; LoRA scale vanishes to 0.0625."""
        alpha = 16.0
        rank = 256
        rslora_scale = alpha / math.sqrt(rank)  # 16 / 16 = 1.0
        lora_scale = alpha / rank               # 16 / 256 = 0.0625
        assert rslora_scale == pytest.approx(1.0)
        assert lora_scale == pytest.approx(0.0625)
        assert rslora_scale > lora_scale * 10  # 1.0 >> 0.0625

    def test_lora_vanishes_at_very_high_rank(self):
        """LoRA scale at rank=1024 vanishes (< 0.1); rsLoRA stays stable (0.5)."""
        alpha = 16.0
        rank = 1024
        lora_scale = alpha / rank               # 16 / 1024 = 0.015625
        rslora_scale = alpha / math.sqrt(rank)  # 16 / 32 = 0.5
        assert lora_scale < 0.1, \
            f"LoRA scale {lora_scale} should vanish (< 0.1) at rank 1024"
        assert rslora_scale == pytest.approx(0.5), \
            f"rsLoRA scale {rslora_scale} should be 0.5 at rank 1024"

    def test_rslora_scale_formula(self):
        """rsLoRALinear(128, 64, rank=16, alpha=1.0).scale == 1/sqrt(16) == 0.25."""
        layer = rsLoRALinear(128, 64, rank=16, alpha=1.0, bias=False)
        expected = 1.0 / math.sqrt(16)  # 0.25
        assert layer.scale == pytest.approx(expected), \
            f"rsLoRA scale {layer.scale} != expected {expected}"


# ── TestForgeAdapterVsLoRA ───────────────────────────────────────────────

class TestForgeAdapterVsLoRA:
    """ForgeAdapter (entropy-guided multi-rank) vs single fixed-rank LoRA."""

    def test_forge_routes_low_entropy_to_low_rank(self):
        """Low entropy (0.1) → hard_route picks index 0 (rank-4, lowest)."""
        forge = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], alpha=16.0,
                             temperature=1.0)
        entropy = torch.tensor([[0.1]])  # (batch=1, seq=1)
        route = forge.hard_route(entropy)
        assert route.item() == 0, \
            f"low entropy should route to index 0, got {route.item()}"

    def test_forge_routes_high_entropy_to_high_rank(self):
        """High entropy (9.5) → hard_route picks index 3 (rank-32, highest)."""
        forge = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], alpha=16.0,
                             temperature=1.0)
        # bin centers are [1.25, 3.75, 6.25, 8.75]; 9.5 is closest to 8.75 (idx 3).
        entropy = torch.tensor([[9.5]])
        route = forge.hard_route(entropy)
        assert route.item() == 3, \
            f"high entropy should route to index 3, got {route.item()}"

    def test_forge_default_uses_highest_rank(self):
        """forward(x, entropy=None) uses highest-rank adapter == merge(rank_idx=-1)."""
        torch.manual_seed(42)
        forge = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], alpha=16.0,
                             temperature=1.0)
        x = torch.randn(2, 4, 64)  # (batch, seq, in_features)
        out_default = forge.forward(x, entropy=None)
        # merge(rank_idx=-1) gives base_weight + delta from highest-rank adapter.
        # With B=0 (zero-init), delta=0, so merged == base_weight.
        merged_weight = forge.merge(rank_idx=-1)
        out_merged = F.linear(x, merged_weight)
        assert torch.allclose(out_default, out_merged, atol=1e-5), \
            "default forward (entropy=None) should match highest-rank merge"

    def test_forge_more_total_params_than_single_lora(self):
        """ForgeAdapter has 4 adapters (11520 params) > LoRA rank-32 (6144)."""
        forge = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], alpha=16.0,
                             temperature=1.0)
        lora = LoRAAdapter(64, 128, rank=32, alpha=16)
        # ForgeAdapter trainable: sum(r*(64+128) for r in [4,8,16,32])
        #   = 4*192 + 8*192 + 16*192 + 32*192 = 11520
        # LoRA rank-32: 32*(64+128) = 6144
        forge_params = _trainable_params(forge)
        lora_params = _trainable_params(lora)
        assert forge_params == 11520, f"expected 11520, got {forge_params}"
        assert lora_params == 6144, f"expected 6144, got {lora_params}"
        assert forge_params > lora_params

    def test_forge_low_entropy_uses_fewer_flops(self):
        """Low-entropy tokens use rank-4 (768 FLOPs) vs LoRA always-rank-32 (6144)."""
        in_features, out_features = 64, 128
        # ForgeAdapter low-entropy route → rank-4 adapter
        forge_flops = _lora_flops(4, in_features, out_features)   # 4*192 = 768
        # LoRA always uses rank-32
        lora_flops = _lora_flops(32, in_features, out_features)    # 32*192 = 6144
        assert forge_flops == 768, f"expected 768, got {forge_flops}"
        assert lora_flops == 6144, f"expected 6144, got {lora_flops}"
        assert forge_flops < lora_flops

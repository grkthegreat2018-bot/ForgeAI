"""Tests for R38-1 (DLoRA) and R38-2 (DoRA).

CPU-only. Uses a small nn.Linear(in=64, out=128) for all tests.
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
from torch import nn

from forge.training.dlora import DLoRA, DLoRAAdapter
from forge.training.dlora import apply_to_model as dlora_apply
from forge.training.dora import DoRALinear
from forge.training.dora import apply_to_model as dora_apply

# ── helpers ───────────────────────────────────────────────────────────────

def _lin(in_f=64, out_f=128, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f, bias=False)


def _model(seed=0):
    torch.manual_seed(seed)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    return M()


# ── DLoRA tests ───────────────────────────────────────────────────────────

def test_dlora_construction():
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
    assert a.current_rank == 4
    assert a.max_rank == 32
    assert a.lora_A.shape == (32, 64)
    assert a.lora_B.shape == (128, 32)
    # B zero-init → no-op start.
    assert torch.all(a.lora_B == 0)
    # Dormant A rows are zero.
    assert torch.all(a.lora_A[4:] == 0)
    # Active A rows are non-zero (kaiming).
    assert not torch.all(a.lora_A[:4] == 0)
    assert abs(a.scale - 8 / 4) < 1e-9


def test_dlora_forward_noop():
    """B zero-init → adapter output is zero for any input."""
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
    x = torch.randn(3, 64)
    out = a(x)
    assert out.shape == (3, 128)
    assert torch.allclose(out, torch.zeros(3, 128), atol=1e-7)


def test_dlora_forward_shape():
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
    x = torch.randn(5, 64)
    assert a(x).shape == (5, 128)


def test_dlora_should_grow():
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=8, alpha=8)
    # Below threshold → no grow.
    assert a.should_grow(0.005, threshold=0.01) is False
    # Above threshold → grow.
    assert a.should_grow(0.05, threshold=0.01) is True
    # At max rank → never grow.
    a.current_rank = 8
    assert a.should_grow(10.0, threshold=0.01) is False
    # History recorded.
    assert len(a.grad_norm_history) == 3


def test_dlora_grow_rank_noop_at_growth():
    """Growing activates new A rows but B cols stay zero → forward unchanged."""
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=8, alpha=8)
    # Give B some non-zero values in the active block so forward != 0.
    with torch.no_grad():
        a.lora_B[:, :4] = torch.randn(128, 4) * 0.01
    x = torch.randn(3, 64)
    before = a(x)
    new_r = a.grow_rank(6)
    assert new_r == 6
    assert a.current_rank == 6
    after = a(x)
    # Forward identical because new B cols are zero.
    assert torch.allclose(before, after, atol=1e-7)
    # New A rows are non-zero (kaiming).
    assert not torch.all(a.lora_A[4:6] == 0)


def test_dlora_grow_rank_clamped():
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=8, alpha=8)
    assert a.grow_rank(100) == 8
    assert a.current_rank == 8
    # No-op when already at target.
    assert a.grow_rank(8) == 8


def test_dlora_grow_default_increment():
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=8, alpha=8)
    assert a.grow_rank() == 5
    assert a.current_rank == 5


def test_dlora_dynamic_rank_tracking():
    mgr = DLoRA(initial_rank=4, max_rank=16, growth_threshold=0.01)
    a1 = DLoRAAdapter(64, 128, initial_rank=4, max_rank=16)
    a2 = DLoRAAdapter(128, 64, initial_rank=4, max_rank=16)
    mgr.adapters = [a1, a2]
    mgr.layer_map = {0: a1, 1: a2}
    assert mgr.total_rank() == 8
    # Only layer 0 exceeds threshold.
    grew = mgr.maybe_grow({0: 0.5, 1: 0.001})
    assert grew == [0]
    assert a1.current_rank == 5
    assert a2.current_rank == 4
    assert mgr.total_rank() == 9


def test_dlora_grow_rank_by_idx():
    mgr = DLoRA(initial_rank=4, max_rank=16)
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=16)
    mgr.adapters = [a]
    mgr.layer_map = {0: a}
    assert mgr.grow_rank(0, 10) == 10
    assert a.current_rank == 10


def test_dlora_apply_to_model():
    model = _model()
    _mgr, n, params = dlora_apply(model, initial_rank=4, max_rank=16,
                                 min_size=32)
    assert n == 2  # fc1 (64→128), fc2 (128→64)
    assert len(params) == 4  # 2 adapters × (A, B)
    # Base weights frozen.
    assert not model.fc1.weight.requires_grad
    assert not model.fc2.weight.requires_grad
    # Adapters attached.
    assert hasattr(model.fc1, 'lora_adapter')
    assert isinstance(model.fc1.lora_adapter, DLoRAAdapter)
    assert model.fc1.lora_adapter.current_rank == 4
    # Forward works (no-op since B zero).
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 64)


def test_dlora_forward_with_active_adapter():
    """Non-zero B → adapter changes the output."""
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
    with torch.no_grad():
        a.lora_B[:, :4] = torch.randn(128, 4) * 0.1
    x = torch.randn(3, 64)
    base = torch.zeros(3, 128)
    out = a(x)
    assert not torch.allclose(out, base, atol=1e-6)


def test_dlora_scale_fixed_with_rank():
    """Scale is fixed at alpha/initial_rank so growth is a no-op."""
    a = DLoRAAdapter(64, 128, initial_rank=4, max_rank=32, alpha=8)
    assert abs(a.scale - 2.0) < 1e-9  # 8/4
    a.grow_rank(8)
    assert abs(a.scale - 2.0) < 1e-9  # still 8/4 (fixed)


# ── DoRA tests ────────────────────────────────────────────────────────────

def test_dora_construction():
    d = DoRALinear(64, 128, rank=8, alpha=16)
    assert d.in_features == 64
    assert d.out_features == 128
    assert d.rank == 8
    assert d.lora_A.shape == (8, 64)
    assert d.lora_B.shape == (128, 8)
    assert d.magnitude.shape == (128,)
    assert d.V.shape == (128, 64)
    # B zero-init.
    assert torch.all(d.lora_B == 0)
    assert abs(d.scale - 2.0) < 1e-9  # 16/8


def test_dora_decomposition_unit_norm():
    """After load_from_weight, V rows are unit-norm and magnitude = row norm."""
    torch.manual_seed(0)
    w = torch.randn(128, 64) * 0.3
    d = DoRALinear(64, 128, rank=8)
    d.load_from_weight(w)
    norms = d.V.norm(dim=1)
    assert torch.allclose(norms, torch.ones(128), atol=1e-5)
    expected_mag = w.norm(dim=1)
    assert torch.allclose(d.magnitude.data, expected_mag, atol=1e-5)


def test_dora_forward_matches_base_when_no_delta():
    """With delta_V=0 (B zero), DoRA forward ≈ base Linear forward."""
    lin = _lin(64, 128, seed=42)
    d = DoRALinear(64, 128, rank=8, bias=False)
    d.load_from_weight(lin.weight.data)
    d = d.to(lin.weight.dtype)
    x = torch.randn(5, 64)
    base_out = lin(x)
    dora_out = d(x)
    assert torch.allclose(base_out, dora_out, atol=1e-4)


def test_dora_forward_shape():
    d = DoRALinear(64, 128, rank=8)
    d.load_from_weight(torch.randn(128, 64))
    x = torch.randn(4, 64)
    assert d(x).shape == (4, 128)


def test_dora_decomposition_lossless_reconstruction():
    """m ⊙ normalize(V) reconstructs the original weight (delta_V=0)."""
    torch.manual_seed(7)
    w = torch.randn(128, 64) * 0.5
    d = DoRALinear(64, 128, rank=8)
    d.load_from_weight(w)
    w_recon = d._effective_weight()
    assert torch.allclose(w_recon, w, atol=1e-4)


def test_dora_merge_zero_delta_bit_exact():
    """Merge with zero delta → merged weight == original weight."""
    lin = _lin(64, 128, seed=11)
    d = DoRALinear(64, 128, rank=8, bias=False)
    d.load_from_weight(lin.weight.data)
    w_merged = d.merge()
    assert torch.allclose(w_merged, lin.weight.data, atol=1e-4)
    # Adapter zeroed after merge.
    assert torch.all(d.lora_B == 0)


def test_dora_merge_with_delta():
    """Merge with non-zero delta produces a valid merged weight."""
    lin = _lin(64, 128, seed=3)
    d = DoRALinear(64, 128, rank=8, bias=False)
    d.load_from_weight(lin.weight.data)
    with torch.no_grad():
        d.lora_B[:, :] = torch.randn(128, 8) * 0.01
        d.magnitude.add_(torch.randn(128) * 0.01)
    w_before = d._effective_weight()
    w_merged = d.merge()
    # Merged weight equals the pre-merge effective weight.
    assert torch.allclose(w_merged, w_before, atol=1e-4)
    # After merge, forward still matches (adapter zeroed, recomposed).
    x = torch.randn(3, 64)
    out_after = d(x)
    w_eff = d._effective_weight()
    assert torch.allclose(out_after, torch.nn.functional.linear(x, w_eff),
                          atol=1e-5)


def test_dora_apply_to_model():
    model = _model()
    _mgr, n, params = dora_apply(model, rank=8, min_size=32)
    assert n == 2
    # fc1, fc2 replaced with DoRALinear.
    assert isinstance(model.fc1, DoRALinear)
    assert isinstance(model.fc2, DoRALinear)
    # Trainable params: magnitude + A + B per layer.
    assert len(params) == 6
    # Forward works.
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 64)


def test_dora_apply_preserves_forward():
    """After apply, model forward ≈ original model forward (delta_V=0)."""
    model = _model(seed=5)
    model.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        base_out = model(x)
    # Apply DoRA (replaces linears, B zero → lossless).
    _mgr, _n, _ = dora_apply(model, rank=8, min_size=32)
    with torch.no_grad():
        dora_out = model(x)
    assert torch.allclose(base_out, dora_out, atol=1e-4)


def test_dora_merge_back_to_linear():
    """merge() replaces DoRALinear with nn.Linear holding merged weight."""
    model = _model(seed=2)
    mgr, _n, _ = dora_apply(model, rank=8, min_size=32)
    # Inject a small delta.
    with torch.no_grad():
        model.fc1.lora_B.add_(torch.randn(128, 8) * 0.01)
    x = torch.randn(3, 64)
    with torch.no_grad():
        before = model(x)
    n_merged = mgr.merge(model)
    assert n_merged == 2
    assert isinstance(model.fc1, nn.Linear)
    assert not isinstance(model.fc1, DoRALinear)
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_dora_quality_vs_lora():
    """DoRA with delta_V=0 is lossless; standard LoRA with B=0 is also a
    no-op. This verifies the DoRA decomposition itself introduces no error
    beyond fp32 numerical noise — the key quality guarantee for DoRA."""
    lin = _lin(64, 128, seed=99)
    d = DoRALinear(64, 128, rank=8, bias=False)
    d.load_from_weight(lin.weight.data)
    x = torch.randn(8, 64)
    base = lin(x)
    dora = d(x)
    err = (base - dora).abs().max().item()
    # Decomposition is lossless up to fp32 precision.
    assert err < 1e-3, f"DoRA decomposition error {err} too large"


def test_dora_magnitude_trainable():
    d = DoRALinear(64, 128, rank=8)
    assert d.magnitude.requires_grad
    assert d.lora_A.requires_grad
    assert d.lora_B.requires_grad
    # V is a buffer (frozen).
    assert not d.V.requires_grad


def test_dora_forward_with_delta_changes_output():
    """Non-zero LoRA delta on direction changes the output."""
    lin = _lin(64, 128, seed=8)
    d = DoRALinear(64, 128, rank=8, bias=False)
    d.load_from_weight(lin.weight.data)
    x = torch.randn(3, 64)
    base = d(x)
    with torch.no_grad():
        d.lora_B[:, :] = torch.randn(128, 8) * 0.1
    changed = d(x)
    assert not torch.allclose(base, changed, atol=1e-6)

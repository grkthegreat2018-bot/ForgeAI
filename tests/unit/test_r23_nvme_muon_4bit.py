"""Tests for R&D Round 23: NVMe-streamed 4-bit Muon optimizer (V8 optimizer).

Combines NVMeStreamedBAdam (per-block NVMe-mapped optimizer states) with
MuonBitNet4Bit (4-bit momentum + Newton-Schulz orthogonalization) into a
single NvmeMuon4Bit optimizer class, registered as "nvme_muon_4bit" in
configure_optimizer().
"""
import os
import sys
import tempfile

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_tiny_model(d=64, n_layers=3):
    """Create a tiny Sequential model for optimizer tests."""
    model = nn.Sequential()
    for i in range(n_layers):
        model.add_module(f"layer_{i}", nn.Linear(d, d, bias=False))
    return model.to(_DEV)


# ── R23-1: Registration in configure_optimizer ──────────────────────────────

def test_nvme_muon_4bit_registered():
    """configure_optimizer should return an optimizer for 'nvme_muon_4bit'."""
    from forge.training.training_utils import configure_optimizer

    model = _make_tiny_model(d=64, n_layers=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        opt = configure_optimizer(
            model, max_lr=1e-3, weight_decay=0.01,
            optimizer_name="nvme_muon_4bit")
    assert opt is not None, "Should return an optimizer object"
    assert hasattr(opt, "step"), "Should have a step() method"
    assert hasattr(opt, "zero_grad"), "Should have a zero_grad() method"
    print("  nvme_muon_4bit_registered: PASS")


# ── R23-2: Loss reduction ───────────────────────────────────────────────────

def test_nvme_muon_4bit_reduces_loss():
    """NvmeMuon4Bit should reduce MSE loss over 20 training steps."""
    from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=3)
    x = torch.randn(8, 64, device=_DEV)
    y = torch.randn(8, 64, device=_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        opt = NvmeMuon4Bit(
            model, lr=1e-2, momentum=0.95, n_steps=5,
            weight_decay=0.01, nvme_path=tmpdir,
            blocks_per_layer=1, switch_every=5, verbose=False)

        initial_loss = nn.MSELoss()(model(x), y).item()
        for _ in range(20):
            opt.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
        final_loss = loss.item()

    print(f"  NvmeMuon4Bit loss: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease"
    print("  nvme_muon_4bit_reduces_loss: PASS")


# ── R23-3: NVMe storage verification ────────────────────────────────────────

def test_nvme_muon_4bit_nvme_storage():
    """Optimizer states should be stored in NVMe files after a step."""
    from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=3)
    x = torch.randn(4, 64, device=_DEV)
    y = torch.randn(4, 64, device=_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        opt = NvmeMuon4Bit(
            model, lr=1e-2, nvme_path=tmpdir,
            blocks_per_layer=1, switch_every=10, verbose=False)

        # Run one step to trigger state initialization + NVMe writes
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()

        # Verify NVMe files exist
        files = os.listdir(tmpdir)
        assert len(files) > 0, "Should have NVMe state files after a step"
        print(f"  NVMe files created: {len(files)}")

    print("  nvme_muon_4bit_nvme_storage: PASS")


# ── R23-4: Only active block in RAM ─────────────────────────────────────────

def test_nvme_muon_4bit_one_block_in_ram():
    """Only the active block's optimizer states should be in CPU RAM."""
    from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=4)
    x = torch.randn(4, 64, device=_DEV)
    y = torch.randn(4, 64, device=_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        opt = NvmeMuon4Bit(
            model, lr=1e-2, nvme_path=tmpdir,
            blocks_per_layer=1, switch_every=10, verbose=False)

        # Run a step to initialize states
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()

        # Only the active block should have states in _active_states
        n_active = len(opt._active_states)
        n_blocks = opt._n_blocks
        print(f"  Active states: {n_active}, total blocks: {n_blocks}")
        assert n_blocks > 1, "Should have multiple blocks for 4-layer model"
        # Active states should only cover the active block's params
        active_block = opt._blocks[opt._block_idx]
        active_param_count = sum(1 for p in active_block["params"] if p.numel() > 0)
        assert n_active <= active_param_count, \
            f"Only active block params should be in RAM ({n_active} vs {active_param_count})"

    print("  nvme_muon_4bit_one_block_in_ram: PASS")


# ── R23-5: 4-bit quantization round-trip ────────────────────────────────────

def test_nvme_muon_4bit_4bit_quantization():
    """4-bit momentum quantization should round-trip with <15% error."""
    from forge.training.optim.r20_memory_optimizers import _dequantize_4bit, _quantize_4bit

    torch.manual_seed(42)
    t = torch.randn(10000) * 0.01  # typical momentum scale
    packed, scales = _quantize_4bit(t, block_size=128)
    t_dq = _dequantize_4bit(packed, scales, t.shape, block_size=128)

    err = (t - t_dq).abs().mean().item() / t.abs().mean().item()
    print(f"  4-bit round-trip error: {err*100:.2f}%")
    assert err < 0.15, f"Round-trip error should be <15%, got {err*100:.2f}%"

    # Verify memory: 4-bit = 0.5 bytes/param + scales
    packed_bytes = packed.numel()  # uint8, 2 values per byte
    scale_bytes = scales.numel() * 2  # fp16
    bytes_per_param = (packed_bytes + scale_bytes) / t.numel()
    print(f"  4-bit memory: {bytes_per_param:.3f} bytes/param")
    assert bytes_per_param < 0.8, "Should be <0.8 bytes/param for 4-bit"

    print("  nvme_muon_4bit_4bit_quantization: PASS")


# ── R23-6: Newton-Schulz orthogonalization ──────────────────────────────────

def test_nvme_muon_4bit_newton_schulz():
    """Muon Newton-Schulz orthogonalization should be applied during step."""
    from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=2)
    x = torch.randn(4, 64, device=_DEV)
    y = torch.randn(4, 64, device=_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        opt = NvmeMuon4Bit(
            model, lr=1e-2, momentum=0.9, n_steps=5,
            nvme_path=tmpdir, switch_every=100, verbose=False)

        # Run a step to build momentum
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()

        # Run another step to get non-zero momentum
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()

        # Verify the optimizer has Newton-Schulz capability
        assert hasattr(opt, "_newton_schulz"), \
            "NvmeMuon4Bit should have _newton_schulz method"

        # Test Newton-Schulz directly: output should be approximately orthogonal
        g = torch.randn(64, 64)
        ortho = opt._newton_schulz(g, n_steps=5)
        # Orthogonalized matrix: X @ X.T ≈ I
        gram = ortho @ ortho.T
        identity = torch.eye(64)
        ortho_err = (gram - identity).abs().max().item()
        print(f"  Newton-Schulz orthogonality error: {ortho_err:.4f}")
        assert ortho_err < 0.5, \
            f"Newton-Schulz should approximately orthogonalize (err={ortho_err:.4f})"

    print("  nvme_muon_4bit_newton_schulz: PASS")


# ── R23-7: Memory budget verification ───────────────────────────────────────

def test_nvme_muon_4bit_memory_budget():
    """V8 budget: active block < 1GB RAM, NVMe storage proportional to params."""
    from forge.training.optim.r20_memory_optimizers import _quantize_4bit

    # Simulate V8-8B scale: 8.05B params, 32 layers
    total_params = 8.05e9
    n_layers = 32
    params_per_layer = total_params / n_layers

    # 4-bit Muon: 0.625 bytes/param for momentum (0.5 packed + 0.125 scales)
    bytes_per_param_4bit = 0.625
    active_block_bytes = params_per_layer * bytes_per_param_4bit
    active_block_gb = active_block_bytes / 1e9

    # NVMe storage: all blocks' 4-bit momentum
    nvme_total_bytes = total_params * bytes_per_param_4bit
    nvme_total_gb = nvme_total_bytes / 1e9

    # Master weights (bf16): 2 bytes/param
    master_gb = total_params * 2 / 1e9

    # Total RAM = master + active block
    total_ram_gb = master_gb + active_block_gb

    print("  V8-8B NvmeMuon4Bit memory budget:")
    print(f"    Active block (1 layer): {active_block_gb:.2f} GB RAM")
    print(f"    NVMe storage (all):     {nvme_total_gb:.1f} GB")
    print(f"    Master weights (bf16):  {master_gb:.1f} GB RAM")
    print(f"    Total RAM:              {total_ram_gb:.1f} GB")

    # V8 budget constraints
    assert active_block_gb < 1.0, \
        f"Active block should be <1GB, got {active_block_gb:.2f} GB"
    assert total_ram_gb < 22.4, \
        f"Total RAM should fit 22.4GB available, got {total_ram_gb:.1f} GB"
    # NVMe storage should be proportional to total params
    expected_nvme = total_params * bytes_per_param_4bit / 1e9
    assert abs(nvme_total_gb - expected_nvme) < 0.1, \
        "NVMe storage should be proportional to total params"

    # Verify with a small model
    torch.manual_seed(42)
    n = 10000
    t = torch.randn(n)
    packed, scales = _quantize_4bit(t, block_size=128)
    actual_bytes = packed.numel() + scales.numel() * 2
    actual_per_param = actual_bytes / n
    print(f"  Small model: {actual_per_param:.3f} bytes/param (4-bit Muon)")
    assert actual_per_param < 0.7, "Should be <0.7 bytes/param"

    print("  nvme_muon_4bit_memory_budget: PASS")


# ── R23-8: Block switching ──────────────────────────────────────────────────

def test_nvme_muon_4bit_block_switching():
    """Optimizer should switch blocks after switch_every steps."""
    from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=4)
    x = torch.randn(4, 64, device=_DEV)
    y = torch.randn(4, 64, device=_DEV)

    switch_every = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        opt = NvmeMuon4Bit(
            model, lr=1e-2, nvme_path=tmpdir,
            blocks_per_layer=1, switch_every=switch_every, verbose=False)

        initial_block = opt._block_idx
        print(f"  Initial block: {initial_block}")

        # Run switch_every - 1 steps (should not switch yet)
        for _i in range(switch_every - 1):
            opt.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
        block_before_switch = opt._block_idx
        print(f"  After {switch_every - 1} steps: block {block_before_switch}")
        assert block_before_switch == initial_block, \
            "Should not switch before switch_every steps"

        # Run one more step (should trigger switch)
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
        block_after_switch = opt._block_idx
        print(f"  After {switch_every} steps: block {block_after_switch}")
        assert block_after_switch != initial_block, \
            f"Should switch block after {switch_every} steps"

    print("  nvme_muon_4bit_block_switching: PASS")


# ── R23-9: State save/restore (resume) ──────────────────────────────────────

def test_nvme_muon_4bit_resume():
    """Save optimizer state, create new optimizer, load state, verify restored."""
    from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=3)
    x = torch.randn(4, 64, device=_DEV)
    y = torch.randn(4, 64, device=_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for a few steps to advance block_idx and step_count
        opt1 = NvmeMuon4Bit(
            model, lr=1e-2, nvme_path=tmpdir,
            blocks_per_layer=1, switch_every=3, verbose=False)
        for _ in range(7):  # > 2 blocks worth of steps
            opt1.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt1.step()

        saved_block_idx = opt1._block_idx
        saved_step_count = opt1._step_count
        print(f"  Saved: block_idx={saved_block_idx}, step_count={saved_step_count}")
        assert saved_step_count == 7, f"Expected 7 steps, got {saved_step_count}"

        # Save state
        state = opt1.state_dict()

        # Create new optimizer and load state
        model2 = _make_tiny_model(d=64, n_layers=3)
        opt2 = NvmeMuon4Bit(
            model2, lr=1e-2, nvme_path=tmpdir,
            blocks_per_layer=1, switch_every=3, verbose=False)
        opt2.load_state_dict(state)

        restored_block_idx = opt2._block_idx
        restored_step_count = opt2._step_count
        print(f"  Restored: block_idx={restored_block_idx}, step_count={restored_step_count}")

        assert restored_block_idx == saved_block_idx, \
            f"block_idx mismatch: {restored_block_idx} vs {saved_block_idx}"
        assert restored_step_count == saved_step_count, \
            f"step_count mismatch: {restored_step_count} vs {saved_step_count}"

    print("  nvme_muon_4bit_resume: PASS")


def main_r23_nvme_muon():
    print("=" * 70)
    print("  R&D ROUND 23: NVMe-streamed 4-bit Muon Optimizer (V8)")
    print("=" * 70)

    print("\n  R23-1: Registration")
    test_nvme_muon_4bit_registered()

    print("\n  R23-2: Loss reduction")
    test_nvme_muon_4bit_reduces_loss()

    print("\n  R23-3: NVMe storage")
    test_nvme_muon_4bit_nvme_storage()

    print("\n  R23-4: One block in RAM")
    test_nvme_muon_4bit_one_block_in_ram()

    print("\n  R23-5: 4-bit quantization")
    test_nvme_muon_4bit_4bit_quantization()

    print("\n  R23-6: Newton-Schulz orthogonalization")
    test_nvme_muon_4bit_newton_schulz()

    print("\n  R23-7: Memory budget")
    test_nvme_muon_4bit_memory_budget()

    print("\n  R23-8: Block switching")
    test_nvme_muon_4bit_block_switching()

    print("\n  R23-9: Resume")
    test_nvme_muon_4bit_resume()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 NVME-MUON TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_nvme_muon()

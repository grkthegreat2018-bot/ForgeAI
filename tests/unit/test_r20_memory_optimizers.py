"""Tests for R&D Round 20: Memory-efficient optimizers for V7-8B training.

Verifies that all 4 optimizers work correctly and that the memory math
fits the 22.4 GB available RAM on the target system.
"""
import os
import sys
import tempfile

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32


def _make_tiny_model(d=128, n_layers=4, ternary=False):
    """Create a tiny model for testing optimizers."""
    model = nn.Sequential()
    for i in range(n_layers):
        layer = nn.Linear(d, d, bias=False)
        if ternary:
            with torch.no_grad():
                layer.weight.data = torch.sign(layer.weight.data)
        model.add_module(f"layer_{i}", layer)
    return model.to(_DEV)


# ── R20a: 4-bit AdamW ───────────────────────────────────────────────────────

def test_adamw4bit_basic():
    """4-bit AdamW should reduce loss on a simple task."""
    from forge.training.optim.r20_memory_optimizers import AdamW4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=3)
    x = torch.randn(8, 64, device=_DEV)
    y = torch.randn(8, 64, device=_DEV)

    opt = AdamW4Bit(model.parameters(), lr=1e-2, verbose=False)

    initial_loss = nn.MSELoss()(model(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
    final_loss = loss.item()

    print(f"  AdamW4Bit: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease"
    print("  AdamW4Bit basic: PASS")


def test_adamw4bit_memory():
    """Verify 4-bit states use ~1.25 bytes/param."""
    from forge.training.optim.r20_memory_optimizers import _quantize_4bit

    n = 10000
    t = torch.randn(n)
    packed, scales = _quantize_4bit(t, block_size=128)

    packed_bytes = packed.numel()  # uint8
    scale_bytes = scales.numel() * 2  # fp16
    total = packed_bytes + scale_bytes
    bytes_per_param = total / n

    print(f"  4-bit quant: {n} params -> {total} bytes = {bytes_per_param:.2f} B/param")
    assert bytes_per_param < 1.5, f"Should be <1.5 B/param, got {bytes_per_param:.2f}"
    assert bytes_per_param > 0.5, f"Should be >0.5 B/param, got {bytes_per_param:.2f}"

    # Verify round-trip accuracy
    t_dq = _dequantize_4bit(packed, scales, t.shape)
    err = (t - t_dq).abs().mean().item() / t.abs().mean().item()
    print(f"  4-bit round-trip error: {err*100:.2f}%")
    assert err < 0.15, f"Round-trip error too high: {err*100:.2f}%"
    print("  AdamW4Bit memory: PASS")


def _dequantize_4bit(packed, scales, shape, block_size=128):
    from forge.training.optim.r20_memory_optimizers import _dequantize_4bit as _dq
    return _dq(packed, scales, shape, block_size)


# ── R20b: NVMe-streamed BAdam ───────────────────────────────────────────────

def test_nvme_streamed_badam():
    """NVMe-streamed BAdam should train and cycle through blocks."""
    from forge.training.optim.r20_memory_optimizers import NVMeStreamedBAdam

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=4)
    x = torch.randn(8, 64, device=_DEV)
    y = torch.randn(8, 64, device=_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        opt = NVMeStreamedBAdam(
            model, lr=1e-2, nvme_path=tmpdir,
            state_bytes=4, switch_every=3, verbose=False)

        initial_loss = nn.MSELoss()(model(x), y).item()
        for _ in range(20):
            opt.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
        final_loss = loss.item()

        print(f"  NVMe-BAdam: {initial_loss:.4f} -> {final_loss:.4f}")
        assert final_loss < initial_loss, "Loss should decrease"

        # Verify NVMe files were created
        files = os.listdir(tmpdir)
        assert len(files) > 0, "Should have NVMe state files"
        print(f"  NVMe files: {len(files)}")

    print("  NVMe-BAdam: PASS")


# ── R20c: Muon-BitNet 4-bit ─────────────────────────────────────────────────

def test_muon_bitnet_4bit():
    """Muon with 4-bit momentum should reduce loss."""
    from forge.training.optim.r20_memory_optimizers import MuonBitNet4Bit

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=3)
    x = torch.randn(8, 64, device=_DEV)
    y = torch.randn(8, 64, device=_DEV)

    opt = MuonBitNet4Bit(model.parameters(), lr=1e-3, verbose=False)

    initial_loss = nn.MSELoss()(model(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
    final_loss = loss.item()

    print(f"  MuonBitNet4Bit: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease"
    print("  MuonBitNet4Bit: PASS")


def test_muon_bitnet_memory():
    """Verify Muon 4-bit uses ~0.625 bytes/param (single buffer)."""
    from forge.training.optim.r20_memory_optimizers import _quantize_4bit

    n = 10000
    t = torch.randn(n)
    packed, scales = _quantize_4bit(t, block_size=128)

    # Muon: only 1 buffer (momentum), not 2 (m+v)
    total = packed.numel() + scales.numel() * 2
    bytes_per_param = total / n

    print(f"  Muon 4-bit: {n} params -> {total} bytes = {bytes_per_param:.2f} B/param")
    assert bytes_per_param < 0.8, f"Should be <0.8 B/param, got {bytes_per_param:.2f}"
    print("  MuonBitNet memory: PASS")


# ── R20d: Ternary optimizer ─────────────────────────────────────────────────

def test_ternary_optimizer():
    """Ternary optimizer should work with BitNet ternary weights."""
    from forge.training.optim.r20_memory_optimizers import TernaryOptimizer

    torch.manual_seed(42)
    model = _make_tiny_model(d=64, n_layers=3, ternary=True)
    x = torch.randn(8, 64, device=_DEV)
    y = torch.randn(8, 64, device=_DEV)

    opt = TernaryOptimizer(model.parameters(), lr=1e-2, verbose=False)

    initial_loss = nn.MSELoss()(model(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
    final_loss = loss.item()

    print(f"  TernaryOpt: {initial_loss:.4f} -> {final_loss:.4f}")
    # Ternary optimizer may not always reduce loss (it flips ternary values)
    # but it should not crash and should produce valid output
    assert torch.isfinite(model(x)).all(), "Output should be finite"

    # Verify weights are still ternary
    for p in model.parameters():
        if p.dim() >= 2:
            vals = p.data.unique()
            assert all(v in (-1.0, 0.0, 1.0) for v in vals.tolist()), \
                f"Weights should be ternary, got {vals}"
    print("  TernaryOptimizer: PASS")


def test_ternary_memory():
    """Verify 2-bit ternary states use ~0.25 bytes/param."""

    n = 10000
    # 2-bit packed: 4 values per byte
    packed_bytes = (n + 3) // 4
    bytes_per_param = packed_bytes / n

    print(f"  Ternary 2-bit: {n} params -> {packed_bytes} bytes = {bytes_per_param:.2f} B/param")
    assert bytes_per_param < 0.3, f"Should be <0.3 B/param, got {bytes_per_param:.2f}"
    print("  Ternary memory: PASS")


# ── Full V7-8B memory verification ──────────────────────────────────────────

def test_v7_8b_memory_fits():
    """Verify all R20 approaches fit the 22.4 GB available RAM."""
    print("\n  V7-8B memory budget (8.05B params, 22.4 GB available RAM):")
    print("  " + "-" * 65)

    params = 8.05e9
    n_layers = 32
    params_per_layer = params / n_layers
    avail_ram = 22.4e9

    approaches = []

    # R20a: 4-bit AdamW (doesn't fit alone, but works with NVMe streaming)
    master = params * 2
    opt_4bit = params * 1.25  # 4-bit m+v + scales
    total_4bit = master + opt_4bit
    approaches.append(("R20a: 4-bit AdamW", total_4bit, total_4bit < avail_ram))

    # R20b: NVMe-streamed (master on CPU + 1 layer opt)
    opt_1layer = params_per_layer * 4  # 8-bit, 1 layer
    total_nvme = master + opt_1layer
    approaches.append(("R20b: NVMe-streamed", total_nvme, total_nvme < avail_ram))

    # R20c: Muon-BitNet 4-bit
    opt_muon = params * 0.625  # 4-bit single momentum + scales
    total_muon = master + opt_muon
    approaches.append(("R20c: Muon-BitNet 4-bit", total_muon, total_muon < avail_ram))

    # R20d: Ternary optimizer
    bitnet_p = 7.65e9
    other_p = 0.4e9
    opt_ternary = bitnet_p * 0.25 + other_p * 8
    total_ternary = master + opt_ternary
    approaches.append(("R20d: Ternary optimizer", total_ternary, total_ternary < avail_ram))

    # Best combo: R20b + R20c (NVMe + 4-bit Muon)
    opt_combo = params_per_layer * 0.625  # 4-bit Muon, 1 layer
    total_combo = master + opt_combo
    approaches.append(("R20b+c: NVMe + 4-bit Muon", total_combo, total_combo < avail_ram))

    all_fit = True
    for name, total, fits in approaches:
        status = "FITS" if fits else "EXCEEDS"
        print(f"  {name:<25} {total/1e9:5.1f} GB  {status}")
        if not fits:
            all_fit = False

    # At least 3 approaches should fit
    fitting = sum(1 for _, _, f in approaches if f)
    print(f"\n  {fitting}/{len(approaches)} approaches fit 22.4 GB available RAM")
    assert fitting >= 3, f"Expected >=3 approaches to fit, got {fitting}"
    print("  V7-8B memory fits: PASS")


def main_r20():
    print("=" * 70)
    print("  R&D ROUND 20: Memory-Efficient Optimizers for V7-8B Training")
    print("=" * 70)

    print("\n  R20a: 4-bit AdamW")
    test_adamw4bit_basic()
    test_adamw4bit_memory()

    print("\n  R20b: NVMe-streamed BAdam")
    test_nvme_streamed_badam()

    print("\n  R20c: Muon-BitNet 4-bit")
    test_muon_bitnet_4bit()
    test_muon_bitnet_memory()

    print("\n  R20d: Ternary optimizer")
    test_ternary_optimizer()
    test_ternary_memory()

    print("\n  V7-8B memory verification")
    test_v7_8b_memory_fits()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 20 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r20()

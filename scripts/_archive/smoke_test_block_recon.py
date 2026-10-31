"""Smoke tests for block-level reconstruction."""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.engine.quant.block_recon import (
    BlockReconstructor,
    _find_block_list,
    _find_embed,
    _get_optimizable_params,
)


class SimpleBlock(nn.Module):
    """A simple transformer-like block for testing."""
    def __init__(self, d=64, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d * 2)
        self.linear2 = nn.Linear(d * 2, d)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.linear2(self.act(self.linear1(h)))
        return x + h


class SimpleModel(nn.Module):
    """A simple model with a block list for testing."""
    def __init__(self, vocab=100, d=64, n_blocks=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([SimpleBlock(d) for _ in range(n_blocks)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.model.layers:
            x = block(x)
        return self.lm_head(x)


def test_find_block_list():
    """Test that block list detection works."""
    print("\n[find_block_list]")
    model = SimpleModel()
    _parent, attr, blocks = _find_block_list(model)
    assert attr == "layers"
    assert len(blocks) == 4
    print(f"  Found {len(blocks)} blocks via {attr}")
    print("  PASS")


def test_find_embed():
    """Test that embedding detection works."""
    print("\n[find_embed]")
    model = SimpleModel()
    embed = _find_embed(model)
    assert embed is not None
    assert isinstance(embed, nn.Embedding)
    print(f"  Found embedding: {embed}")
    print("  PASS")


def test_capture_io():
    """Test block I/O capture."""
    print("\n[capture_block_io]")
    model = SimpleModel(vocab=100, d=64, n_blocks=4)
    model.eval()

    calib = torch.randint(0, 100, (2, 16))

    # We need both orig and quant models — use same model for both
    recon = BlockReconstructor(
        model_orig=model,
        model_quant=model,
        calibration_data=calib,
        device='cpu',
    )
    recon.capture_block_io()

    assert len(recon._block_inputs) == 4
    assert len(recon._block_outputs) == 4
    assert recon._block_inputs[0].shape == (2, 16, 64)
    print(f"  Captured {len(recon._block_inputs)} blocks")
    print(f"  Input shape: {recon._block_inputs[0].shape}")
    print("  PASS")


def test_get_optimizable_params():
    """Test that optimizable parameter extraction works."""
    print("\n[get_optimizable_params]")
    from forge.engine.quant.novel_quant_r48 import NanoQuantLinear

    block = SimpleBlock(d=64)
    # Replace linear1 with NanoQuant
    block.linear1 = NanoQuantLinear.from_linear(block.linear1, rank=8, admm_iters=5)

    params = _get_optimizable_params(block)
    assert len(params) > 0
    assert any("U_soft" in k for k in params)
    assert any("V_soft" in k for k in params)
    assert any("s1" in k for k in params)
    print(f"  Found {len(params)} optimizable params: {list(params.keys())}")
    print("  PASS")


def test_reconstruct_simple():
    """Test that reconstruction runs on a simple model."""
    print("\n[reconstruct_simple]")
    torch.manual_seed(42)

    # Create two identical models
    model_orig = SimpleModel(vocab=100, d=64, n_blocks=2)
    model_quant = SimpleModel(vocab=100, d=64, n_blocks=2)
    model_quant.load_state_dict(model_orig.state_dict())

    # Add some noise to the quant model to simulate quantization error
    with torch.no_grad():
        for p in model_quant.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    calib = torch.randint(0, 100, (2, 16))

    recon = BlockReconstructor(
        model_orig=model_orig,
        model_quant=model_quant,
        calibration_data=calib,
        device='cpu',
    )

    results = recon.reconstruct(n_iters=10, lr=0.001, verbose=True)
    assert len(results) == 2
    for idx, loss in results.items():
        assert loss < float('inf'), f"Block {idx} failed"
    print(f"  Reconstructed {len(results)} blocks")
    print("  PASS")


def test_reconstruct_with_nanoquant():
    """Test reconstruction with actual NanoQuant layers."""
    print("\n[reconstruct_nanoquant]")
    from forge.engine.quant.novel_quant_r48 import (
        quantize_model_nanoquant,
    )

    torch.manual_seed(42)
    model_orig = SimpleModel(vocab=100, d=64, n_blocks=2)
    model_quant = SimpleModel(vocab=100, d=64, n_blocks=2)
    model_quant.load_state_dict(model_orig.state_dict())

    # Quantize the quant model
    n = quantize_model_nanoquant(model_quant, rank=8, admm_iters=10, verbose=False)
    assert n > 0

    calib = torch.randint(0, 100, (2, 16))

    # Measure error before reconstruction
    with torch.no_grad():
        out_orig = model_orig(calib)
        out_quant_before = model_quant(calib)
        err_before = (out_orig - out_quant_before).abs().mean().item()

    # Reconstruct
    recon = BlockReconstructor(
        model_orig=model_orig,
        model_quant=model_quant,
        calibration_data=calib,
        device='cpu',
    )
    results = recon.reconstruct(n_iters=15, lr=0.005, verbose=True)

    # Measure error after reconstruction
    with torch.no_grad():
        out_quant_after = model_quant(calib)
        err_after = (out_orig - out_quant_after).abs().mean().item()

    print(f"  Error before: {err_before:.6f}")
    print(f"  Error after:  {err_after:.6f}")
    print(f"  Improvement:  {err_before - err_after:.6f}")
    print("  PASS")


def main():
    print("Block Reconstruction Smoke Tests\n")
    print("=" * 60)

    tests = [
        test_find_block_list,
        test_find_embed,
        test_capture_io,
        test_get_optimizable_params,
        test_reconstruct_simple,
        test_reconstruct_with_nanoquant,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Smoke tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

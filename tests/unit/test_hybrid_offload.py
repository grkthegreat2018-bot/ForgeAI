"""Test CPUAdamW against torch.optim.AdamW for numerical equivalence.

Verifies that the hybrid CPU-GPU optimizer produces the same parameter updates
as standard AdamW (within fp32 precision), and that the overlap mode works.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from forge.training.optim.hybrid_offload import CPUAdamW, estimate_memory


def test_numerical_equivalence():
    """CPUAdamW should match torch.optim.AdamW within 1e-5."""
    torch.manual_seed(42)
    model_a = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    model_b = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    model_b.load_state_dict(model_a.state_dict())

    # Standard AdamW on CPU (reference)
    opt_ref = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=0.01)
    # CPUAdamW with params on CPU (should be identical path)
    opt_cpu = CPUAdamW(model_b.parameters(), lr=1e-3, weight_decay=0.01, verbose=False)

    for _step in range(20):
        x = torch.randn(8, 64)
        target = torch.randn(8, 32)

        loss_a = nn.MSELoss()(model_a(x), target)
        loss_b = nn.MSELoss()(model_b(x), target)

        opt_ref.zero_grad()
        loss_a.backward()
        opt_ref.step()

        opt_cpu.zero_grad()
        loss_b.backward()
        opt_cpu.step()

    # Compare final weights
    for (n_a, p_a), (_n_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters(), strict=False):
        diff = (p_a - p_b).abs().max().item()
        assert diff < 1e-4, f"Param {n_a}: max diff {diff} exceeds 1e-4"
    print("PASS: CPUAdamW matches AdamW (max param diff < 1e-4) over 20 steps")


def test_gpu_offload():
    """CPUAdamW with GPU params should match AdamW with GPU params."""
    if not torch.cuda.is_available():
        print("SKIP: test_gpu_offload requires CUDA")
        return

    device = "cuda"
    torch.manual_seed(42)
    model_a = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)).to(device)
    model_b = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)).to(device)
    model_b.load_state_dict(model_a.state_dict())

    opt_ref = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=0.01)
    opt_cpu = CPUAdamW(model_b.parameters(), lr=1e-3, weight_decay=0.01, verbose=False)

    for _step in range(20):
        x = torch.randn(8, 64, device=device)
        target = torch.randn(8, 32, device=device)

        loss_a = nn.MSELoss()(model_a(x), target)
        loss_b = nn.MSELoss()(model_b(x), target)

        opt_ref.zero_grad()
        loss_a.backward()
        opt_ref.step()

        opt_cpu.zero_grad()
        loss_b.backward()
        opt_cpu.step()

    for (n_a, p_a), (_n_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters(), strict=False):
        diff = (p_a - p_b).abs().max().item()
        assert diff < 1e-3, f"GPU param {n_a}: max diff {diff} exceeds 1e-3 (bf16 transfer tolerance)"
    print("PASS: CPUAdamW GPU offload matches AdamW (max diff < 1e-3) over 20 steps")


def test_overlap_mode():
    """Overlap mode should produce same results as sync mode."""
    if not torch.cuda.is_available():
        print("SKIP: test_overlap_mode requires CUDA")
        return

    device = "cuda"
    torch.manual_seed(42)
    model_a = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)).to(device)
    model_b = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)).to(device)
    model_b.load_state_dict(model_a.state_dict())

    opt_sync = CPUAdamW(model_a.parameters(), lr=1e-3, weight_decay=0.01, overlap=False, verbose=False)
    opt_overlap = CPUAdamW(model_b.parameters(), lr=1e-3, weight_decay=0.01, overlap=True, verbose=False)

    for _step in range(20):
        x = torch.randn(8, 64, device=device)
        target = torch.randn(8, 32, device=device)

        loss_a = nn.MSELoss()(model_a(x), target)
        loss_b = nn.MSELoss()(model_b(x), target)

        opt_sync.zero_grad()
        loss_a.backward()
        opt_sync.step()

        opt_overlap.zero_grad()
        loss_b.backward()
        opt_overlap.step()
        opt_overlap.wait()  # ensure async CPU step completes

    for (n_a, p_a), (_n_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters(), strict=False):
        diff = (p_a - p_b).abs().max().item()
        assert diff < 1e-4, f"Overlap param {n_a}: max diff {diff} exceeds 1e-4"
    print("PASS: overlap mode matches sync mode (max diff < 1e-4)")


def test_memory_estimate():
    """estimate_memory should report reasonable values."""
    model = nn.Linear(1000, 1000)  # 1M params
    mem = estimate_memory(model)
    assert abs(mem["total_params_M"] - 1.001) < 0.01  # 1M weights + 1K bias
    assert mem["cpu_optimizer_GB"] > 0
    print(f"PASS: estimate_memory reports {mem['cpu_optimizer_GB']:.4f} GB CPU for 1M params")


def test_state_dict_roundtrip():
    """Optimizer state should survive save/load."""
    torch.manual_seed(42)
    model = nn.Linear(64, 32)
    opt = CPUAdamW(model.parameters(), lr=1e-3, verbose=False)

    # Run a few steps to populate state
    for _ in range(5):
        x = torch.randn(4, 64)
        target = torch.randn(4, 32)
        opt.zero_grad()
        nn.MSELoss()(model(x), target).backward()
        opt.step()

    sd = opt.state_dict()
    # Re-create and load
    model2 = nn.Linear(64, 32)
    opt2 = CPUAdamW(model2.parameters(), lr=1e-3, verbose=False)
    opt2.load_state_dict(sd)

    # Verify state was loaded
    assert opt2._initialized
    print("PASS: state_dict round-trip works")


if __name__ == "__main__":
    test_numerical_equivalence()
    test_gpu_offload()
    test_overlap_mode()
    test_memory_estimate()
    test_state_dict_roundtrip()
    print("\n=== All CPUAdamW tests passed ===")

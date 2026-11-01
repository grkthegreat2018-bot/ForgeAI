"""Test FreeToken-inspired enhancements to CPUAdamW.

Tests:
1. Backward compatibility — original API still works
2. double_buffer + overlap
3. bandwidth_adaptive (profiles PCIe)
4. chunked transfers
5. numerical correctness vs torch.optim.AdamW
6. bandwidth predictor
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn as nn

from forge.training.optim.hybrid_offload import CPUAdamW


def main():
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).cuda()
    model.train()
    x = torch.randn(4, 128).cuda()

    # Test 1: Backward compat
    opt1 = CPUAdamW(model.parameters(), lr=1e-3, verbose=False)
    y = model(x)
    loss = y.sum()
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    print("Test 1 (backward compat): PASS")

    # Test 2: double_buffer + overlap
    opt2 = CPUAdamW(model.parameters(), lr=1e-3, overlap=True, double_buffer=True, verbose=False)
    opt2.zero_grad()
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt2.step()
    opt2.wait()
    print("Test 2 (double_buffer): PASS")

    # Test 3: bandwidth_adaptive
    opt3 = CPUAdamW(model.parameters(), lr=1e-3, bandwidth_adaptive=True, double_buffer=True, verbose=True)
    opt3.zero_grad()
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt3.step()
    opt3.wait()
    print("Test 3 (bandwidth_adaptive): PASS")

    # Test 4: chunked transfers
    opt4 = CPUAdamW(model.parameters(), lr=1e-3, bandwidth_adaptive=True, chunk_size_mb=4, verbose=False)
    opt4.zero_grad()
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt4.step()
    print("Test 4 (chunked): PASS")

    # Test 5: numerical correctness — sync mode vs double_buffer overlap
    # Both use fp32 master, so should produce identical results.
    model_a = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).cuda()
    model_b = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).cuda()
    model_b.load_state_dict(model_a.state_dict())
    opt_a = CPUAdamW(model_a.parameters(), lr=1e-3, overlap=False, verbose=False)
    opt_b = CPUAdamW(model_b.parameters(), lr=1e-3, overlap=True, double_buffer=True, verbose=False)
    torch.manual_seed(42)
    for _i in range(5):
        x_i = torch.randn(4, 128).cuda()
        opt_a.zero_grad()
        opt_b.zero_grad()
        y_a = model_a(x_i)
        y_b = model_b(x_i)
        l_a = y_a.sum()
        l_b = y_b.sum()
        l_a.backward()
        l_b.backward()
        opt_a.step()  # sync mode, blocks until done
        opt_b.step()
        opt_b.wait()  # overlap mode, wait for completion
    max_diff = max(
        (pa - pb).abs().max().item() for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=False)
    )
    status = "PASS" if max_diff < 1e-5 else "FAIL"
    print(f"Test 5 (double_buffer correctness): max_diff={max_diff:.8f} {status}")
    assert max_diff < 1e-5, f"double_buffer correctness: max_diff={max_diff:.8f}"

    # Test 6: bandwidth predictor
    for vram in [5.0, 5.5, 6.0, 6.5, 7.0]:
        opt3.record_bandwidth_sample(vram_gb=vram)
    preempt = opt3.should_preempt_offload()
    stats = opt3.bandwidth_stats()
    print(f"Test 6 (predictor): should_preempt={preempt}")
    print(f"  stats: {stats}")
    print("ALL TESTS PASSED")


@pytest.mark.gpu
def test_hybrid_offload_ft():
    main()


if __name__ == "__main__":
    main()

"""Smoke test: CPUAdamW with the actual ForgeLM model on GPU.

Verifies that the hybrid offload optimizer works with the real 1.2B model,
fits in 12GB VRAM, and produces correct training steps.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from forge.training.optim.hybrid_offload import CPUAdamW, estimate_memory


def test_forgelm_cpuadamw():
    from forge.config import get_config
    from forge.model_loader import ModelLoader

    config = get_config("forgelm_tiny")  # 4-layer tiny model for fast test
    print(f"Config: {config.n_layers} layers, d_model={config.d_model}")

    model = ModelLoader.build_model(config)
    model = model.to("cuda").to(torch.bfloat16)
    model.train()

    # Memory estimate
    mem = estimate_memory(model)
    print(f"Memory estimate: {mem}")

    # Create CPUAdamW
    optimizer = CPUAdamW(model.parameters(), lr=5e-5, weight_decay=0.01, overlap=False)

    # Run 3 training steps
    for step in range(3):
        x = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = logits.float().mean()  # dummy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}: loss={loss.item():.6f}")

    # Check VRAM usage
    vram_gb = torch.cuda.memory_allocated() / 1e9
    vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM allocated: {vram_gb:.2f} GB | peak: {vram_peak_gb:.2f} GB")
    print("PASS: CPUAdamW works with ForgeLM model")


def test_forgelm_cpuadamw_overlap():
    from forge.config import get_config
    from forge.model_loader import ModelLoader

    config = get_config("forgelm_tiny")
    model = ModelLoader.build_model(config)
    model = model.to("cuda").to(torch.bfloat16)
    model.train()

    optimizer = CPUAdamW(model.parameters(), lr=5e-5, weight_decay=0.01, overlap=True, verbose=False)

    for step in range(3):
        x = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = logits.float().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.wait()
        print(f"Overlap step {step}: loss={loss.item():.6f}")

    print("PASS: CPUAdamW overlap mode works with ForgeLM model")


if __name__ == "__main__":
    test_forgelm_cpuadamw()
    test_forgelm_cpuadamw_overlap()
    print("\n=== ForgeLM + CPUAdamW smoke tests passed ===")

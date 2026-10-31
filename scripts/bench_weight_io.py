"""Microbenchmark: checkpoint -> GPU weight loading strategies on V2 ckpt.

Compares the production backends for ForgeLM_V2.safetensors (6.39 GB,
463 tensors) plus a correctness spot-check of the pipelined loader:
  1. safetensors per-tensor (fallback path)
  2. load_safetensors_pipelined (production fast path)

Run: venv/Scripts/python.exe scripts/bench_weight_io.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.runtime.configure import configure

configure()

import torch

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "research", "checkpoints", "ForgeLM_V2.safetensors")


def bench(name, fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    n_bytes = sum(t_.numel() * t_.element_size() for t_ in out.values())
    print(f"  {name:<44} {dt:6.2f}s  ({n_bytes / 1e9:.2f} GB, "
          f"{n_bytes / dt / 1e9:.2f} GB/s)")
    del out
    torch.cuda.empty_cache()
    return dt


def via_safetensors_cuda():
    from safetensors import safe_open
    state = {}
    with safe_open(CKPT, framework="pt", device="cuda") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def via_pipelined():
    from forge.checkpoint_io import load_safetensors_pipelined
    return load_safetensors_pipelined(CKPT, "cuda")


def main():
    sz = os.path.getsize(CKPT)
    print(f"checkpoint: {sz / 1e9:.2f} GB — {CKPT}")
    free, _tot = torch.cuda.mem_get_info()
    print(f"VRAM free: {free / 1e9:.1f} GB\n")

    t = time.perf_counter()
    with open(CKPT, "rb") as f:
        while f.read(64 * 1024 * 1024):
            pass
    print(f"  {'plain file read (disk floor)':<44} "
          f"{time.perf_counter() - t:6.2f}s\n")

    # correctness spot-check first
    from safetensors import safe_open
    state = via_pipelined()
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        bad = sum(1 for k in keys[::23]
                  if not torch.equal(f.get_tensor(k), state[k].cpu()))
        print(f"  pipelined correctness: {len(keys[::23]) - bad}"
              f"/{len(keys[::23])} sampled tensors identical\n")
    del state
    torch.cuda.empty_cache()

    bench("safetensors cuda per-tensor (fallback)", via_safetensors_cuda)
    bench("load_safetensors_pipelined (prod)", via_pipelined)
    bench("load_safetensors_pipelined (warm)", via_pipelined)


if __name__ == "__main__":
    main()

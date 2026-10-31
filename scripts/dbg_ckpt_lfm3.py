"""Minimal repro: which block type breaks torch.utils.checkpoint recompute."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from forge.config import get_config
from forge.model.layers import ModularBlock

torch.manual_seed(0)
cfg = get_config("lfm3_1p7b", device="cpu")
cfg.n_layers = 24

x = torch.randn(1, 16, cfg.d_model, dtype=torch.float32, requires_grad=True)


def probe(name, layer_idx):
    blk = ModularBlock(cfg, layer_idx=layer_idx)
    blk._gradient_checkpointing = True
    blk._gradient_checkpointing_strategy = "all"
    blk.train()
    xin = x.detach().clone().requires_grad_(True)
    try:
        out, _ = blk(xin, use_cache=False)
        out.sum().backward()
        print(f"{name} (layer {layer_idx}): OK")
    except Exception as e:
        print(f"{name} (layer {layer_idx}): FAIL {type(e).__name__}: "
              f"{str(e)[:200]}")


probe("conv", 0)
probe("full_attention", 2)
probe("sliding_attention", 4)

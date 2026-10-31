"""Full-model repro of the checkpoint recompute mismatch (training path)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.utils.checkpoint as cp

from forge.config import get_config
from forge.model.llm import ConfigurableResearchLLM

cp.set_checkpoint_debug_enabled(True)

torch.manual_seed(0)
cfg = get_config("lfm3_1p7b", device="cpu")
cfg.n_layers = 5            # conv,conv,full,conv,swa
cfg.vocab_size = 1024       # tiny head for speed
model = ConfigurableResearchLLM(cfg)
model.train()

# mimic sft_train: idx + targets, no cache, grad ckpt on
B, T = 1, 32
idx = torch.randint(0, 1024, (B, T))
tgt = torch.randint(0, 1024, (B, T))

try:
    out = model(idx, tgt)
    loss = out[1] if isinstance(out, tuple) else out
    if not torch.is_tensor(loss):
        loss = out[0].sum()
    loss.backward()
    print("OK")
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")

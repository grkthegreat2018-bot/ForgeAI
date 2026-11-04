"""E2E: engine.consolidate_pairs() — the L2 loop as an engine API.

Feeds facts into the SuffixMemory sidecar, consolidates into LoRA
weights, verifies fact CE drops while control CE stays ~flat.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from forge.engine.forge_engine import ForgeEngine

CKPT = "research/checkpoints/lfm3_1p7b_cc"

FACTS = [
    ("The Meridian project's vault code is", " 7741."),
    ("The harbor relay uses uplink port", " 9917."),
    ("Dr. Okafor keeps the lab notebook in drawer", " kappa-3."),
    ("The Nighthawk migration starts on", " March 17th."),
    ("Our staging cluster's admin token is", " skt-44-breeze."),
    ("The backup cronjob runs at", " 03:40 every night."),
]
REPLAY = [
    ("Write a Python function to check if a number is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n"
     "    return all(n % i for i in range(2, int(n**0.5)+1))"),
    ("What is the capital of France?", "The capital of France is Paris."),
]

eng = ForgeEngine.from_checkpoint(CKPT, config_name="lfm3_1p7b",
                                  device="cuda", auto_activate=False)
mem = eng.enable_suffix_memory()
tok = eng.tokenizer

# Ingest the facts into L1.
for p, a in FACTS:
    mem.feed(tok.encode(p + a))

# Consolidate: L1-gated pairs -> LoRA weights.
out = eng.consolidate_pairs(
    FACTS, replay=REPLAY, lr=2e-4, steps=20, rank=8,
    control="Attention is all you need. Transformers use scaled dot-product.",
    save_adapter="research/checkpoints/_consol_smoke.lora.safetensors")
print("consolidate:", out)
print("lora_info:", eng.lora_info())

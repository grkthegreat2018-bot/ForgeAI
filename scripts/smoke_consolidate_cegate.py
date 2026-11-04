"""Smoke: consolidate_pairs CE gating (min_fit_ce skip + replay_topk)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.engine.forge_engine import ForgeEngine

eng = ForgeEngine.from_checkpoint(
    "research/checkpoints/lfm3_1p7b_cc.distill250_fq",
    config_name="lfm3_1p7b", device="cuda")

pairs = [
    ("The capital of Frobozz is Zxqwville.", "Zxqwville"),
    ("Q: what is 2+2?", "4"),              # known — should CE-skip
    ("The Flibberty river flows through Blandfordia.", "Blandfordia"),
]
replay = [
    ("def add(a,b):", "return a + b"),
    ("France capital", "Paris"),
    ("import os\nos.getcwd()", "returns cwd"),
    ("1+1", "2"),
    ("water formula", "H2O"),
]

for low_ce in (False, True):
    r = eng.consolidate_pairs(
        pairs, replay=replay, lr=2e-4, steps=20, rank=8,
        min_fit_ce=1.0, replay_topk=2, replay_low_ce=low_ce,
        control="Photosynthesis converts sunlight into chemical energy "
                "in plants.")
    print(f"replay_low_ce={low_ce}: {r}")

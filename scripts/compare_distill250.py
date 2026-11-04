"""Honest A/B: base lfm3 vs distill250 on code + agentic prompts.

LFM3-1.7B is an instruct model — instruction prompts are rendered through
the shipped chat_template.jinja; code-continuation prompts stay raw.
Sampling follows the official generation_config (temp 0.1, top_k 50,
rep_penalty 1.1): pure greedy loops on this family.
Sequential engine loads (12GB card).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from forge.engine.forge_engine import ForgeEngine
from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "research/checkpoints/lfm3_1p7b_cc"
TEMPLATE_SRC = (CKPT_DIR / "chat_template.jinja").read_text(encoding="utf-8")

# (prompt, use_chat_template)
PROMPTS = [
    ("Write a Python function that returns the nth Fibonacci number.", True),
    ("def quicksort(arr):", False),
    ("Write a function to read a JSON file and return the value at a "
     "dotted key path.", True),
    ("Call the function `get_weather` with city='Paris' to answer the "
     "user.", True),
]

CKPTS = [
    ("base", "research/checkpoints/lfm3_1p7b_cc"),
    ("distill250_fq", "research/checkpoints/lfm3_1p7b_cc.distill250_fq"),
]

GEN = dict(max_new_tokens=120, temperature=0.1, top_k=50,
           repetition_penalty=1.1)


def render(p: str, use_template: bool) -> str:
    if not use_template:
        return p
    text, _ = render_jinja_template(
        conversations=[[{"role": "user", "content": p}]],
        chat_template=TEMPLATE_SRC, add_generation_prompt=True,
        bos_token="")
    return text[0]


for name, ckpt in CKPTS:
    torch.cuda.empty_cache()
    eng = ForgeEngine.from_checkpoint(
        ckpt, config_name="lfm3_1p7b", device="cuda")
    print(f"\n{'='*60}\n{name}: {ckpt}", flush=True)
    for p, templated in PROMPTS:
        out = eng.generate(render(p, templated), **GEN)
        print(f"\n--- {'[chat]' if templated else '[raw]'} {p[:60]}\n"
              f"{out[:400]}", flush=True)
    del eng
    torch.cuda.empty_cache()

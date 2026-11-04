"""Spark-X2.6-1.7B forge-engine bench — load VRAM, decode tok/s, quality.

Official sampling (generation_config): temp 1.0, top_p 0.95, no top_k,
rep 1.0. Thinking mode defaults ON via chat_template.jinja.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from forge.engine.forge_engine import ForgeEngine
from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "research/checkpoints/spark_x26_1p7b"
TEMPLATE_SRC = (CKPT_DIR / "chat_template.jinja").read_text(encoding="utf-8")

PROMPTS = [
    ("Write a Python function that returns the nth Fibonacci number.", True),
    ("def quicksort(arr):", False),
    ("Call the function `get_weather` with city='Paris' to answer the "
     "user.", True),
]

GEN = dict(max_new_tokens=160, temperature=1.0, top_p=0.95, top_k=0,
           repetition_penalty=1.0)


def render(p: str) -> str:
    text, _ = render_jinja_template(
        conversations=[[{"role": "user", "content": p}]],
        chat_template=TEMPLATE_SRC, add_generation_prompt=True,
        bos_token="")
    return text[0]


import os
_plain = os.environ.get("PLAIN")
ckpt = os.environ.get("CKPT", str(CKPT_DIR))
eng = ForgeEngine.from_checkpoint(
    ckpt, config_name="spark_x26_1p7b", device="cuda",
    **({"load_quantize": "none", "quantize": False,
        "use_compile": False, "use_breakable_cuda_graph": False,
        "use_block_fusion": False, "use_prefix_cache": False,
        "use_learned_prefix_cache": False}
       if _plain else
       {"load_quantize": "none", "quantize": False,
        "use_breakable_cuda_graph": False, "use_block_fusion": False}
       if os.environ.get("NOG") else
       {"load_quantize": "none", "quantize": False,
        "use_block_fusion": False}
       if os.environ.get("NOF") else {}))
print(f"loaded: {torch.cuda.memory_allocated()/2**30:.2f}GB alloc, "
      f"{torch.cuda.memory_reserved()/2**30:.2f}GB reserved", flush=True)

# warmup + timed decode
_ = eng.generate(render("hi"), max_new_tokens=8, temperature=0.0)
for p, templated in PROMPTS:
    t0 = time.perf_counter()
    out = eng.generate(render(p) if templated else p, **GEN)
    dt = time.perf_counter() - t0
    n = len(eng.tokenizer.encode(out)) if hasattr(eng, 'tokenizer') else 0
    print(f"\n--- {'[chat]' if templated else '[raw]'} {p[:60]}\n"
          f"{out[:600]}\n[{dt:.2f}s, ~{n} tok -> {n/dt:.1f} tok/s]",
          flush=True)
print(f"\npeak: {torch.cuda.max_memory_allocated()/2**30:.2f}GB")

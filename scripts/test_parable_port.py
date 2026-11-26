"""Parable-Granite-4.1-3B port parity check — HF reference vs ForgeAI port.

    venv\\Scripts\\python.exe scripts\\test_parable_port.py [--cpu]

The checkpoint dir ships raw HF-format shards; ForgeAI loads them via
ModelLoader._remap_hf_keys (llama-style names -> forge names) under the
``parable_granite41_3b`` preset, which carries the Granite multipliers
(embeds x12, block outs x0.22, attn logits x0.015625, head logits /10).
This script forwards one granite-format prompt through both stacks and
compares logits — argmax agreement is the exactness criterion at fp16.
"""
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CKPT = ROOT / "research" / "checkpoints" / "parable_granite41_3b"
DEVICE = "cpu" if "--cpu" in sys.argv else (
    "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PROMPT = ("<|start_of_role|>user<|end_of_role|>"
          "Write a python function that checks if a number is prime."
          "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>")


def hf_side(ids):
    from transformers import AutoModelForCausalLM
    print(f"[hf] loading GraniteForCausalLM ({DTYPE}) ...")
    m = AutoModelForCausalLM.from_pretrained(
        str(CKPT), dtype=DTYPE).to(DEVICE).eval()
    with torch.no_grad():
        logits = m(ids).logits.float()
    del m
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return logits


def forge_side(ids):
    from forge.config import get_config
    from forge.model.loader import ModelLoader
    print(f"[forge] building parable_granite41_3b ({DTYPE}) ...")
    cfg = get_config("parable_granite41_3b", device=DEVICE)
    m = ModelLoader.build_model_fast(cfg, str(CKPT), dtype=DTYPE).eval()
    with torch.no_grad():
        out = m(ids)
        logits = (out[0] if isinstance(out, tuple) else out).float()
    return logits


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"[in ] {ids.shape[1]} tokens  bos={ids[0, 0].item()}")

    a = hf_side(ids)
    b = forge_side(ids)
    diff = (a - b).abs()
    max_d, mean_d = diff.max().item(), diff.mean().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    top_hf = a[0, -1].topk(5).indices.tolist()
    top_fm = b[0, -1].topk(5).indices.tolist()
    print(f"HF    top5={top_hf}")
    print(f"FORGE top5={top_fm}")
    print(f"max|dlogit|={max_d:.6f}  mean={mean_d:.6f}  "
          f"argmax-agree={agree * 100:.2f}%")
    ok = agree >= 0.995 and max_d < 0.25
    print("PASS" if ok else "FAIL — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

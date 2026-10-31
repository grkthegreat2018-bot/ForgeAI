"""Merge a trained LoRA adapter into the text-only Gemma-4 checkpoint.

Produces a bf16 merged model dir usable by:
  - transformers / vLLM directly
  - llama.cpp convert_hf_to_gguf.py  ->  GGUF for LM Studio

Usage (training venv):
    venv_train\\Scripts\\python.exe scripts\\distill\\merge_lora.py ^
        --base checkpoints\\gemma4_oblit_text ^
        --adapter adapters\\gemma4_roc_lora ^
        --out checkpoints\\gemma4_roc_merged
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("loading base (bf16, CPU -> GPU tiles) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging ...", flush=True)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)
    for fn in ("chat_template.jinja",):
        p = os.path.join(args.base, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, fn))
    print("merged ->", args.out)


if __name__ == "__main__":
    main()

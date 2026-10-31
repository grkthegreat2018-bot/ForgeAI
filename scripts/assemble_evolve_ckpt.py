"""Assemble a trained .safetensors into a loadable Forge checkpoint dir.

Copies tokenizer + forge_convert.json from the source ported dir, then
drops the trained safetensors in as model.safetensors.

Usage:
    python scripts/assemble_evolve_ckpt.py ^
        --src research/checkpoints/qwen25_coder_3b_mla_moe ^
        --ckpt research/checkpoints/qwen3b_evolve_lora.safetensors ^
        --dst research/checkpoints/qwen25_coder_3b_mla_moe_honed
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FILES = ["forge_convert.json", "generation_config.json", "merges.txt",
         "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "special_tokens_map.json", "added_tokens.json",
         "chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        f = src / name
        if f.exists():
            shutil.copy2(f, dst / name)
    shutil.copy2(args.ckpt, dst / "model.safetensors")
    # sanity: report tensor count + size
    from safetensors import safe_open
    with safe_open(str(dst / "model.safetensors"), "pt") as f:
        print(f"{dst}: {len(list(f.keys()))} tensors, "
              f"{(dst / 'model.safetensors').stat().st_size / 1e9:.2f} GB")
    conv = dst / "forge_convert.json"
    if conv.exists():
        print("forge_convert:", json.loads(conv.read_text()))


if __name__ == "__main__":
    main()

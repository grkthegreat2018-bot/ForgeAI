"""Pre-pack a checkpoint into ForgeQuant format (CPU-only, mem-safe).

Writes a new safetensors where every nn.Linear (in>=64, out>=64) is
replaced by its ForgeQuantLinear packed params:
  <name>.dense_packed   (uint8, out x in//2)
  <name>.dense_scales   (fp16, out x n_groups)
  <name>.sparse_packed  (int8, n_sparse x in)
  <name>.sparse_indices (int64, n_sparse)
  <name>.sparse_scales  (fp16, n_sparse x 1)
  <name>.bias           (kept, if present)

ModelLoader materializes ForgeQuantLinear shells on `.dense_packed` keys
and assigns packed params directly — no bf16 GPU peak at load, and
plain-tensor dequant keeps CUDA-graph capture possible.

Usage:
  venv\\Scripts\\python.exe scripts\\prequant_forge_quant.py \
      --preset qwen25_coder_3b_mla_moe \
      --src research/checkpoints/qwen25_coder_3b_mla_moe \
      --dst research/checkpoints/qwen25_coder_3b_mla_moe_fq
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "generation_config.json", "special_tokens_map.json",
             "chat_template.jinja")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--sparse-ratio", type=float, default=0.10)
    ap.add_argument("--rotate", action="store_true",
                    help="Hadamard-rotate the dense weights at pack time "
                         "(QuaRot-style outlier spreading; requires "
                         "power-of-2 --group-size). Loader detects it via "
                         "forge_convert.json fq_rotation.")
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda — pack math device (outputs stay CPU)")
    args = ap.parse_args()
    dev = torch.device(args.device)

    from forge.config import get_config
    from forge.engine.quant.forge_quant import ForgeQuantLinear
    from forge.model.llm import ConfigurableResearchLLM

    cfg = get_config(args.preset)
    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg)

    # Enumerate packable linears (name -> (in, out, has_bias))
    linears = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and (mod.in_features >= args.min_size
                and mod.out_features >= args.min_size):
            linears[name] = (mod.in_features, mod.out_features,
                             mod.bias is not None)
    print(f"[pack] {len(linears)} linears to pack")

    state = load_file(str(args.src / "model.safetensors"))
    # HF-style sources (model.layers.N.*) must be remapped to Forge names
    # (blocks.N.*) before matching — unmapped linears silently pack 0.
    if any(k.startswith("model.") for k in state):
        from forge.model.loader import ModelLoader
        state = ModelLoader._remap_hf_keys(state, cfg)
    out: dict[str, torch.Tensor] = {}
    packed = 0
    packed_bytes = 0
    for k, v in state.items():
        mod_name = k[:-len(".weight")] if k.endswith(".weight") else None
        if mod_name in linears:
            in_f, out_f, _ = linears[mod_name]
            fq = ForgeQuantLinear(in_f, out_f, bias=False,
                                  group_size=args.group_size,
                                  sparse_ratio=args.sparse_ratio,
                                  use_rotation=args.rotate).to(dev)
            fq.load_from_weight(v.float().to(dev))
            out[f"{mod_name}.dense_packed"] = fq.dense_packed.data.cpu()
            out[f"{mod_name}.dense_scales"] = fq.dense_scales.data.cpu()
            out[f"{mod_name}.sparse_packed"] = fq.sparse_packed.data.cpu()
            out[f"{mod_name}.sparse_indices"] = fq.sparse_indices.data.cpu()
            out[f"{mod_name}.sparse_scales"] = fq.sparse_scales.data.cpu()
            packed += 1
            packed_bytes += (fq.dense_packed.numel()
                             + fq.sparse_packed.numel()
                             + fq.dense_scales.numel() * 2
                             + fq.sparse_scales.numel() * 2
                             + fq.sparse_indices.numel() * 8)
            if packed % 200 == 0:
                print(f"[pack] {packed}/{len(linears)} ...")
        else:
            out[k] = v

    args.dst.mkdir(parents=True, exist_ok=True)
    save_file({k: t.contiguous() for k, t in out.items()},
              str(args.dst / "model.safetensors"),
              metadata={"format": "pt"})
    for name in TOK_FILES:
        f = args.src / name
        if f.exists():
            shutil.copy(f, args.dst / name)
    conv = json.loads((args.src / "forge_convert.json").read_text()) \
        if (args.src / "forge_convert.json").exists() else {}
    conv["weight_format"] = "forge_quant"
    if args.rotate:
        conv["fq_rotation"] = True          # loader materializes rot shells
    (args.dst / "forge_convert.json").write_text(json.dumps(conv, indent=2))
    total = sum(t.numel() * t.element_size() for t in out.values())
    print(f"[pack] {packed} linears packed | {total/1e9:.2f}GB written -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

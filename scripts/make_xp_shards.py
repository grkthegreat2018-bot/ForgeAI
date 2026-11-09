"""Split a checkpoint into xp_shards/ for AirLLMStreamer.

Layout the streamer expects (forge/param name space):
  <ckpt_dir>/../xp_shards/shard_0.safetensors  — resident non-layer
      tensors (embeddings, norms, head, tied weights)
  <ckpt_dir>/../xp_shards/shard_{i}.safetensors — blocks.{i-1}.* params

The streamer binds shard i into the model's blocks.(i-1) params by
suffix match. Run once per checkpoint; ~1x ckpt size of sequential
writes (safe on the SMR D: drive).

  venv\\Scripts\\python.exe scripts/make_xp_shards.py \\
    --model research/checkpoints/spark_x26_1p7b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None,
                    help="default <parent>/xp_shards")
    args = ap.parse_args()

    from forge.config import get_config
    from forge.model_loader import ModelLoader
    from safetensors.torch import save_file

    conv = Path(args.model) / "forge_convert.json"
    cfg_name = (json.loads(conv.read_text())["forge_config"]
                if conv.exists() else "spark_x26_1p7b")
    cfg = get_config(cfg_name, device="cpu")
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.model,
                                         dtype=torch.bfloat16)
    sd = model.state_dict()

    out = Path(args.out or (Path(args.model).parent / "xp_shards"))
    out.mkdir(parents=True, exist_ok=True)

    layers: dict[int, dict[str, torch.Tensor]] = {}
    resident: dict[str, torch.Tensor] = {}
    pat = re.compile(r"^blocks\.(\d+)\.")
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            layers.setdefault(int(m.group(1)), {})[k] = v.contiguous()
        else:
            # clone: tied embed/head share storage — safetensors
            # refuses (or silently corrupts) shared tensors
            resident[k] = v.detach().clone().contiguous()

    save_file(resident, str(out / "shard_0.safetensors"))
    print(f"shard_0: {len(resident)} resident tensors "
          f"({sum(v.numel() for v in resident.values())/1e6:.0f}M)")
    for i in sorted(layers):
        save_file(layers[i], str(out / f"shard_{i + 1}.safetensors"))
    print(f"wrote {len(layers)} layer shards -> {out}")


if __name__ == "__main__":
    main()

"""Train LoopGates on a corpus and save for FORGE_LOOPTHINK decode.

  venv\\Scripts\\python.exe scripts/train_loopthink_gates.py \\
    --model research/checkpoints/spark_x26_1p7b \\
    --data data/sft/distill_v10.scored.jsonl --rows 64 --steps 400 \\
    --loop-lo 8 --loop-hi 20 --max-iters 3 \\
    --out research/checkpoints/loopgates_sft.pt

Then:  set FORGE_LOOPTHINK=8,20,3,0.15,research/checkpoints/loopgates_sft.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _rows(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (r.get("text") or r.get("prompt") or r.get("instruction")
                 or "")
            if not t and r.get("messages"):
                t = "\n".join(m.get("content", "") for m in r["messages"])
            if not t and r.get("response"):
                t = str(r["response"])
            if t.strip():
                out.append(t)
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--loop-lo", type=int, default=8)
    ap.add_argument("--loop-hi", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from forge.config import get_config
    from forge.exp.loopdepth import train_loop_gates
    from forge.model_loader import ModelLoader
    from transformers import AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    conv = Path(args.model) / "forge_convert.json"
    cfg_name = (json.loads(conv.read_text())["forge_config"]
                if conv.exists() else "spark_x26_1p7b")
    cfg = get_config(cfg_name, device="cuda")
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.model,
                                         dtype=torch.bfloat16).to(dev).eval()

    texts = _rows(args.data, args.rows)
    print(f"{len(texts)} training texts")
    ids_list = [tok(t, return_tensors="pt", truncation=True,
                    max_length=args.seq_len)["input_ids"].to(dev)
                for t in texts]
    ids_list = [i for i in ids_list if i.shape[1] >= 8]

    gates, losses = train_loop_gates(
        model, ids_list, args.loop_lo, args.loop_hi,
        max_iters=args.max_iters, steps=args.steps, lr=args.lr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"g": gates.g.detach().cpu(),
                "loop_lo": args.loop_lo, "loop_hi": args.loop_hi,
                "train_loss": (losses[0], losses[-1])}, out)
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"gates: {gates.g.detach().cpu().numpy().round(3)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

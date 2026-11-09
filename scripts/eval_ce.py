"""Quick CE eval: checkpoint x data-file -> mean loss. FisherFold verdict:

FisherFold(sft6,base) should keep sft6's skill on ITS corpus (low CE on
data-new) while reverting to base on general text (CE ~= base on
data-base) — the anti-forgetting signature a 50/50 soup can't produce.

  venv\\Scripts\\python.exe scripts/eval_ce.py --model CKPT --data F --rows 16
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
            t = (r.get("text") or r.get("output") or r.get("response")
                 or "")
            if (r.get("prompt") or "") and (r.get("response") or ""):
                t = r["prompt"] + "\n" + r["response"]
            if not t and r.get("messages"):
                t = "\n".join(m.get("content", "") for m in r["messages"])
            if t.strip():
                out.append(t)
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    from forge.config import get_config
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

    losses = []
    with torch.inference_mode():
        for t in _rows(args.data, args.rows):
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=args.seq_len)["input_ids"].to(dev)
            if ids.shape[1] < 8:
                continue
            out = model(ids)
            lg = (out.logits if hasattr(out, "logits")
                  else (out[0] if isinstance(out, tuple) else out))
            losses.append(float(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
                ids[:, 1:].reshape(-1))))
    print(f"{args.model} on {args.data}: "
          f"CE {sum(losses)/len(losses):.4f} (n={len(losses)})")


if __name__ == "__main__":
    main()

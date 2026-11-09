"""LoopThink bench (R71/B5): does verifier-gated recurrent depth help?

Measures on a real checkpoint:
  - standard forward eval loss
  - looped prefill loss (middle blocks iterated, KV committed from last iter)
  - verifier-gated: iters actually used + stop reasons + wall-time cost

  venv\\Scripts\\python.exe scripts/bench_loopthink.py \\
    --model research/checkpoints/spark_x26_1p7b --rows 8 --max-iters 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.exp.loopdepth import (VerifierGate, looped_prefill,
                                 looped_prefill_verified)  # noqa: E402


def _ce(logits, ids):
    return float(torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        ids[:, 1:].reshape(-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/sft/sft9_external.jsonl")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--loop-lo", type=int, default=8)
    ap.add_argument("--loop-hi", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--train-gates", type=int, default=0,
                    help="steps of LoopGates training before eval")
    ap.add_argument("--train-data", default=None,
                    help="separate corpus for gate training (cross-domain test)")
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

    texts = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text") or ((r.get("prompt") or "") + "\n"
                                  + (r.get("response") or ""))
            if t.strip():
                texts.append(t)
            if len(texts) >= args.rows:
                break

    trained = None
    if args.train_gates > 0:
        from forge.exp.loopdepth import train_loop_gates
        if args.train_data:
            train_texts = []
            with open(args.train_data, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    t = (r.get("text") or r.get("prompt")
                         or r.get("instruction") or "")
                    if not t and r.get("messages"):
                        t = "\n".join(m.get("content", "")
                                      for m in r["messages"])
                    if t.strip():
                        train_texts.append(t)
                    if len(train_texts) >= 64:
                        break
            print(f"cross-domain: {len(train_texts)} train rows from "
                  f"{args.train_data} / {len(texts)} eval")
        else:
            half = max(1, len(texts) // 2)
            train_texts, texts = texts[:half], texts[half:]
            print(f"holdout split: {len(train_texts)} train / "
                  f"{len(texts)} eval")
        train_ids = [tok(t, return_tensors="pt", truncation=True,
                         max_length=args.seq_len)["input_ids"].to(dev)
                     for t in train_texts]
        trained, tl = train_loop_gates(
            model, train_ids, args.loop_lo, args.loop_hi,
            max_iters=args.max_iters, steps=args.train_gates, lr=3e-2)
        print(f"gate train: loss {tl[0]:.3f} -> {tl[-1]:.3f} | "
              f"gates {trained.g.detach().cpu().numpy().round(3)}")

    res = {"std": [], "loop": [], "gate": []}
    iters_used, stops, t_std, t_loop, t_gate = [], [], 0.0, 0.0, 0.0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=args.seq_len)["input_ids"].to(dev)
        if ids.shape[1] < 8:
            continue
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            out = model(ids)
            lg = (out.logits if hasattr(out, "logits")
                  else (out[0] if isinstance(out, tuple) else out))
        torch.cuda.synchronize(); t_std += time.time() - t0
        res["std"].append(_ce(lg, ids))

        torch.cuda.synchronize(); t0 = time.time()
        lg, _, st = looped_prefill(model, ids, args.loop_lo, args.loop_hi,
                                   max_iters=args.max_iters)
        torch.cuda.synchronize(); t_loop += time.time() - t0
        res["loop"].append(_ce(lg, ids))

        gate = VerifierGate(min_margin=args.margin, patience=1)
        torch.cuda.synchronize(); t0 = time.time()
        lg, _, st = looped_prefill_verified(
            model, ids, args.loop_lo, args.loop_hi,
            max_iters=args.max_iters,
            gates=trained.g if trained is not None else None,
            gate=gate)
        torch.cuda.synchronize(); t_gate += time.time() - t0
        res["gate"].append(_ce(lg, ids))
        iters_used.append(st["iters_run"])
        stops.append(st["stop_reason"])

    n = len(res["std"])
    print(f"\n=== LoopThink bench ({n} texts, layers {args.loop_lo}-"
          f"{args.loop_hi}, max_iters {args.max_iters}) ===")
    print(f"std loss    : {sum(res['std'])/n:.4f}  ({t_std/n*1000:.0f}ms)")
    print(f"looped loss : {sum(res['loop'])/n:.4f}  ({t_loop/n*1000:.0f}ms)")
    print(f"gated  loss : {sum(res['gate'])/n:.4f}  ({t_gate/n*1000:.0f}ms)")
    print(f"iters used  : {iters_used}  mean {sum(iters_used)/n:.2f}")
    print(f"stop reasons: {stops}")


if __name__ == "__main__":
    main()

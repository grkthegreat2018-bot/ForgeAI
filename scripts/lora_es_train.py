"""LoRA-ES training driver (R71/B3): forward-only adapter SFT.

Attaches rank-r LowRankBypass adapters to matched Linear layers, then
trains them with zeroth-order ES — no backward pass, so peak memory ≈
inference (weights + 1 batch of logits) instead of full SFT.

  venv\\Scripts\\python.exe scripts/lora_es_train.py \\
    --model research/checkpoints/spark_x26_1p7b \\
    --data data/sft/spark_sft_v8.jsonl --steps 200 --n-dirs 8 \\
    --rank 8 --eps 1e-3 --lr 1e-3 --layers-matching mlp \\
    --adapter-out research/adapters/lora_es_r8.pt

--measure-vram also runs one backprop step for a peak-memory comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.training.lora_es import LoRAES, attach_lora  # noqa: E402


def _rows(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text") or ((r.get("prompt") or "") + "\n" +
                                  (r.get("response") or ""))
            if t.strip():
                out.append(t)
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-dirs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--init", default="randn_ab",
                    choices=["zero", "randn", "randn_ab", "pissa"])
    ap.add_argument("--layers-matching", default="ffn",
                    help="substring filter for Linear names")
    ap.add_argument("--perturb-mode", default="full",
                    choices=["full", "block", "fisher"])
    ap.add_argument("--adapter-out", default=None)
    ap.add_argument("--measure-vram", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-decay", choices=["none", "linear", "cosine"],
                    default="linear",
                    help="lr schedule over --steps; ZO noise compounds so "
                         "a constant lr random-walks past the minimum")
    ap.add_argument("--eval-every", type=int, default=10)
    args = ap.parse_args()

    from forge.config import get_config
    from forge.model_loader import ModelLoader
    from transformers import AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    conv = Path(args.model) / "forge_convert.json"
    cfg_name = (json.loads(conv.read_text())["forge_config"]
                if conv.exists() else "spark_x26_1p7b")
    cfg = get_config(cfg_name, device=str(dev))
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.model,
                                         dtype=torch.bfloat16).to(dev)
    model.requires_grad_(False)

    named_lora: dict[str, torch.Tensor] = {}
    train = attach_lora(model, rank=args.rank, init=args.init,
                        name_filter=lambda n: args.layers_matching in n)
    for name, mod in model.named_modules():
        if hasattr(mod, "A") and hasattr(mod, "B"):
            named_lora[f"{name}.A"], named_lora[f"{name}.B"] = mod.A, mod.B
    print(f"attached {len(train)} LoRA tensors "
          f"({sum(p.numel() for p in train)/1e6:.2f}M trainable)")

    texts = _rows(args.data, args.rows)
    rng = torch.Generator().manual_seed(args.seed)
    # fixed eval set (last 8 rows) — per-step batch loss is text-noise,
    # the learning signal only shows on a constant set
    eval_ids = [tok(t, return_tensors="pt", truncation=True,
                    max_length=args.seq_len)["input_ids"].to(dev)
                for t in texts[-8:]]
    texts = texts[:-8]

    @torch.inference_mode()
    def eval_ce():
        tot = 0.0
        for ids in eval_ids:
            out = model(ids[:, :-1])
            lg = (out.logits if hasattr(out, "logits")
                  else (out[0] if isinstance(out, tuple) else out))
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]).float(),
                ids[:, 1:].reshape(-1)))
        return tot / len(eval_ids)

    def make_loss():
        text = texts[torch.randint(len(texts), (1,), generator=rng).item()]
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.seq_len)["input_ids"].to(dev)
        if ids.shape[1] < 4:
            return lambda: 0.0

        def loss_fn():
            with torch.inference_mode():
                out = model(ids[:, :-1])
                logits = (out.logits if hasattr(out, "logits")
                          else (out[0] if isinstance(out, tuple) else out))
                return float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1)))
        return loss_fn

    es = LoRAES(train, names=list(named_lora.keys()), eps=args.eps,
                lr=args.lr, n_dirs=args.n_dirs, perturb_mode=args.perturb_mode,
                seed=args.seed)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"eval@0 CE {eval_ce():.4f}")
    # ZO estimates are noisy — keep the best-eval adapter seen, not the
    # final one (constant lr random-walks past the minimum on long runs).
    best_ce, best_sd = float("inf"), None
    for step in range(args.steps):
        if args.lr_decay == "linear":
            es.lr = args.lr * max(0.0, 1.0 - step / args.steps)
        elif args.lr_decay == "cosine":
            import math
            es.lr = args.lr * 0.5 * (1.0 + math.cos(
                math.pi * step / args.steps))
        info = es.step(make_loss())
        if step % args.eval_every == 0 or step == args.steps - 1:
            ce = eval_ce()
            if ce < best_ce:
                best_ce = ce
                best_sd = {k: v.detach().cpu().clone()
                           for k, v in named_lora.items()}
            peak = (torch.cuda.max_memory_allocated() / 2**30
                    if dev.type == "cuda" else 0)
            print(f"step {step:4d} loss {info['loss']:.4f} "
                  f"|g| {info['grad_norm']:.3f} peakVRAM {peak:.2f}GB "
                  f"evalCE {ce:.4f}")

    if args.measure_vram and dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        p = [t for t in train]
        for t in p:
            t.requires_grad_(True)
        ids = tok(texts[0], return_tensors="pt", truncation=True,
                  max_length=args.seq_len)["input_ids"].to(dev)
        out = model(ids[:, :-1])
        logits = (out.logits if hasattr(out, "logits")
                  else (out[0] if isinstance(out, tuple) else out))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1))
        loss.backward()
        bp = torch.cuda.max_memory_allocated() / 2**30
        print(f"backprop-step peak: {bp:.2f}GB  (ES peak above — "
              f"savings ≈ activations+grads)")

    if args.adapter_out:
        out = Path(args.adapter_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sd = best_sd if best_sd is not None else {
            k: v.detach().cpu() for k, v in named_lora.items()}
        torch.save(sd, out)
        print(f"saved adapter {out} (best evalCE {best_ce:.4f})")


if __name__ == "__main__":
    main()

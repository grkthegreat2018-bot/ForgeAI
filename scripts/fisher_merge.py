"""FisherFold driver (R71/B4): diagonal-Fisher-weighted merge of two checkpoints.

  w' = w_base + (w_new - w_base) * F_new / (F_base + F_new + eps)   [mode=gate]

F_base is the empirical diagonal Fisher of the BASE model on --data-base;
F_new is that of the NEW model on --data-new. Params repurposed by the new
task keep their delta; base-critical or unused params revert to base — the
anti-forgetting merge for rescuing regressed sftN rounds.

  venv\\Scripts\\python.exe scripts/fisher_merge.py \\
    --base research/checkpoints/spark_x26_1p7b \\
    --new  research/checkpoints/spark_x26_1p7b.sft6_fq \\
    --data-base data/sft/replay_general.jsonl --data-new data/sft/spark_sft_v6.jsonl \\
    --n 24 --out research/checkpoints/spark_x26_1p7b.sft6_fisherfold

Data files: jsonl rows with "text", or "prompt"+"response" fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.merge_models import fisher_diag, merge_fisher  # noqa: E402


def _load_texts(path: str, limit: int) -> list[str]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                rows.append(line)
                continue
            t = r.get("text") or r.get("output") or r.get("response")
            if (r.get("prompt") or "") and (r.get("response") or ""):
                t = r["prompt"] + "\n" + r["response"]
            elif r.get("task_desc"):
                t = r["task_desc"] + "\n" + (r.get("solution") or "")
            if t:
                rows.append(t)
            if len(rows) >= limit:
                break
    return rows


def _state_via_safetensors(ckpt: str) -> dict:
    from safetensors.torch import load_file
    p = Path(ckpt)
    if p.is_dir():
        state = {}
        for sh in sorted(p.glob("*.safetensors")):
            state.update(load_file(str(sh)))
        return state
    return load_file(str(p))


def _save(ckpt_out: str, state: dict) -> None:
    from safetensors.torch import save_file
    out = Path(ckpt_out)
    if out.suffix != ".safetensors":
        out.mkdir(parents=True, exist_ok=True)
        out = out / "model.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(out))
    print(f"saved {out} ({sum(v.numel() for v in state.values())/1e9:.2f}B params)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-base", required=True)
    ap.add_argument("--data-new", required=True)
    ap.add_argument("--n", type=int, default=24, help="fisher batches per side")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--rows", type=int, default=512, help="max data rows")
    ap.add_argument("--mode", choices=["gate", "interp"], default="gate")
    ap.add_argument("--eps", type=float, default=1e-20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fisher-device", default=None,
                    help="where fisher accumulates (default: model device; use cpu when tight)")
    args = ap.parse_args()

    from forge.config import get_config
    from forge.model_loader import ModelLoader
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.base)

    base_sd = _state_via_safetensors(args.base)
    new_sd = _state_via_safetensors(args.new)

    texts_b = _load_texts(args.data_base, args.rows)
    texts_n = _load_texts(args.data_new, args.rows)
    print(f"data: {len(texts_b)} base rows, {len(texts_n)} new rows")

    def make_loss(model):
        def loss_fn(model, text):
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.seq_len).to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 4:
                return model(ids).logits.sum() * 0.0
            out = model(ids[:, :-1])
            logits = (out.logits if hasattr(out, "logits")
                      else (out[0] if isinstance(out, tuple) else out))
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                ids[:, 1:].reshape(-1))
        return loss_fn

    conv = Path(args.base) / "forge_convert.json"
    cfg_name = (json.loads(conv.read_text())["forge_config"]
                if conv.exists() else "spark_x26_1p7b")
    cfg = get_config(cfg_name, device=str(device))
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=args.base, dtype=torch.bfloat16)
    model.to(device)
    model.requires_grad_(True)

    loss_fn = make_loss(model)
    print(f"computing F_base over {args.n} batches...")
    f_base = fisher_diag(model, loss_fn, iter(texts_b), args.n,
                         accumulate_device=args.fisher_device)

    # swap in the new weights (param-name space == model.named_parameters)
    named = dict(model.named_parameters())
    loaded = missed = 0
    with torch.no_grad():
        for n, p in named.items():
            for cand in (n, n.replace("blocks.", "model.layers."),
                         "model." + n, n.replace("model.layers.", "blocks.")):
                if cand in new_sd and new_sd[cand].shape == p.shape:
                    p.data.copy_(new_sd[cand].to(p.device, p.dtype))
                    loaded += 1
                    break
            else:
                missed += 1
    print(f"swapped {loaded} params from --new ({missed} unmatched)")

    print(f"computing F_new over {args.n} batches...")
    f_new = fisher_diag(model, loss_fn, iter(texts_n), args.n,
                        accumulate_device=args.fisher_device)
    del model
    torch.cuda.empty_cache()

    # align fisher dicts to checkpoint key space via the same candidates
    def remap(fdict, sd):
        out = {}
        for n, f in fdict.items():
            for cand in (n, n.replace("blocks.", "model.layers."),
                         "model." + n, n.replace("model.layers.", "blocks.")):
                if cand in sd:
                    out[cand] = f
                    break
        return out

    merged = merge_fisher(base_sd, new_sd, remap(f_base, base_sd),
                          remap(f_new, new_sd), eps=args.eps, mode=args.mode)
    _save(args.out, merged)


if __name__ == "__main__":
    main()

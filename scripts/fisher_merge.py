"""FisherFold driver (R71/B4): diagonal-Fisher-weighted merge, real ckpts.

  w' = w_base + (w_new - w_base) * F_new / (F_base + F_new + eps)  [gate]

Loads BOTH checkpoints through ModelLoader (forge name space — HF key
remap incl. fused-qkv is handled once on load), computes empirical
diagonal Fisher for each on its own calibration corpus, merges, saves.

VRAM plan (RTX 5070 12GB): one model at a time, bf16 (~4.3GB) + grads
(~4.3GB) + activations; Fisher accumulates fp32 on CPU (--fisher-device
cpu) so no extra 8.6GB on GPU. ~0.2s/batch D2H cost.

  venv\\Scripts\\python.exe scripts/fisher_merge.py \\
    --base research/checkpoints/spark_x26_1p7b \\
    --new  research/checkpoints/spark_x26_1p7b.sft6_src \\
    --data-base data/sft/nontool_general.jsonl \\
    --data-new  data/sft/sft9_external.jsonl \\
    --n 24 --fisher-device cpu \\
    --out research/checkpoints/spark_x26_1p7b.sft6_fisherfold
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.merge_models import fisher_diag, merge_fisher  # noqa: E402


def _load_texts(path: str, limit: int) -> list[str]:
    rows = []
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
                rows.append(t)
            if len(rows) >= limit:
                break
    return rows


def _load_model(ckpt: str, device, tok_cache: dict):
    from forge.config import get_config
    from forge.model_loader import ModelLoader
    conv = Path(ckpt) / "forge_convert.json"
    cfg_name = (json.loads(conv.read_text())["forge_config"]
                if conv.exists() else tok_cache["cfg_fallback"])
    cfg = get_config(cfg_name, device="cuda")
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt,
                                         dtype=torch.bfloat16)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-base", required=True)
    ap.add_argument("--data-new", required=True)
    ap.add_argument("--cfg-fallback", default="spark_x26_1p7b")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--mode", choices=["gate", "interp"], default="gate")
    ap.add_argument("--eps", type=float, default=1e-20)
    ap.add_argument("--fisher-device", default=None,
                    help="fisher accumulator device (default: model/GPU)")
    ap.add_argument("--n-groups", type=int, default=4,
                    help="param groups for fisher pass — grads held for "
                         "one group at a time (VRAM fit on 12GB)")
    ap.add_argument("--alpha-new", type=float, default=1.0,
                    help="scale F_new — >1 recovers more donor skill")
    ap.add_argument("--save-fisher", default=None,
                    help="dir to save computed fishers (f_base/f_new.pt)")
    ap.add_argument("--load-fisher", default=None,
                    help="dir with precomputed fishers — skips GPU pass")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.base)
    texts_b = _load_texts(args.data_base, args.rows)
    texts_n = _load_texts(args.data_new, args.rows)
    print(f"data: {len(texts_b)} base rows, {len(texts_n)} new rows")

    def encode(t):
        return tok(t, return_tensors="pt", truncation=True,
                   max_length=args.seq_len)["input_ids"].to(device)

    ids_b = [encode(t) for t in texts_b]
    ids_n = [encode(t) for t in texts_n]
    ids_b = [i for i in ids_b if i.shape[1] >= 8]
    ids_n = [i for i in ids_n if i.shape[1] >= 8]

    def loss_fn(model, ids):
        out = model(ids[:, :-1])
        logits = (out.logits if hasattr(out, "logits")
                  else (out[0] if isinstance(out, tuple) else out))
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1))

    tk = {"cfg_fallback": args.cfg_fallback}
    t0 = time.time()
    fdir = Path(args.load_fisher or args.save_fisher or "")
    if args.load_fisher:
        f_base = torch.load(fdir / "f_base.pt", map_location="cpu",
                            weights_only=True)
        f_new = torch.load(fdir / "f_new.pt", map_location="cpu",
                           weights_only=True)
        print("loaded precomputed fishers")
        base_sd = new_sd = None
    else:
        print("loading base model...")
        model = _load_model(args.base, device, tk)
        model.requires_grad_(True)
        base_sd = {k: v.detach().cpu()
                   for k, v in model.state_dict().items()}
        print(f"F_base over {args.n} batches x{args.n_groups} groups...")
        f_base = fisher_diag(model, loss_fn, ids_b, args.n,
                             fisher_dtype=torch.bfloat16,
                             accumulate_device=args.fisher_device,
                             n_groups=args.n_groups)
        del model
        torch.cuda.empty_cache()

        print("loading new model...")
        model = _load_model(args.new, device, tk)
        model.requires_grad_(True)
        new_sd = {k: v.detach().cpu()
                  for k, v in model.state_dict().items()}
        print(f"F_new over {args.n} batches x{args.n_groups} groups...")
        f_new = fisher_diag(model, loss_fn, ids_n, args.n,
                            fisher_dtype=torch.bfloat16,
                            accumulate_device=args.fisher_device,
                            n_groups=args.n_groups)
        del model
        torch.cuda.empty_cache()
        print(f"fisher compute: {time.time()-t0:.0f}s")
        if args.save_fisher:
            fdir.mkdir(parents=True, exist_ok=True)
            torch.save(f_base, fdir / "f_base.pt")
            torch.save(f_new, fdir / "f_new.pt")

    if base_sd is None or new_sd is None:
        # load state dicts without keeping a live model
        print("loading sds for merge...")
        model = _load_model(args.base, device, tk)
        base_sd = {k: v.detach().cpu()
                   for k, v in model.state_dict().items()}
        model.load_state_dict(
            _load_model(args.new, device, tk).state_dict())
        new_sd = {k: v.detach().cpu()
                  for k, v in model.state_dict().items()}
        del model
        torch.cuda.empty_cache()

    merged = merge_fisher(base_sd, new_sd, f_base, f_new,
                          eps=args.eps, mode=args.mode,
                          alpha_new=args.alpha_new)
    del f_base, f_new, base_sd, new_sd

    from safetensors.torch import save_file
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "model.safetensors"
    save_file({k: v.contiguous() for k, v in merged.items()}, str(fp))
    # carry tokenizer/config files so the dir is self-contained
    import shutil
    for f in Path(args.base).iterdir():
        if f.suffix in (".json", ".jinja", ".txt", ".py") or \
                f.name in ("merges.txt",):
            try:
                shutil.copy2(f, out / f.name)
            except OSError:
                pass
    print(f"saved {fp} ({sum(v.numel() for v in merged.values())/1e9:.2f}B)")


if __name__ == "__main__":
    main()

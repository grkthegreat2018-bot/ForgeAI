"""Train V14-RSI's novel heads + (optionally) substrate — mem-safe.

Two supervised signals, both mined free from forge_evolve.db:

1. CopySpan pointer head — data/sft/edit_spans.jsonl rows carry
   src_ids (parent tokens) + ops ([{tok:[...]}|{copy:[s,e]}]).
   For each row: forward src_ids once (prefill) → post-ln_f hiddens →
   key_cache; teacher-force the op stream; CE on literal tokens +
   copy_span_loss on each copy's (start, end). The base LM also learns to
   emit <copy> at op boundaries (CE on the op token).

2. Verifier head — data/sft/evolve_pairs.jsonl (chosen/rejected).
   Bradley-Terry on last-token ln_f(hidden) of each candidate's tokens.
   Frozen trunk: verifier learns a rerank prior without touching the LM.

Memory: heads train on a FROZEN backbone by default (hiddens are
constants → one forward per row, tiny grads) — fits alongside the
proposer or on CPU. --unfreeze-last-k N enables substrate fine-tune
(slow; run alone).

Usage:
    python scripts/train_v14_heads.py --preset forgelm_v14_rsi \
        --ckpt <v14 ckpt or none> --steps 500 --device cuda
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # ops/src_ids were serialized as repr'd lists in gen_edit_sft —
        # accept both real lists and string forms.
        for k in ("src_ids", "ops"):
            if isinstance(r.get(k), str):
                try:
                    r[k] = ast.literal_eval(r[k])
                except Exception:
                    r[k] = None
        if (r.get("src_ids") and r.get("ops")) or (
                r.get("chosen") and r.get("rejected")):
            rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="forgelm_v14_rsi")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint dir/file for the V14 student "
                         "(omit = random init — heads still train: "
                         "pointer/verifier are read-only taps)")
    ap.add_argument("--spans", default="data/sft/edit_spans.jsonl")
    ap.add_argument("--pairs", default="data/sft/evolve_pairs.jsonl")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unfreeze-last-k", type=int, default=0)
    ap.add_argument("--copy-id", type=int, default=None,
                    help="token id of <copy> op (default: auto-detect "
                         "from tokenizer or config.copy_token_id)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--tokenizer",
                    default="research/checkpoints/forgelm_v2_tokenizer")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from forge.config import get_config
    from forge.model.llm import ConfigurableResearchLLM

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    cfg = get_config(args.preset, device=args.device)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        from forge.model.loader import ModelLoader
        model = ModelLoader.build_model_fast(cfg, args.ckpt)
    else:
        with torch.device("meta"):
            model = ConfigurableResearchLLM(cfg)
        model = model.to_empty(device=dev).to(torch.bfloat16)
        model.apply(lambda m: m.reset_parameters()
                    if hasattr(m, "reset_parameters") else None)
    model.to(dev)
    model.eval() if args.unfreeze_last_k == 0 else model.train()

    # Freeze trunk; train heads (+ optionally last-k blocks).
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for name in ("copy_head", "verifier_head", "edit_head",
                     "copyspan_head", "ptr_head"):
        mod = getattr(model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(True)
            trainable.append(name)
    if args.unfreeze_last_k > 0:
        for blk in model.blocks[-args.unfreeze_last_k:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        trainable.append(f"blocks[-{args.unfreeze_last_k}:]")
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"[heads] trainable: {trainable} = {n_tr/1e6:.2f}M params",
          flush=True)
    if not params:
        print("[heads] FATAL: no head modules found on model", flush=True)
        return
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    spans = _load_rows(args.spans, args.rows)
    pairs = _load_rows(args.pairs, args.rows)
    print(f"[heads] spans={len(spans)} pairs={len(pairs)}", flush=True)

    copy_tok = args.copy_id if args.copy_id is not None else getattr(
        cfg, "copy_token_id", 1)

    ptr_head = None
    ver_head = None
    for name in ("copy_head", "copyspan_head", "ptr_head", "edit_head"):
        if getattr(model, name, None) is not None:
            ptr_head = getattr(model, name)
            break
    for name in ("verifier_head", "verifier"):
        if getattr(model, name, None) is not None:
            ver_head = getattr(model, name)
            break

    from forge.model.edit_head import copy_span_loss, verifier_bt_loss

    t0 = time.time()
    step = 0
    it_s = iter(spans * 20) if spans else iter(())
    it_p = iter(pairs * 20) if pairs else iter(())
    while step < args.steps:
        step += 1
        losses = []
        # --- CopySpan step ---
        row = next(it_s, None)
        if row is not None and ptr_head is not None:
            src = torch.tensor(row["src_ids"], device=dev).unsqueeze(0)
            with torch.no_grad() if args.unfreeze_last_k == 0 else torch.enable_grad():
                out = model(src, use_cache=False, return_hidden=True)
            hidden = out[2] if isinstance(out, tuple) else getattr(
                out, "hidden_states", None)
            if hidden is not None:
                h = hidden[0] if hidden.dim() == 3 else hidden
                src_k = ptr_head.key_cache(h)
                # teacher-forced: for each copy op, the token position
                # BEFORE it emits <copy> then pointer (s, e).
                pos = 0
                l_copy = 0.0
                n_copy = 0
                for op in row["ops"]:
                    if "copy" in op:
                        s, e = op["copy"]
                        h_t = h[min(pos, h.shape[0] - 1)].unsqueeze(0)
                        sl, el = ptr_head(h_t, src_k)
                        l_copy = l_copy + copy_span_loss(
                            sl, el,
                            torch.tensor([s], device=dev),
                            torch.tensor([e], device=dev))
                        n_copy += 1
                        pos += e - s + 1
                    else:
                        pos += len(op["tok"])
                if n_copy:
                    losses.append(l_copy / n_copy)
        # --- Verifier step ---
        pr = next(it_p, None)
        if pr is not None and ver_head is not None:
            def _score(code: str):
                ids = torch.tensor(
                    tok.encode(code)[:2048] or [0],
                    device=dev).unsqueeze(0)
                with torch.no_grad() if args.unfreeze_last_k == 0 else torch.enable_grad():
                    o = model(ids, use_cache=False, return_hidden=True)
                hh = o[2] if isinstance(o, tuple) else getattr(
                    o, "hidden_states", None)
                return ver_head(hh[0, -1] if hh is not None else hh[-1])
            sc = _score(pr.get("chosen", ""))
            sr = _score(pr.get("rejected", ""))
            losses.append(verifier_bt_loss(sc, sr))

        if losses:
            loss = sum(losses) / len(losses)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 25 == 0:
                print(f"[heads] step {step}/{args.steps} "
                      f"loss={loss.item():.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

    if args.save:
        sd = {n: p.detach().cpu() for n, p in model.state_dict().items()
              if any(k in n for k in trainable)}
        from safetensors.torch import save_file
        save_file(sd, args.save)
        print(f"[heads] saved head weights -> {args.save}", flush=True)


if __name__ == "__main__":
    main()

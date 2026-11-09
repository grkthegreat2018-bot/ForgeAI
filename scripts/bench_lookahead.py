"""Lookahead decode (n-gram draft + batched verify) on spark — lossless
parallel decode probe. Compares standard autoregressive decode vs
n-gram-drafted verification, checking token equality and timing.

  venv\\Scripts\\python.exe -X utf8 -u scripts/bench_lookahead.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from forge.config import get_config
from forge.model_loader import ModelLoader
from forge.engine.engine_generation import unpack_output_with_kv

CKPT = "research/checkpoints/spark_x26_1p7b"
CFG = "spark_x26_1p7b"
DRAFT_K = 8          # tokens drafted per step
NGRAM_W = 3          # context window for n-gram match
POOL_MAX = 4         # candidate continuations kept per step (try in order)


def find_draft(context: list[int], w: int, k: int) -> list[int] | None:
    """Most recent non-overlapping (w)-gram match -> its continuation."""
    ctx = context[-w:]
    for j in range(len(context) - w - 1, -1, -1):
        if context[j:j + w] == ctx:
            draft = context[j + w:j + w + k]
            return draft if draft else None
    return None


def decode_std(model, ids, max_new):
    logits, past = unpack_output_with_kv(model(ids, use_cache=True))
    out = []
    for _ in range(max_new):
        tok = logits[:, -1, :].argmax(-1, keepdim=True)
        out.append(tok.item())
        logits, past = unpack_output_with_kv(
            model(tok, past_key_values=past, use_cache=True))
    return out


def decode_lookahead(model, ids, max_new, w=NGRAM_W, k=DRAFT_K,
                     adaptive: bool = True):
    logits, past = unpack_output_with_kv(model(ids, use_cache=True))
    seq_len = ids.shape[1]
    out: list[int] = []
    hist = ids[0].tolist()
    cooldown = 0        # steps until drafting re-enables
    backoff = 8         # next cooldown on a bad draft (doubles on streaks)
    dk = 2              # dynamic draft size: grows with acceptance
    while len(out) < max_new:
        draft = (find_draft(hist, w, dk)
                 if (len(hist) > w * 2 and cooldown == 0) else None)
        if not draft:
            cooldown = max(0, cooldown - 1) if adaptive else 0
            tok = logits[:, -1, :].argmax(-1, keepdim=True)
            out.append(tok.item())
            hist.append(tok.item())
            logits, past = unpack_output_with_kv(
                model(tok, past_key_values=past, use_cache=True))
            seq_len += 1
            continue
        # batch-verify: one forward over the draft, logits at each position
        inp = torch.tensor([draft], device=ids.device)
        d_logits, d_past = unpack_output_with_kv(
            model(inp, past_key_values=past, use_cache=True))
        # logits[:,-1] of `past` step predicts draft[0]; d_logits[t] predicts
        # draft[t+1]
        chain = [logits[:, -1, :].argmax(-1).item()]
        chain += d_logits[0, :-1, :].argmax(-1).tolist()
        # accept longest matching prefix
        acc = 0
        for want, got in zip(draft, chain):
            if got != want:
                break
            acc += 1
        # Adaptive backoff: poor drafts (<2 acc) double the cooldown;
        # good drafts reset it.
        if not adaptive or acc >= 2:
            cooldown, backoff = 0, 8
            dk = min(acc + 1, k)          # grow toward k on good drafts
        else:
            cooldown, backoff = backoff, min(backoff * 2, 64)
            dk = 2                        # shrink on rejection
        # Emit ONLY accepted tokens; `logits` is set to the batch position
        # predicting the next token — the next loop's argmax emits the
        # corrected token for free. Every draft costs <=1 forward.
        for t in draft[:acc]:
            out.append(t)
            hist.append(t)
        if acc == len(draft):
            logits = d_logits[:, -1:, :]          # full accept
        elif acc > 0:
            logits = d_logits[:, acc - 1:acc, :]  # predicts next pos
        # acc == 0: `logits` already predicts position seq_len
        def trunc(p, keep):
            if p is None:
                return None
            k_, v_ = p
            return k_[:, :, :keep], v_[:, :, :keep]
        past = [trunc(p, seq_len + acc) for p in d_past]
        seq_len += acc
    return out[:max_new]


def main():
    from forge.engine.forge_engine import ForgeEngine
    e = ForgeEngine.from_checkpoint(
        checkpoint=CKPT, config_name=CFG, auto_activate=False)
    model = e.model
    prompts = [
        "def reverse_words(s):",
        "def quicksort(a):",
        "def binary_search(a, x):",
    ]
    for p in prompts:
        ids = e.tokenizer.encode(p, return_tensors="pt").to("cuda")
        t0 = time.time()
        ref = decode_std(model, ids, 96)
        t_std = time.time() - t0
        t0 = time.time()
        la = decode_lookahead(model, ids, 96)
        t_la = time.time() - t0
        match = ref == la
        print(f"{p!r}: std {t_std*1000/96:.1f}ms/tok  la {t_la*1000/96:.1f}ms/tok"
              f"  speedup {t_std/t_la:.2f}x  match {match}")
        if not match:
            for i, (a, b) in enumerate(zip(ref, la)):
                if a != b:
                    print(f"  first diff @{i}: std={a} la={b}")
                    break


if __name__ == "__main__":
    main()

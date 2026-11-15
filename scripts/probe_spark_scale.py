"""Spark supervision scale-confirm — winners at N_TRAIN=512 / 60 steps.

Reuses probe_spark_supervision helpers; overrides its module constants so
the same production loss path (S.compute_loss chunked-CE) runs at:

  N_TRAIN=512, N_EVAL=96 rows of data/sft/spark_sft_v9.jsonl
  seq=1024, bs=2, 60 steps, AdamW lr=5e-5, LoRA r32/a64
  targets = qkv_proj,out_proj,w_gate,w_up,w_down

Arms:
  ce        plain CE, shuffled dataset order (as collated)
  ord_hard  frontier ordering: base-model per-example mean CE, hardest first

heldout CE is evaluated every 10 steps; curve entries are steps
10/20/30/40/50/60 followed by the final heldout_ce.

One subprocess per arm (VRAM isolation):
  python scripts/probe_spark_scale.py ce
  python scripts/probe_spark_scale.py ord_hard
  python scripts/probe_spark_scale.py all      # driver, spawns both
"""
import json
import os
import random
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

import probe_spark_supervision as P
from forge.training.runners import sft_train as S

N_TRAIN, N_EVAL, STEPS, LR = 512, 96, 60, 5e-5
EVAL_EVERY = 10


def run_arm(arm):
    tr_ds, train_b, ev_ds, eval_b = P.build_data(P.SEQ, N_TRAIN, N_EVAL)
    res = {"arm": arm, "n_train_rows": N_TRAIN, "n_eval_rows": N_EVAL,
           "n_train": len(train_b) * P.BS, "n_eval": len(eval_b) * P.BS,
           "steps": STEPS, "lr": LR}

    student = P.build_model(P.STU_CKPT)
    W_head = student.head.weight

    if arm == "ord_hard":
        t0 = time.perf_counter()
        scores = P.score_examples(student, tr_ds, W_head)
        res["score_s"] = round(time.perf_counter() - t0, 1)
        order = sorted(range(len(tr_ds)), key=lambda i: -scores[i])
        train_b = P.reorder_batches(tr_ds, order)
    elif arm == "ce":
        order = list(range(len(tr_ds)))
        random.Random(0).shuffle(order)
        train_b = P.reorder_batches(tr_ds, order)
    else:
        raise ValueError(arm)

    opt = P.attach_lora(student, lr=LR, rank=32, alpha=64)

    def batch_loss(ids, labels, mask, batch=None):
        loss = S.compute_loss(student, ids, labels, mask,
                              entropy_alpha=0.0)
        return loss, False

    student.train()
    torch.cuda.reset_peak_memory_stats()
    eval_fn = lambda: P.eval_ce(student, eval_b, W_head)
    losses, times, skipped, curve = P.train_loop(
        opt, train_b, batch_loss, steps=STEPS, eval_fn=eval_fn,
        eval_every=EVAL_EVERY)
    res.update(step_ms=round(1000 * sum(times) / max(len(times), 1), 1),
               loss0=round(losses[0], 4) if losses else None,
               lossN=round(losses[-1], 4) if losses else None,
               heldout_curve=curve,
               heldout_ce=round(eval_fn(), 4),
               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               skipped=skipped)
    print("RESULT " + json.dumps(res), flush=True)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arm == "all":
        import subprocess
        for a in ("ce", "ord_hard"):
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            p = subprocess.run([sys.executable, __file__, a],
                               env=env, capture_output=True,
                               text=True, timeout=3600)
            hit = [l for l in p.stdout.splitlines()
                   if l.startswith("RESULT ")]
            print(hit[-1] if hit else
                  f"RESULT {{\"arm\":\"{a}\",\"error\":\""
                  f"{(p.stderr or p.stdout)[-300:].replace(chr(34), chr(39))}\"}}",
                  flush=True)
        return
    run_arm(arm)


if __name__ == "__main__":
    main()

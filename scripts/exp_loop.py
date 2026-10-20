"""EXP-1 looped transformer (recurrent depth) on Qwen2.5-0.5B.

A) fixed-point dynamics of iterating the middle blocks
B) NLL vs iteration count (untrained looping)
C) learned LoopGates (zero-init, ~8 params) — does ANY depth reuse help?
D) decode-time loop cost
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from forge.exp.harness import (QWEN_ID, load_engine, log_result,
                               wikitext_docs)
from forge.exp.loopdepth import (looped_prefill, looped_decode_step,
                                 train_loop_gates)

MAX_LEN = 256
LOOP_LO, LOOP_HI = 8, 16


def docs_ids(eng, texts):
    out = []
    for t in texts:
        ids = eng.tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=MAX_LEN).input_ids.long().to("cuda")
        if ids.shape[1] >= 32:
            out.append(ids)
    return out


def nll_via_loop(model, ids_list, iters, gates=None):
    tot, n = 0.0, 0
    for ids in ids_list:
        logits, _, _ = looped_prefill(model, ids, LOOP_LO, LOOP_HI,
                                      max_iters=iters, gates=gates)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            ids[:, 1:].reshape(-1))
        tot += loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return tot / max(n, 1)


def main():
    eng = load_engine(QWEN_ID, warmup=False)
    model = eng.model
    test_ids = docs_ids(eng, wikitext_docs(4, split="test"))
    train_ids = docs_ids(eng, wikitext_docs(3, split="train"))

    # A) fixed-point dynamics
    for iters, tol in ((8, 0.0), (8, 1e-3)):
        logits, _, st = looped_prefill(model, test_ids[0], LOOP_LO,
                                     LOOP_HI, max_iters=iters, tol=tol)
        log_result("loop", phase="dynamics", iters=iters, tol=tol,
                   iters_run=st["iters_run"],
                   deltas=[round(d, 4) for d in st["deltas"]])

    # B) NLL vs iterations (1 = standard transformer)
    for iters in (1, 2, 4, 8):
        nll = nll_via_loop(model, test_ids, iters)
        log_result("loop", phase="nll_vs_iters", iters=iters,
                   nll=round(nll, 4))

    # C) learned gates: lossless at init, trained on wikitext train
    t0 = time.perf_counter()
    gates, losses = train_loop_gates(model, train_ids, LOOP_LO, LOOP_HI,
                                   max_iters=2, steps=200, lr=3e-2)
    train_s = time.perf_counter() - t0
    g = gates.g.detach()
    log_result("loop", phase="gate_train", train_s=round(train_s, 1),
               loss_first=round(losses[0], 4),
               loss_last=round(losses[-1], 4),
               gates=[round(float(v), 4) for v in g])
    nll_g = nll_via_loop(model, test_ids, 2, gates=g)
    nll_2 = nll_via_loop(model, test_ids, 2)
    log_result("loop", phase="gate_eval", nll_gated=round(nll_g, 4),
               nll_ungated=round(nll_2, 4))

    # D) decode-time loop latency: one step with iters vs plain decode
    ids = test_ids[0][:, :64]
    logits, pasts, _ = looped_prefill(model, ids, LOOP_LO, LOOP_HI,
                                      max_iters=1)
    tok = logits[:, -1].argmax(-1, keepdim=True)
    for iters in (1, 3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(16):
            pasts_c = [(p[0].clone(), p[1].clone())
                       if p is not None else None for p in pasts]
            looped_decode_step(model, tok, pasts_c, LOOP_LO, LOOP_HI,
                               max_iters=iters)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / 16
        log_result("loop", phase="decode_cost", iters=iters,
                   ms_per_tok=round(ms, 2))
    print("[loop] done")


if __name__ == "__main__":
    main()

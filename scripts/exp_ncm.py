"""EXP-1 NCM side-path: closed-form continuous cells on frozen Qwen blocks.

A) lossless attach check (gate=0)
B) train NCM params only (4 last blocks) on wikitext, eval dNLL
C) decay-gate statistics (learned timescales)
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from forge.exp.harness import (QWEN_ID, load_engine, log_result,
                               wikitext_docs)
from forge.exp.ncm import attach_ncm, train_ncm

MAX_LEN = 256
NCM_LAYERS = [20, 21, 22, 23]


def docs_ids(eng, texts):
    out = []
    for t in texts:
        ids = eng.tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=MAX_LEN).input_ids.long().to("cuda")
        if ids.shape[1] >= 32:
            out.append(ids)
    return out


@torch.no_grad()
def model_nll(model, ids_list):
    tot, n = 0.0, 0
    for ids in ids_list:
        loss = model(ids[:, :-1], targets=ids[:, 1:])[1]
        tot += loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return tot / max(n, 1)


def main():
    eng = load_engine(QWEN_ID, warmup=False)
    model = eng.model
    test_ids = docs_ids(eng, wikitext_docs(4, split="test"))
    train_ids = docs_ids(eng, wikitext_docs(4, split="train"))

    nll_base = model_nll(model, test_ids)
    log_result("ncm", phase="baseline", nll=round(nll_base, 4))

    attach_ncm(model, NCM_LAYERS)
    nll0 = model_nll(model, test_ids)
    log_result("ncm", phase="lossless_check", nll=round(nll0, 4),
               delta=round(nll0 - nll_base, 6))

    t0 = time.perf_counter()
    losses = train_ncm(model, train_ids, NCM_LAYERS, steps=150, lr=3e-3)
    log_result("ncm", phase="train", s=round(time.perf_counter() - t0, 1),
               loss_first=round(losses[0], 4),
               loss_mid=round(losses[len(losses) // 2], 4),
               loss_last=round(losses[-1], 4))

    nll_ncm = model_nll(model, test_ids)
    log_result("ncm", phase="eval", nll=round(nll_ncm, 4),
               delta_vs_base=round(nll_ncm - nll_base, 4))

    taus = {i: model.blocks[i]._ncm.log_tau.exp().mean().item()
            for i in NCM_LAYERS}
    log_result("ncm", phase="timescales",
               **{f"L{i}": round(v, 3) for i, v in taus.items()})
    print("[ncm] done")


if __name__ == "__main__":
    main()

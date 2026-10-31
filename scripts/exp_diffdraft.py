"""EXP-1 hybrid diffusion+AR decode (E-diffusion).

Trains a parallel diffusion draft head on frozen Qwen, then measures
greedy spec-verify acceptance and effective decode speed vs AR baseline.
Fidelity check: hybrid output ids must equal plain greedy ids.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from forge.exp.diffdraft import (
    DiffusionDraftHead,
    MedusaDraftHead,
    fixup_step,
    spec_verify,
    train_draft_head,
    train_medusa_head,
)
from forge.exp.harness import QWEN_ID, load_engine, log_result, wikitext_docs

MAX_LEN = 256
K = 4
PROMPTS = ["The quick brown fox jumps over",
           "In the year 1950, the city of",
           "Machine learning models are"]


def docs_ids(eng, texts):
    out = []
    for t in texts:
        ids = eng.tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=MAX_LEN).input_ids.long().to("cuda")
        if ids.shape[1] >= 48:
            out.append(ids)
    return out


@torch.no_grad()
def greedy_ids(model, ids, n):
    out = model(ids, use_cache=True)
    logits, pasts = out[0], out[2]
    gen = []
    for _ in range(n):
        tok = logits[:, -1].argmax(-1, keepdim=True)
        gen.append(int(tok))
        out = model(tok, past_key_values=pasts, use_cache=True)
        logits, pasts = out[0], out[2]
    return gen


@torch.no_grad()
def hybrid_ids(model, head, ids, n):
    out = model(ids, use_cache=True, return_hidden=True)
    logits, pasts, hidden = out[0], out[2], out[3]
    gen, fwds, accepts = [], 0, 0
    while len(gen) < n:
        draft = head.sample(hidden[:, -1].reshape(1, -1),
                            model.embed.weight,
                            head_w=model.head.weight)
        res = spec_verify(model, pasts, logits[:, -1:], draft)
        pasts = res["kv"]
        fwds += 1
        accepts += res["n_accepted"]
        if res["all_accepted"]:
            last_logits, hidden = res["logits_next"], res["hidden_next"]
        else:
            last_logits, pasts, hidden = fixup_step(
                model, pasts, res["committed"][-1])
            fwds += 1
        gen += res["committed"]
        logits = last_logits
    return gen[:n], fwds, accepts


def main():
    eng = load_engine(QWEN_ID, warmup=False)
    model = eng.model
    train_ids = docs_ids(eng, wikitext_docs(6, split="train"))
    test_ids = docs_ids(eng, wikitext_docs(4, split="test"))

    head = DiffusionDraftHead(model.config.d_model, k=K, steps=4).cuda()
    t0 = time.perf_counter()
    losses, accs = train_draft_head(model, head, train_ids, steps=300,
                                    lr=1e-3)
    log_result("diffdraft", phase="train", s=round(
        time.perf_counter() - t0, 1), loss_first=round(losses[0], 3),
        loss_last=round(losses[-1], 3), acc_first=round(accs[0], 3),
        acc_last=round(accs[-1], 3))
    torch.save(head.state_dict(),
               os.path.join(ROOT, ".devin", "exp_diffdraft_head.pt"))

    for tag, ids in zip(("A", "B"), test_ids[:2]):
        prompt = ids[:, :32]
        g = greedy_ids(model, prompt, 48)
        h, fwds, acc = hybrid_ids(model, head, prompt, 48)
        same = g == h
        log_result("diffdraft", phase="fidelity", tag=tag, match=same,
                   fwds=fwds, accepted=acc,
                   tok_per_fwd=round(48 / fwds, 2),
                   accept_rate=round(acc / max(fwds, 1), 3))

    # medusa variant — same verify machinery, direct argmax drafts
    mhead = MedusaDraftHead(model.config.d_model, k=K).cuda()
    t0 = time.perf_counter()
    mloss, macc = train_medusa_head(model, mhead, train_ids, steps=300,
                                  lr=1e-3)
    log_result("diffdraft", phase="medusa_train",
               s=round(time.perf_counter() - t0, 1),
               loss_first=round(mloss[0], 3),
               loss_last=round(mloss[-1], 3),
               acc_first=round(macc[0], 3), acc_last=round(macc[-1], 3))
    torch.save(mhead.state_dict(),
               os.path.join(ROOT, ".devin", "exp_diffdraft_medusa.pt"))
    for tag, ids in zip(("A", "B"), test_ids[:2]):
        prompt = ids[:, :32]
        h, fwds, acc = hybrid_ids(model, mhead, prompt, 48)
        log_result("diffdraft", phase="medusa_fidelity", tag=tag,
                   match=(greedy_ids(model, prompt, 48) == h),
                   fwds=fwds, accepted=acc,
                   tok_per_fwd=round(48 / fwds, 2),
                   accept_rate=round(acc / max(fwds, 1), 3))

    # speed: wall clock for 96 tokens both ways on a prompt
    prompt = test_ids[0][:, :32]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    greedy_ids(model, prompt, 96)
    torch.cuda.synchronize(); t_g = time.perf_counter() - t0
    t0 = time.perf_counter()
    hybrid_ids(model, head, prompt, 96)
    torch.cuda.synchronize(); t_h = time.perf_counter() - t0
    t0 = time.perf_counter()
    hybrid_ids(model, mhead, prompt, 96)
    torch.cuda.synchronize(); t_m = time.perf_counter() - t0
    log_result("diffdraft", phase="speed",
               greedy_tok_s=round(96 / t_g, 1),
               hybrid_tok_s=round(96 / t_h, 1),
               medusa_tok_s=round(96 / t_m, 1))
    print("[diffdraft] done")


if __name__ == "__main__":
    main()

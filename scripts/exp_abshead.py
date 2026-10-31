"""EXP-1 abshead: latent-thought head on frozen Qwen2.5-0.5B.

A) baseline NLL + lossless-attach check (gate=0)
B) train LatentThoughtHead h_t -> h_{t+4} (frozen model), report
   latent-prediction cos-sim / MSE vs the persistence baseline
C) train the gated residual hint on next-token CE, report honest dNLL
D) decode overhead: ms/tok with vs without the hint recompute
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from forge.exp.abshead import LatentThoughtHead, decode_ms_per_tok, eval_latent, hint_nll, train_hint, train_latent
from forge.exp.harness import QWEN_ID, load_engine, log_result, wikitext_docs

MAX_LEN = 256
K = 4


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
    train_ids = docs_ids(eng, wikitext_docs(6, split="train"))

    # A) baseline + lossless attach
    nll_base = model_nll(model, test_ids)
    log_result("abshead", phase="baseline", nll=round(nll_base, 4))

    head = LatentThoughtHead(model.config.d_model, hidden=1024,
                             k=K).cuda()
    log_result("abshead", phase="head", params=head.n_params(), k=K)

    nll0 = hint_nll(model, head, test_ids)
    log_result("abshead", phase="lossless_check", nll=round(nll0, 4),
               delta=round(nll0 - nll_base, 6))

    # B) latent prediction: h_t -> h_{t+k}
    t0 = time.perf_counter()
    tr = train_latent(model, head, train_ids, steps=300, lr=1e-3)
    losses, coss = tr["losses"], tr["cos"]
    log_result("abshead", phase="train_latent",
               s=round(time.perf_counter() - t0, 1),
               loss_first=round(losses[0], 4),
               loss_mid=round(losses[len(losses) // 2], 4),
               loss_last=round(losses[-1], 4),
               cos_first=round(coss[0], 4),
               cos_last=round(coss[-1], 4))
    torch.save(head.state_dict(),
               os.path.join(ROOT, ".devin", "exp_abshead_head.pt"))

    ev = eval_latent(model, head, test_ids)
    log_result("abshead", phase="latent_eval",
               cos_head=round(ev["cos_head"], 4),
               cos_persist=round(ev["cos_persist"], 4),
               mse_head=round(ev["mse_head"], 4),
               mse_persist=round(ev["mse_persist"], 4),
               n_positions=ev["n_positions"])

    # C) gated hint -> honest dNLL
    # C1) gate only (head frozen at latent weights): does the latent
    #     itself help readout?  Scalar gate, higher lr.
    t0 = time.perf_counter()
    gloss = train_hint(model, head, train_ids, steps=150, lr=1e-2,
                       latent_w=0.0, gate_only=True)
    log_result("abshead", phase="train_gate_only",
               s=round(time.perf_counter() - t0, 1),
               loss_first=round(gloss[0], 4),
               loss_last=round(gloss[-1], 4),
               gate=round(float(head.gate.detach()), 5))
    nll_g = hint_nll(model, head, test_ids)
    log_result("abshead", phase="hint_eval_gate_only",
               nll=round(nll_g, 4), dnll=round(nll_g - nll_base, 4))

    # C2) joint gate+head with a latent anchor, gate reset to 0.
    head.gate.data.zero_()
    t0 = time.perf_counter()
    hloss = train_hint(model, head, train_ids, steps=150, lr=1e-3,
                       latent_w=0.1)
    log_result("abshead", phase="train_hint",
               s=round(time.perf_counter() - t0, 1),
               loss_first=round(hloss[0], 4),
               loss_last=round(hloss[-1], 4),
               gate=round(float(head.gate.detach()), 5))

    nll_hint = hint_nll(model, head, test_ids)
    log_result("abshead", phase="hint_eval", nll=round(nll_hint, 4),
               dnll=round(nll_hint - nll_base, 4))

    # latent quality after hint fine-tune (did it stay a latent?)
    ev2 = eval_latent(model, head, test_ids)
    log_result("abshead", phase="latent_eval_post_hint",
               cos_head=round(ev2["cos_head"], 4),
               cos_persist=round(ev2["cos_persist"], 4))

    # D) decode overhead
    prompt = test_ids[0][:, :32]
    ms_base = decode_ms_per_tok(model, prompt, n_tokens=64, head=None)
    ms_hint = decode_ms_per_tok(model, prompt, n_tokens=64, head=head)
    log_result("abshead", phase="overhead",
               ms_per_tok_base=round(ms_base, 2),
               ms_per_tok_hint=round(ms_hint, 2),
               overhead_ms=round(ms_hint - ms_base, 2))
    print("[abshead] done")


if __name__ == "__main__":
    main()

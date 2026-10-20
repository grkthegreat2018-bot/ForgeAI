"""EXP-prevtok driver: prev-token head + self-correcting decode on Qwen2.5-0.5B.

Phase 1: train PrevTokenHead (tied-embedding readout: d x d adapter +
scale + vocab bias over embed.weight, AdamW) on frozen-model hidden
states over wikitext-2 train docs (<=256 tok).
Phase 2: held-out backward NLL / top-1.
Phase 3: 8 held-out prompts x 64 tokens — baseline greedy vs threshold
self-correcting decode (tau sweep) vs always-on rerank (alpha=1), with
and without single-token rollback.  All numbers logged via log_result.

Usage: venv\\Scripts\\python.exe scripts\\exp_prevtok.py [--steps N]
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from forge.exp.harness import (QWEN_ID, load_engine, wikitext_docs,
                               bench_decode, log_result)
from forge.exp.prevtok import (PrevTokenHead, SelfCorrectingDecoder,
                              train_prev_head, eval_prev_head, save_head,
                              seq_prev_nll)

HEAD_PATH = os.path.join(ROOT, ".devin", "exp_prevtok_head.pt")

N_TRAIN_DOCS = 48          # tied head can't memorize; more docs help
MAX_LEN = 256
STEPS = 600
LR = 1e-3                  # small adapter converges slowly at 3e-4
N_EVAL_DOCS = 4
N_PROMPTS = 8
N_GEN = 64
PROMPT_TOK = 12
# Fixed probability thresholds — the held-out p_prev distribution is
# extremely right-skewed (median ~0), so quantile-based thresholds
# collapse; absolute taus give a usable 10-70% flag-rate sweep.
THRESHOLDS = [0.01, 0.05, 0.15]
N_CANDS = 4
ALPHA = 1.0

if "--steps" in sys.argv:
    STEPS = int(sys.argv[sys.argv.index("--steps") + 1])


def docs_to_ids(eng, texts, max_len, device, min_len=16):
    out = []
    for t in texts:
        ids = eng.tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to(device)
        if ids.shape[1] >= min_len:
            out.append(ids)
    return out


def run_suite(dec, prompts, n_gen, mode):
    """Run decode over all prompts; return meaned metrics.

    Also scores each generated sequence's backward NLL (post-hoc, one
    prefill) so we can report joint consistency, not just forward NLL.
    """
    agg = {"nll": 0.0, "tok_s": 0.0, "checks": 0, "corrections": 0,
           "resamples": 0, "rollbacks": 0, "forwards": 0, "n_tokens": 0,
           "acc_plp": 0.0, "prev_nll_gen": 0.0}
    texts = []
    for p in prompts:
        r = dec.generate(p, n_gen, mode=mode) if mode else \
            dec.generate_baseline(p, n_gen)
        agg["nll"] += r["nll"]
        agg["tok_s"] += r["tok_s"]
        for k in ("checks", "corrections", "resamples", "rollbacks",
                  "forwards", "n_tokens"):
            agg[k] += r.get(k, 0)
        agg["acc_plp"] += r.get("mean_accepted_prev_logp", 0.0)
        full = torch.cat(
            [p, torch.tensor([r["ids"]], device=p.device,
                             dtype=p.dtype)], dim=1)
        agg["prev_nll_gen"] += seq_prev_nll(dec.model, dec.head, full)
        texts.append(r["ids"])
    n = max(len(prompts), 1)
    mean_nll = agg["nll"] / n
    mean_pnll = agg["prev_nll_gen"] / n
    out = {"nll": round(mean_nll, 4),
           "prev_nll_gen": round(mean_pnll, 4),
           "joint_nll": round(mean_nll + mean_pnll, 4),
           "tok_s": round(agg["tok_s"] / n, 1),
           "forwards_per_tok": round(
               agg["forwards"] / max(agg["n_tokens"], 1), 3),
           "gen_ids": texts}
    if mode:
        out.update({
            "correction_rate": round(
                agg["corrections"] / max(agg["checks"], 1), 4),
            "retry_rate": round(
                agg["resamples"] / max(agg["n_tokens"], 1), 4),
            "rollbacks": agg["rollbacks"],
            "mean_acc_prev_logp": round(agg["acc_plp"] / n, 3)})
    return out


def main():
    t_start = time.perf_counter()
    eng = load_engine(QWEN_ID)
    model, dev = eng.model, eng.device
    model.eval()
    print(f"[prevtok] engine load={eng._exp_load_s:.1f}s "
          f"peak={eng._exp_peak_vram:.2f}GB")
    log_result("prevtok", phase="load", load_s=round(eng._exp_load_s, 1),
               vram_gb=round(eng._exp_peak_vram, 3))

    # ── Phase 1: train the backward head ──────────────────────────────
    train_docs = wikitext_docs(N_TRAIN_DOCS, min_len=200, split="train")
    train_ids = docs_to_ids(eng, train_docs, MAX_LEN, dev)
    print(f"[prevtok] {len(train_ids)} train docs, "
          f"lens={[i.shape[1] for i in train_ids]}")
    head = PrevTokenHead(model.config.d_model, model.config.vocab_size,
                         out_weight=model.embed.weight,
                         tap_layers=(2, 9)).to(dev)
    torch.cuda.reset_peak_memory_stats()
    tr = train_prev_head(model, head, train_ids, steps=STEPS, lr=LR,
                         optimizer="adamw", log_every=50)
    vram_train = torch.cuda.max_memory_allocated() / 1e9
    print(f"[prevtok] train done: {tr['steps']} steps in {tr['train_s']}s "
          f"loss={tr['final_loss']} acc={tr['final_acc']}")
    for h in tr["history"]:
        log_result("prevtok", phase="train", **h)
    log_result("prevtok", phase="train_done", steps=tr["steps"],
               train_s=tr["train_s"], final_loss=tr["final_loss"],
               final_acc=tr["final_acc"], vram_gb=round(vram_train, 3))

    # ── Phase 2: held-out eval ─────────────────────────────────────────
    eval_docs = wikitext_docs(N_EVAL_DOCS + N_PROMPTS, min_len=200,
                              split="test")
    eval_ids = docs_to_ids(eng, eval_docs[:N_EVAL_DOCS], MAX_LEN, dev)
    ev = eval_prev_head(model, head, eval_ids)
    print(f"[prevtok] held-out prev_nll={ev['prev_nll']:.3f} "
          f"top1={ev['prev_top1']:.3f} pprev={ev['mean_prev_prob']:.3f} "
          f"q05={ev['pprev_q05']:.4f} q15={ev['pprev_q15']:.4f} "
          f"q30={ev['pprev_q30']:.4f}")
    log_result("prevtok", phase="eval", **{k: round(v, 4) for k, v in
                                         ev.items()})
    thresholds = THRESHOLDS
    save_head(head, HEAD_PATH, extra={"steps": tr["steps"], "lr": LR,
                                      "prev_nll": ev["prev_nll"],
                                      "prev_top1": ev["prev_top1"]})
    print(f"[prevtok] head saved -> {HEAD_PATH}")

    # ── Phase 3: decode comparison ─────────────────────────────────────
    src_ids = docs_to_ids(eng, eval_docs[N_EVAL_DOCS:], MAX_LEN, dev,
                          min_len=PROMPT_TOK + 4)
    prompts = [i[:, :PROMPT_TOK] for i in src_ids[:N_PROMPTS]]
    print(f"[prevtok] {len(prompts)} prompts x {N_GEN} tokens")

    dec = SelfCorrectingDecoder(model, head, threshold=0.0,
                                n_cands=N_CANDS, alpha=ALPHA)
    # Warmup: first timed suite would eat lazy-init/cuda-ctx noise.
    dec.generate_baseline(prompts[0], 8)
    dec.generate(prompts[0], 8, mode="threshold")
    base = run_suite(dec, prompts, N_GEN, mode=None)
    log_result("prevtok", phase="decode_baseline",
               nll=base["nll"], prev_nll_gen=base["prev_nll_gen"],
               joint_nll=base["joint_nll"], tok_s=base["tok_s"],
               forwards_per_tok=base["forwards_per_tok"])
    print(f"[prevtok] baseline nll={base['nll']} "
          f"prevNLL={base['prev_nll_gen']} tok/s={base['tok_s']}")

    best = None
    for tau in thresholds:
        dec.threshold = tau
        r = run_suite(dec, prompts, N_GEN, mode="threshold")
        dnll = round(r["nll"] - base["nll"], 4)
        djoint = round(r["joint_nll"] - base["joint_nll"], 4)
        overhead = round(base["tok_s"] / max(r["tok_s"], 1e-6), 3)
        log_result("prevtok", phase="decode_corrected", tau=tau,
                   nll=r["nll"], dnll=dnll,
                   prev_nll_gen=r["prev_nll_gen"],
                   joint_nll=r["joint_nll"], djoint=djoint,
                   correction_rate=r["correction_rate"],
                   retry_rate=r["retry_rate"],
                   forwards_per_tok=r["forwards_per_tok"],
                   tok_s=r["tok_s"], overhead_x=overhead,
                   mean_acc_prev_logp=r["mean_acc_prev_logp"])
        print(f"[prevtok] tau={tau} nll={r['nll']} dnll={dnll} "
              f"djoint={djoint} corr={r['correction_rate']} "
              f"tok/s={r['tok_s']} ({overhead}x)")
        if best is None or r["joint_nll"] < best[0]:
            best = (r["joint_nll"], tau)

    # Rerank variant (always-on, alpha=1)
    r = run_suite(dec, prompts, N_GEN, mode="rerank")
    dnll = round(r["nll"] - base["nll"], 4)
    djoint = round(r["joint_nll"] - base["joint_nll"], 4)
    log_result("prevtok", phase="decode_rerank", alpha=ALPHA,
               nll=r["nll"], dnll=dnll,
               prev_nll_gen=r["prev_nll_gen"],
               joint_nll=r["joint_nll"], djoint=djoint,
               correction_rate=r["correction_rate"],
               forwards_per_tok=r["forwards_per_tok"],
               tok_s=r["tok_s"],
               overhead_x=round(base["tok_s"] / max(r["tok_s"], 1e-6), 3))
    print(f"[prevtok] rerank nll={r['nll']} dnll={dnll} "
          f"djoint={djoint} corr={r['correction_rate']} "
          f"tok/s={r['tok_s']}")

    # Rollback variant at the best-joint-NLL threshold
    dec_rb = SelfCorrectingDecoder(model, head, threshold=best[1],
                                   n_cands=N_CANDS, alpha=ALPHA,
                                   rollback=True)
    r = run_suite(dec_rb, prompts, N_GEN, mode="threshold")
    log_result("prevtok", phase="decode_rollback", tau=best[1],
               nll=r["nll"], dnll=round(r["nll"] - base["nll"], 4),
               prev_nll_gen=r["prev_nll_gen"], joint_nll=r["joint_nll"],
               correction_rate=r["correction_rate"],
               retry_rate=r["retry_rate"], rollbacks=r["rollbacks"],
               tok_s=r["tok_s"],
               overhead_x=round(base["tok_s"] / max(r["tok_s"], 1e-6), 3))
    print(f"[prevtok] rollback(tau={best[1]}) nll={r['nll']} "
          f"rollbacks={r['rollbacks']}")

    # Reference raw decode speed + a sample of generated text
    ref = bench_decode(eng, "The quick brown fox", n_tokens=N_GEN, n_runs=2)
    log_result("prevtok", phase="bench_decode", **ref)
    sample = eng.tokenizer.decode(base["gen_ids"][0])
    log_result("prevtok", phase="sample_baseline", text=sample[:200])
    dec.threshold = best[1]
    r0 = dec.generate(prompts[0], N_GEN, mode="threshold")
    log_result("prevtok", phase="sample_corrected", tau=best[1],
               text=eng.tokenizer.decode(r0["ids"])[:200])

    peak = torch.cuda.max_memory_allocated() / 1e9
    log_result("prevtok", phase="done",
               total_s=round(time.perf_counter() - t_start, 1),
               peak_vram_gb=round(peak, 3))
    print(f"[prevtok] done in {time.perf_counter() - t_start:.0f}s "
          f"peakVRAM={peak:.2f}GB")


if __name__ == "__main__":
    main()

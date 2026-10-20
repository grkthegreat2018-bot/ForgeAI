"""E-dynscale driver: live head-gating + early-exit depth scaling on
Qwen2.5-0.5B via ForgeEngine.

Phases:
  0. baseline NLL + decode tok/s (engine path)
  1. HeadGateController: tau sweeps (rel_mean organic + frac calibrated)
     → frac_gated vs dNLL, plus decode-cost measurement with gates live.
  2. EarlyExitController: tau sweeps over confidence metrics
     → decode tok/s speedup, mean exit depth, greedy-token agreement,
       and teacher-forced first-exit dNLL.
  3. DiagTempScaler (bonus): per-channel gain recalibration (frozen
     substrate, ~200 AdamW steps on wikitext train) folded into
     out_proj/w_down rows → does it recover gated NLL?

Everything measured goes through log_result("dynscale", ...).

Usage: venv\\Scripts\\python.exe scripts\\exp_dynscale.py
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from forge.exp.harness import (QWEN_ID, load_engine, bench_decode,
                               log_result, wikitext_docs)
from forge.exp.dynscale import (HeadGateController, EarlyExitController,
                                DiagTempScaler)
from forge.model_loader import unpack_output_with_kv

PROMPT = ("The quick brown fox jumps over the lazy dog and runs through "
          "the forest while the wind blows softly over the hills.")
N_TOKENS = 64
EVAL_MAXLEN = 256
N_EVAL_DOCS = 5
N_EE_DOCS = 3            # per-position first-exit NLL is 24x head evals/doc
EE_MIN_DEPTH = 4
RECALIB_STEPS = 200
RECALIB_LR = 5e-3


def engine_nll(eng, texts, device="cuda", max_len=EVAL_MAXLEN):
    """NLL via the Forge-side model (targets = next-token labels)."""
    eng.model.eval()
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = eng.tokenizer(t, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
            if ids.shape[1] < 8:
                continue
            out = eng.model(ids[:, :-1].long(), targets=ids[:, 1:].long())
            nll_sum += float(out[1]) * (ids.shape[1] - 1)
            n += ids.shape[1] - 1
    return nll_sum / max(n, 1)


def doc_ids(eng, text, max_len=EVAL_MAXLEN):
    return eng.tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=max_len).input_ids.to(eng.device)


def ee_nll_mean(ee, eng, texts, tau, metric, check_every=1):
    """Mean first-exit NLL + exit depth over held-out docs."""
    nll_w = dep_w = n = 0.0
    for t in texts:
        ids = doc_ids(eng, t)
        if ids.shape[1] < 8:
            continue
        nll, dep = ee.exit_nll(ids, tau, metric=metric,
                               min_depth=EE_MIN_DEPTH, check_every=check_every)
        w = ids.shape[1] - 1
        nll_w += nll * w
        dep_w += dep * w
        n += w
    return (nll_w / max(n, 1), dep_w / max(n, 1))


def timed_ee_decode(ee, eng, ids, n_tokens, n_runs=2, **kw):
    """Time only the decode-token loop (prefill excluded), like
    harness.bench_decode. Best tok/s over n_runs (decode timing on this
    box has ~10-20% run-to-run noise; min-variance wins the report)."""
    best, best_res = -1.0, None
    for _ in range(n_runs):
        logits, past = eng._prefill(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = ee.decode(logits, past, ids.shape[1], n_tokens, **kw)
        torch.cuda.synchronize()
        tps = n_tokens / (time.perf_counter() - t0)
        if tps > best:
            best, best_res = tps, res
        del past, res
    return best_res, best


def model_greedy_tokens(eng, ids, n_tokens):
    """Reference greedy tokens via the stock model() decode path."""
    logits, past = eng._prefill(ids)
    toks = []
    with torch.inference_mode():
        for _ in range(n_tokens):
            tok = logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(tok[0, 0]))
            out = eng.model(tok, past_key_values=past, use_cache=True)
            logits, past = unpack_output_with_kv(out)
    return toks


def main():
    try:
        texts = wikitext_docs(N_EVAL_DOCS, split="test")
        train_texts = wikitext_docs(RECALIB_STEPS, split="train")
    except Exception as e:
        print(f"[dynscale] wikitext load failed ({e}); using canned text")
        texts = ["The quick brown fox jumps over the lazy dog. " * 24]
        train_texts = texts * 4

    eng = load_engine(QWEN_ID, warmup=False)
    m = eng.model
    print(f"[dynscale] loaded in {eng._exp_load_s:.1f}s, "
          f"peak {eng._exp_peak_vram:.2f}GB, "
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params")

    nll0 = engine_nll(eng, texts)
    b0 = bench_decode(eng, PROMPT, n_tokens=N_TOKENS, n_runs=3)
    log_result("dynscale", phase="baseline", nll=round(nll0, 4),
               tok_s=b0["decode_tok_s"], ms_tok=b0["decode_ms_per_tok"],
               vram=round(torch.cuda.max_memory_allocated() / 1e9, 3))

    # == Phase 1: HeadGateController ===================================
    print("\n== Phase 1: HeadGateController (EMA head gating) ==")
    ctl = HeadGateController(m, ema_alpha=0.9, tau_mode="rel_mean",
                             update_every=1)
    n_att = ctl.attach()
    log_result("dynscale", phase="headgate_attach", n_blocks=n_att)

    # warm the EMA once so gating is live during measured evals
    def warm():
        ctl.reset_ema()
        engine_nll(eng, texts[:1])          # first doc populates EMA

    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ctl.set_tau(tau, mode="rel_mean")
        warm()
        nll = engine_nll(eng, texts)
        st = ctl.stats()
        log_result("dynscale", phase="headgate", mode="rel_mean", tau=tau,
                   frac_gated=round(st["frac_gated"], 4),
                   nll=round(nll, 4), dnll=round(nll - nll0, 4),
                   ema_min=st.get("ema_min"), ema_max=st.get("ema_max"))

    # calibrated curve: frac mode → exact gated fraction per layer
    for frac in [0.15, 0.3, 0.5]:
        ctl.set_tau(frac, mode="frac")
        warm()
        nll = engine_nll(eng, texts)
        st = ctl.stats()
        log_result("dynscale", phase="headgate_frac", mode="frac", tau=frac,
                   frac_gated=round(st["frac_gated"], 4),
                   nll=round(nll, 4), dnll=round(nll - nll0, 4))

    # live decode cost of gating at a mid tau — per-step .tolist() syncs
    # vs amortized EMA refresh (update_every=8, saliency lags 8 steps)
    for ue in (1, 8):
        ctl.set_tau(0.5, mode="rel_mean")
        for g in ctl.gates.values():
            g.update_every = ue
        ctl.reset_ema()
        rg = bench_decode(eng, PROMPT, n_tokens=N_TOKENS, n_runs=3)
        st = ctl.stats()
        log_result("dynscale", phase="headgate_decode", tau=0.5,
                   update_every=ue,
                   tok_s=rg["decode_tok_s"], ms_tok=rg["decode_ms_per_tok"],
                   frac_gated=round(st["frac_gated"], 4),
                   slowdown_vs_base=round(b0["decode_tok_s"] /
                                          max(rg["decode_tok_s"], 1e-9), 3))
    ctl.detach()
    torch.cuda.empty_cache()

    # == Phase 2: EarlyExitController ==================================
    print("\n== Phase 2: EarlyExitController (live depth scaling) ==")
    ee = EarlyExitController(eng)
    ids = eng._tokenize(PROMPT)

    # warmup: first manual-loop call pays cold allocator/launch costs —
    # discard it so the baseline isn't depressed by one-time effects.
    timed_ee_decode(ee, eng, ids, 16, tau=None)

    # wiring sanity: manual full-depth decode must match model() greedy
    res_b, tps_b = timed_ee_decode(ee, eng, ids, N_TOKENS, tau=None)
    res_b2, tps_b2 = timed_ee_decode(ee, eng, ids, N_TOKENS, tau=None)
    tps_b = max(tps_b, tps_b2)
    ref_tokens = model_greedy_tokens(eng, ids, N_TOKENS)
    parity = sum(a == b for a, b in zip(res_b["tokens"], ref_tokens)) / N_TOKENS
    log_result("dynscale", phase="ee_baseline", tok_s=round(tps_b, 1),
               mean_depth=res_b["mean_depth"],
               token_parity=round(parity, 4),
               engine_tok_s=b0["decode_tok_s"])

    # teacher-forced baseline parity (exit_nll tau=None == engine_nll)
    nll_inf, dep_inf = ee_nll_mean(ee, eng, texts[:N_EE_DOCS], None, "prob")
    nll_ref, _ = ee_nll_mean(ee, eng, texts[:N_EE_DOCS], float("inf"), "prob")
    log_result("dynscale", phase="ee_nll_parity",
               nll_none=round(nll_inf, 4), nll_inf=round(nll_ref, 4),
               mean_depth=dep_inf)

    for metric, taus, ce in [("margin", [0.02, 0.05, 0.1, 0.2], 1),
                             ("prob", [0.5, 0.8, 0.95], 1),
                             ("prob", [0.8, 0.95], 2)]:
        for tau in taus:
            try:
                res, tps = timed_ee_decode(ee, eng, ids, N_TOKENS,
                                           tau=tau, metric=metric,
                                           min_depth=EE_MIN_DEPTH,
                                           check_every=ce)
                agree = sum(a == b for a, b in
                            zip(res["tokens"], ref_tokens)) / N_TOKENS
                nll_ee, dep_ee = ee_nll_mean(ee, eng, texts[:N_EE_DOCS],
                                             tau, metric, check_every=ce)
                log_result("dynscale", phase="earlyexit", metric=metric,
                           tau=tau, check_every=ce, tok_s=round(tps, 1),
                           speedup=round(tps / tps_b, 3),
                           mean_depth=round(res["mean_depth"], 2),
                           token_agree=round(agree, 4),
                           nll=round(nll_ee, 4),
                           dnll=round(nll_ee - nll_inf, 4),
                           tf_mean_depth=round(dep_ee, 2),
                           vram=round(torch.cuda.max_memory_allocated()
                                      / 1e9, 3))
            except Exception as e:
                log_result("dynscale", phase="earlyexit", metric=metric,
                           tau=tau, check_every=ce,
                           error=type(e).__name__ + ": " + str(e)[:120])
                traceback.print_exc(limit=1)
    torch.cuda.empty_cache()

    # == Phase 3: DiagTempScaler recalibration (bonus) =================
    print("\n== Phase 3: DiagTempScaler — recalibrate around gates ==")
    TAU_G = 0.5
    ctl.attach()
    ctl.set_tau(TAU_G, mode="rel_mean")
    warm()                                   # same live-EMA policy
    nll_gated = engine_nll(eng, texts)
    sc = DiagTempScaler(m)
    n_wraps = sc.attach()
    log_result("dynscale", phase="diagtemp_pre", tau_g=TAU_G,
               n_gains=n_wraps, nll_gated=round(nll_gated, 4),
               dnll_gated=round(nll_gated - nll0, 4))

    train_ids = [doc_ids(eng, t) for t in train_texts[:RECALIB_STEPS]
                 if doc_ids(eng, t).shape[1] >= 8] or [doc_ids(eng, texts[0])]

    def next_ids(step):
        return train_ids[step % len(train_ids)]

    torch.cuda.reset_peak_memory_stats()
    losses = sc.recalibrate(next_ids, steps=RECALIB_STEPS, lr=RECALIB_LR,
                            log_every=50)
    sc.fold()
    warm()                                   # re-warm EMA: same policy as pre
    nll_post = engine_nll(eng, texts)
    log_result("dynscale", phase="diagtemp_post", tau_g=TAU_G,
               steps=RECALIB_STEPS, lr=RECALIB_LR,
               nll_pre=round(nll_gated, 4), nll_post=round(nll_post, 4),
               recovered=round(nll_gated - nll_post, 4),
               dnll_vs_base=round(nll_post - nll0, 4),
               loss_first=round(losses[0], 4), loss_last=round(losses[-1], 4),
               peak_vram=round(torch.cuda.max_memory_allocated() / 1e9, 3))
    ctl.detach()
    torch.cuda.empty_cache()

    print("\n[dynscale] done — results in .devin/exp_results.jsonl")


if __name__ == "__main__":
    main()

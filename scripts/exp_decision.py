"""E-decision driver: train a per-step decision head on Qwen2.5-0.5B
hidden states (heuristic labels from a frozen decode trace), then run a
DecisionController decode loop (CONTINUE/THINK/STOP/VERIFY) and measure
action mix, dNLL vs greedy, and tok/s overhead.

Usage: venv\\Scripts\\python.exe scripts\\exp_decision.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from forge.exp.decision import (
    ACTIONS,
    N_ACTIONS,
    DecisionController,
    DecisionHead,
    collect_trace,
    eval_head,
    gen_nll,
    greedy_generate,
    relabel,
    train_head,
)
from forge.exp.harness import QWEN_ID, load_engine, log_result, wikitext_docs

HEAD_PATH = os.path.join(ROOT, ".devin", "exp_decision_head.pt")

N_TRAIN_DOCS = 36
N_HELDO_DOCS = 8
MAX_LEN = 384          # tokens per doc for the trace
HEAD_STEPS = 400
N_PROMPTS = 6
GEN_TOKENS = 64
VERIFY_DELTA = 0.05


def main():
    torch.manual_seed(0)
    eng = load_engine(QWEN_ID)
    print(f"[decision] engine loaded {eng._exp_load_s:.1f}s "
          f"peak {eng._exp_peak_vram:.2f}GB")
    d_model = eng.model.config.d_model

    docs = wikitext_docs(N_TRAIN_DOCS + N_HELDO_DOCS, split="train")
    train_docs, held_docs = docs[:N_TRAIN_DOCS], docs[N_TRAIN_DOCS:]
    print(f"[decision] {len(train_docs)} train docs, "
          f"{len(held_docs)} held-out docs")

    # ── 1. trace -> labels -> train head ────────────────────────────────
    tr = collect_trace(eng, train_docs, max_len=MAX_LEN,
                       verify_delta=VERIFY_DELTA)
    dist = torch.bincount(tr["labels"], minlength=N_ACTIONS).tolist()
    log_result("decision", phase="trace_train", n=tr["n"],
               think_thresh=round(tr["think_thresh"], 4),
               label_dist=dict(zip(ACTIONS, dist)))
    print(f"[decision] train trace n={tr['n']} labels={dist} "
          f"p25={tr['think_thresh']:.3f}")

    head = DecisionHead(d_model=d_model)
    tinfo = train_head(head, tr["hidden"], tr["labels"],
                       steps=HEAD_STEPS)
    log_result("decision", phase="head_train", steps=HEAD_STEPS, **tinfo)

    ho = collect_trace(eng, held_docs, max_len=MAX_LEN,
                       verify_delta=VERIFY_DELTA)
    ho["labels"] = relabel(ho, think_thresh=tr["think_thresh"],
                           verify_delta=VERIFY_DELTA)
    ev = eval_head(head, ho["hidden"], ho["labels"])
    log_result("decision", phase="head_eval", n=ho["n"], **ev)
    print(f"[decision] held-out acc={ev['acc']} per_class={ev['per_class']}")
    for a, row in zip(ACTIONS, ev["confusion"]):
        print(f"           true={a:8s} -> {row}")

    os.makedirs(os.path.dirname(HEAD_PATH), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "d_model": d_model,
                "think_thresh": tr["think_thresh"],
                "verify_delta": VERIFY_DELTA}, HEAD_PATH)
    print(f"[decision] head saved -> {HEAD_PATH}")

    # ── 2. controller decode vs greedy on held-out prompts ──────────────
    eval_docs = wikitext_docs(N_PROMPTS, split="test")
    prompts = []
    for d in eval_docs:
        ids = eng.tokenizer(d, return_tensors="pt",
                            add_special_tokens=False).input_ids[0, :16]
        prompts.append(ids.view(1, -1))

    ctrl = DecisionController(eng, head, verify_temp=0.7,
                              min_new_tokens=8, seed=0)

    agg_actions = torch.zeros(N_ACTIONS, dtype=torch.long)
    ctrl_toks = greedy_toks = 0
    ctrl_time = greedy_time = 0.0
    dnlls, shifts = [], []
    step_rows = []   # (action, entropy) for coherence analysis
    for i, p in enumerate(prompts):
        g = greedy_generate(eng, p, max_new_tokens=GEN_TOKENS)
        c = ctrl.generate(p, max_new_tokens=GEN_TOKENS)
        nll_g = gen_nll(eng, p, g.tokens)
        nll_c = gen_nll(eng, p, c.tokens)
        dnlls.append(nll_c - nll_g)
        shifts += c.think_logit_shift
        step_rows += list(zip(c.actions, c.entropies))
        agg_actions += torch.bincount(torch.tensor(c.actions or [0]),
                                      minlength=N_ACTIONS)
        ctrl_toks += len(c.tokens); ctrl_time += c.elapsed_s
        greedy_toks += len(g.tokens); greedy_time += g.elapsed_s
        print(f"[decision] p{i}: ctrl {len(c.tokens)}tok "
              f"{c.tok_s:.0f}tok/s hist={c.action_hist} "
              f"stopped={c.stopped_early} | greedy nll={nll_g:.3f} "
              f"ctrl nll={nll_c:.3f} dNLL={nll_c - nll_g:+.3f}")

    ctrl_tps = ctrl_toks / max(ctrl_time, 1e-9)
    greedy_tps = greedy_toks / max(greedy_time, 1e-9)
    overhead = (greedy_tps - ctrl_tps) / greedy_tps * 100.0
    log_result("decision", phase="decode",
               n_prompts=len(prompts), gen_tokens=GEN_TOKENS,
               action_dist=dict(zip(ACTIONS, agg_actions.tolist())),
               ctrl_tok_s=round(ctrl_tps, 1),
               greedy_tok_s=round(greedy_tps, 1),
               overhead_pct=round(overhead, 1),
               mean_dNLL=round(sum(dnlls) / len(dnlls), 4),
               mean_think_logit_shift=(
                   round(sum(shifts) / len(shifts), 4) if shifts else None),
               peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))

    # ── 3. coherence: does THINK fire on high-entropy steps? ────────────
    import statistics
    ents = [e for _, e in step_rows]
    med = statistics.median(ents) if ents else 0.0
    per_action_ent = {}
    think_hi = think_n = 0
    for a, e in step_rows:
        per_action_ent.setdefault(ACTIONS[a], []).append(e)
        if a == 1:
            think_n += 1
            think_hi += int(e > med)
    coh = {"median_entropy": round(med, 3),
           "mean_entropy_by_action": {
               k: round(sum(v) / len(v), 3) for k, v in per_action_ent.items()},
           "think_above_median_frac": (
               round(think_hi / think_n, 3) if think_n else None),
           "n_think": think_n}
    log_result("decision", phase="coherence", **coh)
    print(f"[decision] coherence: {coh}")
    print(f"[decision] peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print("[decision] done")


if __name__ == "__main__":
    main()

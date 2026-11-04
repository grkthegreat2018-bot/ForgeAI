"""exp_l1_real.py — step 2: exact-memory L1 on a REAL backbone.

Attaches SuffixMemory (EMA recency variant) to the int4-packed LFM3-1.7B-CC
engine and runs the FluxLM-v1-killer probes with real text, not synthetic
token pairs:

  1. recall:     doc facts -> prompt-with-answer-missing, CE+top1 of the
                 first answer token, gate on vs off
  2. distractor: +2048 unrelated tokens (AGENTS.md), re-probe
  3. overwrite:  same fact sentence, new value -> new must dominate, and
                 stale-value suppression is measured honestly
  4. unseen:     novel-suffix prompts -> lift must stay flat (no poison)
  5. cost:       ingest tps, per-step lift build time, peak VRAM

Run:  venv\\Scripts\\python.exe scripts\\exp_l1_real.py
"""
import math
import time

import torch
import torch.nn.functional as F

from forge.engine.forge_engine import ForgeEngine
from forge.model.ngram_mem import SuffixMemory

CKPT = "research/checkpoints/lfm3_1p7b_cc_fq"

FACTS = [
    ("The Meridian project's vault code is", " 7741."),
    ("The harbor relay uses uplink port", " 9917."),
    ("Dr. Okafor keeps the lab notebook in drawer", " kappa-3."),
    ("The Nighthawk migration starts on", " March 17th."),
    ("Our staging cluster's admin token is", " skt-44-breeze."),
    ("The backup cronjob runs at", " 03:40 every night."),
    ("Captain Reyes pilots the vessel named", " Ironwood."),
    ("The checksum seed for build nine is", " 0xA3F1."),
]
DOC = " ".join(p + a for p, a in FACTS)
OVERWRITES = [
    ("The Meridian project's vault code is", " 8823."),
    ("The harbor relay uses uplink port", " 4402."),
]
UNSEEN = [
    "The Cygnus relay's beacon code is",
    "Professor Almeida stores the archive key in",
]


def main():
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        CKPT, config_name="lfm3_1p7b", device="cuda",
        quantize=None, load_quantize=None, kv_cache="standard",
        use_compile=False, use_breakable_cuda_graph=False,
        use_block_fusion=False, use_chunked_prefix_cache=False,
        use_adaptive_spec=False, use_suffix_spec=False,
        use_faser=False, use_lookahead_gate=False, use_hotprefix=False,
        use_pod_attention=False, use_hybrid_prefill=False,
        use_fused_qk_norm_rope_cache=False, use_seq_split=False)
    tok = eng.tokenizer
    model = eng.model
    dev = next(model.parameters()).device
    print(f"loaded {CKPT} in {time.time()-t0:.1f}s  "
          f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    def ids_of(text):
        return torch.tensor(tok.encode(text), device=dev)

    def logits_last(ids):
        with torch.no_grad():
            out = model(ids.unsqueeze(0))
            return (out[0] if isinstance(out, tuple) else out)[0, -1].float()

    # size to the MODEL's output dim (128000) — tokenizer.vocab_size
    # (124893) under-sizes the lift vs the logit vector
    n_vocab = int(model.config.vocab_size)
    mem = SuffixMemory(n_vocab, orders=(4, 8, 16, 32), gate=1.0)

    def probe(prefix, answer, gate, p_ids=None):
        """CE + top1 of the first answer token at the missing-answer slot."""
        if p_ids is None:
            p_ids = ids_of(prefix)
        a_ids = tok.encode(answer)
        # 2-step roll: memory lifts the first answer token (usually a
        # shared space); after consuming it, the suffix key includes it
        # and the row fires on the DISTINGUISHING token — the real
        # fact-recall signal
        ids2 = torch.cat([p_ids, torch.tensor([a_ids[0]], device=dev)])
        base = logits_last(ids2)
        mem._tail = ids2[-mem.max_order:].tolist()
        lift = mem.lift_last(device=dev)
        z = base + gate * lift
        tgt = a_ids[1] if len(a_ids) > 1 else a_ids[0]
        ce = F.cross_entropy(z.unsqueeze(0),
                             torch.tensor([tgt], device=dev))
        return float(ce), int(z.argmax()) == tgt, tgt

    # ---- ingest -------------------------------------------------------------
    # sentence-level ingest: feed each fact separately so write-time suffix
    # boundaries align with how fresh queries tokenize (doc-as-one-string
    # merges BPE differently across fact boundaries -> zero key matches)
    fact_spans = []          # (prefix_ids, first_answer_tok)
    t0 = time.time()
    total_tok = 0
    for p, a in FACTS:
        f_ids = ids_of(p + a)
        p_ids = ids_of(p)
        a_ids = tok.encode(a)
        fact_spans.append((p_ids, a_ids[0]))
        mem.feed(f_ids.tolist())
        total_tok += len(f_ids)
    feed_ms = (time.time()-t0)*1e3
    d_ids = ids_of(DOC)
    print(f"doc={total_tok}tok feed={feed_ms:.0f}ms "
          f"({total_tok/max(feed_ms,1e-3)*1e3:.0f}tps)")

    rows = []
    def suite(tag):
        for i, (p, a) in enumerate(FACTS):
            ce0, ok0, atok = probe(p, a, 0.0)
            ce1, ok1, _ = probe(p, a, 1.0)
            rows.append((tag, i, ce0, ce1, ok0, ok1))
    suite("recall")

    # tokenization-boundary check: probe with the DOC-tokenized prefix too
    diag = []
    for i, (p_ids, atok) in enumerate(fact_spans):
        mem._tail = p_ids[-mem.max_order:].tolist()
        matched = any(mem._row_probs(o)[0] is not None for o in mem.orders)
        diag.append(matched)
    print(f"key-match on fresh-tokenized prefixes: {sum(diag)}/{len(diag)}")

    # ---- distractor ---------------------------------------------------------
    noise_ids = ids_of(open("AGENTS.md", encoding="utf-8").read()[:12000])
    t0 = time.time(); mem.feed(noise_ids.tolist()); dms = (time.time()-t0)*1e3
    suite("distract")
    print(f"distractor={len(noise_ids)}tok feed={dms:.0f}ms")

    # ---- overwrite ----------------------------------------------------------
    ow_rows = []
    for p, a in OVERWRITES:
        mem.feed(ids_of(p + a).tolist())
    for i, (p, a) in enumerate(OVERWRITES):
        stale_a = FACTS[i][1]
        ce_new, ok_new, _ = probe(p, a, 1.0)
        ce_stale, ok_stale, _ = probe(p, stale_a, 1.0)
        ow_rows.append((i, ce_new, ce_stale, ok_new, ok_stale))

    # ---- unseen-suffix safety ----------------------------------------------
    us = []
    for p in UNSEEN:
        p_ids = ids_of(p)
        mem._tail = p_ids[-mem.max_order:].tolist()
        lift = mem.lift_last(device=dev)
        base = logits_last(p_ids)
        z = base + lift
        us.append((float(lift.abs().max()),
                   int(z.argmax()) == int(base.argmax())))

    # ---- per-step cost --------------------------------------------------------
    p_ids = ids_of("The Meridian project's vault code is")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(32):
        logits_last(p_ids)
    torch.cuda.synchronize(); fwd_ms = (time.time()-t0)/32*1e3
    mem._tail = p_ids[-mem.max_order:].tolist()
    t0 = time.time()
    for _ in range(64):
        mem.lift_last(device=dev)
    lift_ms = (time.time()-t0)/64*1e3

    # ---- report --------------------------------------------------------------
    print(f"\n{'probe':<10} {'ce_off':>6} {'ce_on':>6} {'hit_off':>7} "
          f"{'hit_on':>6}")
    for tag in ("recall", "distract"):
        rs = [r for r in rows if r[0] == tag]
        print(f"{tag:<10} {sum(r[2] for r in rs)/len(rs):>6.2f} "
              f"{sum(r[3] for r in rs)/len(rs):>6.2f} "
              f"{sum(r[4] for r in rs)/len(rs):>7.2f} "
              f"{sum(r[5] for r in rs)/len(rs):>6.2f}")
    print("\noverwrite (same suffix, new value):")
    for i, cn, cs, on, os_ in ow_rows:
        print(f"  fact{i}: new_ce={cn:.2f} hit={on}  "
              f"stale_ce={cs:.2f} hit={os_}")
    print("\nunseen suffixes (lift_max, top1_preserved):")
    for m, keep in us:
        print(f"  {m:.4f}  {keep}")
    print(f"\ncost: fwd={fwd_ms:.1f}ms/step  lift={lift_ms:.3f}ms/step  "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB  "
          f"cells={mem.info()['cells']}")


if __name__ == "__main__":
    main()

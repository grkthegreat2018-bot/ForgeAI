"""EXP: FluxLM scaling on a real prose corpus (wikitext-103 head).

The pool run proved mixed-domain coverage needs lanes; this run
measures PURE prose scaling: ~15M wikitext BPE tokens vs the docs
baseline.  Two held slices - held_wt (same-domain generalization)
and held_docs (cross-domain transfer, same slice as exp_fluxscale).

No gradient training; ingest = memory.  Same order/channel config as
exp_fluxscale.run so numbers are directly comparable.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import torch

from forge.exp.harness import load_hf, log_result
from forge.model.flux import FluxConfig, FluxLM
from scripts.exp_fluxscale import pool_bytes, tf

V = 151936
WT = ROOT / ".tmp" / "wt103_head.txt"


def run_wt(name, segs, held_wt, held_docs, chunk=1 << 20,
           snap=None, **kw):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n = 0
    if snap and Path(snap).exists():
        # resume: skip re-ingest, eval the saved model
        m = FluxLM.load(snap)
        n = int(getattr(m, "_count", -1))
    else:
        cfg = {"vocab_size": V, "device": "cuda", "cuda_primary": True,
                   "sparse_taps": (), "class_orders": (), "word_orders": (),
                   "hybrid_taps": (), "use_knn": False,
                   "orders": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)}
        cfg.update(kw)
        m = FluxLM(FluxConfig(**cfg))
        for ids, dom in segs:
            for c0 in range(0, len(ids), chunk):
                seg = ids[c0:c0 + chunk]
                m.ingest(seg, tag=dom, domain=dom)
                n += len(seg)
            print(f"[fluxscale_wt] {name}: ingested {n} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if snap:
        try:
            m.snapshot(snap)
            print(f"[fluxscale_wt] snapshot -> {snap}", flush=True)
        except Exception as e:      # MemoryError at this scale
            print(f"[fluxscale_wt] snapshot failed: {e}", flush=True)
    t1 = time.perf_counter()
    # docs first: evals mutate m (clone=False), so run the
    # cross-domain slice before wt-held tokens enter the stream
    t_doc, nll_doc = tf(m, held_docs, clone=False)
    t_wt, nll_wt = tf(m, held_wt, clone=False)
    vram = torch.cuda.max_memory_allocated() / 1e9
    log_result("fluxscale_wt", variant=name, n_train=n,
               ingest_s=round(t1 - t0, 1),
               vram_gb=round(vram, 2),
               top1_wt=round(t_wt, 4), nll_wt=round(nll_wt, 4),
               top1_doc=round(t_doc, 4), nll_doc=round(nll_doc, 4))
    print(f"[fluxscale_wt] {name}: wt nll={nll_wt:.4f} top1={t_wt:.4f} "
          f"| doc nll={nll_doc:.4f} top1={t_doc:.4f} "
          f"| vram={vram:.2f}GB ingest={t1-t0:.0f}s n={n}",
          flush=True)
    return m


def main():
    _hf, tok = load_hf()
    wt_b = WT.read_bytes()
    tr_wt, held_wt_b = wt_b[:-8000], wt_b[-8000:]
    held_wt = tok(held_wt_b.decode("utf-8", "ignore"))["input_ids"]
    _tr_docs, _pool, held_doc_b = pool_bytes()
    held_doc = tok(held_doc_b.decode("utf-8", "ignore"))["input_ids"]
    tr_docs = _tr_docs
    print(f"[fluxscale_wt] held wt={len(held_wt)} doc={len(held_doc)}",
          flush=True)
    del _hf
    torch.cuda.empty_cache()

    tr_wt_ids = tok(tr_wt.decode("utf-8", "ignore"))["input_ids"]
    tr_doc_ids = tok(tr_docs.decode("utf-8", "ignore"))["input_ids"]
    print(f"[fluxscale_wt] train wt={len(tr_wt_ids)} "
          f"doc={len(tr_doc_ids)}", flush=True)

    # prose lanes share statistics (both English prose) - code lane
    # only exists if a mixed pool is added later
    segs = [(tr_wt_ids, "prose"), (tr_doc_ids, "prose")]
    run_wt("wt15m+docs", segs, held_wt, held_doc,
           gpu_table_slots=2 << 20,
           domains=("shared", "prose", "code"),
           domain_route_every=16,
           bloom_admit=2,
           bank_enable=True,
           bank_rows_cap=4 << 20,
           bank_dir=str(ROOT / ".tmp" / "fluxbanks"),
           teacher_channel=True,
           teacher_window=4096,
           teacher_w0=0.6,
           teacher_bulk_chunk=2048,
           bulk_channel_rewards=True,
           snap=str(ROOT / ".tmp" / "wt15m.flux"))
    print("[fluxscale_wt] done")


if __name__ == "__main__":
    main()

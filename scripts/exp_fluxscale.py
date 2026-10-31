"""EXP: FluxLM scaling curve on Qwen-BPE tokens.

At fixed corpus size every auxiliary channel dilutes - the binding
constraint is coverage.  Measure NLL vs corpus size on the SAME
held-out docs slice: docs-only (190K) -> full local text pool
(repo .py/.md/.txt, ~12MB).  No gradient training; ingest = memory.

VRAM: gpu_table_slots raised for the big pool (cells grow
sublinearly with corpus but high orders mostly unique).
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

V = 151936


def pool_bytes():
    eval_list = sorted(
        (ROOT / "docs" / "research").glob("*.md")) + \
        [ROOT / "docs" / "CODEBASE_CRITIQUE_2026-09-16.md"]
    docs = b""
    for p in eval_list:                  # original order = same held
        docs += p.read_bytes()
    eval_files = {p.resolve() for p in eval_list}
    tr_docs, held = docs[:-8000], docs[-8000:]
    pool = bytearray(tr_docs)
    for d in ("forge", "scripts", "tests", "docs", "research"):
        base = ROOT / d
        if not base.is_dir():
            continue
        for r, _dirs, fs in os.walk(base):
            if "venv" in r or "__pycache__" in r:
                continue
            for f in fs:
                if not f.endswith((".py", ".md", ".txt")):
                    continue
                fp = (Path(r) / f).resolve()
                if fp in eval_files:      # held-out source: no leak
                    continue
                pool += fp.read_bytes()
    return tr_docs, bytes(pool), held


def pool_segments(tok):
    """Same pool but as (ids, domain) segments - .py -> "code",
    .md/.txt -> "prose" - for the domain-lane run."""
    tr_docs, _pool, held = pool_bytes()
    eval_list = sorted(
        (ROOT / "docs" / "research").glob("*.md")) + \
        [ROOT / "docs" / "CODEBASE_CRITIQUE_2026-09-16.md"]
    eval_files = {p.resolve() for p in eval_list}
    segs = [(tok(tr_docs.decode("utf-8", "ignore"))["input_ids"],
             "prose")]
    for d in ("forge", "scripts", "tests", "docs", "research"):
        base = ROOT / d
        if not base.is_dir():
            continue
        for r, _dirs, fs in os.walk(base):
            if "venv" in r or "__pycache__" in r:
                continue
            for f in fs:
                fp = (Path(r) / f).resolve()
                if not f.endswith((".py", ".md", ".txt")) \
                        or fp in eval_files:
                    continue
                dom = "code" if f.endswith(".py") else "prose"
                ids = tok(fp.read_bytes().decode("utf-8", "ignore")
                          )["input_ids"]
                segs.append((ids, dom))
    return segs, held


def tf(model_proto, ids, w_override=None, clone=True):
    # clone=False for models whose snapshot exceeds host RAM - eval
    # learns the held ids into m (only safe when m is discarded after)
    m = model_proto.clone() if clone else model_proto
    if w_override is not None:
        m._w = w_override.copy()
        m._wg.copy_(torch.from_numpy(w_override).to(m._dev))
    hit = n = 0
    nll = 0.0
    m._learn_token(ids[0])
    for t in range(1, len(ids)):
        lg = torch.as_tensor(m._predict()).float().cpu()
        truth = ids[t]
        hit += int(int(lg.argmax()) == truth)
        nll -= float(torch.log_softmax(lg, -1)[truth])
        m._learn_token(truth)
        n += 1
    return hit / n, nll / n


def run(name, train_ids, held_ids, ret=None, **kw):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cfg = {"vocab_size": V, "device": "cuda", "cuda_primary": True,
               "sparse_taps": (), "class_orders": (), "word_orders": (),
               "hybrid_taps": (), "use_knn": False,
               "orders": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)}
    cfg.update(kw)
    m = FluxLM(FluxConfig(**cfg))
    m.ingest(train_ids, tag="train")
    t1 = time.perf_counter()
    top1, nll = tf(m, held_ids)
    vram = torch.cuda.max_memory_allocated() / 1e9
    log_result("fluxscale", variant=name, n_train=len(train_ids),
               ingest_s=round(t1 - t0, 1),
               eval_s=round(time.perf_counter() - t1, 1),
               vram_gb=round(vram, 2), top1=round(top1, 4),
               nll=round(nll, 4))
    print(f"[fluxscale] {name}: nll={nll:.4f} top1={top1:.4f} "
          f"vram={vram:.2f}GB ingest={t1-t0:.0f}s "
          f"train={len(train_ids)}", flush=True)
    if ret is not None:
        ret["m"], ret["w"] = m, m._w.copy()
    else:
        del m
        torch.cuda.empty_cache()


def main():
    phases = sys.argv[1].split(",") if len(sys.argv) > 1 \
        else ["all"]
    _hf, tok = load_hf()
    tr_docs, pool, held_b = pool_bytes()
    held = tok(held_b.decode("utf-8", "ignore"))["input_ids"]
    print(f"[fluxscale] held={len(held)} bpe-tokens", flush=True)
    del _hf
    torch.cuda.empty_cache()

    tr_ids = tok(tr_docs.decode("utf-8", "ignore"))["input_ids"]
    dret = {}
    if "base" in phases or "all" in phases:
        run("docs190k", tr_ids, held, ret=dret)
    pool_ids = tok(pool.decode("utf-8", "ignore"))["input_ids"]
    print(f"[fluxscale] pool={len(pool_ids)} bpe-tokens", flush=True)
    if "pool" in phases or "all" in phases:
        pret = {}
        run("pool_bigslots", pool_ids, held, ret=pret,
            gpu_table_slots=1 << 20)
        # attribute the pool regression: eval the pool model under
        # DOCS-adapted hedge weights and under uniform weights -
        # isolates w-mismatch (shared state) from cell pollution.
        import numpy as np
        pm = pret["m"]
        if dret:
            for tag, w in (("docs_w", dret["w"]),
                           ("unif_w", np.full_like(
                               pm._w, 1.0 / len(pm._w)))):
                t1, nl = tf(pm, held, w_override=w)
                print(f"[fluxscale] pool_bigslots+{tag}: nll={nl:.4f} "
                      f"top1={t1:.4f}", flush=True)
        # recency probe: keep ingesting docs over the pool model -
        # does ordering (docs-last) recover?
        pm.ingest(tr_ids, tag="recall")
        t1, nl = tf(pm, held)
        print(f"[fluxscale] pool+docs_last: nll={nl:.4f} "
              f"top1={t1:.4f}", flush=True)
        del pm
        torch.cuda.empty_cache()
    if "dom" not in phases and "all" not in phases:
        print("[fluxscale] done")
        return

    # domain lanes: same pool, per-file domain tags - low-order cells
    # stop being shared across domains, so code mass can't dilute
    # prose contexts.
    segs, _held = pool_segments(tok)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    m = FluxLM(FluxConfig(
        vocab_size=V, device="cuda", cuda_primary=True,
        sparse_taps=(), class_orders=(), word_orders=(),
        hybrid_taps=(), use_knn=False,
        orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
        gpu_table_slots=1 << 20,
        domains=("shared", "code", "prose"),
        domain_route_every=16))
    n_seg = 0
    for ids, dom in segs:
        m.ingest(ids, tag=dom, domain=dom)
        n_seg += len(ids)
    t1s = time.perf_counter()
    t1, nl = tf(m, held)
    vr = torch.cuda.max_memory_allocated() / 1e9
    print(f"[fluxscale] pool_dom: nll={nl:.4f} top1={t1:.4f} "
          f"vram={vr:.2f}GB ingest={t1s-t0:.0f}s "
          f"eval={time.perf_counter()-t1s:.0f}s n={n_seg} "
          f"dom={m._dom_id}", flush=True)
    del m
    torch.cuda.empty_cache()
    print("[fluxscale] done")


if __name__ == "__main__":
    main()

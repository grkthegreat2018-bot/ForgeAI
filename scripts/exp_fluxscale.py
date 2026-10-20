"""EXP: FluxLM scaling curve on Qwen-BPE tokens.

At fixed corpus size every auxiliary channel dilutes - the binding
constraint is coverage.  Measure NLL vs corpus size on the SAME
held-out docs slice: docs-only (190K) -> full local text pool
(repo .py/.md/.txt, ~12MB).  No gradient training; ingest = memory.

VRAM: gpu_table_slots raised for the big pool (cells grow
sublinearly with corpus but high orders mostly unique).
"""
import os, sys, time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import torch
from forge.model.flux import FluxConfig, FluxLM
from forge.exp.harness import log_result, load_hf

V = 151936


def pool_bytes():
    docs = b""
    for p in sorted((ROOT / "docs" / "research").glob("*.md")) + \
            [ROOT / "docs" / "CODEBASE_CRITIQUE_2026-09-16.md"]:
        docs += p.read_bytes()
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
                if f.endswith((".py", ".md", ".txt")):
                    pool += (Path(r) / f).read_bytes()
    return tr_docs, bytes(pool), held


def tf(model_proto, ids):
    m = model_proto.clone()
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


def run(name, train_ids, held_ids, **kw):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cfg = dict(vocab_size=V, device="cuda", cuda_primary=True,
               sparse_taps=(), class_orders=(), word_orders=(),
               hybrid_taps=(), use_knn=False,
               orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
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
    del m
    torch.cuda.empty_cache()


def main():
    _hf, tok = load_hf()
    tr_docs, pool, held_b = pool_bytes()
    held = tok(held_b.decode("utf-8", "ignore"))["input_ids"]
    print(f"[fluxscale] held={len(held)} bpe-tokens", flush=True)
    del _hf
    torch.cuda.empty_cache()

    tr_ids = tok(tr_docs.decode("utf-8", "ignore"))["input_ids"]
    run("docs190k", tr_ids, held)
    pool_ids = tok(pool.decode("utf-8", "ignore"))["input_ids"]
    print(f"[fluxscale] pool={len(pool_ids)} bpe-tokens", flush=True)
    run("pool_bigslots", pool_ids, held,
        gpu_table_slots=1 << 20)
    print("[fluxscale] done")


if __name__ == "__main__":
    main()

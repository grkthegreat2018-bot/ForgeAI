"""EXP: FluxLM on Qwen-BPE tokens vs Qwen2.5-0.5B itself.

Byte-level FluxLM can't be compared to a BPE LLM directly; running
it on the SAME token stream gives a same-unit NLL/token comparison.
No gradient training anywhere - ingest is the whole "training".

Variants: pure order mixture -> +knn -> +deep orders -> +sparse taps.
Metrics: teacher-forced NLL/token + top1 on held-out docs bytes,
VRAM/RAM, ingest/eval seconds.  Reference: HF Qwen NLL same text.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import torch
from forge.model.flux import FluxConfig, FluxLM
from forge.exp.harness import log_result, load_hf, hf_nll, vram_gb

V = 151936


def corpus_ids(tok):
    data = b""
    for p in sorted((ROOT / "docs" / "research").glob("*.md")) + \
            [ROOT / "docs" / "CODEBASE_CRITIQUE_2026-09-16.md"]:
        data += p.read_bytes()
    tr_b, te_b = data[:-8000], data[-8000:]
    tr = tok(tr_b.decode("utf-8", "ignore"))["input_ids"]
    te = tok(te_b.decode("utf-8", "ignore"))["input_ids"]
    return tr, te, te_b.decode("utf-8", "ignore")


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


def run(name, train, held, **kw):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cfg = dict(vocab_size=V, device="cuda", cuda_primary=True,
               sparse_taps=(), class_orders=(), word_orders=(),
               hybrid_taps=())
    cfg.update(kw)
    m = FluxLM(FluxConfig(**cfg))
    m.ingest(train, tag="train")
    t1 = time.perf_counter()
    top1, nll = tf(m, held)
    vram = torch.cuda.max_memory_allocated() / 1e9
    log_result("fluxtok", variant=name,
               n_train=len(train), n_eval=len(held),
               ingest_s=round(t1 - t0, 1),
               eval_s=round(time.perf_counter() - t1, 1),
               vram_gb=round(vram, 2),
               top1=round(top1, 4), nll=round(nll, 4))
    print(f"[fluxtok] {name}: nll={nll:.4f} top1={top1:.4f} "
          f"vram={vram:.2f}GB ingest={t1-t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()


def main():
    _hf, tok = load_hf()
    train, held, held_text = corpus_ids(tok)
    print(f"[fluxtok] train={len(train)} held={len(held)} bpe-tokens",
          flush=True)
    which = os.environ.get("FLUXTOK_VARIANTS", "all")
    if which == "all":
        # Qwen reference on the same held-out text (8k-char chunks)
        chunks = [held_text[i:i + 8000]
                  for i in range(0, len(held_text), 8000)]
        qnll = hf_nll(_hf, tok, chunks)
        log_result("fluxtok", variant="qwen_ref", nll=round(qnll, 4))
        print(f"[fluxtok] qwen_ref nll={qnll:.4f}", flush=True)
    del _hf
    torch.cuda.empty_cache()

    if which == "all":
        base = dict(use_knn=False)
        run("bpe-o1248", train, held,
            orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32), **base)
        run("bpe-o1248-knn", train, held,
            orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
        run("bpe-deep-knn", train, held)      # default orders (3..64)
        run("bpe-o1248-sparse", train, held,
            orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
            sparse_taps=((0, 2), (0, 2, 4), (0, 2, 4, 6),
                         (1, 3), (1, 3, 5)))

    # BPE surface classes: make g<k>/gv meaningful at V=151936
    if which == "cls":
        from forge.model.flux.helpers import bpe_class_table
        _pieces = tok.convert_ids_to_tokens(list(range(V)))
        cls = bpe_class_table(lambda t: _pieces[t] or "", V,
                              set(tok.all_special_ids))
        run("bpe-o1248-sparse", train, held,
            orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
            sparse_taps=((0, 2), (0, 2, 4), (0, 2, 4, 6),
                         (1, 3), (1, 3, 5)))
        oc = dict(orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
                  cls_table=tuple(cls))
        run("bpe-o1248-cls", train, held,
            class_orders=(2, 3, 4), **oc)
        run("bpe-o1248-cls-gv", train, held, class_orders=(2, 3, 4),
            gen_variants=True, gen_fix_tail=0, gen_cls_cap=64, **oc)

    if which == "embc":
        # semantic classes: k-means over Qwen's embedding table ->
        # gv variants become same-MEANING edits, g<k> get real classes
        _hf2, _ = load_hf()
        emb = _hf2.get_input_embeddings().weight.detach()
        from forge.model.flux.helpers import emb_class_table
        ecls, emem = emb_class_table(
            emb, n_cls=256, special_ids=set(tok.all_special_ids))
        del _hf2
        torch.cuda.empty_cache()
        eo = dict(orders=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
                  cls_table=tuple(ecls),
                  cls_members=tuple(tuple(m) for m in emem))
        run("bpe-o1248-embc", train, held,
            class_orders=(2, 3), **eo)
        run("bpe-o1248-embc-gv", train, held, class_orders=(2, 3),
            gen_variants=True, gen_fix_tail=0, gen_cls_cap=48, **eo)
    print("[fluxtok] done")


if __name__ == "__main__":
    main()

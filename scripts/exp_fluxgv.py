"""EXP: FluxLM generalized-variant channel "gv" - read-time, stateless.

gv generalizes the LIVE order tables at predict time by probing
same-class 1-edit suffix variants (rolling-hash O(1) re-keys).  It
owns no state: deleted evidence cannot persist through it.

Protocol:
  A) synthetic class-structured corpus: word templates with digit
     slots - held-out (word,unseen-digit) pairs test generalization
  B) real bytes (docs/research) held-out NLL/top1, base vs +gv
"""
import os, sys, random, time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import torch
from forge.model.flux import FluxConfig, FluxLM
from forge.exp.harness import log_result

WORDS = list(b"abcdefghij")          # 10 lowercase 'word' contexts
DIG_TRAIN = list(b"0123456789")[:8]  # 8 train digits, '8','9' held out
SEP = ord(' ')


def synth_corpus(n, digits, rng):
    """w d SEP triplets - class structure: any digit completes the
    pattern, but only 8 of 10 digits are trained."""
    toks, i = [], 0
    while len(toks) < n:
        toks += [WORDS[i % len(WORDS)], rng.choice(digits), SEP]
        i += 1
    return toks


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


def run(name, vocab, orders, train, held, eval_n, fix_tail=1):
    held = held[:eval_n]
    for tag, kw in [("base", {}), ("gv", dict(gen_variants=True,
                                             gen_fix_tail=fix_tail))]:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        m = FluxLM(FluxConfig(vocab_size=vocab, device="cuda",
                              cuda_primary=True,
                              orders=orders, sparse_taps=(),
                              class_orders=(), word_orders=(),
                              hybrid_taps=(), **kw))
        m.ingest(train, tag="train")
        w = float(m._w[m._chix["gv"]]) if "gv" in m._chix else -1
        t1 = time.perf_counter()
        top1, nll = tf(m, held)
        log_result("fluxgv", corpus=name, variant=tag,
                   ingest_s=round(t1 - t0, 1),
                   eval_s=round(time.perf_counter() - t1, 1),
                   vram_mb=round(
                       torch.cuda.max_memory_allocated() / 1e6, 1),
                   w_gv=round(w, 3), top1=round(top1, 4),
                   nll=round(nll, 4))
        del m
        torch.cuda.empty_cache()


def main():
    rng = random.Random(0)
    # held-out stream contains unseen '8','9' digit slots
    train = synth_corpus(60_000, DIG_TRAIN, rng)
    held = synth_corpus(6_000, list(b"0123456789"), rng)
    run("synth", 256, (2, 3), train, held, 3_000, fix_tail=0)

    data = b""
    for p in sorted((ROOT / "docs" / "research").glob("*.md")) + \
            [ROOT / "docs" / "CODEBASE_CRITIQUE_2026-09-16.md"]:
        data += p.read_bytes()
    data = list(data)
    run("text", 256, (3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
        data[:-8000], data[-8000:], 8_000)
    print("[fluxgv] done")


if __name__ == "__main__":
    main()

"""EXP: FluxLM reverse-consistency channel "rev" (bidirectional memory).

Upgrade under test:
  - sidecar: rev[succ]->{prev:count} + decode rerank fwd+alpha*rev
  - native:  "rev" Hedge channel — _rev1[last]->{succ:cnt} written free
             at ingest; channel votes v_t = uni_t·p(last|t) so the
             candidate frequency denominates the pair count (Bayes flip
             of the forward bigram).  Hedge picks the weight.

Protocol per corpus:
  A) teacher-forced held-out top1/NLL: base | +o1 | +rev | sidecar
  B) free-run greedy continuation: backward consistency (prev_nll
     under the frozen train reverse table) + novel-pair rate
  C) speed overhead

Corpora: synthetic sparse Markov chain (rev's hardest case — the
generative direction is all there is) and real markdown prose at byte
level (degenerate continuations exist for rev to suppress).
"""
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import torch

from forge.exp.harness import log_result
from forge.model.flux import FluxConfig, FluxLM
from forge.model.flux.revmem import ReverseMem, joint_prev_nll

V_MARKOV = 2048
K_CANDS = 4
ALPHAS = (0.3, 1.0, 2.0)


def markov_corpus(n, vocab, rng):
    """Sparse Markov chain: each state has 2-4 allowed successors —
    directional structure the reverse table can verify."""
    trans = {}
    for s in range(vocab):
        k = rng.randint(2, 4)
        ws = rng.sample(range(vocab), k)
        ps = [rng.random() + 0.5 for _ in range(k)]
        tot = sum(ps)
        trans[s] = (ws, [p / tot for p in ps])
    out, s = [], rng.randrange(vocab)
    for _ in range(n):
        out.append(s)
        ws, ps = trans[s]
        s = rng.choices(ws, ps)[0]
    return out


def text_corpus():
    """docs/research/*.md as a byte stream — real prose, V=256."""
    data = b""
    for p in sorted((ROOT / "docs" / "research").glob("*.md")) + \
             [ROOT / "docs" / "CODEBASE_CRITIQUE_2026-09-16.md"]:
        data += p.read_bytes()
    return list(data)


def tf_base_rev(model_proto, ids):
    """Teacher-forced top1 + held-out NLL on a fresh clone."""
    model = model_proto.clone()
    hit = n = 0
    nll = 0.0
    model._learn_token(ids[0])
    for t in range(1, len(ids)):
        lg = torch.as_tensor(model._predict()).float()
        truth = ids[t]
        lp = torch.log_softmax(lg, -1)
        hit += int(int(lg.argmax()) == truth)
        nll -= float(lp[truth])
        model._learn_token(truth)
        n += 1
    return hit / n, nll / n


def tf_sidecar(model_proto, rev, ids, alpha):
    """Baseline mixture + external rev rerank of top-K candidates."""
    model = model_proto.clone()
    hit0 = hitr = n = 0
    last = ids[0]
    model._learn_token(last)
    for t in range(1, len(ids)):
        lg = torch.as_tensor(model._predict()).float()
        topv, topi = lg.topk(min(K_CANDS, lg.numel()))
        truth = ids[t]
        hit0 += int(int(topi[0]) == truth)
        fwd = torch.log_softmax(topv, -1)
        sc = [float(fwd[j]) + alpha * rev.rev_logp(int(topi[j]), last)
              for j in range(topi.numel())]
        hitr += int(int(topi[sc.index(max(sc))]) == truth)
        model._learn_token(truth)
        last = truth
        n += 1
    return hit0 / n, hitr / n


def gen_consistency(model_proto, rev_frozen, prompt, max_new=64):
    """Free-run greedy; score generated text's backward consistency
    under the frozen train-time reverse table."""
    g = model_proto.clone()
    g.ingest(prompt, tag="gen", learn=False)
    ids = []
    for _ in range(max_new):
        x = int(torch.as_tensor(g._predict()).argmax())
        ids.append(x)
        g._learn_token(x)
    pn = joint_prev_nll(g, rev_frozen, list(prompt) + ids)
    return pn["prev_nll"], ids


def run_corpus(name, vocab, train, held, eval_n):
    orders_def = (3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    variants = {
        "base": {"orders": orders_def},
        "o1": {"orders": (1,) + orders_def},
        "rev": {"orders": orders_def, "rev_channel": True},
        "rev+o1": {"orders": (1,) + orders_def, "rev_channel": True},
    }
    held = held[:eval_n]
    rev_f = ReverseMem(vocab)
    rev_f.feed(train)
    pairset = set(zip(train, train[1:]))

    models = {}
    for tag, kw in variants.items():
        t0 = time.perf_counter()
        m = FluxLM(FluxConfig(vocab_size=vocab, device="cpu", **kw))
        m.ingest(train, tag="train")
        models[tag] = m
        ch = "rev" if "rev" in m._chix else \
            ("o1" if "o1" in m._chix else "uni")
        wr = float(m._w[m._chix[ch]])
        log_result("fluxrev", corpus=name, phase="ingest", variant=tag,
                   s=round(time.perf_counter() - t0, 1),
                   rev1_rows=len(getattr(m, "_rev1", {})),
                   w=round(wr, 3))

    for tag, m in models.items():
        top1, nll = tf_base_rev(m, held)
        log_result("fluxrev", corpus=name, phase="tf", variant=tag,
                   top1=round(top1, 4), nll=round(nll, 4))

    base = models["base"]
    for a in ALPHAS:
        _, r = tf_sidecar(base, rev_f, held, a)
        log_result("fluxrev", corpus=name, phase="tf_sidecar", alpha=a,
                   top1_rerank=round(r, 4))

    prompt = held[:8]
    for tag, m in models.items():
        pn, ids = gen_consistency(m, rev_f, prompt)
        novel = sum((x, y) not in pairset for x, y in zip(ids, ids[1:]))
        log_result("fluxrev", corpus=name, phase="gen", variant=tag,
                   prev_nll=round(pn, 3), novel_pairs=novel)


def main():
    rng = random.Random(0)
    corpus = markov_corpus(60_000 + 4_000, V_MARKOV, rng)
    run_corpus("markov", V_MARKOV, corpus[:60_000], corpus[60_000:],
               4_000)

    data = text_corpus()
    split = len(data) - 8_000
    run_corpus("text", 256, data[:split], data[split:], 8_000)
    print("[fluxrev] done")


if __name__ == "__main__":
    main()

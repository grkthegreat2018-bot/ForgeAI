"""A6 head-to-head: FluxLM vs a dependency-free TF-IDF RAG baseline on
held-out, doc-grounded QA.

Competitive audit (docs/COMPETITIVE_AUDIT_2026-10-31.md, A6 + appendix F)
repositions FluxLM as "a revertible memory layer, not an LM".  This script
is the falsification harness: a *weak* RAG stack (in-repo sparse TF-IDF
retriever + an honest second-stage overlap reranker — no faiss/BM25/embedder
deps) is scored on the same held-out QA set as a fresh FluxLM instance
configured with the R69/R70 extractive knn-QA knobs.  If even this baseline
beats FluxLM at held-out QA at corpus scale, the memory-layer positioning
must lean on what RAG cannot do (journaled bit-exact revert, explain()
provenance) rather than on QA accuracy.

    python scripts/bench_flux_vs_rag.py [--docs 2000] [--qa 40] [--seed 0]

Modes:
  default        synthetic corpus: every doc carries templated facts;
                 QA pairs query one fact via DIFFERENT surface phrasing
                 (held-out — the question's suffix is not in the doc).
  --qa-source concise  procedural pairs via
                 forge.self_play.infinite_loop._concise_qa_pairs (the
                 self-play eval machinery — neither leg is expected to
                 score; parity mode).
  --eval-json P  load flux_eval_web.json-format rows (prompt/response/
                 accepts); each row ALSO enters the corpus as a verbatim
                 "Q:..A:.." record (measures seen-record recall, both legs).

Grading: same word-bounded substring check as
InfiniteSelfPlayLoop._flux_check (answer-first or word-bounded anywhere).
RAG leg answers = the best question-overlap SENTENCE extracted from the
top-1 doc (extractive reader); retrieval recall reported separately as
the answer-present-in-doc ceiling (rag_doc_hit1 / rag_doc_hit5).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# R69/R70 tuned extractive-recall knob set (mirrors
# InfiniteSelfPlayLoop._FLUX_KNN_QA/_WRITE — kept in sync manually;
# predict-side knobs are safe on any model, write-side must be set
# BEFORE ingest).
_KNN_QA_WRITE = {"knn_window": 16, "knn_idf": 1.0, "knn_residual": 1.0}
_KNN_QA = {"knn_episode": True, "knn_k": 64, "knn_pool": 256,
           "knn_cover": 32, "knn_cover_w": 4.0, "knn_twin": 1.0,
           "knn_span": 8, "knn_dedup": 1, "knn_subq": True,
           "knn_atype": 1, "knn_lift": 8.0, "knn_lift_steps": 4,
           "knn_lift_deep": 8, "knn_lift_topconf": 0.4}

# CPU-time trims: channels that do not carry doc-grounded QA (semantic-
# spread readouts sem/csem, the templated-fallback tap arms p*/g*/h*,
# and the track_gaps probe bookkeeping) are switched off so the default
# --docs 2000 run fits the <5min budget (~510 vs ~385 tok/s ingest).
# Order/deep/episodic/knn machinery — the QA-relevant retrieval path —
# is untouched.  Measured effect on held_qa at this corpus family: none
# (flux collapsed the hedge mixture to a template channel either way).
_FLUX_BENCH_TRIMS = {"use_sem": False, "use_csem": False,
                     "track_gaps": False, "sparse_taps": (),
                     "class_orders": (), "hybrid_taps": ()}

_STOP = frozenset(
    "a an the of is are was were in on at to for by with and or it its "
    "what who whom whose which when where why how many much name does "
    "do did has have had be been city year".split())

_WORD = re.compile(r"[a-z0-9']+")
_SENT = re.compile(r"[^.!?\n]+[.!?]?")


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _content_terms(text: str) -> list[str]:
    return [t for t in _terms(text) if t not in _STOP]


# ── TF-IDF retriever + weak reranker (the baseline arm) ────────────────

class TfidfRetriever:
    """Sparse TF-IDF + cosine retriever.  ~40 lines, no dependencies —
    intentionally the WEAK baseline: it stands in for the dense
    retriever of a vanilla RAG stack (audit F).  ``rerank`` is an honest
    second-stage score = tfidf cosine + idf-weighted question-term
    coverage + best-sentence overlap, not a learned cross-encoder."""

    def __init__(self, docs):
        self.docs = [str(d) for d in docs]
        self.bags: list[dict[str, int]] = []
        self.postings: dict[str, set[int]] = {}
        df: dict[str, int] = {}
        for i, d in enumerate(self.docs):
            b: dict[str, int] = {}
            for t in _terms(d):
                b[t] = b.get(t, 0) + 1
            self.bags.append(b)
            for t in b:
                df[t] = df.get(t, 0) + 1
                self.postings.setdefault(t, set()).add(i)
        n = len(self.docs)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) + 1.0
                    for t, c in df.items()}
        self.norms = [
            math.sqrt(sum((tf * self.idf[t]) ** 2 for t, tf in b.items()))
            or 1.0 for b in self.bags]

    def _qvec(self, query: str) -> dict[str, float]:
        # query-side stopword removal (standard IR): 'how many does
        # have' are template words — matching them lets a same-template
        # record outrank the entity's own doc (measured: 4/40 misses).
        q: dict[str, float] = {}
        for t in _terms(query):
            if t in self.idf and t not in _STOP:
                q[t] = q.get(t, 0.0) + self.idf[t]
        return q

    def topk(self, query: str, k: int = 5) -> list[tuple[float, int]]:
        q = self._qvec(query)
        if not q:
            return []
        cand: set[int] = set()
        for t in q:
            cand.update(self.postings[t])
        qn = math.sqrt(sum(w * w for w in q.values())) or 1.0
        out = []
        for i in cand:
            b = self.bags[i]
            s = sum(w * b[t] * self.idf[t]
                    for t, w in q.items() if t in b)
            if s > 0:
                out.append((s / (qn * self.norms[i]), i))
        out.sort(reverse=True)
        return out[:k]

    def _sent_overlap(self, doc: str, qidf: dict[str, float]) -> float:
        """Max idf-weighted question-term overlap over the doc's
        sentences, normalized to [0,1] coverage."""
        tot = sum(qidf.values()) or 1.0
        best = 0.0
        for m in _SENT.finditer(doc):
            st = set(_terms(m.group(0)))
            best = max(best, sum(w for t, w in qidf.items() if t in st))
        return best / tot

    def rerank(self, query: str, k: int = 8) -> list[tuple[float, int]]:
        """Second stage over the tfidf top-k pool: base cosine + term
        coverage + best-sentence overlap.  Weak by design — documents
        what even a non-learned reranker buys (usually a few points)."""
        qidf = {t: w for t, w in self._qvec(query).items()
                if t not in _STOP}
        out = []
        for s0, i in self.topk(query, k):
            cover = (len(set(_content_terms(query)) & set(self.bags[i]))
                     / max(len(set(_content_terms(query))), 1))
            s = s0 + 0.25 * cover + 0.35 * self._sent_overlap(
                self.docs[i], qidf)
            out.append((s, i))
        out.sort(reverse=True)
        return out

    def extract(self, doc: str, query: str) -> str:
        """Extractive reader: the doc sentence with the max idf-weighted
        question-term overlap (the 'generated answer' for the RAG leg)."""
        qidf = {t: w for t, w in self._qvec(query).items()
                if t not in _STOP}
        best_s, best_sc = "", -1.0
        for m in _SENT.finditer(doc):
            st = set(_terms(m.group(0)))
            sc = sum(w for t, w in qidf.items() if t in st)
            if sc > best_sc:
                best_s, best_sc = m.group(0).strip(), sc
        return best_s


# ── grader (InfiniteSelfPlayLoop._flux_check semantics) ────────────────

def _check(output: str, answer: str) -> bool:
    a = answer.strip().lower()
    if not a:
        return False
    out = output.strip().lower()
    if out.startswith(a):
        return True
    return re.search(r"(?<![\w.])" + re.escape(a)
                     + r"(?=$|[^\w.]|\.(?!\d))", out) is not None


def _check_any(output: str, pair: dict) -> bool:
    if _check(output, pair.get("response", "")):
        return True
    return any(_check(output, a) for a in pair.get("accepts") or [])


# ── synthetic doc-grounded corpus ──────────────────────────────────────

_SYL_A = ("Zor Nim Kel Vor Tal Qua Bel Xer Mal Dor Fen Gar Hel Jor Lys "
          "Mor Pax Ril Sor Til Vex Wyr Yol Zen Bex Cor Dun Els Ful Nar "
          "Pol Ruv").split()
_SYL_B = ("ax ar el ix or us yn ath em il on ur ek om ul es it av er "
          "oc an ib od um as en ig ot").split()
_FIRST = ("Aldus Bren Cato Daria Elara Fenn Galen Hara Ilya Jora Kael "
          "Liora Mira Niko Oren Pia Quill Ren Sana Tovin Ulma Vey Wren "
          "Xara Yann Zola").split()
_LAST = ("Ashdown Brix Correll Dunmore Ellery Fennick Grell Halvor "
         "Ibsen Jory Kestrel Lorimer Maren Novik Ostren Quist Ryn "
         "Sorrel Talbot Ulric Vane Wexley Yarros Zeff").split()
_KINDS = ("island city archive guild observatory foundry monastery "
          "harbor republic academy vessel garden citadel").split()
_REGIONS = ("Aldermere Bront Cathal Devron Estmere Falken Ghor "
            "Havel Imre Jessel Kantos Lyrath").split()
_COLORS = ("crimson cobalt verdant amber ivory slate scarlet teal "
           "ochre violet").split()
_UNITS = ("towers bridges canons archives docks terraces vaults "
          "spires canals lanterns").split()
_METALS = ("nimbar keralt vostrum aldrite pellium tarsil corvex "
           "dunite elvar brass").split()
_FILLER = ("Travelers describe {e} as {adj} and {adj2}.",
           "The {r} records mention {e} frequently.",
           "Scholars still debate the origins of {e}.",
           "{e} appears on old trade maps of the {r} region.")
_ADJS = ("austere bustling fortified ancient terraced windswept "
         "mercantile scholastic remote gilded").split()


def _entity(rng: random.Random, used: set) -> str:
    while True:
        n = rng.choice(_SYL_A) + rng.choice(_SYL_B)
        if rng.random() < 0.4:
            n += rng.choice(_SYL_B)
        if n not in used:
            used.add(n)
            return n


def _person(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def _fact(rng: random.Random, e: str, region: str) -> dict:
    """One templated fact: (doc sentence, question, answer, accepts)."""
    kind = rng.choice(("capital", "year", "discoverer", "count",
                       "color", "metal", "population"))
    if kind == "capital":
        v = f"{rng.choice(_SYL_A)}{rng.choice(_SYL_B)}"
        return {"kind": kind,
                "sent": f"The capital of {e} is {v}.",
                "q": f"What is the capital of {e}?",
                "a": v, "accepts": []}
    if kind == "year":
        v = str(rng.randint(1200, 1999))
        return {"kind": kind,
                "sent": f"{e} was founded in the year {v}.",
                "q": f"In what year was {e} founded?",
                "a": v, "accepts": []}
    if kind == "discoverer":
        p = _person(rng)
        last = p.split()[-1]
        return {"kind": kind,
                "sent": f"{e} was discovered by {p}.",
                "q": f"Who discovered {e}?",
                "a": p, "accepts": [last]}
    if kind == "count":
        u = rng.choice(_UNITS)
        v = str(rng.randint(2, 99))
        return {"kind": kind,
                "sent": f"{e} has {v} {u}.",
                "q": f"How many {u} does {e} have?",
                "a": v, "accepts": []}
    if kind == "color":
        v = rng.choice(_COLORS)
        return {"kind": kind,
                "sent": f"The flag of {e} is {v}.",
                "q": f"What color is the flag of {e}?",
                "a": v, "accepts": []}
    if kind == "metal":
        v = rng.choice(_METALS)
        return {"kind": kind,
                "sent": f"{e} is built mostly of {v}.",
                "q": f"What is {e} built mostly of?",
                "a": v, "accepts": []}
    v = f"{rng.randint(2, 900) * 1000:,}".replace(",", " ")
    return {"kind": "population",
            "sent": f"The population of {e} is {v}.",
            "q": f"What is the population of {e}?",
            "a": v, "accepts": [v.replace(" ", "")]}


def synth_corpus(n_docs: int, n_qa: int, seed: int):
    """-> (docs, pairs).  Each doc: one invented entity + 3-4 templated
    declarative facts + filler.  Each QA pair targets a DISTINCT doc and
    a fact in it, phrased differently from the doc sentence — the doc
    holds the answer declaratively, the question is held-out phrasing.
    Pair format mirrors FluxGoalSampler.build_eval rows."""
    rng = random.Random(0xBEEF + seed)
    used: set[str] = set()
    docs: list[str] = []
    facts: list[list[dict]] = []
    for _ in range(n_docs):
        e = _entity(rng, used)
        region = rng.choice(_REGIONS)
        kname = rng.choice(_KINDS)
        fl = []
        seen_k = set()
        for _ in range(rng.randint(3, 4)):
            f = _fact(rng, e, region)
            if f["kind"] in seen_k:
                continue
            seen_k.add(f["kind"])
            fl.append(f)
        sents = [f["sent"] for f in fl]
        for _ in range(rng.randint(0, 2)):
            sents.append(rng.choice(_FILLER).format(
                e=e, r=region, adj=rng.choice(_ADJS),
                adj2=rng.choice(_ADJS)))
        docs.append(f"{e} is a {kname} in the {region} region. "
                    + " ".join(sents))
        facts.append(fl)
    idx = rng.sample(range(n_docs), min(n_qa, n_docs))
    pairs = []
    for i in idx:
        fl = facts[i]
        f = fl[rng.randrange(len(fl))] if fl else None
        if f is None:
            continue
        pairs.append({"prompt": f["q"], "response": f["a"],
                      "accepts": f["accepts"], "source": "synth",
                      "doc_idx": i})
    return docs, pairs


def scaffold_records(k: int, seed: int) -> list[tuple[str, str, str]]:
    """Q/A-format demo pairs on INVENTED entities — (question, answer,
    entity).  Teaches the answer-slot surface form (production FluxLM
    deployments get it from self-play teach_text calls).  Callers must
    still drop records whose entity appears in the corpus or whose
    answer equals an eval gold — ``_clean_scaffold`` does both."""
    rng = random.Random(0x5CAFF0 + seed)
    used: set[str] = set()
    out = []
    for _ in range(k):
        e = _entity(rng, used)
        if rng.random() < 0.5:
            u, v = rng.choice(_UNITS), str(rng.randint(2, 99))
            q, a = f"How many {u} does {e} have?", v
        else:
            c = rng.choice(_COLORS)
            q, a = f"What color is the flag of {e}?", c
        out.append((q, a, e))
    return out


def _clean_scaffold(records, docs, pairs):
    """Held-out hygiene: drop scaffold demos whose invented entity name
    appears anywhere in the corpus/eval (would leak a binding) or whose
    answer coincides with an eval gold (right answer for the wrong
    reason).  Returns [(q, a)] records."""
    hay = " ".join(docs).lower() + " " + " ".join(
        p["prompt"].lower() for p in pairs)
    golds = {p.get("response", "") for p in pairs}
    golds |= {a for p in pairs for a in (p.get("accepts") or [])}
    return [(q, a) for q, a, e in records
            if e.lower() not in hay and a not in golds]


def _concise_pairs(n: int, seed: int) -> list[dict]:
    """Procedural pairs via the self-play eval machinery."""
    from forge.self_play.infinite_loop import _concise_qa_pairs
    return [dict(p, kind="qa", source="procedural")
            for p in _concise_qa_pairs(n, seed)]


def _load_eval_json(path: str) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r for r in rows if isinstance(r, dict) and r.get("prompt")]


# ── FluxLM leg ─────────────────────────────────────────────────────────

def _load_tok(args):
    """-> (encode, decode, vocab_size, itos_or_None)."""
    if args.bytes:
        return (lambda s: list(s.encode("utf-8")),
                lambda ids: bytes(int(t) % 256 for t in ids)
                .decode("utf-8", "replace"),
                256, None)
    try:
        from research.tokenizer_cache import get_tokenizer
        from forge.engine.engine_common import _tokenizer_for_vocab
        tok = get_tokenizer(_tokenizer_for_vocab(65536))
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] tokenizer load failed ({e}); "
              "falling back to --bytes vocab=256", file=sys.stderr)
        return _load_tok(argparse.Namespace(**{**vars(args),
                                               "bytes": True}))

    def enc(s: str):
        ids = tok.encode(s, add_special_tokens=False)
        return list(getattr(ids, "input_ids", getattr(ids, "ids", ids)))

    def dec(ids):
        try:
            return tok.decode(list(ids))
        except Exception:                                  # noqa: BLE001
            return tok.decode([int(t) for t in ids])
    itos = [tok.decode([i]) for i in range(65536)]
    return enc, dec, 65536, itos


def run_flux(docs, pairs, enc, dec, vocab, itos, args, scaff=()):
    from forge.model.flux import FluxConfig, FluxLM
    cfg = FluxConfig(vocab_size=vocab, device="cpu")
    for k, v in {**_KNN_QA_WRITE, **_KNN_QA, **_FLUX_BENCH_TRIMS}.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    m = FluxLM(cfg)
    if itos is not None:
        m.attach_vocab(itos[:m.vocab_size])
    t0 = time.perf_counter()
    ntok = 0
    for d in docs:
        m.journal.append(("ws", "corpus"))
        ntok += m.ingest(enc(d), tag="corpus")
        m.journal.append(("we", "corpus"))
        m.soft_reset()
    # Q/A-format scaffold: teach_text hot-binds the answer slot on
    # entities outside the corpus (same call self-play uses on fails).
    # Records are ALSO appended to docs upstream so the RAG leg sees
    # the identical corpus.
    tok_call = (lambda s, **kw: enc(s))
    for q, a in scaff:
        m.teach_text(f"Q: {q}\nA:", f" {a}\n", tok_call,
                     tag="scaffold")
    t_ing = time.perf_counter() - t0
    outs, t_gen = [], 0.0
    for p in pairs:
        q = f"Q: {p['prompt']}\nA:"
        t1 = time.perf_counter()
        ids = m.generate(enc(q), max_new=args.max_new,
                         temperature=0.0, learn=False)
        t_gen += time.perf_counter() - t1
        outs.append(dec(ids))
    hit = sum(_check_any(o, p) for o, p in zip(outs, pairs))
    w = m.channel_weights()
    return {"held_qa_flux": hit / max(len(pairs), 1),
            "ingest_s": round(t_ing, 2),
            "ingest_tps": round(ntok / max(t_ing, 1e-9), 1),
            "gen_s": round(t_gen, 2), "tokens": ntok,
            "cells": sum(len(t) for t in m._tables.values()),
            "flux_w_top": sorted(w.items(), key=lambda kv: -kv[1])[:3]}, outs


def run_rag(docs, pairs, retr_cls=TfidfRetriever):
    t0 = time.perf_counter()
    r = retr_cls(docs)
    t_fit = time.perf_counter() - t0
    hit1 = hitr = dhit1 = dhit5 = 0
    outs1, outsr = [], []
    for p in pairs:
        q = p["prompt"]
        gold = [p.get("response", "")] + list(p.get("accepts") or [])
        pool = r.topk(q, 5)
        if pool:
            d1 = r.docs[pool[0][1]]
            ans = r.extract(d1, q)
            outs1.append(ans)
            hit1 += _check_any(ans, p)
            dhit1 += any(_check(d1, g) for g in gold)
            dhit5 += any(any(_check(r.docs[i], g) for g in gold)
                         for _s, i in pool)
        else:
            outs1.append("")
        rp = r.rerank(q, 8)
        if rp:
            ansr = r.extract(r.docs[rp[0][1]], q)
            outsr.append(ansr)
            hitr += _check_any(ansr, p)
        else:
            outsr.append("")
    n = max(len(pairs), 1)
    return {"held_qa_rag": hit1 / n, "held_qa_rag_rerank": hitr / n,
            "rag_doc_hit1": dhit1 / n, "rag_doc_hit5": dhit5 / n,
            "fit_s": round(t_fit, 3)}, outs1, outsr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--qa", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa-source", default="synth",
                    choices=("synth", "concise"))
    ap.add_argument("--eval-json", default="",
                    help="flux_eval_web.json-format pairs; each row also "
                         "joins the corpus as a verbatim Q:/A: record")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--bytes", action="store_true",
                    help="vocab=256 byte-level FluxLM (no tokenizer dep)")
    ap.add_argument("--scaffold", type=int, default=24,
                    help="Q/A-format demo pairs taught on non-corpus "
                         "entities (0 = pure-doc corpus)")
    ap.add_argument("--skip-flux", action="store_true",
                    help="score only the RAG legs")
    ap.add_argument("--save-eval", default="",
                    help="persist the eval pairs (flux_eval_web.json "
                         "format) to this path")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="results/flux_vs_rag.json")
    args = ap.parse_args()

    t_all = time.perf_counter()
    enc, dec, vocab, itos = _load_tok(args)

    if args.eval_json:
        pairs = _load_eval_json(args.eval_json)[: args.qa or None]
        docs, _ = synth_corpus(args.docs, 0, args.seed)
        # each eval row also exists as a verbatim record in the corpus —
        # this arm measures seen-record recall, not held-out generalization
        docs = docs + [f"Q: {p['prompt']}\nA: {p.get('response', '')}"
                       for p in pairs]
        qa_source = os.path.basename(args.eval_json)
    elif args.qa_source == "concise":
        docs, _ = synth_corpus(args.docs, 0, args.seed)
        pairs = _concise_pairs(args.qa, args.seed)
        qa_source = "concise"
    else:
        docs, pairs = synth_corpus(args.docs, args.qa, args.seed)
        qa_source = "synth"
    if args.save_eval:
        Path(args.save_eval).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_eval).write_text(json.dumps(
            [{k: v for k, v in p.items() if k != "doc_idx"}
             for p in pairs], ensure_ascii=False, indent=1),
            encoding="utf-8")

    scaff = _clean_scaffold(scaffold_records(args.scaffold, args.seed),
                            docs, pairs) if args.scaffold > 0 else []
    # scaffold records join the corpus for BOTH legs — same memory store
    docs = docs + [f"Q: {q}\nA: {a}" for q, a in scaff]

    res = {"docs": len(docs), "n": len(pairs), "seed": args.seed,
           "qa_source": qa_source, "vocab": vocab,
           "scaffold": len(scaff)}
    rag, outs1, outsr = run_rag(docs, pairs)
    res.update(rag)
    flux = None
    if not args.skip_flux:
        flux, fouts = run_flux(docs, pairs, enc, dec, vocab, itos,
                               args, scaff=scaff)
        res.update(flux)
    res["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    if args.verbose:
        for i, p in enumerate(pairs):
            print(f"  [{i}] {p['prompt']!r} -> gold {p['response']!r}")
            print(f"      rag1 : {outs1[i]!r}")
            print(f"      rerk : {outsr[i]!r}")
            if flux:
                print(f"      flux : {fouts[i]!r}")
    line = (f"held_qa: flux={res.get('held_qa_flux')} "
            f"rag={res['held_qa_rag']:.3f} "
            f"rag+rerank={res['held_qa_rag_rerank']:.3f} "
            f"(doc hits: @1={res['rag_doc_hit1']:.3f} "
            f"@5={res['rag_doc_hit5']:.3f}) n={res['n']} "
            f"docs={res['docs']} [{res['elapsed_s']}s]")
    print(line)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

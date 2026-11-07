"""Tests for the A6 FluxLM-vs-RAG harness (scripts/bench_flux_vs_rag.py).

Retriever/reranker/grader correctness only — CPU-fast, no FluxLM load.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from bench_flux_vs_rag import TfidfRetriever, _check, _check_any, _clean_scaffold, scaffold_records, synth_corpus

DOCS = [
    "Zorblax is a city in the Aldermere region. The capital of Zorblax "
    "is Vendrix. Zorblax was founded in the year 1204.",
    "Nimbar is a archive in the Bront region. Nimbar was discovered by "
    "Elara Quist. The flag of Nimbar is cobalt.",
    "Kerathi is a harbor in the Cathal region. Kerathi has 42 docks. "
    "The population of Kerathi is 87 000.",
]


def test_tfidf_top1_returns_answer_doc():
    r = TfidfRetriever(DOCS)
    hits = r.topk("What is the capital of Zorblax?", 1)
    assert hits and hits[0][1] == 0
    hits = r.topk("Who discovered Nimbar?", 1)
    assert hits and hits[0][1] == 1
    hits = r.topk("How many docks does Kerathi have?", 1)
    assert hits and hits[0][1] == 2


def test_tfidf_no_match_returns_empty():
    r = TfidfRetriever(DOCS)
    assert r.topk("quantum entanglement zebra") == []


def test_rerank_keeps_or_improves_top1():
    r = TfidfRetriever(DOCS)
    for q, want in (("What color is the flag of Nimbar?", 1),
                    ("What is the population of Kerathi?", 2)):
        base = r.topk(q, 1)
        re_ = r.rerank(q, 8)
        assert base[0][1] == want
        assert re_[0][1] == want


def test_extract_picks_answer_sentence():
    r = TfidfRetriever(DOCS)
    d = DOCS[0]
    assert "Vendrix" in r.extract(d, "What is the capital of Zorblax?")
    assert "1204" in r.extract(d, "In what year was Zorblax founded?")


def test_check_matches_flux_grader():
    assert _check(" Vendrix, founded in 1204.", "Vendrix")
    assert _check("42", "42")
    assert _check("the answer is 42.", "42")
    # answer-first quirk (verbatim _flux_check semantics): startswith
    # has no word boundary, so a leading "42.5" counts as "42"
    assert _check("42.5 is close", "42")
    assert not _check("x 42.5 is close", "42")
    assert not _check("the Vendrixia is larger", "Vendrix")
    assert _check_any(" Quist", {"response": "Elara Quist",
                                "accepts": ["Quist"]})


def test_synth_corpus_deterministic_and_grounded():
    docs1, pairs1 = synth_corpus(60, 10, 3)
    docs2, pairs2 = synth_corpus(60, 10, 3)
    assert docs1 == docs2 and pairs1 == pairs2
    assert len(pairs1) == 10
    # every gold answer is present verbatim in its source doc
    for p in pairs1:
        assert p["response"] in docs1[p["doc_idx"]]
    # entity isolation: a question's entity names retrieve its own doc
    r = TfidfRetriever(docs1)
    hit = 0
    for p in pairs1:
        top = r.topk(p["prompt"], 1)
        if top and p["response"] in docs1[top[0][1]]:
            hit += 1
    assert hit >= 8            # weak baseline still retrieves ~all


def test_scaffold_records_no_leak():
    docs, pairs = synth_corpus(40, 6, 0)
    raw = scaffold_records(24, 0)
    assert all(len(r) == 3 for r in raw)
    scaff = set(_clean_scaffold(raw, docs, pairs))
    golds = {p["response"] for p in pairs}
    hay = " ".join(docs).lower()
    for q, a, e in raw:
        if (q, a) in scaff:
            assert a not in golds
            assert e.lower() not in hay

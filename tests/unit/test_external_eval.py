"""Unit tests for forge.evaluation.external_eval (AUDIT A1).

CPU-fast: a deterministic stub model that always predicts
``next_id = (prev_id + 1) % 256`` with high confidence, plus a byte
tokenizer stub — no checkpoints or network required.
"""
from __future__ import annotations

import json

import torch

from forge.evaluation import external_eval as ee


class _ByteTok:
    eos_token_id = 0
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return [b for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(b % 256 for b in ids).decode("utf-8", "ignore")


class _ConstModel(torch.nn.Module):
    """Logits concentrated on a fixed vocab id at every position — only
    continuations containing that token score well."""

    def __init__(self, k: int = 99):  # 99 = 'c'
        super().__init__()
        self.k = k

    def forward(self, ids):
        T = ids.shape[1]
        logits = torch.full((1, T, 256), -20.0)
        logits[0, :, self.k] = 20.0
        return logits, None


def _model():
    return _ConstModel().eval()


def test_loglikelihood_scores_correct_continuation():
    m, tok = _model(), _ByteTok()
    lp, ntok, greedy = ee.loglikelihood(
        m, tok, "ab", "c", torch.device("cpu"))
    # 'b'(98)+1 = 99 = 'c' → the continuation is the argmax: lp ≈ 0
    assert ntok == 1 and greedy and lp > -1e-3


def test_loglikelihood_wrong_continuation_scores_low():
    m, tok = _model(), _ByteTok()
    good, _, _ = ee.loglikelihood(m, tok, "ab", "c", torch.device("cpu"))
    bad, _, greedy = ee.loglikelihood(m, tok, "ab", "z", torch.device("cpu"))
    assert good > bad and not greedy


def test_mc_task_picks_gold():
    rows = [{"context": "ab", "choices": ["z", "c", "y"], "answer": 1}]
    res = ee.run_mc_task(_model(), _ByteTok(), rows, torch.device("cpu"),
                         task="arc_easy")
    assert res["acc"] == 1.0 and res["n"] == 1


def test_mmlu_letter_format_prefers_correct_letter():
    # _ConstModel(66) only scores 'B' well; with letter continuations
    # (" A".." D") the ' B' continuation must win → gold=B=1.
    rows = [{"context": "q", "choices": ["x", "y", "z", "w"],
             "answer": 1}]
    res = ee.run_mc_task(_ConstModel(66).eval(), _ByteTok(), rows,
                         torch.device("cpu"), task="mmlu")
    assert res["acc"] == 1.0


def test_mmlu_prompt_renders_letter_choices():
    p = ee._mmlu_prompt("What?", ["aa", "bb"])
    assert "A. aa" in p and "B. bb" in p and p.endswith("Answer:")


def test_load_task_and_limit(tmp_path):
    p = tmp_path / "arc_easy.jsonl"
    rows = [{"context": f"q{i}", "choices": ["a", "b"], "answer": 0}
            for i in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(ee.load_task("arc_easy", data_dir=tmp_path)) == 5
    assert len(ee.load_task("arc_easy", limit=2, data_dir=tmp_path)) == 2


def test_extract_gsm8k_answer():
    assert ee.extract_gsm8k_answer("blah blah #### 42") == 42.0
    assert ee.extract_gsm8k_answer("total 1,234.5 #### 1,234.5") == 1234.5
    assert ee.extract_gsm8k_answer("no marker, last num 17") == 17.0
    assert ee.extract_gsm8k_answer("nothing") is None


def test_compare_reports_gate():
    base = {"headline": {"arc_easy": 0.5, "mmlu": 0.4}}
    assert ee.compare_reports(
        base, {"headline": {"arc_easy": 0.51, "mmlu": 0.40}})["ok"]
    gate = ee.compare_reports(
        base, {"headline": {"arc_easy": 0.40, "mmlu": 0.40}})
    assert not gate["ok"] and gate["deltas"]["arc_easy"] < 0

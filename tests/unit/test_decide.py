"""Unit tests for SystemOneEvaluator (forge/engine/decide.py).

TypeSafe System One-compatible typed decisions: noul / choice / score
questions answered via candidate-continuation scoring in single forward
passes (no generation).

Tests cover:
  - question schema validation (bad types, bounds)
  - prompt building + answer wire shapes
  - both scoring paths (single-token fast path, batched continuation)
  - evaluate() end-to-end on a CPU tiny model + real tokenizer
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.engine.decide import (
    QuestionValidationError,
    SystemOneEvaluator,
    _jsonable,
    _parse_question,
    _peaked_confidence,
)

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _tiny_model():
    """Random-init gen_model_tiny (vocab 65536) on CPU."""
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = get_config("gen_model_tiny")
    cfg.device = "cpu"
    cfg.dtype = "float32"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Real Jamba tokenizer (gigatoken); skips if unavailable."""
    try:
        from research.tokenizer_cache import get_tokenizer
        return get_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def evaluator(tokenizer):
    model = _tiny_model()
    return SystemOneEvaluator(model, tokenizer, torch.device("cpu"))


# â”€â”€ _jsonable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestJsonable:
    def test_none(self):
        assert _jsonable(None) == ""

    def test_str_passthrough(self):
        assert _jsonable("hello") == "hello"

    def test_dict(self):
        assert _jsonable({"a": 1}) == '{"a": 1}'

    def test_list(self):
        assert _jsonable([1, "x"]) == '[1, "x"]'


# â”€â”€ _parse_question validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestParseQuestion:
    def test_noul_minimal(self):
        q = _parse_question("k", {"type": "noul"})
        assert q.qtype == "noul"
        assert q.labels == ["yes", "no"]
        assert q.candidates == [" yes", " no"]

    def test_noul_with_criteria(self):
        q = _parse_question("k", {
            "type": "noul", "instructions": "Is it spam?",
            "criteria": {"true": "spam", "false": "ham"}})
        assert q.noul_descs == {"true": "spam", "false": "ham"}

    def test_choice_requires_criteria(self):
        with pytest.raises(QuestionValidationError, match="criteria"):
            _parse_question("k", {"type": "choice"})

    def test_choice_option_bounds(self):
        with pytest.raises(QuestionValidationError, match="2-255"):
            _parse_question("k", {"type": "choice", "criteria": {"only": None}})
        many = {f"o{i}": None for i in range(256)}
        with pytest.raises(QuestionValidationError, match="2-255"):
            _parse_question("k", {"type": "choice", "criteria": many})
        ok = {f"o{i}": None for i in range(255)}
        q = _parse_question("k", {"type": "choice", "criteria": ok})
        assert len(q.candidates) == 255

    def test_score_level_bounds(self):
        with pytest.raises(QuestionValidationError, match="2-10"):
            _parse_question("k", {"type": "score", "criteria": ["x"]})
        with pytest.raises(QuestionValidationError, match="2-10"):
            _parse_question("k", {"type": "score",
                                  "criteria": [str(i) for i in range(11)]})
        q = _parse_question("k", {"type": "score",
                                  "criteria": ["bad", "ok", "good"]})
        assert q.labels == ["0", "1", "2"]
        assert q.candidates == [" 0", " 1", " 2"]
        assert q.legend == {"0": "bad", "1": "ok", "2": "good"}

    def test_unknown_type(self):
        with pytest.raises(QuestionValidationError, match="unknown type"):
            _parse_question("k", {"type": "essay"})

    def test_missing_type(self):
        with pytest.raises(QuestionValidationError, match="unknown type"):
            _parse_question("k", {"instructions": "hi"})

    def test_non_dict(self):
        with pytest.raises(QuestionValidationError, match="object"):
            _parse_question("k", "not a dict")


# â”€â”€ _peaked_confidence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPeakedConfidence:
    def test_uniform_is_zero(self):
        assert _peaked_confidence([0.5, 0.5]) == pytest.approx(0.0)
        assert _peaked_confidence([0.25] * 4) == pytest.approx(0.0)

    def test_certain_is_one(self):
        assert _peaked_confidence([1.0, 0.0]) == pytest.approx(1.0)
        assert _peaked_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)

    def test_monotonic(self):
        a = _peaked_confidence([0.6, 0.4])
        b = _peaked_confidence([0.9, 0.1])
        assert 0.0 < a < b <= 1.0


# â”€â”€ evaluate() end-to-end â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_STATE = "Acme Corp reported record Q3 revenue of $4.2B, up 18% YoY."


class TestEvaluate:
    def test_noul_wire_shape(self, evaluator):
        out = evaluator.evaluate(
            _STATE, {"good_news": {"type": "noul",
                                   "instructions": "The news is positive."}})
        ans = out["answers"]["good_news"]
        assert ans["type"] == "noul"
        assert 0.0 <= ans["noul"] <= 1.0
        assert set(ans) == {"type", "noul"}
        assert out["usage"]["output_tokens"] == 0
        assert out["usage"]["input_tokens"] > 0
        assert out["usage"]["billing_units"] == out["usage"]["input_tokens"]

    def test_choice_wire_shape(self, evaluator):
        out = evaluator.evaluate(
            _STATE,
            {"sector": {"type": "choice",
                        "instructions": "Which sector does this concern?",
                        "criteria": {"finance": "financial results",
                                     "sports": "athletics",
                                     "weather": "forecasts"}}})
        ans = out["answers"]["sector"]
        assert ans["type"] == "choice"
        assert ans["choice"] in {"finance", "sports", "weather"}
        probs = ans["probabilities"]
        assert set(probs) == {"finance", "sports", "weather"}
        assert sum(probs.values()) == pytest.approx(1.0)
        assert 0.0 <= ans["confidence"] <= 1.0

    def test_score_wire_shape(self, evaluator):
        out = evaluator.evaluate(
            _STATE,
            {"hype": {"type": "score",
                      "instructions": "How bullish is this report?",
                      "criteria": ["bearish", "neutral", "bullish"]}})
        ans = out["answers"]["hype"]
        assert ans["type"] == "score"
        assert 0.0 <= ans["score"] <= 2.0
        assert ans["legend"] == {"0": "bearish", "1": "neutral", "2": "bullish"}
        probs = ans["probabilities"]
        assert set(probs) == {"0", "1", "2"}
        assert sum(probs.values()) == pytest.approx(1.0)
        # score is the probability-weighted index
        expected = sum(int(k) * v for k, v in probs.items())
        assert ans["score"] == pytest.approx(expected)

    def test_mixed_questions_in_one_call(self, evaluator):
        out = evaluator.evaluate(
            _STATE,
            {"pos": {"type": "noul", "instructions": "Positive news?"},
             "sector": {"type": "choice",
                        "criteria": {"finance": None, "sports": None}},
             "hype": {"type": "score", "criteria": ["low", "high"]}})
        assert set(out["answers"]) == {"pos", "sector", "hype"}
        assert out["answers"]["pos"]["type"] == "noul"
        assert out["answers"]["sector"]["type"] == "choice"
        assert out["answers"]["hype"]["type"] == "score"

    def test_state_as_dict(self, evaluator):
        out = evaluator.evaluate(
            {"company": "Acme", "revenue": 4.2},
            {"ok": {"type": "noul", "instructions": "Revenue mentioned?"}})
        assert 0.0 <= out["answers"]["ok"]["noul"] <= 1.0

    def test_empty_state_rejected(self, evaluator):
        with pytest.raises(QuestionValidationError, match="state"):
            evaluator.evaluate("", {"q": {"type": "noul"}})

    def test_none_state_rejected(self, evaluator):
        with pytest.raises(QuestionValidationError, match="state"):
            evaluator.evaluate(None, {"q": {"type": "noul"}})

    def test_empty_questions_rejected(self, evaluator):
        with pytest.raises(QuestionValidationError, match="questions"):
            evaluator.evaluate(_STATE, {})

    def test_deterministic(self, evaluator):
        q = {"q": {"type": "noul", "instructions": "Is this about money?"}}
        a = evaluator.evaluate(_STATE, q)
        b = evaluator.evaluate(_STATE, q)
        assert a["answers"]["q"]["noul"] == b["answers"]["q"]["noul"]

    def test_context_limit_truncates(self, evaluator):
        long_state = "word " * 5000
        out = evaluator.evaluate(
            long_state, {"q": {"type": "noul"}}, context_limit=256)
        assert out["usage"]["input_tokens"] <= 300  # 256 + slack


# â”€â”€ ForgeEngine.decide() â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestEngineDecide:
    def test_decide_via_engine(self, tokenizer):
        from forge.engine.forge_engine import ForgeEngine
        model = _tiny_model()
        engine = ForgeEngine(model, tokenizer, device="cpu")
        try:
            out = engine.decide(
                _STATE,
                {"q": {"type": "choice",
                       "criteria": {"a": "first", "b": "second"}}})
            assert out["answers"]["q"]["choice"] in {"a", "b"}
        finally:
            del engine
            del model


# â”€â”€ POST /v1/systemone route â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _StubRegistry:
    """Minimal ModelRegistry stand-in for route tests."""

    def __init__(self, engine):
        self._engine = engine

    def list_models(self):
        return [{"id": "forgelm-v2-jamba", "awake": True,
                 "vram_budget_gb": 8.0, "config": "forgelm_v2",
                 "last_used_ago_s": 0.0, "generations": 0}]

    def get_engine(self, model_id):
        return self._engine if model_id == "forgelm-v2-jamba" else None


class _StubEngine:
    """Exposes decide() backed by the real evaluator."""

    def __init__(self, evaluator):
        self._ev = evaluator

    def decide(self, state, questions, **kwargs):
        return self._ev.evaluate(state, questions)


@pytest.fixture(scope="module")
def route_client(evaluator):
    from fastapi.testclient import TestClient

    from forge.engine.forge_server import ForgeServer
    server = ForgeServer(registry=_StubRegistry(_StubEngine(evaluator)))
    return TestClient(server.app)


class TestSystemOneRoute:
    def test_systemone_happy_path(self, route_client):
        resp = route_client.post("/v1/systemone", json={
            "state": _STATE,
            "model": "jev-latest",  # unregistered alias -> falls back
            "questions": {
                "pos": {"type": "noul", "instructions": "Positive?"},
                "sector": {"type": "choice",
                           "criteria": {"finance": None, "sports": None}},
            }})
        assert resp.status_code == 200
        assert resp.headers.get("x-typesafe-request-id")
        body = resp.json()
        assert body["model"] == "forgelm-v2-jamba"
        assert body["answers"]["pos"]["type"] == "noul"
        assert body["answers"]["sector"]["type"] == "choice"
        assert "usage" in body

    def test_systemone_validation_422(self, route_client):
        resp = route_client.post("/v1/systemone", json={
            "state": _STATE,
            "model": "forgelm-v2-jamba",
            "questions": {"bad": {"type": "choice"}}})  # missing criteria
        assert resp.status_code == 422

    def test_systemone_bad_type_422(self, route_client):
        resp = route_client.post("/v1/systemone", json={
            "state": _STATE,
            "questions": {"bad": {"type": "essay"}}})
        assert resp.status_code == 422

    def test_models_typesafe_shape(self, route_client):
        resp = route_client.get("/v1/models",
                          headers={"X-TypeSafe-SDK": "python/0.6.0"})
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert models[0]["name"] == "forgelm-v2-jamba"
        assert "description" in models[0]

    def test_models_openai_shape(self, route_client):
        resp = route_client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == "forgelm-v2-jamba"

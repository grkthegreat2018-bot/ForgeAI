"""Tests for research.training_free — forward-only alignment techniques.

Runs on CPU with the tiny_test config (no GPU required).
"""
import pytest
import torch

from forge.config import get_config
from forge.model_loader import ConfigurableResearchLLM
from research.training_free import (
    ActivationSteerer,
    RAINGenerator,
    ReflexionBuffer,
    TrainingFreeSolver,
    build_prompt,
    make_template_reflection,
)


@pytest.fixture(scope="module")
def tiny_model():
    cfg = get_config("forgelm_tiny")
    cfg.device = "cpu"
    model = ConfigurableResearchLLM(cfg)
    model.eval()
    return model


class _DummyEncoding:
    def __init__(self, ids_tensor):
        self.input_ids = ids_tensor

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        return self


class _DummyTokenizer:
    """Minimal HF-interface stub (ids < vocab_size=256) for the tiny model."""

    eos_token_id = 2

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def _ids(self, text: str) -> list[int]:
        return [len(t) % self.vocab_size for t in text.split()][:64]

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = self._ids(text)
        if return_tensors == "pt":
            return _DummyEncoding(torch.tensor([ids], dtype=torch.long))
        return {"input_ids": ids}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._ids(text)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return " ".join(f"tok{i}" for i in ids)


@pytest.fixture(scope="module")
def tiny_tokenizer():
    """Whitespace tokenizer stub so tests don't depend on the LFM tokenizer."""
    return _DummyTokenizer(vocab_size=256)


# ── URIAL ────────────────────────────────────────────────────────────────────


class TestURIAL:
    def test_prompt_contains_system_and_examples(self):
        p = build_prompt("Solve it")
        assert "You are a helpful" in p
        assert "sum_even" in p
        assert p.endswith("User: Solve it\nAssistant:")

    def test_extra_context_inserted_before_task(self):
        p = build_prompt("Solve it", extra_context="## Past attempts")
        assert p.index("## Past attempts") < p.index("User: Solve it")

    def test_example_count_respected(self):
        p0 = build_prompt("Solve it", n_examples=0)
        p3 = build_prompt("Solve it", n_examples=3)
        assert "sum_even" not in p0
        assert p0.count("User:") == 1
        assert p3.count("User:") == 4


# ── Reflexion ────────────────────────────────────────────────────────────────


class TestReflexion:
    def test_buffer_bounded(self):
        buf = ReflexionBuffer(max_entries=3, max_chars=1000)
        for i in range(5):
            buf.add(f"task{i}", error=f"err{i}", success=(i % 2 == 0))
        assert len(buf) == 3
        assert buf.successes == 2

    def test_template_reflection_mentions_error(self):
        r = make_template_reflection("t", code="", error="NameError: x")
        assert "NameError: x" in r

    def test_prompt_includes_memory(self):
        buf = ReflexionBuffer(max_entries=2)
        buf.add("fib", error="syntax", success=False)
        p = buf.prompt_for("new task")
        assert "Past attempts" in p
        assert "syntax" in p
        assert "new task" in p

    def test_context_bounded(self):
        buf = ReflexionBuffer(max_entries=8, max_chars=120)
        for i in range(8):
            buf.add(f"task{i}", error="e" * 50, success=False)
        assert len(buf.context_block()) <= 200  # 120 + truncation suffix


# ── Steering / task vectors ──────────────────────────────────────────────────


class TestSteering:
    def test_task_vectors_shape_and_normalization(self, tiny_model,
                                                  tiny_tokenizer):
        steerer = ActivationSteerer(tiny_model)
        pos = steerer.collect_activations(
            tiny_tokenizer, ["solve a b c", "solve d e f"], device="cpu")
        neg = steerer.collect_activations(
            tiny_tokenizer, ["fail x y z", "fail q r s"], device="cpu")
        assert pos and neg
        vectors = ActivationSteerer.task_vectors(pos, neg, normalize=True)
        assert set(vectors.keys()) == set(pos.keys())
        for v in vectors.values():
            assert v.dim() == 1
            assert abs(v.norm().item() - 1.0) < 1e-3

    def test_injection_changes_output_and_removes_cleanly(self, tiny_model,
                                                          tiny_tokenizer):
        steerer = ActivationSteerer(tiny_model)
        pos = steerer.collect_activations(
            tiny_tokenizer, ["solve a b c"], device="cpu")
        neg = steerer.collect_activations(
            tiny_tokenizer, ["fail x y z"], device="cpu")
        vectors = ActivationSteerer.task_vectors(pos, neg)
        assert vectors

        input_ids = tiny_tokenizer("solve a b c", return_tensors="pt").input_ids

        with torch.inference_mode():
            base, _ = tiny_model(input_ids)
            steerer.apply(vectors, alpha=5.0)
            assert steerer.active
            steered, _ = tiny_model(input_ids)
            steerer.remove()
            assert not steerer.active
            restored, _ = tiny_model(input_ids)

        assert not torch.allclose(base, steered, atol=1e-4)
        assert torch.allclose(base, restored, atol=1e-5)

    def test_context_manager_removes(self, tiny_model, tiny_tokenizer):
        steerer = ActivationSteerer(tiny_model)
        with steerer:
            pass
        assert not steerer.active


# ── RAIN ─────────────────────────────────────────────────────────────────────


class TestRAIN:
    def test_rewinds_when_score_low(self, tiny_model, tiny_tokenizer):
        calls = {"n": 0}

        def bad_eval(text, logprobs):
            calls["n"] += 1
            return 0.1  # always below threshold -> force rewinds

        gen = RAINGenerator(
            tiny_model, tiny_tokenizer, device="cpu",
            eval_fn=bad_eval, threshold=0.5, rewind_tokens=2,
            max_rewinds=2, max_tokens=8,
        )
        _text, score, scores = gen.generate("solve a b c")
        assert gen.rewinds_used == 2
        assert len(scores) == 3  # initial + 2 rewinds
        assert score < 0.5

    def test_no_rewind_when_score_ok(self, tiny_model, tiny_tokenizer):
        gen = RAINGenerator(
            tiny_model, tiny_tokenizer, device="cpu",
            eval_fn=lambda t, l: 0.9, threshold=0.5,
            max_rewinds=3, max_tokens=8,
        )
        _text, _score, scores = gen.generate("solve a b c")
        assert gen.rewinds_used == 0
        assert len(scores) == 1


# ── TrainingFreeSolver ───────────────────────────────────────────────────────


class TestTrainingFreeSolver:
    def test_end_to_end(self, tiny_model, tiny_tokenizer):
        solver = TrainingFreeSolver(
            tiny_model, tiny_tokenizer, device="cpu",
            max_tokens=8, capture_activations=True,
        )
        out = solver.generate("write a function")
        assert isinstance(out, str) and out

        solver.record("t1", output="def f(): pass", error="", success=True)
        solver.record("t2", output="def g():", error="SyntaxError", success=False)
        solver.record("t3", output="def h(): pass", error="", success=True)

        stats = solver.stats()
        assert stats["memory_entries"] == 3
        assert stats["successes"] == 2
        assert stats["pos_activation_sets"] == 2
        assert stats["neg_activation_sets"] == 1

        vectors = solver.build_task_vector()
        assert vectors, "task vector needs pos+neg activations"
        assert solver.apply_steering(alpha=1.0) is True
        assert solver.steering_active
        solver.clear_steering()
        assert not solver.steering_active

    def test_no_vector_without_negatives(self, tiny_model, tiny_tokenizer):
        solver = TrainingFreeSolver(
            tiny_model, tiny_tokenizer, device="cpu",
            max_tokens=4, capture_activations=False,
        )
        assert solver.build_task_vector() == {}
        assert solver.apply_steering() is False


# ── SelfPlaySandbox integration ──────────────────────────────────────────────


class TestSandboxIntegration:
    def test_run_task_feeds_training_free(self, tiny_model, tiny_tokenizer):
        import tempfile

        from forge.self_play.self_play_sandbox import SelfPlaySandbox

        solver = TrainingFreeSolver(
            tiny_model, tiny_tokenizer, device="cpu", max_tokens=8,
            capture_activations=True,
        )
        sandbox = SelfPlaySandbox(
            tiny_model, tiny_tokenizer, device="cpu",
            max_gen_tokens=8, training_free=solver,
            temp_dir=tempfile.gettempdir(),
        )

        class FakeExec:
            def execute(self, code, expected_output):
                return {"stdout": "55", "stderr": "", "returncode": 0,
                        "output_matches_expected": True, "io_score": 1.0,
                        "exec_time_ms": 0.1, "peak_memory_kb": 0,
                        "file_size_bytes": 10, "timed_out": False}

        sandbox.executor = FakeExec()
        packet = sandbox.run_task("Compute 5+5", expected_output="10")
        assert packet["quality_score"] > 0
        assert len(solver.memory) == 1
        assert solver.stats()["successes"] == 1
        assert solver.stats()["pos_activation_sets"] == 1

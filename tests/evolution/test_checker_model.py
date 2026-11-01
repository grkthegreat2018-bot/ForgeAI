"""Tests for SharedCheckerModel and HeuristicChecker.

Run with:
    set PYTHONPATH=<repo-root>
    python -m pytest tests/evolution/test_checker_model.py -v
"""

import pytest

from forge.evolution.checker_model import (
    HeuristicChecker,
    SharedCheckerModel,
    _BoundedLRUCache,
    _parse_score,
    reset_checker,
)

# ── HeuristicChecker tests ──────────────────────────────────────────────────


class TestHeuristicChecker:
    def setup_method(self):
        self.checker = HeuristicChecker()

    def test_good_answer(self):
        """Answer containing all requirement keywords scores high."""
        score = self.checker.check(
            question="Explain photosynthesis",
            answer="Photosynthesis converts sunlight into chemical energy using chlorophyll in plants.",
            requirements="Must mention photosynthesis, sunlight, and chemical energy",
        )
        assert 60 <= score <= 100, f"Expected high score, got {score}"

    def test_short_answer(self):
        """Very short answer gets penalized."""
        score = self.checker.check(
            question="Explain photosynthesis",
            answer="Yes.",
            requirements="Must mention photosynthesis, sunlight, and chemical energy",
        )
        assert score < 50, f"Expected low score for short answer, got {score}"

    def test_empty_answer(self):
        score = self.checker.check(
            question="What is 2+2?",
            answer="",
            requirements="Must contain the number 4",
        )
        assert score < 20, f"Expected very low score for empty answer, got {score}"

    def test_long_answer(self):
        """Very long answer gets penalized."""
        long_answer = "word " * 600
        score = self.checker.check(
            question="Explain X",
            answer=long_answer,
            requirements="Must explain X clearly",
        )
        assert score < 80, f"Expected penalty for long answer, got {score}"

    def test_no_requirements(self):
        """No extractable requirements → neutral keyword score."""
        score = self.checker.check(
            question="What is 2+2?",
            answer="The answer is four.",
            requirements="",
        )
        assert 0 <= score <= 100

    def test_score_range(self):
        """Score is always in [0, 100]."""
        score = self.checker.check("q", "a", "r")
        assert 0 <= score <= 100


# ── Score parsing tests ─────────────────────────────────────────────────────


class TestParseScore:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Score: 85", 85.0),
            ("score: 42", 42.0),
            ("Score = 73", 73.0),
            ("85/100", 85.0),
            ("85", 85.0),
            ("  90  \n", 90.0),
            ("85 points", 85.0),
            ("85 pts", 85.0),
            ("The score is 95 out of 100", 95.0),  # fallback catches 95
            ("hello world", None),
            ("", None),
            ("Score: 0", 0.0),
            ("Score: 100", 100.0),
            ("Score: 150", None),  # > 100, should not match first pattern
        ],
    )
    def test_parse(self, text, expected):
        result = _parse_score(text)
        assert result == expected, f"parse({text!r}) = {result}, expected {expected}"


# ── Bounded LRU Cache tests ─────────────────────────────────────────────────


class TestBoundedLRUCache:
    def test_basic_get_put(self):
        cache = _BoundedLRUCache(max_entries=5)
        cache.put("a", 1.0)
        assert cache.get("a") == 1.0
        assert cache.get("missing") is None

    def test_lru_eviction(self):
        cache = _BoundedLRUCache(max_entries=3)
        cache.put("a", 1.0)
        cache.put("b", 2.0)
        cache.put("c", 3.0)
        # Access "a" to make it recently used
        cache.get("a")
        # Insert "d" — should evict "b" (least recently used)
        cache.put("d", 4.0)
        assert cache.get("b") is None
        assert cache.get("a") == 1.0
        assert cache.get("c") == 3.0
        assert cache.get("d") == 4.0

    def test_clear(self):
        cache = _BoundedLRUCache(max_entries=5)
        cache.put("a", 1.0)
        cache.put("b", 2.0)
        cache.clear()
        assert cache.get("a") is None
        assert len(cache) == 0

    def test_update_existing(self):
        cache = _BoundedLRUCache(max_entries=5)
        cache.put("a", 1.0)
        cache.put("a", 2.0)
        assert cache.get("a") == 2.0
        assert len(cache) == 1


# ── SharedCheckerModel tests (heuristic fallback path) ──────────────────────


class TestSharedCheckerModel:
    def test_heuristic_only(self):
        """When model build fails, check() falls back to heuristic."""
        # Create instance with a bad config name to force fallback
        checker = SharedCheckerModel.__new__(SharedCheckerModel)
        checker._lock = __import__("threading").RLock()
        checker._cache = _BoundedLRUCache()
        checker._heuristic = HeuristicChecker()
        checker._is_sleeping = False
        checker._model_failed = True
        checker._model = None
        checker._tokenizer = None
        checker._config = None
        checker._vocab_size = 256
        checker._max_seq_len = 128
        checker._unpack_output = None
        checker._device = __import__("torch").device("cpu")

        score = checker.check("What is 2+2?", "The answer is 4.", "Must contain 4")
        assert 0 <= score <= 100

    def test_caching(self):
        """Same input returns cached score on second call."""
        checker = SharedCheckerModel.__new__(SharedCheckerModel)
        import threading

        import torch

        checker._lock = threading.RLock()
        checker._cache = _BoundedLRUCache()
        checker._heuristic = HeuristicChecker()
        checker._is_sleeping = False
        checker._model_failed = True
        checker._model = None
        checker._tokenizer = None
        checker._config = None
        checker._vocab_size = 256
        checker._max_seq_len = 128
        checker._unpack_output = None
        checker._device = torch.device("cpu")

        score1 = checker.check("q", "a", "r")
        score2 = checker.check("q", "a", "r")
        assert score1 == score2
        assert checker.cache_size() == 1

    def test_clear_cache(self):
        checker = SharedCheckerModel.__new__(SharedCheckerModel)
        import threading

        import torch

        checker._lock = threading.RLock()
        checker._cache = _BoundedLRUCache()
        checker._heuristic = HeuristicChecker()
        checker._is_sleeping = False
        checker._model_failed = True
        checker._model = None
        checker._tokenizer = None
        checker._config = None
        checker._vocab_size = 256
        checker._max_seq_len = 128
        checker._unpack_output = None
        checker._device = torch.device("cpu")

        checker.check("q", "a", "r")
        assert checker.cache_size() == 1
        checker.clear_cache()
        assert checker.cache_size() == 0

    def test_batch_heuristic(self):
        """Batch check with heuristic fallback."""
        checker = SharedCheckerModel.__new__(SharedCheckerModel)
        import threading

        import torch

        checker._lock = threading.RLock()
        checker._cache = _BoundedLRUCache()
        checker._heuristic = HeuristicChecker()
        checker._is_sleeping = False
        checker._model_failed = True
        checker._model = None
        checker._tokenizer = None
        checker._config = None
        checker._vocab_size = 256
        checker._max_seq_len = 128
        checker._unpack_output = None
        checker._device = torch.device("cpu")

        items = [
            {"question": "q1", "answer": "good answer with keywords", "requirements": "must have keywords"},
            {"question": "q2", "answer": "no", "requirements": "must explain in detail"},
            {"question": "q3", "answer": "another detailed answer here", "requirements": "must be detailed"},
        ]
        scores = checker.check_batch(items)
        assert len(scores) == 3
        for s in scores:
            assert 0 <= s <= 100

    def test_batch_empty(self):
        checker = SharedCheckerModel.__new__(SharedCheckerModel)
        import threading

        import torch

        checker._lock = threading.RLock()
        checker._cache = _BoundedLRUCache()
        checker._heuristic = HeuristicChecker()
        checker._is_sleeping = False
        checker._model_failed = True
        checker._model = None
        checker._tokenizer = None
        checker._config = None
        checker._vocab_size = 256
        checker._max_seq_len = 128
        checker._unpack_output = None
        checker._device = torch.device("cpu")

        assert checker.check_batch([]) == []

    def test_sleep_wake_noop_when_failed(self):
        """sleep/wake are no-ops when model failed to build."""
        checker = SharedCheckerModel.__new__(SharedCheckerModel)
        import threading

        import torch

        checker._lock = threading.RLock()
        checker._cache = _BoundedLRUCache()
        checker._heuristic = HeuristicChecker()
        checker._is_sleeping = False
        checker._model_failed = True
        checker._model = None
        checker._tokenizer = None
        checker._config = None
        checker._vocab_size = 256
        checker._max_seq_len = 128
        checker._unpack_output = None
        checker._device = torch.device("cpu")

        # These should not raise
        checker.sleep()
        checker.wake()
        assert checker.is_awake is True
        assert checker.is_llm_available is False


# ── Singleton tests ─────────────────────────────────────────────────────────


class TestSingleton:
    def test_reset_clears_instance(self):
        """reset_checker() clears the singleton reference."""
        reset_checker()
        import forge.evolution.checker_model as cm

        assert cm._checker_instance is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

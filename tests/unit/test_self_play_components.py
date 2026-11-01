"""Unit tests for self-play support components.

Covers the previously-orphaned/wired components and the I/O-focus revamp:
  - io_match: semantic output matching
  - ReplayBuffer: FOREVER forgetting-curve replay
  - DataQualityPipeline: multi-level dedup + QDC scoring
  - SelfPlayMonitor: failure-mode alerts

All CPU-only, fast, no model weights required.
"""

from forge.self_play.data_quality import DataQualityPipeline
from forge.self_play.io_match import io_match, io_similarity
from forge.self_play.monitoring import SelfPlayMonitor
from forge.self_play.replay_buffer import ReplayBuffer

# ── io_match ────────────────────────────────────────────────────────────────


class TestIoMatch:
    def test_exact_match(self):
        assert io_similarity("55", "55") == 1.0
        assert io_match("55", "55")

    def test_whitespace_tolerance(self):
        assert io_match("55", "  55\n")
        assert io_match("hello world", "hello   world")

    def test_numeric_tolerance(self):
        # Float formatting differences must not fail verification
        assert io_match("3.0", "3.00")
        assert io_match("0.30000000000000004", "0.3")
        assert io_similarity("55", "55.0") == 1.0

    def test_numeric_partial_credit(self):
        # Close but wrong number → partial credit, not a match
        s = io_similarity("100", "105")
        assert 0.0 < s < 0.99
        assert not io_match("100", "105")

    def test_unordered_multiset(self):
        assert io_similarity("hello world", "world hello") == 0.95

    def test_python_list_formatting(self):
        # Printed list vs space-separated values
        assert io_match("[1, 2, 3]", "1 2 3")
        assert io_match("(1, 2)", "1 2")

    def test_numeric_sequence(self):
        assert io_match("1 2 3", "1.0 2.0 3.0")

    def test_case_insensitive(self):
        assert io_similarity("True", "true") == 0.98

    def test_mismatch(self):
        assert io_similarity("abc", "xyz") == 0.0
        assert not io_match("abc", "xyz")

    def test_empty_and_none(self):
        assert io_similarity(None, "x") == 0.0
        assert io_similarity("", "") == 1.0
        assert io_similarity("x", "") == 0.0
        assert not io_match(None, "anything")

    def test_partial_overlap_below_threshold(self):
        # Half the tokens right → partial credit, never a match
        s = io_similarity("a b c d", "a b x y")
        assert 0.0 < s < 0.9
        assert not io_match("a b c d", "a b x y")


# ── ReplayBuffer ────────────────────────────────────────────────────────────


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=10)
        buf.add({"prompt": "p", "solution": "s"}, optimizer_magnitude=1.0)
        assert len(buf) == 1
        assert buf.cumulative_magnitude == 1.0

    def test_fifo_eviction(self):
        buf = ReplayBuffer(max_size=3)
        for i in range(5):
            buf.add({"idx": i})
        assert len(buf) == 3
        kept = sorted(s["idx"] for s in buf._buffer)
        assert kept == [2, 3, 4]

    def test_sample_returns_copies(self):
        buf = ReplayBuffer(max_size=10)
        buf.add({"prompt": "p", "solution": "s"})
        out = buf.sample(n=1, current_magnitude=0.0)
        assert len(out) == 1
        out[0]["prompt"] = "MUTATED"
        assert buf._buffer[0]["prompt"] == "p"

    def test_sample_empty(self):
        buf = ReplayBuffer()
        assert buf.sample(n=5, current_magnitude=0.0) == []

    def test_sample_respects_n(self):
        buf = ReplayBuffer(max_size=10)
        for i in range(4):
            buf.add({"idx": i})
        assert len(buf.sample(n=100, current_magnitude=0.0)) == 4

    def test_forgetting_curve_prefers_distant(self):
        # With a small tau, samples added at a very different model-time
        # should be replayed more often than recent ones.
        buf = ReplayBuffer(max_size=100, stability_constant=1.0)
        buf.add({"tag": "old"}, optimizer_magnitude=0.0)
        buf.update_magnitude(100.0)
        for _ in range(9):
            buf.add({"tag": "recent"}, optimizer_magnitude=0.0)
        counts = {"old": 0, "recent": 0}
        import numpy as np

        rng_state = np.random.get_state()
        np.random.seed(0)
        try:
            for _ in range(200):
                for s in buf.sample(n=1, current_magnitude=100.0):
                    counts[s["tag"]] += 1
        finally:
            np.random.set_state(rng_state)
        # The distant memory (due for replay per the forgetting curve) should
        # dominate sampling: weight ≈ 1 vs ≈ 1e-3 for fresh samples.
        assert counts["old"] > 100

    def test_retrieval_count_increments(self):
        buf = ReplayBuffer(max_size=10)
        buf.add({"x": 1})
        buf.sample(n=1, current_magnitude=0.0)
        assert buf._buffer[0]["retrieval_count"] == 1

    def test_stats(self):
        buf = ReplayBuffer(max_size=10)
        assert buf.stats()["size"] == 0
        buf.add({"x": 1}, optimizer_magnitude=2.0)
        st = buf.stats()
        assert st["size"] == 1
        assert st["cumulative_magnitude"] == 2.0


# ── DataQualityPipeline ─────────────────────────────────────────────────────


class TestDataQualityPipeline:
    def _sample(self, prompt, solution, quality=0.8):
        return {"prompt": prompt, "solution": solution, "quality": quality}

    def test_dedup_exact_removes_copies(self):
        dq = DataQualityPipeline()
        a = self._sample("compute fibonacci of ten", "def fib(n): pass")
        b = self._sample("compute fibonacci of ten", "def fib(n): pass")
        out = dq.dedup_exact([a, b])
        assert len(out) == 1

    def test_dedup_exact_keeps_distinct(self):
        dq = DataQualityPipeline()
        a = self._sample("sort a list of integers ascending", "print(sorted(x))")
        b = self._sample("reverse a string completely", "print(s[::-1])")
        assert len(dq.dedup_exact([a, b])) == 2

    def test_dedup_ast_structural(self):
        dq = DataQualityPipeline()
        # Same AST structure, different identifiers/comments
        a = self._sample("task a", "def f(x):\n    return x + 1\nprint(f(2))")
        b = self._sample("task b", "def g(y):\n    return y + 1\nprint(g(2))")
        out = dq.dedup_ast([a, b])
        assert len(out) == 1

    def test_dedup_ast_syntax_error_safe(self):
        dq = DataQualityPipeline()
        a = self._sample("t", "def broken(:\n")
        b = self._sample("t", "def also_broken(:\n")
        # Both unparseable → identical empty signatures → deduped
        assert len(dq.dedup_ast([a, b])) == 1

    def test_filter_difficulty_goldilocks(self):
        dq = DataQualityPipeline()
        samples = [self._sample(f"p{i}", "s") for i in range(4)]
        rates = [0.1, 0.4, 0.65, 0.95]
        out = dq.filter_difficulty(samples, rates)
        assert out == [samples[1], samples[2]]

    def test_filter_difficulty_mismatched_lengths(self):
        dq = DataQualityPipeline()
        samples = [self._sample("p", "s")]
        assert dq.filter_difficulty(samples, []) == samples

    def test_diversity_score_range(self):
        dq = DataQualityPipeline()
        samples = [
            self._sample("sort integers", "print(sorted(x))"),
            self._sample("reverse string", "print(s[::-1])"),
            self._sample("compute gcd", "import math\nprint(math.gcd(a,b))"),
        ]
        d = dq.compute_diversity_score(samples)
        assert 0.0 <= d <= 1.0

    def test_qdc_score_weights_quality(self):
        dq = DataQualityPipeline()
        hi = dq.compute_qdc_score(self._sample("p", "print(1)", quality=0.9))
        lo = dq.compute_qdc_score(self._sample("p", "print(1)", quality=0.1))
        assert hi > lo

    def test_run_pipeline_stats(self):
        dq = DataQualityPipeline()
        samples = [
            self._sample("compute fibonacci of ten", "def fib(n): pass"),
            self._sample("compute fibonacci of ten", "def fib(n): pass"),
            self._sample("reverse a string", "print(s[::-1])"),
        ]
        out, stats = dq.run_pipeline(samples)
        assert stats["n_input"] == 3
        assert len(out) == stats["n_after_difficulty"]
        assert "mean_qdc" in stats
        assert "diversity_score" in stats


# ── SelfPlayMonitor ─────────────────────────────────────────────────────────


class TestSelfPlayMonitor:
    def _healthy_step(self):
        return {
            "mean_reward": 0.5,
            "diversity_score": 0.9,
            "kl_divergence": 0.1,
            "intrinsic_reward": 0.5,
            "grounded_reward": 0.5,
            "advantage_collapse_rate": 0.0,
        }

    def test_no_alerts_when_healthy(self):
        m = SelfPlayMonitor(window_size=10)
        for _ in range(10):
            m.record_step(self._healthy_step())
        assert m.check_alerts() == []

    def test_acr_alert(self):
        m = SelfPlayMonitor(window_size=10)
        step = self._healthy_step()
        step["advantage_collapse_rate"] = 0.5
        for _ in range(10):
            m.record_step(step)
        alerts = m.check_alerts()
        assert any(a["metric"] == "ACR" for a in alerts)

    def test_diversity_collapse_alert(self):
        m = SelfPlayMonitor(window_size=10)
        step = self._healthy_step()
        step["diversity_score"] = 0.2
        for _ in range(10):
            m.record_step(step)
        assert any(a["metric"] == "diversity" for a in m.check_alerts())

    def test_kl_alert(self):
        m = SelfPlayMonitor(window_size=10)
        step = self._healthy_step()
        step["kl_divergence"] = 50.0
        for _ in range(10):
            m.record_step(step)
        assert any(a["metric"] == "kl_div" for a in m.check_alerts())

    def test_ig_gap_critical(self):
        m = SelfPlayMonitor(window_size=10)
        step = self._healthy_step()
        step["intrinsic_reward"] = 0.9
        step["grounded_reward"] = 0.2
        for _ in range(10):
            m.record_step(step)
        alerts = m.check_alerts()
        ig = [a for a in alerts if a["metric"] == "IG_gap"]
        assert ig and ig[0]["level"] == "critical"

    def test_summary_keys(self):
        m = SelfPlayMonitor(window_size=10)
        m.record_step(self._healthy_step())
        s = m.summary()
        for key in ("step_count", "mean_reward", "diversity_score", "kl_divergence", "intrinsic_ground_gap"):
            assert key in s

    def test_plot_data_lengths(self):
        m = SelfPlayMonitor(window_size=10)
        for _ in range(5):
            m.record_step(self._healthy_step())
        pd = m.plot_data()
        assert len(pd["mean_reward"]) == 5
        assert len(pd["steps"]) == 5

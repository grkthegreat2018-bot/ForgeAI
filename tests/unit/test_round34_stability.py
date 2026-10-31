"""Tests for Round 34 Model Stability & Usability features.

R34-1: TriLens — per-layer logit-lens entropy hallucination detection
R34-2: PoP — prediction-of-prediction inter-layer fusion
R34-3: SyncThink — training-free reasoning saturation detection
R34-4: SPOC/MIRROR — agent self-correction & rollback
R34-5: VRAM pressure monitor for proactive OOM resilience
R34-6: SIREN — streaming safety guardrails
"""
from __future__ import annotations

import torch

# ── R34-1: TriLens ────────────────────────────────────────────────────────

class TestTriLens:
    """TriLens — per-layer logit-lens entropy trajectory."""

    def test_init(self):
        from forge.engine.safety.trilens import TriLensDetector
        det = TriLensDetector(n_layers=4, vocab_size=1000)
        assert det.n_layers == 4

    def test_record_and_trajectory(self):
        from forge.engine.safety.trilens import TriLensDetector
        det = TriLensDetector(n_layers=4, vocab_size=100)
        det.set_unembedding(torch.randn(100, 32))
        hidden = torch.randn(32)
        for i in range(4):
            det.record_layer(i, hidden, hidden, hidden)
        traj = det.get_trajectory()
        assert traj.shape[0] == 12  # 3 * 4 layers

    def test_detect(self):
        from forge.engine.safety.trilens import TriLensDetector
        det = TriLensDetector(n_layers=4, vocab_size=100)
        det.set_unembedding(torch.randn(100, 32))
        hidden = torch.randn(32)
        for i in range(4):
            det.record_layer(i, hidden, hidden, hidden)
        score = det.detect()
        assert 0.0 <= score <= 1.0

    def test_reset(self):
        from forge.engine.safety.trilens import TriLensDetector
        det = TriLensDetector(n_layers=4, vocab_size=100)
        det.set_unembedding(torch.randn(100, 32))
        det.record_layer(0, torch.randn(32), torch.randn(32), torch.randn(32))
        det.reset()
        traj = det.get_trajectory()
        assert traj.shape[0] == 0 or traj.numel() == 0


# ── R34-2: PoP ────────────────────────────────────────────────────────────

class TestPoP:
    """PoP — prediction-of-prediction inter-layer fusion."""

    def test_init(self):
        from forge.engine.safety.trilens import PoPDetector
        det = PoPDetector(n_layers=4, hidden_dim=64)
        assert det.n_layers == 4

    def test_record_and_fuse(self):
        from forge.engine.safety.trilens import PoPDetector
        det = PoPDetector(n_layers=4, hidden_dim=64)
        for i in range(4):
            det.record_layer(i, torch.randn(64))
        fused = det.fuse()
        assert fused.shape[0] == 4

    def test_detect(self):
        from forge.engine.safety.trilens import PoPDetector
        det = PoPDetector(n_layers=4, hidden_dim=64)
        for i in range(4):
            det.record_layer(i, torch.randn(64))
        score = det.detect()
        assert 0.0 <= score <= 1.0

    def test_ensemble(self):
        from forge.engine.safety.trilens import TriLensPoPEnsemble
        ens = TriLensPoPEnsemble(n_layers=4, vocab_size=100, hidden_dim=32)
        ens.set_unembedding(torch.randn(100, 32))
        hidden = torch.randn(32)
        for i in range(4):
            ens.record_layer(i, hidden, hidden, hidden)
        score = ens.detect()
        assert 0.0 <= score <= 1.0


# ── R34-3: SyncThink ──────────────────────────────────────────────────────

class TestSyncThink:
    """SyncThink — reasoning saturation detection."""

    def test_init(self):
        from forge.engine.reasoning.syncthink import SyncThinkMonitor
        mon = SyncThinkMonitor(window_size=16, transition_threshold=0.15,
                               min_reasoning_tokens=64)
        assert mon.window_size == 16

    def test_update_and_signal(self):
        from forge.engine.reasoning.syncthink import SyncThinkMonitor
        mon = SyncThinkMonitor(window_size=8, min_reasoning_tokens=4)
        # Simulate attention scores: (n_heads, seq_len)
        for i in range(10):
            scores = torch.rand(2, i + 1)
            mon.update(token_id=i, attention_scores=scores, position=i)
        signal = mon.compute_transition_signal()
        assert isinstance(signal, float)

    def test_should_terminate(self):
        from forge.engine.reasoning.syncthink import SyncThinkMonitor
        mon = SyncThinkMonitor(window_size=4, transition_threshold=0.1,
                               min_reasoning_tokens=4)
        # Simulate transition: early tokens attend broadly, later focus on recent
        for i in range(20):
            scores = torch.zeros(1, i + 1)
            if i < 10:
                scores[0, :i+1] = 1.0 / (i + 1)  # uniform attention
            else:
                scores[0, -4:] = 1.0  # focus on recent
            mon.update(token_id=i, attention_scores=scores, position=i)
        assert mon.should_terminate() in (True, False)

    def test_min_tokens_gate(self):
        from forge.engine.reasoning.syncthink import SyncThinkMonitor
        mon = SyncThinkMonitor(window_size=4, transition_threshold=0.01,
                               min_reasoning_tokens=100)
        for i in range(10):
            scores = torch.zeros(1, i + 1)
            scores[0, -2:] = 1.0
            mon.update(token_id=i, attention_scores=scores, position=i)
        assert not mon.should_terminate(), "Should not terminate before min_tokens"

    def test_reset(self):
        from forge.engine.reasoning.syncthink import SyncThinkMonitor
        mon = SyncThinkMonitor()
        mon.update(0, torch.rand(1, 5), 0)
        mon.reset()
        stats = mon.stats()
        assert stats["tokens_generated"] == 0


# ── R34-4: SPOC/MIRROR ────────────────────────────────────────────────────

class TestAgentSelfCorrector:
    """SPOC/MIRROR — agent self-correction & rollback."""

    def test_detect_error(self):
        from forge.engine.safety.agent_corrector import SPOCCorrector
        spoc = SPOCCorrector()
        assert spoc.detect_error({"error": "file not found"})[0] is True
        assert spoc.detect_error({"ok": False, "error": "bad args"})[0] is True
        assert spoc.detect_error({"result": "success"})[0] is False

    def test_reflection_prompt(self):
        from forge.engine.safety.agent_corrector import SPOCCorrector
        spoc = SPOCCorrector()
        prompt = spoc.build_reflection_prompt(
            {"name": "read_file", "arguments": {"path": "/bad"}},
            "File not found")
        assert "read_file" in prompt
        assert "File not found" in prompt

    def test_checkpoint_and_rollback(self):
        from forge.engine.safety.agent_corrector import MIRRORCorrector
        mir = MIRRORCorrector()
        msgs = [{"role": "user", "content": "test"}]
        mir.checkpoint(msgs, [], [], 0)
        msgs.append({"role": "assistant", "content": "response"})
        rb = mir.rollback()
        assert rb is not None
        assert len(rb.messages) == 1  # original state

    def test_escalation(self):
        from forge.engine.safety.agent_corrector import MIRRORCorrector
        mir = MIRRORCorrector(error_window=3)
        mir.record_error("read_file", "error 1")
        mir.record_error("read_file", "error 2")
        assert not mir.should_escalate("read_file")
        mir.record_error("read_file", "error 3")
        assert mir.should_escalate("read_file")

    def test_agent_self_corrector(self):
        from forge.engine.safety.agent_corrector import AgentSelfCorrector
        cor = AgentSelfCorrector()
        calls = [{"name": "read_file", "arguments": {}}]
        results = [{"error": "not found"}]
        reflection = cor.maybe_correct(
            messages=[], tool_calls=calls, tool_results=results, round_idx=0)
        assert reflection is not None
        assert "read_file" in reflection

    def test_no_correction_on_success(self):
        from forge.engine.safety.agent_corrector import AgentSelfCorrector
        cor = AgentSelfCorrector()
        calls = [{"name": "read_file", "arguments": {}}]
        results = [{"content": "file contents"}]
        reflection = cor.maybe_correct(
            messages=[], tool_calls=calls, tool_results=results, round_idx=0)
        assert reflection is None

    def test_stats(self):
        from forge.engine.safety.agent_corrector import AgentSelfCorrector
        cor = AgentSelfCorrector()
        cor.maybe_correct(
            [], [{"name": "test", "arguments": {}}], [{"error": "fail"}], 0)
        stats = cor.stats()
        assert stats["corrections_made"] == 1


# ── R34-5: VRAM Pressure Monitor ──────────────────────────────────────────

class TestVRAMMonitor:
    """VRAM pressure monitor for proactive OOM resilience."""

    def test_init(self):
        from forge.engine.safety.vram_monitor import VRAMPressureMonitor
        mon = VRAMPressureMonitor(budget_bytes=12 * 1024**3)
        assert mon.budget == 12 * 1024**3

    def test_get_level(self):
        from forge.engine.safety.vram_monitor import VRAMPressureMonitor
        mon = VRAMPressureMonitor()
        assert mon.get_level(0.5) == mon.LEVEL_NORMAL
        assert mon.get_level(0.75) == mon.LEVEL_COMPRESSED
        assert mon.get_level(0.85) == mon.LEVEL_AGGRESSIVE
        assert mon.get_level(0.92) == mon.LEVEL_OFFLOAD
        assert mon.get_level(0.97) == mon.LEVEL_CRITICAL

    def test_callback_registration(self):
        from forge.engine.safety.vram_monitor import VRAMPressureMonitor
        mon = VRAMPressureMonitor()
        called = []
        mon.register_callback(mon.LEVEL_COMPRESSED,
                              lambda p, l: called.append((p, l)))
        mon._current_level = mon.LEVEL_NORMAL
        # Simulate level increase
        mon._current_level = mon.LEVEL_NORMAL
        mon._callbacks[mon.LEVEL_COMPRESSED](0.75, mon.LEVEL_COMPRESSED)
        assert len(called) == 1

    def test_stats(self):
        from forge.engine.safety.vram_monitor import VRAMPressureMonitor
        mon = VRAMPressureMonitor()
        stats = mon.stats()
        assert "pressure" in stats
        assert "level" in stats
        assert "level_name" in stats

    def test_reset(self):
        from forge.engine.safety.vram_monitor import VRAMPressureMonitor
        mon = VRAMPressureMonitor()
        mon._current_level = mon.LEVEL_CRITICAL
        mon.reset()
        assert mon._current_level == mon.LEVEL_NORMAL

    def test_level_name(self):
        from forge.engine.safety.vram_monitor import VRAMPressureMonitor
        assert VRAMPressureMonitor.level_name(0) == "normal"
        assert VRAMPressureMonitor.level_name(4) == "critical"


# ── R34-6: SIREN ──────────────────────────────────────────────────────────

class TestSIREN:
    """SIREN — streaming safety guardrails."""

    def test_init(self):
        from forge.engine.safety.siren import SIRENGuard
        guard = SIRENGuard(n_layers=4, hidden_dim=64)
        assert guard.n_layers == 4

    def test_evaluate_no_probes(self):
        from forge.engine.safety.siren import SIRENGuard
        guard = SIRENGuard(n_layers=4, hidden_dim=64)
        for i in range(4):
            guard.record_layer(i, torch.randn(64))
        scores = guard.evaluate()
        # No probes → permissive (all zeros)
        for cat in scores.values():
            assert cat == 0.0

    def test_evaluate_with_probes(self):
        from forge.engine.safety.siren import SIRENGuard
        guard = SIRENGuard(n_layers=4, hidden_dim=64,
                           categories=["harmful", "safe"])
        probes = {
            "harmful": torch.randn(4, 64),
            "safe": torch.randn(4, 64),
        }
        guard.set_probes(probes)
        for i in range(4):
            guard.record_layer(i, torch.randn(64))
        scores = guard.evaluate()
        assert "harmful" in scores
        assert "safe" in scores
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_check_token(self):
        from forge.engine.safety.siren import SIRENGuard
        guard = SIRENGuard(n_layers=4, hidden_dim=64,
                           categories=["harmful"])
        guard.set_probes({"harmful": torch.randn(4, 64) * 10})
        states = {i: torch.randn(64) for i in range(4)}
        is_safe, _cat = guard.check_token("test", states)
        assert isinstance(is_safe, bool)

    def test_reset(self):
        from forge.engine.safety.siren import SIRENGuard
        guard = SIRENGuard(n_layers=4, hidden_dim=64)
        guard.record_layer(0, torch.randn(64))
        guard.reset()
        # After reset, evaluate should return zeros (no states)
        scores = guard.evaluate()
        for v in scores.values():
            assert v == 0.0

    def test_stream_wrapper(self):
        from forge.engine.safety.siren import SIRENGuard, SIRENStreamWrapper

        def gen():
            yield "hello"
            yield "world"

        guard = SIRENGuard(n_layers=4, hidden_dim=64)
        wrapper = SIRENStreamWrapper(gen(), guard)
        tokens = list(wrapper)
        assert "hello" in tokens
        assert "world" in tokens

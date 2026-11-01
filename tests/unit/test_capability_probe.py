"""A14: frozen capability probe + status/history wiring tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from forge.self_play.infinite_loop import _CAPABILITY_PROBE, InfiniteSelfPlayLoop, LoopConfig


def _loop(enabled: bool = True) -> InfiniteSelfPlayLoop:
    loop = InfiniteSelfPlayLoop("dummy_ckpt", LoopConfig(capability_probe=enabled))
    return loop


def test_probe_scores_frozen_set_with_mock_engine():
    loop = _loop()
    engine = MagicMock()
    # Return the expected answer for whichever probe item is asked —
    # inspect the prompt to echo the gold string back.
    answers = {q["prompt"]: q["answer"] for q in _CAPABILITY_PROBE}
    engine.generate.side_effect = lambda prompt, **kw: answers[prompt]
    loop._engine = engine

    result = loop._capability_probe()

    assert result["n"] == len(_CAPABILITY_PROBE) == 8
    assert result["correct"] == 8
    assert result["score"] == 1.0


def test_probe_counts_wrong_answers():
    loop = _loop()
    engine = MagicMock()
    engine.generate.return_value = "I don't know."
    loop._engine = engine

    result = loop._capability_probe()

    assert result["correct"] == 0
    assert result["score"] == 0.0


def test_probe_disabled_returns_none():
    loop = _loop(enabled=False)
    loop._engine = MagicMock()
    assert loop._capability_probe() is None


def test_probe_no_engine_returns_none():
    loop = _loop()
    loop._engine = None
    assert loop._capability_probe() is None

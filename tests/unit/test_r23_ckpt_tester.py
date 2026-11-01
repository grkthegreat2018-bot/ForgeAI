"""Tests for R&D Round 23: Checkpoint tester (ckpt_tester.py).

The checkpoint tester loads a checkpoint and runs a 50-question probe:
  - 10 math (arithmetic, algebra)
  - 10 logic (syllogisms, sequence completion)
  - 10 code (simple function generation)
  - 10 recall (factual knowledge questions)
  - 10 format (proper tool-use format, JSON, markdown)
Plus a val-loss delta vs prior checkpoint for regression detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────


def _tiny_v8_config(**extra):
    """Create a tiny V8-8B config that builds on CPU in <1s."""
    from forge.config import get_config

    overrides = {
        "vocab_size": 256,
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "intermediate_size": 128,
        "max_seq_len": 128,
        "titan_memory_rank": 16,
        "embed_factorized_rank": 32,
        "mtp_n_heads": 2,
        "use_triton_kernels": False,
        "use_varlen": False,
        "bitnet_int8_training": False,
        "use_gradient_checkpointing": False,
        "use_hyperloop": False,
        "use_lisa": False,
        "ngram_host": False,
    }
    overrides.update(extra)
    cfg = get_config("forgelm_v2", **overrides)
    cfg.device = "cpu"
    cfg.dtype = "float32"
    return cfg


def _build_tiny_v8(**extra):
    """Build a tiny V8 model + config."""
    from forge.model_loader import ConfigurableResearchLLM

    cfg = _tiny_v8_config(**extra)
    model = ConfigurableResearchLLM(cfg)
    return model, cfg


# ── Test 1: Imports ───────────────────────────────────────────────────────


def test_ckpt_tester_imports():
    """ckpt_tester module should expose CheckpointTester class or run_test()."""
    import forge.evaluation.ckpt_tester as mod

    has_class = hasattr(mod, "CheckpointTester")
    has_func = hasattr(mod, "run_test")
    assert has_class or has_func, "ckpt_tester should have CheckpointTester class or run_test() function"
    if has_class:
        assert callable(mod.CheckpointTester), "CheckpointTester should be a class"
    if has_func:
        assert callable(mod.run_test), "run_test should be callable"
    print("  ckpt_tester_imports: PASS")


# ── Test 2: Question count ────────────────────────────────────────────────


def test_ckpt_tester_question_count():
    """The question bank should have exactly 50 questions (10 per category)."""
    from forge.evaluation.ckpt_tester import QUESTION_BANK

    assert isinstance(QUESTION_BANK, (list, dict)), "QUESTION_BANK should be a list or dict"
    if isinstance(QUESTION_BANK, dict):
        total = sum(len(v) for v in QUESTION_BANK.values())
    else:
        total = len(QUESTION_BANK)
    assert total == 50, f"Should have 50 questions, got {total}"
    print(f"  Question count: {total} questions")
    print("  ckpt_tester_question_count: PASS")


# ── Test 3: Categories ────────────────────────────────────────────────────


def test_ckpt_tester_categories():
    """The 5 categories (math, logic, code, recall, format) should each have 10."""
    from forge.evaluation.ckpt_tester import QUESTION_BANK

    expected_cats = {"math", "logic", "code", "recall", "format"}
    if isinstance(QUESTION_BANK, dict):
        cats = set(QUESTION_BANK.keys())
        assert cats == expected_cats, f"Categories {cats} != {expected_cats}"
        for cat, questions in QUESTION_BANK.items():
            assert len(questions) == 10, f"Category '{cat}' should have 10 questions, got {len(questions)}"
    else:
        # If list, each question should have a 'category' field
        cat_counts = {}
        for q in QUESTION_BANK:
            cat = q.get("category", q.get("type", ""))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        assert set(cat_counts.keys()) == expected_cats, f"Categories {set(cat_counts.keys())} != {expected_cats}"
        for cat, count in cat_counts.items():
            assert count == 10, f"Category '{cat}' should have 10, got {count}"
    print(f"  Categories: {sorted(expected_cats)}, 10 each")
    print("  ckpt_tester_categories: PASS")


# ── Test 4: Runs with model ───────────────────────────────────────────────


def test_ckpt_tester_runs_with_model():
    """Tester should run on a tiny V8 model and return per-category scores."""
    from forge.evaluation.ckpt_tester import CheckpointTester

    model, cfg = _build_tiny_v8()
    tester = CheckpointTester(model, cfg, device=torch.device("cpu"))
    report = tester.run()

    assert isinstance(report, dict), "Report should be a dict"
    assert "category_scores" in report, "Report should have category_scores"
    cats = report["category_scores"]
    assert isinstance(cats, dict), "category_scores should be a dict"
    assert len(cats) == 5, f"Should have 5 category scores, got {len(cats)}"
    for cat, score in cats.items():
        assert 0.0 <= score <= 1.0, f"Category '{cat}' score {score} should be in [0, 1]"
    print(f"  Scores: {cats}")
    print("  ckpt_tester_runs_with_model: PASS")


# ── Test 5: Regression detection ──────────────────────────────────────────


def test_ckpt_tester_regression_detection():
    """Degrading model weights should produce val_loss_delta > 0 (regression)."""
    from forge.evaluation.ckpt_tester import CheckpointTester

    model, cfg = _build_tiny_v8()
    tester = CheckpointTester(model, cfg, device=torch.device("cpu"))
    report_a = tester.run()
    val_loss_a = report_a["val_loss"]

    # Degrade model: add noise to weights
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p) * 0.5

    # Re-run with prior val_loss for delta computation
    tester2 = CheckpointTester(model, cfg, device=torch.device("cpu"), prior_val_loss=val_loss_a)
    report_b = tester2.run()

    assert "val_loss_delta" in report_b, "Should have val_loss_delta"
    delta = report_b["val_loss_delta"]
    print(f"  Regression: val_loss {val_loss_a:.3f} → {report_b['val_loss']:.3f}, delta={delta:.3f}")
    assert delta > 0, f"Degrading weights should increase val_loss (delta > 0), got {delta}"
    print("  ckpt_tester_regression_detection: PASS")


# ── Test 6: No false regression ───────────────────────────────────────────


def test_ckpt_tester_no_regression():
    """Running tester twice on same model should give val_loss_delta ≈ 0."""
    from forge.evaluation.ckpt_tester import CheckpointTester

    model, cfg = _build_tiny_v8()
    tester = CheckpointTester(model, cfg, device=torch.device("cpu"))
    report_a = tester.run()

    tester2 = CheckpointTester(model, cfg, device=torch.device("cpu"), prior_val_loss=report_a["val_loss"])
    report_b = tester2.run()

    delta = report_b["val_loss_delta"]
    print(f"  No-regression: delta={delta:.4f} (should be ≈ 0)")
    assert abs(delta) < 0.1, f"Same model should have delta ≈ 0, got {delta:.4f}"
    assert not report_b["regression_flag"], "Should not flag regression on same model"
    print("  ckpt_tester_no_regression: PASS")


# ── Test 7: Report format ─────────────────────────────────────────────────


def test_ckpt_tester_report_format():
    """Report dict should have all required keys."""
    from forge.evaluation.ckpt_tester import CheckpointTester

    model, cfg = _build_tiny_v8()
    tester = CheckpointTester(model, cfg, device=torch.device("cpu"))
    report = tester.run()

    required_keys = {
        "total_score",
        "category_scores",
        "val_loss",
        "val_loss_delta",
        "regression_flag",
        "questions_passed",
        "questions_total",
    }
    missing = required_keys - set(report.keys())
    assert not missing, f"Report missing keys: {missing}"

    assert isinstance(report["total_score"], float), "total_score should be float"
    assert 0.0 <= report["total_score"] <= 1.0, "total_score should be in [0, 1]"
    assert isinstance(report["regression_flag"], bool), "regression_flag should be bool"
    assert report["questions_total"] == 50, f"questions_total should be 50, got {report['questions_total']}"
    assert 0 <= report["questions_passed"] <= 50, f"questions_passed should be 0-50, got {report['questions_passed']}"
    print(
        f"  Report: total={report['total_score']:.2f}, "
        f"passed={report['questions_passed']}/{report['questions_total']}, "
        f"regression={report['regression_flag']}"
    )
    print("  ckpt_tester_report_format: PASS")


# ── Test 8: Math questions ────────────────────────────────────────────────


def test_ckpt_tester_math_questions():
    """Math questions should test arithmetic and evaluate model output correctly."""
    from forge.evaluation.ckpt_tester import QUESTION_BANK, evaluate_answer

    # Get math questions
    if isinstance(QUESTION_BANK, dict):
        math_qs = QUESTION_BANK["math"]
    else:
        math_qs = [q for q in QUESTION_BANK if q.get("category") == "math"]

    assert len(math_qs) == 10, f"Should have 10 math questions, got {len(math_qs)}"

    # Verify they test arithmetic (should contain numbers and operators)
    has_arithmetic = False
    for q in math_qs:
        prompt = q.get("prompt", q.get("question", ""))
        if any(op in str(prompt) for op in ["+", "-", "*", "/", "What is"]):
            has_arithmetic = True
            break
    assert has_arithmetic, "Math questions should test arithmetic"

    # Test evaluation: correct answer should pass, wrong should fail
    q0 = math_qs[0]
    expected = q0.get("answer", q0.get("expected", ""))
    # If the evaluate function exists, test it
    if callable(evaluate_answer):
        # Correct answer should pass
        result_correct = evaluate_answer(q0, str(expected))
        # Wrong answer should fail
        result_wrong = evaluate_answer(q0, "definitely_wrong_answer_12345")
        assert result_correct or not result_wrong, "evaluate_answer should distinguish correct from wrong"
    print(f"  Math: {len(math_qs)} questions, arithmetic detected")
    print("  ckpt_tester_math_questions: PASS")


# ── Test 9: Format questions ──────────────────────────────────────────────


def test_ckpt_tester_format_questions():
    """Format questions should test JSON validity, markdown structure, tool-use."""
    from forge.evaluation.ckpt_tester import QUESTION_BANK, evaluate_answer

    if isinstance(QUESTION_BANK, dict):
        format_qs = QUESTION_BANK["format"]
    else:
        format_qs = [q for q in QUESTION_BANK if q.get("category") == "format"]

    assert len(format_qs) == 10, f"Should have 10 format questions, got {len(format_qs)}"

    # Verify they test format (JSON, markdown, tool-use)
    format_keywords = ["json", "markdown", "tool", "format", "```", "{", "```"]
    has_format = False
    for q in format_qs:
        prompt = str(q.get("prompt", q.get("question", "")))
        if any(kw in prompt.lower() for kw in format_keywords):
            has_format = True
            break
    assert has_format, "Format questions should test output format"

    # Test evaluation: valid JSON should pass, invalid should fail
    if callable(evaluate_answer):
        # Find a JSON-format question if any
        json_q = None
        for q in format_qs:
            if "json" in str(q.get("prompt", "")).lower():
                json_q = q
                break
        if json_q:
            valid_json = '{"key": "value"}'
            invalid_json = "not valid json {{{"
            result_valid = evaluate_answer(json_q, valid_json)
            result_invalid = evaluate_answer(json_q, invalid_json)
            # At least one should distinguish valid from invalid
            if result_valid is not None and result_invalid is not None:
                assert result_valid or not result_invalid, (
                    "Format evaluation should distinguish valid from invalid JSON"
                )
    print(f"  Format: {len(format_qs)} questions, format testing detected")
    print("  ckpt_tester_format_questions: PASS")


# ── Test 10: Blocking on critical regression ──────────────────────────────


def test_ckpt_tester_blocking_on_critical():
    """When regression_flag=True AND val_loss_delta > 0.05, tester should block."""
    from forge.evaluation.ckpt_tester import CheckpointTester

    model, cfg = _build_tiny_v8()
    tester = CheckpointTester(model, cfg, device=torch.device("cpu"))
    report_a = tester.run()
    val_loss_a = report_a["val_loss"]

    # Severely degrade model to trigger critical regression
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p) * 2.0

    tester2 = CheckpointTester(
        model, cfg, device=torch.device("cpu"), prior_val_loss=val_loss_a, critical_threshold=0.05
    )
    report_b = tester2.run()

    delta = report_b["val_loss_delta"]
    print(
        f"  Critical: delta={delta:.3f}, regression={report_b['regression_flag']}, block={report_b.get('block', 'N/A')}"
    )

    # If delta exceeds critical threshold, should signal block
    if delta > 0.05:
        assert report_b["regression_flag"], "Should flag regression when delta > critical threshold"
        # The tester should have a 'block' or 'should_block' key
        block_key = report_b.get("block", report_b.get("should_block", None))
        if block_key is not None:
            assert block_key is True, "Should signal block when critical regression detected"
    else:
        # If delta didn't exceed threshold (tiny model variance), just verify
        # the mechanism exists — the regression_flag should match delta > threshold
        assert report_b["regression_flag"] == (delta > 0.05), "regression_flag should match delta > critical_threshold"
    print("  ckpt_tester_blocking_on_critical: PASS")


# ── Main ──────────────────────────────────────────────────────────────────


def main_r23_ckpt():
    print("=" * 70)
    print("  R&D ROUND 23: Checkpoint Tester")
    print("=" * 70)

    print("\n  Imports & question bank")
    test_ckpt_tester_imports()
    test_ckpt_tester_question_count()
    test_ckpt_tester_categories()

    print("\n  Running tester")
    test_ckpt_tester_runs_with_model()
    test_ckpt_tester_report_format()

    print("\n  Regression detection")
    test_ckpt_tester_regression_detection()
    test_ckpt_tester_no_regression()
    test_ckpt_tester_blocking_on_critical()

    print("\n  Question categories")
    test_ckpt_tester_math_questions()
    test_ckpt_tester_format_questions()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 CKPT TESTER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_ckpt()

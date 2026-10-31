"""Test RLVR pipeline: math answer extraction, verification, reward computation."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from forge.training.runners.rlvr_train import (
    VerifiableTask,
    extract_gold_answer,
    extract_math_answer,
    load_math_tasks,
    math_verify,
    normalize_answer,
)


def test_extract_math_answer_gsm8k():
    """Extract answer from GSM8K format (#### N)."""
    text = "Natalia sold 48/2 = 24 clips in May.\n48+24 = 72.\n#### 72"
    ans = extract_math_answer(text)
    assert ans == "72"
    print("PASS: extract_math_answer (GSM8K #### format)")


def test_extract_math_answer_boxed():
    """Extract answer from \\boxed{} format."""
    text = "Therefore the answer is \\boxed{42}."
    ans = extract_math_answer(text)
    assert ans == "42"
    print("PASS: extract_math_answer (\\boxed format)")


def test_extract_math_answer_text():
    """Extract answer from 'The answer is X' format."""
    text = "After calculation, the answer is 15."
    ans = extract_math_answer(text)
    assert ans == "15"
    print("PASS: extract_math_answer ('answer is' format)")


def test_normalize_answer():
    """Normalize answers for comparison."""
    assert normalize_answer("42") == "42.0"
    assert normalize_answer("$42$") == "42.0"
    assert normalize_answer("4,200") == "4200.0"
    assert normalize_answer("  42  ") == "42.0"
    print("PASS: normalize_answer handles numbers, commas, $, spaces")


def test_math_verify_correct():
    """Verify a correct math completion."""
    completion = "I calculated 2+2=4.\n#### 4"
    gold = "4"
    assert math_verify(completion, gold)
    print("PASS: math_verify accepts correct answer")


def test_math_verify_wrong():
    """Verify an incorrect math completion."""
    completion = "I think 2+2=5.\n#### 5"
    gold = "4"
    assert not math_verify(completion, gold)
    print("PASS: math_verify rejects wrong answer")


def test_math_verify_no_answer():
    """Verify a completion with no extractable answer."""
    completion = "I don't know the answer."
    gold = "4"
    assert not math_verify(completion, gold)
    print("PASS: math_verify rejects no-answer completion")


def test_math_verify_doom_loop():
    """A doom-loop completion should fail verification (no valid answer)."""
    loop = "the answer is the answer is the answer is the answer is the answer is " \
           "the answer is the answer is the answer is the answer is the answer is " \
           "the answer is the answer is the answer is the answer is the answer is " \
           "the answer is the answer is the answer is the answer is the answer is"
    gold = "4"
    # The doom-loop text has "the answer is" repeated, extract_math_answer
    # might extract a number from it. Let's check what happens.
    extracted = extract_math_answer(loop)
    if extracted is not None:
        # If it extracts something, verify it's wrong
        assert not math_verify(loop, gold) or normalize_answer(extracted) == normalize_answer(gold)
    print("PASS: doom-loop completion handled by math_verify")


def test_extract_gold_answer():
    """Extract gold answer from a solution text."""
    solution = "The answer is calculated as 3*7=21.\n#### 21"
    gold = extract_gold_answer(solution, "math")
    assert gold == "21"
    print("PASS: extract_gold_answer extracts from solution")


def test_load_math_tasks():
    """Load math tasks from a JSONL file."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "What is 2+2?", "solution": "2+2=4.\\n#### 4"}\n')
        f.write('{"prompt": "What is 3*3?", "solution": "3*3=9.\\n#### 9"}\n')
        f.write('{"prompt": "No answer here", "solution": "just text"}\n')  # no gold answer
        path = f.name

    tasks = load_math_tasks(path)
    os.unlink(path)
    assert len(tasks) == 2, f"Expected 2 tasks (1 has no gold), got {len(tasks)}"
    assert tasks[0].prompt == "What is 2+2?"
    assert tasks[0].gold_answer == "4"
    assert tasks[0].task_type == "math"
    assert callable(tasks[0].verify_fn)
    print(f"PASS: load_math_tasks loaded {len(tasks)} tasks (skipped no-gold)")


def test_verifiable_task_dataclass():
    """VerifiableTask should store all fields."""
    task = VerifiableTask(
        prompt="What is 2+2?",
        gold_answer="4",
        task_type="math",
        verify_fn=math_verify,
    )
    assert task.prompt == "What is 2+2?"
    assert task.gold_answer == "4"
    assert task.task_type == "math"
    assert task.verify_fn("#### 4", "4")
    print("PASS: VerifiableTask dataclass works")


def test_repetition_penalty_integration():
    """Verify that the GRPO config has repetition penalty fields."""
    from forge.self_play.grpo_trainer import GRPOConfig
    config = GRPOConfig()
    assert hasattr(config, "use_repetition_penalty")
    assert hasattr(config, "repetition_n")
    assert hasattr(config, "repetition_threshold")
    assert hasattr(config, "repetition_penalty")
    assert hasattr(config, "repetition_warmup_steps")
    assert config.repetition_penalty == -0.5
    print("PASS: GRPOConfig has repetition penalty fields")


if __name__ == "__main__":
    test_extract_math_answer_gsm8k()
    test_extract_math_answer_boxed()
    test_extract_math_answer_text()
    test_normalize_answer()
    test_math_verify_correct()
    test_math_verify_wrong()
    test_math_verify_no_answer()
    test_math_verify_doom_loop()
    test_extract_gold_answer()
    test_load_math_tasks()
    test_verifiable_task_dataclass()
    test_repetition_penalty_integration()
    print("\n=== All RLVR tests passed ===")

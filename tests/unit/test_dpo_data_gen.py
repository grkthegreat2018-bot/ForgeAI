"""Test DPO preference data generation: candidate scoring, doom-loop detection, pair construction."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from forge.training.runners.curriculum_sft import ngram_repetition_ratio
from forge.training.runners.dpo_data_gen import (
    JUDGE_SYSTEM_PROMPT,
    Candidate,
    construct_preference_pair,
    load_prompts,
)


def test_candidate_dataclass():
    """Candidate dataclass should store all required fields."""
    c = Candidate(text="hello world", temperature=0.8, is_greedy=False)
    assert c.text == "hello world"
    assert c.temperature == 0.8
    assert not c.is_greedy
    assert c.judge_score == 0.0
    assert not c.is_loop
    print("PASS: Candidate dataclass works")


def test_construct_pair_normal():
    """Normal case: best non-looping chosen, worst rejected."""
    candidates = [
        Candidate(text="Good answer that is correct and clear.", temperature=0.8, judge_score=8.0),
        Candidate(text="Bad answer that is wrong.", temperature=0.8, judge_score=3.0),
        Candidate(text="Mediocre answer.", temperature=0.8, judge_score=5.0),
    ]
    pair = construct_preference_pair(candidates, "What is 2+2?")
    assert pair is not None
    assert pair["chosen"] == "Good answer that is correct and clear."
    assert pair["rejected"] == "Bad answer that is wrong."
    assert pair["chosen_score"] == 8.0
    assert pair["rejected_score"] == 3.0
    assert not pair["chosen_is_loop"]
    assert not pair["rejected_is_loop"]
    print("PASS: construct_preference_pair (normal case)")


def test_construct_pair_with_doom_loop():
    """Doom-loop candidate should always be rejected, even if judge score is high."""
    # The looping candidate has a high judge score but should still be rejected
    loop_text = (
        "the answer is the answer is the answer is the answer is the answer is "
        "the answer is the answer is the answer is the answer is the answer is "
        "the answer is the answer is the answer is the answer is the answer is "
        "the answer is the answer is the answer is the answer is the answer is"
    )
    candidates = [
        Candidate(text="Good non-looping answer.", temperature=0.8, judge_score=7.0),
        Candidate(text=loop_text, temperature=0.8, judge_score=9.0),  # high score but loops
        Candidate(text="Another good answer.", temperature=0.8, judge_score=6.0),
    ]
    pair = construct_preference_pair(candidates, "test prompt")
    assert pair is not None
    # Chosen should be the best non-looping (score 7.0)
    assert pair["chosen"] == "Good non-looping answer."
    assert not pair["chosen_is_loop"]
    # Rejected should be the doom-loop (even though it has score 9.0)
    assert pair["rejected"] == loop_text
    assert pair["rejected_is_loop"]
    print("PASS: construct_preference_pair rejects doom-loop even with high judge score")


def test_construct_pair_all_loops():
    """If all candidates are doom-loops, return None (no valid pair)."""
    loop_text = (
        "the answer is the answer is the answer is the answer is the answer is "
        "the answer is the answer is the answer is the answer is the answer is "
        "the answer is the answer is the answer is the answer is the answer is "
        "the answer is the answer is the answer is the answer is the answer is"
    )
    candidates = [
        Candidate(text=loop_text, temperature=0.8, judge_score=5.0),
        Candidate(text=loop_text, temperature=0.8, judge_score=3.0),
    ]
    pair = construct_preference_pair(candidates, "test")
    assert pair is None
    print("PASS: construct_preference_pair returns None when all candidates loop")


def test_construct_pair_identical_chosen_rejected():
    """If chosen and rejected are the same text, return None."""
    candidates = [
        Candidate(text="Only one unique answer here.", temperature=0.8, judge_score=7.0),
    ]
    pair = construct_preference_pair(candidates, "test")
    # With only one candidate, chosen == rejected, should return None
    assert pair is None
    print("PASS: construct_preference_pair returns None for identical chosen/rejected")


def test_construct_pair_greedy_vs_temp():
    """Greedy candidate should be treated like any other for pair construction."""
    candidates = [
        Candidate(text="Greedy answer.", temperature=0.0, is_greedy=True, judge_score=6.0),
        Candidate(text="Sampled answer is better.", temperature=0.8, is_greedy=False, judge_score=8.0),
    ]
    pair = construct_preference_pair(candidates, "test")
    assert pair is not None
    assert pair["chosen"] == "Sampled answer is better."
    assert pair["rejected"] == "Greedy answer."
    print("PASS: construct_preference_pair handles greedy vs sampled correctly")


def test_load_prompts():
    """Load prompts from a JSONL file."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "What is 2+2?", "response": "4"}\n')
        f.write('{"prompt": "What is 3+3?", "response": "6"}\n')
        f.write('{"prompt": "What is 2+2?", "response": "duplicate"}\n')  # dedup
        path = f.name

    prompts = load_prompts(path)
    os.unlink(path)
    assert len(prompts) == 2
    assert "What is 2+2?" in prompts
    assert "What is 3+3?" in prompts
    print(f"PASS: load_prompts loaded {len(prompts)} unique prompts (dedup works)")


def test_judge_prompt_format():
    """Judge system prompt should contain scoring criteria."""
    assert "1-10" in JUDGE_SYSTEM_PROMPT
    assert "Correctness" in JUDGE_SYSTEM_PROMPT
    assert "JSON" in JUDGE_SYSTEM_PROMPT
    print("PASS: JUDGE_SYSTEM_PROMPT has correct format")


def test_repetition_ratio_consistency():
    """Repetition ratio should be consistent between curriculum_sft and dpo_data_gen."""
    # Both modules import from curriculum_sft, so this is trivially true,
    # but we verify the import chain works.
    text = "the the the the the the the the the the the the the the the the"
    ratio = ngram_repetition_ratio(text, n=3)
    assert ratio > 0.3
    print(f"PASS: repetition ratio consistency ({ratio:.2f})")


if __name__ == "__main__":
    test_candidate_dataclass()
    test_construct_pair_normal()
    test_construct_pair_with_doom_loop()
    test_construct_pair_all_loops()
    test_construct_pair_identical_chosen_rejected()
    test_construct_pair_greedy_vs_temp()
    test_load_prompts()
    test_judge_prompt_format()
    test_repetition_ratio_consistency()
    print("\n=== All DPO data generation tests passed ===")

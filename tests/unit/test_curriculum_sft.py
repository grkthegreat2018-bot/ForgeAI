"""Test curriculum SFT data pipeline: doom-loop detection, CoT classification, mix distillation."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from forge.training.runners.curriculum_sft import (
    classify_cot_length,
    estimate_token_count,
    filter_doom_loops,
    has_cot_markers,
    is_doom_loop,
    load_jsonl,
    mix_distillation,
    ngram_repetition_ratio,
    save_jsonl,
)


def test_ngram_repetition_ratio():
    """Normal text should have low repetition, looping text high."""
    normal = "The quick brown fox jumps over the lazy dog. " * 2
    assert ngram_repetition_ratio(normal, n=8) < 0.3

    looping = "the answer is the answer is the answer is the answer is the answer is"
    ratio = ngram_repetition_ratio(looping, n=3)
    assert ratio > 0.3, f"Expected high repetition ratio, got {ratio}"
    print(f"PASS: ngram_repetition_ratio (normal<0.3, looping={ratio:.2f})")


def test_is_doom_loop():
    """Doom-loop detector should flag repetitive text but not normal text."""
    normal = "To solve this problem, we need to calculate the sum of all numbers in the list. " \
             "We can iterate through each element and add it to a running total. " \
             "This gives us O(n) time complexity."
    assert not is_doom_loop(normal), "Normal text flagged as doom loop"

    looping = "the answer is the answer is the answer is the answer is the answer is " \
              "the answer is the answer is the answer is the answer is the answer is " \
              "the answer is the answer is the answer is the answer is the answer is " \
              "the answer is the answer is the answer is the answer is the answer is"
    assert is_doom_loop(looping), "Doom-loop text not detected"
    print("PASS: is_doom_loop correctly identifies doom loops")


def test_filter_doom_loops():
    """Filter should remove doom-loop examples and keep clean ones."""
    examples = [
        {"prompt": "q1", "text": "Normal short answer that is fine and not repetitive at all."},
        {"prompt": "q2", "text": "the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is the answer is"},
        {"prompt": "q3", "text": "Another normal answer with different content entirely."},
    ]
    filtered, removed = filter_doom_loops(examples)
    assert removed == 1, f"Expected 1 removed, got {removed}"
    assert len(filtered) == 2
    print(f"PASS: filter_doom_loops removed {removed}, kept {len(filtered)}")


def test_estimate_token_count():
    """Token count estimate should be roughly len/4."""
    text = "This is a test sentence with about ten words in it."
    tc = estimate_token_count(text)
    assert tc > 0
    assert tc == len(text) // 4
    print(f"PASS: estimate_token_count ({tc} tokens for {len(text)} chars)")


def test_has_cot_markers():
    """CoT marker detection should find reasoning markers."""
    assert has_cot_markers("Let me think about this problem step by step.")
    assert has_cot_markers("<|begin_of_thought|>\nOkay, let's solve this...")
    assert has_cot_markers("Step 1: Calculate the sum.\nStep 2: Divide by n.")
    assert not has_cot_markers("The answer is 42.")
    print("PASS: has_cot_markers detects CoT markers correctly")


def test_classify_cot_length():
    """Classify examples into short/long/neutral buckets."""
    examples = [
        # Short (≤150 tokens ≈ 600 chars)
        {"prompt": "q1", "text": "The answer is 42."},
        # Long with CoT markers (≥300 tokens ≈ 1200 chars)
        {"prompt": "q2", "text": "Let me think about this. " + "x " * 700},
        # Long without CoT markers
        {"prompt": "q3", "text": "def solve():\n    return " + "42 " * 700},
        # Medium (between short and long)
        {"prompt": "q4", "text": "x " * 200},  # ~50 tokens
    ]
    short, long_cot, neutral = classify_cot_length(
        examples, short_max_tokens=150, long_min_tokens=300
    )
    assert len(short) >= 1, f"Expected at least 1 short, got {len(short)}"
    assert len(long_cot) >= 2, f"Expected at least 2 long, got {len(long_cot)}"
    print(f"PASS: classify_cot_length (short={len(short)}, long={len(long_cot)}, neutral={len(neutral)})")


def test_mix_distillation():
    """Mix distillation should blend long + short at target ratio."""
    long_ex = [{"prompt": f"q{i}", "text": f"Long CoT solution {i} " * 100} for i in range(100)]
    short_ex = [{"prompt": f"q{i}", "text": f"Short answer {i}"} for i in range(100)]

    mixed = mix_distillation(long_ex, short_ex, mix_ratio=0.5, target_size=100)
    assert len(mixed) == 100
    # Check that both types are present
    long_count = sum(1 for ex in mixed if "Long CoT" in ex["text"])
    short_count = sum(1 for ex in mixed if "Short answer" in ex["text"])
    assert long_count == 50, f"Expected 50 long, got {long_count}"
    assert short_count == 50, f"Expected 50 short, got {short_count}"
    print(f"PASS: mix_distillation (50 long + 50 short = {len(mixed)} total)")

    # Test ratio 0.7 (70% long)
    mixed_70 = mix_distillation(long_ex, short_ex, mix_ratio=0.7, target_size=100)
    long_70 = sum(1 for ex in mixed_70 if "Long CoT" in ex["text"])
    assert long_70 == 70, f"Expected 70 long at 0.7 ratio, got {long_70}"
    print(f"PASS: mix_distillation ratio 0.7 ({long_70} long)")


def test_load_save_jsonl():
    """Load and save JSONL round-trip."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "q1", "solution": "a1"}\n')
        f.write('{"prompt": "q2", "response": "a2"}\n')
        in_path = f.name

    examples = load_jsonl([in_path])
    assert len(examples) == 2
    assert examples[0]["text"] == "a1"
    assert examples[1]["text"] == "a2"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        out_path = f.name
    save_jsonl(examples, out_path)

    # Reload and verify format
    with open(out_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    assert "prompt" in lines[0] and "response" in lines[0]

    os.unlink(in_path)
    os.unlink(out_path)
    print("PASS: load/save JSONL round-trip works")


def test_doom_loop_on_real_data():
    """Test doom-loop filtering on realistic reasoning data."""
    # Simulate a CoT trace that starts good but doom-loops
    good_cot = "Let me think about this step by step. " \
               "First, I need to understand what the problem is asking. " \
               "The problem asks us to find the sum of the first 100 natural numbers. " \
               "I can use the formula n*(n+1)/2 = 100*101/2 = 5050."
    assert not is_doom_loop(good_cot), "Good CoT flagged as doom loop"

    # Simulate a doom-loop CoT
    loop_cot = "So the answer is so the answer is so the answer is so the answer is " \
               "so the answer is so the answer is so the answer is so the answer is " \
               "so the answer is so the answer is so the answer is so the answer is " \
               "so the answer is so the answer is so the answer is so the answer is"
    assert is_doom_loop(loop_cot), "Doom-loop CoT not detected"
    print("PASS: doom-loop detection works on realistic CoT data")


if __name__ == "__main__":
    test_ngram_repetition_ratio()
    test_is_doom_loop()
    test_filter_doom_loops()
    test_estimate_token_count()
    test_has_cot_markers()
    test_classify_cot_length()
    test_mix_distillation()
    test_load_save_jsonl()
    test_doom_loop_on_real_data()
    print("\n=== All curriculum SFT tests passed ===")

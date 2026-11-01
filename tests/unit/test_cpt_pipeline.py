"""Test CPT data pipeline: loading, tokenization, packing, mixed sampling."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from forge.training.runners.cpt_train import (
    MixedDataSampler,
    load_jsonl_examples,
    render_cpt_text,
    tokenize_and_pack,
)


def test_load_jsonl_examples():
    """Load examples from a temp JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "What is 2+2?", "solution": "2+2=4. The answer is 4."}\n')
        f.write('{"prompt": "Write a function", "solution": "def f(): pass"}\n')
        f.write('{"prompt": "What is 2+2?", "solution": "duplicate, should be skipped"}\n')
        f.write('{"prompt": "", "solution": "empty prompt, skipped"}\n')
        path = f.name

    examples = load_jsonl_examples([path])
    os.unlink(path)
    assert len(examples) == 2, f"Expected 2 examples, got {len(examples)}"
    assert examples[0]["prompt"] == "What is 2+2?"
    assert examples[0]["text"] == "2+2=4. The answer is 4."
    print(f"PASS: load_jsonl_examples loaded {len(examples)} examples (dedup works)")


def test_render_cpt_text():
    """Render function should produce plain text with prompt + text."""
    ex = {"prompt": "What is 2+2?", "text": "2+2=4"}
    rendered = render_cpt_text(ex)
    assert "What is 2+2?" in rendered
    assert "2+2=4" in rendered
    assert rendered.endswith("\n\n")
    print("PASS: render_cpt_text produces correct format")


def test_tokenize_and_pack(v2_assets):
    """Tokenize and pack examples into fixed-length sequences."""
    from research.tokenizer_cache import get_tokenizer

    tokenizer = get_tokenizer()

    examples = [
        {"prompt": "What is 2+2?", "text": "2+2=4. The answer is 4."},
        {"prompt": "What is 3+3?", "text": "3+3=6. The answer is 6."},
        {"prompt": "Write code", "text": "def f(): return 42"},
    ]
    seq_len = 32
    packed = tokenize_and_pack(examples, tokenizer, seq_len)
    assert packed.dim() == 2
    assert packed.shape[1] == seq_len
    assert packed.shape[0] > 0
    assert packed.dtype == torch.long
    print(f"PASS: tokenize_and_pack produced {packed.shape[0]} sequences of {seq_len}")


def test_mixed_data_sampler():
    """MixedDataSampler should produce batches with correct ratio."""
    reasoning = torch.randint(0, 1000, (20, 64))
    general = torch.randint(0, 1000, (20, 64))
    sampler = MixedDataSampler(reasoning, general, batch_size=4, reasoning_ratio=0.6)

    batch, labels = sampler.get_batch("cpu")
    assert batch.shape == (4, 64)
    assert labels.shape == (4, 64)
    # 60% of 4 = 2 reasoning, 2 general
    assert sampler.n_reasoning_per_batch == 2
    assert sampler.n_general_per_batch == 2
    print(f"PASS: MixedDataSampler produces batch {batch.shape} with 2 reasoning + 2 general")


def test_mixed_data_sampler_edge_cases():
    """Empty pools should be handled gracefully."""
    # Empty general pool
    sampler = MixedDataSampler(
        torch.randint(0, 1000, (10, 64)),
        torch.empty(0, 64),
        batch_size=4,
        reasoning_ratio=0.6,
    )
    batch, _ = sampler.get_batch("cpu")
    assert batch.shape == (4, 64)
    assert sampler.n_general_per_batch == 0
    print("PASS: MixedDataSampler handles empty general pool")

    # Empty reasoning pool
    sampler = MixedDataSampler(
        torch.empty(0, 64),
        torch.randint(0, 1000, (10, 64)),
        batch_size=4,
        reasoning_ratio=0.6,
    )
    batch, _ = sampler.get_batch("cpu")
    assert batch.shape == (4, 64)
    assert sampler.n_reasoning_per_batch == 0
    print("PASS: MixedDataSampler handles empty reasoning pool")


if __name__ == "__main__":
    test_load_jsonl_examples()
    test_render_cpt_text()
    test_tokenize_and_pack()
    test_mixed_data_sampler()
    test_mixed_data_sampler_edge_cases()
    print("\n=== All CPT data pipeline tests passed ===")

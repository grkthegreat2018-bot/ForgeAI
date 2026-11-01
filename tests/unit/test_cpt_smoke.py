"""Smoke test: CPT training loop with tiny model + CPUAdamW."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F

from forge.config import get_config
from forge.model_loader import ModelLoader
from forge.training.optim.hybrid_offload import CPUAdamW
from forge.training.runners.cpt_train import (
    MixedDataSampler,
    load_jsonl_examples,
    tokenize_and_pack,
)
from research.tokenizer_cache import get_tokenizer


def test_cpt_training_smoke(v2_assets, gpu_available):
    """Run 3 CPT steps with tiny model + CPUAdamW."""
    # Create temp data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for i in range(20):
            f.write(f'{{"prompt": "Question {i}", "solution": "Answer {i} with reasoning."}}\n')
        reasoning_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for i in range(20):
            f.write(f'{{"prompt": "General {i}", "response": "General answer {i}."}}\n')
        general_path = f.name

    tokenizer = get_tokenizer()
    reasoning_ex = load_jsonl_examples([reasoning_path])
    general_ex = load_jsonl_examples([general_path])

    seq_len = 64
    reasoning_seqs = tokenize_and_pack(reasoning_ex, tokenizer, seq_len)
    general_seqs = tokenize_and_pack(general_ex, tokenizer, seq_len)

    # Tiny model has vocab_size=256; clamp token IDs to fit
    vocab = 256
    reasoning_seqs = reasoning_seqs % vocab
    general_seqs = general_seqs % vocab

    sampler = MixedDataSampler(reasoning_seqs, general_seqs, batch_size=2, reasoning_ratio=0.6)

    # Build tiny model
    config = get_config("forgelm_tiny")
    model = ModelLoader.build_model(config)
    model = model.to("cuda").to(torch.bfloat16)
    model.train()

    optimizer = CPUAdamW(model.parameters(), lr=1e-4, weight_decay=0.1, verbose=False)

    # Run 3 steps
    for step in range(3):
        input_ids, labels = sampler.get_batch("cuda")
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  CPT step {step}: loss={loss.item():.4f}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM: {vram:.2f} GB")
    os.unlink(reasoning_path)
    os.unlink(general_path)
    print("PASS: CPT training loop runs with CPUAdamW")


if __name__ == "__main__":
    test_cpt_training_smoke()
    print("\n=== CPT smoke test passed ===")

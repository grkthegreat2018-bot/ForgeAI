#!/usr/bin/env python
"""Test block reconstruction on real Qwen 2.5 0.5B with NanoQuant."""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.engine.quant.block_recon import BlockReconstructor
from forge.engine.quant.novel_quant_r48 import (
    estimate_r48_memory,
    quantize_model_nanoquant,
    quantize_model_ternary_ptq,
)


def compute_perplexity(model, input_ids, device):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        return torch.exp(loss).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="nanoquant", choices=["nanoquant", "ternary"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--recon-iters", type=int, default=30)
    parser.add_argument("--recon-lr", type=float, default=0.005)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=0, help="0=all")
    parser.add_argument("--mode", default="progressive",
                        choices=["standard", "progressive", "error_mitigation", "kl_calib"],
                        help="reconstruction mode")
    parser.add_argument("--admm-iters", type=int, default=400,
                        help="ADMM iterations (paper default: 400)")
    parser.add_argument("--kl-iters", type=int, default=20,
                        help="KL calibration iterations (after block recon)")
    args = parser.parse_args()

    device = args.device
    model_id = args.model

    print(f"\n{'='*70}")
    print(f"Block Reconstruction Test — {model_id}")
    print(f"Method: {args.method}, Device: {device}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    # Load original model
    print("\nLoading original model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True).to(device).eval()

    # Test sequence
    test_text = ("Quantization is a technique used to reduce the memory footprint "
                 "of large language models by representing weights with lower "
                 "precision numbers.")
    input_ids = tokenizer(test_text, return_tensors="pt", truncation=True,
                          max_length=128)["input_ids"].to(device)

    # Calibration data
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be compressed using quantization.",
        "Neural networks process information through layers of neurons.",
        "Transformers use attention mechanisms to weigh input importance.",
        "Gradient descent optimizes model parameters iteratively.",
        "Deep learning has revolutionized natural language processing.",
        "Model compression enables deployment on resource-constrained devices.",
        "Post-training quantization reduces model size without retraining.",
    ]
    calib_ids = tokenizer(calib_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)["input_ids"].to(device)
    # Use a subset for calibration
    if args.n_calib < calib_ids.shape[0]:
        calib_ids = calib_ids[:args.n_calib]

    # FP16 baseline
    fp16_ppl = compute_perplexity(model_orig, input_ids, device)
    print(f"\nFP16 PPL: {fp16_ppl:.4f}")

    # Load fresh model for quantization
    print(f"\nLoading fresh model for {args.method} quantization...")
    model_quant = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True).to(device).eval()

    # Quantize
    if args.method == "nanoquant":
        n = quantize_model_nanoquant(model_quant, rank=args.rank,
                                     admm_iters=args.admm_iters, verbose=True)
    else:
        n = quantize_model_ternary_ptq(model_quant, activations=None,
                                       refine_iters=0, verbose=True)

    mem = estimate_r48_memory(model_quant)
    print(f"Quantized: {n} layers, {mem['total_mb']:.1f} MB ({mem['avg_eff_bits']:.3f} bits)")

    # PPL before reconstruction
    ppl_before = compute_perplexity(model_quant, input_ids, device)
    print(f"\nPPL before reconstruction: {ppl_before:.4f}")

    # Block reconstruction
    print(f"\n{'='*70}")
    print(f"Starting block reconstruction ({args.recon_iters} iters, lr={args.recon_lr})")
    print(f"{'='*70}")

    max_blocks = args.n_blocks if args.n_blocks > 0 else None

    recon = BlockReconstructor(
        model_orig=model_orig,
        model_quant=model_quant,
        calibration_data=calib_ids,
        device=device,
        max_blocks=max_blocks,
    )

    t0 = time.time()
    if args.mode == "progressive":
        results = recon.reconstruct_progressive(
            n_iters=args.recon_iters, lr=args.recon_lr, verbose=True)
    elif args.mode == "error_mitigation":
        results = recon.reconstruct_with_error_mitigation(
            n_iters=args.recon_iters, lr=args.recon_lr, verbose=True)
    elif args.mode == "kl_calib":
        # First do progressive block recon, then KL calibration
        results = recon.reconstruct_progressive(
            n_iters=args.recon_iters, lr=args.recon_lr, verbose=True)
        if args.kl_iters > 0:
            kl_loss = recon.calibrate_kl(
                n_iters=args.kl_iters, lr=args.recon_lr * 0.1, verbose=True)
    else:
        results = recon.reconstruct(
            n_iters=args.recon_iters, lr=args.recon_lr, verbose=True)
    elapsed = time.time() - t0
    print(f"\nReconstruction took {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/block)")

    # PPL after reconstruction
    ppl_after = compute_perplexity(model_quant, input_ids, device)
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"FP16 PPL:              {fp16_ppl:.4f}")
    print(f"PPL before recon:      {ppl_before:.4f} (delta={ppl_before-fp16_ppl:+.4f})")
    print(f"PPL after recon:       {ppl_after:.4f} (delta={ppl_after-fp16_ppl:+.4f})")
    print(f"Improvement:           {ppl_before - ppl_after:.4f} PPL points")
    print(f"Reconstruction time:   {elapsed:.1f}s")

    # Memory
    mem_after = estimate_r48_memory(model_quant)
    print(f"Memory: {mem_after['total_mb']:.1f} MB ({mem_after['avg_eff_bits']:.3f} bits)")


if __name__ == "__main__":
    main()

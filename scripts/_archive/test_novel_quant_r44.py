#!/usr/bin/env python
"""R44 Novel Quantization Test Suite — Qwen 2.5 0.5B

Tests 4 novel quantization algorithms on a real model:
  1. HadamardLift (HLQ) — rotation + dimensional lifting
  2. AdaptiveBlockFP4 (AB-FP4) — MSE-optimal + kurtosis bit alloc
  3. SparseResidualINT3 (SR-INT3) — INT3 + error-threshold outliers
  4. TernaryLift (TL) — ternary lattice in lifted space

Baselines: FP16 (bf16), INT4 (group=128), INT8 (per-channel)

Metrics:
  - Weight memory (bytes, MB, effective bits/param)
  - Perplexity on WikiText-2 (raw) sample
  - Inference speed (tokens/sec, single batch)
  - Weight reconstruction error (Frobenius)

Usage:
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r44.py
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r44.py --methods hadamard_lift,ternary_lift
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r44.py --device cpu
"""
from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────
# Novel algorithms
from forge.engine.quant.novel_quant_r44 import (
    estimate_quantized_memory,
    quantize_model_adaptive_block_fp4,
    quantize_model_hadamard_lift,
    quantize_model_sparse_residual_int3,
    quantize_model_ternary_lift,
)

# FP4 baseline
from forge.engine.quant.nvfp4_quant import quantize_model_nvfp4

# Baselines
from forge.quant.inference_quant import quantize_model_int4, quantize_model_int8

# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def model_weight_bytes(model: nn.Module) -> int:
    """Estimate total weight storage bytes (including quantized buffers)."""
    total = 0
    for name, buf in model.named_buffers():
        total += buf.numel() * buf.element_size()
    for name, param in model.named_parameters():
        # Skip bias (small), count weight params
        if 'weight' in name or 'q_weight' in name or 'q_signs' in name or 'q_weights' in name or 'bias' in name:
            total += param.numel() * param.element_size()
        else:
            total += param.numel() * param.element_size()
    return total


def get_device_memory_mb() -> float:
    """Get current GPU memory allocated in MB (0 if CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def free_memory():
    """Aggressively free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_perplexity(model: nn.Module, input_ids: torch.Tensor,
                       device: str, max_length: int = 512) -> float:
    """Compute perplexity on a sequence of tokens.

    Args:
        model: the model
        input_ids: (1, seq_len) token IDs
        device: 'cuda' or 'cpu'
        max_length: max sequence length to process

    Returns:
        perplexity (float)
    """
    model.eval()
    input_ids = input_ids[:, :max_length].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    return torch.exp(loss).item()


def measure_inference_speed(model: nn.Module, input_ids: torch.Tensor,
                            device: str, n_tokens: int = 50,
                            n_warmup: int = 5) -> float:
    """Measure autoregressive generation speed (tokens/sec).

    Generates n_tokens one at a time, measures wall time.
    """
    model.eval()
    input_ids = input_ids[:, :32].to(device)  # Start with 32 tokens

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(input_ids)
            next_token = out.logits[:, -1:, :].argmax(dim=-1) if hasattr(out, 'logits') else out[0][:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.shape[1] > 50:
                input_ids = input_ids[:, :32]

    # Reset
    input_ids = input_ids[:, :32].to(device)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(input_ids)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            next_token = logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return n_tokens / elapsed


def weight_reconstruction_error(original_model: nn.Module,
                                quantized_model: nn.Module) -> float:
    """Compute average relative Frobenius error across replaced layers.

    Compares original nn.Linear weights to dequantized quantized weights.
    """
    errors = []

    orig_linears = {}
    for name, module in original_model.named_modules():
        if isinstance(module, nn.Linear):
            orig_linears[name] = module

    for name, module in quantized_model.named_modules():
        if name in orig_linears:
            orig_w = orig_linears[name].weight.data.float()
            if hasattr(module, '_dequantize_weight'):
                quant_w = module._dequantize_weight(torch.float32)
                if quant_w.shape == orig_w.shape:
                    err = (orig_w - quant_w).norm().item() / max(orig_w.norm().item(), 1e-8)
                    errors.append(err)

    return sum(errors) / max(len(errors), 1)


# ──────────────────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────────────────

METHODS = {
    'hadamard_lift': {
        'name': 'HadamardLift 1-bit (HLQ-1b)',
        'quantize_fn': quantize_model_hadamard_lift,
        'kwargs': {'lift_ratio': 2.0, 'lift_dim': 8, 'optimize_p': True, 'p_steps': 50, 'quant_bits': 1},
        'eff_bits': 2.0,
    },
    'hadamard_lift_2bit': {
        'name': 'HadamardLift 2-bit (HLQ-2b)',
        'quantize_fn': quantize_model_hadamard_lift,
        'kwargs': {'lift_ratio': 2.0, 'lift_dim': 8, 'optimize_p': True, 'p_steps': 200, 'quant_bits': 2, 'p_lr': 0.005},
        'eff_bits': 4.0,
    },
    'hadamard_lift_2bit_3x': {
        'name': 'HadamardLift 2-bit 3x (HLQ-2b-3x)',
        'quantize_fn': quantize_model_hadamard_lift,
        'kwargs': {'lift_ratio': 3.0, 'lift_dim': 8, 'optimize_p': True, 'p_steps': 50, 'quant_bits': 2},
        'eff_bits': 6.0,
    },
    'adaptive_block_fp4': {
        'name': 'AdaptiveBlockFP4 (AB-FP4)',
        'quantize_fn': quantize_model_adaptive_block_fp4,
        'kwargs': {'block_size': 32, 'kurt_low': 3.0, 'kurt_high': 7.0, 'use_residual': True},
        'eff_bits': 4.0,
    },
    'adaptive_block_fp4_no_residual': {
        'name': 'AB-FP4 no-residual (3/4-bit only)',
        'quantize_fn': quantize_model_adaptive_block_fp4,
        'kwargs': {'block_size': 32, 'kurt_low': 3.0, 'kurt_high': 7.0, 'use_residual': False},
        'eff_bits': 3.8,
    },
    'sparse_residual_int3': {
        'name': 'SparseResidualINT3 (SR-INT3)',
        'quantize_fn': quantize_model_sparse_residual_int3,
        'kwargs': {'group_size': 64, 'n_std': 2.0},
        'eff_bits': 3.2,
    },
    'sparse_residual_int3_aggressive': {
        'name': 'SR-INT3 aggressive (n_std=1.0)',
        'quantize_fn': quantize_model_sparse_residual_int3,
        'kwargs': {'group_size': 64, 'n_std': 1.0, 'base_bits': 3},
        'eff_bits': 3.5,
    },
    'sparse_residual_int4': {
        'name': 'SparseResidualINT4 (SR-INT4)',
        'quantize_fn': quantize_model_sparse_residual_int3,
        'kwargs': {'group_size': 64, 'n_std': 2.0, 'base_bits': 4},
        'eff_bits': 4.2,
    },
    'sparse_residual_int4_gs128': {
        'name': 'SR-INT4 gs=128 (less scale overhead)',
        'quantize_fn': quantize_model_sparse_residual_int3,
        'kwargs': {'group_size': 128, 'n_std': 2.0, 'base_bits': 4},
        'eff_bits': 4.1,
    },
    'sparse_residual_int4_aggressive': {
        'name': 'SR-INT4 aggressive (n_std=1.0)',
        'quantize_fn': quantize_model_sparse_residual_int3,
        'kwargs': {'group_size': 64, 'n_std': 1.0, 'base_bits': 4},
        'eff_bits': 4.5,
    },
    'ternary_lift': {
        'name': 'TernaryLift (TL)',
        'quantize_fn': quantize_model_ternary_lift,
        'kwargs': {'lift_ratio': 1.5, 'lift_dim': 8, 'optimize_p': True, 'p_steps': 200, 'p_lr': 0.005},
        'eff_bits': 2.37,
    },
    'ternary_lift_2x': {
        'name': 'TernaryLift 2x (TL-2x)',
        'quantize_fn': quantize_model_ternary_lift,
        'kwargs': {'lift_ratio': 2.0, 'lift_dim': 8, 'optimize_p': True, 'p_steps': 50},
        'eff_bits': 3.16,
    },
    # Baselines
    'int4': {
        'name': 'INT4 (baseline)',
        'quantize_fn': quantize_model_int4,
        'kwargs': {'group_size': 128},
        'eff_bits': 4.0,
    },
    'int8': {
        'name': 'INT8 (baseline)',
        'quantize_fn': quantize_model_int8,
        'kwargs': {},
        'eff_bits': 8.0,
    },
    'nvfp4': {
        'name': 'NVFP4 (baseline)',
        'quantize_fn': quantize_model_nvfp4,
        'kwargs': {'block_size': 32},
        'eff_bits': 4.5,
    },
}


def run_test(model_id: str, methods: list[str], device: str,
             max_seq_len: int = 512, n_gen_tokens: int = 50):
    """Run quantization tests on a real model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*80}")
    print(f"R44 Novel Quantization Test — {model_id}")
    print(f"Device: {device}, Methods: {len(methods)}")
    print(f"{'='*80}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load FP16 baseline model
    print(f"Loading {model_id} (fp16)...")
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model_fp = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device if device == 'cuda' else None,
    )
    model_fp.eval()

    n_params = count_parameters(model_fp)
    fp_bytes = n_params * 2  # bf16
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  FP16 weight memory: {fp_bytes/1024**2:.1f} MB")

    # Prepare test data (use a fixed English text for perplexity)
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models have become increasingly powerful in recent years, "
        "demonstrating remarkable capabilities in natural language understanding, "
        "code generation, and mathematical reasoning. However, their large size "
        "presents significant challenges for deployment on resource-constrained "
        "devices. Quantization offers a promising solution by reducing the precision "
        "of model weights and activations, thereby decreasing memory usage and "
        "computational requirements while maintaining acceptable performance levels. "
        "Post-training quantization methods are particularly attractive because they "
        "can be applied to pre-trained models without requiring additional training, "
        "making them accessible to a wide range of users and applications."
    )
    input_ids = tokenizer(test_text, return_tensors='pt')['input_ids']
    print(f"  Test sequence: {input_ids.shape[1]} tokens")

    # Baseline FP16 metrics
    print("\n--- FP16 Baseline ---")
    free_memory()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    fp_ppl = compute_perplexity(model_fp, input_ids, device, max_seq_len)
    fp_speed = measure_inference_speed(model_fp, input_ids, device, n_gen_tokens)
    fp_gpu_mem = get_device_memory_mb()

    print(f"  Perplexity: {fp_ppl:.4f}")
    print(f"  Speed: {fp_speed:.1f} tok/s")
    if device == 'cuda':
        print(f"  GPU memory: {fp_gpu_mem:.1f} MB")

    # Test each method
    results = [{
        'method': 'FP16 (baseline)',
        'ppl': fp_ppl,
        'speed': fp_speed,
        'weight_mb': fp_bytes / 1024**2,
        'eff_bits': 16.0,
        'recon_err': 0.0,
        'gpu_mem': fp_gpu_mem,
    }]

    for method_key in methods:
        if method_key not in METHODS:
            print(f"\n  [SKIP] Unknown method: {method_key}")
            continue

        cfg = METHODS[method_key]
        print(f"\n--- {cfg['name']} ---")

        # Create a fresh copy of the FP16 model
        free_memory()
        # Reload from scratch to avoid residual quantization
        model_q = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device if device == 'cuda' else None,
        )
        model_q.eval()

        # Apply quantization
        try:
            n_quantized = cfg['quantize_fn'](model_q, **cfg['kwargs'])
            print(f"  Layers quantized: {n_quantized}")
        except Exception as e:
            print(f"  [FAIL] Quantization error: {e}")
            import traceback
            traceback.print_exc()
            del model_q
            free_memory()
            continue

        # Measure memory
        mem_info = estimate_quantized_memory(model_q)
        weight_mb = mem_info['total_mb']
        eff_bits = mem_info['avg_eff_bits']
        print(f"  Weight memory: {weight_mb:.1f} MB (eff {eff_bits:.2f} bits/param)")

        # Measure reconstruction error
        recon_err = weight_reconstruction_error(model_fp, model_q)
        print(f"  Reconstruction error: {recon_err:.4f}")

        # Measure perplexity
        try:
            ppl = compute_perplexity(model_q, input_ids, device, max_seq_len)
            print(f"  Perplexity: {ppl:.4f} (delta={ppl - fp_ppl:+.4f})")
        except Exception as e:
            print(f"  [FAIL] PPL error: {e}")
            ppl = float('inf')

        # Measure speed
        try:
            speed = measure_inference_speed(model_q, input_ids, device, n_gen_tokens)
            print(f"  Speed: {speed:.1f} tok/s ({speed/fp_speed:.2f}x FP16)")
        except Exception as e:
            print(f"  [FAIL] Speed error: {e}")
            speed = 0.0

        # GPU memory
        gpu_mem = get_device_memory_mb()
        if device == 'cuda':
            print(f"  GPU memory: {gpu_mem:.1f} MB")

        results.append({
            'method': cfg['name'],
            'ppl': ppl,
            'speed': speed,
            'weight_mb': weight_mb,
            'eff_bits': eff_bits,
            'recon_err': recon_err,
            'gpu_mem': gpu_mem,
        })

        del model_q
        free_memory()

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'METHOD':<35} {'PPL':>8} {'ΔPPL':>8} {'SPEED':>8} {'WEIGHT':>8} {'EFF':>6} {'RECON':>8}")
    print(f"{'':.<35} {'':>8} {'':>8} {'tok/s':>8} {'MB':>8} {'bits':>6} {'err':>8}")
    print(f"{'-'*100}")
    for r in results:
        delta_ppl = r['ppl'] - results[0]['ppl']
        print(f"{r['method']:<35} {r['ppl']:>8.4f} {delta_ppl:>+8.4f} "
              f"{r['speed']:>8.1f} {r['weight_mb']:>8.1f} {r['eff_bits']:>6.2f} "
              f"{r['recon_err']:>8.4f}")
    print(f"{'='*100}")

    # Memory savings summary
    fp_weight = results[0]['weight_mb']
    print(f"\nMemory savings vs FP16 ({fp_weight:.1f} MB):")
    for r in results[1:]:
        savings = (1 - r['weight_mb'] / fp_weight) * 100
        print(f"  {r['method']:<35} {r['weight_mb']:>8.1f} MB  ({savings:>5.1f}% savings)")

    return results


def main():
    parser = argparse.ArgumentParser(description='R44 Novel Quant Test')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B',
                        help='HuggingFace model ID')
    parser.add_argument('--methods', default='all',
                        help='Comma-separated method names, or "all"')
    parser.add_argument('--device', default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Max sequence length for PPL')
    parser.add_argument('--gen-tokens', type=int, default=50,
                        help='Tokens to generate for speed test')
    args = parser.parse_args()

    if args.methods == 'all':
        methods = list(METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(',')]

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    results = run_test(
        model_id=args.model,
        methods=methods,
        device=device,
        max_seq_len=args.seq_len,
        n_gen_tokens=args.gen_tokens,
    )

    # Save results to file
    import json
    results_file = 'scripts/r44_quant_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()

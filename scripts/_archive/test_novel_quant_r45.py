#!/usr/bin/env python
"""R45 Novel Quantization Benchmark — Qwen 2.5 0.5B

Tests R45 novel quantization algorithms on a real model:
  1. WaveletLift (WL) — Haar wavelet + SVD low-rank binary
  2. SchurAB-FP4 — Schur-complement corrected AB-FP4
  3. SVDLiftBinary — SVD low-rank + binarize (LittleBit PTQ)
  4. ReQuant refinement on R44 AB-FP4
  5. LloydMaxRotatedKV — KV cache quantization (measured separately)

Baselines: FP16, R44 AB-FP4 no-residual, NVFP4

Usage:
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r45.py
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r45.py --methods wavelet_lift_128,svd_lift_128 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time

import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r44 import (
    estimate_quantized_memory as estimate_r44_memory,
)

# R44 algorithms (for comparison + ReQuant refinement)
from forge.engine.quant.novel_quant_r44 import (
    quantize_model_adaptive_block_fp4,
)

# R45 algorithms
from forge.engine.quant.novel_quant_r45 import (
    LloydMaxRotatedKVQuantizer,
    estimate_r45_memory,
    quantize_model_schur_ab_fp4,
    quantize_model_svd_lift_binary,
    quantize_model_wavelet_lift,
    requant_refine_model,
)
from forge.engine.quant.nvfp4_quant import quantize_model_nvfp4

# Baselines

# ──────────────────────────────────────────────────────────────────────────
# Utilities (shared with R44 test)
# ──────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_perplexity(model, input_ids, device, max_length=512):
    model.eval()
    input_ids = input_ids[:, :max_length].to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))
    return torch.exp(loss).item()


def measure_inference_speed(model, input_ids, device, n_tokens=50, n_warmup=5):
    model.eval()
    input_ids = input_ids[:, :32].to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(input_ids)
            next_token = out.logits[:, -1:, :].argmax(dim=-1) if hasattr(out, 'logits') else out[0][:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.shape[1] > 50:
                input_ids = input_ids[:, :32]
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


def weight_reconstruction_error(original_model, quantized_model):
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
# Method definitions
# ──────────────────────────────────────────────────────────────────────────

METHODS = {
    # R45 novel methods
    'wavelet_lift_64': {
        'name': 'WaveletLift r=64',
        'quantize_fn': quantize_model_wavelet_lift,
        'kwargs': {'rank': 64},
        'eff_bits': 0.29,
    },
    'wavelet_lift_128': {
        'name': 'WaveletLift r=128',
        'quantize_fn': quantize_model_wavelet_lift,
        'kwargs': {'rank': 128},
        'eff_bits': 0.57,
    },
    'wavelet_lift_256': {
        'name': 'WaveletLift r=256',
        'quantize_fn': quantize_model_wavelet_lift,
        'kwargs': {'rank': 256},
        'eff_bits': 1.14,
    },
    'schur_ab_fp4': {
        'name': 'SchurAB-FP4',
        'quantize_fn': quantize_model_schur_ab_fp4,
        'kwargs': {'block_size': 32, 'schur_iters': 3},
        'eff_bits': 4.0,
    },
    'svd_lift_64': {
        'name': 'SVDLiftBinary r=64',
        'quantize_fn': quantize_model_svd_lift_binary,
        'kwargs': {'rank': 64},
        'eff_bits': 0.29,
    },
    'svd_lift_128': {
        'name': 'SVDLiftBinary r=128',
        'quantize_fn': quantize_model_svd_lift_binary,
        'kwargs': {'rank': 128},
        'eff_bits': 0.57,
    },
    'svd_lift_256': {
        'name': 'SVDLiftBinary r=256',
        'quantize_fn': quantize_model_svd_lift_binary,
        'kwargs': {'rank': 256},
        'eff_bits': 1.14,
    },
    'svd_lift_448': {
        'name': 'SVDLiftBinary r=448 (~2bit)',
        'quantize_fn': quantize_model_svd_lift_binary,
        'kwargs': {'rank': 448},
        'eff_bits': 2.0,
    },
    # R44 comparison
    'ab_fp4_no_residual': {
        'name': 'AB-FP4 no-residual (R44)',
        'quantize_fn': quantize_model_adaptive_block_fp4,
        'kwargs': {'block_size': 32, 'kurt_low': 3.0, 'kurt_high': 7.0, 'use_residual': False},
        'eff_bits': 3.8,
    },
    # Baselines
    'nvfp4': {
        'name': 'NVFP4 (baseline)',
        'quantize_fn': quantize_model_nvfp4,
        'kwargs': {'block_size': 32},
        'eff_bits': 4.5,
    },
}


def run_test(model_id, methods, device, max_seq_len=512, n_gen_tokens=50,
             use_requant=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*80}")
    print(f"R45 Novel Quantization Test — {model_id}")
    print(f"Device: {device}, Methods: {len(methods)}, ReQuant: {use_requant}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model_fp = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    n_params = count_parameters(model_fp)
    fp16_mb = sum(p.numel() * p.element_size() for p in model_fp.parameters()) / 1024**2
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"FP16 weight memory: {fp16_mb:.1f} MB")

    # Test sequence
    test_text = ("Quantization is a technique used to reduce the memory footprint "
                 "of large language models by representing weights with lower "
                 "precision numbers. Post-training quantization applies this after "
                 "training without requiring gradient updates.")
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=max_seq_len)
    input_ids = inputs["input_ids"]
    print(f"Test sequence: {input_ids.shape[1]} tokens\n")

    # FP16 baseline
    print("--- FP16 Baseline ---")
    fp16_ppl = compute_perplexity(model_fp, input_ids, device)
    fp16_speed = measure_inference_speed(model_fp, input_ids, device, n_gen_tokens)
    fp16_gpu = get_device_memory_mb()
    print(f"  Perplexity: {fp16_ppl:.4f}")
    print(f"  Speed: {fp16_speed:.1f} tok/s")
    print(f"  GPU memory: {fp16_gpu:.1f} MB\n")

    results = []

    for method_key in methods:
        if method_key not in METHODS:
            print(f"  Unknown method: {method_key}, skipping")
            continue

        cfg = METHODS[method_key]
        print(f"--- {cfg['name']} ---")

        # Deep copy original model
        model_q = copy.deepcopy(model_fp)
        free_memory()

        # Quantize
        try:
            n_layers = cfg['quantize_fn'](model_q, verbose=True, **cfg['kwargs'])
        except Exception as e:
            print(f"  FAILED: {e}")
            del model_q
            free_memory()
            continue

        # ReQuant refinement
        if use_requant:
            requant_refine_model(model_q, model_fp, verbose=True)

        # Memory estimation
        try:
            mem_r45 = estimate_r45_memory(model_q)
            mem_r44 = estimate_r44_memory(model_q)
            if mem_r45['total_bytes'] > 0:
                weight_mb = mem_r45['total_mb']
                eff_bits = mem_r45['avg_eff_bits']
            elif mem_r44['total_bytes'] > 0:
                weight_mb = mem_r44['total_mb']
                eff_bits = mem_r44['avg_eff_bits']
            else:
                weight_mb = sum(p.numel() * p.element_size() for p in model_q.parameters()) / 1024**2
                eff_bits = 16.0
        except Exception as e:
            print(f"  Memory estimation failed: {e}")
            weight_mb = 0
            eff_bits = 0

        # Reconstruction error
        recon_err = weight_reconstruction_error(model_fp, model_q)

        # Perplexity
        try:
            ppl = compute_perplexity(model_q, input_ids, device)
        except Exception as e:
            print(f"  PPL computation failed: {e}")
            ppl = float('inf')

        # Speed
        try:
            speed = measure_inference_speed(model_q, input_ids, device, n_gen_tokens)
        except Exception as e:
            print(f"  Speed measurement failed: {e}")
            speed = 0.0

        gpu_mem = get_device_memory_mb()

        delta_ppl = ppl - fp16_ppl
        print(f"  Layers quantized: {n_layers}")
        print(f"  Weight memory: {weight_mb:.1f} MB (eff {eff_bits:.2f} bits/param)")
        print(f"  Reconstruction error: {recon_err:.4f}")
        print(f"  Perplexity: {ppl:.4f} (delta={delta_ppl:+.4f})")
        print(f"  Speed: {speed:.1f} tok/s ({speed/fp16_speed:.2f}x FP16)")
        print(f"  GPU memory: {gpu_mem:.1f} MB\n")

        results.append({
            'method': cfg['name'],
            'method_key': method_key,
            'ppl': ppl,
            'delta_ppl': delta_ppl,
            'speed': speed,
            'weight_mb': weight_mb,
            'eff_bits': eff_bits,
            'recon_err': recon_err,
            'gpu_mem': gpu_mem,
            'n_layers': n_layers,
        })

        del model_q
        free_memory()

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'METHOD':<40} {'PPL':>8} {'dPPL':>8} {'SPEED':>8} {'WEIGHT':>8} {'EFF':>6} {'RECON':>8}")
    print(f"{'':.<40} {'':>8} {'':>8} {'tok/s':>8} {'MB':>8} {'bits':>6} {'err':>8}")
    print(f"{'-'*100}")
    print(f"{'FP16 (baseline)':<40} {fp16_ppl:>8.4f} {0.0:>8.4f} {fp16_speed:>8.1f} {fp16_mb:>8.1f} {16.0:>6.2f} {0.0:>8.4f}")
    for r in results:
        print(f"{r['method']:<40} {r['ppl']:>8.4f} {r['delta_ppl']:>+8.4f} {r['speed']:>8.1f} {r['weight_mb']:>8.1f} {r['eff_bits']:>6.2f} {r['recon_err']:>8.4f}")
    print(f"{'='*100}")

    # Memory savings
    print(f"\nMemory savings vs FP16 ({fp16_mb:.1f} MB):")
    for r in results:
        savings = (1 - r['weight_mb'] / fp16_mb) * 100
        print(f"  {r['method']:<40} {r['weight_mb']:>8.1f} MB  ({savings:>+6.1f}% savings)")

    # Save results
    output = {
        'model': model_id,
        'device': device,
        'fp16_ppl': fp16_ppl,
        'fp16_speed': fp16_speed,
        'fp16_mb': fp16_mb,
        'results': results,
    }
    with open('scripts/r45_quant_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to scripts/r45_quant_results.json")


def test_kv_cache_quantization(model_id, device, bits_list=(2, 3)):
    """Test LloydMax KV cache quantization separately."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*80}")
    print(f"R45 KV Cache Quantization Test — {model_id}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    # Get model config
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"Model: hidden={config.hidden_size}, heads={config.num_attention_heads}, head_dim={head_dim}")

    test_text = ("Quantization reduces memory by using lower precision numbers "
                 "to represent model weights and activations.")
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"]

    # Generate KV cache from a forward pass
    with torch.no_grad():
        outputs = model(input_ids.to(device), output_hidden_states=False, use_cache=True)
        past_key_values = outputs.past_key_values

    if past_key_values is None:
        print("  Model did not produce KV cache, skipping KV test")
        return

    # Test each KV layer
    for bits in bits_list:
        quantizer = LloydMaxRotatedKVQuantizer(bits=bits, head_dim=head_dim, use_qjl=True)
        comp_ratio = quantizer.compression_ratio()

        total_orig_bytes = 0
        total_packed_bytes = 0
        total_recon_err = []

        for layer_idx in range(len(past_key_values)):
            k_cache, v_cache = past_key_values[layer_idx]
            # k_cache: (batch, heads, seq, head_dim)
            for kv_tensor, _name in [(k_cache, 'K'), (v_cache, 'V')]:
                packed = quantizer.quantize(kv_tensor.float())
                recon = quantizer.dequantize(packed)

                orig_bytes = kv_tensor.numel() * 2  # fp16
                packed_bytes = packed['packed'].numel() + packed['scales'].numel() * 2
                if packed['qjl_packed'] is not None:
                    packed_bytes += packed['qjl_packed'].numel()

                err = (kv_tensor.float() - recon).norm().item() / max(kv_tensor.float().norm().item(), 1e-8)

                total_orig_bytes += orig_bytes
                total_packed_bytes += packed_bytes
                total_recon_err.append(err)

        avg_err = sum(total_recon_err) / len(total_recon_err)
        actual_ratio = total_orig_bytes / max(total_packed_bytes, 1)

        print(f"  LloydMax {bits}-bit + QJL: comp={actual_ratio:.1f}x (theory={comp_ratio:.1f}x), "
              f"avg recon err={avg_err:.4f}")

    del model
    free_memory()


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R45 Novel Quantization Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model ID")
    parser.add_argument("--methods", default="all", help="Comma-separated method keys or 'all'")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--gen-tokens", type=int, default=30, help="Tokens for speed test")
    parser.add_argument("--requant", action="store_true", help="Apply ReQuant refinement")
    parser.add_argument("--kv-test", action="store_true", help="Also test KV cache quantization")
    args = parser.parse_args()

    if args.methods == "all":
        methods = list(METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    run_test(args.model, methods, args.device, n_gen_tokens=args.gen_tokens,
             use_requant=args.requant)

    if args.kv_test:
        test_kv_cache_quantization(args.model, args.device)

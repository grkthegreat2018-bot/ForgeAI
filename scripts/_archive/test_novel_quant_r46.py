#!/usr/bin/env python
"""R46 Novel Quantization Benchmark — Qwen 2.5 0.5B

Tests R46 algorithms:
  1. HadamardRotatedFP4 (HR-FP4) — rotation, no calibration
  2. GPTQFP4 — error compensation, needs calibration
  3. AWQFP4 — activation-aware, needs calibration
  4. OptimalGridFP4 (OG-FP4) — data-dependent codebook, no calibration
  5. HadamardGPTQFP4 (HR-GPTQ-FP4) — combined rotation + GPTQ

Usage:
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r46.py
    venv\\Scripts\\python.exe scripts\\test_novel_quant_r46.py --methods hr_fp4,og_fp4 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time

import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r45 import (
    estimate_r45_memory,
    quantize_model_schur_ab_fp4,
)
from forge.engine.quant.novel_quant_r46 import (
    collect_activations,
    estimate_r46_memory,
    quantize_model_awq_fp4,
    quantize_model_gptq_fp4,
    quantize_model_hadamard_awq_fp4,
    quantize_model_hadamard_gptq_fp4,
    quantize_model_hadamard_rotated_fp4,
    quantize_model_optimal_grid_fp4,
)
from forge.engine.quant.nvfp4_quant import quantize_model_nvfp4


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_device_memory_mb():
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

def measure_inference_speed(model, input_ids, device, n_tokens=30, n_warmup=5):
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


# Methods that need calibration
CALIBRATION_METHODS = {'gptq_fp4', 'awq_fp4', 'hr_gptq_fp4', 'hr_awq_fp4'}

METHODS = {
    'hr_fp4': {
        'name': 'HadamardRotatedFP4',
        'quantize_fn': quantize_model_hadamard_rotated_fp4,
        'kwargs': {'block_size': 32},
        'needs_cal': False,
    },
    'gptq_fp4': {
        'name': 'GPTQFP4',
        'quantize_fn': quantize_model_gptq_fp4,
        'kwargs': {'block_size': 32, 'group_size': 128},
        'needs_cal': True,
    },
    'awq_fp4': {
        'name': 'AWQFP4',
        'quantize_fn': quantize_model_awq_fp4,
        'kwargs': {'block_size': 32},
        'needs_cal': True,
    },
    'og_fp4': {
        'name': 'OptimalGridFP4',
        'quantize_fn': quantize_model_optimal_grid_fp4,
        'kwargs': {'block_size': 32, 'n_lloyd_iters': 20},
        'needs_cal': False,
    },
    'hr_gptq_fp4': {
        'name': 'HadamardGPTQFP4',
        'quantize_fn': quantize_model_hadamard_gptq_fp4,
        'kwargs': {'block_size': 32, 'group_size': 128},
        'needs_cal': True,
    },
    'hr_awq_fp4': {
        'name': 'HadamardAWQFP4',
        'quantize_fn': quantize_model_hadamard_awq_fp4,
        'kwargs': {'block_size': 32},
        'needs_cal': True,
    },
    # R45 comparison
    'schur_ab_fp4': {
        'name': 'SchurAB-FP4 (R45)',
        'quantize_fn': quantize_model_schur_ab_fp4,
        'kwargs': {'block_size': 32, 'schur_iters': 3},
        'needs_cal': False,
    },
    # Baseline
    'nvfp4': {
        'name': 'NVFP4 (baseline)',
        'quantize_fn': quantize_model_nvfp4,
        'kwargs': {'block_size': 32},
        'needs_cal': False,
    },
}


def run_test(model_id, methods, device, max_seq_len=512, n_gen_tokens=30,
             n_calib_samples=128):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*80}")
    print(f"R46 Novel Quantization Test — {model_id}")
    print(f"Device: {device}, Methods: {len(methods)}")
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
    print(f"Test sequence: {input_ids.shape[1]} tokens")

    # Calibration data — diverse texts for better activation statistics
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be compressed using quantization techniques.",
        "Neural networks process information through layers of interconnected neurons.",
        "Transformers use self-attention mechanisms to weigh the importance of input tokens.",
        "Gradient descent optimizes model parameters by following the loss gradient.",
        "Deep learning has revolutionized natural language processing and computer vision.",
        "Model compression enables deployment of large models on resource-constrained devices.",
        "Post-training quantization reduces model size without requiring retraining.",
        "The attention mechanism allows models to focus on relevant parts of the input.",
        "Large language models generate text by predicting the next token in a sequence.",
        "Knowledge distillation transfers knowledge from a large teacher to a small student model.",
        "Mixed precision training uses both 16-bit and 32-bit floating point numbers.",
        "The transformer architecture was introduced in the landmark paper Attention is All You Need.",
        "Inference speed is critical for real-time applications like chatbots and virtual assistants.",
        "Memory efficiency allows running larger models on consumer hardware with limited VRAM.",
        "Quantization aware training incorporates quantization noise during the training process.",
        "The GPU accelerator parallelizes matrix multiplication operations for faster computation.",
        "Token embeddings map discrete tokens to continuous vector representations.",
        "Positional encoding injects order information into the token embeddings.",
        "Layer normalization stabilizes training by normalizing activations within each layer.",
    ]
    # Tokenize all calibration texts and concatenate
    calib_input_ids_list = []
    for text in calib_texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        calib_input_ids_list.append(ids["input_ids"])
    print(f"Calibration: {len(calib_texts)} diverse texts")

    # Check if any method needs calibration
    needs_calib = any(METHODS.get(m, {}).get('needs_cal', False) for m in methods)
    activations = None
    if needs_calib:
        print(f"\nCollecting calibration activations ({n_calib_samples} samples per text)...")
        all_activations = {}
        for _i, calib_input_ids in enumerate(calib_input_ids_list):
            acts = collect_activations(model_fp, calib_input_ids,
                                        n_samples=n_calib_samples // len(calib_input_ids_list),
                                        device=device)
            for name, act in acts.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act)
        # Concatenate activations from all texts
        activations = {}
        for name, act_list in all_activations.items():
            activations[name] = torch.cat(act_list, dim=0)
        print(f"  Collected from {len(activations)} layers")
        for name, act in list(activations.items())[:3]:
            print(f"    {name}: {act.shape}")

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
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

        model_q = copy.deepcopy(model_fp)
        free_memory()

        try:
            if cfg['needs_cal']:
                n_layers = cfg['quantize_fn'](model_q, activations,
                                              verbose=True, **cfg['kwargs'])
            else:
                n_layers = cfg['quantize_fn'](model_q, verbose=True, **cfg['kwargs'])
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            del model_q
            free_memory()
            continue

        # Memory estimation
        try:
            mem_r46 = estimate_r46_memory(model_q)
            mem_r45 = estimate_r45_memory(model_q)
            if mem_r46['total_bytes'] > 0:
                weight_mb = mem_r46['total_mb']
                eff_bits = mem_r46['avg_eff_bits']
            elif mem_r45['total_bytes'] > 0:
                weight_mb = mem_r45['total_mb']
                eff_bits = mem_r45['avg_eff_bits']
            else:
                weight_mb = sum(p.numel() * p.element_size() for p in model_q.parameters()) / 1024**2
                eff_bits = 16.0
        except Exception as e:
            print(f"  Memory estimation failed: {e}")
            weight_mb = 0
            eff_bits = 0

        recon_err = weight_reconstruction_error(model_fp, model_q)

        try:
            ppl = compute_perplexity(model_q, input_ids, device)
        except Exception as e:
            print(f"  PPL computation failed: {e}")
            ppl = float('inf')

        try:
            speed = measure_inference_speed(model_q, input_ids, device, n_gen_tokens)
        except Exception as e:
            print(f"  Speed measurement failed: {e}")
            speed = 0.0

        gpu_mem = get_device_memory_mb()
        delta_ppl = ppl - fp16_ppl

        print(f"  Layers: {n_layers}")
        print(f"  Weight: {weight_mb:.1f} MB (eff {eff_bits:.2f} bits)")
        print(f"  Recon err: {recon_err:.4f}")
        print(f"  PPL: {ppl:.4f} (delta={delta_ppl:+.4f})")
        print(f"  Speed: {speed:.1f} tok/s ({speed/fp16_speed:.2f}x FP16)")
        print(f"  GPU mem: {gpu_mem:.1f} MB\n")

        results.append({
            'method': cfg['name'], 'method_key': method_key,
            'ppl': ppl, 'delta_ppl': delta_ppl, 'speed': speed,
            'weight_mb': weight_mb, 'eff_bits': eff_bits,
            'recon_err': recon_err, 'gpu_mem': gpu_mem, 'n_layers': n_layers,
        })

        del model_q
        free_memory()

    # Summary
    print(f"\n{'='*100}")
    print(f"{'METHOD':<35} {'PPL':>8} {'dPPL':>8} {'SPEED':>8} {'WEIGHT':>8} {'EFF':>6} {'RECON':>8}")
    print(f"{'':.<35} {'':>8} {'':>8} {'tok/s':>8} {'MB':>8} {'bits':>6} {'err':>8}")
    print(f"{'-'*100}")
    print(f"{'FP16 (baseline)':<35} {fp16_ppl:>8.4f} {0.0:>8.4f} {fp16_speed:>8.1f} {fp16_mb:>8.1f} {16.0:>6.2f} {0.0:>8.4f}")
    for r in results:
        print(f"{r['method']:<35} {r['ppl']:>8.4f} {r['delta_ppl']:>+8.4f} {r['speed']:>8.1f} {r['weight_mb']:>8.1f} {r['eff_bits']:>6.2f} {r['recon_err']:>8.4f}")
    print(f"{'='*100}")

    print(f"\nMemory savings vs FP16 ({fp16_mb:.1f} MB):")
    for r in results:
        savings = (1 - r['weight_mb'] / fp16_mb) * 100
        print(f"  {r['method']:<35} {r['weight_mb']:>8.1f} MB  ({savings:>+6.1f}%)")

    output = {
        'model': model_id, 'device': device,
        'fp16_ppl': fp16_ppl, 'fp16_speed': fp16_speed, 'fp16_mb': fp16_mb,
        'results': results,
    }
    with open('scripts/r46_quant_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to scripts/r46_quant_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R46 Novel Quantization Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gen-tokens", type=int, default=30)
    parser.add_argument("--calib-samples", type=int, default=128)
    args = parser.parse_args()

    if args.methods == "all":
        methods = list(METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    run_test(args.model, methods, args.device,
             n_gen_tokens=args.gen_tokens, n_calib_samples=args.calib_samples)

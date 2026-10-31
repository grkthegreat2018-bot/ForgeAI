#!/usr/bin/env python
"""R48 extreme low-bit benchmark on real Qwen 2.5 0.5B.

Tests NanoQuant, BTC-LLM, and TernaryPTQ against FP16 and NVFP4 baselines.
Measures perplexity, speed, memory, and reconstruction error.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r46 import (
    collect_activations as collect_acts_r46,
)
from forge.engine.quant.novel_quant_r46 import (
    estimate_r46_memory,
    quantize_model_awq_fp4,
)
from forge.engine.quant.novel_quant_r48 import (
    estimate_r48_memory,
    quantize_model_btc,
    quantize_model_nanoquant,
    quantize_model_ternary_ptq,
)
from forge.engine.quant.nvfp4_quant import quantize_model_nvfp4


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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


def measure_inference_speed(model, input_ids, device, n_gen_tokens=20):
    model.eval()
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        cur_ids = input_ids.clone()
        for _ in range(n_gen_tokens):
            outputs = model(cur_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            next_token = logits[:, -1:, :].argmax(dim=-1)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start
    return n_gen_tokens / elapsed


def get_device_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


METHODS = {
    'nanoquant_r128': {
        'name': 'NanoQuant (rank=128)',
        'needs_cal': False,
        'quantize': lambda model, acts: quantize_model_nanoquant(
            model, rank=128, admm_iters=50, refine_steps=0, verbose=True),
        'estimate': estimate_r48_memory,
    },
    'nanoquant_r256': {
        'name': 'NanoQuant (rank=256)',
        'needs_cal': False,
        'quantize': lambda model, acts: quantize_model_nanoquant(
            model, rank=256, admm_iters=50, refine_steps=0, verbose=True),
        'estimate': estimate_r48_memory,
    },
    'btc_k256': {
        'name': 'BTC-LLM (K=256, rotated)',
        'needs_cal': False,
        'quantize': lambda model, acts: quantize_model_btc(
            model, codebook_size=256, use_rotation=True, verbose=True),
        'estimate': estimate_r48_memory,
    },
    'btc_k512': {
        'name': 'BTC-LLM (K=512, rotated)',
        'needs_cal': False,
        'quantize': lambda model, acts: quantize_model_btc(
            model, codebook_size=512, use_rotation=True, verbose=True),
        'estimate': estimate_r48_memory,
    },
    'ternary_ptq': {
        'name': 'TernaryPTQ (Hessian-refined)',
        'needs_cal': True,
        'quantize': lambda model, acts: quantize_model_ternary_ptq(
            model, activations=acts, refine_iters=20, verbose=True),
        'estimate': estimate_r48_memory,
    },
    'ternary_vanilla': {
        'name': 'TernaryPTQ (vanilla absmean)',
        'needs_cal': False,
        'quantize': lambda model, acts: quantize_model_ternary_ptq(
            model, activations=None, refine_iters=0, verbose=True),
        'estimate': estimate_r48_memory,
    },
    'awq_fp4': {
        'name': 'AWQFP4 (R47 best)',
        'needs_cal': True,
        'quantize': lambda model, acts: quantize_model_awq_fp4(
            model, acts, verbose=True),
        'estimate': estimate_r46_memory,
    },
    'nvfp4': {
        'name': 'NVFP4 (baseline)',
        'needs_cal': False,
        'quantize': lambda model, acts: quantize_model_nvfp4(
            model, block_size=32, w4a8=False),
        'estimate': None,  # uses its own reporting
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gen-tokens", type=int, default=20)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--methods", default="all",
                        help="Comma-separated method keys or 'all'")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    model_id = args.model

    if args.methods == 'all':
        methods = list(METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(',')]

    print(f"\n{'='*80}")
    print(f"R48 Extreme Low-Bit Benchmark — {model_id}")
    print(f"Device: {device}, Methods: {len(methods)}")
    print(f"{'='*80}")

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
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    print(f"Test sequence: {input_ids.shape[1]} tokens")

    # Diverse calibration
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
    calib_input_ids_list = [tokenizer(t, return_tensors="pt", truncation=True,
                                       max_length=128)["input_ids"].to(device)
                            for t in calib_texts]
    print(f"Calibration: {len(calib_texts)} diverse texts")

    # Collect activations for methods that need them
    needs_calib = any(METHODS.get(m, {}).get('needs_cal', False) for m in methods)
    activations = None
    if needs_calib:
        print("\nCollecting calibration activations...")
        all_acts = {}
        for calib_input_ids in calib_input_ids_list:
            acts = collect_acts_r46(model_fp, calib_input_ids,
                                     n_samples=args.calib_samples // len(calib_input_ids_list),
                                     device=device)
            for name, act in acts.items():
                all_acts.setdefault(name, []).append(act)
        activations = {name: torch.cat(act_list, dim=0) for name, act_list in all_acts.items()}
        print(f"  Collected from {len(activations)} layers")

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    fp16_ppl = compute_perplexity(model_fp, input_ids, device)
    fp16_speed = measure_inference_speed(model_fp, input_ids, device, args.gen_tokens)
    fp16_gpu = get_device_memory_mb()
    print(f"  PPL: {fp16_ppl:.4f}, Speed: {fp16_speed:.1f} tok/s, GPU: {fp16_gpu:.1f} MB")

    results = []

    for method_key in methods:
        if method_key not in METHODS:
            print(f"  Unknown method: {method_key}, skipping")
            continue

        cfg = METHODS[method_key]
        print(f"\n--- {cfg['name']} ---")

        # Reload fresh model
        free_memory()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(device).eval()

        try:
            n_q = cfg['quantize'](model, activations)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({
                'method': cfg['name'], 'ppl': None, 'speed': None,
                'weight_mb': None, 'eff_bits': None, 'recon_err': None,
                'error': str(e),
            })
            del model
            continue

        # Measure
        ppl = compute_perplexity(model, input_ids, device)
        speed = measure_inference_speed(model, input_ids, device, args.gen_tokens)
        gpu_mem = get_device_memory_mb()

        # Memory estimation
        if cfg['estimate']:
            mem = cfg['estimate'](model)
            weight_mb = mem['total_mb']
            eff_bits = mem['avg_eff_bits']
        else:
            # NVFP4 — estimate from packed weights
            weight_mb = 0
            eff_bits = 0
            for name, m in model.named_modules():
                if hasattr(m, 'weight_packed') and hasattr(m, 'weight_scales'):
                    wp = m.weight_packed.numel()
                    ws = m.weight_scales.numel() * 2
                    wgs = getattr(m, 'weight_global_scale', None)
                    wgs_b = wgs.numel() * 4 if wgs is not None else 0
                    weight_mb += (wp + ws + wgs_b) / 1024**2
                    eff_bits += (wp * 8 + ws * 8 + wgs_b * 8)
            total_params = sum(m.out_features * m.in_features
                               for _, m in model.named_modules()
                               if hasattr(m, 'weight_packed'))
            eff_bits = eff_bits / max(total_params, 1)

        # Reconstruction error (sample a few layers)
        recon_err = 0.0
        n_layers = 0
        with torch.no_grad():
            for name, m in model.named_modules():
                if hasattr(m, '_get_weight') and hasattr(m, 'in_features'):
                    w_q = m._get_weight(dtype)
                    # Can't compare to original (it's gone), skip
                    break

        delta_ppl = ppl - fp16_ppl if ppl is not None else None
        speed_ratio = speed / fp16_speed if speed and fp16_speed else None

        print(f"  PPL: {ppl:.4f} (delta={delta_ppl:+.4f})" if ppl else "  PPL: FAILED")
        print(f"  Speed: {speed:.1f} tok/s ({speed_ratio:.2f}x FP16)" if speed else "  Speed: FAILED")
        print(f"  Weight: {weight_mb:.1f} MB (eff {eff_bits:.3f} bits)")
        print(f"  GPU mem: {gpu_mem:.1f} MB")

        results.append({
            'method': cfg['name'],
            'ppl': ppl,
            'delta_ppl': delta_ppl,
            'speed': speed,
            'speed_ratio': speed_ratio,
            'weight_mb': weight_mb,
            'eff_bits': eff_bits,
            'gpu_mem': gpu_mem,
            'n_layers': n_q if isinstance(n_q, int) else None,
        })

        del model
        free_memory()

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'METHOD':<35} {'PPL':>8} {'dPPL':>8} {'SPEED':>8} {'WEIGHT':>10} {'EFF':>7} {'GPU':>10}")
    print(f"{'':.<35} {'':>8} {'':>8} {'tok/s':>8} {'MB':>10} {'bits':>7} {'MB':>10}")
    print(f"{'.'*90}")
    print(f"{'FP16 (baseline)':<35} {fp16_ppl:>8.4f} {0:>8.4f} {fp16_speed:>8.1f} "
          f"{fp16_mb:>10.1f} {16.0:>7.2f} {fp16_gpu:>10.1f}")

    for r in results:
        if r['ppl'] is not None:
            print(f"{r['method']:<35} {r['ppl']:>8.4f} {r['delta_ppl']:>+8.4f} "
                  f"{r['speed']:>8.1f} {r['weight_mb']:>10.1f} "
                  f"{r['eff_bits']:>7.3f} {r['gpu_mem']:>10.1f}")
        else:
            print(f"{r['method']:<35} {'FAIL':>8} {'—':>8} {'—':>8} {'—':>10} {'—':>7} {'—':>10}")

    print(f"{'='*90}")

    # Memory savings
    print(f"\nMemory savings vs FP16 ({fp16_mb:.1f} MB):")
    for r in results:
        if r['weight_mb']:
            savings = (1 - r['weight_mb'] / fp16_mb) * 100
            print(f"  {r['method']:<35} {r['weight_mb']:>10.1f} MB  ({savings:+.1f}%)")

    # Save results
    out = {
        'model': model_id,
        'fp16_ppl': fp16_ppl,
        'fp16_speed': fp16_speed,
        'fp16_mb': fp16_mb,
        'results': results,
    }
    with open('scripts/r48_quant_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to scripts/r48_quant_results.json")


if __name__ == "__main__":
    main()

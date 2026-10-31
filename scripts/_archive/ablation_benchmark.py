"""Automated ablation benchmark for ForgeLM keys.

Tests each runtime key individually against the base model:
- Perplexity on calibration data
- Generation speed (tokens/sec)
- VRAM usage
- Output cosine similarity to baseline (losslessness check)
- Runtime overhead (ms/token)

Usage:
    python ablation_benchmark.py
    python ablation_benchmark.py --keys rope_share,wisparse,spec_attn
    python ablation_benchmark.py --quick  # fewer samples, faster
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import paths as _paths

V2_CHECKPOINT = _paths.as_str(_paths.V2_CHECKPOINT)
TOKENIZER_PATH = _paths.as_str(_paths.FORGE_TOKENIZER_DIR)
RESULTS_DIR = _paths.as_str(_paths.ABLATION_RESULTS_DIR)

# Test prompts for perplexity + generation speed
TEST_PROMPTS = [
    "def fibonacci(n):",
    "def is_prime(n):",
    "def binary_search(arr, target):",
    "def reverse_string(s):",
    "def factorial(n):",
    "class Stack:",
    "def merge_sort(arr):",
    "def count_vowels(s):",
    "def gcd(a, b):",
    "def power(base, exp):",
]


@torch.no_grad()
def measure_perplexity(model, tokenizer, prompts, device="cuda"):
    """Measure average perplexity across test prompts."""
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        full = f'"""{prompt}"""\n'
        input_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids, use_cache=False)
        logits = out[0] if isinstance(out, tuple) else out
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


@torch.no_grad()
def measure_generation_speed(model, tokenizer, prompt, device="cuda",
                             max_tokens=30):
    """Measure generation speed (tokens/sec) and VRAM."""
    full = f'"""{prompt}"""\n'
    input_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    torch.cuda.synchronize() if device == "cuda" else None
    vram_before = torch.cuda.memory_allocated() // (1024 * 1024) if device == "cuda" else 0
    t0 = time.perf_counter()

    # Generate with KV cache
    out = model(input_ids, use_cache=True)
    logits = out[0] if isinstance(out, tuple) else out
    past_kvs = out[2] if isinstance(out, tuple) and len(out) > 2 else (out[1] if isinstance(out, tuple) else None)
    next_token = logits[0, -1].argmax().unsqueeze(0)
    generated = [next_token.item()]

    for _ in range(max_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        cur = next_token.unsqueeze(0).to(device)
        out = model(cur, past_key_values=past_kvs, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kvs = out[2] if isinstance(out, tuple) and len(out) > 2 else (out[1] if isinstance(out, tuple) else None)
        next_token = logits[0, -1].argmax().unsqueeze(0)
        generated.append(next_token.item())

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0
    vram_after = torch.cuda.memory_allocated() // (1024 * 1024) if device == "cuda" else 0
    vram_peak = torch.cuda.max_memory_allocated() // (1024 * 1024) if device == "cuda" else 0

    tokens_gen = len(generated)
    tokens_per_sec = tokens_gen / elapsed if elapsed > 0 else 0
    ms_per_token = (elapsed * 1000) / tokens_gen if tokens_gen > 0 else 0

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "vram_mb": vram_after,
        "vram_peak_mb": vram_peak,
        "tokens_generated": tokens_gen,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True)[:100],
    }


@torch.no_grad()
def measure_output_similarity(model, tokenizer, prompts, baseline_outputs, device="cuda"):
    """Measure cosine similarity of logits vs baseline (losslessness check)."""
    sims = []
    for prompt, base_logits in zip(prompts, baseline_outputs):
        full = f'"""{prompt}"""\n'
        input_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
        out = model(input_ids, use_cache=False)
        logits = out[0] if isinstance(out, tuple) else out
        # Compare last-token logits
        cur = logits[0, -1].float().flatten()
        base = base_logits.float().flatten()
        cos = torch.nn.functional.cosine_similarity(cur.unsqueeze(0), base.unsqueeze(0)).item()
        sims.append(cos)
    return sum(sims) / len(sims) if sims else 0.0


def collect_baseline_outputs(model, tokenizer, prompts, device="cuda"):
    """Collect baseline logits for similarity comparison."""
    outputs = []
    for prompt in prompts:
        full = f'"""{prompt}"""\n'
        input_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
        out = model(input_ids, use_cache=False)
        logits = out[0] if isinstance(out, tuple) else out
        outputs.append(logits[0, -1].clone())
    return outputs


# ── Key registry: (name, apply_fn, revert_fn) ──────────────────────
def get_testable_keys():
    """Return list of (key_name, apply_fn, revert_fn) for ablation testing."""
    keys = []

    # RoPE Buffer Sharing
    def apply_rope_share(model):
        from forge.keys.position.rope_share_key import RoPEShareKey
        k = RoPEShareKey()
        k.apply(model)
        return k
    keys.append(("rope_share", apply_rope_share, None))

    # WiSparse
    def apply_wisparse(model):
        from forge.keys.compression.wisparse_key import WiSparseKey
        k = WiSparseKey()
        k.apply(model)
        return k
    keys.append(("wisparse", apply_wisparse, None))

    # Per-Query Temperature
    def apply_per_query_temp(model):
        from forge.keys.training.per_query_temp_key import apply_per_query_temp as _apply
        _apply(model)
        return None
    keys.append(("per_query_temp", apply_per_query_temp, None))

    # Norm-Gated MoD
    def apply_norm_gated_mod(model):
        from forge.keys.normalization.norm_gated_mod_key import apply_norm_gated_mod as _apply
        _apply(model)
        return None
    keys.append(("norm_gated_mod", apply_norm_gated_mod, None))

    # Logit Cap (runtime flag, no apply method — uses forward)
    def apply_logit_cap(model):
        from forge.keys.training.logit_cap_key import LogitCapKey
        k = LogitCapKey()
        # LogitCap patches the model's forward via monkey-patching
        if hasattr(k, 'apply'):
            k.apply(model)
        elif hasattr(k, 'patch_model'):
            k.patch_model(model)
        return k
    keys.append(("logit_cap", apply_logit_cap, None))

    # MAC-Attention
    def apply_mac_attn(model):
        from forge.keys.attention.attn_reuse_key import AttnReuseKey
        k = AttnReuseKey(max_entries=16, match_threshold=0.85)
        k.apply(model)
        return k
    keys.append(("mac_attn", apply_mac_attn, None))

    # Speculative Attention
    def apply_spec_attn(model):
        from forge.keys.speculative.speculative_keys import SpeculativeAttentionKey
        k = SpeculativeAttentionKey(draft_rank=32)
        k.apply(model)
        return k
    keys.append(("spec_attn", apply_spec_attn, None))

    return keys


def main():
    parser = argparse.ArgumentParser(description="Automated ablation benchmark")
    parser.add_argument("--keys", type=str, default="",
                        help="Comma-separated keys to test (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer prompts, fewer tokens")
    parser.add_argument("--output", default=RESULTS_DIR,
                        help="Output directory for results")
    args = parser.parse_args()

    print("=" * 70)
    print("ForgeLM Key Ablation Benchmark")
    print("=" * 70)

    from transformers import AutoTokenizer

    from forge.config import get_config
    from forge.model_loader import ModelLoader

    device = "cuda"
    prompts = TEST_PROMPTS[:4] if args.quick else TEST_PROMPTS
    gen_tokens = 15 if args.quick else 30

    # Load model
    print("\n[1] Loading ForgeLM V2 (bf16)...")
    cfg = get_config("forgelm_v2", device=device)
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=V2_CHECKPOINT, moe_top_k=0,
        dtype=torch.bfloat16)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect baseline
    print("\n[2] Measuring baseline...")
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    baseline_ppl = measure_perplexity(model, tokenizer, prompts, device)
    baseline_speed = measure_generation_speed(model, tokenizer, prompts[0], device, gen_tokens)
    baseline_outputs = collect_baseline_outputs(model, tokenizer, prompts, device)
    print(f"  Perplexity: {baseline_ppl:.2f}")
    print(f"  Speed: {baseline_speed['tokens_per_sec']:.1f} tok/s "
          f"({baseline_speed['ms_per_token']:.1f} ms/tok)")
    print(f"  VRAM: {baseline_speed['vram_mb']} MB (peak: {baseline_speed['vram_peak_mb']} MB)")

    results = [{
        "key": "baseline",
        "perplexity": baseline_ppl,
        "tokens_per_sec": baseline_speed["tokens_per_sec"],
        "ms_per_token": baseline_speed["ms_per_token"],
        "vram_mb": baseline_speed["vram_mb"],
        "vram_peak_mb": baseline_speed["vram_peak_mb"],
        "cosine_sim": 1.0,
        "overhead_ms": 0.0,
    }]

    # Test each key
    all_keys = get_testable_keys()
    if args.keys:
        selected = args.keys.split(",")
        all_keys = [k for k in all_keys if k[0] in selected]

    print(f"\n[3] Testing {len(all_keys)} keys...")
    for key_name, apply_fn, _revert_fn in all_keys:
        print(f"\n  [{key_name}]")
        try:
            # Reload fresh model to avoid key contamination
            del model
            import gc; gc.collect()
            torch.cuda.empty_cache()
            model = ModelLoader.build_model_fast(
                cfg, checkpoint_path=V2_CHECKPOINT, moe_top_k=0,
                dtype=torch.bfloat16)
            model.to(device).eval()

            # Apply key
            key_obj = apply_fn(model)
            # Calibrate keys that need it (wisparse)
            if hasattr(key_obj, 'calibrate'):
                sample_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)
                try:
                    key_obj.calibrate(model, sample_ids)
                except Exception as e:
                    print(f"    (calibration skipped: {e})")
            torch.cuda.empty_cache()

            # Measure
            torch.cuda.reset_peak_memory_stats()
            ppl = measure_perplexity(model, tokenizer, prompts, device)
            speed = measure_generation_speed(model, tokenizer, prompts[0], device, gen_tokens)
            cos_sim = measure_output_similarity(model, tokenizer, prompts, baseline_outputs, device)

            overhead = speed["ms_per_token"] - baseline_speed["ms_per_token"]
            ppl_change = (ppl - baseline_ppl) / baseline_ppl * 100
            speed_change = (speed["tokens_per_sec"] - baseline_speed["tokens_per_sec"]) / baseline_speed["tokens_per_sec"] * 100

            result = {
                "key": key_name,
                "perplexity": ppl,
                "tokens_per_sec": speed["tokens_per_sec"],
                "ms_per_token": speed["ms_per_token"],
                "vram_mb": speed["vram_mb"],
                "vram_peak_mb": speed["vram_peak_mb"],
                "cosine_sim": cos_sim,
                "overhead_ms": overhead,
                "ppl_change_pct": ppl_change,
                "speed_change_pct": speed_change,
            }
            results.append(result)

            # Print summary
            lossless = "LOSSLESS" if cos_sim > 0.999 else f"cos={cos_sim:.4f}"
            print(f"    PPL: {ppl:.2f} ({ppl_change:+.1f}%)")
            print(f"    Speed: {speed['tokens_per_sec']:.1f} tok/s ({speed_change:+.1f}%)")
            print(f"    VRAM: {speed['vram_mb']} MB (peak: {speed['vram_peak_mb']} MB)")
            print(f"    {lossless}, overhead: {overhead:+.1f} ms/tok")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"key": key_name, "error": str(e)})

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"{'Key':<20} {'PPL':>8} {'PPL%':>8} {'tok/s':>8} {'spd%':>8} "
          f"{'VRAM':>8} {'cos':>8} {'ms/tok':>8}")
    print(f"{'-' * 80}")
    for r in results:
        if "error" in r:
            print(f"{r['key']:<20} {'ERROR':>8}")
            continue
        print(f"{r['key']:<20} {r['perplexity']:>8.2f} "
              f"{r.get('ppl_change_pct', 0):>+7.1f}% "
              f"{r['tokens_per_sec']:>7.1f} "
              f"{r.get('speed_change_pct', 0):>+7.1f}% "
              f"{r['vram_mb']:>7}MB "
              f"{r['cosine_sim']:>8.4f} "
              f"{r['ms_per_token']:>7.1f}")
    print(f"{'=' * 80}")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

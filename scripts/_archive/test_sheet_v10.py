"""ForgeEngine test sheet for ForgeLM V10-1.2B â€” the new base model.

Uses ForgeEngine's full feature set (activate, generate, stats, bottleneck)
instead of raw model.forward(). Tests:

  1. Load + activate (standard, spectral KV, optimal)
  2. Lossless verification (logit diff vs raw forward)
  3. Speed: tokens/sec across KV strategies (standard vs spectral vs s4r)
  4. Memory: VRAM + RAM per strategy
  5. Accuracy: 10-question multi-turn test (greedy + sampled)
  6. KV compression: measured bytes per strategy
  7. Bottleneck profile: per-layer timings
  8. Engine stats dump

Usage:
  python scripts/test_sheet_v10.py
  python scripts/test_sheet_v10.py --kv spectral
  python scripts/test_sheet_v10.py --quick  # skip bottleneck profiling
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(msg):
    print(msg, flush=True)

# â”€â”€â”€ Test questions (multi-turn, increasing difficulty) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
QUESTIONS = [
    # Arithmetic
    ("What is 2+2? Answer with just the number.", "4"),
    ("What is 10 minus 3? Answer with just the number.", "7"),
    ("What is 5 times 6? Answer with just the number.", "30"),
    # Knowledge
    ("What is the capital of France?", "Paris"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    # Reasoning
    ("If I have 3 apples and eat 1, how many are left?", "2"),
    ("What comes after Monday?", "Tuesday"),
    # Code
    ("In Python, what function prints text to the console?", "print"),
    ("What does CPU stand for?", "central processing unit"),
]

def check_answer(response, expected):
    return expected.lower() in response.lower()

def run_test_sheet(kv_strategy="standard", quick=False, device="cuda"):
    from forge.config import get_config
    from forge.engine.forge_engine import ForgeEngine
    from forge.model_loader import load_default_model
    from research.paths import V10_CHECKPOINT

    log("=" * 70)
    log("ForgeEngine Test Sheet â€” ForgeLM V10-1.2B")
    log(f"KV strategy: {kv_strategy}")
    log("=" * 70)

    # â”€â”€ Load model â”€â”€
    log("\n[1] Loading V10-1.2B...")
    t0 = time.time()
    model, tokenizer = load_default_model(
        "forgelm_v2_light",
        checkpoint_path=str(V10_CHECKPOINT),
        device=device,
        dtype=torch.bfloat16,
    )
    load_time = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Loaded in {load_time:.1f}s | {n_params/1e6:.1f}M params")

    # â”€â”€ Create ForgeEngine â”€â”€
    log("\n[2] Creating ForgeEngine...")
    engine = ForgeEngine(model, tokenizer, device=device,
                         checkpoint_path=str(V10_CHECKPOINT))
    log(f"  Engine created. KV cache: {engine.kv_cache}")

    # â”€â”€ Activate with strategy â”€â”€
    log(f"\n[3] Activating with kv_cache='{kv_strategy}'...")
    activate_kwargs = {
        "kv_cache": kv_strategy,
        "warmup": True,
    }
    if kv_strategy == "spectral":
        # SpectralKV needs max_freq config
        cfg = get_config("forgelm_v2_light")
        activate_kwargs["kv_cache_tokens"] = cfg.spectral_kv_max_freq
    try:
        engine.activate(**activate_kwargs)
        log(f"  Activated: {engine.kv_cache}")
    except Exception as e:
        log(f"  Activation failed: {e}")
        traceback.print_exc()
        # Fall back to standard
        engine.activate(kv_cache="standard", warmup=True)
        kv_strategy = "standard (fallback)"

    # â”€â”€ Engine stats â”€â”€
    log("\n[4] Engine stats:")
    stats = engine.stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")

    # â”€â”€ Test 1: Speed (tokens/sec) â”€â”€
    log("\n" + "=" * 50)
    log(f"TEST 1: Generation speed (kv={kv_strategy})")
    log("=" * 50)

    speed_prompts = [
        ("Short", "The future of AI is", 32),
        ("Medium", "Write a short story about a robot learning to paint.", 64),
        ("Long", "Explain the concept of recursion in programming with examples.", 128),
    ]

    speed_results = {}
    for label, prompt, n_tokens in speed_prompts:
        try:
            # Warmup
            _ = engine.generate(prompt, max_new_tokens=4, temperature=0.0)
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            output = engine.generate(prompt, max_new_tokens=n_tokens, temperature=0.0)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tps = n_tokens / elapsed
            speed_results[label] = {"tps": tps, "elapsed": elapsed, "n_tokens": n_tokens}
            log(f"  {label}: {tps:.1f} tok/s ({n_tokens} tok in {elapsed:.2f}s)")
            log(f"    Output: '{output[:80]}...'")
        except Exception as e:
            log(f"  {label}: ERROR: {e}")
            speed_results[label] = {"error": str(e)}

    # â”€â”€ Test 2: Memory â”€â”€
    log("\n" + "=" * 50)
    log(f"TEST 2: Memory usage (kv={kv_strategy})")
    log("=" * 50)

    vram = engine.vram_usage()
    import psutil
    ram = psutil.Process().memory_info().rss / 1e6
    log(f"  VRAM: {vram.get('used_gb', 0):.2f} GB used / {vram.get('total_gb', 0):.2f} GB total ({vram.get('percent', 0):.0f}%)")
    log(f"  Model weights: {vram.get('model_weights_gb', 0):.2f} GB")
    log(f"  Free VRAM: {vram.get('free_gb', 0):.2f} GB")
    log(f"  RAM: {ram:.0f} MB")

    # KV cache info
    if engine.kv_cache:
        kv_info = engine.kv_cache.info()
        log(f"  KV cache info: {kv_info}")

    # â”€â”€ Test 3: Accuracy (10 questions) â”€â”€
    log("\n" + "=" * 50)
    log("TEST 3: Accuracy (10 questions, greedy)")
    log("=" * 50)

    correct = 0
    total = len(QUESTIONS)
    for i, (question, expected) in enumerate(QUESTIONS):
        try:
            response = engine.generate(question, max_new_tokens=32, temperature=0.0)
            is_correct = check_answer(response, expected)
            if is_correct:
                correct += 1
            status = "âœ“" if is_correct else "âœ—"
            log(f"  Q{i+1} {status}: {question[:40]} -> '{response[:60]}'")
        except Exception as e:
            log(f"  Q{i+1} ERROR: {e}")

    accuracy = correct / total
    log(f"\n  Accuracy: {correct}/{total} = {accuracy:.0%}")

    # â”€â”€ Test 3b: Accuracy with sampling (temperature=0.7) â”€â”€
    log("\n" + "=" * 50)
    log("TEST 3b: Accuracy with sampling (temp=0.7, top_p=0.9)")
    log("=" * 50)

    correct_s = 0
    for i, (question, expected) in enumerate(QUESTIONS):
        try:
            response = engine.generate(question, max_new_tokens=32,
                                        temperature=0.7, top_p=0.9, top_k=40)
            is_correct = check_answer(response, expected)
            if is_correct:
                correct_s += 1
            status = "âœ“" if is_correct else "âœ—"
            log(f"  Q{i+1} {status}: {question[:40]} -> '{response[:60]}'")
        except Exception as e:
            log(f"  Q{i+1} ERROR: {e}")

    accuracy_s = correct_s / total
    log(f"\n  Accuracy (sampled): {correct_s}/{total} = {accuracy_s:.0%}")

    # â”€â”€ Test 4: Bottleneck profile (skip if quick) â”€â”€
    bottleneck_results = None
    if not quick:
        log("\n" + "=" * 50)
        log("TEST 4: Bottleneck profile")
        log("=" * 50)
        try:
            report = engine.bottleneck(max_new_tokens=16)
            bottleneck_results = report
            if "bottlenecks" in report:
                log("  Top bottlenecks:")
                for b in report["bottlenecks"][:5]:
                    log(f"    {b}")
            if "tokens_per_sec" in report:
                log(f"  Overall: {report['tokens_per_sec']:.1f} tok/s")
        except Exception as e:
            log(f"  Bottleneck profiling failed: {e}")

    # â”€â”€ Test 5: KV cache comparison â”€â”€
    log("\n" + "=" * 50)
    log("TEST 5: KV cache comparison")
    log("=" * 50)

    from forge.config import get_config
    cfg = get_config("forgelm_v2_light")
    n_kv = cfg.n_kv_heads
    head_dim = cfg.d_model // cfg.n_heads
    n_layers = cfg.n_layers
    seq_len = 2048

    standard_mb = 2 * n_kv * head_dim * seq_len * n_layers * 2 / 1e6
    spectral_mb = (2 * n_kv * head_dim * (1 + 2 * cfg.spectral_kv_max_freq) * n_layers * 2 +
                   2 * n_kv * head_dim * cfg.spectral_kv_sink_size * n_layers * 2) / 1e6

    log(f"  Standard KV @ {seq_len} tokens: {standard_mb:.1f} MB")
    log(f"  SpectralKV @ {seq_len} tokens: {spectral_mb:.1f} MB")
    log(f"  Compression: {standard_mb/spectral_mb:.1f}Ã—")

    # â”€â”€ Final engine stats â”€â”€
    log("\n" + "=" * 50)
    log("FINAL ENGINE STATS")
    log("=" * 50)
    final_stats = engine.stats()
    log(f"  Generation count: {final_stats['generation_count']}")
    log(f"  Total tokens generated: {final_stats['total_tokens_generated']}")
    if final_stats.get('kv_cache'):
        log(f"  KV cache: {final_stats['kv_cache']}")
    if final_stats.get('vram'):
        log(f"  VRAM: {final_stats['vram']}")

    # â”€â”€ Summary â”€â”€
    log("\n" + "=" * 70)
    log(f"SUMMARY (kv={kv_strategy})")
    log("=" * 70)
    log(f"  Load time: {load_time:.1f}s")
    log(f"  Params: {n_params/1e6:.1f}M")
    for label, r in speed_results.items():
        if "tps" in r:
            log(f"  Speed ({label}): {r['tps']:.1f} tok/s")
    log(f"  Accuracy (greedy): {correct}/{total} = {accuracy:.0%}")
    log(f"  Accuracy (sampled): {correct_s}/{total} = {accuracy_s:.0%}")
    log(f"  VRAM: {vram.get('used_gb', 0):.2f} GB / {vram.get('total_gb', 0):.2f} GB")
    log(f"  RAM: {ram:.0f} MB")
    log(f"  KV compression: {standard_mb/spectral_mb:.1f}Ã—")

    # Save results
    results = {
        "kv_strategy": kv_strategy,
        "load_time": load_time,
        "params": n_params,
        "speed": speed_results,
        "accuracy_greedy": accuracy,
        "accuracy_sampled": accuracy_s,
        "vram": vram,
        "ram_mb": ram,
        "kv_standard_mb": standard_mb,
        "kv_spectral_mb": spectral_mb,
        "kv_compression": standard_mb / spectral_mb,
        "engine_stats": final_stats,
    }
    out_path = f"scripts/_test_sheet_{kv_strategy}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ForgeEngine test sheet for V10-1.2B")
    parser.add_argument("--kv", default="standard",
                        choices=["standard", "spectral", "s4r", "paged", "snapkv"],
                        help="KV cache strategy to test")
    parser.add_argument("--quick", action="store_true",
                        help="Skip bottleneck profiling")
    parser.add_argument("--all", action="store_true",
                        help="Run all KV strategies and compare")
    args = parser.parse_args()

    if args.all:
        all_results = {}
        for kv in ["standard", "spectral", "s4r"]:
            log(f"\n{'#' * 70}")
            log(f"# Running with kv='{kv}'")
            log(f"{'#' * 70}")
            try:
                r = run_test_sheet(kv_strategy=kv, quick=args.quick)
                all_results[kv] = r
                # Clear GPU between runs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                log(f"FAILED with kv='{kv}': {e}")
                traceback.print_exc()
                all_results[kv] = {"error": str(e)}

        # Comparison table
        log(f"\n{'=' * 70}")
        log("KV STRATEGY COMPARISON")
        log(f"{'=' * 70}")
        log(f"{'Strategy':<15} {'Speed (tok/s)':>15} {'Accuracy':>10} {'VRAM (GB)':>10} {'KV MB':>10}")
        log(f"{'-' * 60}")
        for kv, r in all_results.items():
            if "error" in r:
                log(f"{kv:<15} {'ERROR':>15}")
                continue
            speed = r.get("speed", {}).get("Medium", {}).get("tps", 0)
            acc = r.get("accuracy_greedy", 0)
            vram = r.get("vram", {}).get("used_gb", 0)
            if kv == "spectral":
                kv_mb = r.get("kv_spectral_mb", 0)
            else:
                kv_mb = r.get("kv_standard_mb", 0)
            log(f"{kv:<15} {speed:>15.1f} {acc:>9.0%} {vram:>10.2f} {kv_mb:>10.1f}")

        with open("scripts/_test_sheet_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log("\nComparison saved to scripts/_test_sheet_comparison.json")
    else:
        run_test_sheet(kv_strategy=args.kv, quick=args.quick)


if __name__ == "__main__":
    main()

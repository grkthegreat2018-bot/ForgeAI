"""ForgeEngine core backend benchmark — identifies bottlenecks across
KV cache, quantization, decoding, and acceleration strategies.

Usage:
    python scripts/bench_engine.py [--model qwen|forgelm]
"""
import gc
import json
import os
import sys
import time
import traceback

os.environ.setdefault("FORGE_NO_COMPILE", "1")  # avoid triton/inductor issues

import torch
import torch.cuda as cuda


def log(msg):
    print(f"[bench] {msg}", flush=True)

def vram_mb():
    if not cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
    alloc = cuda.memory_allocated() / 1e6
    reserved = cuda.memory_reserved() / 1e6
    free, total = cuda.mem_get_info()
    return {"allocated": round(alloc, 1), "reserved": round(reserved, 1),
            "free": round(free/1e6, 1), "total": round(total/1e6, 1)}

def clear_cache():
    gc.collect()
    if cuda.is_available():
        cuda.empty_cache()
        cuda.synchronize()

# ─── Model setup ───────────────────────────────────────────────────────

def load_forgelm():
    from forge.engine.forge_engine import ForgeEngine
    ckpt = "research/checkpoints/ForgeLM_V2_Light.sft.safetensors"
    cfg = "forgelm_v2_light"
    tok = "research/checkpoints/lfm25_tokenizer"
    log(f"Loading ForgeLM V2 Light from {ckpt}...")
    engine = ForgeEngine.from_checkpoint(
        checkpoint=ckpt, config_name=cfg, tokenizer_path=tok,
        auto_activate=False)
    return engine, "ForgeLM V2 Light (1.2B)"

def load_qwen():
    """Load Qwen 2.5 0.5B by creating a matching config and downloading."""
    from transformers import AutoTokenizer

    from forge.config import ModelConfig
    from forge.engine.forge_engine import ForgeEngine
    from forge.model_loader import ModelLoader

    # Qwen 2.5 0.5B architecture
    qwen_cfg = ModelConfig(
        vocab_size=151936,
        d_model=896,
        n_layers=24,
        n_heads=14,
        n_kv_heads=2,
        intermediate_size=4864,
        attn_type="gqa",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        norm_eps=1e-6,
        use_embed_norm=False,
        use_final_norm=True,
        rope_base=1_000_000.0,
        max_seq_len=32768,
        use_qk_norm=False,  # Qwen 2.5 does NOT have QK-norm (Qwen3 does)
        layer_types=["attention"] * 24,
        tie_embeddings=True,  # Qwen 2.5 0.5B ties embed/head
        use_bitnet=False,
        use_bitnet_residual=False,
        ffn_compression="none",
        nlrq_rank=0,
        use_factorized_embeddings=False,
        embed_factorized_rank=0,
        use_pit=False,
        use_iri_fp4=False,
        use_spectral_kv=False,
        zero_init_residual=True,
        batch_size=1,
        seq_len=2048,
        max_steps=50000,
        warmup_steps=2000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    # Download Qwen 2.5 0.5B
    from huggingface_hub import snapshot_download
    model_id = "Qwen/Qwen2.5-0.5B"
    log(f"Downloading {model_id}...")
    model_dir = snapshot_download(model_id)
    log(f"Downloaded to {model_dir}")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    log(f"Tokenizer loaded: vocab={tok.vocab_size}")

    # Build model from HF checkpoint
    log("Building model from HF checkpoint...")
    model = ModelLoader.build_model_fast(
        qwen_cfg, checkpoint_path=model_dir, dtype=torch.bfloat16)
    model.eval()

    engine = ForgeEngine(model, tok, device="cuda",
                         checkpoint_path=model_dir, auto_activate=False)
    return engine, "Qwen 2.5 0.5B"

# ─── Benchmark helpers ─────────────────────────────────────────────────

PROMPT = "The quick brown fox jumps over the lazy dog. " * 5  # ~60 tokens
PROMPT_LONG = "The quick brown fox jumps over the lazy dog. " * 50  # ~600 tokens

def bench_generate(engine, label, prompt=PROMPT, max_new_tokens=50, n_runs=3):
    """Benchmark generation throughput."""
    log(f"  [{label}] benchmarking {max_new_tokens} tokens, {n_runs} runs...")
    try:
        result = engine.benchmark(prompt, max_new_tokens=max_new_tokens, n_runs=n_runs)
        v = vram_mb()
        log(f"  [{label}] {result['tokens_per_sec']:.1f} tok/s, "
            f"{result['latency_ms']:.0f}ms latency, "
            f"VRAM {v['allocated']}MB alloc / {v['free']}MB free")
        return {**result, "vram_mb": v, "label": label}
    except Exception as e:
        log(f"  [{label}] FAILED: {e}")
        traceback.print_exc()
        return {"label": label, "error": str(e), "vram_mb": vram_mb()}

def bench_strategy(engine, label, **activate_kwargs):
    """Activate a strategy set and benchmark it."""
    clear_cache()
    try:
        engine.activate(**activate_kwargs)
    except Exception as e:
        log(f"  [{label}] activation FAILED: {e}")
        return {"label": label, "error": f"activation: {e}"}
    return bench_generate(engine, label)

def bench_bottleneck(engine, label):
    """Run per-layer bottleneck profiling."""
    log(f"  [{label}] profiling bottlenecks...")
    try:
        result = engine.bottleneck(prompt=PROMPT, max_new_tokens=16)
        v = vram_mb()
        bottlenecks = result.get("bottlenecks", [])[:5]
        for b in bottlenecks:
            log(f"    {b['type']}#{b['index']}: {b['time_ms']:.2f}ms")
        return {**result, "vram_mb": v, "label": label}
    except Exception as e:
        log(f"  [{label}] bottleneck FAILED: {e}")
        traceback.print_exc()
        return {"label": label, "error": str(e)}

# ─── Main benchmark suite ─────────────────────────────────────────────

def run_suite(engine, model_name):
    results = {"model": model_name, "timestamp": time.strftime("%Y-%m-%d %H:%M")}
    v = vram_mb()
    log(f"=== Benchmarking {model_name} ===")
    log(f"Initial VRAM: {v}")

    # 1. Baseline (no activation)
    results["baseline"] = bench_strategy(engine, "baseline")

    # 2. KV cache strategies
    kv_modes = ["standard", "paged", "rotorquant", "hadamard_int4",
                "snapkv", "cpu_offload", "s4r"]
    results["kv_cache"] = {}
    for kv in kv_modes:
        results["kv_cache"][kv] = bench_strategy(
            engine, f"kv={kv}", kv_cache=kv, decoding="standard")

    # 3. Quantization modes
    quant_modes = [None, "int8", "int4", "w8a8", "nvfp4", "fp8"]
    results["quantization"] = {}
    for q in quant_modes:
        qlabel = q or "none"
        results["quantization"][qlabel] = bench_strategy(
            engine, f"quant={qlabel}", quantize=q, kv_cache="standard",
            decoding="standard")

    # 4. Decoding strategies (with standard KV, no quant)
    decode_modes = ["standard", "speculative", "mtp_selfspec", "eagle3"]
    results["decoding"] = {}
    for d in decode_modes:
        results["decoding"][d] = bench_strategy(
            engine, f"decode={d}", decoding=d, kv_cache="standard")

    # 5. Acceleration
    results["acceleration"] = {}
    results["acceleration"]["prefix_cache"] = bench_strategy(
        engine, "prefix_cache", use_prefix_cache=True, kv_cache="standard")
    results["acceleration"]["chunked_prefill"] = bench_strategy(
        engine, "chunked_prefill", use_chunked_prefill=True, kv_cache="standard")
    results["acceleration"]["cuda_graph"] = bench_strategy(
        engine, "cuda_graph", acceleration="cuda_graph", kv_cache="standard")
    results["acceleration"]["compile"] = bench_strategy(
        engine, "compile", use_compile=True, kv_cache="standard")

    # 6. Optimal (auto-activate)
    clear_cache()
    try:
        engine.activate_optimal()
        results["optimal"] = bench_generate(engine, "optimal")
    except Exception as e:
        log(f"  [optimal] FAILED: {e}")
        results["optimal"] = {"error": str(e)}

    # 7. Bottleneck profiling (with optimal activation)
    results["bottleneck"] = bench_bottleneck(engine, "optimal_bottleneck")

    # 8. Long prompt benchmark
    results["long_prompt"] = bench_generate(
        engine, "long_prompt_600tok", prompt=PROMPT_LONG,
        max_new_tokens=50, n_runs=2)

    # 9. Batch generation
    try:
        clear_cache()
        prompts = [PROMPT] * 4
        log("  [batch=4] benchmarking...")
        t0 = time.perf_counter()
        if cuda.is_available():
            cuda.synchronize()
        outputs = engine.generate_batch(prompts, max_new_tokens=30)
        if cuda.is_available():
            cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_tokens = 30 * len(prompts)
        tps = total_tokens / elapsed if elapsed else 0
        v = vram_mb()
        log(f"  [batch=4] {tps:.1f} tok/s, {elapsed:.2f}s, VRAM {v['allocated']}MB")
        results["batch"] = {"tokens_per_sec": tps, "time_s": elapsed,
                           "batch_size": 4, "vram_mb": v}
    except Exception as e:
        log(f"  [batch=4] FAILED: {e}")
        results["batch"] = {"error": str(e)}

    # 10. VRAM summary
    results["vram_final"] = vram_mb()
    log(f"Final VRAM: {results['vram_final']}")

    return results

# ─── Main ──────────────────────────────────────────────────────────────

def main():
    model_type = "qwen" if "--model" in sys.argv and "qwen" in sys.argv else "forgelm"

    if model_type == "qwen":
        engine, name = load_qwen()
    else:
        engine, name = load_forgelm()

    results = run_suite(engine, name)

    # Save results
    out_path = "scripts/_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Results saved to {out_path}")

    # Print summary table
    log("\n=== SUMMARY ===")
    log(f"Model: {results['model']}")
    for category in ["baseline", "kv_cache", "quantization", "decoding",
                     "acceleration", "optimal", "batch"]:
        if category not in results:
            continue
        val = results[category]
        if isinstance(val, dict) and "error" not in val:
            if "tokens_per_sec" in val:
                log(f"  {category}: {val['tokens_per_sec']:.1f} tok/s")
            else:
                for k, v in val.items():
                    if isinstance(v, dict) and "tokens_per_sec" in v:
                        log(f"  {category}/{k}: {v['tokens_per_sec']:.1f} tok/s")
                    elif isinstance(v, dict) and "error" in v:
                        log(f"  {category}/{k}: ERROR")

if __name__ == "__main__":
    main()

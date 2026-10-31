"""Quick re-test: verify quantization state is sticky across activate() calls,
and test if re-loading the engine fresh fixes the 3.5x slowdown."""
import gc
import os

os.environ["FORGE_NO_COMPILE"] = "1"
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

import torch.cuda as cuda


def log(msg):
    print(f"[retest] {msg}", flush=True)

def vram_mb():
    alloc = cuda.memory_allocated() / 1e6
    free, _total = cuda.mem_get_info()
    return {"allocated": round(alloc, 1), "free": round(free/1e6, 1)}

PROMPT = "The quick brown fox jumps over the lazy dog. " * 5

def quick_bench(engine, label, n_runs=3):
    try:
        result = engine.benchmark(PROMPT, max_new_tokens=50, n_runs=n_runs)
        log(f"  [{label}] {result['tokens_per_sec']:.1f} tok/s, VRAM {vram_mb()}")
        return result["tokens_per_sec"]
    except Exception as e:
        log(f"  [{label}] FAILED: {e}")
        return None

def main():
    from forge.engine.forge_engine import ForgeEngine

    ckpt = "research/checkpoints/ForgeLM_V2_Light.sft.safetensors"
    cfg = "forgelm_v2_light"
    tok = "research/checkpoints/lfm25_tokenizer"

    # Test 1: Fresh engine, no quantization, standard KV
    log("=== Test 1: Fresh engine, no quant ===")
    engine = ForgeEngine.from_checkpoint(
        checkpoint=ckpt, config_name=cfg, tokenizer_path=tok,
        auto_activate=False)
    engine.activate(kv_cache="standard", decoding="standard")
    tps1 = quick_bench(engine, "fresh_standard")
    del engine
    gc.collect(); cuda.empty_cache(); cuda.synchronize()

    # Test 2: Fresh engine, int4 quantization, standard KV
    log("=== Test 2: Fresh engine, int4 quant ===")
    engine = ForgeEngine.from_checkpoint(
        checkpoint=ckpt, config_name=cfg, tokenizer_path=tok,
        auto_activate=False)
    engine.activate(kv_cache="standard", decoding="standard", quantize="int4")
    tps2 = quick_bench(engine, "fresh_int4")
    del engine
    gc.collect(); cuda.empty_cache(); cuda.synchronize()

    # Test 3: Fresh engine, w8a8 quantization
    log("=== Test 3: Fresh engine, w8a8 quant ===")
    engine = ForgeEngine.from_checkpoint(
        checkpoint=ckpt, config_name=cfg, tokenizer_path=tok,
        auto_activate=False)
    engine.activate(kv_cache="standard", decoding="standard", quantize="w8a8")
    tps3 = quick_bench(engine, "fresh_w8a8")
    del engine
    gc.collect(); cuda.empty_cache(); cuda.synchronize()

    # Test 4: Fresh engine, fp8 quantization
    log("=== Test 4: Fresh engine, fp8 quant ===")
    engine = ForgeEngine.from_checkpoint(
        checkpoint=ckpt, config_name=cfg, tokenizer_path=tok,
        auto_activate=False)
    engine.activate(kv_cache="standard", decoding="standard", quantize="fp8")
    tps4 = quick_bench(engine, "fresh_fp8")

    # Test 5: Same engine, deactivate quantization, test if speed recovers
    log("=== Test 5: Same engine, re-activate without quant ===")
    engine.activate(kv_cache="standard", decoding="standard", quantize=None)
    tps5 = quick_bench(engine, "recovered_no_quant")

    # Test 6: Check if generate_batch works with correct API
    log("=== Test 6: generate_batch API check ===")
    try:
        import inspect

        from forge.engine.decoding import BatchedDecoding
        sig = inspect.signature(BatchedDecoding.generate_batch)
        log(f"  generate_batch signature: {sig}")
    except Exception as e:
        log(f"  API check failed: {e}")

    # Test 7: Check prefix cache error
    log("=== Test 7: prefix_cache error trace ===")
    del engine
    gc.collect(); cuda.empty_cache(); cuda.synchronize()
    engine = ForgeEngine.from_checkpoint(
        checkpoint=ckpt, config_name=cfg, tokenizer_path=tok,
        auto_activate=False)
    try:
        engine.activate(kv_cache="standard", decoding="standard", use_prefix_cache=True)
        quick_bench(engine, "prefix_cache")
    except Exception as e:
        import traceback
        log(f"  prefix_cache FAILED: {e}")
        traceback.print_exc()

    log("\n=== SUMMARY ===")
    log(f"Fresh standard (no quant): {tps1} tok/s")
    log(f"Fresh int4: {tps2} tok/s")
    log(f"Fresh w8a8: {tps3} tok/s")
    log(f"Fresh fp8: {tps4} tok/s")
    log(f"Recovered no-quant: {tps5} tok/s")

if __name__ == "__main__":
    main()

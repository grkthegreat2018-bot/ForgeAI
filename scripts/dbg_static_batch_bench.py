"""Bench generate_batch static-graph path on _fq: B in {8}, tok/s target ~1k."""
import sys, time, torch
sys.path.insert(0, ".")
from forge.engine.forge_engine import ForgeEngine

IM = chr(60) + "|im_start|" + chr(62)
IE = chr(60) + "|im_end|" + chr(62)
SEED = "def top_k_mask(logits, k):\n    top = logits.topk(k, dim=-1).values[..., -1:]\n    return logits < top\n"
USER = ("Improve this function for speed. Reply with ONLY a ```python fenced block "
        "containing the complete improved function.\n\n```python\n" + SEED + "```")
PROMPT = IM + "user\n" + USER + IE + "\n" + IM + "assistant\n```python\ndef top_k_mask(logits, k):"

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
eng = ForgeEngine.from_checkpoint(
    "research/checkpoints/lfm3_1p7b_cc_fq", config_name="lfm3_1p7b",
    device="cuda", kv_cache="standard", use_compile=False,
    use_breakable_cuda_graph=True,
    use_block_fusion=False, use_chunked_prefix_cache=True,
    use_learned_prefix_cache=False, use_adaptive_spec=False,
    use_suffix_spec=False, use_faser=False, use_lookahead_gate=False,
    use_hotprefix=False, use_pod_attention=False, use_hybrid_prefill=False)
print("static_batch:", hasattr(eng.model, "_ensure_static_batch"))

# mode=cublas: dequant weights to bf16 once, cuBLAS GEMM per step
if len(sys.argv) > 2 and sys.argv[2] == "cublas":
    import torch.nn as nn
    n = 0
    for m in eng.model.modules():
        if hasattr(m, "_fused_gemv") and hasattr(m, "_dequantize_weight"):
            m._fused_gemv = False
            m._cache_w = True
            n += 1
    print(f"cuBLAS mode: {n} linears switched (fused_gemv=0, cache_w=1)")

prompts = [PROMPT] * B
im_end = chr(60) + "|im_end|" + chr(62)
for trial in range(3):
    t0 = time.time()
    outs = eng.generate_batch(
        prompts, max_new_tokens=200, temperature=0.9, top_p=0.95,
        top_k=80, stops=[[im_end]] * B, skip_special_tokens=True)
    dt = time.time() - t0
    # ~200 tok/row unless EOS'd early; report wall rate on max tokens
    toks = 200 * B
    print(f"B={B} trial={trial}: {dt:.1f}s -> {toks/dt:.0f} tok/s agg (max)")

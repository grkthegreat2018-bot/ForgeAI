"""Token-equality test: static-graph batched decode MUST emit the same
token ids as the functional path for identical prompts/seeds. Any
divergence = garbage (the OOV panic's root cause)."""
import sys, torch
sys.path.insert(0, ".")
from forge.engine.forge_engine import ForgeEngine

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
IM = chr(60) + "|im_start|" + chr(62)
IE = chr(60) + "|im_end|" + chr(62)
SEED = "def top_k_mask(logits, k):\n    top = logits.topk(k, dim=-1).values[..., -1:]\n    return logits < top\n"
USER = ("Improve this function for speed. Reply with ONLY a ```python fenced block "
        "containing the complete improved function.\n\n```python\n" + SEED + "```")
PROMPT = IM + "user\n" + USER + IE + "\n" + IM + "assistant\n```python\ndef top_k_mask(logits, k):"

def build(graphs):
    eng = ForgeEngine.from_checkpoint(
        "research/checkpoints/lfm3_1p7b_cc_fq", config_name="lfm3_1p7b",
        device="cuda", kv_cache="standard", use_compile=False,
        use_breakable_cuda_graph=graphs,
        use_block_fusion=False, use_chunked_prefix_cache=False,
        use_learned_prefix_cache=False, use_adaptive_spec=False,
        use_suffix_spec=False, use_faser=False, use_lookahead_gate=False,
        use_hotprefix=False, use_pod_attention=False,
        use_hybrid_prefill=False)
    for m in eng.model.modules():
        if hasattr(m, "_fused_gemv"):
            m._fused_gemv = False
            m._cache_w = True
    return eng

kw = dict(max_new_tokens=64, temperature=0.0, top_p=1.0, top_k=0)
prompts = [PROMPT] * B

eng0 = build(False)
out0 = eng0.generate_batch(prompts, **kw)
del eng0; torch.cuda.empty_cache()

eng1 = build(True)
out1 = eng1.generate_batch(prompts, **kw)

same = sum(1 for a, b in zip(out0, out1) if a == b)
print(f"B={B}: {same}/{B} rows identical")
for i, (a, b) in enumerate(zip(out0, out1)):
    if a != b:
        print(f"--- row {i} FUNCTIONAL: {a[:120]!r}")
        print(f"--- row {i} GRAPH    : {b[:120]!r}")

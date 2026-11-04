"""A/B: HF generate vs ForgeEngine generate_batch on Qwen2.5-Coder-3B.

Pre-port = HF AutoModel bf16 batched decode.
Post-port = qwen25_coder_3b_mla_moe_fq (INT4-packed GLA latent-KV +
static-KV decode graphs — the production port config).
Same prompts, same max_new, tok/s at batch 1/4/8.
MEM-SAFE: models load sequentially, del + empty_cache between; the
engine side uses the 2.35GB packed checkpoint, not the 6.4GB bf16 one.
"""
import os
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPTS = [
    "Write a Python function `sliding_max(xs, k)` returning the "
    "sliding-window max in O(n). Output one ```python block only.",
    "Write a Python function `dedup_stable(xs)` that removes "
    "duplicates preserving order, O(n). Output one ```python block.",
    "Write a Python function `rmsnorm(x, w, eps)` implementing "
    "RMSNorm over the last dim with torch. One ```python block.",
    "Write a Python function `prefix_sum(xs)` returning the "
    "cumulative sum without numpy. One ```python block.",
    "Write a Python function `kth_largest(xs, k)` O(n) expected "
    "via quickselect. One ```python block.",
    "Write a Python function `batch_normalize(v)` normalizing a "
    "list of vectors to zero mean unit std. One ```python block.",
    "Write a Python function `cosine_sim(a, b)` pure-python cosine "
    "similarity on two vectors. One ```python block.",
    "Write a Python function `topk_freq(xs, k)` returning the k "
    "most frequent items with counts. One ```python block.",
]
MAX_NEW = 300


def bench_hf(bs: int) -> float:
    import torch
    from transformers import (AutoModelForCausalLM,
                              AutoTokenizer)
    hf = "Qwen/Qwen2.5-Coder-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(
        hf, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True) + "```python\n"
        for p in PROMPTS[:bs]]
    enc = tok(texts, return_tensors="pt",
              padding=True).to(model.device)
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=32,
                       do_sample=False,
                       pad_token_id=tok.eos_token_id)   # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    gened = int((out[:, enc.input_ids.shape[1]:]
                 != tok.eos_token_id).sum().item())
    del model, out, enc
    torch.cuda.empty_cache()
    return gened / dt


def bench_forge(bs: int) -> float:
    import torch
    from forge.engine.forge_engine import ForgeEngine
    eng = ForgeEngine.from_checkpoint(
        "research/checkpoints/qwen25_coder_3b_mla_moe_fq",
        use_breakable_cuda_graph=True,
        use_block_fusion=True)
    prompts = PROMPTS[:bs]
    eng.generate_batch(prompts, max_new_tokens=32,
                       temperature=0.0)                # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    raws = eng.generate_batch(prompts, max_new_tokens=MAX_NEW,
                              temperature=0.0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    gened = sum(len(eng.tokenizer.encode(r)) for r in raws)
    del eng
    torch.cuda.empty_cache()
    return gened / dt


def main():
    import torch
    print(f"{'batch':>5} {'HF tok/s':>10} {'Forge tok/s':>12} "
          f"{'VRAM_GB':>8}", flush=True)
    for bs in (1, 4, 8):
        hf_t = forge_t = 0.0
        err = ""
        try:
            hf_t = bench_hf(bs)
        except Exception as e:
            err += f" hf:{str(e)[:60]}"
        v = torch.cuda.memory_allocated() / 1e9
        try:
            forge_t = bench_forge(bs)
        except Exception as e:
            err += f" forge:{str(e)[:60]}"
        print(f"{bs:>5} {hf_t:>10.1f} {forge_t:>12.1f} {v:>8.1f}"
              f"{err}", flush=True)


if __name__ == "__main__":
    main()

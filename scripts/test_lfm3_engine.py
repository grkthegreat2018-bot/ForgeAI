"""LFM3 ForgeEngine smoke test — load + generate + generate_batch on CUDA.

Run:  venv\\Scripts\\python.exe scripts\\test_lfm3_engine.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc"


def main():
    from forge.engine.forge_engine import ForgeEngine

    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        str(CKPT), config_name="lfm3_1p7b",
        kv_cache="standard", use_compile=False,
        use_breakable_cuda_graph=False, use_block_fusion=False,
        use_adaptive_spec=False, use_suffix_spec=False,
        use_faser=False, use_lookahead_gate=False,
        use_hotprefix=False, use_pod_attention=False,
        use_hybrid_prefill=False)
    print(f"engine load: {time.time() - t0:.1f}s  "
          f"vram={torch.cuda.memory_allocated() / 2**30:.2f}GB")

    prompt = ("<|im_start|>user\n"
              "Write a Python function fizzbuzz(n).<|im_end|>\n"
              "<|im_start|>assistant\n")
    out = eng.generate(prompt, max_new_tokens=64, temperature=0.0,
                       top_k=1, finish_sentence=False)
    text = out if isinstance(out, str) else out.get("text", "")
    print("=== single generate ===")
    print(text[:400])

    prompts = [prompt,
               ("<|im_start|>user\n"
                "Write a Python function quicksort(xs).<|im_end|>\n"
                "<|im_start|>assistant\n")]
    t0 = time.time()
    outs = eng.generate_batch(prompts, max_new_tokens=48,
                              temperature=0.0, top_k=1)
    dt = time.time() - t0
    print("=== batch generate ===")
    for i, o in enumerate(outs):
        t = o if isinstance(o, str) else o.get("text", "")
        print(f"-- [{i}] {t[:200]}")
    print(f"batch time {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

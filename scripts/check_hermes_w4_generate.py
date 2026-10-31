"""Hermes-3 8B W4 verification: VRAM-spike fix + prefix-cache corruption fix.

Exercises the real Local_To_Chub path:
  ForgeEngine.from_checkpoint(hermes3-8b-lorablated, llama31_8b,
                              load_quantize="int4") -> engine.generate()

Checks:
  1. Prompt-time VRAM no longer spikes from full [1,T,128k] logits
     (peak delta reported per request).
  2. A second request sharing the prompt prefix (prefix-cache HIT path)
     produces the same greedy output as a fresh no-cache run — previously
     corrupted by the top-left SDPA is_causal mask on T>1 continuation
     forwards.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from forge.engine.forge_engine import ForgeEngine

CKPT = "research/checkpoints/hermes3-8b-lorablated"

IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"

SYSTEM = (
    "You are Hermes, a helpful and knowledgeable AI assistant. "
    "You answer questions accurately and concisely. " * 40
)


def chatml(user: str) -> str:
    return (
        f"{IM_START}system\n{SYSTEM}{IM_END}\n"
        f"{IM_START}user\n{user}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def main():
    print(f"Loading {CKPT} (llama31_8b, load_quantize=int4)...", flush=True)
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        CKPT, config_name="llama31_8b", tokenizer_path=CKPT,
        load_quantize="int4", device="cuda")
    print(f"loaded in {time.time() - t0:.1f}s — "
          f"prefix_cache={type(eng._prefix_cache).__name__}", flush=True)
    base = torch.cuda.memory_allocated() / 2**30
    print(f"resident VRAM: {base:.2f} GiB", flush=True)

    def gen(prompt, n=64):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        out = eng.generate(prompt, max_new_tokens=n, temperature=0.0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        return out, time.time() - t, peak

    prompts = [
        chatml("What is 17 * 23? Show your reasoning briefly."),
        chatml("Name three prime numbers greater than 50."),
    ]

    print("\n=== request 1 (cache miss) ===", flush=True)
    out1, dt1, pk1 = gen(prompts[0])
    print(f"peak VRAM {pk1:.2f} GiB (+{pk1 - base:.2f} transient)  {dt1:.1f}s")
    print(f"OUT1: {out1!r}")

    print("\n=== request 2 (same prompt — prefix-cache HIT) ===", flush=True)
    out2, dt2, pk2 = gen(prompts[0])
    print(f"peak VRAM {pk2:.2f} GiB (+{pk2 - base:.2f} transient)  {dt2:.1f}s")
    print(f"OUT2: {out2!r}")
    print(f"MATCH: {out1 == out2}")

    print("\n=== request 3 (shared prefix, new question — HIT) ===", flush=True)
    out3, dt3, pk3 = gen(prompts[1])
    print(f"peak VRAM {pk3:.2f} GiB (+{pk3 - base:.2f} transient)  {dt3:.1f}s")
    print(f"OUT3: {out3!r}")

    print("\n=== no-cache reference for request 3 ===", flush=True)
    saved_cache = eng._prefix_cache
    eng._prefix_cache = None
    out3b, dt3b, pk3b = gen(prompts[1])
    eng._prefix_cache = saved_cache
    print(f"peak VRAM {pk3b:.2f} GiB  {dt3b:.1f}s")
    print(f"OUT3b: {out3b!r}")
    print(f"MATCH vs cached: {out3 == out3b}")

    # ── Server-style prompt: render_messages_for_config + repeated
    # identical requests (exercises the exact Local_To_Chub cache path)
    print("\n=== server-path: rendered ChatML prompt, repeated hits ===",
          flush=True)
    from forge.self_play.discovery.qwen_adapter import render_messages_for_config
    card = ("Seraphina is a sharp-witted elven ranger traveling with "
            "the user. She speaks plainly and teases often.\n\n")
    hist = "".join(
        f"User: turn {i} done.\nSeraphina: noted, keep walking.\n\n"
        for i in range(30))
    user_blob = card + hist + "User: What's that smell?\nSeraphina:"
    srv_prompt = render_messages_for_config(
        [{"role": "user", "content": user_blob}],
        config_name="llama31_8b")
    print(f"rendered {len(srv_prompt)} chars")
    for i in range(3):
        out, dt, _pk = gen(srv_prompt, n=48)
        n_tok = len(eng.tokenizer.encode(out, add_special_tokens=False))
        print(f"  req{i+1}: {dt:.1f}s out_len={len(out)} "
              f"OUT={out[:120]!r}")
    print(f"cache entries: "
          f"{len(eng._prefix_cache._cache) if eng._prefix_cache is not None else 'none'}")

    print("\nDone.")


if __name__ == "__main__":
    main()

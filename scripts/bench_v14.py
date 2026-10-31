"""V14-RSI bench: params (total/active), VRAM, decode tok/s.

Usage: python scripts/bench_v14.py [--routed] [--bf16-decode N]

Reports the architecture-level numbers that gate the V14-RSI design:
total params (int4 footprint), active params/step (decode bandwidth),
peak VRAM, and eager decode tok/s (engine path adds CUDA graphs + int4 +
MTP self-spec — this script measures the raw-model floor only).
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, r"D:\windsurf\ForgeAI")

from forge.config import get_config
from forge.model.llm import ConfigurableResearchLLM


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _active_params(m):
    """Weight bytes actually read per decode step (routed MoE top-k)."""
    def n(x):
        return sum(p.numel() for p in x.parameters())
    act = n(m.head)
    if m.ln_f is not None:
        act += n(m.ln_f)
    if m.edit_head is not None:
        act += n(m.edit_head)
    if m.verifier_head is not None:
        act += n(m.verifier_head)
    for b in m.blocks:
        act += n(b.attn) + n(b.ln1) + n(b.ln2)
        f = b.ffn
        if hasattr(f, "experts"):
            act += n(f.experts[0]) * f.top_k
            if getattr(f, "has_shared", False):
                act += n(f.shared)
            if hasattr(f, "router"):
                act += n(f.router)
        else:
            act += n(f)
        # Gate-zero side-paths are skipped at forward — only count when open.
        if b._gdn2 is not None and float(b._gdn2.gate) != 0.0:
            act += n(b._gdn2)
        if b._memory is not None and float(b._memory.gate) != 0.0:
            act += n(b._memory)
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routed", action="store_true",
                    help="disable moe_dense_bypass (production decode mode)")
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    cfg = get_config(
        "forgelm_v14_rsi",
        **({"moe_dense_bypass": False} if args.routed else {}))
    m = ConfigurableResearchLLM(cfg)
    total = _params(m)
    active = _active_params(m)
    print(f"total params   : {total/1e9:.3f}B  (int4 ~{total*0.5/1e9:.2f}GB,"
          f" bf16 {total*2/1e9:.2f}GB)")
    print(f"active/step    : {active/1e9:.3f}B  (int4 ~{active*0.5/1e9:.2f}GB"
          f" read -> ~{570/(active*0.5/1e9):.0f} fwd/s @570GB/s)")

    if not torch.cuda.is_available():
        print("no CUDA — param report only")
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m = m.to("cuda", dtype=dtype).eval()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, cfg.vocab_size, (1, args.prefill), device="cuda")
    with torch.inference_mode():
        t0 = time.time()
        logits, _, past, _ = m(
            ids, use_cache=True, return_hidden=True, logits_last_only=True)
        torch.cuda.synchronize()
        print(f"prefill {args.prefill}: {time.time()-t0:.2f}s")
        tok = logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(4):  # warmup
            logits, _, past, _ = m(
                tok, past_key_values=past, use_cache=True,
                return_hidden=True, logits_last_only=True)
            tok = logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.decode):
            logits, _, past, _ = m(
                tok, past_key_values=past, use_cache=True,
                return_hidden=True, logits_last_only=True)
            tok = logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.decode
        print(f"eager decode   : {1/dt:.1f} tok/s ({dt*1e3:.1f} ms/tok)")
        print(f"peak VRAM      : {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()

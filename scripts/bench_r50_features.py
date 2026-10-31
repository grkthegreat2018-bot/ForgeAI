"""R50/R49-2 feature benchmark — real ForgeLM V2 (3.2B bf16, RTX 5070).

Numbers, not assertions: decode tok/s + peak VRAM per feature, rep-3
repetition rate for DRY, KDA param/VRAM overhead + decode cost at
gate=0 (lossless) and gate=1 (worst case, naive fp32 scan), filler-KV
byte savings for V2 (MQA) and qwen3_4b (8 KV heads) geometries.

Run: venv/Scripts/python.exe scripts/bench_r50_features.py
"""
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROWS = []


def row(name, tok_s, vram_gb, note=""):
    ROWS.append((name, tok_s, vram_gb, note))
    print(f"  {name:<28} {tok_s:>7.1f} tok/s  {vram_gb:>5.2f} GiB  {note}")


def bench(gen, model, ids, n, **kw):
    # warmup: first generate on a fresh model/config pays cuBLAS workspace
    # alloc + kernel autotune — discard it so measured runs are hot.
    gen.generate(model, ids, max_new_tokens=8, temperature=0.0, **kw)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = gen.generate(model, ids, max_new_tokens=n, temperature=0.0, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out, n / dt, torch.cuda.max_memory_allocated() / 2**30


def rep_n(ids, n=3):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def main():
    from forge.engine.decoding import DoLaDecoding, StandardDecoding
    from forge.model_loader import load_default_model

    print("Loading ForgeLM V2 (bf16, cuda)...")
    model, tok = load_default_model("forgelm_v2", device="cuda")
    base_gb = torch.cuda.memory_allocated() / 2**30
    print(f"  weights resident: {base_gb:.2f} GiB")

    std = StandardDecoding()
    prompt = ("Write a story about a cat who lives in a small village "
              "and explores the forest every day.")
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    N = 40

    print("\n== Decode throughput (greedy, V2) ==")
    _, s, v = bench(std, model, ids, N)
    row("baseline", s, v)

    _, s, v = bench(std, model, ids, N, dry_multiplier=1.0, dry_base=1.75,
                    dry_allowed_length=2, dry_penalty_last_n=512)
    row("+ DRY", s, v)

    dola_fix = DoLaDecoding(early_layer=8, candidate_top_k=64)
    _, s, v = bench(dola_fix, model, ids, N)
    row("+ DoLa fixed L8", s, v)

    dola_dyn = DoLaDecoding(candidate_top_k=64)
    _, s, v = bench(dola_dyn, model, ids, N)
    row("+ DoLa dynamic", s, v, "JSD over all layers")

    print("\n== DRY quality (rep-3 on repetition-prone prompt) ==")
    rep_prompt = ("The cat sat on the mat. The cat sat on the mat. "
                  "The cat sat on the mat. The cat sat")
    rids = tok(rep_prompt, return_tensors="pt").input_ids.cuda()
    o_off = std.generate(model, rids, max_new_tokens=64, temperature=0.8,
                         top_p=0.9, top_k=64, repetition_penalty=1.0)
    o_on = std.generate(model, rids, max_new_tokens=64, temperature=0.8,
                        top_p=0.9, top_k=64, repetition_penalty=1.0,
                        dry_multiplier=1.0, dry_base=1.75,
                        dry_allowed_length=2, dry_penalty_last_n=512)
    r_off = rep_n(o_off[0, rids.shape[1]:].tolist())
    r_on = rep_n(o_on[0, rids.shape[1]:].tolist())
    print(f"  rep-3  dry=off: {r_off:.3f}   dry=on: {r_on:.3f}   "
          f"({(r_off - r_on) / max(r_off, 1e-9) * 100:+.0f}% repetition)")

    print("\n== KDA overhead ==")
    n_base = sum(p.numel() for p in model.parameters())
    del model
    gc.collect(); torch.cuda.empty_cache()

    import dataclasses

    from safetensors.torch import load_file

    from forge.config import get_config
    from forge.keys.attention.kda_key import KDAKey
    from forge.model.llm import ConfigurableResearchLLM
    from research.paths import V2_CHECKPOINT

    cfg = dataclasses.replace(get_config("forgelm_v2"), use_kda=True)
    key = KDAKey(cfg.d_model, cfg.n_heads,
                 head_dim=cfg.d_model // cfg.n_heads)
    sd = load_file(str(V2_CHECKPOINT))
    res = key.convert_model_state(sd)
    m_kda = ConfigurableResearchLLM(cfg)
    m_kda.load_state_dict(res.weights, strict=True)
    m_kda = m_kda.to("cuda", torch.bfloat16).eval()
    del res, sd
    gc.collect(); torch.cuda.empty_cache()
    n_kda = sum(p.numel() for p in m_kda.parameters())
    kda_gb = torch.cuda.memory_allocated() / 2**30
    d_params = n_kda - n_base
    print(f"  params: {n_base/1e9:.3f}B -> {n_kda/1e9:.3f}B "
          f"(+{d_params/1e6:.0f}M, +{d_params*2/2**30:.2f} GiB bf16)")
    print(f"  weights resident: {kda_gb:.2f} GiB")

    _, s, v = bench(std, m_kda, ids, N)
    row("+ KDA gate=0 (lossless)", s, v, "fast-path skip")

    with torch.no_grad():
        for n_, p in m_kda.named_parameters():
            if n_.endswith("_kda.gate"):
                p.fill_(1.0)
    _, s, v = bench(std, m_kda, ids, N)
    row("+ KDA gate=1 (worst case)", s, v, "naive fp32 scan")
    del m_kda
    gc.collect(); torch.cuda.empty_cache()

    print("\n== Filler-KV byte savings (4096-tok ctx, ~30% filler, "
          "budget 1024) ==")
    from forge.engine.kv.filler_kv import FillerKVCache

    def filler_mb(n_layers, n_kv, head_dim, label):
        c = FillerKVCache(observation_window=128, budget=1024,
                          n_kv_heads=n_kv, head_dim=head_dim,
                          filler_pred=lambda t: t % 10 < 3)
        B = 1
        k = torch.randn(B, n_kv, 1, head_dim, device="cuda",
                        dtype=torch.bfloat16)
        v = torch.randn_like(k)
        scores = torch.rand(B, n_kv, 1, 1, device="cuda",
                            dtype=torch.bfloat16)
        for pos in range(4096):
            c.append(k, v, pos, attention_weights=scores,
                     token_ids=[pos])
        bytes_tok = 2 * n_kv * head_dim * 2 * n_layers
        kept, evict = c.seq_len, 4096 - c.seq_len
        full_mb = 4096 * bytes_tok / 2**20
        kept_mb = kept * bytes_tok / 2**20
        print(f"  {label}: {kept}/{4096} slots kept "
              f"({c.filler_evicted} filler + {c.score_evicted} scored), "
              f"KV {full_mb:.0f} -> {kept_mb:.0f} MiB "
              f"({bytes_tok} B/tok, saved {full_mb-kept_mb:.0f} MiB)")

    filler_mb(2, 1, 128, "V2 (2 attn, MQA)")
    filler_mb(36, 8, 128, "qwen3_4b (36 attn, 8KV)")

    print("\n== Summary ==")
    for name, s, v, note in ROWS:
        print(f"  {name:<28} {s:>7.1f} tok/s  {v:>5.2f} GiB  {note}")


if __name__ == "__main__":
    main()

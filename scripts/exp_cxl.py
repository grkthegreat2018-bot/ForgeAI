"""E-cxl driver: CXL-style pooled/tiered KV cache + weight tier.

Phase A (synthetic KV workload, Qwen2.5-0.5B dims: 24 layers, 2 KV heads,
head_dim 64, bf16): capacity ratio at fixed GPU budget, page-miss
latency, and decode-step tok/s for stream vs fetch vs full-GPU at
8k/32k context.

Phase B (real model): Qwen2.5-0.5B decode over an ~8k-token prompt.
Strategies: full_gpu baseline, tiered fetch-all, tiered stream,
tiered stream + link prefetch, and a weight-tier variant (FFN of last
blocks parked in the CXL pool).  Measures tok/s, H2D traffic, miss
latency, and checks argmax agreement vs baseline.

GPU runs are kept short; results land in .devin/exp_results.jsonl.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from forge.exp.harness import QWEN_ID, load_engine, log_result, wikitext_docs
from forge.exp.cxlmem import (TieredKV, attach_tiered_kv, detach_tiered_kv,
                              kv_bytes_per_token, unwrap_ffn_tiered,
                              wrap_ffn_tiered)
from forge.model_loader import unpack_output_with_kv

# Qwen2.5-0.5B KV geometry
N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 24, 14, 2, 64
DTYPE = torch.bfloat16
TOK_BYTES_LAYER = 2 * N_KV * HEAD_DIM * 2          # k+v, bf16 = 512 B


def _r(*shape):
    return torch.randn(*shape, dtype=DTYPE, device="cuda")


# ── Phase A: synthetic tier bench ─────────────────────────────────────

def synth_fill(tier, seq_len, chunk=2048):
    """Fill all layers with random post-RoPE K/V (bulk appends)."""
    for li in range(tier.n_layers):
        for off in range(0, seq_len, chunk):
            n = min(chunk, seq_len - off)
            tier.append(li, _r(1, N_KV, n, HEAD_DIM), _r(1, N_KV, n, HEAD_DIM))


def synth_step_ms(tier, mode, steps):
    """Simulated decode: one new token per layer per step."""
    if mode == "full_gpu":
        pasts = [(k.clone(), v.clone()) for k, v in
                     (tier.materialize(li) for li in range(tier.n_layers))]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _s in range(steps):
            q = _r(1, N_HEADS, 1, HEAD_DIM)
            kn, vn = _r(1, N_KV, 1, HEAD_DIM), _r(1, N_KV, 1, HEAD_DIM)
            for li in range(tier.n_layers):
                pk, pv = pasts[li]
                k = torch.cat([pk, kn], 2)
                v = torch.cat([pv, vn], 2)
                pasts[li] = (k, v)
                kf = k.repeat_interleave(N_HEADS // N_KV, dim=1)
                vf = v.repeat_interleave(N_HEADS // N_KV, dim=1)
                F.scaled_dot_product_attention(q, kf, vf)
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0) / steps
    tier.link.reset_stats()
    if mode == "prefetch":
        tier.prefetch_ahead = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _s in range(steps):
        q = _r(1, N_HEADS, 1, HEAD_DIM)
        kn, vn = _r(1, N_KV, 1, HEAD_DIM), _r(1, N_KV, 1, HEAD_DIM)
        if mode == "prefetch":
            tier.prefetch_async(0)
        for li in range(tier.n_layers):
            if mode == "stream" or mode == "prefetch":
                tier.attend(li, q, kn, vn)
                if mode == "prefetch":
                    tier.prefetch_async(li + 1)
            else:  # fetch-all
                kf, vf = tier.materialize(li)
                k = torch.cat([kf, kn], 2)
                v = torch.cat([vf, vn], 2)
                F.scaled_dot_product_attention(
                    q, k.repeat_interleave(N_HEADS // N_KV, dim=1),
                    v.repeat_interleave(N_HEADS // N_KV, dim=1))
                tier.append(li, kn, vn)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / steps


def bench_page_miss(tier, layer=0, reps=8):
    """Sync-timed page-miss latency: fetch the first cold page H2D.

    Returns (single_page_ms, burst_ms, burst_pages) — wall-clock,
    not event time, so per-op interconnect latency is honest.
    """
    st = tier._st[layer]
    if st.cold_len == 0 or not tier.link.is_cuda:
        return None
    page = tier.page
    # single page
    dst = torch.empty(page, *st.cold_k.shape[1:], dtype=st.cold_k.dtype,
                      device="cuda")
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(st.cold_k[:page], non_blocking=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    page_ms = sorted(times)[len(times) // 2]
    # burst (8 pages)
    n = min(tier.burst_pages * page, st.cold_len)
    dstb = torch.empty(n, *st.cold_k.shape[1:], dtype=st.cold_k.dtype,
                       device="cuda")
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dstb.copy_(st.cold_k[:n], non_blocking=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return page_ms, sorted(times)[len(times) // 2], n // page


def run_synthetic():
    for seq_len in (8192, 32768):
        window = 1024
        page = 128
        tier = TieredKV(N_LAYERS, gpu_window_tokens=window,
                        page_tokens=page,
                        cxl_capacity_tokens=seq_len + 4096,
                        burst_pages=8, device="cuda")
        synth_fill(tier, seq_len)
        # correctness spot-check: materialize round-trip, layer 0
        kk, vv = tier.materialize(0)
        assert kk.shape[2] == seq_len and vv.shape[2] == seq_len
        st = tier.link.stats
        pm = bench_page_miss(tier)
        log_result(
            "cxl", phase="synth_capacity", seq_len=seq_len, window=window,
            gpu_resident_mb=round(tier.persistent_gpu_bytes() / 1e6, 1),
            full_gpu_mb=round(tier.logical_kv_bytes() / 1e6, 1),
            cxl_pool_mb=round(tier.cpu_pool_bytes() / 1e6, 1),
            capacity_ratio=round(seq_len / window, 1),
            d2h_gb=round(st["d2h_bytes"] / 1e9, 3),
            page_miss_ms=round(pm[0], 4) if pm else None,
            burst_miss_ms=round(pm[1], 4) if pm else None,
            burst_pages=pm[2] if pm else None)
        del kk, vv
        # decode-step cost per strategy; burst=8 (fine) and burst=32
        for burst in (8, 32):
            for mode in ("stream", "fetch", "prefetch"):
                t2 = TieredKV(N_LAYERS, gpu_window_tokens=window,
                              page_tokens=page,
                              cxl_capacity_tokens=seq_len + 4096,
                              burst_pages=burst, device="cuda")
                synth_fill(t2, seq_len)
                t2.link.reset_stats()
                ms = synth_step_ms(t2, mode, steps=16)
                st2 = t2.stats()
                log_result(
                    "cxl", phase="synth_decode", mode=mode,
                    seq_len=seq_len, burst_pages=burst,
                    ms_per_step=round(ms, 2),
                    tok_s=round(1e3 / ms, 1),
                    h2d_mb_per_step=round(st2["h2d_bytes"] / 1e6 / 16, 1),
                    misses=st2["misses"],
                    miss_ms_mean=st2["miss_ms_mean"],
                    fetch_ops=st2["fetch_ops"])
                del t2
                torch.cuda.empty_cache()
        # full-GPU resident baseline
        t2 = TieredKV(N_LAYERS, gpu_window_tokens=window, page_tokens=page,
                      cxl_capacity_tokens=seq_len + 4096, device="cuda")
        synth_fill(t2, seq_len)
        ms = synth_step_ms(t2, "full_gpu", steps=16)
        log_result("cxl", phase="synth_decode", mode="full_gpu",
                   seq_len=seq_len, ms_per_step=round(ms, 2),
                   tok_s=round(1e3 / ms, 1))
        del t2, tier
        torch.cuda.empty_cache()


# ── Phase B: real model decode ────────────────────────────────────────

def build_ids(eng, n_tokens):
    docs = wikitext_docs(96, min_len=200, split="test")
    parts = []
    total = 0
    for t in docs:
        ids = eng.tokenizer(t, return_tensors="pt",
                            add_special_tokens=False).input_ids
        parts.append(ids)
        total += ids.shape[1]
        if total >= n_tokens * 2:
            break
    ids = torch.cat(parts, dim=1)[:, :n_tokens].long().to("cuda")
    if ids.shape[1] < n_tokens:      # pad by repetition if short
        reps = (n_tokens // max(ids.shape[1], 1)) + 1
        ids = ids.repeat(1, reps)[:, :n_tokens]
    return ids


def decode_run(eng, ids, n_tokens, tier=None, mode=None,
               ffn_tier_blocks=0, link=None):
    """One decode bench. Returns (tok_s, first_logits, gen_ids)."""
    model = eng.model
    if tier is not None:
        tier.reset()
        # "prefetch" is the streaming attention path + link read-ahead
        attach_tiered_kv(model, tier, "fetch" if mode == "fetch" else "stream")
    finfo = None
    if ffn_tier_blocks and link is not None:
        ids_off = list(range(N_LAYERS - ffn_tier_blocks, N_LAYERS))
        finfo = wrap_ffn_tiered(model, ids_off, link)
    logits, past = eng._prefill(ids)
    gen = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _i in range(n_tokens):
        tok = logits[:, -1].argmax(-1, keepdim=True)
        gen.append(int(tok[0, 0]))
        with torch.inference_mode():
            out = model(tok, past_key_values=past, use_cache=True)
            logits, past = unpack_output_with_kv(out)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if ffn_tier_blocks:
        unwrap_ffn_tiered(model)
    if tier is not None:
        detach_tiered_kv(model)
    return n_tokens / dt, logits, gen, finfo


def run_model():
    eng = load_engine(QWEN_ID, warmup=False)
    ctx = 8192
    n_tok = 64
    ids = build_ids(eng, ctx)
    log_result("cxl", phase="model_load", load_s=round(eng._exp_load_s, 2),
               vram_gb=round(eng._exp_peak_vram, 2), ctx=ctx)

    # baseline: full GPU-resident KV
    torch.cuda.reset_peak_memory_stats()
    tok_s, logits0, gen0, _ = decode_run(eng, ids, n_tok)
    base_vram = torch.cuda.max_memory_allocated() / 1e9
    log_result("cxl", phase="decode_full_gpu", ctx=ctx, n_tok=n_tok,
               tok_s=round(tok_s, 1), ms_per_tok=round(1e3 / tok_s, 2),
               peak_vram_gb=round(base_vram, 2))

    full_kv_mb = ctx * kv_bytes_per_token(N_LAYERS, N_KV, HEAD_DIM) / 1e6

    for mode in ("fetch", "stream", "prefetch"):
        window = 1024
        tier = TieredKV(N_LAYERS, gpu_window_tokens=window, page_tokens=128,
                        cxl_capacity_tokens=ctx + 4096, burst_pages=32,
                        prefetch_ahead=(mode == "prefetch"), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        tok_s, logits, gen, _ = decode_run(eng, ids, n_tok,
                                           tier=tier, mode=mode)
        st = tier.stats()
        agree = sum(a == b for a, b in zip(gen, gen0)) / len(gen0)
        log_result(
            "cxl", phase=f"decode_{mode}", ctx=ctx, n_tok=n_tok,
            window=window,
            tok_s=round(tok_s, 1), ms_per_tok=round(1e3 / tok_s, 2),
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            kv_resident_mb=round(tier.persistent_gpu_bytes() / 1e6, 1),
            kv_full_mb=round(full_kv_mb, 1),
            capacity_ratio=round(ctx / window, 1),
            h2d_mb=round(st["h2d_bytes"] / 1e6, 1),
            stage_mb=round(st["stage_bytes"] / 1e6, 1),
            misses=st["misses"],
            miss_ms_mean=st["miss_ms_mean"],
            argmax_agree=round(agree, 3))
        del tier
        torch.cuda.empty_cache()

    # weight tier: last 8 blocks' FFN in the CXL pool, stream mode KV
    from forge.exp.cxlmem import CxLLink
    link = CxLLink("cuda")
    tier = TieredKV(N_LAYERS, gpu_window_tokens=1024, page_tokens=128,
                    cxl_capacity_tokens=ctx + 4096, burst_pages=32,
                    prefetch_ahead=True, device="cuda", link=link)
    tok_s, logits, gen, finfo = decode_run(eng, ids, 32, tier=tier,
                                           mode="stream",
                                           ffn_tier_blocks=8, link=link)
    agree = sum(a == b for a, b in zip(gen, gen0[:32])) / 32
    st = tier.stats()
    log_result(
        "cxl", phase="decode_weight_tier", ctx=ctx, n_tok=32,
        ffn_blocks=8, tok_s=round(tok_s, 1),
        ms_per_tok=round(1e3 / tok_s, 2),
        ffn_freed_mb=round(finfo["freed_gpu_bytes"] / 1e6, 1),
        w_h2d_mb=round(st["w_h2d_bytes"] / 1e6, 1),
        argmax_agree=round(agree, 3))
    del tier
    torch.cuda.empty_cache()

    # long-context contrast: 24k prompt, baseline vs stream
    ctx2 = 24576
    ids2 = build_ids(eng, ctx2)
    tok_s, _, _, _ = decode_run(eng, ids2, 32)
    log_result("cxl", phase="decode_full_gpu_24k", ctx=ctx2, n_tok=32,
               tok_s=round(tok_s, 1))
    tier = TieredKV(N_LAYERS, gpu_window_tokens=1024, page_tokens=128,
                    cxl_capacity_tokens=ctx2 + 4096, burst_pages=32,
                    prefetch_ahead=True, device="cuda")
    tok_s, logits, gen, _ = decode_run(eng, ids2, 32, tier=tier,
                                       mode="stream")
    st = tier.stats()
    log_result(
        "cxl", phase="decode_stream_24k", ctx=ctx2, n_tok=32,
        tok_s=round(tok_s, 1), ms_per_tok=round(1e3 / tok_s, 2),
        kv_resident_mb=round(tier.persistent_gpu_bytes() / 1e6, 1),
        kv_full_mb=round(ctx2 * kv_bytes_per_token(N_LAYERS, N_KV,
                                                   HEAD_DIM) / 1e6, 1),
        h2d_mb=round(st["h2d_bytes"] / 1e6, 1),
        miss_ms_mean=st["miss_ms_mean"],
        misses=st["misses"])
    del tier
    torch.cuda.empty_cache()
    print("[cxl] done")


def main():
    run_synthetic()
    run_model()


if __name__ == "__main__":
    main()

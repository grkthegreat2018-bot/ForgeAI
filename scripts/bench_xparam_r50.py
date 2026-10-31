#!/usr/bin/env python
"""R50 Novel Parameter-Format Benchmark — Qwen2.5-0.5B

Evaluates novel weight *storage formats* (forge/engine/quant/novel_quant_r50.py)
against the original bf16 model:
  - weight reconstruction error (rel. Frobenius)
  - perplexity on wikitext-2 (sliding-window chunks)
  - KL divergence + top-1 agreement vs original logits on eval prompts
  - serialized memory (exact bit accounting incl. metadata)
  - decode tok/s with on-the-fly dequant (XParamLinear forward decodes every call)

Usage:
  venv\\Scripts\\python.exe scripts\\bench_xparam_r50.py --formats esc_q,prcb
  venv\\Scripts\\python.exe scripts\\bench_xparam_r50.py --formats all --include-embed
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.engine.quant.novel_quant_r50 import (
    XParamLinear,
    estimate_xparam_memory,
    quantize_model_xparam,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_eval_data(tokenizer, device, n_chunks=8, seq=256):
    """wikitext-2 test -> list of (n_chunks, seq) token tensors."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(n_chunks):
        off = i * seq
        chunks.append(ids[off:off + seq].unsqueeze(0).to(device))
    return chunks


@torch.no_grad()
def eval_quality(model_q, model_fp, chunks):
    """PPL (mean over chunks), KL(fp||q) on logits, top-1 agreement."""
    model_q.eval(); model_fp.eval()
    ppls, kls, agrees = [], [], []
    loss_fn = nn.CrossEntropyLoss()
    for ids in chunks:
        lo = model_fp(ids).logits
        lq = model_q(ids).logits
        sl = lq[:, :-1].reshape(-1, lq.size(-1))
        tl = ids[:, 1:].reshape(-1)
        ppls.append(math.exp(loss_fn(sl, tl).item()))
        p_fp = F.log_softmax(lo[:, :-1].float(), -1)
        p_q = F.log_softmax(lq[:, :-1].float(), -1)
        kl = F.kl_div(p_q, p_fp, log_target=True, reduction="batchmean")
        kls.append(kl.item())
        agrees.append((lo[:, :-1].argmax(-1) == lq[:, :-1].argmax(-1))
                      .float().mean().item())
    return (sum(ppls) / len(ppls), sum(kls) / len(kls),
            sum(agrees) / len(agrees))


@torch.no_grad()
def measure_speed(model, ids, device, n_tokens=40, warmup=8):
    model.eval()
    x = ids[:, :32].clone()
    for _ in range(warmup):
        nx = model(x).logits[:, -1:].argmax(-1)
        x = torch.cat([x, nx], 1)[:, -48:]
    x = ids[:, :32].clone()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_tokens):
        nx = model(x).logits[:, -1:].argmax(-1)
        x = torch.cat([x, nx], 1)[:, -48:]
    if device == "cuda":
        torch.cuda.synchronize()
    return n_tokens / (time.time() - t0)


@torch.no_grad()
def recon_err(model_fp, model_q):
    errs = []
    fp_lin = {n: m for n, m in model_fp.named_modules() if isinstance(m, nn.Linear)}
    for n, m in model_q.named_modules():
        if isinstance(m, XParamLinear) and n in fp_lin:
            ow = fp_lin[n].weight.float()
            qw = m._dequantize_weight(torch.float32)
            errs.append(((ow - qw).norm() / ow.norm().clamp(min=1e-8)).item())
    return sum(errs) / max(len(errs), 1)


def run(formats, device, include_embed, codec_kwargs, n_chunks=8, n_gen=40):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_fp = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model_fp.parameters())
    fp_mb = sum(p.numel() * p.element_size() for p in model_fp.parameters()) / 2**20
    print(f"params {n_params/1e6:.0f}M, bf16 {fp_mb:.0f} MB")

    chunks = build_eval_data(tok, device, n_chunks=n_chunks)
    print(f"eval: {n_chunks} chunks x {chunks[0].shape[1]} tokens")

    fp_ppl, _, _ = eval_quality(model_fp, model_fp, chunks[:2])
    fp_ppl = eval_quality(model_fp, model_fp, chunks)[0]
    fp_speed = measure_speed(model_fp, chunks[0], device, n_gen)
    print(f"FP16: ppl={fp_ppl:.3f} speed={fp_speed:.1f} tok/s")

    results = []
    for fmt in formats:
        print(f"\n--- {fmt} {codec_kwargs.get(fmt, {})} ---")
        free_mem()
        model_q = copy.deepcopy(model_fp)
        kw = codec_kwargs.get(fmt, {})
        t0 = time.time()
        try:
            n_layers = quantize_model_xparam(
                model_q, fmt, verbose=False, include_embed=include_embed, **kw)
        except Exception as e:
            print(f"  ENCODE FAIL: {e}")
            import traceback; traceback.print_exc()
            del model_q; free_mem(); continue
        enc_t = time.time() - t0
        mem = estimate_xparam_memory(model_q)
        model_q.to(device)
        try:
            ppl, kl, agree = eval_quality(model_q, model_fp, chunks)
        except Exception as e:
            print(f"  EVAL FAIL: {e}")
            ppl, kl, agree = float("nan"), float("nan"), float("nan")
        try:
            speed = measure_speed(model_q, chunks[0], device, n_gen)
        except Exception as e:
            print(f"  SPEED FAIL: {e}"); speed = 0.0
        re = recon_err(model_fp, model_q)
        print(f"  layers={n_layers} enc={enc_t:.1f}s  quant={mem['quant_mb']:.0f}MB "
              f"total={mem['total_mb']:.0f}MB avg_bpw={mem['avg_bpw']:.3f}")
        print(f"  ppl={ppl:.3f} (d={ppl-fp_ppl:+.3f})  kl={kl:.5f} "
              f"top1={agree:.4f}  recon={re:.4f}  speed={speed:.1f} tok/s "
              f"({speed/fp_speed:.2f}x)")
        results.append({"format": fmt, "kw": kw, "ppl": ppl,
                        "dppl": ppl - fp_ppl, "kl": kl, "top1": agree,
                        "recon": re, "speed": speed,
                        "speed_ratio": speed / fp_speed,
                        "quant_mb": mem["quant_mb"], "total_mb": mem["total_mb"],
                        "avg_bpw": mem["avg_bpw"], "enc_s": enc_t})
        del model_q; free_mem()

    print(f"\n{'='*104}")
    print(f"{'format':<16} {'bpw':>6} {'quantMB':>8} {'ppl':>8} {'dppl':>8} "
          f"{'kl':>9} {'top1':>7} {'recon':>7} {'tok/s':>7}")
    print(f"{'bf16':<16} {16.0:>6} {fp_mb:>8.0f} {fp_ppl:>8.3f} {'':>8} "
          f"{'':>9} {'':>7} {'':>7} {fp_speed:>7.1f}")
    for r in results:
        print(f"{r['format']:<16} {r['avg_bpw']:>6.3f} {r['quant_mb']:>8.0f} "
              f"{r['ppl']:>8.3f} {r['dppl']:>+8.3f} {r['kl']:>9.5f} "
              f"{r['top1']:>7.4f} {r['recon']:>7.4f} {r['speed']:>7.1f}")
    out = {"model": MODEL_ID, "fp_ppl": fp_ppl, "fp_speed": fp_speed,
           "fp_mb": fp_mb, "include_embed": include_embed, "results": results}
    os.makedirs("scripts", exist_ok=True)
    with open("scripts/r50_xparam_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved scripts/r50_xparam_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="esc_q,prcb,geoq,ash_q,mrq,ppc_w")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--include-embed", action="store_true")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n-levels", type=int, default=5)
    ap.add_argument("--r", type=float, default=1.5)
    args = ap.parse_args()

    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    kw = {}
    for f in fmts:
        if f in ("int_u", "esc_q", "ash_q", "geoq", "dpcm_w", "det_q"):
            kw[f] = {"bits": args.bits, "bs": args.bs}
        if f == "geoq":
            kw[f] = {"bits": args.bits, "bs": args.bs, "r": args.r}
        if f == "mrq":
            kw[f] = {"n_levels": args.n_levels, "bs": args.bs, "lloyd": True}
        if f == "ppc_w":
            kw[f] = {"r_bits": 4, "t_bits": 4, "bs": args.bs}
        if f == "prcb":
            kw[f] = {"bits": args.bits, "lloyd_iters": 30}
        if f == "inr_w":
            kw[f] = {"hid": 32, "steps": 300, "res_bits": args.bits}
        if f == "dctq":
            kw[f] = {"bits_hi": 8, "bits_lo": args.bits, "frac": 0.25}
        if f == "pairrot":
            kw[f] = {"bu": 5, "bv": 3, "bs": args.bs}
    run(fmts, args.device, args.include_embed, kw, args.chunks, args.gen)

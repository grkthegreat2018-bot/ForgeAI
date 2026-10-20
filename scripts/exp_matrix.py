"""EXP-1 speed/memory matrix: decode x KV x quant sweep on Qwen2.5-0.5B.

Phase A: decoding x kv_cache on bf16 weights.
Phase B: quantize sweep (weights restored between cells by engine).
Logs to .devin/exp_results.jsonl.
"""
import os, sys, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from forge.exp.harness import (QWEN_ID, load_engine, bench_engine,
                               bench_decode, log_result, wikitext_docs)

PROMPT = "The quick brown fox jumps over the lazy dog and runs through the forest."
DECODINGS = ["standard", "ngram_speculative", "uno", "dola", "batched"]
KVS = ["standard", "paged", "rotorquant", "snapkv"]
QUANTS = [None, "int8", "w8a8", "fp8", "int4", "nvfp4", "forge_quant",
          "ternary_ptq"]


def engine_nll(eng, texts, device="cuda", max_len=256):
    """NLL via the Forge-side model (targets = next-token labels)."""
    eng.model.eval()
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = eng.tokenizer(t, return_tensors="pt",
                                truncation=True,
                                max_length=max_len).input_ids.to(device)
            if ids.shape[1] < 8:
                continue
            out = eng.model(ids[:, :-1].long(),
                            targets=ids[:, 1:].long())
            loss = out[1]
            nll_sum += loss.item() * (ids.shape[1] - 1)
            n += ids.shape[1] - 1
    return nll_sum / max(n, 1)


def main():
    texts = wikitext_docs(4)
    eng = load_engine(QWEN_ID, warmup=False)
    print(f"[matrix] loaded in {eng._exp_load_s:.1f}s, "
          f"peak {eng._exp_peak_vram:.2f}GB")

    nll0 = engine_nll(eng, texts)
    log_result("matrix", phase="baseline_nll", nll=round(nll0, 4))

    # Phase A: decode x KV on bf16
    for d in DECODINGS:
        for k in KVS:
            cell = f"{d}|{k}"
            try:
                eng.activate(decoding=d, kv_cache=k, quantize=None,
                             warmup=False)
                r = bench_decode(eng, PROMPT)
                log_result("matrix", phase="decode_kv", cell=cell,
                           tok_s=r["decode_tok_s"],
                           ms_tok=r["decode_ms_per_tok"],
                           vram=round(torch.cuda.max_memory_allocated() / 1e9, 3))
            except Exception as e:
                log_result("matrix", phase="decode_kv", cell=cell,
                           error=type(e).__name__ + ": " + str(e)[:120])
                traceback.print_exc(limit=1)

    # Phase B: quantize sweep (standard decode + standard KV)
    for q in QUANTS:
        cell = f"quant|{q}"
        try:
            eng.activate(decoding="standard", kv_cache="standard",
                         quantize=q, warmup=False)
            r = bench_decode(eng, PROMPT)
            nll = engine_nll(eng, texts)
            log_result("matrix", phase="quant", cell=cell,
                       tok_s=r["decode_tok_s"], ms_tok=r["decode_ms_per_tok"],
                       vram=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                       nll=round(nll, 4),
                       dnll=round(nll - nll0, 4))
        except Exception as e:
            log_result("matrix", phase="quant", cell=cell,
                       error=type(e).__name__ + ": " + str(e)[:120])
            traceback.print_exc(limit=1)

    print("[matrix] done")


if __name__ == "__main__":
    main()

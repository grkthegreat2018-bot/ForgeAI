"""Shared harness for the EXP-1 R&D round on Qwen2.5-0.5B.

Common pieces: snapshot resolution, HF/Forge loading, NLL eval,
tok/s bench, JSONL result logging to .devin/exp_results.jsonl.
"""
from __future__ import annotations

import glob
import json
import os
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, ".devin", "exp_results.jsonl")

QWEN_ID = "Qwen/Qwen2.5-0.5B"
QWEN_CODER_ID = "Qwen/Qwen2.5-Coder-0.5B"


def snap_dir(repo_id: str) -> str:
    """Local HF snapshot dir for a cached model."""
    folder = "models--" + repo_id.replace("/", "--")
    hits = glob.glob(os.path.expandvars(
        rf"%USERPROFILE%\.cache\huggingface\hub\{folder}\snapshots\*"))
    if not hits:
        raise FileNotFoundError(f"no HF snapshot for {repo_id}")
    return hits[0]


def load_engine(repo_id: str = QWEN_ID, **kw):
    """ForgeEngine on a HF snapshot. Defaults: clean bf16, no cudagraph
    trees (crashes on this box), standard KV, standard decode."""
    from forge.engine.forge_engine import ForgeEngine
    kw.setdefault("config_name", "qwen25_05b")
    kw.setdefault("device", "cuda")
    kw.setdefault("quantize", None)
    kw.setdefault("kv_cache", "standard")
    kw.setdefault("use_compile", False)
    kw.setdefault("use_breakable_cuda_graph", False)
    kw.setdefault("use_block_fusion", False)
    kw.setdefault("use_learned_prefix_cache", False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = ForgeEngine.from_checkpoint(snap_dir(repo_id), **kw)
    eng._exp_load_s = time.perf_counter() - t0
    eng._exp_peak_vram = torch.cuda.max_memory_allocated() / 1e9
    return eng


def load_hf(repo_id: str = QWEN_ID, device: str = "cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, dtype=dtype).to(device).eval()
    return model, tok


def wikitext_docs(n: int = 16, min_len: int = 200, split: str = "test"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return [t for t in ds["text"] if len(t) > min_len][:n]


@torch.no_grad()
def hf_nll(model, tok, texts, device: str = "cuda") -> float:
    model.eval()
    nll_sum, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        out = model(ids, labels=ids)
        nll_sum += out.loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return nll_sum / max(n, 1)


def bench_engine(eng, prompt: str = "The quick brown fox",
                 max_new_tokens: int = 128, n_runs: int = 5) -> dict:
    torch.cuda.reset_peak_memory_stats()
    res = eng.benchmark(prompt, max_new_tokens=max_new_tokens, n_runs=n_runs)
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return res


def bench_decode(eng, prompt: str = "The quick brown fox",
                 n_tokens: int = 128, n_runs: int = 3) -> dict:
    """Raw per-step decode benchmark — ignores EOS so short-stop models
    (base Qwen) still give a valid tok/s. Times the model forward loop
    only (no detokenization)."""
    from forge.model_loader import unpack_output_with_kv
    ids = eng._tokenize(prompt)
    # warmup
    logits, past = eng._prefill(ids)
    with torch.inference_mode():
        out = eng.model(logits[:, -1].argmax(-1, keepdim=True),
                        past_key_values=past, use_cache=True)
    del past, out
    rates = []
    for _ in range(n_runs):
        logits, past = eng._prefill(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _i in range(n_tokens):
            tok = logits[:, -1].argmax(-1, keepdim=True)
            with torch.inference_mode():
                out = eng.model(tok, past_key_values=past, use_cache=True)
                logits, past = unpack_output_with_kv(out)
        torch.cuda.synchronize()
        rates.append(n_tokens / (time.perf_counter() - t0))
        del past
    return {"decode_tok_s": round(min(rates) if not rates else
                                sum(rates) / len(rates), 1),
            "decode_ms_per_tok": round(1000.0 / (sum(rates) / len(rates)), 2),
            "runs": n_runs, "n_tokens": n_tokens}


def log_result(exp: str, **kv):
    kv["exp"] = exp
    kv["ts"] = time.strftime("%H:%M:%S")
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(kv) + "\n")
    print(f"[{exp}] " + " ".join(f"{k}={v}" for k, v in kv.items()
                                 if k not in ("exp", "ts")))


def vram_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9

"""Spark gen-speed matrix — which decode stack brute-forces winners fastest.

Usage: python scripts/bench_spark_genmatrix.py MODE
  MODE in {plain,bcg,bcg_ngram,bcg_suffix,compile_bcg,batch,batch_fq,fq,all}
  "all" runs each mode as its own subprocess (VRAM isolation) and prints
  a summary table at the end.

Each subprocess: load -> warmup -> 3x B=1 gens (160 tok, temp 0.7)
-> optional generate_batch B=8/B=16 aggregate throughput (the RSI
brute-force path). Prints one RESULT line per config for the driver.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
BASE_CKPT = ROOT / "research/checkpoints/spark_x26_1p7b"
FQ_CKPT = ROOT / "research/checkpoints/spark_x26_1p7b_fq"
TEMPLATE_SRC = (BASE_CKPT / "chat_template.jinja").read_text(encoding="utf-8")

PROMPTS = [
    "Write a Python function `f(a)` that returns the sum of even squares <= a.",
    "Write `reverse_words(s)` that reverses word order but keeps punctuation.",
    "Implement `lru_get(cap, ops)` simulating an LRU cache; return final state.",
]
SOLVE_PROMPTS = PROMPTS * 6  # 18 rows for B=8/B=16 slicing
GEN = dict(max_new_tokens=160, temperature=0.7, top_p=0.95, top_k=80,
           repetition_penalty=1.0)


def render(p: str) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template
    text, _ = render_jinja_template(
        conversations=[[{"role": "user", "content": p}]],
        chat_template=TEMPLATE_SRC, add_generation_prompt=True,
        bos_token="")
    return text[0]


def run_mode(mode: str) -> dict:
    import torch
    from forge.engine.forge_engine import ForgeEngine

    fq = mode in ("fq", "batch_fq")
    ckpt = str(FQ_CKPT if fq else BASE_CKPT)
    eng = ForgeEngine.from_checkpoint(
        ckpt, config_name="spark_x26_1p7b", device="cuda",
        auto_activate=False)
    res = {"mode": mode, "ckpt": Path(ckpt).name}

    act = dict(kv_cache="standard", decoding="standard", quantize=None,
               acceleration=None, warmup=True)
    if mode.startswith("bcg") or mode in ("batch", "batch_fq", "compile_bcg"):
        act["use_breakable_cuda_graph"] = True
    if mode == "compile_bcg":
        act["use_compile"] = True
    if mode == "bcg_ngram":
        act["decoding"] = "ngram_speculative"
    if mode == "bcg_suffix":
        eng.enable_suffix_memory()
        act["decoding"] = "suffix_draft"
    eng.activate(**act)
    res["alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)

    prompts = [render(p) for p in PROMPTS]
    t0 = time.perf_counter()
    _ = eng.generate(prompts[0], max_new_tokens=16, temperature=0.0)
    res["warmup_s"] = round(time.perf_counter() - t0, 2)

    # B=1 latency runs
    toks, dt = 0, 0.0
    for p in prompts:
        t0 = time.perf_counter()
        out = eng.generate(p, **GEN)
        dt += time.perf_counter() - t0
        toks += len(eng.tokenizer.encode(out))
    res["b1_toks"] = toks
    res["b1_tok_s"] = round(toks / max(dt, 1e-9), 1)
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # Aggregate throughput (the brute-force mode)
    if mode in ("batch", "batch_fq"):
        blist = os.environ.get("BLIST", "8 16").split()
        for B in (int(b) for b in blist):
            bp = [render(p) for p in SOLVE_PROMPTS[:B]]
            t0 = time.perf_counter()
            outs = eng.generate_batch(
                bp, max_new_tokens=160, temperature=0.7, top_p=0.95,
                top_k=80, repetition_penalty=1.0)
            d = time.perf_counter() - t0
            n = sum(len(eng.tokenizer.encode(o)) for o in outs)
            res[f"b{B}_tok_s"] = round(n / max(d, 1e-9), 1)
            res[f"b{B}_s"] = round(d, 2)
            res[f"peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 2)

    print("RESULT " + json.dumps(res), flush=True)
    return res


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    modes = ["plain", "bcg", "bcg_ngram", "bcg_suffix", "batch", "fq",
             "batch_fq"] if mode == "all" else [mode]
    if len(modes) == 1:
        run_mode(modes[0])
        return
    rows = []
    for m in modes:
        env = dict(os.environ)
        p = subprocess.run(
            [sys.executable, __file__, m], env=env,
            capture_output=True, text=True, timeout=1800)
        hit = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if hit:
            rows.append(json.loads(hit[-1][7:]))
        else:
            rows.append({"mode": m, "error": (p.stderr or p.stdout)[-400:]})
        print(f"[{m}] done rc={p.returncode}", flush=True)
    print("\n=== SPARK GEN MATRIX ===")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()

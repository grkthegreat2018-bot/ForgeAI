"""A5 proposer head-to-head: gate-rate of proposal code per candidate model.

Runs N real evolution proposal prompts through each candidate proposer
and reports the metrics that decide the default — not vibes:

    valid%     = code fences extracted -> compiles -> defines the
                 required function names (what actually gates in evolve)
    syntax%    = compiles at all
    len_med    = median generated chars (verbosity proxy)

    .\\venv\\Scripts\\python.exe scripts\\bench_proposer.py ^
        --models "research/checkpoints/lfm3_1p7b_cc_fq" ^
                 "Qwen/Qwen2.5-Coder-3B-Instruct" --prompts 12

The report (stdout JSON + results/proposer_bench.json) is the A5
upgrade decision: flip the default only if a coder model beats LFM3's
valid-rate on the same prompts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from forge.evolution.code_evolve import _extract_code  # noqa: E402

# Real proposal prompts sampled from the bundled targets — same shape
# the proposer sees in a basin run (instruction + seed source + required
# def names). Kept tiny and deterministic so the comparison is fair.
_SEED_FAST_SUM = '''NAME = "fast_sum"
def fast_sum(xs):
    total = 0.0
    for x in xs:
        total += x
    return total
'''

_PROPOSALS = [
    ("rewrite this kernel to be faster while keeping identical output",
     _SEED_FAST_SUM, ["fast_sum"]),
    ("vectorize the loop with torch while keeping identical output",
     _SEED_FAST_SUM, ["fast_sum"]),
    ("parallelize the reduction using cumulative ops, identical output",
     _SEED_FAST_SUM, ["fast_sum"]),
]


def _score_completion(raw: str, required: list[str]) -> dict:
    code = _extract_code(raw)
    out = {"has_code": bool(code.strip()), "syntax": False,
           "names": False, "chars": len(raw)}
    try:
        compile(code, "<c>", "exec")
        out["syntax"] = True
    except SyntaxError:
        return out
    out["names"] = all(
        re.search(rf"def\s+{re.escape(n)}\b|{re.escape(n)}\s*=",
                  code) for n in required)
    return out


def _load_hf(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


def _hf_generate(model, tok, prompt: str, max_new: int = 512) -> str:
    import torch
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    text += "```python\n"          # same code-fence prefill as LFM3 path
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def bench_model(model_id: str, prompts: list, device: str,
                max_new: int, lora_tag: str | None = None,
                lora_root: str | None = None) -> dict:
    from forge.evolution.code_evolve import (CodeEvolveConfig,
                                            LLMProposer)
    # Go through the real proposer machinery — it picks forge-engine for
    # converted dirs, HF fallback for repo ids.
    cfg = CodeEvolveConfig(target="x", model_id=model_id,
                           proposer_device=device,
                           max_gen_tokens=max_new,
                           lora_tag=lora_tag, lora_root=lora_root)
    prop = LLMProposer(cfg)
    rows = []
    t0 = time.time()
    for instr, seed, required in prompts:
        prompt = (f"{instr}\n\n```python\n{seed}\n```\n\nReturn only "
                  "verbatim Python inside one ```python fence.")
        try:
            raw = prop._gen_raw(prompt)
        except Exception as e:
            rows.append({"error": str(e)[:200]})
            continue
        s = _score_completion(raw, required)
        rows.append(s)
    n = len([r for r in rows if "error" not in r]) or 1
    return {
        "model": model_id,
        "n": len(rows),
        "valid_pct": sum(1 for r in rows
                         if r.get("names")) / len(rows) * 100,
        "syntax_pct": sum(1 for r in rows
                          if r.get("syntax")) / len(rows) * 100,
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", type=int, default=12,
                    help="proposal prompts to cycle (repeats the 3-seed "
                    "grid)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--lora-tag", default=None,
                    help="attach this registry adapter tag to every benched "
                         "model (run trained models in their own invocation)")
    ap.add_argument("--lora-root", default=None,
                    help="adapter registry dir (default: loras_lfm3)")
    ap.add_argument("--out", default="results/proposer_bench.json")
    args = ap.parse_args()

    grid = [_PROPOSALS[i % len(_PROPOSALS)]
            for i in range(args.prompts)]
    reports = []
    for mid in args.models:
        print(f"=== {mid}")
        try:
            reports.append(bench_model(mid, grid, args.device,
                                       args.max_new, args.lora_tag,
                                       args.lora_root))
        except Exception as e:
            reports.append({"model": mid, "fatal": str(e)[:300]})
        print(json.dumps({k: v for k, v in reports[-1].items()
                          if k != "rows"}, indent=2))
        # Release the shared engine before loading the next model —
        # 12GB VRAM can't hold two proposers.
        from forge.evolution.code_evolve import LLMProposer
        LLMProposer.release_engines(fast=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

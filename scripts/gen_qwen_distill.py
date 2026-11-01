"""Qwen-teacher distill set for the LFM3 proposer.

Draws the REAL prompt distribution — logged proposal prompts from
code_programs (stratified per target so every basin is covered) plus
template prompts over every seed in evolution/targets — and asks
Qwen2.5-Coder-3B-Instruct for the completion. Rows that don't extract
to compilable code are dropped (teacher quality filter).

Then: LoRA 'qwen_distill' on lfm3_1p7b_cc via the sft_train path —
teaches Qwen's terse fenced-output style + capability while keeping
the 1.7B resident footprint.

Usage:
    venv\\Scripts\\python.exe scripts/gen_qwen_distill.py \
        --rows 200 --out results/qwen_distill.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from forge.evolution.code_evolve import _extract_code  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"


def _real_prompts(db: str, per_target: int, total: int) -> list[str]:
    """Stratified sample of logged proposal prompts — the true basin
    distribution, deduped by prompt text."""
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT target, prompt FROM code_programs "
        "WHERE prompt IS NOT NULL AND length(prompt) > 200 "
        "AND prompt LIKE '# Rewrite this Python program%' "
        "ORDER BY RANDOM()").fetchall()
    seen: set[str] = set()
    by_target: dict[str, int] = {}
    out: list[str] = []
    for tgt, p in rows:
        if len(out) >= total:
            break
        if by_target.get(tgt, 0) >= per_target:
            continue
        h = p[:300]
        if h in seen:
            continue
        seen.add(h)
        by_target[tgt] = by_target.get(tgt, 0) + 1
        out.append(p)
    return out


def _target_prompts(targets_dir: Path) -> list[str]:
    """One template prompt per target seed — guaranteed basin coverage
    even where logged prompts are thin."""
    import hashlib
    out = []
    for f in sorted(targets_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        spec = {}
        src = f.read_text(encoding="utf-8", errors="replace")
        head = src[:2500]
        for k in ("NAME", "GOAL", "METRIC", "REQUIRED", "BUDGETS"):
            m = re.search(rf"^{k}\s*=\s*(.+)$", head, re.M)
            if m:
                try:
                    spec[k] = eval(m.group(1))
                except Exception:
                    pass
        m = re.search(r"SEED_CODE\s*=\s*'((?:[^'\\]|\\.)*)'", src, re.S)
        seed = m.group(1).encode().decode('unicode_escape') if m else ""
        if not seed.strip():
            m = re.search(r'SEED_CODE\s*=\s*"((?:[^"\\]|\\.)*)"', src, re.S)
            seed = m.group(1).encode().decode('unicode_escape') if m else ""
        if not seed.strip():
            continue
        required = spec.get("REQUIRED", [])
        tag = hashlib.sha1(seed.encode()).hexdigest()[:8]
        metric = spec.get("METRIC", "score")
        goal = spec.get("GOAL", "min").upper() + "IMIZE"
        goal = "MINIMIZE" if "MIN" in goal else "MAXIMIZE"
        out.append(
            "# Rewrite this Python program to be much better.\n"
            f"# Candidate: {tag}\n"
            f"# Direction: {metric} {goal}\n"
            f"# Hard budgets (must satisfy): {spec.get('BUDGETS', {})}\n"
            f"# Rules: keep public names ({', '.join(required)}); "
            "stdlib+numpy+torch only;\n"
            "# output ONLY one python code block; identical semantics.\n"
            "#\n# Current best:\n"
            f"```python\n{seed.strip()}\n```\n"
            "Reply with ONLY one ```python code block containing the "
            "complete improved implementation.")
    return out


def _gen(model, tok, prompt: str, max_new: int = 700) -> str:
    import torch
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    text += "```python\n"
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--per-target", type=int, default=25)
    ap.add_argument("--out", default="results/qwen_distill.jsonl")
    ap.add_argument("--max-new", type=int, default=700)
    args = ap.parse_args()

    prompts = _real_prompts(args.db, args.per_target, args.rows)
    seed_prompts = _target_prompts(REPO / "forge" / "evolution" / "targets")
    prompts = prompts + [p for p in seed_prompts if p[:300]
                         not in {q[:300] for q in prompts}]
    print(f"[gen] {len(prompts)} prompts "
          f"({len(seed_prompts)} target-template)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            try:
                raw = _gen(model, tok, p, args.max_new)
            except Exception as e:
                print(f"[gen] {i} gen error: {str(e)[:120]}", flush=True)
                n_bad += 1
                continue
            code = _extract_code(raw)
            ok = bool(code.strip())
            if ok:
                try:
                    compile(code, "<c>", "exec")
                except SyntaxError:
                    ok = False
            if ok:
                f.write(json.dumps({"prompt": p, "response": raw,
                                    "target": "qwen_distill"}) + "\n")
                n_ok += 1
            else:
                n_bad += 1
            if (i + 1) % 20 == 0:
                print(f"[gen] {i+1}/{len(prompts)} ok={n_ok} bad={n_bad}",
                      flush=True)
    print(f"[gen] done: {n_ok} usable / {n_bad} dropped -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

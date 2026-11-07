"""Fetch external code-task DBs into the selfgen task-pool format.

First supply per AGENTS.md "Spark selfgen staged plan": OpenCoder-LLM/
opc-sft-stage2 `educational_instruct` (compiler-validated triples:
instruction + reference code + literal assert testcases).

Output: JSON list of {name, code, desc, tests:[{args, expected}], key,
source}. Only rows with >=2 fully-literal asserts are kept (tree/object
setup lines can't be literal-evaled and are skipped).

Usage:
  python scripts/fetch_external_tasks.py --limit 500 `
      --out data/external/opc_sft2_tasks.json
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def literal_asserts(testcases, name):
    """Parse `assert name(<literal args>) == <literal expected>` lines."""
    out = []
    for tc in testcases or []:
        m = re.match(
            rf"\s*assert\s+{re.escape(name)}\s*\((.*?)\)\s*==\s*(.+?)\s*$",
            tc.strip())
        if not m:
            continue
        args, expected = m.group(1).strip(), m.group(2).strip()
        try:
            ast.literal_eval(f"({args},)")
            ast.literal_eval(expected)
        except Exception:
            continue
        out.append({"args": args, "expected": expected})
    return out


def reflection_rows(row):
    """ReflectEvo D+ -> up to 2 {prompt, response} rows teaching the
    self-repair loop faithfully (instruction/response stay consistent):
      reflect: reflect_prompt (embeds question+failed attempt) -> reflections
      repair:  question + own error analysis -> corrected reasoning + answer
    The raw reason_prompt[1] is unusable (its history leaks the correct
    trajectory), so the repair prompt is synthesized without answer leak."""
    q = (row.get("question") or "").strip()
    refl = (row.get("reflections") or "").strip()
    rp = (row.get("reflect_prompt") or "").strip()
    cot = (row.get("second_trial_reasoning") or "").strip()
    ans = (row.get("second_trial_answer") or row.get("answer") or "").strip()
    out = []
    if rp and refl:
        out.append(dict(prompt=rp, response=refl,
                        source="reflectionevo_reflect"))
    if q and refl and cot and ans:
        prompt = (q + "\n\nReflection on your previous incorrect "
                  "attempt:\n" + refl)
        out.append(dict(prompt=prompt,
                        response=cot + f"\nAnswer: {ans}",
                        source="reflectionevo_repair"))
    elif q and cot and ans:
        out.append(dict(prompt=q, response=cot + f"\nAnswer: {ans}",
                        source="reflectionevo"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="OpenCoder-LLM/opc-sft-stage2")
    ap.add_argument("--config", default="educational_instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--kind", choices=["code", "reflection"], default="code")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N eligible rows (held-out slices)")
    ap.add_argument("--out", default="data/external/opc_sft2_tasks.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split=args.split,
                      streaming=True)
    if args.kind == "reflection":
        rows = []
        for row in ds:
            rows.extend(reflection_rows(row))
            if len(rows) >= args.limit:
                break
        rows = rows[:args.limit]
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} reflection rows -> {out}")
        return
    tasks = []
    n_seen = 0
    n_eligible = 0
    for row in ds:
        n_seen += 1
        name = (row.get("entry_point") or "").strip()
        code = (row.get("code") or "").strip()
        desc = (row.get("instruction") or "").strip()
        if not name or not code or not desc:
            continue
        try:
            compile(code, "<row>", "exec")
        except SyntaxError:
            continue
        tests = literal_asserts(row.get("testcase"), name)
        if len(tests) < 2:
            continue
        if len({t["expected"] for t in tests}) < 2:
            continue  # constant-output tasks carry no signal
        if n_eligible < args.skip:
            n_eligible += 1
            continue
        n_eligible += 1
        sig_m = re.search(r"def\s+\w+\s*(\([^)]*\))", code, re.S)
        signature = re.sub(r"\s+", " ", sig_m.group(1)) if sig_m else "(...)"
        tasks.append(dict(name=name, code=code, desc=desc, tests=tests,
                          signature=signature,
                          key=hashlib.md5((name + code).encode())
                          .hexdigest()[:12], source="opc_sft2"))
        if len(tasks) >= args.limit:
            break
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(tasks)} tasks (from {n_seen} rows) -> {out}")


if __name__ == "__main__":
    main()

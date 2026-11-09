"""Runnable-task eval (the "240-task" harness): generate a function per
task, then exec it against the task's arg/expected tests in a separate
subprocess (generated code is untrusted — never exec in-process).

  python scripts/eval_runnable_tasks.py --ckpt research/checkpoints/X \
      --config spark_x26_1p7b --n 240 --out .devin/gen_X.jsonl
  python scripts/eval_runnable_tasks.py --run-tests .devin/gen_X.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time


def _extract_fn(text: str, name: str) -> str:
    """Pull the first def block out of a completion."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    lines = code.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("def "):
            start = i
            break
    if start is None:
        return ""
    body = [lines[start]]
    for ln in lines[start + 1:]:
        if ln and not ln[0] in " \t#":
            break
        body.append(ln)
    return "\n".join(body)


def cmd_generate(args):
    sys.path.insert(0, ".")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from forge.engine.forge_engine import ForgeEngine

    tasks = json.load(open(args.tasks))[: args.n]
    e = ForgeEngine.from_checkpoint(
        checkpoint=args.ckpt, config_name=args.config, auto_activate=False)
    rows = []
    for i, t in enumerate(tasks):
        prompt = (f"# {t['desc']}\n"
                  f"# signature: {t['name']}{t['signature']}\n")
        try:
            out = e.generate(prompt, max_new_tokens=args.max_new,
                             temperature=0.0, finish_sentence=False)
        except Exception as ex:  # noqa: BLE001
            out = f"<GENFAIL {ex}>"
        rows.append({"key": t.get("key", i), "name": t["name"],
                     "signature": t["signature"], "tests": t["tests"],
                     "completion": out})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(tasks)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {args.out}")


def cmd_run_tests(args):
    rows = [json.loads(x) for x in open(args.gens, encoding="utf-8")]
    npass = fail_compile = fail_run = 0
    results = []
    for r in rows:
        code = _extract_fn(r["completion"], r["name"])
        ok = 0
        verdict = "no_fn"
        if code:
            env: dict = {}
            try:
                exec(compile(code, "<gen>", "exec"), env)
                fn = env.get(r["name"])
                if not callable(fn):
                    verdict = "name_missing"
                else:
                    verdict = "pass"
                    for tc in r["tests"]:
                        try:
                            got = fn(*eval(f"({tc['args']},)"))
                            if repr(got) == tc["expected"]:
                                ok += 1
                            else:
                                verdict = "wrong_answer"
                        except Exception:  # noqa: BLE001
                            verdict = "runtime_error"
                    if verdict == "pass" and ok < len(r["tests"]):
                        verdict = "wrong_answer"
            except SyntaxError:
                verdict = "compile_error"
                fail_compile += 1
        passed = verdict == "pass" and ok == len(r["tests"])
        npass += passed
        fail_run += verdict in ("wrong_answer", "runtime_error")
        results.append({"key": r["key"], "name": r["name"],
                        "verdict": verdict, "tests_ok": ok})
    n = len(results)
    print(f"PASS {npass}/{n}  ({100.0 * npass / max(n, 1):.1f}%)")
    from collections import Counter
    print("verdicts:", dict(Counter(r["verdict"] for r in results)))
    if args.report:
        json.dump(results, open(args.report, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="spark_x26_1p7b")
    ap.add_argument("--tasks", default="data/external/opc_sft2_tasks.json")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--out", default="")
    ap.add_argument("--run-tests", dest="gens", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    if args.gens:
        cmd_run_tests(args)
    else:
        assert args.ckpt and args.out, "--ckpt and --out required"
        cmd_generate(args)


if __name__ == "__main__":
    main()

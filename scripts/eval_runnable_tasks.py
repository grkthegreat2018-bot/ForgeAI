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


def _make_prompt(t: dict, fmt: str) -> str:
    """Standard code-eval prompt formats (HumanEval/MBPP conventions)."""
    if fmt == "humaneval":
        # Completion-style: signature + docstring incl. doctest asserts;
        # model literally continues the body (what code LMs are tuned on).
        doc = t["desc"]
        for tc in t["tests"][:2]:
            doc += f"\n    >>> {t['name']}({tc['args']})\n    {tc['expected']}"
        return f"def {t['name']}{t['signature']}:\n    \"\"\"{doc}\n    \"\"\"\n    "
    if fmt == "mbpp":
        # Instruct-style with fenced gen_prefix (evalscope convention).
        tests = "".join(f"assert {t['name']}({tc['args']}) == "
                        f"{tc['expected']}\n" for tc in t["tests"][:3])
        return ("You are an expert Python programmer, and here is your "
                f"task: {t['desc']} Your code should pass these tests:\n\n"
                f"{tests}\n```python\n")
    # 'comment' — original minimal format.
    return (f"# {t['desc']}\n"
            f"# signature: {t['name']}{t['signature']}\n")


def _extract_fn(text: str, name: str, fmt: str = "comment",
                signature: str = "()") -> str:
    """Pull the first def block out of a completion."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    if fmt == "humaneval":
        # Completion was body-only (signature lived in the prompt). If the
        # model re-emitted the def line, strip it; then re-attach so the
        # exec env always sees a complete function.
        lines = code.splitlines()
        while lines and (not lines[0].strip() or
                         lines[0].startswith(("def ", '"""', "#"))):
            lines.pop(0)
        body = [ln for ln in lines if ln.strip()]
        if not body:
            return ""
        if not all(ln.startswith((" ", "\t")) for ln in body):
            body = ["    " + ln for ln in body]
        # Salvage truncation: completions may cut mid-line at max_new —
        # drop trailing lines until the function compiles.
        while body:
            cand = f"def {name}{signature}:\n" + "\n".join(body)
            try:
                compile(cand, "<gen>", "exec")
                return cand
            except SyntaxError:
                body.pop()
        return ""
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
        prompt = _make_prompt(t, args.prompt_format)
        outs = []
        for s in range(args.samples):
            try:
                out = e.generate(
                    prompt, max_new_tokens=args.max_new,
                    temperature=0.0 if s == 0 else args.temp,
                    top_p=0.95 if s > 0 else 1.0,
                    finish_sentence=False)
            except Exception as ex:  # noqa: BLE001
                out = f"<GENFAIL {ex}>"
            outs.append(out)
        rows.append({"key": t.get("key", i), "name": t["name"],
                     "signature": t["signature"], "tests": t["tests"],
                     "fmt": args.prompt_format,
                     "completion": outs[0],
                     "completions": outs if args.samples > 1 else None})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(tasks)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {args.out}")


def _judge_one(r: dict, completion: str) -> tuple[str, int]:
    code = _extract_fn(completion, r["name"], r.get("fmt", "comment"),
                       r["signature"])
    if not code:
        return "no_fn", 0
    env: dict = {}
    try:
        exec(compile(code, "<gen>", "exec"), env)
    except SyntaxError:
        return "compile_error", 0
    fn = env.get(r["name"])
    if not callable(fn):
        return "name_missing", 0
    ok = 0
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
    if ok < len(r["tests"]) and verdict == "pass":
        verdict = "wrong_answer"
    return verdict, ok


def _judge_worker(path: str) -> None:
    """Subprocess entry: judge every completion in one task row, printing
    one JSON line each so the parent can harvest partial results before
    killing a hang."""
    row = json.load(open(path, encoding="utf-8"))
    comps = row.get("completions") or [row["completion"]]
    for i, c in enumerate(comps):
        v, ok = _judge_one(row, c)
        print(json.dumps({"i": i, "verdict": v, "ok": ok}), flush=True)


def _judge_row_subproc(row: dict, timeout: float = 30.0) -> list[tuple[str, int]]:
    """Judge a row in a worker subprocess. Generated code is untrusted and
    may hang (infinite loops) — the parent enforces the timeout; samples
    not judged before it fires get verdict 'timeout'."""
    import os
    import subprocess
    import tempfile
    comps = row.get("completions") or [row["completion"]]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(row, f)
        path = f.name
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--judge-worker", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        got = {}
        for line in out.splitlines():
            try:
                d = json.loads(line)
                got[d["i"]] = (d["verdict"], d["ok"])
            except Exception:  # noqa: BLE001
                pass
        return [got.get(i, ("timeout", 0)) for i in range(len(comps))]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al.): prob a size-k subset of n samples
    with c correct contains >=1 pass."""
    from math import comb
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def cmd_run_tests(args):
    rows = [json.loads(x) for x in open(args.gens, encoding="utf-8")]
    from collections import Counter
    results = []
    nsamp = 0
    for r in rows:
        comps = r.get("completions") or [r["completion"]]
        if args.judge_inline:
            verdicts = [_judge_one(r, c)[0] for c in comps]
        else:
            verdicts = [v for v, _ in _judge_row_subproc(r, args.judge_timeout)]
        c_pass = verdicts.count("pass")
        nsamp = max(nsamp, len(comps))
        results.append({"key": r["key"], "name": r["name"],
                        "verdict": verdicts[0],
                        "verdicts": verdicts,
                        "n": len(comps), "c": c_pass})
    n = len(results)
    npass = sum(v["verdict"] == "pass" for v in results)
    print(f"PASS@1 {npass}/{n}  ({100.0 * npass / max(n, 1):.1f}%)")
    print("verdicts:", dict(Counter(
        v for r in results for v in r["verdicts"])))
    # format vs solve decomposition
    fn = [r for r in results if r["verdict"] != "no_fn"]
    print(f"fn_emitted {len(fn)}/{n} ({100.0 * len(fn) / n:.1f}%)  "
          f"pass|fn {npass}/{max(len(fn),1)} "
          f"({100.0 * npass / max(len(fn), 1):.1f}%)")
    if nsamp > 1:
        for k in (1, 2, 4, 8):
            if k <= nsamp:
                pk = sum(_pass_at_k(r["n"], r["c"], k)
                         for r in results) / n
                print(f"pass@{k}: {100.0 * pk:.1f}%")
    if args.report:
        json.dump(results, open(args.report, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="spark_x26_1p7b")
    ap.add_argument("--tasks", default="data/external/opc_sft2_tasks.json")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--samples", type=int, default=1,
                    help="completions per task (sample 0 greedy; rest at "
                         "--temp for pass@k)")
    ap.add_argument("--temp", type=float, default=0.8,
                    help="sampling temperature for samples 1..n-1")
    ap.add_argument("--prompt-format", dest="prompt_format",
                    default="comment",
                    choices=["comment", "humaneval", "mbpp"],
                    help="comment=#-desc style (original); humaneval="
                         "def+docstring completion (Chen et al.); "
                         "mbpp=instruct+asserts+fence prefix")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--out", default="")
    ap.add_argument("--run-tests", dest="gens", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--judge-worker", dest="judge_worker", default="",
                    help=argparse.SUPPRESS)
    ap.add_argument("--judge-inline", action="store_true",
                    help="exec generated code in-process (unsafe: no "
                         "timeout — hangs block the whole run)")
    ap.add_argument("--judge-timeout", type=float, default=30.0,
                    help="seconds per task row before the worker is "
                         "killed (default 30)")
    args = ap.parse_args()
    if args.judge_worker:
        _judge_worker(args.judge_worker)
    elif args.gens:
        cmd_run_tests(args)
    else:
        assert args.ckpt and args.out, "--ckpt and --out required"
        cmd_generate(args)


if __name__ == "__main__":
    main()

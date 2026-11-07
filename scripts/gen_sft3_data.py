"""Generate verified SFT data (RFT) from base_fq + sft2_fq.

Two modes:
  proposer — evolve-wrap prompts from forge/evolution/targets (base's
             strength): instruction + seed + required names. Verified rows
             teach completion-continues-open-fence (the runtime prefill).
  direct   — instruction prompts from existing corpora (sft2's
             strength): verified fenced answers.

Verify: _extract_code -> compile() -> required names -> no dangerous
imports. Dedup by normalized-code hash, keep <=2 variants per prompt.
Append-mode JSONL; resume skips finished prompt keys.

Usage:
  python gen_sft3_data.py --model <ckpt> --mode proposer --k 4 --reps 4
  python gen_sft3_data.py --model <ckpt> --mode direct --k 3 --limit 900
  python gen_sft3_data.py --model <ckpt> --mode selfgen --tasks 64 --k 4

selfgen (codebase-grounded anti-overfitting): tasks are REAL functions
harvested from ForgeAI production code (forge/scripts/research, GUI
excluded) — self-contained, stdlib-only, documented, deterministic. The
sandbox executor runs each reference on type-annotation-fuzzed inputs
for ground truth (>=2 distinct outputs required), solver rollouts must
reproduce it on all tests, and only Goldilocks-informative tasks
(0 < solve_rate < 1) are kept. 8-gram novelty gate + resume markers;
the pool is the whole codebase, not a fixed prompt list.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from forge.engine.forge_engine import ForgeEngine  # noqa: E402
from forge.evolution.code_evolve import _extract_code  # noqa: E402

SOS = "<" + chr(0xFF5C) + "start" + chr(0x2581) + "of" + chr(0x2581) + \
    "sentence" + chr(0xFF5C) + ">"
EOS = "<" + chr(0xFF5C) + "end" + chr(0x2581) + "of" + chr(0x2581) + \
    "sentence" + chr(0xFF5C) + ">"

_DANGEROUS = {"os", "sys", "subprocess", "shutil", "ctypes", "socket",
              "requests", "urllib", "pathlib", "builtins", "importlib",
              "pickle", "eval", "exec", "open", "__import__", "compile",
              "globals", "locals", "getattr", "setattr", "delattr"}


def wrap(user_text):
    return (SOS + "<" + chr(0x7C) + "User" + chr(0x7C) + ">" + user_text
            + EOS + SOS + "<" + chr(0x7C) + "Bot" + chr(0x7C) + ">")


def wrap_proposer(user_text):
    """Exact runtime wrap: empty </think> prefill + open python fence."""
    return (SOS + "<" + chr(0x7C) + "User" + chr(0x7C) + ">" + user_text
            + EOS + SOS + "<" + chr(0x7C) + "Bot" + chr(0x7C) + ">"
            + "<" + chr(0x2F) + "think>" + "```python\n")


def safe_code(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _DANGEROUS:
                    return False
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _DANGEROUS:
                return False
        elif isinstance(node, ast.Name) and node.id in _DANGEROUS:
            return False
    return True


def verify(raw: str, required: list[str]) -> str | None:
    code = _extract_code(raw) or ""
    if not code.strip() or not safe_code(code):
        return None
    try:
        compile(code, "<c>", "exec")
    except SyntaxError:
        return None
    if required and not all(
            re.search(rf"def\s+{re.escape(n)}\b|{re.escape(n)}\s*=",
                      code) for n in required):
        return None
    return code


def norm_hash(code: str) -> str:
    n = re.sub(r"\s+", " ", code).strip()
    return hashlib.md5(n.encode()).hexdigest()[:12]


def proposer_prompts(reps: int) -> list[dict]:
    out = []
    tdir = REPO / "forge" / "evolution" / "targets"
    for f in sorted(tdir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        seed = getattr(mod, "SEED_CODE", "")
        req = list(getattr(mod, "REQUIRED", []) or [])
        extra = getattr(mod, "PROMPT", "")
        metric = getattr(mod, "METRIC", "score")
        goal = getattr(mod, "GOAL", "min")
        budgets = getattr(mod, "BUDGETS", None) or {}
        if not seed or not req:
            continue
        bud = json.dumps(budgets) if budgets else "none"
        cand = hashlib.md5(seed.encode()).hexdigest()[:8]
        instr = (
            "# Rewrite this Python program to be much better.\n"
            f"# Candidate: {cand}\n"
            f"# Direction: {metric} "
            f"{'MINIMIZE' if goal == 'min' else 'MAXIMIZE'}\n"
            f"# Hard budgets (must satisfy): {bud}\n"
            f"# Rules: keep public names ({', '.join(req)}); "
            "stdlib+numpy+torch only;\n"
            "# output ONLY one python code block; identical semantics.\n"
            f"# Extra: {extra}\n#\n"
            f"# Current best ({metric}=1.0, metrics={{\"{metric}\": 1.0}}):")
        for _ in range(reps):
            out.append(dict(key=f.stem, instr=instr, seed=seed, req=req))
    return out


def direct_prompts(limit: int) -> list[dict]:
    out, seen = [], set()
    sources = ["data/sft/spark_sft_v1.jsonl",
               "data/sft/distill_v10.scored.jsonl",
               "data/sft/glaive_fc_pythonic.jsonl",
               "data/sft/magicoder.jsonl"]
    for src in sources:
        p = REPO / src
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8", errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = r.get("prompt") or r.get("instruction") or ""
            if not q or len(q) > 3000 or len(q) < 15:
                continue
            h = hashlib.md5(q[:300].encode()).hexdigest()[:12]
            if h in seen:
                continue
            seen.add(h)
            req = []
            m = re.search(r"`([a-zA-Z_][a-zA-Z_0-9]{2,})`", q)
            if m:
                req = [m.group(1)]
            out.append(dict(key=h, instr=q, seed="", req=req))
            if len(out) >= limit * 4:
                break
        if len(out) >= limit * 4:
            break
    import random
    random.Random(7).shuffle(out)
    return out[:limit]


def done_keys(path: Path) -> set:
    s = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                s.add(json.loads(line)["gen_key"])
            except Exception:
                pass
    return s


# ── SELFGEN (codebase-grounded): real functions -> execution-verified RFT ──
# Anti-fixed-pool-overfitting: tasks are harvested from REAL ForgeAI
# production functions (valid, executable, diverse — never a fixed prompt
# pool). The sandbox executes the harvested reference on type-fuzzed
# inputs for ground truth; solver rollouts must MATCH it on all tests;
# only Goldilocks-informative tasks (0 < solve_rate < 1) enter the corpus.
# Measured dead end (2026-11-07): model-authored task proposals (from-
# scratch AND seed-variation) = 0/10 valid on the 1.7B — placeholder
# copies, syntax errors, verbatim seed clones, fence spam. The model
# solves well; it does not invent well. Tasks come from the codebase.

_HARVEST_ROOTS = ("forge", "scripts", "research")
_HARVEST_EXCLUDE_DIRS = {
    "__pycache__", "forge_gui", "forge_gui_server", "tests", "checkpoints",
    "proposer_ckpt", "proposer_mla_moe", "loras", "loras_lfm3", "hf_cache",
    "torch_cache", "tmp", "node_modules", "qwen25-05b", "ruffbase",
}
_HARVEST_ALLOWED_MODULES = {"math", "re", "itertools", "collections",
                            "functools", "string", "operator"}
_HARVEST_BANNED_TOKENS = (
    "torch", "np.", "numpy", "cuda", "open(", "print(", "eval(", "exec(",
    "globals(", "input(", "random", "time.", "datetime", "requests",
    "subprocess", "os.", "sys.", "Path(", "ForgeEngine", "self.",
    "cls.", "__import__", "pickle", "socket", "ctypes", "shutil",
)
_FUZZ_VALUES = {
    "int": ["0", "1", "-3", "42"],
    "float": ["0.0", "1.5", "-2.25", "3.25"],
    "str": ["''", "'a'", "'hello world'", "'a,b,c'"],
    "bool": ["True", "False", "True", "False"],
    "list[int]": ["[]", "[1]", "[1, 2, 3]", "[-4, 0, 7, 7]"],
    "list[str]": ["[]", "['a']", "['a', 'bb']", "['x', 'yy', 'zzz']"],
    "list[float]": ["[]", "[1.0]", "[1.5, -2.0]", "[0.1, 0.2, 0.3]"],
    "list": ["[]", "[1]", "[1, 2, 3]", "[-4, 0, 7, 7]"],
    "dict[str,int]": ["{}", "{'a': 1}", "{'a': 1, 'b': 2}", "{'x': 0}"],
}


def _annotation_str(node) -> str | None:
    """Normalize a param annotation to a fuzzable type key."""
    if node is None:
        return None
    try:
        s = ast.unparse(node)
    except Exception:
        return None
    s = (s.replace("typing.", "").replace("List", "list")
         .replace("Dict", "dict").replace(" ", ""))
    return s if s in _FUZZ_VALUES else None


_NAME_TYPE_RULES = (
    (("s", "text", "string", "prompt", "desc", "word", "sentence", "line",
      "query", "name", "msg", "message"), "str"),
    (("xs", "arr", "array", "nums", "numbers", "values", "items", "seq",
      "lst", "list", "ts", "scores", "ids", "tokens"), "list[int]"),
    (("d", "dct", "dict", "mapping", "counts", "freq", "freqs", "counter",
      "hist", "histogram"), "dict[str,int]"),
    (("flag", "enabled", "disabled", "verbose", "strict", "reverse"), "bool"),
    (("x", "y", "v", "val", "value", "score", "ratio", "alpha", "lr",
      "tol", "eps", "threshold", "rate", "scale", "weight"), "float"),
)


def _name_type(name: str) -> str:
    """Type inference for unannotated params (validated later by the
    reference-execution filter — a wrong guess just drops the task)."""
    n = name.lower()
    if n.startswith(("use_", "is_", "has_", "do_", "should_", "allow_")):
        return "bool"
    for keys, t in _NAME_TYPE_RULES:
        if n in keys:
            return t
    return "int"  # ints are the most common scalar in this codebase


def _function_to_task(node: ast.FunctionDef, source: str,
                      path: Path) -> dict | None:
    """Eligibility: top-level, public, undecorated, 3-45 lines, 1-4
    fully-annotated params, docstring description, stdlib+builtins only,
    no module-level dependencies, deterministic."""
    if (node.name.startswith("_") or node.decorator_list
            or isinstance(node, ast.AsyncFunctionDef)):
        return None
    n_lines = (node.end_lineno or node.lineno) - node.lineno + 1
    if not 3 <= n_lines <= 45:
        return None
    params = list(node.args.posonlyargs) + list(node.args.args)
    if (not 1 <= len(params) <= 4 or node.args.vararg or node.args.kwarg
            or node.args.kwonlyargs):
        return None
    types = [(_annotation_str(p.annotation) or _name_type(p.arg))
             for p in params]
    desc = (ast.get_docstring(node) or "").strip().splitlines()
    desc = desc[0].strip() if desc else ""
    if desc and (len(desc) < 15 or not desc.isascii()):
        return None
    src = ast.get_source_segment(source, node)
    if not src or any(tok in src for tok in _HARVEST_BANNED_TOKENS):
        return None
    # Free-name analysis: every loaded name must be a param, a local, a
    # builtin, an allowed stdlib module, or the function itself (recursion).
    local_names = {p.arg for p in params} | {node.name}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            local_names.add(sub.id)
        elif isinstance(sub, ast.arg):
            local_names.add(sub.arg)
        elif isinstance(sub, (ast.FunctionDef, ast.Lambda)):
            for p in (sub.args.posonlyargs + sub.args.args
                      + sub.args.kwonlyargs):
                local_names.add(p.arg)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            local_names.add(sub.name)
        elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
            for n2 in ast.walk(sub.optional_vars):
                if isinstance(n2, ast.Name):
                    local_names.add(n2.id)
    import builtins as _b
    allowed = (local_names | set(dir(_b)) | _HARVEST_ALLOWED_MODULES
               | {"True", "False", "None"})
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id not in allowed:
                return None
    sig_m = re.search(r"def\s+\w+\s*(\([^)]*\))", src, re.S)
    signature = re.sub(r"\s+", " ", sig_m.group(1)) if sig_m else "(...)"
    try:
        rel = str(path.relative_to(REPO))
    except ValueError:
        rel = path.name
    return dict(name=node.name, code=src, desc=desc, signature=signature,
                types=types, file=rel,
                key=hashlib.md5(src.encode()).hexdigest()[:12])


def harvest_functions(roots=_HARVEST_ROOTS, limit: int = 0) -> list[dict]:
    """Extract execution-ready RFT tasks from production code."""
    out = []
    seen = set()
    for root in roots:
        for f in sorted((REPO / root).rglob("*.py")):
            if any(part in _HARVEST_EXCLUDE_DIRS for part in f.parts):
                continue
            try:
                if f.stat().st_size > 300_000:
                    continue
                source = f.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except Exception:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    task = _function_to_task(node, source, f)
                    if task is not None and task["key"] not in seen:
                        seen.add(task["key"])
                        out.append(task)
                        if limit and len(out) >= limit:
                            return out
    return out


def fuzz_inputs(types: list[str], n: int = 4) -> list[str]:
    """Deterministic literal arg strings per param annotation."""
    out = []
    for i in range(n):
        out.append(", ".join(_FUZZ_VALUES[t][i % len(_FUZZ_VALUES[t])]
                             for t in types))
    return out


def describe_function_prompt(code: str) -> str:
    """Describe-existing-code pass for undocumented functions — a much
    easier skill than task invention, and yields on-policy prompt text."""
    return ("Describe in ONE English sentence (15-40 words) what this "
            "Python function computes, including its edge-case behavior. "
            "Reply with the sentence only — no code, no preamble, no "
            "'This function' opener.\n```python\n" + code + "\n```")


def valid_description(text: str) -> str:
    d = " ".join(text.strip().split()).strip('"`').strip()
    for pre in ("this function ", "the function "):
        if d.lower().startswith(pre):
            d = d[len(pre):]
            if d:
                d = d[0].upper() + d[1:]
            break
    if not 15 <= len(d) <= 300 or not d.isascii() or "\n" in d:
        return ""
    if re.search(r"\bdef\b|```|[{}]", d):
        return ""  # code leaked into the description
    return d


def input_gen_prompt(task: dict) -> str:
    """Model-written literal test calls for a GIVEN working function —
    rescues tasks whose params aren't obvious from name heuristics."""
    return ("Write 4 test calls for this Python function as comment lines "
            f"`# {task['name']}(<args>)`, covering typical inputs plus one "
            "edge case. Args must be Python LITERALS (numbers, strings, "
            "lists, dicts, bools) — no variables, no objects, no calls.\n"
            "```python\n" + task["code"] + "\n```\n"
            "Reply with ONLY the 4 comment lines.")


def parse_input_calls(raw: str, name: str) -> list[str]:
    """Extract literal arg strings from `# name(<args>)` comment lines."""
    out = []
    for m in re.finditer(rf"#\s*{re.escape(name)}\s*\((.*?)\)\s*(?:\n|$)",
                         raw):
        a = m.group(1).strip()
        if not a:
            continue
        try:
            ast.literal_eval(f"({a},)")
            out.append(a)
        except Exception:
            continue
    return out


def codebase_solver_prompt(task: dict) -> str:
    return (f"Task: {task['desc']}\n"
            f"Signature: `{task['name']}{task['signature']}`\n"
            "Constraints: stdlib only, deterministic, no prints, no I/O.\n"
            "Respond with ONE python code block containing only the "
            f"function (named `{task['name']}`).")


def _harness(code: str, tests: list[str], name: str = "solve",
             nonce: str = "") -> str:
    """One-shot script: run the function on every test input, print reprs
    after a nonced marker (isolates results from user print() noise —
    and a candidate can't forge the marker: the nonce is generated
    per-exec and the sandbox denies frame/source inspection)."""
    lines = ["import math, re, itertools, collections, functools, "
             "string, operator", code, "",
             "import json as _json", "_out = []"]
    # Per-input isolation: an out-of-domain fuzz input must not kill the
    # whole suite — it just marks that input invalid for ground truth.
    for a in tests:
        lines += ["try:", f"    _out.append(repr({name}({a})))",
                  "except Exception as _e:",
                  "    _out.append('__ERROR__' + repr(_e))"]
    lines += [f"print('__SELF_GEN_RESULT__{nonce}:' "
              "+ _json.dumps([_out, None]))"]
    return "\n".join(lines)


def run_tests(executor, code: str, tests: list[str], name: str = "solve"):
    """Execute the function on all test inputs in the sandbox. Returns
    (repr_outputs, error) — outputs None on timeout/crash/exception.
    Only the line carrying this call's nonce parses — anything else a
    candidate prints (including a copied marker prefix) is ignored."""
    import secrets
    nonce = secrets.token_hex(6)
    r = executor.execute(_harness(code, tests, name, nonce))
    if r.get("timed_out") or r.get("returncode") != 0:
        return None, r.get("stderr", "failed")[:200]
    prefix = f"__SELF_GEN_RESULT__{nonce}:"
    for line in reversed(r.get("stdout", "").splitlines()):
        if line.startswith(prefix):
            try:
                outs, err = json.loads(line[len(prefix):])
            except Exception:
                return None, "bad result payload"
            if err is not None:
                return None, str(err)[:200]
            return outs, None
    return None, "no result marker"


def _approx_eq(a: str, b: str, tol: float = 1e-6) -> bool:
    """Compare two repr'd outputs; floats (also nested) get rel tolerance."""
    try:
        va, vb = ast.literal_eval(a), ast.literal_eval(b)
    except Exception:
        return a == b

    def rec(x, y):
        if isinstance(x, float) or isinstance(y, float):
            try:
                return abs(float(x) - float(y)) <= tol * max(
                    1.0, abs(float(x)), abs(float(y)))
            except Exception:
                return False
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(rec(u, v) for u, v in zip(x, y))
        return x == y
    return rec(va, vb)


def novelty_ngrams(text: str, n: int = 8) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    if not w:
        return set()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def is_novel(desc: str, seen_grams: set, threshold: float = 0.5) -> bool:
    """8-gram Jaccard novelty vs the accumulated description set."""
    g = novelty_ngrams(desc)
    if not g:
        return False
    if not seen_grams:
        return True
    return len(g & seen_grams) / len(g) < threshold


def goldilocks_ok(solve_rate: float) -> bool:
    """Informative tasks only: always-solved = no learning signal (format
    overfitting), never-solved = broken/too-hard task."""
    return 0.0 < solve_rate < 1.0


def keep_decision(solve_rate: float, easy_prob: float, rng) -> str | None:
    """-> 'goldilocks' | 'easy' | None. Easy tasks (all-k passes) are the
    strongest-verified; easy_prob controls how many enter the corpus —
    SFT learns from every correct row, RL would see zero signal."""
    if goldilocks_ok(solve_rate):
        return "goldilocks"
    if solve_rate == 1.0 and easy_prob > 0 and rng.random() < easy_prob:
        return "easy"
    return None


def run_selfgen(args, out_path: Path, finished: set):
    """Codebase-grounded wave loop: harvested real functions -> sandbox
    ground truth on fuzzed inputs -> solver rollouts -> execution-verified,
    Goldilocks + novelty gated rows. Every wave uses NEW functions."""
    import random
    from research.persistent_executor import PersistentCodeExecutor

    rng = random.Random(args.seed)
    executor = PersistentCodeExecutor(timeout_s=6.0, memory_limit_mb=512)
    if args.external:
        pool = json.loads(Path(args.external).read_text(encoding="utf-8"))
        print(f"selfgen: {len(pool)} external tasks from {args.external}",
              flush=True)
    else:
        pool = harvest_functions()
        print(f"selfgen: harvested {len(pool)} candidate functions",
              flush=True)
    rng.shuffle(pool)
    pool = [t for t in pool if f"selfgen:{t['key']}" not in finished]
    e = ForgeEngine.from_checkpoint(args.model)
    BS = args.batch_init or 16
    stats = dict(executable=0, gen=0, kept=0, goldilocks=0, easy=0)
    t0 = time.time()

    def gen_batch(ws, seeds):
        nonlocal BS
        while True:
            try:
                return e.generate_batch(
                    ws, max_new_tokens=args.max_new, temperature=args.temp,
                    top_p=0.95, seeds=seeds)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if BS > 4:
                    BS = max(4, BS // 2)
                    print(f"  OOM -> batch={BS}", flush=True)
                    ws, seeds = ws[:BS], seeds[:BS]
                    continue
                return [None] * len(ws)
            except Exception as ex:
                print(f"  batch err: {str(ex)[:150]}", flush=True)
                return [None] * len(ws)

    # ── Describe-first pass: undocumented functions get a model-written
    # one-line description (describe-existing-code is a learnable skill,
    # unlike task invention; the text is on-policy prompt material).
    need = [t for t in pool if not t["desc"]]
    for d0 in range(0, len(need), BS):
        chunk = need[d0:d0 + BS]
        ws = [wrap(describe_function_prompt(t["code"])) for t in chunk]
        raws = gen_batch(ws, [rng.randrange(1 << 30) for _ in ws])
        for t, raw in zip(chunk, raws):
            if raw is not None:
                t["desc"] = valid_description(raw)
    pool = [t for t in pool if t["desc"]]
    print(f"selfgen: {len(pool)} described tasks "
          f"({len(need)} needed describe-pass)", flush=True)

    seen_grams: set = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("task_desc"):
                    seen_grams |= novelty_ngrams(r["task_desc"])
            except Exception:
                pass
    pool = [t for t in pool if is_novel(t["desc"], seen_grams)]
    pool = pool[:args.tasks]
    print(f"selfgen: {len(pool)} fresh tasks after resume+novelty filters",
          flush=True)

    def resolve_gt(t, tests) -> bool:
        expected, err = run_tests(executor, t["code"], tests, t["name"])
        if expected is None:
            return False
        # Ground truth = inputs the reference itself survives (fuzz
        # inputs can be outside the function's domain). Require >=2
        # valid inputs and >=2 distinct outputs (non-degenerate).
        valid = [i for i, o in enumerate(expected)
                 if not o.startswith("__ERROR__")]
        if len(valid) < 2 or len({expected[i] for i in valid}) < 2:
            return False
        t["tests"] = tests
        t["expected"] = expected
        t["valid_idx"] = valid
        return True

    def resolve_external_gt(t) -> bool:
        """External DB rows: the reference must pass every literal assert
        (>=2), else the row's tests are unreliable — drop it."""
        argstrs = [x["args"] for x in t["tests"]]
        expected, err = run_tests(executor, t["code"], argstrs, t["name"])
        if expected is None:
            return False
        valid = []
        for i, x in enumerate(t["tests"]):
            o = expected[i]
            if o.startswith("__ERROR__"):
                continue
            if not _approx_eq(o, x["expected"]):
                return False
            valid.append(i)
        if len(valid) < 2:
            return False
        t["tests"] = argstrs
        t["expected"] = expected
        t["valid_idx"] = valid
        return True

    for w0 in range(0, len(pool), BS):
        # ── Phase 1: executor ground truth (fuzz, then model-input rescue) ──
        tasks = []
        retry = []
        for t in pool[w0:w0 + BS]:
            if args.external:
                (tasks if resolve_external_gt(t) else retry).append(t)
            else:
                (tasks if resolve_gt(t, fuzz_inputs(t["types"], n=4))
                 else retry).append(t)
        stats["fuzz_ok"] = stats.get("fuzz_ok", 0) + len(tasks)
        for r0 in range(0, len(retry), BS) if not args.external else []:
            chunk = retry[r0:r0 + BS]
            ws = [wrap(input_gen_prompt(t)) for t in chunk]
            raws = gen_batch(ws, [rng.randrange(1 << 30) for _ in ws])
            for t, raw in zip(chunk, raws):
                if raw is None:
                    continue
                calls = parse_input_calls(raw, t["name"])
                if len(calls) >= 2 and resolve_gt(t, calls):
                    tasks.append(t)
                    stats["rescued"] = stats.get("rescued", 0) + 1
        stats["executable"] += len(tasks)
        print(f"[wave {w0//BS}] executable={len(tasks)}/{BS} -> solving",
              flush=True)

        # ── Phase 2: k solver rollouts per task ──
        queue = [(ti, ki) for ti in range(len(tasks))
                 for ki in range(args.k)]
        passes: dict[int, list] = {ti: [] for ti in range(len(tasks))}
        for q0 in range(0, len(queue), BS):
            chunk = queue[q0:q0 + BS]
            ws = [wrap(codebase_solver_prompt(tasks[ti])) for ti, _ in chunk]
            raws = gen_batch(ws, [rng.randrange(1 << 30) for _ in ws])
            for (ti, _ki), raw in zip(chunk, raws):
                if raw is None:
                    continue
                stats["gen"] += 1
                t = tasks[ti]
                code = verify(raw, [t["name"]])
                if code is None:
                    continue
                outs, err = run_tests(executor, code, t["tests"], t["name"])
                if outs is None or len(outs) != len(t["expected"]):
                    continue
                if all(not outs[i].startswith("__ERROR__")
                       and _approx_eq(outs[i], t["expected"][i])
                       for i in t["valid_idx"]):
                    passes[ti].append((norm_hash(code), raw))

        # ── Phase 3: Goldilocks gate + keep <=2 variants + markers ──
        # Easy tasks (rate==1.0): all-k passes = strongest verification and
        # still novel-task variety — keep 1 row w.p. keep_easy_prob. This is
        # SFT (every correct row teaches), not RL (zero gradient signal).
        for ti, t in enumerate(tasks):
            rate = len(passes[ti]) / args.k
            gk = f"selfgen:{t['key']}"
            verdict = keep_decision(rate, args.keep_easy_prob, rng)
            easy = verdict == "easy"
            if verdict:
                stats[verdict] += 1
                seen_grams |= novelty_ngrams(t["desc"])
                cap = 1 if easy else 2
                kept = 0
                seen_h = set()
                for hh, raw in passes[ti]:
                    if hh in seen_h or kept >= cap:
                        continue
                    seen_h.add(hh)
                    row = dict(
                        prompt=codebase_solver_prompt(t), response=raw,
                        source=("rft_external_easy" if easy else
                                "rft_external") if args.external
                        else ("rft_codebase_easy" if easy else
                              "rft_codebase"), gen_key=gk,
                        task_desc=t["desc"],
                        domain="external" if args.external
                        else t["file"].split("/")[0],
                        difficulty="auto", solve_rate=rate,
                        n_tests=len(t["tests"]), seed_code=t["code"],
                        required=[t["name"]])
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                    stats["kept"] += 1
            else:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(
                        prompt=None, response=None, gen_key=gk,
                        task_desc=t["desc"], solve_rate=rate)) + "\n")
        el = time.time() - t0
        print(f"[wave {w0//BS} done] goldilocks={stats['goldilocks']} "
              f"easy={stats['easy']}/{len(tasks)} kept={stats['kept']} "
              f"gen={stats['gen']} bs={BS} {el/60:.1f}min", flush=True)
    executor.close()
    print(f"SELFGEN DONE {stats} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["proposer", "direct", "selfgen"],
                    required=True)
    ap.add_argument("--tasks", type=int, default=64,
                    help="selfgen: fresh tasks to attempt")
    ap.add_argument("--external", default=None,
                    help="selfgen: external task-pool JSON from "
                         "scripts/fetch_external_tasks.py (reference + "
                         "literal asserts; skips harvest/describe/fuzz)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--batch-init", type=int, default=0,
                    help="0=auto from free VRAM")
    ap.add_argument("--batch-min", type=int, default=4)
    ap.add_argument("--batch-max", type=int, default=40)
    ap.add_argument("--keep-easy-prob", type=float, default=0.0,
                    help="selfgen: probability of keeping 1 verified row "
                         "from always-solved tasks (0=strict Goldilocks). "
                         "All-k-pass rows are the strongest-verified and "
                         "still add novel-task variety for SFT.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(out_path)

    if args.mode == "selfgen":
        run_selfgen(args, out_path, finished)
        return

    if args.mode == "proposer":
        prompts = proposer_prompts(args.reps)
    else:
        prompts = direct_prompts(args.limit)
    prompts = [p for p in prompts
               if f"{p['key']}:{p['instr'][:40]}" not in finished]
    print(f"{len(prompts)} prompts todo ({len(finished)} done)")

    e = ForgeEngine.from_checkpoint(args.model)
    stats = dict(gen=0, kept=0)
    t0 = time.time()

    # Flatten (prompt x k) into one task queue; batched decode is
    # latency-bound so width scales with VRAM. Adaptive sizing:
    # start safe (bs8), measure the REAL per-seq peak incl. KV growth
    # (max_memory_allocated across a full batch), then predict how
    # many more seqs fit under the 88% ceiling and jump there — at
    # most 2 grow-steps, since every batch-size change triggers a new
    # CUDA-graph capture (the suspected HYPERVISOR_ERROR trigger).
    # OOM -> halve, stop growing.
    free_b, total_b = torch.cuda.mem_get_info()
    ceiling = 0.88 * total_b
    if args.batch_init > 0:
        BS = args.batch_init
        grow_left = 0
    else:
        BS = 8
        grow_left = 2
    BS = max(args.batch_min, min(args.batch_max, BS))
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    print(f"batch={BS} (free {free_b/1e9:.1f}GB/{total_b/1e9:.1f}GB, "
          f"ceiling {ceiling/1e9:.1f}GB)", flush=True)
    tasks = []
    for pi, p in enumerate(prompts):
        for ki in range(args.k):
            tasks.append((pi, ki, p))

    kept_hashes = {i: set() for i in range(len(prompts))}
    b0 = 0
    while b0 < len(tasks):
        chunk = tasks[b0:b0 + BS]
        ws = [(wrap_proposer(t[2]["instr"]) if args.mode == "proposer"
               else wrap(t[2]["instr"])) for t in chunk]
        seeds = [t[1] * 7919 + b0 for t in chunk]
        try:
            raws = e.generate_batch(
                ws, max_new_tokens=args.max_new, temperature=args.temp,
                top_p=0.95, seeds=seeds)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            grow_left = 0
            if BS > args.batch_min:
                BS = max(args.batch_min, BS // 2)
                print(f"  OOM -> batch={BS}", flush=True)
                continue
            raws = [None] * len(chunk)
        except Exception as ex:
            print(f"  batch err @{b0}: {ex}")
            raws = [None] * len(chunk)
        else:
            if grow_left > 0 and raws:
                peak = torch.cuda.max_memory_allocated()
                per_seq = max((peak - base_alloc) / len(chunk), 1)
                head = ceiling - peak
                add = int(head // per_seq)
                new_bs = min(args.batch_max, BS + max(0, add))
                if new_bs > BS:
                    print(f"  peak {peak/1e9:.2f}GB, ~{per_seq/1e6:.0f}MB/seq"
                          f" -> batch={new_bs}", flush=True)
                    BS = new_bs
                    grow_left -= 1
                    torch.cuda.reset_peak_memory_stats()
                else:
                    grow_left = 0
        for (pi, _ki, p), raw in zip(chunk, raws):
            if raw is None:
                continue
            stats["gen"] += 1
            code = verify(raw, p["req"])
            if code is None:
                continue
            hh = norm_hash(code)
            kh = kept_hashes[pi]
            if hh in kh or len(kh) >= 2:
                continue
            kh.add(hh)
            resp = ("<" + chr(0x2F) + "think>\n```python\n" + code
                    + "\n```") if args.mode == "proposer" else raw
            row = dict(prompt=p["instr"], response=resp,
                       source=f"rft_{args.mode}",
                       gen_key=f"{p['key']}:{p['instr'][:40]}",
                       seed_code=p["seed"], required=p["req"])
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
        done_p = (b0 + BS) // args.k
        if (b0 // BS) % 8 == 0:
            el = time.time() - t0
            print(f"[~{done_p}/{len(prompts)}] gen={stats['gen']} "
                  f"kept={stats['kept']} bs={BS} {el/60:.0f}min",
                  flush=True)
        b0 += BS

    # mark all-zero-keep prompts finished
    for pi, p in enumerate(prompts):
        if not kept_hashes[pi]:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(
                    prompt=None, response=None,
                    gen_key=f"{p['key']}:{p['instr'][:40]}")) + "\n")
    print(f"DONE gen={stats['gen']} kept={stats['kept']} "
          f"-> {out_path}")


if __name__ == "__main__":
    main()

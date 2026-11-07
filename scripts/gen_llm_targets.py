"""LLM-generated evolution goals — unbounded target space.

The fixed 26 hand/auto targets cap the distill corpus (prompt dedup
saturates). This script asks an API teacher to INVENT new optimization
goals as {name, prompt, seed_code, gen_cases} JSON, validates each in
a subprocess (AST gate + self-oracle + determinism + runtime budget),
then writes forge/evolution/targets/llm_<name>.py in the auto_targets
spec format — the generic _ORACLE verifies candidate-vs-seed semantics,
so every generated goal slots into fn_evolve / gen_qwen_distill /
merge_distill with zero downstream changes.

Goal inspiration comes from three broad channels (--sources):
  code   — real function bodies scanned out of forge/ (the teacher
           sees actual hot-path code and invents a goal around the
           same compute pattern)
  web    — live Tavily search snippets on kernel/optimization
           benchmark topics (the teacher grounds goals in external
           benchmark ideas)
  topics — the built-in topic x difficulty x library seed grid

Usage:
    python scripts/gen_llm_targets.py --count 60 --backend mistral
    python scripts/gen_llm_targets.py --count 30 --sources code,web
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from gen_qwen_distill import BACKENDS, _api_gen  # noqa: E402
from forge.evolution.llm_domain_gen import _check_domain_code  # noqa: E402
from forge.evolution.auto_targets import _ORACLE, _SPEC_HEADER_IMPORTS  # noqa: E402

TARGETS_DIR = REPO / "forge" / "evolution" / "targets"

_TOPICS = [
    "string parsing and tokenization", "graph traversal algorithms",
    "tensor indexing and reshaping", "bit packing and manipulation",
    "memoization and caching", "sorting and searching",
    "probability and sampling", "compression and encoding",
    "numerical linear algebra kernels", "data structure operations",
    "scheduling and interval problems", "simulation and state updates",
    "text normalization and distance", "matrix transformations",
    "quantization and rounding", "hashing and fingerprinting",
    "signal smoothing and filtering", "geometry and spatial queries",
    "combinatorics and counting", "finite state machines",
    "attention-like tensor reductions", "prefix sums and scans",
    "tree flattening and serialization", "run-length encoding",
    "sliding-window aggregations", "set operations and membership",
]
_DIFF = ["easy", "medium", "hard"]
_LIBS = [
    "python stdlib only (no numpy/torch)",
    "numpy allowed",
    "torch allowed",
]

_WEB_QUERIES = [
    "python performance benchmark micro-optimizations common",
    "GPU kernel optimization benchmark suite list",
    "numpy vectorization performance patterns examples",
    "LLM inference hot path operations softmax sampling",
    "python string algorithm optimization interview problems",
    "data structure algorithms commonly optimized list",
    "tensor operations deep learning kernels list",
    "cache-friendly algorithms python examples",
    "python regex parsing performance bottlenecks",
    "numerical algorithms optimization benchmarks",
]

# Modules worth scanning for goal-worthy compute patterns — skipped:
# tests/, gui, server plumbing, pure-config files.
_CODE_SKIP = re.compile(
    r"(tests|forge_gui|scratchpad|__pycache__|sweep|bench_|"
    r"agent_tools|server|session_manager|tool_security)", re.I)
_CODE_HIT = re.compile(
    r"(quant|attn|attention|decode|sampling|cache|kv|kernel|conv|rope|"
    r"mamba|scan|tokeniz|canon|merge|prune|score|rank|spec|draft|"
    r"gate|router|pool|pack|compress|hash|trie|ngram|sparse)", re.I)


def _code_contexts(cap: int) -> list[dict]:
    """Harvest real function bodies from forge/ as goal inspiration.

    Picks standalone-looking FunctionDefs (6-55 source lines) in
    perf-relevant modules; returns [{'src', 'origin'}] contexts."""
    out = []
    files = [p for p in (REPO / "forge").rglob("*.py")
             if not _CODE_SKIP.search(str(p))]
    random.shuffle(files)
    for f in files:
        if len(out) >= cap:
            break
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except Exception:
            continue
        for node in ast.walk(tree):
            if len(out) >= cap:
                break
            if not isinstance(node, ast.FunctionDef):
                continue
            seg = ast.get_source_segment(text, node)
            if not seg:
                continue
            lines = seg.splitlines()
            if not (6 <= len(lines) <= 55):
                continue
            if not _CODE_HIT.search(node.name + f.stem):
                continue
            out.append({"src": "\n".join(lines)[:3000],
                        "origin": f"{f.relative_to(REPO)}::{node.name}"})
    return out


def _web_contexts(key: str, cap: int) -> list[dict]:
    """Tavily snippets on optimization topics -> goal inspiration."""
    if not key:
        return []
    import urllib.request
    out = []
    for q in random.sample(_WEB_QUERIES, min(cap, len(_WEB_QUERIES))):
        try:
            body = json.dumps({
                "api_key": key, "query": q, "max_results": 3,
                "search_depth": "basic"}).encode()
            req = urllib.request.Request(
                "https://api.tavily.com/search", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.loads(r.read())
            for hit in d.get("results", []):
                txt = f"{hit.get('title','')}\n{hit.get('content','')}"
                out.append({"src": txt[:1800], "origin": f"web:{q}"})
        except Exception:
            continue
    return out[:cap]


_PROMPT_TOPIC = """Generate ONE new self-contained Python optimization goal \
for a code-evolution benchmark, as a single JSON object.

Topic: {topic}
Difficulty: {difficulty}
Libraries: {libs}

{schema}
"""

_PROMPT_CODE = """Here is a REAL function from a production LLM inference \
engine ({origin}). Invent ONE new self-contained Python optimization \
goal for a code-evolution benchmark INSPIRED BY its compute pattern — \
same kind of hot-path work, but your goal must be standalone (do NOT \
copy the function; design a fresh simpler version that is optimizable \
in the same way).

```python
{src}
```

Difficulty: {difficulty}
Libraries: {libs}

{schema}
"""

_PROMPT_WEB = """Here is a snippet about code optimization / kernel \
benchmarks from the web ({origin}). Invent ONE new self-contained \
Python optimization goal for a code-evolution benchmark inspired by \
the kind of work described — pick ONE concrete computable task.

{src}

Difficulty: {difficulty}
Libraries: {libs}

{schema}
"""

_SCHEMA = """JSON schema (no markdown, no commentary):
{{
  "name": "snake_case_name",
  "prompt": "one-paragraph instruction describing what to optimize and \
the correctness constraint",
  "seed_code": "<python source: imports + exactly one required function \
`def <name>(...)` — a CORRECT but deliberately unoptimized \
implementation>",
  "gen_cases": "<python source of `def _gen_cases(seed):` — a \
DETERMINISTIC function OF THE SEED returning a list of 5-15 inputs; a \
tuple item is called f(*item), anything else f(item). Bind randomness \
to `seed` (random.seed(seed) / np.random.seed(seed % (2**32)) / \
torch.Generator().manual_seed(seed)); literal corpora may ignore it.>"
}}

Rules:
- seed_code is fully self-contained: no file/network/env access, no \
prints, no input()
- the required function name in seed_code MUST equal "name"
- the function must be clearly optimizable (naive loops, repeated \
work, O(n^2) that could be O(n log n), unnecessary copies)
- _gen_cases must be a deterministic function OF THE SEED (same seed -> \
identical cases) and produce realistic inputs that exercise the function
- output ONLY the JSON object"""


_VALIDATOR = r'''
import json, sys, time
ns = {{}}
code = {code!r}
gen = {gen!r}
required = {required!r}
try:
    compile(code, "<s>", "exec"); compile(gen, "<g>", "exec")
    exec(code, ns)
    if not callable(ns.get(required)):
        print(json.dumps({{"ok": False, "why": "required not callable"}})); sys.exit()
    exec(gen, ns)
    t0 = time.perf_counter()
    cases1 = ns["_gen_cases"](7)
    cases2 = ns["_gen_cases"](7)
    if not isinstance(cases1, list) or not cases1:
        print(json.dumps({{"ok": False, "why": "gen_cases empty/not list"}})); sys.exit()
    if repr(cases1) != repr(cases2):
        print(json.dumps({{"ok": False, "why": "gen_cases nondeterministic"}})); sys.exit()
    if time.perf_counter() - t0 > 5:
        print(json.dumps({{"ok": False, "why": "gen_cases slow"}})); sys.exit()
    fn = ns[required]
    t0 = time.perf_counter()
    for c in cases1:
        fn(*c) if isinstance(c, tuple) else fn(c)
    if time.perf_counter() - t0 > 20:
        print(json.dumps({{"ok": False, "why": "seed too slow"}})); sys.exit()
    print(json.dumps({{"ok": True, "ncases": len(cases1)}}))
except Exception as e:
    print(json.dumps({{"ok": False, "why": f"{{type(e).__name__}}: {{e}}"[:160]}}))
'''


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _write_spec(name: str, prompt: str, seed: str, gen: str) -> Path:
    text = (
        f'"""LLM-generated evolution target — gen_llm_targets.py."""\n\n'
        f"NAME = {name!r}\nGOAL = 'min'\nMETRIC = 'ms_per_call'\n"
        f"TAGS = ['llm-generated']\nBUDGETS = {{}}\nBENCH_REPS = 3\n"
        f"REQUIRED = [{name!r}]\nPROMPT = {prompt!r}\n\n"
        f"SEED_CODE = {seed!r}\n\n"
        + "\n".join(_SPEC_HEADER_IMPORTS) + "\n\n" + gen + "\n" + _ORACLE)
    p = TARGETS_DIR / f"llm_{name}.py"
    p.write_text(text, encoding="utf-8")
    return p


def _self_oracle(path: Path) -> bool:
    """Run the assembled spec's own test(ns) on its seed in-process."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_llm_t", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = {}
    exec(mod.SEED_CODE, ns)
    return bool(mod.test(ns).get("ok"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mistral",
                    choices=list(BACKENDS))
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=2600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sources", default="code,web,topics",
                    help="goal-inspiration channels")
    args = ap.parse_args()

    url, key_env, model, extra = BACKENDS[args.backend]
    if args.api_model:
        model = args.api_model
    key = os.environ.get(key_env, "")
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == key_env and not key:
                    key = v.strip()
                if k.strip() == "TAVILY_API_KEY" and not tavily_key:
                    tavily_key = v.strip()
    if not key:
        print(f"[targets] no {key_env}")
        return 1
    print(f"[targets] {model} @ {url}")

    # ── build the inspiration pool across all enabled channels ─────
    pool: list[dict] = []
    srcs = {s.strip() for s in args.sources.split(",")}
    if "code" in srcs:
        ctx = _code_contexts(args.count)
        pool += [{"kind": "code", **c} for c in ctx]
        print(f"[targets] code contexts: {len(ctx)}")
    if "web" in srcs:
        ctx = _web_contexts(tavily_key, min(len(_WEB_QUERIES),
                                          max(4, args.count // 4)))
        pool += [{"kind": "web", **c} for c in ctx]
        print(f"[targets] web contexts: {len(ctx)}")
    if "topics" in srcs:
        combos = [{"kind": "topics", "topic": t, "libs": l}
                  for t in _TOPICS for l in _LIBS]
        pool += combos
    random.shuffle(pool)
    pool = pool[: args.count * 2]          # slots w/ retry room

    existing = {p.stem[4:] for p in TARGETS_DIR.glob("llm_*.py")}
    seen_hash = {
        hashlib.md5(p.read_bytes()).hexdigest()
        for p in TARGETS_DIR.glob("*.py")}
    stats = {"ok": 0, "invalid": 0, "dupe": 0}
    lock = threading.Lock()

    def worker(i: int):
        ctx = pool[i % len(pool)]
        diff = random.choice(_DIFF)
        libs = ctx.get("libs") or random.choice(_LIBS)
        for attempt in range(args.retries + 1):
            if ctx["kind"] == "code":
                prompt = _PROMPT_CODE.format(
                    src=ctx["src"], origin=ctx["origin"],
                    difficulty=diff, libs=libs, schema=_SCHEMA)
            elif ctx["kind"] == "web":
                prompt = _PROMPT_WEB.format(
                    src=ctx["src"], origin=ctx["origin"],
                    difficulty=diff, libs=libs, schema=_SCHEMA)
            else:
                prompt = _PROMPT_TOPIC.format(
                    topic=ctx["topic"], difficulty=diff, libs=libs,
                    schema=_SCHEMA)
            try:
                raw, _ = _api_gen(
                    url, key, model, prompt, max_new=args.max_new,
                    temperature=min(0.8 + attempt * 0.1, 1.0),
                    extra_body=extra)
            except Exception:
                continue
            spec = _parse_json(raw) if raw else None
            if not spec or not all(
                    spec.get(k) for k in
                    ("name", "prompt", "seed_code", "gen_cases")):
                continue
            name = re.sub(r"[^a-z0-9_]+", "_",
                          str(spec["name"]).lower()).strip("_")[:48]
            if not name or name[0].isdigit() or name in existing:
                continue
            seed, gen = str(spec["seed_code"]), str(spec["gen_cases"])
            if f"def {name}" not in seed \
                    or "def _gen_cases(seed" not in gen:
                continue
            ok1, _ = _check_domain_code(seed)
            ok2, _ = _check_domain_code(gen)
            if not (ok1 and ok2):
                continue
            vcode = _VALIDATOR.format(
                code=seed, gen=gen, required=name)
            try:
                r = subprocess.run(
                    [sys.executable, "-c", vcode],
                    capture_output=True, text=True, timeout=45)
                verdict = json.loads(
                    r.stdout.strip().splitlines()[-1]) \
                    if r.stdout.strip() else {"ok": False}
            except Exception:
                verdict = {"ok": False}
            if not verdict.get("ok"):
                with lock:
                    stats["invalid"] += 1
                continue
            path = _write_spec(name, str(spec["prompt"]), seed, gen)
            try:
                if not _self_oracle(path):
                    path.unlink()
                    with lock:
                        stats["invalid"] += 1
                    continue
            except Exception:
                path.unlink(missing_ok=True)
                with lock:
                    stats["invalid"] += 1
                continue
            h = hashlib.md5(path.read_bytes()).hexdigest()
            with lock:
                if h in seen_hash:
                    path.unlink(missing_ok=True)
                    stats["dupe"] += 1
                    continue
                seen_hash.add(h)
                existing.add(name)
                stats["ok"] += 1
                print(f"[targets] {i}: kept llm_{name} "
                      f"({ctx['kind']}:{ctx.get('topic') or ctx['origin']})",
                      flush=True)
            return
        with lock:
            stats["invalid"] += 1

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(worker, range(min(len(pool), args.count))))
    print(f"[targets] done: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

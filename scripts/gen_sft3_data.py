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

selfgen (AZR anti-overfitting): every wave proposes FRESH tasks (domain x
difficulty x uniqueness-seed rotation, cliche banlist, 8-gram novelty
gate), the sandbox executor derives ground truth from the reference
implementation, solver rollouts must reproduce it on all tests, and only
Goldilocks-informative tasks (0 < solve_rate < 1) are kept. Fixed corpora
are never reused, so the prompt distribution expands every round.
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


# ── SELFGEN (AZR): fresh tasks + executor ground truth + Goldilocks ──
# Anti-fixed-pool-overfitting: the model proposes NEW tasks (never the
# fixed corpora), the sandbox executor derives ground truth from the
# reference implementation, solver rollouts must MATCH it on tests, and
# only informative tasks (0 < solve_rate < 1) enter the corpus.

_SELFGEN_DOMAINS = [
    "string manipulation", "array algorithms", "hash maps and counting",
    "two pointers", "stack and queue simulation", "greedy algorithms",
    "recursion and backtracking", "math and number theory",
    "date and interval logic", "matrix and grid traversal",
    "dynamic programming (small)", "parsing and tokenizing",
]
_SELFGEN_DIFFICULTIES = ["easy", "medium", "medium", "hard"]
_RE_TEST_INPUT = re.compile(
    r'#\s*solve\s*\((.*?)\)\s*(?:->[^\n]*)?(?:\n|$)')


def selfgen_proposal_prompt(domain: str, difficulty: str, seed: str) -> str:
    return (
        "Invent ONE novel Python programming exercise for a coding student.\n"
        f"Domain: {domain}. Difficulty: {difficulty}.\n"
        f"Uniqueness seed: {seed} (use it to pick an unusual topic).\n"
        "Banned cliches: Fibonacci, palindrome, LRU cache, sorting, "
        "binary search, factorial, prime checking, debounce, flatten.\n"
        "Rules: stdlib only; deterministic (no randomness, no I/O); the "
        "reference solution must be a single function `solve`.\n"
        "Reply in EXACTLY this format:\n"
        "# Task: <one-sentence description of what solve computes>\n"
        "```python\n"
        "def solve(<args>):\n"
        "    <reference implementation>\n"
        "# solve(<literal input 1>)\n"
        "# solve(<literal input 2>)\n"
        "# solve(<literal input 3>)\n"
        "# solve(<literal input 4, an edge case>)\n"
        "```\n"
        "Test inputs must be Python literals; do NOT write expected outputs.")


def parse_proposal(raw: str) -> dict | None:
    """Extract (description, reference code, test arg strings) from a
    proposal generation. Returns None when the contract isn't met."""
    m = re.search(r"#\s*Task:\s*(.+?)(?:\n|$)", raw)
    if not m:
        return None
    desc = m.group(1).strip()
    if len(desc) < 15:
        return None
    code = _extract_code(raw) or ""
    if not code.strip() or not safe_code(code):
        return None
    try:
        compile(code, "<proposal>", "exec")
    except SyntaxError:
        return None
    if not re.search(r"def\s+solve\s*\(", code):
        return None
    tests = [a.strip() for a in _RE_TEST_INPUT.findall(code)]
    tests = [a for a in tests if a]
    if len(tests) < 3:
        return None
    for a in tests:
        try:
            ast.literal_eval(f"({a},)")
        except Exception:
            return None
    return dict(desc=desc, code=code, tests=tests)


def _harness(code: str, tests: list[str]) -> str:
    """One-shot script: run solve on every test input, print reprs after a
    marker. The marker isolates results from user print() noise."""
    lines = [code, "", "import json as _json", "_out = []", "_err = None",
             "try:"]
    for a in tests:
        lines.append(f"    _out.append(repr(solve({a})))")
    lines += ["except Exception as _e:", "    _err = repr(_e)",
              "print('__SELF_GEN_RESULT__' + _json.dumps([_out, _err]))"]
    return "\n".join(lines)


def run_tests(executor, code: str, tests: list[str]):
    """Execute solve() on all test inputs in the sandbox. Returns
    (repr_outputs, error) — outputs None on timeout/crash/exception."""
    r = executor.execute(_harness(code, tests))
    if r.get("timed_out") or r.get("returncode") != 0:
        return None, r.get("stderr", "failed")[:200]
    for line in reversed(r.get("stdout", "").splitlines()):
        if line.startswith("__SELF_GEN_RESULT__"):
            try:
                outs, err = json.loads(line[len("__SELF_GEN_RESULT__"):])
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


def selfgen_solver_prompt(desc: str, code: str) -> str:
    m = re.search(r"def\s+solve\s*\([^)]*\)", code)
    sig = m.group(0) if m else "def solve(...)"
    return (f"Write a Python function that {desc[0].lower() + desc[1:]}\n"
            f"Signature: `{sig}`\n"
            "Constraints: stdlib only, deterministic, no prints.\n"
            "Respond with ONE python code block containing only the "
            "function (name it `solve`).")


def run_selfgen(args, out_path: Path, finished: set):
    """AZR wave loop: propose fresh tasks -> executor ground truth ->
    solver rollouts -> execution-verified, Goldilocks + novelty gated rows.
    Every wave is NEW tasks — the fixed corpora are never reused."""
    import random
    import uuid
    from research.persistent_executor import PersistentCodeExecutor

    rng = random.Random(args.seed)
    executor = PersistentCodeExecutor(timeout_s=6.0, memory_limit_mb=512)
    e = ForgeEngine.from_checkpoint(args.model)
    BS = args.batch_init or 16
    stats = dict(proposed=0, valid=0, novel=0, executable=0, gen=0, kept=0,
                 goldilocks=0)
    t0 = time.time()
    domain_stats: dict[str, list] = {}
    seen_grams: set = set()
    # Rebuild novelty memory from prior selfgen rows (resume-safe).
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("task_desc"):
                    seen_grams |= novelty_ngrams(r["task_desc"])
            except Exception:
                pass
    print(f"selfgen: {len(seen_grams)} prior novelty grams, "
          f"{len(finished)} finished tasks", flush=True)

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

    n_waves = (args.tasks + BS - 1) // BS
    for wave in range(n_waves):
        # ── Phase 1: propose fresh tasks ──
        specs = []
        for _ in range(min(BS, args.tasks - wave * BS)):
            domain = rng.choice(_SELFGEN_DOMAINS)
            # Goldilocks difficulty adaptation per domain: high recent
            # solve rate -> harder, low -> easier.
            diff = rng.choice(_SELFGEN_DIFFICULTIES)
            hist = domain_stats.get(domain, [])
            if len(hist) >= 4:
                rate = sum(hist[-8:]) / len(hist[-8:])
                if rate > 0.6:
                    diff = "hard"
                elif rate < 0.4:
                    diff = "easy"
            specs.append((domain, diff, uuid.uuid4().hex[:8]))
        ws = [wrap(selfgen_proposal_prompt(d, df, s)) for d, df, s in specs]
        raws = gen_batch(ws, [rng.randrange(1 << 30) for _ in ws])
        tasks = []
        for (domain, diff, _seed), raw in zip(specs, raws):
            stats["proposed"] += 1
            if raw is None:
                continue
            p = parse_proposal(raw)
            if p is None:
                continue
            stats["valid"] += 1
            if not is_novel(p["desc"], seen_grams):
                continue
            stats["novel"] += 1
            expected, err = run_tests(executor, p["code"], p["tests"])
            if expected is None:
                continue
            stats["executable"] += 1
            p.update(domain=domain, difficulty=diff, expected=expected,
                     key=hashlib.md5(p["desc"].encode()).hexdigest()[:12])
            tasks.append(p)
        print(f"[wave {wave}] proposals valid={stats['valid']}/"
              f"{stats['proposed']} novel={stats['novel']} "
              f"executable={stats['executable']} -> solving {len(tasks)}",
              flush=True)

        # ── Phase 2: k solver rollouts per fresh task ──
        queue = [(ti, ki) for ti in range(len(tasks))
                 for ki in range(args.k)]
        passes: dict[int, list] = {ti: [] for ti in range(len(tasks))}
        for q0 in range(0, len(queue), BS):
            chunk = queue[q0:q0 + BS]
            ws = [wrap(selfgen_solver_prompt(
                tasks[ti]["desc"], tasks[ti]["code"])) for ti, _ in chunk]
            raws = gen_batch(ws, [rng.randrange(1 << 30) for _ in ws])
            for (ti, _ki), raw in zip(chunk, raws):
                if raw is None:
                    continue
                stats["gen"] += 1
                t = tasks[ti]
                code = verify(raw, ["solve"])
                if code is None:
                    continue
                outs, err = run_tests(executor, code, t["tests"])
                if outs is None:
                    continue
                if len(outs) == len(t["expected"]) and all(
                        _approx_eq(o, ex) for o, ex in
                        zip(outs, t["expected"])):
                    passes[ti].append((norm_hash(code), raw))

        # ── Phase 3: Goldilocks gate + keep <=2 variants + markers ──
        for ti, t in enumerate(tasks):
            rate = len(passes[ti]) / args.k
            domain_stats.setdefault(t["domain"], []).append(rate)
            gk = f"selfgen:{t['key']}"
            if goldilocks_ok(rate):
                stats["goldilocks"] += 1
                seen_grams |= novelty_ngrams(t["desc"])
                kept = 0
                seen_h = set()
                for hh, raw in passes[ti]:
                    if hh in seen_h or kept >= 2:
                        continue
                    seen_h.add(hh)
                    row = dict(
                        prompt=selfgen_solver_prompt(t["desc"], t["code"]),
                        response=raw, source="rft_selfgen", gen_key=gk,
                        task_desc=t["desc"], domain=t["domain"],
                        difficulty=t["difficulty"], solve_rate=rate,
                        n_tests=len(t["tests"]), seed_code=t["code"],
                        required=["solve"])
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
        print(f"[wave {wave} done] goldilocks={stats['goldilocks']}/"
              f"{len(tasks)} kept={stats['kept']} gen={stats['gen']} "
              f"bs={BS} {el/60:.1f}min", flush=True)
    executor.close()
    print(f"SELFGEN DONE {stats} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["proposer", "direct", "selfgen"],
                    required=True)
    ap.add_argument("--tasks", type=int, default=64,
                    help="selfgen: fresh tasks to attempt")
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

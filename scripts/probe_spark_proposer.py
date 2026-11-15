"""Spark proposer-prompt probe — which propose-prompt variant yields the
most executor-valid self-play tasks?

Loads the Spark engine ONCE, then generates N=16 proposals per prompt
variant (4 per domain: algorithms/math/strings/logic) through the
PRODUCTION InfiniteCurriculum engine path (_generate_batch with the
propose prefill + think-cap processors — identical to propose_tasks).

Metrics per proposal:
  parse-valid    _parse_proposal returns a ProposedTask
  solvable-valid _validate_task AND proposer_confidence == 1.0 — the
                 reference impl actually ran on its own test inputs in
                 the persistent_executor sandbox (AZR derived-output
                 semantics). This is the metric the RSI data supply
                 cares about. Soft accepts (conf < 1.0: parses-but-
                 fails / no ref code) are counted separately.

Variants:
  baseline   CHAT_PROPOSE_PROMPT + single desc_hint (production)
  dual       two crossed desc_hints (idea mixing: "h1 + h2")
  constraint baseline + appended hard-requirements block
  exemplar   baseline + one gold validated task spec prepended (1-shot)

All variants share the same per-position seeds so comparisons are paired.
Prints one RESULT {json} line per variant + a final SUMMARY {json}.

Usage: python scripts/probe_spark_proposer.py [baseline dual constraint exemplar]
"""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from forge.engine.forge_engine import ForgeEngine  # noqa: E402
from forge.self_play.infinite_curriculum import (  # noqa: E402
    _CHAT_MIN_PROPOSE_TOKENS,
    _CHAT_PROPOSE_CLOSE,
    _CHAT_PROPOSE_PREFILL,
    _CHAT_PROPOSE_THINK_BUDGET,
    _DESC_HINT_BANK,
    CHAT_PROPOSE_PROMPT,
    InfiniteCurriculum,
)

CKPT = ROOT / "research/checkpoints/spark_x26_1p7b"
CONFIG_NAME = "spark_x26_1p7b"
DOMAINS = ["algorithms", "math", "strings", "logic"]
DIFFICULTY = "medium"
N_PER_VARIANT = 16
VARIANTS = ["baseline", "dual", "constraint", "exemplar"]

_CONSTRAINT_BLOCK = (
    "\n\nHard requirements for the task you create:"
    "\n- include 3+ test cases with edge cases (empty, zero, negative, "
    "single-element where meaningful)"
    "\n- solvable in under 30 lines of Python"
    "\n- deterministic: no randomness, no I/O, no dates/times"
    "\n- standard library only"
)

# One hand-verified spec in the exact production format (bare test-input
# lines; executor derives outputs). Deliberately different from the
# digit-sum example already inside CHAT_PROPOSE_PROMPT.
_GOLD_SPEC = """Here is a GOLD task spec that passed executor validation in a previous round — match its quality:

```python
# Task: Return the count of elements in lst that are strictly greater than the average of all elements.
def solve(lst: list) -> int:
    if len(lst) == 0:
        return 0
    avg = sum(lst) / len(lst)
    count = 0
    for x in lst:
        if x > avg:
            count += 1
    return count

# solve([1, 2, 3, 4, 5])
# solve([])
# solve([-2, -1, 0, 1, 2])
# solve([7])
# solve([3, 3, 3])
```

"""


def build_prompts(variant: str, cur: InfiniteCurriculum,
                  rng: random.Random) -> tuple[list[str], list[str]]:
    """Build N_PER_VARIANT instruction strings for a variant.

    Returns (prompts, domains) — domains[i] is the domain slot for
    prompts[i] (round-robin over DOMAINS).
    """
    prompts, doms = [], []
    for i in range(N_PER_VARIANT):
        domain = DOMAINS[i % len(DOMAINS)]
        sig_hint = cur._random_signature(domain)
        hints = _DESC_HINT_BANK.get(domain, _DESC_HINT_BANK["algorithms"])
        h1 = rng.choice(hints)
        desc_hint = h1
        if variant == "dual":
            h2 = h1
            while h2 == h1 and len(hints) > 1:
                h2 = rng.choice(hints)
            desc_hint = f"{h1} + {h2}"
        prompt = CHAT_PROPOSE_PROMPT.format(
            domain=domain, difficulty=DIFFICULTY,
            signature=sig_hint, desc_hint=desc_hint)
        if variant == "constraint":
            prompt = prompt + _CONSTRAINT_BLOCK
        elif variant == "exemplar":
            prompt = _GOLD_SPEC + prompt
        prompts.append(prompt)
        doms.append(domain)
    return prompts, doms


def gen_all(cur: InfiniteCurriculum, prompts: list[str],
            seeds: list[int]) -> tuple[list[str], float]:
    """_generate_batch with OOM chunk fallback (16 -> 8 -> 4)."""
    try:
        t0 = time.perf_counter()
        out = cur._generate_batch(
            prompts, min_tokens=_CHAT_MIN_PROPOSE_TOKENS,
            think_budget=_CHAT_PROPOSE_THINK_BUDGET,
            close_suffix=_CHAT_PROPOSE_CLOSE,
            prefill=_CHAT_PROPOSE_PREFILL, seeds=seeds)
        return out, time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) <= 4:
            raise
        half = len(prompts) // 2
        a, ta = gen_all(cur, prompts[:half], seeds[:half])
        b, tb = gen_all(cur, prompts[half:], seeds[half:])
        return a + b, ta + tb


def run_variant(cur: InfiniteCurriculum, eng: ForgeEngine,
                variant: str, rng: random.Random) -> dict:
    prompts, doms = build_prompts(variant, cur, rng)
    seeds = [1000 + i for i in range(len(prompts))]
    completions, elapsed = gen_all(cur, prompts, seeds)
    n_tok = sum(len(eng.tokenizer.encode(c, add_special_tokens=False))
                for c in completions)

    per_dom = {d: [0, 0, 0] for d in DOMAINS}  # [n, parse_ok, valid]
    n_parse = n_valid = n_soft = 0
    descs = []
    for comp, dom in zip(completions, doms):
        per_dom[dom][0] += 1
        task = cur._parse_proposal(comp, dom, DIFFICULTY, "induction")
        if task is None:
            continue
        n_parse += 1
        per_dom[dom][1] += 1
        if cur._validate_task(task):
            if task.proposer_confidence >= 1.0:
                n_valid += 1
                per_dom[dom][2] += 1
            else:
                n_soft += 1
            if len(descs) < 4:
                descs.append(task.description[:70])

    return {
        "variant": variant, "n": len(completions),
        "n_parse": n_parse, "parse_rate": round(n_parse / len(completions), 3),
        "n_valid": n_valid, "valid_rate": round(n_valid / len(completions), 3),
        "n_soft": n_soft, "tok_s": round(n_tok / max(elapsed, 1e-9), 1),
        "elapsed_s": round(elapsed, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "per_domain": per_dom, "sample_desc": descs,
    }


def main() -> None:
    variants = sys.argv[1:] or VARIANTS
    variants = [v for v in variants if v in VARIANTS]
    # Manual activation (bench_spark_genmatrix "batch" mode): auto_activate
    # measured 0 free VRAM mid-load -> capped KV at 64 tokens + hit a
    # contiguous-bias bug in the static-KV graph path. Standard KV + BCG
    # is the proven B=8/16 throughput path on this box.
    eng = ForgeEngine.from_checkpoint(
        str(CKPT), config_name=CONFIG_NAME, device="cuda",
        auto_activate=False, load_quantize="none", quantize=False)
    eng.activate(kv_cache="standard", decoding="standard", quantize=None,
                 acceleration=None, warmup=True,
                 use_breakable_cuda_graph=True)
    print(f"loaded: {torch.cuda.memory_allocated()/2**30:.2f}GB alloc",
          flush=True)
    _ = eng.generate("hi", max_new_tokens=4, temperature=0.0)

    cur = InfiniteCurriculum(
        eng.model, eng.tokenizer, device="cuda", engine=eng,
        config_name=CONFIG_NAME,
        task_queue_dir=tempfile.mkdtemp(prefix="propose_probe_"))

    rng = random.Random(20260214)
    rows = []
    for v in variants:
        print(f"\n=== variant: {v} ===", flush=True)
        res = run_variant(cur, eng, v, rng)
        rows.append(res)
        print("RESULT " + json.dumps(res), flush=True)

    base = next((r for r in rows if r["variant"] == "baseline"), None)
    summary = {"variants": [r["variant"] for r in rows]}
    if base is not None:
        summary["deltas_vs_baseline"] = {
            r["variant"]: round(r["valid_rate"] - base["valid_rate"], 3)
            for r in rows if r["variant"] != "baseline"}
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

"""Code evolution - feed a function/script, get a verified-better one back.

The loop: generator (small coder LLM) writes brand-new implementations;
each candidate passes a correctness test + benchmark in an isolated
subprocess; a challenger becomes champion only by beating the incumbent's
re-measured mean by a noise-aware margin. Compares vs original, last
best, and the round's own candidates. Stops on stagnation.

Examples:
    # Evolve the bundled softmax target with the coder model
    python scripts/fn_evolve.py --target fast_softmax --evals 100

    # Scaffold a target file for a real codebase function, then edit it
    python scripts/fn_evolve.py --emit-target forge.engine.decoding:sample \
        --metric ms_per_call --goal min

    # After a run: best program + metrics from the DB
    python scripts/fn_evolve.py --target fast_softmax --report

    # Dry-run plumbing without a model (test hook)
    python scripts/fn_evolve.py --target fast_softmax --proposer none \
        --evals 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.evolution.code_evolve import (  # noqa: E402
    CodeEvolve, CodeEvolveConfig)


def _emit_target(ref: str, metric: str, goal: str) -> str:
    """Scaffold a target spec from an existing function's source."""
    import importlib
    import inspect
    mod_name, _, fn_name = ref.rpartition(":")
    if not mod_name or not fn_name:
        raise SystemExit("--emit-target expects module:function, e.g. "
                         "forge.engine.decoding:sample")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise SystemExit(f"cannot get source for {ref}: {e}")
    name = fn_name.replace("-", "_")
    return f'''"""Evolution target: {ref} - auto-scaffolded by fn_evolve.py.

Fill in the TODO inputs in test()/bench(), then run:
    python scripts/fn_evolve.py --target {name}.py
"""

NAME = "{name}"
GOAL = "{goal}"
METRIC = "{metric}"
TAGS = ["auto"]          # add domain tags: kv, quant, kernel, decoding...
BUDGETS = {{}}            # e.g. {{"peak_vram_gb": ("<=", 12.0)}}
REQUIRED = ["{fn_name}"]
PROMPT = ""

SEED_CODE = \'\'\'
{src}
\'\'\'

# -- correctness gate ---------------------------------------------------
def _inputs():
    # TODO: representative argument tuples for {fn_name}
    return [()]


def test(ns):
    from {mod_name} import {fn_name} as ref
    fn = ns.get("{fn_name}")
    if not callable(fn):
        return {{"ok": False, "detail": "no {fn_name}"}}
    for i, args in enumerate(_inputs()):
        try:
            y, y0 = fn(*args), ref(*args)
        except Exception as e:
            return {{"ok": False, "detail": f"input {{i}} raised {{e}}"}}
        try:
            import torch
            same = torch.allclose(torch.as_tensor(y, dtype=torch.float64),
                                  torch.as_tensor(y0, dtype=torch.float64),
                                  atol=1e-5, rtol=1e-4)
        except Exception:
            same = y == y0
        if not same:
            return {{"ok": False, "detail": f"input {{i}} mismatch"}}
    return {{"ok": True}}


def bench(ns):
    import time
    fn = ns["{fn_name}"]
    args = _inputs()[0]
    for _ in range(5):
        fn(*args)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        fn(*args)
    return {{"{metric}": (time.perf_counter() - t0) / n * 1000}}
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="target name in forge/evolution/targets/ "
                    "or path to a target spec file")
    ap.add_argument("--emit-target", metavar="MOD:FN",
                    help="print a scaffolded target spec for a codebase function")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--goal", choices=["min", "max"], default=None)
    ap.add_argument("--evals", type=int, default=200, help="eval budget")
    ap.add_argument("--reps", type=int, default=2, help="verify evals per side")
    ap.add_argument("--islands", type=int, default=3)
    ap.add_argument("--patience", type=int, default=30,
                    help="stop after N evals with no verified promotion")
    ap.add_argument("--hop-after", type=int, default=12)
    ap.add_argument("--proposer", choices=["mut", "mix", "llm", "basin",
                                         "none"],
                    default="mut",
                    help="mut=free AST mutations (default, no model), "
                         "mix=mutations + LLM leap every --llm-every, "
                         "llm=coder model only, basin=async LLM parents "
                         "+ deterministic mutator fan-out, none=seed only")
    ap.add_argument("--llm-every", type=int, default=6,
                    help="mix mode: LLM proposal every Nth cycle")
    ap.add_argument("--basin-batch", type=int, default=8,
                    help="basin mode: LLM parents per generate_batch pass")
    ap.add_argument("--fanout", type=int, default=32,
                    help="basin mode: mutator variants enumerated per parent")
    ap.add_argument("--no-eval-pool", dest="eval_pool",
                    action="store_false",
                    help="disable persistent eval workers (subprocess/eval)")
    ap.set_defaults(eval_pool=True)
    ap.add_argument("--model", default=None, help="HF repo id or local "
                    "checkpoint dir for proposer (default: "
                    "research/checkpoints/lfm3_1p7b_cc_fq — the LFM3 "
                    "Claude-Code finetune; the Qwen port is archived)")
    ap.add_argument("--proposer-device", default="cuda")
    ap.add_argument("--engine-config", default=None,
                    help="ForgeEngine preset for --model (e.g. qwen3_4b)")
    ap.add_argument("--proposer-quantize", default=None,
                    choices=["auto", "int4", "fp8", "int8"])
    ap.add_argument("--proposer-compile", action="store_true",
                    help="torch.compile the proposer (~3-4min inductor "
                         "boot tax; decode already replays static-KV "
                         "graphs so this only speeds prefill)")
    ap.add_argument("--proposer-think", action="store_true",
                    help="LFM3 proposer only: open think prefill "
                         "(reason before code) instead of the "
                         "template's empty-think direct answer")
    ap.add_argument("--lora-root", default=None,
                    help="tagged-LoRA registry; hot-swaps per target")
    ap.add_argument("--lora-tag", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel eval workers; 0 = AUTO — ramps "
                         "submission until ~85% CPU, gated by RAM (<80%%) "
                         "and, for CUDA-bench targets, free VRAM. "
                         "1 = serial subprocess-per-eval (debug path)")
    ap.add_argument("--max-workers", type=int, default=24,
                    help="hard cap for --workers 0 (auto)")
    ap.add_argument("--worker-mem-mb", type=int, default=2048,
                    help="per-worker RSS cap in MB — kill+respawn on "
                         "breach (0 disables; not recommended)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--eval-log", action="store_true",
                    help="one stderr line per eval result — the "
                         "crash-locator flag (last line before a hang "
                         "names the exact candidate/stage)")
    ap.add_argument("--log-gens", action="store_true",
                    help="dump every raw model emission to "
                         "<db>.gens.jsonl (prompt+raw, unredacted)")
    ap.add_argument("--status-file", metavar="JSON", default=None,
                    help="live progress JSON (~2s refresh; point "
                         "evolve_watch --status at it for the in-flight "
                         "job view)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore this target's history (no resume)")
    ap.add_argument("--export-pairs", metavar="JSONL", default=None,
                    help="export DPO pairs (chosen=improved child, "
                         "rejected=parent) from the db and exit")
    ap.add_argument("--export-validity", metavar="JSONL", default=None,
                    help="export validity-DPO pairs (chosen=gated "
                         "program, rejected=malformed/dup output) from "
                         "the db and exit — teaches format compliance")
    args = ap.parse_args()

    if args.emit_target:
        print(_emit_target(args.emit_target,
                           args.metric or "ms_per_call",
                           args.goal or "min"))
        return 0
    if args.export_pairs:
        from forge.evolution.code_evolve import export_pairs
        n = export_pairs(args.db, args.export_pairs, args.target)
        print(f"wrote {n} preference pairs -> {args.export_pairs}")
        return 0
    if args.export_validity:
        from forge.evolution.code_evolve import export_validity_pairs
        n = export_validity_pairs(args.db, args.export_validity,
                                  args.target)
        print(f"wrote {n} validity pairs -> {args.export_validity}")
        return 0
    if not args.target:
        ap.error("--target or --emit-target required")

    cfg = CodeEvolveConfig(
        target=args.target, metric=args.metric, goal=args.goal,
        evals_budget=args.evals, reps=args.reps, islands=args.islands,
        patience=args.patience, hop_after=args.hop_after,
        proposer=args.proposer, llm_every=args.llm_every,
        basin_batch=args.basin_batch, fanout=args.fanout,
        eval_pool=args.eval_pool,
        resume=not args.fresh,
        model_id=args.model,
        engine_config=args.engine_config,
        proposer_quantize=args.proposer_quantize,
        proposer_think=args.proposer_think,
        proposer_compile=args.proposer_compile,
        lora_root=args.lora_root, lora_tag=args.lora_tag,
        proposer_device=args.proposer_device, temperature=args.temperature,
        db_path=args.db, timeout_s=args.timeout, workers=args.workers,
        max_workers=args.max_workers, worker_mem_mb=args.worker_mem_mb,
        seed=args.seed, verbose=not args.quiet,
        eval_log=args.eval_log, log_gens=args.log_gens,
        status_file=args.status_file)
    ce = CodeEvolve(cfg)
    if args.report:
        print(ce.report())
        return 0
    res = ce.run()
    print("=" * 60)
    print(f"stop: {res.get('stop_reason')}")
    if "error" in res:
        print(f"ERROR: {res['error']} - {res.get('detail')}")
        return 1
    print(f"evals={res['evals']} improvement={res['improvement_pct']:+.1f}% "
          f"({res['elapsed_s']}s)")
    print(f"best metrics: {res['best_metrics']}")
    print("=" * 60)
    print("BEST PROGRAM:")
    print(res["best_code"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

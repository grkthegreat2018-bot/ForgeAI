"""Multi-domain code-evolution campaign.

Cycles every registered target (or a subset) for N rounds on a SHARED
db, so the mutation-operator bandit, code_insights, elite archive, and
seen-hashes all carry over between targets and rounds. Each target
resumes its own historical best (CodeEvolve resume=1 default).

Example (long background run):
    python scripts/evolve_all.py --rounds 4 --evals 150 --proposer mut ^
        --db forge_evolve.db > evolve_campaign.log 2>&1

Smaller smoke:
    python scripts/evolve_all.py --targets fast_softmax --evals 10 --rounds 1
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.evolution.code_evolve import (  # noqa: E402
    CodeEvolve, CodeEvolveConfig)

# name -> per-run eval budget. Ordered roughly most-promising first.
# Every target is a ForgeEngine decode/load hot path or an evolve-loop
# self-target — a verified win ports straight back into production code.
DEFAULT_TARGETS = {
    "sample_pipeline": 150,
    "min_k_filter": 150,
    "top_p_filter": 150,
    "top_p_probs": 150,
    "suffix_mem": 150,
    "dequantize_4bit": 150,
    "quantize_4bit": 150,
    "dry_penalties": 120,
    "radix_prefix": 120,
    "stop_matcher": 120,
    "prefix_hash": 100,
    "canon_key": 100,
    "fast_softmax": 100,
    "decode_step": 100,   # audit A11: e2e decode-step meta-target
    # Domain targets (targets/domains/) — resolvable via Target.resolve.
    "cev_math_solver": 100,
    "cev_rle_codec": 100,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=None,
                    help="comma-separated subset (default: all known targets)")
    ap.add_argument("--evals", type=int, default=None,
                    help="override per-target eval budget")
    ap.add_argument("--rounds", type=int, default=3,
                    help="campaign passes over the target list; "
                         "0 = run until --quiet-rounds consecutive "
                         "rounds produce zero promotions anywhere")
    ap.add_argument("--quiet-rounds", type=int, default=3,
                    help="infinite mode: stop after N straight rounds "
                         "with zero verified promotions on ANY target")
    ap.add_argument("--proposer", choices=["mut", "mix", "llm", "basin"],
                    default="mut")
    ap.add_argument("--llm-every", type=int, default=6)
    ap.add_argument("--basin-batch", type=int, default=8,
                    help="basin proposer: LLM parents per generate_batch pass")
    ap.add_argument("--fanout", type=int, default=32,
                    help="basin proposer: mutator variants per parent")
    ap.add_argument("--model", default=None)
    ap.add_argument("--engine-config", default=None,
                    help="ForgeEngine preset for --model (e.g. qwen3_4b)")
    ap.add_argument("--proposer-device", default="cuda")
    ap.add_argument("--proposer-quantize", default=None,
                    choices=[None, "auto", "int4", "fp8", "int8"],
                    help="packed-weight quantize for the proposer model")
    ap.add_argument("--lora-root", default=None,
                    help="tagged-LoRA registry dir; proposer hot-swaps "
                         "adapters per target")
    ap.add_argument("--lora-tag", default=None)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--hop-after", type=int, default=15)
    ap.add_argument("--islands", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel eval workers per target run; 0 = auto "
                         "(persistent EvalPool — subprocess spawn cost "
                         "paid once per worker, ~25x throughput vs the "
                         "per-eval subprocess path)")
    ap.add_argument("--status", default=None,
                    help="status file stem: <stem>.<target>.json per "
                         "target (evolve_watch / GUI poll these)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="run up to N targets concurrently; 0 = AUTO "
                         "(cpu_count//8, min 2). Shared db — WAL+"
                         "busy_timeout already handle the writes; "
                         "parallel*workers cuda-eval subprocesses share "
                         "the 12GB GPU")
    ap.add_argument("--max-workers", type=int, default=12,
                    help="per-run hard cap when --workers 0")
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-targets", action="store_true",
                    help="merge auto-harvested targets (@self_improve "
                         "specs materialized as auto_*) into the round")
    ap.add_argument("--export-pairs", metavar="JSONL",
                    default="data/sft/evolve_pairs.jsonl")
    ap.add_argument("--export-validity", metavar="JSONL", default=None,
                    help="also export validity-DPO pairs (chosen=gated "
                         "program, rejected=malformed/dup output) — the "
                         "format-compliance teaching signal")
    args = ap.parse_args()

    auto: dict = {}
    if args.auto_targets:
        try:
            from forge.evolution.auto_targets import collect_targets
            auto = collect_targets()
            if auto:
                print(f"[evolve_all] +{len(auto)} auto-harvested "
                      f"targets: " + ", ".join(sorted(auto)), flush=True)
        except Exception as e:
            print(f"[evolve_all] auto-target harvest failed ({e}); "
                  "hand seeds only", flush=True)
    budget_map = {**DEFAULT_TARGETS, **auto}
    if args.targets:
        names = [t.strip() for t in args.targets.split(",") if t.strip()]
        targets = {n: budget_map.get(n, args.evals or 120)
                   for n in names}
    else:
        targets = dict(budget_map)
    if args.evals:
        targets = {n: args.evals for n in targets}

    if args.parallel <= 0:
        import os
        args.parallel = max(2, min(len(targets), (os.cpu_count() or 8) // 8))
        print(f"[evolve_all] parallel=auto -> {args.parallel} targets")
    else:
        args.parallel = min(args.parallel, len(targets))

    def _run_target(name: str, budget: int, rnd: int):
        cfg = CodeEvolveConfig(
            target=name, evals_budget=budget, reps=args.reps,
            islands=args.islands, patience=args.patience,
            hop_after=args.hop_after, proposer=args.proposer,
            llm_every=args.llm_every,
            basin_batch=args.basin_batch, fanout=args.fanout,
            model_id=args.model,
            engine_config=args.engine_config,
            proposer_device=args.proposer_device,
            proposer_quantize=args.proposer_quantize,
            lora_root=args.lora_root, lora_tag=args.lora_tag,
            db_path=args.db, timeout_s=args.timeout,
            workers=args.workers, max_workers=args.max_workers,
            status_file=(f"{args.status}.{name}.json"
                         if args.status else None),
            seed=args.seed + rnd * 1000 + zlib.crc32(
                name.encode()) % 997,
            resume=True, verbose=True)
        try:
            res = CodeEvolve(cfg).run()
        except Exception:
            return (name, rnd, "CRASH", 0.0, traceback.format_exc(), 0)
        return (name, rnd, res.get("stop_reason", "?"),
                res.get("improvement_pct", 0.0), res.get("evals"),
                res.get("promotions", 0))

    t_all = time.time()
    summary = []
    quiet = 0
    rnd = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    while True:
        rnd += 1
        if args.rounds > 0 and rnd > args.rounds:
            break
        tag_r = (f"{rnd}/{args.rounds}" if args.rounds > 0
                 else f"{rnd} (inf, quiet {quiet}/{args.quiet_rounds})")
        print(f"\n{'=' * 66}\nCAMPAIGN ROUND {tag_r} "
              f"({time.strftime('%H:%M:%S')})\n{'=' * 66}", flush=True)
        items = list(targets.items())
        round_promos = 0
        if args.parallel > 1 and len(items) > 1:
            futs = {}
            with ThreadPoolExecutor(max_workers=args.parallel,
                                    thread_name_prefix="cev_target") as tp:
                for name, budget in items:
                    futs[tp.submit(_run_target, name, budget, rnd)] = name
                for fut in as_completed(futs):
                    name, r, tag, imp, extra, promos = fut.result()
                    round_promos += promos
                    if tag == "CRASH":
                        print(f"[{name}] CRASHED:\n{extra}", flush=True)
                    else:
                        print(f">>> [{name}] stop={tag} evals={extra} "
                              f"improvement={imp:+.1f}% "
                              f"promos={promos}", flush=True)
                    summary.append((name, r, tag, imp))
        else:
            for name, budget in items:
                name, r, tag, imp, extra, promos = _run_target(
                    name, budget, rnd)
                round_promos += promos
                if tag == "CRASH":
                    print(f"[{name}] CRASHED:\n{extra}", flush=True)
                else:
                    print(f">>> [{name}] stop={tag} evals={extra} "
                          f"improvement={imp:+.1f}% promos={promos}",
                          flush=True)
                summary.append((name, r, tag, imp))
        if args.rounds > 0:
            continue
        quiet = quiet + 1 if round_promos == 0 else 0
        print(f"[evolve_all] round {rnd}: {round_promos} promotions "
              f"(quiet streak {quiet}/{args.quiet_rounds})", flush=True)
        if quiet >= args.quiet_rounds:
            print(f"[evolve_all] deep stagnation: {quiet} promotion-free "
                  f"rounds — stopping", flush=True)
            break

    print(f"\n{'=' * 66}\nCAMPAIGN SUMMARY "
          f"({(time.time() - t_all) / 60:.1f} min)\n{'=' * 66}")
    for name, rnd, tag, imp in summary:
        print(f"  r{rnd} {name:20s} {tag:12s} {imp:+.1f}%")

    if args.export_pairs:
        try:
            Path(args.export_pairs).parent.mkdir(
                parents=True, exist_ok=True)
            from forge.evolution.code_evolve import export_pairs
            n = export_pairs(args.db, args.export_pairs)
            print(f"\nexported {n} preference pairs -> {args.export_pairs}")
        except Exception as e:
            print(f"pair export failed: {e}")
    if args.export_validity:
        try:
            from forge.evolution.code_evolve import (
                export_validity_pairs)
            n = export_validity_pairs(args.db, args.export_validity)
            print(f"exported {n} validity pairs -> {args.export_validity}")
        except Exception as e:
            print(f"validity export failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

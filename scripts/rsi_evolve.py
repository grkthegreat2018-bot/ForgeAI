"""RSI campaign: evolve ForgeAI LLM-research kernels, then hone the
proposer on its own verified wins — recursive self-improvement loop.

Round structure:
  1. CodeEvolve on every target (proposer=basin: async Qwen basin
     emitter + deterministic mutator fan-out + persistent eval pool)
  2. export_sft() -> every verified child>parent pair becomes a
     prompt->improved-code SFT row (the true task distribution)
  3. finetune_proposer --mode lora -> tagged adapters + router
  4. next round: cfg.lora_root stays set — LLMProposer hot-swaps the
     adapters it trained on its own wins. THAT is the RSI step: the
     proposer gets better at proposing exactly the kind of code that
     verified, on the targets that matter to ForgeAI inference.

Scientific-importance bar: only VERIFIED wins enter training data
(test() semantics gate + bench + replicated margin vs incumbent), and
"major win" = a verified promotion, logged as a code_insights row.
Duds never reach the model.

Stop: --rounds, --minutes, or --quiet-rounds straight zero-promotion
rounds. Status: per-target <status>.<target>.json (evolve_watch/GUI
format) + campaign events appended to <status>.events.jsonl.

Example:
    python scripts/rsi_evolve.py --rounds 0 --evals 200 --workers 0 ^
        --db forge_evolve.db --status results/rsi_status.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# LLM-research target set: real ForgeAI decode/load hot paths plus
# evolve-loop self-targets — a verified win ports straight back into
# production code. Ordered roughly most-promising first; --evals overrides.
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
    # System-level meta-target (audit A11): e2e decode-step us/token —
    # the metric that actually moves system tok/s, unlike µs micro-kernels.
    "decode_step": 100,
    # Non-code demonstration domains (forge/evolution/domains/cev_*.py —
    # spec-contract targets, resolved via Target.resolve's domains/ leg):
    # a math/reasoning solver scored on accuracy and a byte codec scored
    # on compression ratio. Proof the loop isn't codegen-only.
    "cev_math_solver": 60,
    "cev_rle_codec": 60,
}


def _emit(args, ev: dict) -> None:
    """Append one campaign event to <status>.events.jsonl."""
    if not args.status:
        return
    try:
        p = Path(f"{args.status}.events.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        ev["ts"] = time.time()
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")
    except Exception:
        pass


def _finetune(args, sft_path: str, n_rows: int) -> bool:
    """LoRA-mode hone: tagged adapters per target + router. Runs in a
    subprocess (own CUDA context); the evolve engine's proposer picks
    the new adapters up next round via the registry."""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "finetune_proposer.py"),
        "--db", args.db, "--data", sft_path, "--force-export",
        "--mode", "lora", "--lora-root", args.lora_root,
        "--lora-steps", str(args.lora_steps),
        "--lora-r", str(args.lora_r), "--lora-alpha", str(args.lora_alpha),
        "--lora-min-new", str(args.min_new),
        "--batch-size", str(args.train_batch),
        "--seq-len", str(args.seq_len),
        "--ram-limit", "96",
    ]
    if args.model:
        cmd += ["--base", args.model]
    if args.proposer_quantize:
        cmd += ["--proposer-quantize", args.proposer_quantize]
    print(f"[rsi] honing proposer: {n_rows} SFT rows -> {args.lora_root}",
          flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    ok = rc == 0 and (Path(args.lora_root) / "manifest.json").exists()
    print(f"[rsi] finetune {'ok' if ok else f'FAILED rc={rc}'}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=None,
                    help="comma-separated subset (default: all LLM targets)")
    ap.add_argument("--no-auto-targets", dest="auto_targets",
                    action="store_false",
                    help="skip auto-harvested targets (production "
                         "functions marked @self_improve); on by default")
    ap.set_defaults(auto_targets=True)
    ap.add_argument("--evals", type=int, default=None,
                    help="override per-target eval budget")
    ap.add_argument("--rounds", type=int, default=0,
                    help="campaign passes; 0 = until --quiet-rounds or "
                         "--minutes")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="wall-clock cap (0 = none)")
    ap.add_argument("--quiet-rounds", type=int, default=3,
                    help="stop after N straight zero-promotion rounds")
    ap.add_argument("--proposer", default="basin",
                    choices=["basin", "mix", "mut", "llm"],
                    help="basin=async LLM parents + mutator fan-out")
    ap.add_argument("--llm-every", type=int, default=6)
    ap.add_argument("--basin-batch", type=int, default=8)
    ap.add_argument("--fanout", type=int, default=32)
    ap.add_argument("--model", default=None)
    ap.add_argument("--engine-config", default=None)
    ap.add_argument("--proposer-device", default="cuda")
    ap.add_argument("--proposer-backend", default="forge",
                    choices=["forge", "ov"],
                    help="ov = OpenVINO GenAI server (NPU/CPU offload)")
    ap.add_argument("--proposer-ov-url", default="http://127.0.0.1:8790")
    ap.add_argument("--proposer-quantize", default=None,
                    choices=[None, "auto", "int4", "fp8", "int8"])
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--hop-after", type=int, default=15)
    ap.add_argument("--islands", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto-ramp eval pool workers")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", default=None,
                    help="status file stem for GUI/evolve_watch")
    # ── honing knobs ────────────────────────────────────────────────
    ap.add_argument("--no-train", dest="train", action="store_false",
                    help="evolve only — skip the proposer-honing step")
    ap.set_defaults(train=True)
    ap.add_argument("--lora-root", default="forge/evolution/loras")
    ap.add_argument("--finetune-every", type=int, default=1,
                    help="hone every Nth round (rounds with new wins)")
    ap.add_argument("--min-new", type=int, default=8,
                    help="min NEW verified rows before a hone run fires")
    ap.add_argument("--lora-steps", type=int, default=120)
    ap.add_argument("--lora-r", type=int, default=32)
    # alpha=rank -> scale 1.0. The 64/32 (scale 2.0) default produced
    # the 'default' adapter that drove the proposer to 0% parseable
    # output; finetune_proposer already corrected its own default.
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--train-batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048)
    # ── web-research tools for the proposer ─────────────────────────
    ap.add_argument("--research", action="store_true",
                    help="give the proposer web tools: refresh "
                         "docs/evolve_research.md at round start "
                         "(tavily/exa keys from .env), inject the "
                         "digest into prompts, and let the model emit "
                         "'SEARCH: <query>' lines — answered via "
                         "research.py, results injected, ONE regen")
    ap.add_argument("--research-budget", type=int, default=32,
                    help="uncached web searches per process (~daily "
                         "quota; cached queries are free)")
    args = ap.parse_args()

    from forge.evolution.code_evolve import (
        CodeEvolve, CodeEvolveConfig, export_sft)

    # Auto-harvested targets: production functions that opted in via
    # @self_improve get materialized as auto_* spec files and merged in.
    auto: dict = {}
    if args.auto_targets:
        try:
            from forge.evolution.auto_targets import collect_targets
            auto = collect_targets()
            if auto:
                print(f"[rsi] +{len(auto)} auto-harvested targets: "
                      + ", ".join(sorted(auto)), flush=True)
        except Exception as e:
            print(f"[rsi] auto-target harvest failed ({e}); "
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

    sft_path = str(Path(args.db).with_suffix(".rsi_sft.jsonl"))
    prev_rows = 0
    try:
        prev_rows = export_sft(args.db, sft_path)
    except Exception:
        pass
    trained_once = Path(args.lora_root, "manifest.json").exists()

    t_all = time.time()
    rnd = 0
    quiet = 0
    total_promos = 0
    while True:
        rnd += 1
        if args.rounds > 0 and rnd > args.rounds:
            break
        if args.minutes and (time.time() - t_all) / 60 >= args.minutes:
            print(f"[rsi] time budget {args.minutes}min reached", flush=True)
            break
        print(f"\n{'=' * 66}\nRSI ROUND {rnd} "
              f"({time.strftime('%H:%M:%S')}) lora={'on' if trained_once else 'pending'}"
              f"\n{'=' * 66}", flush=True)
        if args.research:
            # Refresh the technique digest at round start — but only when
            # missing or stale (>20h): rounds can start many times a day
            # and each refresh is ~12 API queries.
            try:
                from forge.evolution.research import research as _res
                digest = ROOT / "docs" / "evolve_research.md"
                if (not digest.exists()
                        or time.time() - digest.stat().st_mtime
                        > 20 * 3600):
                    _res()
            except Exception as e:
                print(f"[rsi] research refresh failed: {e}", flush=True)
        round_promos = 0
        for name, budget in targets.items():
            if args.minutes and (time.time() - t_all) / 60 \
                    >= args.minutes:
                break
            cfg = CodeEvolveConfig(
                target=name, evals_budget=budget, reps=args.reps,
                islands=args.islands, patience=args.patience,
                hop_after=args.hop_after, proposer=args.proposer,
                llm_every=args.llm_every, basin_batch=args.basin_batch,
                fanout=args.fanout, model_id=args.model,
                engine_config=args.engine_config,
                proposer_device=args.proposer_device,
                proposer_backend=args.proposer_backend,
                proposer_ov_url=args.proposer_ov_url,
                proposer_quantize=args.proposer_quantize,
                lora_root=args.lora_root,
                research=args.research,
                research_budget=args.research_budget,
                research_digest=(
                    str(ROOT / "docs" / "evolve_research.md")
                    if args.research else None),
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
                print(f"[{name}] CRASHED:\n{traceback.format_exc()}",
                      flush=True)
                continue
            p = res.get("promotions", 0)
            round_promos += p
            total_promos += p
            print(f">>> [{name}] stop={res.get('stop_reason')} "
                  f"evals={res.get('evals')} "
                  f"improvement={res.get('improvement_pct', 0):+.1f}% "
                  f"promos={p}", flush=True)
            _emit(args, {"kind": "target_done", "round": rnd,
                         "target": name,
                         "evals": res.get("evals"),
                         "improvement_pct": res.get("improvement_pct"),
                         "promotions": p,
                         "stop_reason": res.get("stop_reason")})

        # ── hone the proposer on its own verified wins ───────────────
        # Release the proposer's VRAM before the hone — the lifecycle is
        # load → propose → unload → judge → train → loop. _engine()
        # lazily rebuilds on the next round's first proposal.
        try:
            from forge.evolution.code_evolve import LLMProposer
            LLMProposer.release_engines()
        except Exception:
            pass
        n_rows = 0
        try:
            n_rows = export_sft(args.db, sft_path)
        except Exception as e:
            print(f"[rsi] sft export failed: {e}", flush=True)
        # Validity-DPO rows alongside SFT: malformed/dup outputs as
        # 'rejected' — teaches the proposer the format contract, which
        # is where most lost generations die.
        try:
            from forge.evolution.code_evolve import \
                export_validity_pairs
            n_vp = export_validity_pairs(
                args.db, str(Path(sft_path).with_suffix(".validity.jsonl")))
            if n_vp:
                print(f"[rsi] {n_vp} validity-DPO pairs exported",
                      flush=True)
        except Exception as e:
            print(f"[rsi] validity export failed: {e}", flush=True)
        new_rows = max(0, n_rows - prev_rows)
        prev_rows = n_rows
        if (args.train and new_rows >= args.min_new
                and rnd % args.finetune_every == 0):
            trained_once = _finetune(args, sft_path, n_rows) or trained_once
            _emit(args, {"kind": "finetune", "round": rnd,
                         "sft_rows": n_rows, "new_rows": new_rows,
                         "ok": trained_once})
        else:
            print(f"[rsi] round {rnd}: {new_rows} new verified rows "
                  f"(need {args.min_new}) — hone skipped", flush=True)

        quiet = quiet + 1 if round_promos == 0 else 0
        print(f"[rsi] round {rnd}: {round_promos} promotions, "
              f"{new_rows} new SFT rows (quiet {quiet}/"
              f"{args.quiet_rounds})", flush=True)
        _emit(args, {"kind": "round_done", "round": rnd,
                     "promotions": round_promos,
                     "total_promotions": total_promos,
                     "new_sft_rows": new_rows, "quiet": quiet})
        if args.rounds <= 0 and quiet >= args.quiet_rounds:
            print(f"[rsi] deep stagnation — {quiet} promotion-free "
                  "rounds, stopping", flush=True)
            break

    mins = (time.time() - t_all) / 60
    print(f"\n{'=' * 66}\nRSI DONE: {rnd} rounds, {total_promos} verified "
          f"promotions in {mins:.1f} min\n{'=' * 66}")
    _emit(args, {"kind": "campaign_done", "rounds": rnd,
                 "total_promotions": total_promos, "minutes": mins})
    return 0


if __name__ == "__main__":
    sys.exit(main())

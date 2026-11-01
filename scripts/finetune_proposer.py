"""Fine-tune the code-evolution proposer — canonical trainer path.

Data sources:
  - OUR verified wins: export_sft() pulls parent->improved-child pairs
    from code_programs (the true task distribution: our targets, our
    prompt format, verified improvements).
  - PIE (Performance-Improving Edits, rootacess/pie-perf): ~40K real
    slow->fast Python edits with measured speedups — teaches the
    semantics-preserving-speedup prior at scale.

Two-stage recipe (PIE prior, then domain specialization):
  stage1: sft_train --hf-model <base> on PIE rows (+ ours mixed in)
  stage2: sft_train --hf-model <stage1 dir> on OUR rows only, lower LR
The saved .safetensors is assembled into a loadable HF dir each stage
(base tokenizer/config copied in); fn_evolve --model <dir> loads it.

Usage:
    python scripts/finetune_proposer.py --db forge_evolve.db \
        --pie 8000 --out forge/evolution/proposer_ckpt --max-steps 500
    python scripts/fn_evolve.py --target dry_penalties --proposer mix \
        --model forge/evolution/proposer_ckpt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HF_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json",
)


def _assemble_hf_dir(ckpt: Path, base_dir: Path, out: Path) -> Path:
    """Trained .safetensors + base tokenizer/config -> loadable HF dir."""
    out.mkdir(parents=True, exist_ok=True)
    for name in _HF_FILES:
        src = base_dir / name
        if src.exists() and src.resolve() != (out / name).resolve():
            shutil.copy2(src, out / name)
    if ckpt.resolve() != (out / "model.safetensors").resolve():
        shutil.copy2(ckpt, out / "model.safetensors")
    return out


def _pie_prompt(code_v0: str, rt0: float | None) -> str:
    """Our prompt template over a PIE slow->fast pair. Chat-path ending
    (matches the deployed proposer: primed=False -> 'Reply with ONLY')."""
    tag = hashlib.sha1(code_v0.encode()).hexdigest()[:8]
    cur = (f"runtime={rt0:.4g}s" if rt0 else "runtime=?")
    return (
        "# Rewrite this Python program to be much better.\n"
        f"# Candidate: {tag}\n"
        "# Direction: runtime MINIMIZE\n"
        "# Hard budgets (must satisfy): {}\n"
        "# Rules: keep public names; stdlib+numpy+torch only;\n"
        "# output ONLY one python code block; identical semantics.\n"
        "#\n"
        f"# Current best ({cur}):\n"
        "```python\n" + code_v0.strip() + "\n```\n"
        "Reply with ONLY one ```python code block containing the "
        "complete improved implementation.")


def import_pie(dataset_id: str, n: int, out_path: str,
               min_improve: float = 5.0,
               max_chars: int = 2600) -> int:
    """PIE -> our {prompt, response} rows. Keeps pairs with a real
    measured speedup and code short enough to fit seq_len 2048."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[pie] datasets not installed")
        return 0
    try:
        ds = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"[pie] load failed ({e}); trying 'python' config")
        ds = load_dataset(dataset_id, "python", split="train")
    cols = set(ds.column_names)
    print(f"[pie] {len(ds)} rows, cols={sorted(cols)[:12]}")
    n_out = 0
    seen: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if n_out >= n:
                break
            if str(row.get("language", "python")).lower() != "python":
                continue
            frac = float(row.get("improvement_frac") or 0)
            if frac < min_improve:
                continue
            # skip trivial pairs: whitespace-only diffs / import swaps
            if str(row.get("code_same")) == "True" or \
                    str(row.get("diff_only_import_comment")) == "True":
                continue
            c0 = row.get("input") or row.get("code_v0") or ""
            c1 = row.get("target") or row.get("code_v1") or ""
            c0, c1 = c0.strip(), c1.strip()
            if not c0 or not c1 or c0 == c1:
                continue
            if len(c0) > max_chars or len(c1) > max_chars + 800:
                continue
            h = hashlib.sha1(c0.encode()).hexdigest()[:12]
            if h in seen:
                continue
            seen.add(h)
            try:
                compile(c1, "<pie>", "exec")
            except SyntaxError:
                continue
            rt0 = row.get("measured_runtime_v0") or row.get("cpu_time_v0")
            f.write(json.dumps({
                "prompt": _pie_prompt(c0, rt0),
                "response": f"```python\n{c1}\n```",
                "delta": frac,
                "target": "pie",
            }) + "\n")
            n_out += 1
    print(f"[pie] -> {n_out} usable rows -> {out_path}")
    return n_out


def _run_sft(data: str, base: str, ckpt: Path, args,
             lr: float, max_steps: int) -> int:
    cmd = [
        sys.executable, "-m", "forge.training.runners.sft_train",
        "--data", data,
        "--hf-model", base,
        # sft_train tokenizes the dataset BEFORE its --hf-model branch
        # swaps in the HF tokenizer — without this the rows are encoded
        # with the canonical ForgeLM-v2 tokenizer (missing on this
        # checkout -> crash; where present -> silent vocab corruption).
        "--tokenizer", base,
        "--save", str(ckpt),
        "--max-steps", str(max_steps),
        "--lr", str(lr),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--optimizer", args.optimizer,
        "--no-pack-sequences",               # prompt+code pairs —
        "--no-bitnet-everywhere",            # packing/bitnet keys would
        "--ram-limit-percent", str(args.ram_limit),
    ]                                        # corrupt rows/HF reload
    print("[sft] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


# LoRA target module names, per deployed Forge arch. These MUST match the
# checkpoint the proposer actually loads: training on HF key names
# (model.layers.N.self_attn.*) produced adapters that could never attach to
# the Forge engine's blocks.N.* modules — the hone loop was a silent no-op.
_LORA_TARGETS = {
    "qwen25_05b_mla_moe":
        "q_proj,out_proj,kv_down_proj,k_up_proj,v_up_proj,"
        "k_rope_proj,w1,w2,w3",
    # LFM3: 'out_proj' hits BOTH attention o_proj and conv out_proj (conv
    # lives at blocks.N.attn.out_proj too); 'in_proj' is conv-only.
    # 'gate_proj' (16 out-features) is skipped by min_size=64 anyway.
    "lfm3_1p7b":
        "q_proj,k_proj,v_proj,out_proj,in_proj,w_gate,w_up,w_down",
    "*": "q_proj,k_proj,v_proj,out_proj,w_gate,w_up,w_down",
}


def _default_proposer_base() -> str:
    """Same resolution as LLMProposer._model_id: the LFM3 Claude-Code
    finetune (_fq forge_quant first, then the fp16 dir). The Qwen port
    is archived — adapters trained on it mismatch the deployed model."""
    repo = Path(__file__).resolve().parent.parent
    for d in ("lfm3_1p7b_cc_fq", "lfm3_1p7b_cc"):
        p = repo / "research" / "checkpoints" / d
        if (p / "forge_convert.json").exists():
            return str(p)
    raise FileNotFoundError(
        "no LFM3 proposer checkpoint under research/checkpoints/ — "
        "pass --base <dir> explicitly")


def _resolve_forge_base(base: str):
    """-> (checkpoint_dir, forge_config, lora_targets) for sft_train's
    ForgeEngine load path. Training on the deployed checkpoint format is
    what makes the saved adapter keys match at hot-load time."""
    p = Path(base)
    if not p.is_dir():
        from forge.exp.harness import snap_dir
        p = Path(snap_dir(base))
    named = "lfm3_1p7b"
    conv = p / "forge_convert.json"
    if conv.exists():
        named = json.loads(conv.read_text()).get("forge_config", named)
    return p, named, _LORA_TARGETS.get(named, _LORA_TARGETS["*"])


def _stale_adapter_tags(db_path: str, min_children: int = 24,
                        win_floor: float = 0.05) -> set[str]:
    """Adapter tags whose post-train proposals stopped beating parents.

    code_programs.adapter records which adapter produced each evaluated
    child. A tag with >= min_children evaluated children and a
    child-beats-parent win rate below win_floor has regressed/overfit —
    the manager then RETRAINS it from scratch instead of warm-starting
    stale weights forever (nothing previously marked tags stale, so the
    retrain path was unreachable)."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(code_programs)")}
        if "adapter" not in cols:
            return set()
        scores = dict(conn.execute(
            "SELECT code_hash, score FROM code_programs WHERE ok=1"))
        kids: dict[str, int] = {}
        wins: dict[str, int] = {}
        for ad, ph, s, ok in conn.execute(
                "SELECT adapter, parent_hash, score, ok FROM code_programs"
                " WHERE adapter IS NOT NULL"):
            if not ad:
                continue
            kids[ad] = kids.get(ad, 0) + 1
            if ok and ph and s > scores.get(ph, -1e18):
                wins[ad] = wins.get(ad, 0) + 1
        return {t for t, n in kids.items()
                if n >= min_children and wins.get(t, 0) / n < win_floor}
    except Exception:
        return set()


def _lora_mode(own: list[dict], args) -> int:
    """Manager-planned LoRA training: update existing adapters or create
    new ones only when a topic earns it (anti-spam). Writes one
    '<tag>.lora.safetensors' per plan + registry entry."""
    from forge.evolution.lora_registry import LoRAManager
    mgr = LoRAManager(args.lora_root)
    fresh = {t.strip() for t in args.lora_fresh.split(",") if t.strip()}
    stale = _stale_adapter_tags(args.db)
    for t in sorted(stale):
        mgr.reg.mark(t, stale=True)
        print(f"[lora] '{t}' marked stale — post-train win rate below "
              f"floor; retraining fresh", flush=True)
    fresh |= stale
    plans = mgr.plan(own, min_new=args.lora_min_new, fresh=fresh)
    if not plans:
        print("[lora] manager produced no plans")
        return 1
    for p in plans:
        print(f"[lora] plan: {p.action} '{p.tag}' "
              f"({len(p.rows)} rows{f', init={p.init_from}' if p.init_from else ''})")

    if args.base is None:
        args.base = _default_proposer_base()
    ckpt_dir, forge_cfg, lora_targets = _resolve_forge_base(args.base)
    print(f"[lora] base={ckpt_dir} config={forge_cfg} "
          f"targets={lora_targets}", flush=True)
    root = Path(args.lora_root)
    root.mkdir(parents=True, exist_ok=True)
    # A20: sft_train's --save writes the FULL merged model — a multi-GB
    # torch.save blob that used to land extensionless inside
    # forge/evolution/loras*/ (3.98GB 'min_k_filter', etc). Keep it out
    # of the package: merged saves go to
    # research/checkpoints/lora_merged/<registry>/, and only the
    # hot-swap adapter (<tag>.lora.safetensors) is moved back into the
    # registry root after a successful run.
    merged_dir = (Path("research/checkpoints/lora_merged")
                  / Path(args.lora_root).name)
    merged_dir.mkdir(parents=True, exist_ok=True)
    rc_all = 0
    for p in plans:
        data = root / f"{p.tag}.lora.jsonl"
        with open(data, "w", encoding="utf-8") as f:
            for r in p.rows:
                f.write(json.dumps(
                    {"prompt": r["prompt"], "response": r["response"]}) + "\n")
        stem = root / p.tag
        save_stem = merged_dir / f"{p.tag}.merged.pt"
        cmd = [
            sys.executable, "-m", "forge.training.runners.sft_train",
            "--data", str(data),
            "--checkpoint", str(ckpt_dir),
            "--config", forge_cfg,
            "--tokenizer", str(ckpt_dir),
            "--save", str(save_stem),
            "--max-steps", str(args.lora_steps), "--lr", str(args.lora_lr),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--optimizer", args.optimizer,
            "--no-pack-sequences", "--no-bitnet-everywhere",
            "--manual-lora", "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--lora-targets", lora_targets,
            "--save-lora-adapter",
            "--val-every", str(args.lora_val_every),
            "--val-size", str(args.lora_val_size),
            "--early-stop-patience", str(args.lora_early_stop),
            "--ram-limit-percent", str(args.ram_limit),
        ]
        if p.init_from and Path(p.init_from).exists():
            cmd += ["--lora-init", p.init_from]
        print("[lora] " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        # sft_train derives the adapter path from --save, so it lands
        # next to the merged save in merged_dir — move it into the
        # registry root where LoRARegistry/manifest expects it.
        adapter_src = save_stem.with_suffix(".lora.safetensors")
        adapter = stem.with_suffix(".lora.safetensors")
        if rc != 0 or not adapter_src.exists():
            print(f"[lora] train failed for '{p.tag}' rc={rc}")
            rc_all = 1
            continue
        shutil.move(str(adapter_src), str(adapter))
        if len(p.rows) < args.min_rows:
            print(f"[lora] '{p.tag}' trained on {len(p.rows)} rows "
                  f"(< min-rows {args.min_rows}) — NOT registering "
                  "(thin adapters corrupt the proposer)", flush=True)
            continue
        prev = mgr.reg.get(p.tag)
        mgr.reg.register(
            p.tag, adapter.name, rank=args.lora_r, alpha=args.lora_alpha,
            target_modules=lora_targets.split(","),
            base=args.base,
            trained_rows=(prev.meta.get("trained_rows", 0) if prev else 0)
            + len(p.rows),
            last_action=p.action)
        print(f"[lora] '{p.tag}' -> {adapter} (registered)")

    # ── decision-model router: embeds via the proposer model itself ──
    if args.train_router and not rc_all:
        tag_prompts = {}
        for p in plans:
            ent = mgr.reg.get(p.tag)
            if ent is not None:
                tag_prompts[p.tag] = [r["prompt"] for r in p.rows]
        if len(tag_prompts) >= 2:
            try:
                from forge.evolution.lora_router import LoRARouter
                eng = _router_engine(args)
                meta = LoRARouter.fit(root, eng.embed, tag_prompts)
                print(f"[router] trained on {meta['trained_rows']} embeds "
                      f"over {meta['tags']} (val_loss "
                      f"{meta['val_loss']:.4f}) -> {root / 'router.json'}")
            except Exception as e:
                print(f"[router] training skipped: {e}")
    return rc_all


def _router_engine(args):
    """Load the proposer engine for router-fit embeddings — same model
    that serves the adapters, so features match deployment."""
    from forge.engine.forge_engine import ForgeEngine
    from forge.exp.harness import snap_dir
    base = Path(args.base)
    ckpt = str(base.resolve()) if base.is_dir() else snap_dir(args.base)
    named = "lfm3_1p7b"
    conv = Path(ckpt) / "forge_convert.json"
    if conv.exists():
        named = json.loads(conv.read_text()).get("forge_config", named)
    return ForgeEngine.from_checkpoint(
        ckpt, config_name=named, device="cuda",
        load_quantize=getattr(args, "proposer_quantize", None),
        use_compile=False, use_breakable_cuda_graph=False,
        use_block_fusion=False, use_learned_prefix_cache=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--data", default="proposer_sft.jsonl")
    ap.add_argument("--pie", type=int, default=0,
                    help="import N PIE rows and run stage-1 training")
    ap.add_argument("--pie-dataset", default="rootacess/pie-perf")
    ap.add_argument("--pie-data", default="pie_sft.jsonl")
    ap.add_argument("--base", default=None,
                    help="HF id or checkpoint dir (default: LFM3 "
                         "Claude-Code ckpt under research/checkpoints/) "
                         "— same artifact the proposer deploys, so "
                         "adapter keys match")
    ap.add_argument("--out", default="forge/evolution/proposer_ckpt")
    ap.add_argument("--max-steps", type=int, default=150,
                    help="stage-2 steps (ours only)")
    ap.add_argument("--pie-steps", type=int, default=500,
                    help="stage-1 steps (PIE + ours)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--pie-lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--optimizer", default="muon_sf")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--ram-limit", type=float, default=96.0,
                    help="sft_train RAM throttle % — default 90 stalls "
                         "forever when the host is already at ~91%")
    ap.add_argument("--force-export", action="store_true")
    ap.add_argument("--mode", choices=["merge", "lora"], default="merge",
                    help="lora: train tagged adapters instead of a merged "
                         "checkpoint (manager decides update-vs-new)")
    ap.add_argument("--lora-root", default="forge/evolution/loras")
    ap.add_argument("--lora-min-new", type=int, default=10,
                    help="min verified rows before a new tag is justified")
    ap.add_argument("--lora-fresh", default="",
                    help="comma-separated tags to retrain from scratch")
    ap.add_argument("--lora-r", type=int, default=32)
    # alpha=rank -> LoRA scale 1.0. The old 64/32 (scale 2.0) trained a
    # 'default' adapter that drove the 1.7B proposer to 0% parseable
    # output (babble) once attached.
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--min-rows", type=int, default=32,
                    help="skip registering an adapter trained on fewer "
                         "rows than this — a 17-row adapter was confirmed "
                         "to corrupt the proposer")
    ap.add_argument("--lora-steps", type=int, default=120,
                    help="steps per LoRA group (adapters train fast)")
    ap.add_argument("--lora-lr", type=float, default=1e-4,
                    help="LoRA-mode LR (default 1e-4 — adapter-scale; the "
                         "merge-mode --lr 2e-5 was far too small for "
                         "120-step hone runs)")
    ap.add_argument("--lora-val-every", type=int, default=15,
                    help="held-out val check every N LoRA steps "
                         "(0 disables)")
    ap.add_argument("--lora-val-size", type=float, default=0.15,
                    help="fraction of the tag's rows held out for the "
                         "val/early-stop check")
    ap.add_argument("--lora-early-stop", type=int, default=2,
                    help="stop a tag's run after N non-improving val "
                         "checks (0 disables) — the overfitting brake")
    ap.add_argument("--no-train-router", dest="train_router",
                    action="store_false",
                    help="skip decision-model router training after "
                         "adapter registration")
    ap.set_defaults(train_router=True)
    ap.add_argument("--proposer-quantize", default=None,
                    choices=["auto", "int4", "fp8", "int8"],
                    help="quantize for router-fit engine + deployments "
                         "matching evolve_all --proposer-quantize")
    args = ap.parse_args()

    from forge.evolution.code_evolve import export_sft
    if not Path(args.data).exists() or args.force_export:
        n = export_sft(args.db, args.data)
        print(f"[sft] exported {n} examples -> {args.data}")
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if r.get("response") and r.get("prompt")]
    if args.min_delta:
        rows = [r for r in rows if r.get("delta", 0) > args.min_delta]
    own = [{"prompt": r["prompt"], "response": r["response"],
            "target": r.get("target") or "misc"}
           for r in rows]
    print(f"[sft] {len(own)} usable verified rows "
          f"across {sorted({r['target'] for r in rows})}")
    if not own and not args.pie:
        print("[sft] no data — evolve more or pass --pie")
        return 1

    # ── LoRA mode: manager decides update-vs-new per tag, adapters only ──
    if args.mode == "lora":
        return _lora_mode(own, args)

    if args.base is None:
        args.base = _default_proposer_base()
    from forge.exp.harness import snap_dir
    base_dir = Path(args.base)
    if not base_dir.exists():
        base_dir = Path(snap_dir(args.base))

    out_dir = Path(args.out)
    ckpt = out_dir.with_suffix(".safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── stage 1: PIE prior (+ our rows mixed in for format anchoring) ──
    if args.pie:
        np_ = import_pie(args.pie_dataset, args.pie, args.pie_data)
        if np_ == 0:
            print("[pie] import produced nothing — skipping stage 1")
        else:
            s1 = str(Path(args.pie_data).with_suffix(".mix.jsonl"))
            with open(s1, "w", encoding="utf-8") as f:
                for l in open(args.pie_data, encoding="utf-8"):
                    f.write(l)
                for r in own:
                    f.write(json.dumps(r) + "\n")
            rc = _run_sft(s1, args.base, ckpt, args,
                          lr=args.pie_lr, max_steps=args.pie_steps)
            if rc != 0 or not ckpt.exists():
                print(f"[sft] stage-1 failed rc={rc}")
                return 1
            _assemble_hf_dir(ckpt, base_dir, out_dir)
            args.base = str(out_dir)   # stage 2 builds on stage 1
            base_dir = out_dir
            print(f"[sft] stage-1 done -> {out_dir}")

    # ── stage 2: our verified rows only (domain specialization) ──
    if own:
        own_file = str(Path(args.data).with_suffix(".own.jsonl"))
        with open(own_file, "w", encoding="utf-8") as f:
            for r in own:
                f.write(json.dumps(r) + "\n")
        rc = _run_sft(own_file, args.base, ckpt, args,
                      lr=args.lr, max_steps=args.max_steps)
        if rc != 0 or not ckpt.exists():
            print(f"[sft] stage-2 failed rc={rc}")
            return 1
    elif not args.pie:
        return 1

    _assemble_hf_dir(ckpt, base_dir, out_dir)
    print(f"[sft] proposer checkpoint -> {out_dir}")
    print(f"[sft] use: fn_evolve.py --proposer mix --model {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

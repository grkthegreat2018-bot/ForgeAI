"""Run external benchmarks against a Forge checkpoint (AUDIT A1).

Produces the baseline/regression-gate report for ForgeLM V2:

    .\\venv\\Scripts\\python.exe scripts\\download_eval_data.py   # once
    .\\venv\\Scripts\\python.exe scripts\\run_external_eval.py ^
        --tasks arc_easy arc_challenge hellaswag mmlu gsm8k ^
        --limit 200 --out results/external_eval_v2.json

Gate a candidate checkpoint against the stored baseline:

    --compare results/external_eval_v2.json --tol 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from forge.config import ModelConfig  # noqa: E402
from forge.evaluation.external_eval import (  # noqa: E402
    ALL_TASKS, compare_reports, run_eval, save_report)
from forge.model_loader import ConfigurableResearchLLM  # noqa: E402

DEFAULT_CKPT = REPO / "research" / "checkpoints" / "ForgeLM_V2.safetensors"
DEFAULT_CFG = REPO / "research" / "checkpoints" / "ForgeLM_V2_config.json"
DEFAULT_TOK = REPO / "research" / "checkpoints" / "forgelm_v2_tokenizer"


def load_model(ckpt: Path, cfg_path: Path, device: str):
    from forge.checkpoint_io import load_safetensors_pipelined
    cfg_dict = json.loads(cfg_path.read_text())
    kw = dict(
        vocab_size=cfg_dict["vocab_size"], d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"], n_heads=cfg_dict["n_heads"],
        n_kv_heads=cfg_dict["n_kv_heads"],
        intermediate_size=cfg_dict["intermediate_size"],
        max_seq_len=2048, layer_types=cfg_dict["layer_types"],
        mamba_d_state=cfg_dict["mamba_d_state"],
        mamba_d_conv=cfg_dict["mamba_d_conv"],
        mamba_expand=cfg_dict["mamba_expand"],
        mamba_dt_rank=cfg_dict["mamba_dt_rank"],
        mamba_bias=cfg_dict["mamba_bias"],
        mamba_conv_bias=cfg_dict["mamba_conv_bias"],
        norm_eps=cfg_dict["norm_eps"],
        tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        use_final_norm=cfg_dict["use_final_norm"],
        use_embed_norm=cfg_dict["use_embed_norm"])
    dev = torch.device(device)
    # Meta-device construction + assign-load: params materialize straight in
    # bf16 on-device from the pipelined (12.9GB/s) GPU state — no fp32 CPU
    # init (~12.8GB RNG writes), no CPU staging copies, no double VRAM peak.
    # Fall back to the classic CPU path if any init is meta-unsafe.
    try:
        with torch.device("meta"):
            model = ConfigurableResearchLLM(ModelConfig(**kw))
        state = load_safetensors_pipelined(str(ckpt), dev)
        missing, unexpected = model.load_state_dict(
            state, strict=False, assign=True)
        del state
        model = model.to(torch.bfloat16)
        mode = "meta+bf16"
    except Exception as e:
        print(f"  meta-device init failed ({e}); CPU load path")
        model = ConfigurableResearchLLM(ModelConfig(**kw))
        model = model.to(torch.bfloat16)
        state = load_safetensors_pipelined(str(ckpt), dev)
        missing, unexpected = model.load_state_dict(state, strict=False)
        del state
        torch.cuda.empty_cache()
        mode = "cpu+bf16"
    if missing or unexpected:
        print(f"  load: {len(missing)} missing / {len(unexpected)} "
              f"unexpected keys")
    model = model.eval()
    if next(model.parameters()).device != dev:
        model = model.to(dev)
    print(f"  load mode: {mode} "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B "
          f"{next(model.parameters()).dtype})")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOK))
    ap.add_argument("--tasks", nargs="*", default=list(ALL_TASKS))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gsm8k-limit", type=int, default=None,
                    help="separate cap for gsm8k (greedy gen is slow)")
    ap.add_argument("--out", default="results/external_eval_v2.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--compare", default=None,
                    help="baseline report json to gate against")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--gsm8k-max-new", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="context cap per scored forward — the eager SSM "
                    "scan's activation grows with T; 2048 keeps a 3B "
                    "bf16 eval inside 12GB (A1 OOM fix)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"loading model: {args.ckpt}")
    model = load_model(Path(args.ckpt), Path(args.config), args.device)

    limit: int | dict = args.limit
    if args.gsm8k_limit is not None:
        limit = {t: args.limit for t in args.tasks}
        limit["gsm8k"] = args.gsm8k_limit
    report = run_eval(model, tok, args.device, tasks=args.tasks,
                      limit=limit, gsm8k_max_new=args.gsm8k_max_new,
                      max_seq_len=args.max_seq,
                      partial_out=str(Path(args.out).with_suffix(
                          ".partial.json")))
    out = save_report(report, Path(args.out),
                      model_tag=Path(args.ckpt).stem,
                      meta={"limit": args.limit, "device": args.device})
    print(f"\n== headline ==")
    for t, v in report["headline"].items():
        print(f"  {t:16s} {v:.4f}")
    print(f"report -> {out}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        gate = compare_reports(base, report, tol=args.tol)
        print(f"gate vs {args.compare}: {gate}")
        return 0 if gate["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

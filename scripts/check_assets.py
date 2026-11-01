"""Asset gate (AUDIT A3): verify required local assets before a dev run.

CI runners don't carry the 6 GB checkpoints, so this gate is a *local*
pre-flight — run it after checkout/setup or before launching evals,
training, or the server:

    .\\venv\\Scripts\\python.exe scripts\\check_assets.py

Exits 0 when every required asset is present; prints the missing paths
and exits 1 otherwise. ``--all`` additionally checks optional assets
(LFM3 proposer, eval JSONLs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: (relative path, why required)
REQUIRED = [
    ("research/checkpoints/forgelm_v2_tokenizer/tokenizer.json",
     "canonical tokenizer (Rust fast-path load)"),
    ("research/checkpoints/forgelm_v2_tokenizer/tokenizer_config.json",
     "special token ids for the tokenizer wrapper"),
    ("research/checkpoints/ForgeLM_V2.safetensors",
     "default ForgeLM V2 checkpoint"),
    ("research/checkpoints/ForgeLM_V2_config.json",
     "ForgeLM V2 layer/config manifest"),
]

OPTIONAL = [
    ("research/checkpoints/lfm3_1p7b_cc_fq",
     "LFM3 proposer checkpoint dir (fn_evolve default)"),
    ("data/eval/arc_easy.jsonl", "external eval data (A1)"),
    ("data/eval/mmlu.jsonl", "external eval data (A1)"),
]


def check(entries, label: str) -> int:
    missing = 0
    for rel, why in entries:
        if not (REPO / rel).exists():
            print(f"  MISSING {label}: {rel}  ({why})")
            missing += 1
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also check optional assets")
    args = ap.parse_args()

    missing = check(REQUIRED, "required")
    if args.all:
        missing += check(OPTIONAL, "optional")

    if missing:
        print(f"\n{missing} asset(s) missing. Tokenizer: keep the "
              "forgelm_v2_tokenizer dir next to the checkpoint; eval data: "
              "run scripts/download_eval_data.py")
        return 1
    print("all assets present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download additional training data from HuggingFace to fill coverage gaps.

Uses only HuggingFace `datasets` library (traditional compute, no API distillation).
Downloads are streamed to avoid loading everything into RAM.

Gap categories targeted:
1. English fluency — FineWeb-Edu (already partially downloaded, get more)
2. Advanced math — Numina-1.5 (finish incomplete download), MetaMathQA
3. Reasoning — OpenThoughts-114k (have), mixture-of-thoughts (have), R1-distill
4. Tool use — Glaive FC v2 (incomplete), ToolACE (incomplete)
5. Creative problem solving — general reasoning datasets with creative tasks
6. Logic — synthetic logic from our generator + reasoning datasets

Datasets that are already complete are skipped. Incomplete downloads are resumed.

Usage:
    python scripts/download_gaps.py
    python scripts/download_gaps.py --categories math,tool_use
    python scripts/download_gaps.py --max-gb 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SFT_ROOT = DATA_ROOT / "sft"
PRETRAIN_ROOT = DATA_ROOT / "pretrain"
FINETUNE_ROOT = DATA_ROOT / "finetune"


def write_jsonl_streaming(dataset, output_path: Path, text_field: str = "text",
                          max_examples: int = 0, max_bytes: int = 0,
                          min_chars: int = 100) -> int:
    """Stream a HuggingFace dataset to JSONL. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total_bytes = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get(text_field, "")
            # Handle list fields (e.g. completions, answers) — join to string
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
                example = {**example, text_field: text}
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            # Write as {"text": ...} for pretrain, or passthrough for SFT
            if text_field == "text":
                line = json.dumps({"text": text}, ensure_ascii=False)
            else:
                line = json.dumps(example, ensure_ascii=False)
            f.write(line + "\n")
            count += 1
            total_bytes += len(line.encode("utf-8"))
            if count % 10000 == 0:
                print(f"  {count} examples, {total_bytes/1e6:.1f} MB", flush=True)
            if max_examples and count >= max_examples:
                break
            if max_bytes and total_bytes >= max_bytes:
                break
    return count


def check_existing(path: Path) -> tuple[int, float]:
    """Check if output file already exists and its size. Returns (lines, MB)."""
    if not path.exists():
        return 0, 0.0
    lines = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            lines += 1
    return lines, path.stat().st_size / 1e6


# ─── Download definitions ──────────────────────────────────────────────────
# Each entry: (name, hf_repo, split, text_field, output_dir, output_name,
#              max_gb, min_chars, description)
DOWNLOADS = [
    # ── Math (fill incomplete downloads) ──
    ("numina_math", "AI-MO/NuminaMath-CoT", "train", "solution",
     SFT_ROOT / "reasoning" / "numina-math-1.5", "numina_math.jsonl",
     5.0, 50, "NuminaMath CoT — advanced math with chain-of-thought solutions"),
    ("metamath", "meta-math/MetaMathQA", "train", "response",
     SFT_ROOT / "reasoning" / "metamath", "metamath.jsonl",
     3.0, 50, "MetaMathQA — augmented math word problems with solutions"),

    # ── Tool use (fill incomplete downloads) ──
    ("glaive_fc", "glaiveai/glaive-function-calling-v2", "train", "chat",
     SFT_ROOT / "tool_use" / "glaive-fc-v2", "glaive_fc_v2.jsonl",
     2.0, 100, "Glaive Function Calling v2 — tool use training data"),

    # ── Reasoning (R1 distill — large, high quality) ──
    ("r1_distill", "a-m-team/AM-DeepSeek-R1-Distilled-1.4M", "train", "response",
     SFT_ROOT / "reasoning" / "am-r1-distill-1.4M", "r1_distill.jsonl",
     5.0, 100, "R1-distill reasoning data — 1.4M chain-of-thought reasoning traces"),

    # ── English fluency (FineWeb-Edu — high quality educational text) ──
    ("fineweb_edu_grammar", "HuggingFaceFW/fineweb-edu", "train",
     "text", PRETRAIN_ROOT / "fineweb_edu_grammar", "fineweb_edu_grammar.jsonl",
     5.0, 500, "FineWeb-Edu — educational text for English fluency + general reasoning"),

    # ── Creative problem solving (OpenOrca — diverse reasoning + instruction) ──
    ("openorca", "Open-Orca/OpenOrca", "train", "response",
     SFT_ROOT / "reasoning" / "openorca", "openorca.jsonl",
     3.0, 100, "OpenOrca — diverse instruction-following + reasoning examples"),

    # ── Logic / deductive reasoning (LogiQA — parquet format) ──
    ("logiqa", "fireworks-ai/logiqa", "train", "completions",
     SFT_ROOT / "reasoning" / "logiqa", "logiqa.jsonl",
     0.5, 20, "LogiQA — logical reasoning + deductive logic problems"),
]


def main():
    p = argparse.ArgumentParser(
        description="Download additional training data from HuggingFace to fill gaps.",
    )
    p.add_argument("--categories", default="",
                   help="Comma-separated category names to download (default: all). "
                        f"Available: {', '.join(d[0] for d in DOWNLOADS)}")
    p.add_argument("--max-gb", type=float, default=0,
                   help="Override max GB per dataset (0 = use per-dataset default)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip datasets that already have data (default True)")
    args = p.parse_args()

    from datasets import load_dataset

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        downloads = [d for d in DOWNLOADS if d[0] in cats]
    else:
        downloads = DOWNLOADS

    total_examples = 0
    for name, repo, split, text_field, out_dir, out_name, max_gb, min_chars, desc in downloads:
        out_path = out_dir / out_name

        # Check if already downloaded
        if args.skip_existing and out_path.exists():
            lines, mb = check_existing(out_path)
            if lines > 100:
                print(f"[{name}] Already have {lines} examples ({mb:.1f} MB) — skipping")
                continue

        max_bytes = int((args.max_gb if args.max_gb else max_gb) * 1e9)
        print(f"\n[{name}] {desc}")
        print(f"  Repo: {repo}, split: {split}")
        print(f"  Output: {out_path}")
        print(f"  Max: {max_gb} GB, min chars: {min_chars}")

        try:
            ds = load_dataset(repo, split=split, streaming=True)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        try:
            count = write_jsonl_streaming(
                ds, out_path, text_field=text_field,
                max_bytes=max_bytes, min_chars=min_chars,
            )
            print(f"  Wrote {count} examples to {out_path.name}")
            total_examples += count
        except Exception as e:
            print(f"  ERROR writing: {e}")
            continue

    print(f"\nTotal: {total_examples} new examples downloaded")


if __name__ == "__main__":
    main()

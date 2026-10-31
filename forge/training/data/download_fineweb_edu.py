"""Download FineWeb-Edu, tokenize with LFM2.5 tokenizer, and pack for V7 training.

FineWeb-Edu: high-quality educational web text from HuggingFace.
  - Sample 10 shards (default) = ~150M tokens
  - Tokenized with our 65536-vocab LFM2.5 tokenizer
  - Packed as int32 train.bin / val.bin (same format as v7_train)

Usage:
    python -m forge.training.data.download_fineweb_edu --shards 10 --seq-len 2048
    python -m forge.training.data.download_fineweb_edu --shards 50 --seq-len 2048
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = ROOT / "research" / "checkpoints" / "lfm25_tokenizer"
OUT_DIR = ROOT / "research" / "data" / "fineweb_edu"


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Download + tokenize FineWeb-Edu")
    parser.add_argument("--shards", type=int, default=10,
                        help="number of 1M-row shards to download (each ~15M tokens)")
    parser.add_argument("--target-tokens-m", type=int, default=500,
                        help="stop after this many millions of tokens (0 = use all shards)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.02,
                        help="fraction of data for validation")
    parser.add_argument("--source", choices=("fineweb-edu", "fineweb", "openwebtext"),
                        default="fineweb-edu",
                        help="which HuggingFace dataset to pull from")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    log(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    log(f"Tokenizer: {type(tok).__name__}, vocab={tok.vocab_size}")

    # ── Dataset config ──
    dataset_configs = {
        "fineweb-edu": {
            "repo": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",  # 10BT sample = 10 shards of ~1M rows each
            "text_col": "text",
        },
        "fineweb": {
            "repo": "HuggingFaceFW/fineweb",
            "config": "sample-10BT",
            "text_col": "text",
        },
        "openwebtext": {
            "repo": "Skylion007/openwebtext",
            "config": None,
            "text_col": "text",
        },
    }
    cfg = dataset_configs[args.source]

    # ── Download + tokenize ──
    from datasets import load_dataset

    log(f"Loading {args.source} ({cfg['config'] or 'default'}) — {args.shards} shards...")
    t0 = time.perf_counter()

    if cfg["config"]:
        ds = load_dataset(cfg["repo"], cfg["config"], split="train",
                          streaming=True)
    else:
        ds = load_dataset(cfg["repo"], split="train",
                          streaming=True)

    all_tokens: list[int] = []
    n_docs = 0
    target_docs = args.shards * 1_000_000  # ~1M docs per shard
    target_tokens = args.target_tokens_m * 1_000_000 if args.target_tokens_m > 0 else 0

    log(f"Tokenizing (target: {target_tokens/1e6:.0f}M tokens or {target_docs:,} docs)...")
    for doc in ds:
        text = doc[cfg["text_col"]]
        if not text or len(text.strip()) < 50:
            continue
        # Tokenize without special tokens — just raw text tokens
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 10:
            continue
        all_tokens.extend(ids)
        n_docs += 1
        if n_docs % 50_000 == 0:
            elapsed = time.perf_counter() - t0
            log(f"  {n_docs:,} docs | {len(all_tokens)/1e6:.1f}M tokens | "
                f"{elapsed:.0f}s | {len(all_tokens)/elapsed/1e3:.0f}K tok/s")
        if target_tokens and len(all_tokens) >= target_tokens:
            log(f"  Reached target of {target_tokens/1e6:.0f}M tokens, stopping.")
            break
        if n_docs >= target_docs:
            break

    elapsed = time.perf_counter() - t0
    total_tokens = len(all_tokens)
    log(f"Done: {n_docs:,} docs, {total_tokens/1e6:.1f}M tokens in {elapsed:.0f}s")

    # ── Pack into sequences ──
    seq_len = args.seq_len
    n_tokens = len(all_tokens)
    n_seqs = n_tokens // seq_len
    log(f"Packing into {n_seqs:,} sequences of {seq_len} tokens...")

    arr = np.array(all_tokens[:n_seqs * seq_len], dtype=np.int32)
    packed = arr.reshape(n_seqs, seq_len)

    # ── Train/val split ──
    n_val = max(1, int(n_seqs * args.val_frac))
    n_train = n_seqs - n_val
    log(f"Split: {n_train:,} train seqs, {n_val:,} val seqs")

    # Shuffle sequences (deterministic)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_seqs)
    packed = packed[perm]
    train = packed[:n_train]
    val = packed[n_train:]

    # ── Save ──
    train_path = OUT_DIR / "train.bin"
    val_path = OUT_DIR / "val.bin"
    train.tofile(str(train_path))
    val.tofile(str(val_path))

    train_gb = train.nbytes / 1e9
    log(f"Saved: {train_path.name} ({train.shape}, {train_gb:.2f} GB)")
    log(f"Saved: {val_path.name} ({val.shape}, {val.nbytes/1e6:.1f} MB)")
    log(f"Total tokens: {(n_train + n_val) * seq_len / 1e6:.1f}M")
    log(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

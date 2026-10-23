#!/usr/bin/env python
"""Download a curated subset of English Wikipedia for pretraining.

Streams wikimedia/wikipedia (2023-11-01.en), filters out low-quality
articles (disambiguation, list-only, <500 chars), and writes JSONL
shards of ~500 MB each until ~5 GB total is reached.

Usage:
    set PYTHONPATH=<repo-root>
    python scripts/download_wikipedia.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
# Repo-root anchored (EC-091 — was a machine-specific absolute path).
OUTPUT_DIR = (Path(__file__).resolve().parents[1]
              / "research" / "data" / "pretrain" / "wikipedia")
FILE_PREFIX = "wikipedia"
MAX_FILE_BYTES = 500 * 1024 * 1024          # 500 MB per shard
TARGET_TOTAL_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB total
MIN_CHARS = 500
PROGRESS_INTERVAL = 10_000

# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

# Patterns that indicate a disambiguation or list-only page.
_DISAMBIG_PATTERNS = [
    re.compile(r"\{\{\s*disambiguation", re.IGNORECASE),
    re.compile(r"\{\{\s*disambig", re.IGNORECASE),
    re.compile(r"\{\{\s dab ", re.IGNORECASE),
    re.compile(r"\{\{\s*geodis", re.IGNORECASE),
    re.compile(r"\{\{\s*hndis", re.IGNORECASE),
    re.compile(r"\{\{\s*schooldis", re.IGNORECASE),
    re.compile(r"\{\{\s*roadis", re.IGNORECASE),
    re.compile(r"\{\{\s*numberdis", re.IGNORECASE),
    re.compile(r"\{\{\s*call sign disambiguation", re.IGNORECASE),
    re.compile(r"\{\{\s*lakeindex", re.IGNORECASE),
    re.compile(r"\{\{\s*mountainindex", re.IGNORECASE),
    re.compile(r"\{\{\s*riverdisambig", re.IGNORECASE),
    re.compile(r"\{\{\s*shipindex", re.IGNORECASE),
    re.compile(r"\{\{\s*surnameindex", re.IGNORECASE),
    re.compile(r"\{\{\s*given name index", re.IGNORECASE),
]

_LIST_PATTERNS = [
    re.compile(r"\{\{\s*list of", re.IGNORECASE),
    re.compile(r"\{\{\s*lists of", re.IGNORECASE),
    re.compile(r"\{\{\s*portal:lists", re.IGNORECASE),
    re.compile(r"\{\{\s*wp:list", re.IGNORECASE),
    re.compile(r"\{\{\s*standalone list", re.IGNORECASE),
    re.compile(r"\{\{\s*dynamic list", re.IGNORECASE),
    re.compile(r"\{\{\s*sia", re.IGNORECASE),  # set index articles
]


def is_disambiguation(text: str) -> bool:
    """Return True if the article looks like a disambiguation page."""
    low = text[:4000]  # only check the start for speed
    return any(p.search(low) for p in _DISAMBIG_PATTERNS)


def is_list_only(text: str) -> bool:
    """Return True if the article looks like a list-only / set-index page."""
    low = text[:4000]
    return any(p.search(low) for p in _LIST_PATTERNS)


def is_list_title(title: str) -> bool:
    """Quick title-level check for 'List of …' style articles."""
    t = title.strip().lower()
    return t.startswith("list of") or t.startswith("lists of") or t.startswith("index of")


def should_skip(title: str, text: str) -> bool:
    """Decide whether to skip an article based on title + text."""
    if len(text) < MIN_CHARS:
        return True
    if is_list_title(title):
        return True
    if is_disambiguation(text):
        return True
    if is_list_only(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Main streaming + writing loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Late import so the script can be --help'd without the venv active
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Activate the ForgeAI venv first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset {DATASET_NAME} / {DATASET_CONFIG} (streaming=True) …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    print("Stream opened. Beginning filtering + write loop.\n")

    total_articles = 0
    written_articles = 0
    skipped_articles = 0
    total_bytes = 0
    file_index = 0
    current_file_bytes = 0
    current_fh = None
    start_time = time.time()

    def open_new_shard(idx: int):
        path = OUTPUT_DIR / f"{FILE_PREFIX}_{idx:02d}.jsonl"
        print(f"  >> Opening shard {path.name}")
        return open(path, "w", encoding="utf-8", buffering=1024 * 1024)

    current_fh = open_new_shard(file_index)

    try:
        for example in ds:
            total_articles += 1

            title = example.get("title", "")
            text = example.get("text", "")

            if should_skip(title, text):
                skipped_articles += 1
                if total_articles % PROGRESS_INTERVAL == 0:
                    _print_progress(total_articles, written_articles, skipped_articles,
                                    total_bytes, start_time)
                continue

            record = {"text": text, "title": title}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            # Check whether adding this line would exceed the shard size
            if current_file_bytes + line_bytes > MAX_FILE_BYTES and current_file_bytes > 0:
                current_fh.close()
                file_index += 1
                current_fh = open_new_shard(file_index)
                current_file_bytes = 0

            current_fh.write(line)
            current_file_bytes += line_bytes
            total_bytes += line_bytes
            written_articles += 1

            if total_articles % PROGRESS_INTERVAL == 0:
                _print_progress(total_articles, written_articles, skipped_articles,
                                total_bytes, start_time)

            # Stop at target
            if total_bytes >= TARGET_TOTAL_BYTES:
                print("\n*** Reached ~5 GB target — stopping. ***")
                break

    except KeyboardInterrupt:
        print("\n*** Interrupted by user — closing files. ***")
    finally:
        if current_fh:
            current_fh.close()

    _print_progress(total_articles, written_articles, skipped_articles,
                    total_bytes, start_time, final=True)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Total articles scanned : {total_articles:,}")
    print(f"  Articles written       : {written_articles:,}")
    print(f"  Articles skipped       : {skipped_articles:,}")
    print(f"  Total size             : {_human_bytes(total_bytes)}")
    print(f"  Shard files            : {file_index + 1}")
    print(f"  Output dir             : {OUTPUT_DIR}")
    print(f"  Elapsed                : {_human_duration(time.time() - start_time)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _print_progress(total, written, skipped, total_bytes, start_time, final=False):
    elapsed = time.time() - start_time
    rate = written / elapsed if elapsed > 0 else 0
    pct = (total_bytes / TARGET_TOTAL_BYTES) * 100
    tag = "FINAL" if final else "PROGRESS"
    print(
        f"[{tag}] scanned={total:,}  written={written:,}  "
        f"skipped={skipped:,}  size={_human_bytes(total_bytes)}  "
        f"({pct:.1f}% of 5GB)  rate={rate:.0f} art/s  "
        f"elapsed={_human_duration(elapsed)}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download OpenWebMath dataset and save as JSONL files.

- Dataset: open-web-math/open-web-math (streaming)
- Format: {"text": text_content}
- Split into 500MB files: openwebmath_00.jsonl, etc.
- Filter: skip pages < 500 chars
- Target: ~5GB total
"""

import json
import os
import time

# Repo-root anchored (EC-078 — previously wrote into scripts/).
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "data", "pretrain", "openwebmath")
FILE_PREFIX = "openwebmath"
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500MB per file
TARGET_TOTAL_BYTES = 5 * 1024 * 1024 * 1024  # 5GB total
MIN_CHARS = 500
PROGRESS_INTERVAL = 10000


def main():
    from datasets import load_dataset

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[OpenWebMath] Output dir: {OUTPUT_DIR}")
    print(f"[OpenWebMath] Target: {TARGET_TOTAL_BYTES / 1e9:.1f} GB, "
          f"max file: {MAX_FILE_BYTES / 1e6:.0f} MB, min chars: {MIN_CHARS}")

    ds = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True,
    )

    file_idx = 0
    total_written = 0
    total_skipped = 0
    total_bytes = 0
    current_file_bytes = 0
    current_fh = None
    start_time = time.time()

    def open_new_file():
        nonlocal current_fh, current_file_bytes, file_idx
        if current_fh is not None:
            current_fh.close()
            print(f"[OpenWebMath] Closed {FILE_PREFIX}_{file_idx - 1:02d}.jsonl "
                  f"({current_file_bytes / 1e6:.1f} MB)")
        fname = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_{file_idx:02d}.jsonl")
        current_fh = open(fname, "w", encoding="utf-8")
        current_file_bytes = 0
        file_idx += 1
        print(f"[OpenWebMath] Opened {fname}")

    open_new_file()

    for item in ds:
        text = item.get("text", "")
        if len(text) < MIN_CHARS:
            total_skipped += 1
            continue

        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        # Check if we need a new file
        if current_file_bytes + line_bytes > MAX_FILE_BYTES:
            open_new_file()

        current_fh.write(line)
        current_file_bytes += line_bytes
        total_bytes += line_bytes
        total_written += 1

        if total_written % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = total_written / elapsed if elapsed > 0 else 0
            print(f"[OpenWebMath] Progress: {total_written:,} written, "
                  f"{total_skipped:,} skipped, "
                  f"{total_bytes / 1e9:.2f} GB, "
                  f"{rate:.0f} items/s, "
                  f"file #{file_idx}")

        if total_bytes >= TARGET_TOTAL_BYTES:
            print(f"[OpenWebMath] Reached target {TARGET_TOTAL_BYTES / 1e9:.1f} GB")
            break

    if current_fh is not None:
        current_fh.close()
        print(f"[OpenWebMath] Closed {FILE_PREFIX}_{file_idx - 1:02d}.jsonl "
              f"({current_file_bytes / 1e6:.1f} MB)")

    elapsed = time.time() - start_time
    print("\n[OpenWebMath] DONE")
    print(f"  Total articles written: {total_written:,}")
    print(f"  Total articles skipped: {total_skipped:,}")
    print(f"  Total size: {total_bytes / 1e9:.2f} GB ({total_bytes:,} bytes)")
    print(f"  Number of files: {file_idx}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

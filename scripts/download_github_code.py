"""Download github code from codeparrot (ungated, streaming).
Targets ~5GB of Python/JS/TS/Rust/Go/C++/Java code.
"""
import json
import os
import time
from pathlib import Path

# Repo-root anchored output (EC-078 — previously wrote into scripts/).
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "research" / "data" / "pretrain" / "github_code"
TARGET_BYTES = 5 * 1024**3
SHARD_SIZE = 500 * 1024**2
LANGUAGES = {"python", "javascript", "typescript", "rust", "go", "c++", "c", "java"}
MAX_FILE_BYTES = 50_000
MIN_FILE_CHARS = 100

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")

    # Load HF token from repo-root .env (parents[1] — this file is in scripts/)
    env_path = Path(__file__).resolve().parents[1] / ".env"
    token = None
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip()
                    break

    import datasets
    print("[github_code] Loading codeparrot/github-code (streaming=True) ...", flush=True)
    ds = datasets.load_dataset("codeparrot/github-code", split="train",
                               streaming=True, token=token, trust_remote_code=True)

    shard_idx = 0
    shard_path = OUTPUT_DIR / f"github_code_{shard_idx:02d}.jsonl"
    shard_fh = open(shard_path, "w", encoding="utf-8")
    shard_bytes = 0
    total_bytes = 0
    total_files = 0
    skipped = 0
    lang_counts = {}
    t0 = time.time()

    for ex in ds:
        lang = ex.get("language", "").lower()
        if lang not in LANGUAGES:
            skipped += 1
            continue
        content = ex.get("code", ex.get("content", ""))
        if not content or len(content) < MIN_FILE_CHARS or len(content.encode("utf-8")) > MAX_FILE_BYTES:
            skipped += 1
            continue
        # Skip generated/minified
        first_line = content.split("\n", 1)[0].lower()[:200]
        if any(m in first_line for m in ["generated", "auto-generated", "do not edit", "minified", "webpack"]):
            skipped += 1
            continue

        obj = {"text": content, "language": lang, "name": ex.get("name", ex.get("path", ""))}
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        shard_fh.write(line)
        shard_bytes += line_bytes
        total_bytes += line_bytes
        total_files += 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if shard_bytes >= SHARD_SIZE:
            shard_fh.close()
            shard_idx += 1
            shard_path = OUTPUT_DIR / f"github_code_{shard_idx:02d}.jsonl"
            shard_fh = open(shard_path, "w", encoding="utf-8")
            shard_bytes = 0

        if total_files % 10000 == 0:
            elapsed = time.time() - t0
            pct = total_bytes / TARGET_BYTES * 100
            rate = total_files / elapsed if elapsed > 0 else 0
            print(f"  [{total_files:>7d} files] {total_bytes/1e9:.2f} GB ({pct:.1f}%) "
                  f"| skipped {skipped} | {rate:.0f} files/s | {elapsed:.0f}s", flush=True)

        if total_bytes >= TARGET_BYTES:
            print(f"\n  Target reached ({total_bytes/1e9:.2f} GB)")
            break

    shard_fh.close()
    if shard_bytes == 0 and shard_idx > 0:
        shard_path.unlink()

    elapsed = time.time() - t0
    print("\n=== GitHub Code Download Complete ===")
    print(f"  Files: {total_files}")
    print(f"  Total: {total_bytes/1e9:.2f} GB")
    print(f"  Skipped: {skipped}")
    print(f"  Shards: {shard_idx + 1}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Languages: {lang_counts}")
    print(f"  Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

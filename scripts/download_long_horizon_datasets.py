"""Download long-horizon agent training datasets from Hugging Face.

Pulls:
  1. Lite-Coder/LiteCoder-Terminal-SFT   (11,255 trajectories, ~1.2 GB)
  2. bunnybhaiya/agentgym-sft-trajectories (101,926 examples, ~459 MB)

Into: data\\sft\\<dataset_name>\\ under the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Repo-root anchored (EC-091 — was a machine-specific absolute path).
BASE = Path(__file__).resolve().parents[1] / "data" / "sft"

DATASETS = [
    {
        "repo_id": "Lite-Coder/LiteCoder-Terminal-SFT",
        "local_dir": BASE / "LiteCoder-Terminal-SFT",
    },
    {
        "repo_id": "bunnybhaiya/agentgym-sft-trajectories",
        "local_dir": BASE / "agentgym-sft-trajectories",
    },
]


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for d in DATASETS:
        repo_id = d["repo_id"]
        local_dir: Path = d["local_dir"]
        print(f"\n=== {repo_id} -> {local_dir} ===", flush=True)
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"  [skip] target already populated: {local_dir}", flush=True)
            continue
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                max_workers=4,
            )
            print(f"  [ok] downloaded to {local_dir}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {repo_id}: {e}", flush=True)
            failures.append(repo_id)
    if failures:
        print(f"\nFAILED: {failures}", flush=True)
        return 1
    print("\nAll datasets downloaded.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

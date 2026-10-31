"""Download and organize all training datasets for ForgeLM V4.

Organizes into:
  research/data/pretrain/     — continued pretraining corpora
  research/data/sft/coding/   — code instruction tuning
  research/data/sft/reasoning/ — math/logic/reasoning CoT
  research/data/sft/tool_use/  — function calling / tool use
  research/data/sft/agentic/   — multi-step agent trajectories

Usage:
    set HF_TOKEN=...
    python scripts/download_datasets.py [--category all|pretrain|coding|reasoning|tool_use|agentic]
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Fix Windows encoding
os.environ.setdefault("PYTHONUTF8", "1")

# Repo-root anchored: this file lives in scripts/, parents[1] is the repo
# root. Datasets land under research/data/ (EC-078 — previously wrote
# into scripts/ itself).
DATA_DIR = Path(__file__).resolve().parents[1] / "research" / "data"

# ─── Dataset registry ─────────────────────────────────────────────────────

DATASETS = {
    # ── Continued pretraining (large, download selectively) ──
    "pretrain": [
        {
            "name": "OpenCoder-LLM/opc-annealing-corpus",
            "local": "pretrain/annealing",
            "note": "24GB synthetic + algorithmic corpus for annealing",
            "repo_type": "dataset",
        },
        {
            "name": "OpenCoder-LLM/opc-fineweb-math-corpus",
            "local": "pretrain/fineweb-math",
            "note": "10GB math-related web text",
            "repo_type": "dataset",
        },
        # opc-fineweb-code-corpus is 148GB — skip for now, too large
        # RefineCode meta is just metadata, actual code fetched from GitHub
        # The Stack v2 — use streaming, too large to download
    ],

    # ── SFT: Coding instruction ──
    "coding": [
        {
            "name": "OpenCoder-LLM/opc-sft-stage2",
            "local": "sft/coding/opc-sft-stage2",
            "note": "375K examples: educational_instruct, evol_instruct, mceval_instruct, package_instruct",
            "repo_type": "dataset",
        },
        {
            "name": "ise-uiuc/Magicoder_oss_instruct_75k",
            "local": "sft/coding/magicoder-oss",
            "note": "75K OSS-Instruct synthetic code",
            "repo_type": "dataset",
        },
        {
            "name": "ise-uiuc/Magicoder_evol_instruct_110k",
            "local": "sft/coding/magicoder-evol",
            "note": "110K evolved code instructions",
            "repo_type": "dataset",
        },
        {
            "name": "theblackcat102/evol-codealpaca-v1",
            "local": "sft/coding/evol-codealpaca",
            "note": "Evolved CodeAlpaca, longer conversations",
            "repo_type": "dataset",
        },
        {
            "name": "OpenCoder-LLM/opc-sft-stage1",
            "local": "sft/coding/opc-sft-stage1",
            "note": "4.21M examples: realuser_instruct, filtered_infinity, largescale_diverse",
            "repo_type": "dataset",
            "large": True,
        },
    ],

    # ── SFT: Reasoning (math/logic/problem-solving) ──
    "reasoning": [
        {
            "name": "open-r1/Mixture-of-Thoughts",
            "local": "sft/reasoning/mixture-of-thoughts",
            "note": "350K verified R1-distilled traces (math+code+science)",
            "repo_type": "dataset",
        },
        {
            "name": "open-thoughts/OpenThoughts-114k",
            "local": "sft/reasoning/openthoughts-114k",
            "note": "114K R1 reasoning traces (math, science, code, puzzles)",
            "repo_type": "dataset",
        },
        {
            "name": "AI-MO/NuminaMath-1.5",
            "local": "sft/reasoning/numina-math-1.5",
            "note": "900K competition math problems with CoT",
            "repo_type": "dataset",
        },
        {
            "name": "meta-math/MetaMathQA",
            "local": "sft/reasoning/metamath",
            "note": "395K bootstrapped math QA from GSM8K+MATH",
            "repo_type": "dataset",
        },
        {
            "name": "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
            "local": "sft/reasoning/am-r1-distill-1.4M",
            "note": "1.4M R1-distilled reasoning traces, verified",
            "repo_type": "dataset",
            "large": True,
        },
    ],

    # ── SFT: Tool use / function calling ──
    "tool_use": [
        {
            "name": "Team-ACE/ToolACE",
            "local": "sft/tool_use/toolace",
            "note": "26,507 APIs, dual-layer verified, wins BFCL",
            "repo_type": "dataset",
        },
        {
            "name": "glaiveai/glaive-function-calling-v2",
            "local": "sft/tool_use/glaive-fc-v2",
            "note": "100K+ function calling, balanced no-call/single/multi",
            "repo_type": "dataset",
        },
        {
            "name": "interstellarninja/hermes_reasoning_tool_use",
            "local": "sft/tool_use/hermes-reasoning-tool",
            "note": "51K ShareGPT, teaches WHEN to call tools (relevance)",
            "repo_type": "dataset",
        },
        {
            "name": "nvidia/Nemotron-SFT-Agentic-v2",
            "local": "sft/tool_use/nemotron-agentic-v2",
            "note": "1.2M synthetic tool-use trajectories, LLM-judge filtered",
            "repo_type": "dataset",
            "large": True,
        },
    ],

    # ── SFT: Agentic trajectories ──
    "agentic": [
        {
            "name": "Petrouil/opencode-agentic-mini",
            "local": "sft/agentic/opencode-agentic-mini",
            "note": "19K sessions, 113K tool calls, real agentic coding",
            "repo_type": "dataset",
        },
        {
            "name": "zai-org/AgentInstruct",
            "local": "sft/agentic/agent-instruct",
            "note": "1,866 ReAct trajectories across 6 task types",
            "repo_type": "dataset",
        },
        {
            "name": "xlangai/AgentTrek",
            "local": "sft/agentic/agent-trek",
            "note": "52K web agent dialogue turns",
            "repo_type": "dataset",
        },
        {
            "name": "togethercomputer/CoderForge-Preview",
            "local": "sft/agentic/coderforge",
            "note": "51K test-verified coding agent trajectories",
            "repo_type": "dataset",
        },
        {
            "name": "nvidia/Open-SWE-Traces",
            "local": "sft/agentic/open-swe-traces",
            "note": "207K SWE-agent + OpenHands trajectories on real issues",
            "repo_type": "dataset",
            "large": True,
        },
    ],
}


def download_dataset(entry: dict, base_dir: Path) -> bool:
    """Download a single dataset using huggingface_hub snapshot_download."""
    from huggingface_hub import snapshot_download

    name = entry["name"]
    local = entry["local"]
    repo_type = entry.get("repo_type", "dataset")
    note = entry.get("note", "")
    is_large = entry.get("large", False)

    target = base_dir / local
    target.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (has files)
    # snapshot_download is resumable — it will skip existing files and only
    # download missing ones. So we always call it to ensure completeness.
    existing = list(target.rglob("*"))
    existing_files = [f for f in existing if f.is_file()]
    if existing_files:
        print(f"  [RESUME] {name} — has {len(existing_files)} files, checking for missing...")

    print(f"\n  [DOWNLOAD] {name}")
    print(f"    -> {local}")
    print(f"    Note: {note}")
    if is_large:
        print("    [LARGE DATASET — may take a while]")

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=name,
            repo_type=repo_type,
            local_dir=str(target),
            token=os.environ.get("HF_TOKEN"),
            max_workers=4,
        )
        elapsed = time.time() - t0
        # Calculate size
        total_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        size_gb = total_size / 1e9
        print(f"    Done in {elapsed:.0f}s — {size_gb:.2f} GB")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED in {elapsed:.0f}s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ForgeLM V4 training datasets")
    parser.add_argument("--category", type=str, default="all",
                        choices=["all", "pretrain", "coding", "reasoning", "tool_use", "agentic"],
                        help="Which category to download")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip datasets marked as large (>1M examples)")
    parser.add_argument("--only-large", action="store_true",
                        help="Only download large datasets")
    args = parser.parse_args()

    base_dir = DATA_DIR
    print(f"Base directory: {base_dir}")
    print(f"HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

    categories = list(DATASETS.keys()) if args.category == "all" else [args.category]

    total = 0
    success = 0
    failed = 0
    skipped = 0
    failed_list = []

    for cat in categories:
        print(f"\n{'='*60}")
        print(f"  CATEGORY: {cat.upper()} ({len(DATASETS[cat])} datasets)")
        print(f"{'='*60}")

        for entry in DATASETS[cat]:
            is_large = entry.get("large", False)
            if args.skip_large and is_large:
                print(f"\n  [SKIP-LARGE] {entry['name']}")
                skipped += 1
                continue
            if args.only_large and not is_large:
                continue

            total += 1
            ok = download_dataset(entry, base_dir)
            if ok:
                success += 1
            else:
                failed += 1
                failed_list.append(entry["name"])

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    if failed_list:
        print("  Failed datasets:")
        for name in failed_list:
            print(f"    - {name}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

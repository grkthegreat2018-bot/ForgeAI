"""Curriculum SFT: Mix Distillation + two-stage curriculum for small models.

Implements the SFT stage of the ForgeLM V10-Thinking recipe, augmented with
insights from the Small Model Learnability Gap research (arXiv 2502.12143):

  Small models (≤3B) do NOT benefit from long CoT distillation. They perform
  better on shorter, simpler reasoning chains that match their intrinsic
  learning capacity. Direct distillation from strong teachers with long CoT
  often HURTS small models.

Solutions implemented here:
  1. Mix Distillation (arXiv 2502.12143): blend long CoT (teacher) + short
     CoT (self/student or simpler teacher) at a target ratio (default 50/50).
  2. Curriculum Distillation (EMNLP 2025): two-stage training:
     - Stage 1 (Internal Solver): SFT on SHORT CoT + direct answers.
       Builds robust internal problem-solving ability.
     - Stage 2 (Externalize): SFT on longer CoT traces. Teaches the model
       to externalize its latent reasoning as explicit chains.
  3. Doom-loop filtering: n-gram repetition detector removes training
     examples with excessive repetition (the #1 failure mode of reasoning
     models, per Liquid AI's ForgeLM V10-Thinking report: 15.74% baseline).
  4. CoT length classification: automatically splits data into short/long
     CoT buckets based on solution length and reasoning marker presence.

## Data format

Input: JSONL files with {"prompt": ..., "solution"/"response": ...}
Output: Two filtered JSONL files (stage1_short.jsonl, stage2_long.jsonl)
        ready for sft_train.py.

## Usage

  # Step 1: Prepare curriculum data
  python -m research.training.runners.curriculum_sft prepare \\
      --input data/hf_datasets/gsm8k.jsonl \\
             data/hf_datasets/openr1_math.jsonl \\
             data/hf_datasets/openthoughts_114k.jsonl \\
      --output-dir research/data/curriculum \\
      --short-cot-max-tokens 150 \\
      --long-cot-min-tokens 300 \\
      --mix-ratio 0.5 \\
      --filter-doom-loops

  # Step 2: Run Stage 1 (short CoT — internal solver)
  python -m research.training.runners.curriculum_sft train-stage1 \\
      --data research/data/curriculum/stage1_short.jsonl \\
      --checkpoint research/checkpoints/forgelm_v2_CPT.safetensors \\
      --save research/checkpoints/forgelm_v2_SFT1.safetensors \\
      --lr 5e-5 --max-steps 1000 --optimizer cpu_offload

  # Step 3: Run Stage 2 (long CoT — externalize reasoning)
  python -m research.training.runners.curriculum_sft train-stage2 \\
      --data research/data/curriculum/stage2_long.jsonl \\
      --checkpoint research/checkpoints/forgelm_v2_SFT1.safetensors \\
      --save research/checkpoints/forgelm_v2_SFT2.safetensors \\
      --lr 2e-5 --max-steps 1500 --optimizer cpu_offload

  # Or run the full pipeline in one command:
  python -m research.training.runners.curriculum_sft full \\
      --input ... --output-dir ... --checkpoint ... --save ...
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Doom-loop detection ────────────────────────────────────────────────────

# Reasoning markers that indicate CoT traces
_COT_MARKERS = (
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
    "<think>",
    "</think>",
    "Let me think",
    "Let's think",
    "Step 1:",
    "Step 1.",
    "First,",
    "To solve this",
)


def ngram_repetition_ratio(text: str, n: int = 8) -> float:
    """Compute the n-gram repetition ratio of a text.

    A high ratio (>0.3) indicates doom-looping — the text repeats the same
    n-grams excessively. This is the #1 failure mode of reasoning models.

    Args:
        text: the text to check
        n: n-gram size (8 is a good default — catches phrase-level repetition)

    Returns:
        ratio in [0, 1]: fraction of n-grams that are repeated.
        0.0 = no repetition, 1.0 = all n-grams repeated.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def is_doom_loop(text: str, n: int = 8, threshold: float = 0.3,
                 min_length: int = 50) -> bool:
    """Check if a text is a doom-loop (excessive n-gram repetition).

    Args:
        text: the text to check
        n: n-gram size
        threshold: repetition ratio above this = doom loop
        min_length: don't check texts shorter than this (too short to loop)

    Returns:
        True if the text is a doom-loop, False otherwise.
    """
    if len(text.split()) < min_length:
        return False
    return ngram_repetition_ratio(text, n) > threshold


def filter_doom_loops(examples: list[dict], n: int = 8,
                      threshold: float = 0.3) -> tuple[list[dict], int]:
    """Filter out doom-loop examples from a list.

    Returns (filtered_examples, n_removed).
    """
    filtered = []
    removed = 0
    for ex in examples:
        text = ex.get("text", ex.get("solution", ex.get("response", "")))
        if is_doom_loop(text, n, threshold):
            removed += 1
        else:
            filtered.append(ex)
    return filtered, removed


# ── CoT length classification ──────────────────────────────────────────────

def estimate_token_count(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English/code)."""
    return max(1, len(text) // 4)


def has_cot_markers(text: str) -> bool:
    """Check if a solution contains explicit CoT reasoning markers."""
    text_lower = text.lower()
    return any(marker.lower() in text_lower for marker in _COT_MARKERS)


def classify_cot_length(examples: list[dict],
                        short_max_tokens: int = 150,
                        long_min_tokens: int = 300) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify examples into short CoT, long CoT, and neutral buckets.

    - Short CoT: solution is short (≤ short_max_tokens) — direct answers,
      brief explanations. Good for Stage 1 (internal solver).
    - Long CoT: solution is long (≥ long_min_tokens) AND has CoT markers.
      Good for Stage 2 (externalize reasoning).
    - Neutral: doesn't fit either bucket (medium length without CoT markers).

    Args:
        examples: list of {"prompt": ..., "text": ...} dicts
        short_max_tokens: max token count for "short" classification
        long_min_tokens: min token count for "long" classification

    Returns:
        (short_examples, long_examples, neutral_examples)
    """
    short, long_cot, neutral = [], [], []
    for ex in examples:
        text = ex.get("text", ex.get("solution", ex.get("response", "")))
        n_tokens = estimate_token_count(text)
        has_cot = has_cot_markers(text)

        if n_tokens <= short_max_tokens:
            short.append(ex)
        elif n_tokens >= long_min_tokens and has_cot:
            long_cot.append(ex)
        elif n_tokens >= long_min_tokens and not has_cot:
            # Long but no CoT markers — could be code, essays, etc.
            # Include in long bucket for Stage 2 (externalization practice)
            long_cot.append(ex)
        else:
            neutral.append(ex)

    return short, long_cot, neutral


# ── Mix distillation ───────────────────────────────────────────────────────

def mix_distillation(long_examples: list[dict],
                     short_examples: list[dict],
                     mix_ratio: float = 0.5,
                     target_size: int | None = None) -> list[dict]:
    """Blend long CoT (teacher) + short CoT (student/simple) data.

    Mix Distillation (arXiv 2502.12143): small models perform better when
    trained on a MIX of long and short CoT, rather than only long CoT from
    a strong teacher. The mix_ratio controls the fraction of long CoT.

    Args:
        long_examples: long CoT traces (from strong teacher)
        short_examples: short CoT / direct answers (from student or simple teacher)
        mix_ratio: fraction of long CoT in the output (0.5 = 50/50 mix)
        target_size: total number of examples (None = use all available)

    Returns:
        Mixed list of examples, shuffled.
    """
    if not long_examples and not short_examples:
        return []
    if not long_examples:
        return short_examples[:target_size] if target_size else short_examples
    if not short_examples:
        return long_examples[:target_size] if target_size else long_examples

    if target_size:
        n_long = int(target_size * mix_ratio)
        n_short = target_size - n_long
    else:
        # Use all of both, but cap to maintain ratio
        n_long = len(long_examples)
        n_short = int(n_long * (1 - mix_ratio) / mix_ratio)
        n_short = min(n_short, len(short_examples))

    long_sample = long_examples[:n_long]
    short_sample = short_examples[:n_short]

    mixed = long_sample + short_sample
    random.shuffle(mixed)
    return mixed


# ── Data preparation ───────────────────────────────────────────────────────

def load_jsonl(paths: list[str]) -> list[dict]:
    """Load examples from JSONL files, normalizing to {"prompt", "text"}."""
    examples = []
    seen = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        n_loaded = 0
        with open(p, encoding="utf-8") as f:
            content = f.read()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "")
            text = obj.get("solution", obj.get("response", ""))
            if not prompt or not text:
                continue
            key = prompt[:200]
            if key in seen:
                continue
            seen.add(key)
            examples.append({"prompt": prompt, "text": text})
            n_loaded += 1
        print(f"  {path}: {n_loaded} loaded")
    return examples


def save_jsonl(examples: list[dict], path: str):
    """Save examples to a JSONL file in sft_train.py format."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Convert to sft_train.py format: {"prompt": ..., "response": ...}
            out = {"prompt": ex["prompt"], "response": ex["text"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} examples to {path}")


def prepare_curriculum_data(args):
    """Prepare Stage 1 (short CoT) and Stage 2 (long CoT) data files."""
    random.seed(42)
    print(f"\n{'='*70}")
    print("  CURRICULUM DATA PREPARATION")
    print(f"{'='*70}")

    # Load all data
    print("\nLoading data...")
    examples = load_jsonl(args.input)
    print(f"Total loaded: {len(examples)} examples")

    if not examples:
        print("No data loaded. Check --input paths.")
        return

    # Filter doom loops
    if args.filter_doom_loops:
        print(f"\nFiltering doom loops (n={args.doom_loop_n}, threshold={args.doom_loop_threshold})...")
        before = len(examples)
        examples, removed = filter_doom_loops(
            examples, n=args.doom_loop_n, threshold=args.doom_loop_threshold
        )
        print(f"  Removed {removed} doom-loop examples ({removed/before*100:.1f}%)")
        print(f"  Remaining: {len(examples)} examples")

    # Classify by CoT length
    print(f"\nClassifying CoT length "
          f"(short≤{args.short_cot_max_tokens}tok, long≥{args.long_cot_min_tokens}tok)...")
    short, long_cot, neutral = classify_cot_length(
        examples, args.short_cot_max_tokens, args.long_cot_min_tokens
    )
    print(f"  Short CoT: {len(short)} examples (Stage 1 — internal solver)")
    print(f"  Long CoT:  {len(long_cot)} examples (Stage 2 — externalize)")
    print(f"  Neutral:   {len(neutral)} examples (medium length, no CoT markers)")

    # Stage 1: short CoT + neutral (direct answers build internal solver)
    stage1_data = short + neutral
    random.shuffle(stage1_data)

    # Stage 2: mix distillation (long CoT + some short for the mix)
    # Use mix_ratio to blend long + short
    stage2_data = mix_distillation(
        long_cot, short, mix_ratio=args.mix_ratio
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = str(output_dir / "stage1_short.jsonl")
    stage2_path = str(output_dir / "stage2_long.jsonl")

    print("\nSaving curriculum data...")
    save_jsonl(stage1_data, stage1_path)
    save_jsonl(stage2_data, stage2_path)

    # Summary
    print(f"\n{'='*70}")
    print("  CURRICULUM DATA READY")
    print(f"{'='*70}")
    print(f"Stage 1 (internal solver): {stage1_path} ({len(stage1_data)} examples)")
    print(f"Stage 2 (externalize):     {stage2_path} ({len(stage2_data)} examples)")
    print("\nNext steps:")
    print("  1. Train Stage 1: python -m research.training.runners.curriculum_sft train-stage1 \\")
    print(f"       --data {stage1_path} --checkpoint <cpt_ckpt> --save <sft1_ckpt>")
    print("  2. Train Stage 2: python -m research.training.runners.curriculum_sft train-stage2 \\")
    print(f"       --data {stage2_path} --checkpoint <sft1_ckpt> --save <sft2_ckpt>")


# ── Training stages (delegates to sft_train.py) ────────────────────────────

def _run_sft(data: str, checkpoint: str, save: str, lr: float,
             max_steps: int, optimizer: str, config: str,
             seq_len: int, batch_size: int, grad_accum: int,
             extra_args: list[str] | None = None) -> int:
    """Run sft_train.py as a subprocess with the given arguments.

    Returns the subprocess exit code.
    """
    cmd = [
        sys.executable, "-m", "forge.training.runners.sft_train",
        "--data", data,
        "--config", config,
        "--checkpoint", checkpoint,
        "--save", save,
        "--lr", str(lr),
        "--max-steps", str(max_steps),
        "--optimizer", optimizer,
        "--seq-len", str(seq_len),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def train_stage1(args):
    """Stage 1: SFT on short CoT + direct answers (internal solver)."""
    print(f"\n{'='*70}")
    print("  STAGE 1: INTERNAL SOLVER (short CoT)")
    print(f"{'='*70}")
    print(f"Data: {args.data}")
    print(f"Checkpoint: {args.checkpoint} → {args.save}")
    print(f"LR: {args.lr} | Steps: {args.max_steps} | Optimizer: {args.optimizer}")

    # Stage 1 uses higher LR (building new capabilities) and standard seq len
    extra = []
    if args.anchor:
        extra.extend(["--anchor", args.anchor, "--l2-lambda", str(args.l2_lambda)])
    if args.ema:
        extra.append("--ema")
    if getattr(args, 'mtp_weight', 0.0) > 0.0:
        extra.extend(["--mtp-weight", str(args.mtp_weight)])

    return _run_sft(
        data=args.data, checkpoint=args.checkpoint, save=args.save,
        lr=args.lr, max_steps=args.max_steps, optimizer=args.optimizer,
        config=args.config, seq_len=args.seq_len,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        extra_args=extra,
    )


def train_stage2(args):
    """Stage 2: SFT on long CoT (externalize reasoning)."""
    print(f"\n{'='*70}")
    print("  STAGE 2: EXTERNALIZE REASONING (long CoT + mix distillation)")
    print(f"{'='*70}")
    print(f"Data: {args.data}")
    print(f"Checkpoint: {args.checkpoint} → {args.save}")
    print(f"LR: {args.lr} | Steps: {args.max_steps} | Optimizer: {args.optimizer}")

    # Stage 2 uses lower LR (refining, not building from scratch)
    # and longer seq len (CoT traces are longer)
    stage2_seq_len = args.seq_len * 2 if args.seq_len < 2048 else args.seq_len
    extra = []
    if args.anchor:
        extra.extend(["--anchor", args.anchor, "--l2-lambda", str(args.l2_lambda)])
    if args.ema:
        extra.append("--ema")
    if getattr(args, 'mtp_weight', 0.0) > 0.0:
        extra.extend(["--mtp-weight", str(args.mtp_weight)])

    return _run_sft(
        data=args.data, checkpoint=args.checkpoint, save=args.save,
        lr=args.lr, max_steps=args.max_steps, optimizer=args.optimizer,
        config=args.config, seq_len=stage2_seq_len,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        extra_args=extra,
    )


def run_full_pipeline(args):
    """Run the full curriculum SFT pipeline: prepare → stage1 → stage2."""
    # Step 1: Prepare data
    prepare_args = argparse.Namespace(
        input=args.input, output_dir=args.output_dir,
        short_cot_max_tokens=args.short_cot_max_tokens,
        long_cot_min_tokens=args.long_cot_min_tokens,
        mix_ratio=args.mix_ratio,
        filter_doom_loops=args.filter_doom_loops,
        doom_loop_n=args.doom_loop_n,
        doom_loop_threshold=args.doom_loop_threshold,
    )
    prepare_curriculum_data(prepare_args)

    stage1_path = str(Path(args.output_dir) / "stage1_short.jsonl")
    stage2_path = str(Path(args.output_dir) / "stage2_long.jsonl")

    # Step 2: Stage 1 training
    sft1_save = args.save.replace(".safetensors", "_SFT1.safetensors")
    stage1_args = argparse.Namespace(
        data=stage1_path, checkpoint=args.checkpoint, save=sft1_save,
        lr=args.stage1_lr, max_steps=args.stage1_steps,
        optimizer=args.optimizer, config=args.config,
        seq_len=args.seq_len, batch_size=args.batch_size,
        grad_accum=args.grad_accum, anchor=args.anchor,
        l2_lambda=args.l2_lambda, ema=args.ema,
        mtp_weight=getattr(args, 'mtp_weight', 0.0),
    )
    rc = train_stage1(stage1_args)
    if rc != 0:
        print(f"Stage 1 failed (exit code {rc}). Aborting.")
        return rc

    # Step 3: Stage 2 training
    stage2_args = argparse.Namespace(
        data=stage2_path, checkpoint=sft1_save, save=args.save,
        lr=args.stage2_lr, max_steps=args.stage2_steps,
        optimizer=args.optimizer, config=args.config,
        seq_len=args.seq_len, batch_size=args.batch_size,
        grad_accum=args.grad_accum, anchor=args.anchor,
        l2_lambda=args.l2_lambda, ema=args.ema,
        mtp_weight=getattr(args, 'mtp_weight', 0.0),
    )
    rc = train_stage2(stage2_args)
    if rc != 0:
        print(f"Stage 2 failed (exit code {rc}).")
    else:
        print(f"\n{'='*70}")
        print("  CURRICULUM SFT COMPLETE")
        print(f"{'='*70}")
        print(f"Stage 1 checkpoint: {sft1_save}")
        print(f"Stage 2 checkpoint: {args.save}")
        print("\nNext: DPO doom-loop mitigation (Phase 3)")
    return rc


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum SFT: Mix Distillation + two-stage curriculum"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── prepare ──
    p_prep = subparsers.add_parser("prepare", help="Prepare curriculum data files")
    p_prep.add_argument("--input", nargs="+", required=True,
                        help="Input JSONL files with {prompt, solution/response}")
    p_prep.add_argument("--output-dir", required=True,
                        help="Output directory for stage1/stage2 JSONL files")
    p_prep.add_argument("--short-cot-max-tokens", type=int, default=150,
                        help="Max token count for 'short' CoT classification")
    p_prep.add_argument("--long-cot-min-tokens", type=int, default=300,
                        help="Min token count for 'long' CoT classification")
    p_prep.add_argument("--mix-ratio", type=float, default=0.5,
                        help="Long:short ratio for Stage 2 mix distillation (0.5=50/50)")
    p_prep.add_argument("--filter-doom-loops", action="store_true", default=True,
                        help="Filter out doom-loop examples (n-gram repetition)")
    p_prep.add_argument("--doom-loop-n", type=int, default=8,
                        help="N-gram size for doom-loop detection")
    p_prep.add_argument("--doom-loop-threshold", type=float, default=0.3,
                        help="Repetition ratio threshold for doom-loop filtering")

    # ── train-stage1 ──
    p_s1 = subparsers.add_parser("train-stage1", help="Stage 1: short CoT (internal solver)")
    p_s1.add_argument("--data", required=True, help="Stage 1 JSONL data file")
    p_s1.add_argument("--checkpoint", required=True, help="Starting checkpoint (CPT output)")
    p_s1.add_argument("--save", required=True, help="Output checkpoint")
    p_s1.add_argument("--config", default="forgelm_v2")
    p_s1.add_argument("--lr", type=float, default=5e-5)
    p_s1.add_argument("--max-steps", type=int, default=1000)
    p_s1.add_argument("--optimizer", default="cpu_offload")
    p_s1.add_argument("--seq-len", type=int, default=1024)
    p_s1.add_argument("--batch-size", type=int, default=2)
    p_s1.add_argument("--grad-accum", type=int, default=4)
    p_s1.add_argument("--anchor", default="", help="Anchor checkpoint for L2-SP")
    p_s1.add_argument("--l2-lambda", type=float, default=0.0)
    p_s1.add_argument("--ema", action="store_true")
    p_s1.add_argument("--mtp-weight", type=float, default=0.0,
                      help="MTP auxiliary loss weight (0=disabled, 0.3=DeepSeek-V3)")

    # ── train-stage2 ──
    p_s2 = subparsers.add_parser("train-stage2", help="Stage 2: long CoT (externalize)")
    p_s2.add_argument("--data", required=True, help="Stage 2 JSONL data file")
    p_s2.add_argument("--checkpoint", required=True, help="Starting checkpoint (Stage 1 output)")
    p_s2.add_argument("--save", required=True, help="Output checkpoint")
    p_s2.add_argument("--config", default="forgelm_v2")
    p_s2.add_argument("--lr", type=float, default=2e-5)
    p_s2.add_argument("--max-steps", type=int, default=1500)
    p_s2.add_argument("--optimizer", default="cpu_offload")
    p_s2.add_argument("--seq-len", type=int, default=2048)
    p_s2.add_argument("--batch-size", type=int, default=2)
    p_s2.add_argument("--grad-accum", type=int, default=4)
    p_s2.add_argument("--anchor", default="", help="Anchor checkpoint for L2-SP")
    p_s2.add_argument("--l2-lambda", type=float, default=0.0)
    p_s2.add_argument("--ema", action="store_true")
    p_s2.add_argument("--mtp-weight", type=float, default=0.0,
                      help="MTP auxiliary loss weight (0=disabled, 0.3=DeepSeek-V3)")

    # ── full ──
    p_full = subparsers.add_parser("full", help="Run full pipeline: prepare + stage1 + stage2")
    p_full.add_argument("--input", nargs="+", required=True)
    p_full.add_argument("--output-dir", required=True)
    p_full.add_argument("--checkpoint", required=True, help="CPT checkpoint to start from")
    p_full.add_argument("--save", required=True, help="Final SFT checkpoint")
    p_full.add_argument("--config", default="forgelm_v2")
    p_full.add_argument("--short-cot-max-tokens", type=int, default=150)
    p_full.add_argument("--long-cot-min-tokens", type=int, default=300)
    p_full.add_argument("--mix-ratio", type=float, default=0.5)
    p_full.add_argument("--filter-doom-loops", action="store_true", default=True)
    p_full.add_argument("--doom-loop-n", type=int, default=8)
    p_full.add_argument("--doom-loop-threshold", type=float, default=0.3)
    p_full.add_argument("--stage1-lr", type=float, default=5e-5)
    p_full.add_argument("--stage1-steps", type=int, default=1000)
    p_full.add_argument("--stage2-lr", type=float, default=2e-5)
    p_full.add_argument("--stage2-steps", type=int, default=1500)
    p_full.add_argument("--optimizer", default="cpu_offload")
    p_full.add_argument("--seq-len", type=int, default=1024)
    p_full.add_argument("--batch-size", type=int, default=2)
    p_full.add_argument("--grad-accum", type=int, default=4)
    p_full.add_argument("--anchor", default="")
    p_full.add_argument("--l2-lambda", type=float, default=0.0)
    p_full.add_argument("--ema", action="store_true")
    p_full.add_argument("--mtp-weight", type=float, default=0.0,
                        help="MTP auxiliary loss weight (0=disabled, 0.3=DeepSeek-V3)")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_curriculum_data(args)
    elif args.command == "train-stage1":
        sys.exit(train_stage1(args))
    elif args.command == "train-stage2":
        sys.exit(train_stage2(args))
    elif args.command == "full":
        sys.exit(run_full_pipeline(args))


if __name__ == "__main__":
    main()

"""Continued pre-training (CPT) with reasoning trace injection.

Implements the midtraining stage of the ForgeLM V10-Thinking recipe:
  - Mix reasoning traces into continued pretraining
  - Teaches the model the "reason first, then answer" pattern
  - Uses full-sequence next-token prediction (no completion masking, unlike SFT)
  - Mixes reasoning data with general knowledge data to prevent catastrophic
    forgetting (Qwen3 Stage 2 approach: STEM + coding + reasoning intensive)

Data format:
  Input JSONL files with {"prompt": ..., "solution": ...} (same as our HF
  datasets). The trainer renders each example as plain text (not chat format)
  and trains on the full sequence with standard cross-entropy.

Text rendering for reasoning examples:
  "{prompt}\n{solution}\n\n"
  (The solution contains the CoT trace + answer, e.g. <|begin_of_thought|>...)

For general knowledge examples (no reasoning):
  "{prompt}\n{response}\n\n"

Data mixing:
  - reasoning_ratio: fraction of batch that is reasoning data (default 0.6)
  - The rest is general knowledge data to maintain broad capabilities
  - Packs multiple short examples into a single sequence (packing) for efficiency

Usage:
  python -m research.training.runners.cpt_train \\
      --reasoning-data data/hf_datasets/openr1_math.jsonl \\
                       data/hf_datasets/openthoughts_114k.jsonl \\
                       data/hf_datasets/dolphin_r1.jsonl \\
      --general-data data/hf_datasets/orca_math.jsonl \\
                     data/hf_datasets/metamath.jsonl \\
      --config forgelm_v2 \\
      --checkpoint research/checkpoints/forgelm_v2_Base.safetensors \\
      --save research/checkpoints/forgelm_v2_CPT.safetensors \\
      --max-steps 5000 --lr 1e-4 --batch-size 2 --seq-len 2048 \\
      --optimizer cpu_offload --reasoning-ratio 0.6
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from forge.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    save_training_checkpoint,
    step_checkpoint_path,
)
from forge.config import get_config
from forge.model_loader import ModelLoader
from forge.runtime.task_logger import task_scope
from forge.training.training_utils import (
    add_safeguard_args,
    configure_optimizer,
    get_lr,
    has_nan_params,
    init_ema,
    load_anchor_cached,
    oom_guard,
    patch_triton_cache_for_windows,
    restore_ema,
    update_ema,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
)
from research.tokenizer_cache import get_tokenizer

# ── Data loading ────────────────────────────────────────────────────────────

def load_jsonl_examples(paths: list[str], max_examples: int | None = None) -> list[dict]:
    """Load examples from JSONL files with {"prompt": ..., "solution"/"response": ...}.

    Returns list of {"prompt": str, "text": str} where text is the full
    completion (solution or response).
    """
    examples: list[dict] = []
    seen: set = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        n_loaded = 0
        n_skipped = 0
        with open(p, encoding="utf-8") as f:
            content = f.read()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            prompt = obj.get("prompt", "")
            # Accept "solution" (reasoning datasets) or "response" (general)
            text = obj.get("solution", obj.get("response", ""))
            if not prompt or not text:
                n_skipped += 1
                continue
            key = prompt[:200]
            if key in seen:
                continue
            seen.add(key)
            examples.append({"prompt": prompt, "text": text})
            n_loaded += 1
            if max_examples and len(examples) >= max_examples:
                break
        print(f"  {path}: {n_loaded} loaded, {n_skipped} skipped")
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def render_cpt_text(example: dict) -> str:
    """Render an example as plain text for CPT (no chat format).

    Reasoning examples: "{prompt}\n{text}\n\n"
    The text already contains CoT markers (<|begin_of_thought|>, think tags, etc.)
    from the source dataset.
    """
    return f"{example['prompt']}\n{example['text']}\n\n"


# ── Sequence packing ────────────────────────────────────────────────────────

def tokenize_and_pack(
    examples: list[dict],
    tokenizer,
    seq_len: int,
    pad_id: int = 0,
    max_examples: int | None = None,
) -> torch.Tensor:
    """Tokenize examples and pack into fixed-length sequences.

    Packs multiple short examples into a single seq_len window for efficiency
    (standard CPT technique — avoids padding waste). Each example is
    concatenated with a newline separator.

    Returns: (N, seq_len) tensor of token IDs (packed, no padding).
    """
    all_tokens: list[int] = []
    n = 0
    for ex in examples:
        text = render_cpt_text(ex)
        enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
        # gigatoken fast path returns list[Encoding]; extract .ids from each
        if ids and hasattr(ids[0], "ids"):
            ids = ids[0].ids
        if not isinstance(ids, list):
            ids = list(ids)
        all_tokens.extend(int(x) for x in ids)
        n += 1
        if max_examples and n >= max_examples:
            break

    # Pack into seq_len chunks; pad the final partial chunk with pad_id.
    n_full = len(all_tokens) // seq_len
    remainder = len(all_tokens) % seq_len
    if n_full == 0 and remainder == 0:
        print("Warning: 0 tokens, nothing to pack")
        return torch.empty(0, seq_len, dtype=torch.long)
    if remainder > 0:
        # Pad the final partial chunk to seq_len so all data is used.
        all_tokens = all_tokens + [pad_id] * (seq_len - remainder)
        n_full += 1
    packed = torch.tensor(all_tokens[: n_full * seq_len], dtype=torch.long)
    packed = packed.view(n_full, seq_len)
    print(f"Packed {len(all_tokens)} tokens into {n_full} sequences of {seq_len}")
    return packed


# ── Mixed data batch sampler ────────────────────────────────────────────────

class MixedDataSampler:
    """Samples batches with a fixed ratio of reasoning vs general data.

    Maintains two separate packed-sequence pools and samples from each
    according to reasoning_ratio. This ensures every batch has the right
    mix, rather than relying on random shuffling of a merged pool.
    """

    def __init__(
        self,
        reasoning_seqs: torch.Tensor,
        general_seqs: torch.Tensor,
        batch_size: int,
        reasoning_ratio: float = 0.6,
    ):
        self.reasoning = reasoning_seqs
        self.general = general_seqs
        self.batch_size = batch_size
        self.reasoning_ratio = reasoning_ratio

        self.n_reasoning_per_batch = max(1, int(batch_size * reasoning_ratio))
        self.n_general_per_batch = batch_size - self.n_reasoning_per_batch

        # Handle edge cases where one pool is empty
        if len(reasoning_seqs) == 0:
            self.n_reasoning_per_batch = 0
            self.n_general_per_batch = batch_size
        if len(general_seqs) == 0:
            self.n_general_per_batch = 0
            self.n_reasoning_per_batch = batch_size

        print(f"MixedDataSampler: {self.n_reasoning_per_batch} reasoning + "
              f"{self.n_general_per_batch} general per batch "
              f"(ratio={reasoning_ratio})")

    def get_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one mixed batch. Returns (input_ids, labels)."""
        parts = []
        if self.n_reasoning_per_batch > 0 and len(self.reasoning) > 0:
            idx = torch.randint(0, len(self.reasoning), (self.n_reasoning_per_batch,))
            parts.append(self.reasoning[idx])
        if self.n_general_per_batch > 0 and len(self.general) > 0:
            idx = torch.randint(0, len(self.general), (self.n_general_per_batch,))
            parts.append(self.general[idx])
        if not parts:
            raise RuntimeError("No data available for batch")
        batch = torch.cat(parts, dim=0).to(device)
        # CPT: labels = input_ids shifted by 1 (next-token prediction)
        # The model's forward handles the shift internally; we provide
        # input_ids and labels = input_ids (model shifts internally)
        labels = batch.clone()
        return batch, labels


# ── Main training loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPT with reasoning trace injection")
    parser.add_argument("--reasoning-data", nargs="+", required=True,
                        help="JSONL files with reasoning traces (prompt + solution)")
    parser.add_argument("--general-data", nargs="+", default=[],
                        help="JSONL files with general data (prompt + response/solution)")
    parser.add_argument("--config", default="forgelm_v2",
                        help="Model config preset name")
    parser.add_argument("--checkpoint", required=True,
                        help="Starting checkpoint (safetensors)")
    parser.add_argument("--save", required=True,
                        help="Output checkpoint path (safetensors)")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="CPT learning rate (higher than SFT, lower than pretrain)")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length (2048-4096 for reasoning traces)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation (effective batch = bs * grad_accum)")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--optimizer", default="cpu_offload",
                        help="Optimizer: cpu_offload (default), fused, bnb, muon")
    parser.add_argument("--reasoning-ratio", type=float, default=0.6,
                        help="Fraction of batch that is reasoning data (0.6 = 60%)")
    parser.add_argument("--mtp-weight", type=float, default=0.0,
                        help="MTP auxiliary loss weight (Nemotron Lightning / DeepSeek-V3). "
                             "When > 0, enables multi-token prediction auxiliary loss. "
                             "0 = disabled (default). 0.3 = DeepSeek-V3 default.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples to load per data source (for debugging)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--heartbeat", action="store_true", default=True)
    parser.add_argument("--ema", action="store_true",
                        help="Enable EMA shadow weights")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--anchor", default="",
                        help="Anchor checkpoint for L2-SP regularization")
    parser.add_argument("--l2-lambda", type=float, default=0.0,
                        help="L2-SP lambda (0=disabled)")
    add_safeguard_args(parser)
    args = parser.parse_args()

    patch_triton_cache_for_windows()
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"\n{'='*70}")
    print("  CPT WITH REASONING TRACE INJECTION")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Save: {args.save}")
    print(f"Steps: {args.max_steps} | LR: {args.lr} | Seq: {args.seq_len}")
    print(f"Optimizer: {args.optimizer} | Reasoning ratio: {args.reasoning_ratio}")

    # ── Load tokenizer ──
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print(f"Tokenizer: vocab={tokenizer.vocab_size}, pad_id={pad_id}")

    # ── Load data ──
    print("\nLoading reasoning data...")
    reasoning_examples = load_jsonl_examples(args.reasoning_data, args.max_examples)
    print(f"Total reasoning examples: {len(reasoning_examples)}")

    general_examples = []
    if args.general_data:
        print("\nLoading general data...")
        general_examples = load_jsonl_examples(args.general_data, args.max_examples)
        print(f"Total general examples: {len(general_examples)}")

    if not reasoning_examples and not general_examples:
        raise RuntimeError("No data loaded. Check file paths.")

    # ── Tokenize and pack ──
    print(f"\nTokenizing and packing (seq_len={args.seq_len})...")
    reasoning_seqs = tokenize_and_pack(
        reasoning_examples, tokenizer, args.seq_len, pad_id, args.max_examples
    ) if reasoning_examples else torch.empty(0, args.seq_len, dtype=torch.long)

    general_seqs = tokenize_and_pack(
        general_examples, tokenizer, args.seq_len, pad_id, args.max_examples
    ) if general_examples else torch.empty(0, args.seq_len, dtype=torch.long)

    sampler = MixedDataSampler(
        reasoning_seqs, general_seqs, args.batch_size, args.reasoning_ratio
    )

    # ── Build model ──
    print(f"\nBuilding model ({args.config})...")
    config = get_config(args.config)
    # MTP auxiliary loss (Nemotron Lightning / DeepSeek-V3)
    if args.mtp_weight > 0.0:
        config.use_mtp = True
        config.mtp_loss_weight = args.mtp_weight
        print(f"MTP auxiliary loss enabled (weight={args.mtp_weight})")
    model = ModelLoader.build_model(config, checkpoint_path=args.checkpoint)
    model = model.to(device).to(dtype)
    model.train()

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None
        # Fallback: set attribute that our model checks
        if hasattr(model, "use_grad_checkpoint"):
            model.use_grad_checkpoint = True

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # Auto-default to CPUAdamW on small GPUs (<15GB) if user didn't override
    if args.optimizer == "adamw" and torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_vram_gb < 15.0:
            print(f"  [Auto] GPU has {total_vram_gb:.1f}GB VRAM — switching to CPUAdamW optimizer")
            args.optimizer = "cpu_offload"

    # ── Optimizer ──
    optimizer = configure_optimizer(
        model, args.lr, args.weight_decay, optimizer_name=args.optimizer
    )

    # ── EMA ──
    ema_state = None
    if args.ema:
        ema_state = init_ema(model)
        print(f"EMA enabled (decay={args.ema_decay})")

    # ── L2-SP anchor ──
    anchor_named_params = None
    if args.anchor and args.l2_lambda > 0:
        print(f"Loading anchor checkpoint for L2-SP: {args.anchor}")
        anchor_sd = load_anchor_cached(args.anchor)
        anchor_named_params = {}
        for name, p in model.named_parameters():
            if name in anchor_sd:
                anchor_named_params[name] = anchor_sd[name].to(device).to(p.dtype)
        print(f"  L2-SP active: {len(anchor_named_params)} matched params")

    # ── Training loop ──
    grad_accum = max(1, args.grad_accum)
    eff_batch = args.batch_size * grad_accum
    print(f"\nTraining {args.max_steps} steps | batch {args.batch_size} | "
          f"grad_accum {grad_accum} | eff_batch {eff_batch} | "
          f"lr {args.lr} | seq_len {args.seq_len}")

    step = 0
    accum_count = 0
    last_loss = 0.0
    t0 = time.time()

    with task_scope("cpt"):
        while step < args.max_steps:
            if vram_exceeded(args.vram_limit_gb, device):
                print("VRAM limit exceeded; emergency save + abort.")
                emergency_save(model, args.save, "emergency", step, optimizer=optimizer)
                break

            input_ids, labels = sampler.get_batch(str(device))

            # Forward — CPT uses full-sequence CE (no completion masking)
            with oom_guard(str(device), label="cpt_fwd") as safe:
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=("cuda" in str(device)),
                ):
                    out = model(input_ids)
                    logits = out[0] if isinstance(out, tuple) else out

                # Next-token prediction: logits[:, :-1] predict labels[:, 1:]
                # CE upcasts to fp32 internally — no need for explicit .float()
                # (which would double VRAM by materializing a fp32 logits copy)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                )

                # L2-SP regularization
                if anchor_named_params and args.l2_lambda > 0:
                    l2_sp = 0.0
                    for name, p in model.named_parameters():
                        if name in anchor_named_params:
                            l2_sp += F.mse_loss(p, anchor_named_params[name])
                    loss = loss + args.l2_lambda * l2_sp

                # Backward
                (loss / grad_accum).backward()
            if safe.skipped:
                optimizer.zero_grad()
                accum_count = 0
                continue
            accum_count += 1
            last_loss = loss.item()

            if accum_count < grad_accum:
                continue

            # Optimizer step
            step += 1
            with torch.no_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            # DeepSeek-V3 aux-loss-free: update expert bias after step.
            try:
                from forge.moe.moe import disable_dense_bypass, update_moe_biases
                update_moe_biases(model)
                # Disable dense_bypass after warmup so router activates.
                warmup_steps = getattr(config, 'moe_dense_bypass_warmup_steps', 0)
                if warmup_steps > 0 and step == warmup_steps:
                    disable_dense_bypass(model)
            except Exception:
                logger.debug("MoE dense_bypass disable failed (no-op for dense models)", exc_info=True)  # no-op for dense models
            accum_count = 0

            # EMA update
            if ema_state:
                update_ema(ema_state, model, args.ema_decay)

            # Logging
            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t0
                tok_s = step * eff_batch * args.seq_len / max(elapsed, 1)
                print(f"  step {step}/{args.max_steps} | loss {last_loss:.4f} | "
                      f"lr {lr:.2e} | {tok_s:.0f} tok/s | "
                      f"{elapsed:.0f}s")
                if args.heartbeat:
                    write_heartbeat("cpt")
                write_status_json("cpt", step, args.max_steps, last_loss, lr)

            # Periodic save
            if step % args.save_every == 0 and step < args.max_steps:
                ckpt_path = step_checkpoint_path(args.save, step)
                save_training_checkpoint(model, ckpt_path, optimizer=optimizer,
                                         ema_state=ema_state, step=step)
                print(f"  Saved step checkpoint: {ckpt_path}")

            # NaN check
            if step % 50 == 0 and has_nan_params(model):
                print(f"NaN detected at step {step}! Emergency save + abort.")
                emergency_save(model, args.save, "nan", step, optimizer=optimizer)
                break

    # ── Final save ──
    print(f"\nSaving final checkpoint: {args.save}")
    if ema_state:
        # Apply EMA weights before saving (use shared utility, not inline copy)
        restore_ema(ema_state, model)
        print("  Applied EMA weights before save")
    save_training_checkpoint(model, args.save, optimizer=optimizer,
                             ema_state=ema_state, step=step)
    cleanup_step_checkpoints(args.save, step)
    print(f"Done. Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

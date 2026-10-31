"""R31: Self-play format QLoRA training on top of R30.

Trains the model to:
1. Emit valid pythonic tool calls: <|tool_call_start|>[tool(args)]<|tool_call_end|>
2. Write working Python code
3. Do multi-turn agent trajectories (think → run_script → interpret → record)

Starts from R30 merged checkpoint, adds new LoRA adapters, trains on r31 data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

FORGE_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = FORGE_ROOT / "research" / "data" / "finetune"
R30_CHECKPOINT = FORGE_ROOT / "research" / "checkpoints" / "ForgeLM_V2_Light_R30.safetensors"
TOKENIZER_DIR = FORGE_ROOT / "research" / "checkpoints" / "lfm25_tokenizer"
CACHE_DIR = FORGE_ROOT / "research" / "data" / "tok_cache_r31"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
BOS = "<|startoftext|>"

# V10 FFN-trio: 30720 params/rank/layer, 16 layers = 491520 params/rank
V10_PARAMS_PER_RANK = 3 * (2048 + 8192) * 16  # 491520
MAX_RANK = 64


def load_examples(path: str) -> list[dict]:
    """Load JSONL examples (prompt/response or messages format)."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def tokenize_single(ex: dict, tokenizer, max_seq_len: int) -> dict | None:
    """Tokenize prompt/response format with completion-only masking."""
    prompt = ex.get("prompt", "")
    response = ex.get("response", "")
    if not prompt or not response:
        return None

    prompt_text = f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    completion_text = f"{response}{IM_END}\n"
    full_text = prompt_text + completion_text

    input_ids = tokenizer(full_text, add_special_tokens=False,
                          truncation=True, max_length=max_seq_len)["input_ids"]
    if len(input_ids) < 4 or len(input_ids) > max_seq_len:
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                           truncation=True, max_length=max_seq_len)["input_ids"]
    comp_start = min(len(prompt_ids), len(input_ids))

    labels = [-100] * comp_start + list(input_ids[comp_start:])
    labels = labels[:len(input_ids)]
    while len(labels) < len(input_ids):
        labels.append(-100)

    return {"input_ids": input_ids, "labels": labels}


def tokenize_multi_turn(ex: dict, tokenizer, max_seq_len: int) -> dict | None:
    """Tokenize messages format (multi-turn) with assistant-only masking.

    Renders ChatML: <|im_start|>role\ncontent<|im_end|>\n for each message.
    Only assistant turns have labels (others are -100).
    """
    messages = ex.get("messages", [])
    if not messages:
        return None

    input_ids = []
    labels = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            text = f"{IM_START}system\n{content}{IM_END}\n"
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))
        elif role == "user":
            text = f"{IM_START}user\n{content}{IM_END}\n"
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))
        elif role == "assistant":
            text = f"{IM_START}assistant\n{content}{IM_END}\n"
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(ids)
            labels.extend(ids)  # Train on assistant content
        elif role == "tool":
            # Tool response: wrapped in tool_response tokens
            text = f"<|tool_response_start|>{content}<|tool_response_end|>{IM_END}\n"
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))  # Don't train on tool outputs

    if len(input_ids) < 4:
        return None
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    # Count trainable tokens
    n_train = sum(1 for l in labels if l != -100)
    if n_train < 2:
        return None

    return {"input_ids": input_ids, "labels": labels}


def tokenize_all(examples: list[dict], tokenizer, max_seq_len: int) -> list[dict]:
    """Tokenize all examples (handles both formats)."""
    dataset = []
    n_single = 0
    n_multi = 0
    n_filtered = 0

    for ex in examples:
        if "messages" in ex:
            result = tokenize_multi_turn(ex, tokenizer, max_seq_len)
            if result:
                n_multi += 1
        elif "prompt" in ex:
            result = tokenize_single(ex, tokenizer, max_seq_len)
            if result:
                n_single += 1
        else:
            n_filtered += 1
            continue

        if result:
            dataset.append(result)
        else:
            n_filtered += 1

    print(f"  Tokenized: {len(dataset)} examples ({n_single} single-turn, {n_multi} multi-turn, {n_filtered} filtered)")
    return dataset


def collate_batch(batch: list[dict], pad_id: int, device: str) -> tuple:
    """Left-pad batch for causal LM."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    b = len(batch)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((b, max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        labs = ex["labels"]
        n = len(ids)
        offset = max_len - n
        input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
        labels[i, offset:] = torch.tensor(labs, dtype=torch.long)
        attn_mask[i, offset:] = 1
    return (input_ids.to(device), labels.to(device), attn_mask.to(device))


def compute_loss(model, input_ids, labels, attn_mask) -> torch.Tensor:
    """Completion-only CE loss."""
    outputs = model(input_ids, attention_mask=attn_mask)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    if not mask.any():
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    loss = F.cross_entropy(
        shift_logits[mask], shift_labels[mask],
        reduction="mean"
    )
    return loss


class AsyncPrefetcher:
    """Simple async prefetcher for overlapping data loading with GPU."""
    def __init__(self, dataset, batch_size, pad_id, device, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.device = device
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        self.pos = 0
        self._next_batch = None
        self._stream = torch.cuda.Stream() if device == "cuda" else None

    def _get_next(self):
        if self.pos >= len(self.indices):
            return None
        batch_indices = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        batch = [self.dataset[i] for i in batch_indices]
        return collate_batch(batch, self.pad_id, self.device)

    def prefetch(self):
        if self._next_batch is None:
            self._next_batch = self._get_next()

    def next(self):
        if self._next_batch is not None:
            batch = self._next_batch
            self._next_batch = None
            return batch
        return self._get_next()

    def reset(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.indices)
        self._next_batch = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(FINETUNE_DIR / "r31_v3_training.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--resume-step", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("  R31: Self-Play Format QLoRA Training")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    examples = load_examples(args.data)
    print(f"  Loaded: {len(examples)} examples from {args.data}")

    # Split train/val
    random.seed(42)
    random.shuffle(examples)
    n_val = max(10, int(len(examples) * args.val_size))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")

    # ── Load tokenizer ────────────────────────────────────────────────────
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer(str(TOKENIZER_DIR))
    pad_id = tok.pad_token_id or 0

    # ── Tokenize ──────────────────────────────────────────────────────────
    print("\n[2/4] Tokenizing...")
    t0 = time.time()
    train_data = tokenize_all(train_examples, tok, args.max_len)
    val_data = tokenize_all(val_examples, tok, args.max_len)
    print(f"  Tokenization: {time.time()-t0:.1f}s")
    print(f"  Train tokens: {sum(len(ex['input_ids']) for ex in train_data)}")
    print(f"  Val tokens: {sum(len(ex['input_ids']) for ex in val_data)}")

    # ── Load model + LoRA ─────────────────────────────────────────────────
    print("\n[3/4] Loading model + QLoRA...")
    ckpt = args.resume_from or str(R30_CHECKPOINT)
    print(f"  Base: {ckpt}")

    from forge.engine.forge_engine import ForgeEngine
    engine = ForgeEngine.from_checkpoint(
        ckpt, config_name="forgelm_v2_light",
        device="cuda", auto_activate=False
    )
    model = engine.model
    model.eval()

    # Add LoRA adapters — FFN + attention (attention needed for tool-name copying from system prompt)
    from forge.training.bitnet_lora import add_lora_adapters
    n_adapters, lora_params_list = add_lora_adapters(
        model, rank=args.max_rank, alpha=args.max_rank * 2,
        target_modules=["w_gate", "w_up", "w_down", "q_proj", "v_proj", "out_proj", "in_proj"]
    )
    total_params = sum(p.numel() for p in lora_params_list)
    print(f"  QLoRA: {n_adapters} adapters, rank={args.max_rank}, {total_params:,} params ({total_params/1e6:.1f}M)")

    # Resume from checkpoint if provided
    if args.resume_from and args.resume_step > 0:
        # Load LoRA weights from a separate LoRA checkpoint
        lora_ckpt = args.resume_from
        if os.path.exists(lora_ckpt):
            from safetensors.torch import load_file
            state = load_file(lora_ckpt)
            loaded = 0
            for name, param in model.named_parameters():
                if "lora_" in name and name in state:
                    param.data.copy_(state[name])
                    loaded += 1
            print(f"  Resumed LoRA from {lora_ckpt} ({loaded} params loaded)")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[4/4] Training...")
    eff_batch = args.batch_size * args.grad_accum
    n_steps = (len(train_data) * args.epochs) // eff_batch
    # Round up to nearest multiple of eff_batch
    n_steps = ((n_steps + eff_batch - 1) // eff_batch) * eff_batch
    print(f"  {len(train_data)} examples, {args.epochs} epochs, {n_steps} steps (eff_batch={eff_batch})")

    # Collect LoRA params
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.95))

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, n_steps - args.warmup_steps)
        return args.lr * 0.1 ** (progress * 2)  # decay to 1% of peak

    prefetcher = AsyncPrefetcher(train_data, args.batch_size, pad_id, "cuda")
    step = args.resume_step
    t0 = time.time()
    best_val_loss = float("inf")
    lr = args.lr

    for _epoch in range(args.epochs):
        prefetcher.reset()
        optimizer.zero_grad()

        while True:
            if step >= n_steps:
                break
            prefetcher.prefetch()
            batch = prefetcher.next()
            if batch is None:
                break

            input_ids, labels, attn_mask = batch
            loss = compute_loss(model, input_ids, labels, attn_mask)
            loss = loss / args.grad_accum
            loss.backward()

            if (step - args.resume_step + 1) % args.grad_accum == 0:
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  step {step}/{n_steps} loss={loss.item()*args.grad_accum:.4f} lr={lr:.2e} {elapsed:.0f}s")

            # Validation
            if step % args.val_every == 0 and step > 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for i in range(0, len(val_data), args.batch_size):
                        vbatch = val_data[i:i+args.batch_size]
                        if not vbatch:
                            continue
                        vi, vl, va = collate_batch(vbatch, pad_id, "cuda")
                        vl_loss = compute_loss(model, vi, vl, va)
                        val_losses.append(vl_loss.item())
                val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
                train_loss = loss.item() * args.grad_accum
                print(f"  [Val] step {step}: val_loss={val_loss:.4f} (train_loss={train_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                model.train()

            # Checkpoint
            if step % args.save_every == 0 and step > 0:
                ckpt_path = FORGE_ROOT / "research" / "checkpoints" / f"ForgeLM_V2_Light_R31_step{step}.safetensors"
                lora_state = {n: p.cpu() for n, p in model.named_parameters() if "lora_" in n}
                from safetensors.torch import save_file
                save_file(lora_state, str(ckpt_path))
                print(f"  [Checkpoint] LoRA saved: {ckpt_path}")

            step += 1

        if step >= n_steps:
            break

    # ── Final save: LoRA adapter only (base R30 stays intact) ─────────────
    print("\n  Saving LoRA adapter (base model not merged)...")
    from safetensors.torch import save_file as _save
    lora_state = {n: p.cpu() for n, p in model.named_parameters() if "lora_" in n}
    final_path = FORGE_ROOT / "research" / "checkpoints" / "ForgeLM_V2_Light_R31_lora.safetensors"
    _save(lora_state, str(final_path))
    print(f"  Saved LoRA: {final_path} ({len(lora_state)} tensors, {sum(p.numel() for p in lora_state.values())/1e6:.1f}M params)")
    print(f"  Base model (R30) untouched: {R30_CHECKPOINT}")

    # Final validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, len(val_data), args.batch_size):
            vbatch = val_data[i:i+args.batch_size]
            if not vbatch:
                continue
            vi, vl, va = collate_batch(vbatch, pad_id, "cuda")
            vl_loss = compute_loss(model, vi, vl, va)
            val_losses.append(vl_loss.item())
    final_val = sum(val_losses) / len(val_losses) if val_losses else 0
    print(f"\n  Final val_loss: {final_val:.4f}")
    print("\n=== R31 complete ===")


if __name__ == "__main__":
    main()

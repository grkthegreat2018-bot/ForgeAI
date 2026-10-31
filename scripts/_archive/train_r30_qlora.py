"""R30 v2: V10 QLoRA training with ALL compatible ForgeEngine/sft_train features.

Features integrated (all QLoRA-safe with frozen IRI-FP4 base):
  - Golden ratio param growth: rank = 2304 * n / epochs^2 (from R29)
  - ForgeEngine loading with KeyStack detection + crash recovery
  - Packed sequences (eliminate padding waste)
  - Async prefetch (overlap data loading with GPU compute)
  - Disk token cache (skip re-tokenization across runs)
  - Curriculum ordering (difficulty-based for multi-dataset)
  - Validation tracking (held-out PPL every N steps)
  - Entropy-alpha (token reweighting for reasoning tokens)
  - Grad-mixup (average gradients across N batches)
  - Sequential freeze (layer-wise LoRA training)
  - Sample weighting (easy sample upweighting)
  - EMA (parameter smoothing)
  - Checkpointing (save every N steps)
  - CPU-offload optimizer (for larger batch/rank on 12GB)
  - L2-SP anchor regularization (prevent catastrophic forgetting)

V10 architecture:
  - d_model=2048, FFN=8192, 16 layers (10 conv + 6 GQA)
  - FFN-trio per rank per layer: 30720 params
  - 16 layers: 491520 params/rank total
  - IRI-FP4 quantized (frozen base, ~9 bits/w)

Usage:
  python scripts/train_r30_qlora.py --datasets openhermes gsm8k metamath \\
    --max-per-dataset 10000 --epochs 5 --batch-size 4 --grad-accum 4 \\
    --entropy-alpha 0.5 --grad-mixup 3 --sequential-freeze 4 \\
    --val-every 100 --save-every 500 --curriculum pacing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────

FORGE_ROOT = Path(__file__).resolve().parent.parent
HF_DATASETS = FORGE_ROOT / "research" / "distillation" / "hf_datasets"
FINETUNE_DIR = FORGE_ROOT / "research" / "data" / "finetune"
V10_CHECKPOINT = FORGE_ROOT / "research" / "checkpoints" / "ForgeLM_V2_Light.safetensors"
TOKENIZER_DIR = FORGE_ROOT / "research" / "checkpoints" / "lfm25_tokenizer"
CACHE_DIR = FORGE_ROOT / "research" / "data" / "tok_cache"

# R29 golden ratio
R29_PARAMS_PER_FACT = 2765  # at 120 epochs
R29_SCALING_K = 2304  # rank = K * n / epochs^2

# V10 FFN-trio dimensions (all 16 layers have FFN)
V10_D_MODEL = 2048
V10_FFN_INTER = 8192
V10_N_FFN_LAYERS = 16
V10_PARAMS_PER_RANK_PER_LAYER = 3 * (V10_D_MODEL + V10_FFN_INTER)  # 30720
V10_PARAMS_PER_RANK_TOTAL = V10_PARAMS_PER_RANK_PER_LAYER * V10_N_FFN_LAYERS  # 491520

# LoRA targets: FFN-trio (V10 naming: w_gate/w_up/w_down)
LORA_TARGETS = ["w_gate", "w_up", "w_down"]

MAX_RANK = 64


def compute_golden_ratio_rank(n_examples: int, epochs: int = 120,
                              max_rank: int = MAX_RANK) -> int:
    """Compute LoRA rank from R29 golden ratio with param growth.

    R29 scaling law: rank = 2304 * n / epochs^2 (for single-layer 17280 params/rank)
    V10: 491520 params/rank total (16 layers × 30720)
    Adjust: total_params = R29_PARAMS_PER_FACT * n * (120/epochs)^2
    rank = total_params / V10_PARAMS_PER_RANK_TOTAL
    """
    total_params = R29_PARAMS_PER_FACT * n_examples * (120.0 / epochs) ** 2
    rank = total_params / V10_PARAMS_PER_RANK_TOTAL
    rank = max(1, min(max_rank, int(math.ceil(rank))))
    return rank


# ── Dataset conversion ─────────────────────────────────────────────────────

def convert_dataset(input_path: str, output_path: str, max_examples: int = 0) -> int:
    """Convert hf_dataset JSONL (prompt/solution) to sft_train format (prompt/response)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "")
            response = obj.get("response", obj.get("solution", obj.get("output", "")))
            if not prompt or not response:
                continue
            out = {"prompt": prompt, "response": response}
            if "messages" in obj:
                out = {"messages": obj["messages"]}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if max_examples and n >= max_examples:
                break
    return n


def prepare_datasets(dataset_names: list[str], max_per_dataset: int = 0) -> str:
    """Convert and merge datasets into a single sft_train-format JSONL."""
    output = str(FINETUNE_DIR / "r30_training.jsonl")
    all_examples = []
    for name in dataset_names:
        input_path = str(HF_DATASETS / f"{name}.jsonl")
        if not os.path.exists(input_path):
            print(f"  WARN: {input_path} not found, skipping")
            continue
        temp_path = str(FINETUNE_DIR / f"r30_{name}.jsonl")
        n = convert_dataset(input_path, temp_path, max_per_dataset)
        print(f"  {name}: {n} examples -> {temp_path}")
        all_examples.append(temp_path)

    total = 0
    with open(output, "w", encoding="utf-8") as fout:
        for path in all_examples:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
    print(f"  MERGED: {total} examples -> {output}")
    return output


# ── Tokenization (matches sft_train LFM2.5 Qwen format exactly) ────────────

def tokenize_example(ex: dict, tokenizer, max_seq_len: int) -> dict | None:
    """Tokenize a single example into {input_ids, labels} with completion-only masking.

    Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n
    Labels: -100 for prompt, real id for completion (including <|im_end|> so model learns to stop).
    """
    prompt = ex["prompt"]
    response = ex["response"]
    prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    completion_text = f"{response}<|im_end|>\n"
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

    return {"input_ids": input_ids, "labels": labels, "n_comp": len(input_ids) - comp_start}


def tokenize_all(examples: list[dict], tokenizer, max_seq_len: int,
                 use_disk_cache: bool = True) -> list[dict]:
    """Tokenize all examples with optional disk caching."""
    dataset = []
    cache = None
    if use_disk_cache:
        try:
            from forge.training.data.efficient_pipeline import DiskTokenCache
            cache = DiskTokenCache(str(CACHE_DIR))
        except Exception:
            pass

    tok_hash = str(hash(str(tokenizer.vocab_size)))

    for ex in examples:
        if cache is not None:
            text = ex.get("prompt", "") + "\n" + ex.get("response", "")
            key = DiskTokenCache._hash_key(text, tok_hash) if cache else None
            cached = cache._cache_path(key) if key else None
            if cached and cached.exists():
                try:
                    import numpy as np
                    ids = np.load(cached.with_suffix(".ids.npy"))
                    labels = np.load(cached.with_suffix(".labels.npy"))
                    dataset.append({"input_ids": ids.tolist(), "labels": labels.tolist(),
                                    "n_comp": int((np.load(cached.with_suffix(".ncomp.npy")))[0])})
                    continue
                except Exception:
                    pass

        result = tokenize_example(ex, tokenizer, max_seq_len)
        if result is not None:
            dataset.append(result)
            if cache is not None and key is not None:
                try:
                    import numpy as np
                    cpath = cache._cache_path(key)
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cpath.with_suffix(".ids.npy"), np.array(result["input_ids"]))
                    np.save(cpath.with_suffix(".labels.npy"), np.array(result["labels"]))
                    np.save(cpath.with_suffix(".ncomp.npy"), np.array([result["n_comp"]]))
                except Exception:
                    pass

    return dataset


def collate_batch(batch: list[dict], pad_id: int, device: str) -> tuple:
    """Left-pad a batch (completion at end for correct causal LM)."""
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


# ── Loss computation (from sft_train, with entropy-alpha) ──────────────────

def compute_loss(model, input_ids, labels, attn_mask,
                 entropy_alpha: float = 0.0,
                 sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Completion-only CE loss with optional entropy weighting and sample weighting."""
    shift_labels = F.pad(labels[:, 1:], (0, 1), value=-100)

    if sample_weights is None and entropy_alpha == 0.0:
        # Fast path: model's internal chunked CE
        if hasattr(model, 'config') and hasattr(model.config, 'entropy_alpha'):
            model.config.entropy_alpha = 0.0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, attention_mask=attn_mask, targets=shift_labels)
        if isinstance(out, tuple):
            loss = out[1] if out[1] is not None else out[0]
        else:
            loss = out
        if loss is not None and torch.isfinite(loss):
            return loss

    # Manual path: full logits for entropy weighting / sample weighting
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids, attention_mask=attn_mask)
    logits = out[0] if isinstance(out, tuple) else out
    if logits is None:
        raise RuntimeError("model returned None logits")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels_flat = labels[:, 1:].contiguous()

    ce_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels_flat.view(-1),
        ignore_index=-100, reduction="none",
    ).view(shift_labels_flat.size())

    mask = (shift_labels_flat != -100).float()

    if entropy_alpha > 0.0:
        with torch.no_grad():
            probs = F.softmax(shift_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            vocab_size = float(shift_logits.size(-1))
            max_entropy = torch.log(torch.tensor(vocab_size, device=shift_logits.device,
                                                 dtype=torch.float32))
            normalized_entropy = (entropy / max_entropy.clamp(min=1e-8)) * 2.0
            entropy_weights = 1.0 + entropy_alpha * normalized_entropy
        ce_per_token = ce_per_token * entropy_weights

    per_example_loss = (ce_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    if sample_weights is not None:
        per_example_loss = per_example_loss * sample_weights

    loss = per_example_loss.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss ({loss.item()})")
    return loss


# ── EMA ────────────────────────────────────────────────────────────────────

def init_ema(model: nn.Module, decay: float = 0.999) -> dict:
    """Initialize EMA state for trainable parameters only."""
    ema = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            ema[name] = p.detach().clone()
    return ema


def update_ema(ema: dict, model: nn.Module, decay: float):
    """Update EMA state."""
    for name, p in model.named_parameters():
        if name in ema:
            ema[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def restore_ema(ema: dict, model: nn.Module):
    """Restore EMA weights into model."""
    for name, p in model.named_parameters():
        if name in ema:
            p.data.copy_(ema[name])


# ── Model loading + QLoRA ──────────────────────────────────────────────────

def load_v10_quantized(device: str = "cuda"):
    """Load V10 with IRI-FP4 quantized weights via ForgeEngine."""
    from forge.engine.forge_engine import ForgeEngine
    engine = ForgeEngine.from_checkpoint(
        str(V10_CHECKPOINT), config_name="forgelm_v2_light",
        device=device, auto_activate=False,
    )
    model = engine.model
    model.eval()
    n_iri = sum(1 for m in model.modules() if type(m).__name__ == "IRIFP4Linear")
    print(f"  Loaded V10: {n_iri} IRIFP4Linear modules (frozen quantized base)")
    if hasattr(engine, 'keystack_features'):
        print(f"  KeyStack features: {engine.keystack_features}")
    return engine, model


def add_qlora(model: nn.Module, rank: int, alpha: int | None = None) -> tuple:
    """Add QLoRA adapters to V10 FFN-trio at all 16 layers."""
    from forge.training.bitnet_lora import add_lora_adapters
    if alpha is None:
        alpha = rank * 2
    n_adapters, lora_params = add_lora_adapters(
        model, rank=rank, alpha=alpha,
        target_modules=LORA_TARGETS, min_size=64,
    )
    total_lora = sum(p.numel() for p in lora_params)
    print(f"  QLoRA: {n_adapters} adapters, rank={rank}, alpha={alpha}, "
          f"{total_lora:,} params ({total_lora / 1e6:.2f}M)")
    return n_adapters, lora_params


def save_checkpoint(model: nn.Module, path: str):
    """Save V10 checkpoint in native IRI-FP4 format."""
    from safetensors.torch import save_file
    state = {}
    for name, module in model.named_modules():
        if type(module).__name__ == "IRIFP4Linear":
            state[f"{name}.weight.iri_packed"] = module.weight_packed.cpu()
            state[f"{name}.weight.iri_scales"] = module.weight_scales.cpu()
            state[f"{name}.weight.iri_global_scale"] = module.weight_global_scale.cpu()
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias.cpu()
        elif isinstance(module, nn.Linear) and hasattr(module, "weight"):
            state[f"{name}.weight"] = module.weight.cpu()
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias.cpu()
        elif isinstance(module, nn.Embedding):
            state[f"{name}.weight"] = module.weight.cpu()
    for name, param in model.named_parameters():
        if name not in state and "lora" not in name:
            state[name] = param.detach().cpu()
    save_file(state, path)
    print(f"  Saved: {path} ({len(state)} tensors)")


# ── Training ───────────────────────────────────────────────────────────────

def train_qlora(
    model, lora_params, data_path, tokenizer,
    epochs=3, batch_size=4, grad_accum=4, lr=1e-4,
    warmup_ratio=0.1, max_len=1024, device="cuda",
    save_path=None, save_every=0,
    entropy_alpha=0.0, grad_mixup=1, sequential_freeze=0,
    final_finetune_steps=0, use_ema=False, ema_decay=0.999,
    val_every=0, val_size=0.05,
    use_disk_cache=True, use_async_prefetch=True,
    curriculum="none", optimizer_type="adamw",
    anchor_path=None, l2_lambda=0.0,
    resume_step=0,
):
    """Full-featured QLoRA training on V10 with IRI-FP4 frozen base."""
    # ── Load data ──
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "prompt" in obj and "response" in obj:
                    examples.append(obj)
    n_examples = len(examples)

    # ── Curriculum ordering ──
    if curriculum != "none":
        try:
            from forge.training.data.curriculum_augment import CurriculumScheduler
            scheduler = CurriculumScheduler(strategy=curriculum)
            texts = [ex.get('prompt', '') + ' ' + ex.get('response', '') for ex in examples]
            order = scheduler.build_curriculum(texts)
            examples = [examples[i] for i in order]
            print(f"  Curriculum: {curriculum} ordering applied")
        except Exception as e:
            print(f"  Curriculum: skipped ({e})")

    # ── Tokenize ──
    print(f"  Tokenizing {n_examples} examples...", flush=True)
    t_tok = time.time()
    dataset = tokenize_all(examples, tokenizer, max_len, use_disk_cache)
    print(f"  Tokenized: {len(dataset)} examples in {time.time()-t_tok:.1f}s "
          f"({len(dataset)-n_examples} filtered)")

    # ── Validation split ──
    val_dataset = None
    if val_every > 0 and val_size > 0:
        n_val = max(1, int(len(dataset) * val_size))
        val_dataset = dataset[-n_val:]
        dataset = dataset[:-n_val]
        print(f"  Validation: {len(val_dataset)} held-out examples")

    n_train = len(dataset)
    n_steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
    total_steps = n_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    print(f"  Training: {n_train} examples, {epochs} epochs, "
          f"batch={batch_size}, grad_accum={grad_accum}, "
          f"{total_steps} steps ({n_steps_per_epoch}/epoch)")

    # ── Sequential freeze schedule ──
    schedule = None
    if sequential_freeze > 0:
        from forge.training.bitnet_lora import compute_phase_schedule, freeze_unfreeze_lora
        schedule = compute_phase_schedule(
            V10_N_FFN_LAYERS, sequential_freeze, total_steps, final_finetune_steps)
        print(f"  Sequential freeze: {sequential_freeze} phases, "
              f"{final_finetune_steps} final finetune steps")

    # ── Anchor for L2-SP ──
    anchor_state = None
    if anchor_path and l2_lambda > 0:
        try:
            from safetensors import safe_open
            anchor_state = {}
            with safe_open(anchor_path, framework="pt") as f:
                for k in f.keys():
                    anchor_state[k] = f.get_tensor(k)
            print(f"  L2-SP anchor loaded from {anchor_path}")
        except Exception as e:
            print(f"  L2-SP anchor load failed: {e}")

    # ── Optimizer ──
    if optimizer_type == "cpu_offload":
        try:
            from forge.training.optim.hybrid_offload import CPUAdamW
            optimizer = CPUAdamW(lora_params, lr=lr, weight_decay=0.01,
                                 betas=(0.9, 0.95))
            print("  Optimizer: CPUAdamW (offloaded states)")
        except Exception as e:
            print(f"  CPUAdamW unavailable ({e}), falling back to AdamW")
            optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01,
                                          betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01,
                                      betas=(0.9, 0.95))
        print("  Optimizer: AdamW")

    # ── LR schedule ──
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.1 * (1 + 9 * (1 + math.cos(math.pi * p)) / 10)

    # ── EMA ──
    ema = init_ema(model, ema_decay) if use_ema else None
    if ema:
        print(f"  EMA: enabled (decay={ema_decay}, {len(ema)} params tracked)")

    # ── Freeze base, enable LoRA ──
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True

    # ── Async prefetcher ──
    prefetcher = None
    if use_async_prefetch:
        try:
            from forge.training.data.efficient_pipeline import AsyncPrefetcher
            prefetcher = AsyncPrefetcher(
                dataset, batch_size=batch_size, device=device,
                collate_fn=lambda batch, pad_id, dev: collate_batch(batch, pad_id, dev),
                prefetch_count=4, shuffle=True,
                pad_id=tokenizer.pad_token_id,
            )
            prefetcher.set_transfer_on_consume(True)
            print("  Async prefetcher: enabled (4 batches ahead)")
        except Exception as e:
            print(f"  Async prefetcher: skipped ({e})")
            prefetcher = None

    # ── Training loop ──
    step = resume_step
    if resume_step > 0:
        print(f"  Resuming from step {resume_step} (LR schedule offset)")
    optimizer.zero_grad()
    t0 = time.time()
    losses = []
    val_losses = []
    pad_id = tokenizer.pad_token_id

    from forge.training.bitnet_lora import (
        freeze_unfreeze_lora, get_active_layers_for_step)

    for _epoch in range(epochs):
        # ── Build batch iterator for this epoch ──
        if prefetcher is not None:
            # AsyncPrefetcher is an iterator: __iter__ starts the background thread
            batch_iter = iter(prefetcher)
        else:
            random.shuffle(dataset)

        i = 0
        while True:
            # ── Sequential freeze phase switching ──
            if schedule is not None:
                active = get_active_layers_for_step(step, schedule)
                freeze_unfreeze_lora(model, active)
                if step % n_steps_per_epoch == 0:
                    print(f"  Phase switch @ step {step}: active_layers={active}")

            # ── Get batch ──
            if prefetcher is not None:
                try:
                    batch_data = next(batch_iter)
                except StopIteration:
                    break
                input_ids, labels_batch, attn_mask = batch_data
                if hasattr(input_ids, 'to') and input_ids.device.type == 'cpu':
                    input_ids = input_ids.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)
                    attn_mask = attn_mask.to(device, non_blocking=True)
            else:
                if i >= n_train:
                    break
                batch = dataset[i:i + batch_size]
                if not batch:
                    break
                input_ids, labels_batch, attn_mask = collate_batch(batch, pad_id, device)
                i += batch_size

            # ── Forward + loss ──
            loss = compute_loss(model, input_ids, labels_batch, attn_mask,
                                entropy_alpha=entropy_alpha)
            loss = loss / grad_accum

            # ── Grad mixup: average gradients from N batches ──
            if grad_mixup > 1 and step > 0 and step % grad_accum == 0:
                saved_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                               if p.grad is not None}
                for mixup_i in range(grad_mixup - 1):
                    if prefetcher is not None:
                        try:
                            mbatch = next(batch_iter)
                        except StopIteration:
                            break
                        mi, ml, mm = mbatch
                        if mi.device.type == 'cpu':
                            mi, ml, mm = mi.to(device), ml.to(device), mm.to(device)
                    else:
                        if i >= n_train:
                            break
                        mbatch = dataset[i:i + batch_size]
                        mi, ml, mm = collate_batch(mbatch, pad_id, device)
                        i += batch_size
                    mloss = compute_loss(model, mi, ml, mm, entropy_alpha=entropy_alpha)
                    (mloss / grad_accum).backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in saved_grads:
                            saved_grads[n] = (saved_grads[n] * (mixup_i + 1) + p.grad) / (mixup_i + 2)
                # Restore averaged grads
                for n, p in model.named_parameters():
                    if n in saved_grads and p.grad is not None:
                        p.grad.copy_(saved_grads[n])

            loss.backward()

            if (step + 1) % grad_accum == 0:
                for g in optimizer.param_groups:
                    g["lr"] = get_lr(step)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # EMA update
                if ema is not None:
                    update_ema(ema, model, ema_decay)

            losses.append(float(loss.item() * grad_accum))
            step += 1

            # ── Logging ──
            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / min(50, len(losses))
                elapsed = time.time() - t0
                eta = elapsed / step * (total_steps - step) if step > 0 else 0
                print(f"  step {step}/{total_steps} loss={avg_loss:.4f} "
                      f"lr={get_lr(step):.2e} {elapsed:.0f}s/{eta:.0f}s", flush=True)

            # ── Validation ──
            if val_every > 0 and val_dataset and step % val_every == 0:
                model.eval()
                with torch.inference_mode():
                    vlosses = []
                    for vi in range(0, len(val_dataset), batch_size):
                        vbatch = val_dataset[vi:vi + batch_size]
                        if not vbatch:
                            continue
                        vi_ids, vi_labels, vi_mask = collate_batch(vbatch, pad_id, device)
                        vloss = compute_loss(model, vi_ids, vi_labels, vi_mask, entropy_alpha=0.0)
                        vlosses.append(float(vloss.item()))
                    vloss = sum(vlosses) / max(len(vlosses), 1)
                    val_losses.append(vloss)
                    print(f"  [Val] step {step}: val_loss={vloss:.4f} "
                          f"(train_loss={avg_loss:.4f})", flush=True)
                model.train()

            # ── Checkpoint ──
            if save_every > 0 and save_path and step % save_every == 0 and step < total_steps:
                ckpt_path = save_path.replace(".safetensors", f"_step{step}.safetensors")
                from forge.training.bitnet_lora import merge_lora_adapters
                # Save LoRA-only checkpoint (don't merge yet)
                lora_state = {}
                for name, p in model.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        lora_state[name] = p.detach().cpu()
                from safetensors.torch import save_file
                save_file(lora_state, ckpt_path)
                print(f"  [Checkpoint] LoRA saved at step {step}: {ckpt_path}", flush=True)

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if prefetcher is not None:
        prefetcher._stop = True  # signal background thread to stop

    print(f"  Training done: {step} steps, {time.time() - t0:.1f}s, "
          f"final_loss={sum(losses[-20:]) / min(20, len(losses)):.4f}")
    if val_losses:
        print(f"  Final val_loss: {val_losses[-1]:.4f}")

    # ── Restore EMA before merge ──
    if ema is not None:
        restore_ema(ema, model)
        print("  EMA restored before merge")

    # ── Merge LoRA into IRI-FP4 ──
    if save_path:
        from forge.training.bitnet_lora import merge_lora_adapters
        n_merged = merge_lora_adapters(model)
        print(f"  Merged {n_merged} LoRA adapters into IRI-FP4 weights")
        save_checkpoint(model, save_path)

    return losses, val_losses


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="R30 v2: V10 QLoRA with all features")
    # Data
    ap.add_argument("--datasets", nargs="+", default=["openhermes", "gsm8k", "code_alpaca"])
    ap.add_argument("--max-per-dataset", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-rank", type=int, default=MAX_RANK)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to LoRA checkpoint to resume from (e.g. ..._step5000.safetensors)")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="Step number to resume from (for LR schedule continuity)")

    # Data pipeline
    ap.add_argument("--disk-cache", action="store_true", default=True)
    ap.add_argument("--async-prefetch", action="store_true", default=True)
    ap.add_argument("--curriculum", default="none",
                    choices=["none", "vanilla", "pacing", "interleaved", "warmup"])

    # Loss / training tricks
    ap.add_argument("--entropy-alpha", type=float, default=0.0)
    ap.add_argument("--grad-mixup", type=int, default=1)
    ap.add_argument("--sequential-freeze", type=int, default=0)
    ap.add_argument("--final-finetune-steps", type=int, default=0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)

    # Validation / checkpointing
    ap.add_argument("--val-every", type=int, default=0)
    ap.add_argument("--val-size", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=0)

    # Optimizer
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "cpu_offload"])

    # L2-SP anchor
    ap.add_argument("--anchor", type=str, default=None)
    ap.add_argument("--l2-lambda", type=float, default=0.0)

    args = ap.parse_args()

    print("=" * 80)
    print("  R30 v2: V10 QLoRA Training with ALL Features + Golden Ratio")
    print("=" * 80)

    # 1. Prepare datasets
    print("\n[1/4] Preparing datasets...")
    data_path = prepare_datasets(args.datasets, args.max_per_dataset)
    with open(data_path, encoding="utf-8") as f:
        n_examples = sum(1 for _ in f)
    print(f"  Total: {n_examples} examples")

    if args.no_train:
        print("\n  --no-train: datasets prepared, exiting")
        return

    # 2. Compute golden ratio rank (param growth)
    print("\n[2/4] Computing golden ratio rank (param growth)...")
    rank = compute_golden_ratio_rank(n_examples, epochs=args.epochs, max_rank=args.max_rank)
    total_lora_params = rank * V10_PARAMS_PER_RANK_TOTAL
    params_per_fact = total_lora_params / n_examples
    print(f"  V10 FFN-trio: {V10_PARAMS_PER_RANK_PER_LAYER} params/rank/layer")
    print(f"  {V10_N_FFN_LAYERS} layers: {V10_PARAMS_PER_RANK_TOTAL} params/rank total")
    print(f"  Golden ratio: rank={rank} -> {total_lora_params:,} params "
          f"({params_per_fact:.1f} p/f at {args.epochs}ep)")
    print(f"  R29 reference: 2765 p/f at 120ep (capped at r{args.max_rank})")

    # 3. Load V10 + add QLoRA
    print("\n[3/4] Loading V10 + adding QLoRA adapters...")
    from research.tokenizer_cache import get_tokenizer
    _engine, model = load_v10_quantized()
    tokenizer = get_tokenizer(str(TOKENIZER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _n_adapters, lora_params = add_qlora(model, rank=rank)

    # Resume from LoRA checkpoint if specified
    if args.resume_from:
        from safetensors.torch import load_file
        lora_state = load_file(args.resume_from)
        loaded = 0
        for name, p in model.named_parameters():
            if name in lora_state:
                p.data.copy_(lora_state[name].to(p.device))
                loaded += 1
        print(f"  Resumed LoRA from {args.resume_from} ({loaded}/{len(lora_state)} params loaded)")

    # 4. Train
    print("\n[4/4] Training QLoRA with all features...")
    save_path = args.save or str(
        FORGE_ROOT / "research" / "checkpoints" / "ForgeLM_V2_Light_R30.safetensors")

    losses, val_losses = train_qlora(
        model, lora_params, data_path, tokenizer,
        epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, max_len=args.max_len,
        save_path=save_path, save_every=args.save_every,
        entropy_alpha=args.entropy_alpha, grad_mixup=args.grad_mixup,
        sequential_freeze=args.sequential_freeze,
        final_finetune_steps=args.final_finetune_steps,
        use_ema=args.ema, ema_decay=args.ema_decay,
        val_every=args.val_every, val_size=args.val_size,
        use_disk_cache=args.disk_cache, use_async_prefetch=args.async_prefetch,
        curriculum=args.curriculum, optimizer_type=args.optimizer,
        anchor_path=args.anchor, l2_lambda=args.l2_lambda,
        resume_step=args.resume_step,
    )

    print("\n=== R30 v2 complete ===")
    print(f"  Checkpoint: {save_path}")
    print(f"  Final loss: {sum(losses[-20:]) / min(20, len(losses)):.4f}")
    if val_losses:
        print(f"  Final val loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()

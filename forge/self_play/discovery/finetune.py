"""Fine-tune the discovery model on its own curated DB content.

When the LLM deems its DB high-quality and large enough, this module builds
an SFT dataset from the DB's best content and fine-tunes a LoRA adapter on
the current best model. The result is saved as a LoRA checkpoint, then
handed to the epoch manager for best-vs-loser comparison.

Training data sources (only curated, verified content — bloat is excluded):
  - discoveries  -> "Summarize what you discovered." -> summary
  - theories     (status='supported') -> "State the hypothesis." -> statement
  - scripts      (returncode=0, non-empty stdout) -> prompt-stub -> code
  - research     -> "Summarize findings on {query}." -> summary
  - tool_trajectories -> multi-turn tool-use examples

Uses ForgeAI's QLoRA system (IRI-FP4 frozen base + trainable LoRA adapters)
instead of HuggingFace PEFT, which is incompatible with IRI-FP4 quantized
weights. LoRA adapters are saved separately (not merged) so they can be
hot-loaded via engine.load_lora() without reloading the base model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from forge.self_play.discovery.discovery_db import DiscoveryDB
from research.paths import DATA_DIR, V2_CHECKPOINT

_EPOCHS_DIR = DATA_DIR / "discovery" / "epochs"


@dataclass
class FinetuneConfig:
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 8
    max_lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 30
    max_seq_len: int = 512
    weight_decay: float = 0.01
    vram_limit_gb: float = 11.0
    # QLoRA config (IRI-FP4 frozen base + trainable LoRA)
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Target modules: FFN + attention (attention needed for tool-name copying)
    lora_targets: tuple = ("w_gate", "w_up", "w_down", "q_proj", "v_proj", "out_proj", "in_proj")


def build_sft_dataset(db: DiscoveryDB, tokenizer, max_seq_len: int) -> list[dict]:
    """Build (prompt, completion) pairs from curated DB content.

    Now includes tool-use trajectories from the tool_trajectories table.
    Trajectories are rendered in Qwen chat format and split into per-turn
    examples (same approach as sft_train.py).
    """
    pairs: list[tuple[str, str]] = []

    for r in db.query("SELECT summary FROM discoveries WHERE summary IS NOT NULL AND length(summary) > 10"):
        pairs.append(("Record a confirmed discovery.", r["summary"]))

    for r in db.query("SELECT statement, notes FROM theories WHERE status='supported' AND length(statement) > 10"):
        comp = r["statement"] + (f" Evidence: {r['notes']}" if r["notes"] else "")
        pairs.append(("State a supported hypothesis.", comp))

    for r in db.query("SELECT code, stdout FROM scripts WHERE returncode=0 AND length(code) > 20 AND length(stdout) > 0 LIMIT 200"):
        pairs.append(("Write useful Python code.", r["code"][:800]))

    for r in db.query("SELECT query, summary FROM research WHERE summary IS NOT NULL AND length(summary) > 10 LIMIT 100"):
        pairs.append((f"Summarize findings on: {r['query']}", r["summary"]))

    # Tool-use trajectories: render as Qwen-format per-turn examples.
    traj_pairs = _build_trajectory_pairs(db)
    pairs.extend(traj_pairs)

    # Tokenize into input_ids + completion_start for CE-on-completion.
    out = []
    eos = tokenizer.eos_token_id
    for prompt, completion in pairs:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if eos is not None:
            c_ids = c_ids + [eos]
        ids = (p_ids + c_ids)[:max_seq_len]
        if len(ids) < 8:
            continue
        comp_start = min(len(p_ids), len(ids) - 1)
        out.append({"ids": ids, "comp_start": comp_start})
    return out


def _build_trajectory_pairs(db: DiscoveryDB,
                            min_reward: float = 0.5,
                            max_pairs: int = 200) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs from tool-use trajectories."""
    from forge.self_play.discovery.qwen_adapter import IM_END, qwen_render_messages

    trajectories = db.get_trajectories(min_reward=min_reward, limit=max_pairs)
    pairs = []

    for traj in trajectories:
        messages = traj["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            continue

        for i in range(1, len(messages)):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue

            prompt_msgs = messages[:i]
            prompt = qwen_render_messages(
                prompt_msgs, tools=None, add_generation_prompt=True)

            body_parts = []
            if msg.get("content"):
                body_parts.append(msg["content"])
            if msg.get("tool_calls"):
                from forge.self_play.discovery.chat_template import (
                    render_tool_calls,
                )
                body_parts.append(render_tool_calls(msg["tool_calls"]))
            body = "\n".join(body_parts)

            completion = f"{body}{IM_END}\n"

            if len(prompt) > 10 and len(body) > 2:
                pairs.append((prompt, completion))

    return pairs


def _collate(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to max length in batch. Returns (input_ids, targets, comp_mask)."""
    maxlen = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
        cs = b["comp_start"]
        tgt[i, cs:L] = ids[i, cs:L]
        mask[i, cs:L] = True
    return ids.to(device), tgt.to(device), mask.to(device)


def finetune_from_db(db: DiscoveryDB, base_checkpoint: str | None = None,
                     config: FinetuneConfig | None = None,
                     device: str = "cuda") -> str:
    """Fine-tune a QLoRA adapter on the DB and return the checkpoint path.

    Uses ForgeAI's QLoRA system (IRI-FP4 frozen base + trainable LoRA adapters)
    instead of HuggingFace PEFT. The LoRA adapter is saved separately (not
    merged into base weights) so it can be hot-loaded via engine.load_lora().

    Args:
        db: the discovery database (training source + epoch bookkeeping).
        base_checkpoint: path to the parent checkpoint (None = R30 base).
        config: FinetuneConfig.
    Returns:
        Path to the newly saved LoRA adapter checkpoint.
    """
    from safetensors.torch import save_file as _save

    from forge.engine.forge_engine import ForgeEngine
    from forge.training.bitnet_lora import add_lora_adapters

    cfg = config or FinetuneConfig()

    # Load base model via ForgeEngine (IRI-FP4 quantized, stays packed in VRAM)
    ckpt = base_checkpoint or str(V2_CHECKPOINT.parent / "ForgeLM_V2.safetensors")
    print(f"  [finetune] Loading base: {ckpt}")
    engine = ForgeEngine.from_checkpoint(ckpt, config_name="forgelm_v2",
                                          device=device, auto_activate=False)
    model = engine.model
    model.train()
    tokenizer = engine.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    dataset = build_sft_dataset(db, tokenizer, cfg.max_seq_len)
    if len(dataset) < 8:
        raise ValueError(f"DB too small to fine-tune: only {len(dataset)} usable samples")
    print(f"  [finetune] Dataset: {len(dataset)} samples")

    # ── QLoRA: add LoRA adapters to IRI-FP4 frozen base ──
    n_adapters, lora_params = add_lora_adapters(
        model, rank=cfg.lora_r, alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_targets))
    total_params = sum(p.numel() for p in lora_params)
    print(f"  [finetune] QLoRA: {n_adapters} adapters, rank={cfg.lora_r}, "
          f"{total_params:,} params ({total_params/1e6:.1f}M)")

    # Optimizer (only LoRA params are trainable)
    optimizer = torch.optim.AdamW(lora_params, lr=cfg.max_lr,
                                   weight_decay=cfg.weight_decay,
                                   betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(dataset) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    print(f"  [finetune] {cfg.epochs} epochs, {total_steps} steps")

    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.max_lr * step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return cfg.max_lr * 0.1 ** (progress * 2)

    step = 0
    t0 = time.time()
    for _ep in range(cfg.epochs):
        order = list(range(len(dataset)))
        order.reverse()  # variety without importing random
        optimizer.zero_grad()

        for sp in range(steps_per_epoch):
            # Build batch
            batch_indices = [order[(sp * cfg.batch_size + j) % len(order)]
                            for j in range(cfg.batch_size)]
            batch = [dataset[idx] for idx in batch_indices]
            ids, tgt, _mask = _collate(batch, pad_id, device)

            # Forward (targets passed for chunked CE)
            shift_tgt = tgt[:, 1:].contiguous()
            pad_col = torch.full((shift_tgt.size(0), 1), -100,
                                 dtype=shift_tgt.dtype, device=device)
            shift_tgt = torch.cat([shift_tgt, pad_col], dim=1)
            out = model(ids, targets=shift_tgt)
            loss = out[1] if isinstance(out, tuple) else None
            if loss is None:
                logits = out[0] if isinstance(out, tuple) else out
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.view(-1), ignore_index=-100)

            (loss / cfg.grad_accum).backward()

            if (sp + 1) % cfg.grad_accum == 0:
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"    step {step}/{total_steps} loss={loss.item():.4f} lr={lr:.2e}")

    elapsed = time.time() - t0
    print(f"  [finetune] Done: {step} steps in {elapsed:.1f}s")

    # ── Save LoRA adapter only (base not merged) ──
    _EPOCHS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_num = db.last_epoch_num() + 1
    path = str(_EPOCHS_DIR / f"epoch{epoch_num}_lora.safetensors")
    lora_state = {n: p.cpu() for n, p in model.named_parameters() if "lora_" in n}
    _save(lora_state, path)
    print(f"  [finetune] Saved LoRA: {path} ({len(lora_state)} tensors)")

    db.add_epoch(epoch_num, path, parent_epoch=None, kind="finetune",
                 status="candidate")
    db.emit("finetune_done", {"epoch": epoch_num, "checkpoint": path,
                              "samples": len(dataset), "steps": step})
    return path

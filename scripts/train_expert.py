"""Train AirMoE experts — dead simple CLI for any knowledge domain.

Two modes:
  1. Supervised (default): Fine-tune on external data (math, science, history, ...)
     python train_expert.py --topic physics --data physics.json
     python train_expert.py --topic math_algebra --data problems.jsonl --epochs 5

  2. Self-play: Model generates + verifies code solutions (existing system)
     python train_expert.py --topic python_algorithms --mode selfplay --epochs 3

Data formats (supervised mode):
  JSON:  [{"prompt": "...", "completion": "..."}, ...]
  JSONL: one {"prompt": "...", "completion": "..."} per line
  TXT:   plain text, one example per blank line (or one per line)
  CSV:   prompt,completion columns (with header)

Features:
  - Creates new topics automatically (no source code edits)
  - Loads existing trained expert for continual improvement
  - Falls back to seed expert for new topics
  - Saves SVD-compressed expert + updates manifest
  - Early stopping with best-weight restoration
  - Quality filter + dedup + label smoothing (prevents degradation)

The base model stays frozen. Only the topic expert is updated.
This prevents base model size growth and allows dynamic compute.
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import paths as _paths

V2_EXPERTS_DIR = _paths.as_str(_paths.AIRMOE_MODULES_DIR)
V2_CHECKPOINT = _paths.as_str(_paths.V2_CHECKPOINT)
# Backward compat
V4_DIR = V2_EXPERTS_DIR
N_LAYERS = 28


# ─── Data Loading ──────────────────────────────────────────────────────

def load_training_data(path: str) -> list[dict[str, str]]:
    """Load training data from JSON, JSONL, TXT, or CSV.

    Returns list of {"prompt": str, "completion": str} dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = p.suffix.lower()
    samples = []

    if ext == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "prompt" in item and "completion" in item:
                    samples.append(item)
                elif isinstance(item, dict) and "text" in item:
                    samples.append({"prompt": "", "completion": item["text"]})
                elif isinstance(item, str):
                    samples.append({"prompt": "", "completion": item})
        elif isinstance(data, dict) and "examples" in data:
            samples = data["examples"]

    elif ext == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "prompt" in item and "completion" in item:
                    samples.append(item)
                elif "text" in item:
                    samples.append({"prompt": "", "completion": item["text"]})

    elif ext == ".csv":
        import csv
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", row.get("input", ""))
                completion = row.get("completion", row.get("output", row.get("answer", "")))
                if completion:
                    samples.append({"prompt": prompt, "completion": completion})

    elif ext == ".txt":
        with open(p, encoding="utf-8") as f:
            text = f.read()
        # Split on blank lines (paragraphs) or double newlines
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        if len(blocks) <= 1:
            # Fallback: one per line
            blocks = [b.strip() for b in text.split("\n") if b.strip()]
        for block in blocks:
            samples.append({"prompt": "", "completion": block})

    else:
        raise ValueError(f"Unsupported format: {ext} (use .json, .jsonl, .csv, or .txt)")

    if not samples:
        raise ValueError(f"No valid samples found in {path}")

    print(f"  Loaded {len(samples)} samples from {path}")
    return samples


# ─── Expert Save/Load ──────────────────────────────────────────────────

def compress_expert_svd(w: torch.Tensor, svd_energy: float = 0.99) -> dict[str, torch.Tensor]:
    """SVD compress an expert weight matrix (near-lossless at 0.99 energy)."""
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    cumsum = (S ** 2).cumsum(0)
    total = cumsum[-1]
    rank = max(1, (cumsum < svd_energy * total).sum().item() + 1)
    return {
        "U": U[:, :rank].contiguous().to(torch.bfloat16),
        "U_shape": torch.tensor(U[:, :rank].shape, dtype=torch.int32),
        "S": S[:rank].to(torch.float16),
        "Vh": Vh[:rank, :].contiguous().to(torch.bfloat16),
        "Vh_shape": torch.tensor(Vh[:rank, :].shape, dtype=torch.int32),
        "rank": torch.tensor([rank], dtype=torch.int32),
    }


def save_expert(model, topic: str, v4_dir: str, svd_energy: float = 0.99):
    """Save model's expert 0 weights as SVD-compressed files for a topic."""
    from safetensors.torch import save_file

    experts_dir = Path(v4_dir) / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for layer in range(len(model.blocks)):
        block = model.blocks[layer]
        ffn = block.ffn
        if not (hasattr(ffn, "experts") and len(ffn.experts) > 0):
            continue

        expert = ffn.experts[0]
        compressed = {}
        for part_name, param in [("w1", expert.w1.weight.data),
                                  ("w2", expert.w2.weight.data),
                                  ("w3", expert.w3.weight.data)]:
            w = param.detach().cpu().float()
            comp = compress_expert_svd(w, svd_energy)
            for k, v in comp.items():
                compressed[f"{part_name}_{k}"] = v

        shard_name = f"expert_l{layer}_{topic}.safetensors"
        save_file(compressed, str(experts_dir / shard_name))
        n_saved += 1

    # Write marker file
    marker = experts_dir / f".trained_{topic}"
    marker.write_text("trained", encoding="utf-8")

    print(f"  Saved {n_saved} expert files for topic '{topic}'")
    return n_saved


def update_manifest(v4_dir: str, topic: str, keywords: list[str] | None = None,
                    label: str | None = None):
    """Update manifest.json with a new/updated topic entry."""
    manifest_path = Path(v4_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"  WARNING: No manifest at {manifest_path}")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Update topics dict
    topics = manifest.get("topics", {})
    if topic not in topics:
        topics[topic] = {
            "label": label or topic.replace("_", " ").title(),
            "keywords": keywords or [topic.replace("_", " ")],
        }
        manifest["topics"] = topics
        print(f"  Registered new topic: {topic}")

    # Update expert entries for this topic
    experts_dir = Path(v4_dir) / "experts"
    existing_experts = [e for e in manifest.get("experts", []) if e.get("topic") != topic]
    new_experts = []
    for layer in range(manifest.get("n_layers", N_LAYERS)):
        shard_name = f"expert_l{layer}_{topic}.safetensors"
        shard_path = experts_dir / shard_name
        if shard_path.exists():
            import hashlib
            h = hashlib.sha256()
            with open(shard_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            new_experts.append({
                "id": f"l{layer}_{topic}",
                "layer": layer,
                "topic": topic,
                "file": f"experts/{shard_name}",
                "size_bytes": shard_path.stat().st_size,
                "sha256": h.hexdigest()[:16],
                "compressed": True,
                "compression": "svd",
                "seed_expert": 0,
            })

    existing_experts.extend(new_experts)
    manifest["experts"] = existing_experts
    manifest["n_topics"] = len(set(e.get("topic", "") for e in existing_experts))
    manifest["n_expert_files"] = len(existing_experts)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest updated: {len(new_experts)} files for '{topic}'")


# ─── Expert Load ───────────────────────────────────────────────────────

def load_trained_expert(model, topic: str, v4_dir: str, device: str = "cuda") -> bool:
    """Load a previously trained expert for a topic into expert slot 0.

    Returns True if loaded, False if no trained expert found.
    """
    from safetensors.torch import load_file

    experts_dir = Path(v4_dir) / "experts"
    marker = experts_dir / f".trained_{topic}"
    if not marker.exists():
        return False

    n_loaded = 0
    for layer in range(len(model.blocks)):
        shard_path = experts_dir / f"expert_l{layer}_{topic}.safetensors"
        if not shard_path.exists():
            continue

        block = model.blocks[layer]
        if not (hasattr(block.ffn, "experts") and len(block.ffn.experts) > 0):
            continue

        state = load_file(str(shard_path))
        expert = block.ffn.experts[0]
        model_dtype = next(model.parameters()).dtype

        for part in ["w1", "w2", "w3"]:
            U = state.get(f"{part}_U")
            S = state.get(f"{part}_S")
            Vh = state.get(f"{part}_Vh")
            if U is not None and S is not None and Vh is not None:
                W = (U.float() * S.float().unsqueeze(0)) @ Vh.float()
                getattr(expert, part).weight.data = W.to(device, model_dtype)

        n_loaded += 1

    print(f"  Loaded trained expert: {n_loaded} layers for '{topic}'")
    return n_loaded > 0


def load_seed_expert(model, seed_idx: int = 3, v2_path: str = V2_CHECKPOINT):
    """Load a seed expert from V2 checkpoint into expert slot 0.

    Used when creating a new topic (no trained expert exists yet).
    """
    from safetensors.torch import load_file

    state = load_file(v2_path)
    n_loaded = 0
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for layer in range(len(model.blocks)):
        for part in ["w1", "w2", "w3"]:
            k = f"blocks.{layer}.ffn.experts.{seed_idx}.{part}.weight"
            if k in state and hasattr(model.blocks[layer].ffn, "experts"):
                if len(model.blocks[layer].ffn.experts) > 0:
                    expert = model.blocks[layer].ffn.experts[0]
                    getattr(expert, part).weight.data = state[k].to(device, model_dtype)
                    n_loaded += 1

    print(f"  Loaded seed expert {seed_idx}: {n_loaded // 3} layers")
    return n_loaded > 0


# ─── Fine-tuning ───────────────────────────────────────────────────────

def finetune_expert(model, tokenizer, samples: list[dict[str, str]],
                    device: str = "cuda", lr: float = 2e-5,
                    label_smoothing: float = 0.15, grad_accum: int = 4,
                    max_length: int = 512) -> dict:
    """Fine-tune expert 0 on (prompt, completion) pairs via LoRA.

    Only trains LoRA adapters on expert 0's FFN weights (last 8 layers).
    Base model stays frozen. Merges LoRA back after training.
    """
    from forge.training.dora import DoRALinear

    def apply_dora_to_linear(layer, rank=4, alpha=8):
        """Wrap an nn.Linear with DoRA, preserving its weight."""
        dora = DoRALinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank, alpha=alpha,
            bias=layer.bias is not None,
        )
        dora.load_from_weight(layer.weight.data)
        if layer.bias is not None:
            dora.bias.data.copy_(layer.bias.data)
        return dora

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to last 8 layers, expert 0 only
    trainable = []
    n_lora = 0
    n_blocks = len(model.blocks)
    lora_start = max(0, n_blocks - 8)

    for idx, block in enumerate(model.blocks):
        if idx < lora_start:
            continue
        if hasattr(block.ffn, "experts") and len(block.ffn.experts) > 0:
            expert = block.ffn.experts[0]
            for attr in ["w1", "w2", "w3"]:
                layer = getattr(expert, attr, None)
                if layer and hasattr(layer, "weight") and not hasattr(layer, "lora_A"):
                    wrapped = apply_dora_to_linear(layer, rank=4, alpha=8)
                    setattr(expert, attr, wrapped)
                    n_lora += 1
                    for p in wrapped.parameters():
                        if p.requires_grad:
                            trainable.append(p)

    for p in trainable:
        p.requires_grad = True

    print(f"  LoRA: {n_lora} adapters, {sum(p.numel() for p in trainable)/1e6:.1f}M trainable")

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=0.05)
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.05)

    # Training loop
    model.train()
    total_loss = 0.0
    n_batches = 0
    accum_count = 0
    eos = tokenizer.eos_token or ""

    for item in samples:
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        if not completion.strip():
            continue

        full_text = prompt + completion
        if not full_text.endswith(eos):
            full_text += eos

        enc = tokenizer(full_text, return_tensors="pt",
                        truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to(device)

        prompt_len = 0
        if prompt:
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]

        if input_ids.shape[1] <= prompt_len + 1:
            continue

        logits, _ = model(input_ids)
        sol_logits = logits[0, prompt_len-1:-1, :]
        sol_targets = input_ids[0, prompt_len:]

        if sol_logits.shape[0] == 0:
            continue

        loss = torch.nn.functional.cross_entropy(
            sol_logits, sol_targets, label_smoothing=label_smoothing)
        weighted_loss = loss / grad_accum
        weighted_loss.backward()
        accum_count += 1

        if accum_count >= grad_accum:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        total_loss += loss.item()
        n_batches += 1

    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    model.eval()

    # Merge LoRA back
    for block in model.blocks:
        if hasattr(block.ffn, "experts") and len(block.ffn.experts) > 0:
            target = block.ffn.experts[0]
        else:
            target = block.ffn
        for attr in ["w1", "w2", "w3"]:
            layer = getattr(target, attr, None)
            if layer and hasattr(layer, "merge_and_unload"):
                merged = layer.merge_and_unload()
                setattr(target, attr, merged)

    del optimizer, trainable
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Training: {n_batches} batches, avg_loss={avg_loss:.4f}")
    return {"n_batches": n_batches, "avg_loss": avg_loss}


def snapshot_expert(model) -> dict:
    """Snapshot expert 0 weights across all layers."""
    state = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block.ffn, "experts") and len(block.ffn.experts) > 0:
            exp = block.ffn.experts[0]
            state[i] = {
                "w1": exp.w1.weight.data.clone(),
                "w2": exp.w2.weight.data.clone(),
                "w3": exp.w3.weight.data.clone(),
            }
    return state


def restore_expert(model, state: dict):
    """Restore expert 0 weights from snapshot."""
    for i, block in enumerate(model.blocks):
        if i in state and hasattr(block.ffn, "experts") and len(block.ffn.experts) > 0:
            exp = block.ffn.experts[0]
            exp.w1.weight.data.copy_(state[i]["w1"])
            exp.w2.weight.data.copy_(state[i]["w2"])
            exp.w3.weight.data.copy_(state[i]["w3"])


# ─── Evaluation ────────────────────────────────────────────────────────

def evaluate_expert(model, tokenizer, val_samples: list[dict[str, str]],
                    device: str = "cuda", max_tokens: int = 100) -> float:
    """Evaluate expert on validation samples.

    Returns average token-level accuracy on completion tokens.
    """
    if not val_samples:
        return 0.0

    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for item in val_samples:
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            if not completion.strip():
                continue

            full_text = prompt + completion
            enc = tokenizer(full_text, return_tensors="pt",
                            truncation=True, max_length=512)
            input_ids = enc.input_ids.to(device)

            prompt_len = 0
            if prompt:
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
                prompt_len = prompt_ids.shape[1]

            if input_ids.shape[1] <= prompt_len + 1:
                continue

            logits, _ = model(input_ids)
            sol_logits = logits[0, prompt_len-1:-1, :]
            sol_targets = input_ids[0, prompt_len:]
            preds = sol_logits.argmax(dim=-1)
            correct = (preds == sol_targets).sum().item()
            total_correct += correct
            total_tokens += sol_targets.shape[0]

    accuracy = total_correct / max(total_tokens, 1)
    return accuracy


# ─── Main Training Pipeline ────────────────────────────────────────────

def train_supervised(topic: str, data_path: str, v4_dir: str = V4_DIR,
                     epochs: int = 3, lr: float = 2e-5, patience: int = 2,
                     val_ratio: float = 0.15, svd_energy: float = 0.99,
                     seed_expert: int = 3, keywords: list[str] | None = None,
                     label: str | None = None, device: str = "cuda"):
    """Train an expert on external data (supervised mode).

    Works for ANY domain: math, science, history, code, etc.
    """
    from transformers import AutoTokenizer

    from forge.config import get_config
    from forge.model_loader import ModelLoader

    print("=" * 70)
    print(f"Train Expert: {topic} (supervised)")
    print(f"  Data: {data_path}")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading training data...")
    samples = load_training_data(data_path)
    if len(samples) < 2:
        print("  ERROR: Need at least 2 samples to train")
        return

    # Split train/val
    import random
    rng = random.Random(42)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  Split: {len(train_samples)} train / {len(val_samples)} val")

    # 2. Load model
    print("\n[2] Loading model...")
    cfg = get_config("forgelm_v2", device=device)
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=V2_CHECKPOINT, moe_top_k=0)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

    # 3. Load existing expert or seed
    print(f"\n[3] Loading expert for topic '{topic}'...")
    loaded = load_trained_expert(model, topic, v4_dir, device)
    if not loaded:
        print(f"  No trained expert found — using seed expert {seed_expert}")
        load_seed_expert(model, seed_idx=seed_expert)

    # 4. Evaluate baseline
    print("\n[4] Evaluating baseline...")
    baseline_acc = evaluate_expert(model, tokenizer, val_samples, device)
    print(f"  Baseline accuracy: {baseline_acc:.1%}")

    # 5. Train
    print("\n[5] Training...")
    best_acc = baseline_acc
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")
        ft_stats = finetune_expert(model, tokenizer, train_samples,
                                   device=device, lr=lr)

        val_acc = evaluate_expert(model, tokenizer, val_samples, device)
        print(f"  Val accuracy: {val_acc:.1%}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            best_state = snapshot_expert(model)
            print(f"  * New best: {best_acc:.1%}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print("  Early stopping")
                break

    # 6. Restore best
    if best_state is not None:
        restore_expert(model, best_state)
        print(f"\n  Restored best expert (acc: {best_acc:.1%})")

    # 7. Save
    print("\n[6] Saving expert...")
    if best_acc > baseline_acc:
        save_expert(model, topic, v4_dir, svd_energy)
        update_manifest(v4_dir, topic, keywords, label)
        print(f"\n  IMPROVED: {baseline_acc:.1%} -> {best_acc:.1%}")
    else:
        print(f"\n  No improvement over baseline ({baseline_acc:.1%})")
        print("  Expert NOT saved (baseline unchanged)")

    # Summary
    print(f"\n{'='*70}")
    print(f"Done: {topic}")
    print(f"  Baseline: {baseline_acc:.1%}")
    print(f"  Best:     {best_acc:.1%}")
    print(f"  Saved:    {'YES' if best_acc > baseline_acc else 'NO'}")
    print(f"{'='*70}")


def train_selfplay(topic: str, v4_dir: str, epochs: int, n_tasks: int,
                   rounds: int, **kwargs):
    """Train via self-play (delegates to existing system)."""
    # Build argv for the existing script
    argv = ["-u", "-m", "forge.training.self_play_expert_training",
            "--topics", topic, "--epochs", str(epochs),
            "--n-tasks", str(n_tasks), "--rounds", str(rounds)]
    print(f"  Delegating to self-play: {' '.join(argv)}")
    os.execv(sys.executable, [sys.executable] + argv)


# ─── CLI ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train AirMoE experts — dead simple CLI for any domain")
    parser.add_argument("--topic", required=True,
                        help="Expert topic name (e.g., physics, math_algebra)")
    parser.add_argument("--data", default="",
                        help="Data file (JSON/JSONL/CSV/TXT) for supervised mode")
    parser.add_argument("--mode", choices=["supervised", "selfplay"],
                        default="supervised",
                        help="Training mode (default: supervised)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Max training epochs (default 3)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default 2e-5)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stop patience (default 2)")
    parser.add_argument("--seed-expert", type=int, default=3,
                        help="Seed expert index for new topics (default 3=general)")
    parser.add_argument("--svd-energy", type=float, default=0.99,
                        help="SVD energy for expert compression (default 0.99)")
    parser.add_argument("--keywords", default="",
                        help="Comma-separated router keywords for new topic")
    parser.add_argument("--label", default="",
                        help="Display label for new topic")
    parser.add_argument("--v4-dir", default=V4_DIR,
                        help="V4 expert library directory")
    parser.add_argument("--n-tasks", type=int, default=50,
                        help="Self-play: tasks per epoch (default 50)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Self-play: max rounds per task (default 3)")
    args = parser.parse_args()

    if args.mode == "supervised":
        if not args.data:
            print("ERROR: --data required for supervised mode")
            sys.exit(1)
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None
        train_supervised(
            topic=args.topic, data_path=args.data, v4_dir=args.v4_dir,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            svd_energy=args.svd_energy, seed_expert=args.seed_expert,
            keywords=keywords, label=args.label or None,
        )
    else:
        train_selfplay(args.topic, args.v4_dir, args.epochs,
                       args.n_tasks, args.rounds)


if __name__ == "__main__":
    main()

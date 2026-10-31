"""bake.py — embed improvements into the .safetensors without training.

1. bake_task_vector: offline task arithmetic — add scale*(finetuned - base)
   onto an *arbitrary target* checkpoint (e.g. the self-play checkpoint).
2. extract_distill_dataset: context distillation — from self-play packet
   logs keep only (problem, final correct answer) pairs for a low-epoch SFT.
3. fuse_lora: offline LoRA fusion — fold a PEFT adapter into base weights,
   producing a standalone checkpoint identical in size to the original.

CLI:
    python -m research.training_free.bake task-vector --base B --delta F --target T --alpha 0.8 --out O
    python -m research.training_free.bake distill --packets p.jsonl --out d.jsonl --min-score 0.7
    python -m research.training_free.bake fuse-lora --base B.safetensors --adapter lora_dir --out O.safetensors
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from forge.checkpoint_io import load_checkpoint, save_checkpoint
from research.merge_models import _load_state_dict, _task_vectors, merge_task_arith

# ---------------------------------------------------------------------------
# 1. Task arithmetic baking
# ---------------------------------------------------------------------------

def bake_task_vector(
    target_path: str,
    finetuned_path: str,
    base_path: str,
    alpha: float = 1.0,
    out_path: str | None = None,
) -> str:
    """Add the task vector (finetuned - base) * alpha onto target weights.

    Unlike merge_models' task_arith (which merges into the *base*), this
    applies the capability delta to an arbitrary target model — typically the
    current self-play checkpoint. Pure offline tensor arithmetic, no training.

    Args:
        target_path: checkpoint that receives the capability (safetensors).
        finetuned_path: model fine-tuned for the capability.
        base_path: the pre-fine-tuning base model.
        alpha: task vector strength (0.5-1.5 typical; 0 = no-op).
        out_path: output path (default: <target stem>.baked<alpha>.safetensors).

    Returns:
        Path written.
    """
    alpha = float(alpha)
    target = _load_state_dict(target_path)
    finetuned = _load_state_dict(finetuned_path)
    base = _load_state_dict(base_path)

    tv = _task_vectors(finetuned, base)
    if not tv:
        raise ValueError("No overlapping tensors between finetuned and base.")

    merged = merge_task_arith(target, [tv], [alpha])
    # merge_task_arith casts to target dtype; keep original meta.
    state = load_checkpoint(target_path, map_location="cpu")
    state.update(merged)

    if out_path is None:
        stem, ext = os.path.splitext(target_path)
        out_path = f"{stem}.baked{alpha}{ext}"

    save_checkpoint(state, out_path)
    n = len(merged)
    print(f"Baked {n} tensors: {target_path} + {alpha}*(finetuned - base) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. Context distillation
# ---------------------------------------------------------------------------

def extract_distill_dataset(
    packets_paths: list[str],
    out_jsonl: str,
    min_score: float = 0.0,
    require_correct: bool = True,
    max_examples: int = 0,
) -> int:
    """Build an SFT dataset from self-play packets (context distillation).

    Self-play produced the perfect solutions using heavy context (Reflexion
    critiques, rewinds, corrections). We discard all of that and keep only
    (task -> final correct answer) pairs; a short, low-epoch SFT pass then
    moves the knowledge from the context window into the weights.

    Args:
        packets_paths: JSONL files of SelfPlaySandbox data packets.
        out_jsonl: destination (sft_train JSONL format: prompt/response).
        min_score: drop packets with quality_score below this.
        require_correct: only keep packets whose output matched expected.
        max_examples: cap on exported examples (0 = unlimited).

    Returns:
        Number of examples written.
    """
    written = 0
    seen = set()
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for path in packets_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pkt = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if max_examples and written >= max_examples:
                        return written
                    exec_r = pkt.get("execution") or {}
                    if pkt.get("quality_score", 0.0) < min_score:
                        continue
                    if require_correct and not exec_r.get(
                            "output_matches_expected", False):
                        continue
                    if exec_r.get("returncode", -1) != 0:
                        continue
                    task = (pkt.get("prompt") or pkt.get("task") or "").strip()
                    code = (pkt.get("generated_code") or "").strip()
                    if not task or not code:
                        continue
                    key = (task, code)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(json.dumps({
                        "prompt": task,
                        "response": code,
                    }, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Distilled {written} (task, answer) pairs -> {out_jsonl}")
    return written


# ---------------------------------------------------------------------------
# 3. Offline LoRA fusion
# ---------------------------------------------------------------------------

def _strip_peft_prefix(key: str) -> str:
    """'base_model.model.blocks.0.attn.q_proj.lora_A.weight' ->
    'blocks.0.attn.q_proj.lora_A.weight' (works with/without prefix)."""
    prefix = "base_model.model."
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def fuse_lora(
    base_path: str,
    adapter_dir: str,
    out_path: str | None = None,
    alpha_override: float | None = None,
) -> str:
    """Fold a PEFT LoRA adapter into base weights offline.

    W' = W + (lora_alpha / r) * (B @ A)   [standard PEFT scaling]

    The adapter directory must contain adapter_config.json +
    adapter_model.safetensors (as saved by peft). The result is a standalone
    checkpoint identical in size to the base — no PEFT dependency at inference.

    Args:
        base_path: base model checkpoint (safetensors).
        adapter_dir: directory with the PEFT adapter files.
        out_path: output path (default: <base stem>.lora_fused.safetensors).
        alpha_override: override lora_alpha from adapter_config.json.

    Returns:
        Path written.
    """
    from safetensors import safe_open

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path}")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Missing {adapter_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    r = int(cfg.get("r", 16))
    lora_alpha = float(alpha_override if alpha_override is not None
                       else cfg.get("lora_alpha", 2 * r))
    scale = lora_alpha / r

    base = _load_state_dict(base_path)
    adapters: dict[str, torch.Tensor] = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            adapters[key] = f.get_tensor(key)

    # Group lora_A/lora_B by target module key.
    a_pairs: dict[str, torch.Tensor] = {}
    b_pairs: dict[str, torch.Tensor] = {}
    for key, t in adapters.items():
        k = _strip_peft_prefix(key)
        if k.endswith(".lora_A.weight"):
            a_pairs[k[: -len(".lora_A.weight")]] = t
        elif k.endswith(".lora_B.weight"):
            b_pairs[k[: -len(".lora_B.weight")]] = t

    if not a_pairs:
        raise ValueError(
            "No lora_A/lora_B tensors found in adapter (not a PEFT LoRA?).")

    n_fused = 0
    for module_key, a in a_pairs.items():
        b = b_pairs.get(module_key)
        if b is None:
            continue
        w_key = module_key + ".weight"
        if w_key not in base:
            continue  # target module not present in this architecture
        delta = (b.to(a.dtype) @ a) * scale  # (out, in)
        w = base[w_key]
        if w.shape != delta.shape:
            # Bias/embedding guards: skip incompatible shapes.
            continue
        base[w_key] = (w.float() + delta.float()).to(w.dtype)
        n_fused += 1

    if n_fused == 0:
        raise ValueError("No adapter modules matched base checkpoint keys.")

    state = load_checkpoint(base_path, map_location="cpu")
    state.update(base)

    if out_path is None:
        stem, ext = os.path.splitext(base_path)
        out_path = f"{stem}.lora_fused{ext}"

    save_checkpoint(state, out_path)
    print(f"Fused {n_fused} LoRA modules (r={r}, alpha={lora_alpha}) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(
        prog="research.training_free.bake",
        description="Bake improvements into a .safetensors without training.")
    sub = p.add_subparsers(dest="cmd", required=True)

    tv = sub.add_parser("task-vector", help="Add (finetuned-base)*alpha onto target")
    tv.add_argument("--base", required=True, help="pre-finetuning base model")
    tv.add_argument("--delta", required=True, help="finetuned model (capability source)")
    tv.add_argument("--target", required=True, help="checkpoint to bake into")
    tv.add_argument("--alpha", type=float, default=1.0, help="task vector strength")
    tv.add_argument("--out", default=None, help="output path")

    ds = sub.add_parser("distill", help="Extract (task, answer) SFT data from packets")
    ds.add_argument("--packets", nargs="+", required=True, help="self-play packet JSONL files")
    ds.add_argument("--out", required=True, help="output JSONL")
    ds.add_argument("--min-score", type=float, default=0.0, help="quality_score floor")
    ds.add_argument("--no-require-correct", action="store_true",
                    help="keep packets even if output did not match expected")
    ds.add_argument("--max-examples", type=int, default=0, help="cap (0 = unlimited)")

    lf = sub.add_parser("fuse-lora", help="Fold a PEFT LoRA adapter into base weights")
    lf.add_argument("--base", required=True, help="base checkpoint (safetensors)")
    lf.add_argument("--adapter", required=True, help="PEFT adapter directory")
    lf.add_argument("--out", default=None, help="output path")
    lf.add_argument("--alpha", type=float, default=None, help="override lora_alpha")

    args = p.parse_args()
    if args.cmd == "task-vector":
        bake_task_vector(args.target, args.delta, args.base,
                         alpha=args.alpha, out_path=args.out)
    elif args.cmd == "distill":
        extract_distill_dataset(
            args.packets, args.out, min_score=args.min_score,
            require_correct=not args.no_require_correct,
            max_examples=args.max_examples)
    elif args.cmd == "fuse-lora":
        fuse_lora(args.base, args.adapter, out_path=args.out,
                  alpha_override=args.alpha)


if __name__ == "__main__":
    _cli()

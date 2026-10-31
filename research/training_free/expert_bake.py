"""expert_bake.py — AirMoE expert → dense consolidation via task arithmetic.

AirMoE (forge/moe/) trains isolated topic experts as standalone per-layer
FFN files (expert_l{layer}_{topic}.safetensors, keys w1/w2/w3, optionally
SVD/INT4 compressed) and hotswaps them from disk at runtime.

This module does the *static offline* equivalent: instead of keeping the
expert as a fragmented, router-dependent module, its task vector
(expert − base FFN) is added directly into the dense FFN matrices of a
target checkpoint. The capability is locked into the .safetensors file —
zero inference overhead, zero router cost, zero disk I/O at generation time.

Usage:
    python -m research.training_free.expert_bake \
        --target model.safetensors \
        --expert experts/expert_l0_math.safetensors \
        --alpha 0.8 --out consolidated.safetensors
"""
from __future__ import annotations

import argparse
import os
import re

import torch

from forge.checkpoint_io import load_checkpoint, save_checkpoint

# Expert part -> dense FFN parameter name in the base model.
_PART_MAP = {"w1": "w_gate", "w2": "w_up", "w3": "w_down"}
_RE_EXPERT_LAYER = re.compile(r"expert_l(\d+)_")


# ---------------------------------------------------------------------------
# Expert file decoding (mirrors forge/moe/airmoe_infinite.py formats)
# ---------------------------------------------------------------------------

def decompress_expert(state: dict[str, torch.Tensor],
                      device: str = "cpu") -> dict[str, torch.Tensor]:
    """Decode an expert state dict to full {w1, w2, w3} weight matrices.

    Handles three on-disk formats:
      1. Raw SwiGLU:   w1.weight / w2.weight / w3.weight
      2. SVD-only:     <part>_U, <part>_S, <part>_Vh   (W = U·S·Vh)
      3. SVD + INT4:   <part>_U_q, <part>_U_scale, <part>_S,
                       <part>_Vh_q, <part>_Vh_scale, <part>_U_shape,
                       <part>_Vh_shape (per-group dequant, group=128)
    LatentMoE (up/down) experts operate in a latent space and cannot be
    folded into dense SwiGLU FFNs — they are skipped with a warning.
    """
    out: dict[str, torch.Tensor] = {}

    # Raw format.
    if "w1.weight" in state or "w2.weight" in state or "w3.weight" in state:
        for part in ("w1", "w2", "w3"):
            k = f"{part}.weight"
            if k in state:
                out[part] = state[k].to(device)

    # SVD formats.
    for part in ("w1", "w2", "w3"):
        if f"{part}_U" in state and f"{part}_S" in state and f"{part}_Vh" in state:
            U = state[f"{part}_U"].float()
            S = state[f"{part}_S"].float()
            Vh = state[f"{part}_Vh"].float()
            # Guard against full_matrices=True SVDs (U/Vh larger than k).
            k = S.numel()
            U = U[:, :k] if U.shape[1] > k else U
            Vh = Vh[:k] if Vh.shape[0] > k else Vh
            out[part] = (U * S.unsqueeze(0)) @ Vh
        elif (f"{part}_U_q" in state and f"{part}_S" in state
              and f"{part}_Vh_q" in state):
            U_q, U_scale = state[f"{part}_U_q"], state[f"{part}_U_scale"]
            Vh_q, Vh_scale = state[f"{part}_Vh_q"], state[f"{part}_Vh_scale"]
            U_shape = state.get(f"{part}_U_shape")
            Vh_shape = state.get(f"{part}_Vh_shape")
            U = (U_q.float() * U_scale.float().unsqueeze(-1)).reshape(-1)
            Vh = (Vh_q.float() * Vh_scale.float().unsqueeze(-1)).reshape(-1)
            if U_shape is not None:
                U = U[: int(U_shape[0]) * int(U_shape[1])].reshape(
                    int(U_shape[0]), int(U_shape[1]))
            if Vh_shape is not None:
                Vh = Vh[: int(Vh_shape[0]) * int(Vh_shape[1])].reshape(
                    int(Vh_shape[0]), int(Vh_shape[1]))
            S = state[f"{part}_S"].float()
            k = S.numel()
            U = U[:, :k] if U.shape[1] > k else U
            Vh = Vh[:k] if Vh.shape[0] > k else Vh
            out[part] = (U * S.unsqueeze(0)) @ Vh

    if not out and any(k.startswith(("up.", "down.")) for k in state):
        print("  [ExpertBake] Skipping LatentMoE-format expert "
              "(latent-space, not foldable into dense SwiGLU FFN)")
    return out


def _layer_from_filename(path: str) -> int | None:
    m = _RE_EXPERT_LAYER.search(os.path.basename(path))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def bake_expert(
    target_path: str,
    expert_paths: list[str],
    alpha: float = 1.0,
    layers: list[int] | None = None,
    out_path: str | None = None,
) -> str:
    """Fold AirMoE expert task vectors into the dense FFN of a target model.

    For each expert: delta_part = expert_part − base_ffn_part (per layer),
    then target += alpha * mean_delta. Multiple experts for the same layer
    are averaged (model-soup of task vectors) to avoid magnitude blowup.

    Args:
        target_path: checkpoint to bake into (safetensors).
        expert_paths: expert_l{layer}_{topic}.safetensors files.
        alpha: overall task-vector strength.
        layers: optional per-expert target layer (default: parsed from
            filenames; must match len(expert_paths) when given).
        out_path: output path (default: <target stem>.expert_baked.safetensors).

    Returns:
        Path written.
    """
    alpha = float(alpha)
    if layers is not None and len(layers) != len(expert_paths):
        raise ValueError("layers must match len(expert_paths)")

    target = load_checkpoint(target_path, map_location="cpu")
    base_ffn: dict[int, dict[str, torch.Tensor]] = {}

    # Per-expert deltas grouped by layer.
    deltas: dict[int, list[dict[str, torch.Tensor]]] = {}
    for i, ep in enumerate(expert_paths):
        layer = layers[i] if layers is not None else _layer_from_filename(ep)
        if layer is None:
            raise ValueError(
                f"Cannot infer layer from {ep}; pass --layers explicitly")
        state = load_checkpoint(ep, map_location="cpu")
        expert = decompress_expert(state)
        if not expert:
            continue

        if layer not in base_ffn:
            base_ffn[layer] = {}
            for part, pname in _PART_MAP.items():
                k = f"blocks.{layer}.ffn.{pname}.weight"
                if k in target:
                    base_ffn[layer][part] = target[k]

        delta: dict[str, torch.Tensor] = {}
        for part, pname in _PART_MAP.items():
            if part not in expert:
                continue
            base_w = base_ffn[layer].get(part)
            if base_w is None or base_w.shape != expert[part].shape:
                continue
            delta[pname] = (expert[part].float() - base_w.float())
        if delta:
            deltas.setdefault(layer, []).append(delta)

    if not deltas:
        raise ValueError("No foldable expert weights matched the target FFN.")

    # Apply: target += alpha * mean(delta per layer).
    n_folded = 0
    for layer, dlist in deltas.items():
        for pname in _PART_MAP.values():
            parts = [d[pname] for d in dlist if pname in d]
            if not parts:
                continue
            mean_delta = torch.stack(parts).mean(dim=0)
            k = f"blocks.{layer}.ffn.{pname}.weight"
            w = target[k]
            target[k] = (w.float() + alpha * mean_delta).to(w.dtype)
            n_folded += 1

    if out_path is None:
        stem, ext = os.path.splitext(target_path)
        out_path = f"{stem}.expert_baked{ext}"
    save_checkpoint(target, out_path)
    print(f"[ExpertBake] Folded {len(deltas)} layer(s) / {n_folded} tensors "
          f"(alpha={alpha}) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(
        prog="research.training_free.expert_bake",
        description="Consolidate AirMoE experts into dense FFN weights "
                    "(task arithmetic, no training).")
    p.add_argument("--target", required=True, help="base/target checkpoint")
    p.add_argument("--expert", nargs="+", required=True,
                   help="expert_l{layer}_{topic}.safetensors file(s)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="task vector strength (default 1.0)")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="per-expert target layer (default: from filenames)")
    p.add_argument("--out", default=None, help="output path")
    args = p.parse_args()

    bake_expert(args.target, args.expert, alpha=args.alpha,
                layers=args.layers, out_path=args.out)


if __name__ == "__main__":
    _cli()

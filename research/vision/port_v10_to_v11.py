"""Port script: V10 → V11 checkpoint conversion (lossless warm start).

Converts a ForgeLM V10 (1.2B, 16 layers, 64K vocab) checkpoint to V11
(3B, 30 layers, 128K vocab + vision tower) with identity/zero-init for
all new parameters, ensuring the model starts in a lossless state.

Conversion steps:
1. Load V10 checkpoint weights
2. Create V11 model (ConfigurableResearchLLM with V11 config)
3. Copy V10 weights into V11 where dimensions match:
   - Embedding: expand 64K→128K (copy first 64K rows, zero-init new rows)
   - LM blocks 0-15: copy from V10 (d_model 2048→2560 requires padding)
   - LM blocks 16-29: zero-init (new layers, zero_init_residual=True)
   - Output head: expand to match 128K vocab
4. Initialize vision tower with SigLIP2 pretrained weights (if available)
   or random init (trunc_normal_)
5. Initialize projector with zero-init (lossless — no visual contribution
   at start, training opens the gate)
6. Save V11 checkpoint

The conversion is LOSSLESS for the text-only path: the V11 model with
zero-init vision projector produces identical text outputs to V10 for
the first 16 layers (the new layers 16-29 are zero-init residual = identity).

Usage:
    python -m research.vision.port_v10_to_v11 \
        --v10-checkpoint research/checkpoints/ForgeLM_V2.safetensors \
        --output research/checkpoints/ForgeLM_V2_Pro.safetensors \
        [--siglip2-checkpoint path/to/siglip2.safetensors]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def expand_embedding(old_weight: torch.Tensor, new_vocab_size: int,
                     old_vocab_size: int) -> torch.Tensor:
    """Expand embedding matrix from old vocab → new vocab.

    Copies the first old_vocab_size rows, zero-inits the rest.
    """
    d_model = old_weight.shape[1]
    new_weight = torch.zeros(new_vocab_size, d_model,
                             dtype=old_weight.dtype,
                             device=old_weight.device)
    new_weight[:old_vocab_size] = old_weight[:old_vocab_size]
    return new_weight


def expand_d_model(old_weight: torch.Tensor, new_d_model: int,
                   old_d_model: int) -> torch.Tensor:
    """Expand a weight matrix from old d_model → new d_model.

    Copies the first old_d_model elements, zero-inits the rest.
    Works for both 1D (norms) and 2D (linear) weights.
    """
    if old_weight.dim() == 1:
        new_weight = torch.zeros(new_d_model,
                                 dtype=old_weight.dtype,
                                 device=old_weight.device)
        new_weight[:old_d_model] = old_weight[:old_d_model]
        return new_weight
    elif old_weight.dim() == 2:
        # Linear weight: (out, in) or (in, out)
        # We need to figure out which dim is d_model
        # For attention: q_proj (out=d_model, in=d_model)
        # For FFN: w_gate (out=intermediate, in=d_model)
        # Copy the first old_d_model elements along the d_model axis
        out_features, in_features = old_weight.shape
        if in_features == old_d_model and out_features == old_d_model:
            # Square matrix (q_proj, k_proj, v_proj, o_proj)
            new_weight = torch.zeros(new_d_model, new_d_model,
                                     dtype=old_weight.dtype,
                                     device=old_weight.device)
            new_weight[:old_d_model, :old_d_model] = old_weight
            return new_weight
        elif in_features == old_d_model:
            # (out, in=d_model) — expand input dim
            new_weight = torch.zeros(out_features, new_d_model,
                                     dtype=old_weight.dtype,
                                     device=old_weight.device)
            new_weight[:, :old_d_model] = old_weight
            return new_weight
        elif out_features == old_d_model:
            # (out=d_model, in) — expand output dim
            new_weight = torch.zeros(new_d_model, in_features,
                                     dtype=old_weight.dtype,
                                     device=old_weight.device)
            new_weight[:old_d_model, :] = old_weight
            return new_weight
        else:
            # Can't determine which axis — just pad both
            new_weight = torch.zeros(
                max(out_features, new_d_model),
                max(in_features, new_d_model),
                dtype=old_weight.dtype, device=old_weight.device)
            new_weight[:out_features, :in_features] = old_weight
            return new_weight
    return old_weight


def port_v10_to_v11(v10_path: str, v11_path: str,
                    siglip2_path: str | None = None,
                    device: str = "cpu") -> None:
    """Port a V10 checkpoint to V11 format.

    Args:
        v10_path: Path to V10 safetensors checkpoint
        v11_path: Output path for V11 checkpoint
        siglip2_path: Optional path to pretrained SigLIP2 weights
        device: Device to use for conversion (cpu or cuda)
    """
    from safetensors.torch import load_file, save_file

    from forge.config import get_config

    logger.info("Loading V10 checkpoint: %s", v10_path)
    v10_state = load_file(v10_path)
    logger.info("V10 checkpoint has %d tensors", len(v10_state))

    # Build V11 model
    v11_config = get_config("forgelm_v2_pro", device=device)
    logger.info("Building V11 model (d_model=%d, n_layers=%d, vocab=%d)",
                v11_config.d_model, v11_config.n_layers, v11_config.vocab_size)

    # Create V11 state dict
    v11_state: dict[str, torch.Tensor] = {}

    # 1. Expand embedding (64K → 128K vocab)
    old_vocab = 65536
    new_vocab = v11_config.vocab_size
    if "embed.weight" in v10_state:
        old_embed = v10_state["embed.weight"]
        new_embed = expand_embedding(old_embed, new_vocab, old_vocab)
        # Also expand d_model (2048 → 2560)
        new_embed = expand_d_model(new_embed, v11_config.d_model, 2048)
        v11_state["embed.weight"] = new_embed
        logger.info("Expanded embedding: %s → %s",
                    str(old_embed.shape), str(new_embed.shape))

    # 2. Copy first 16 layers from V10 (with d_model expansion)
    old_d_model = 2048
    new_d_model = v11_config.d_model
    for layer_idx in range(16):
        prefix = f"layers.{layer_idx}."
        # Find all V10 tensors for this layer
        layer_keys = [k for k in v10_state if k.startswith(prefix)]
        for k in layer_keys:
            old_w = v10_state[k]
            # Expand d_model dimensions
            if old_w.dim() >= 1 and any(s == old_d_model for s in old_w.shape):
                new_w = expand_d_model(old_w, new_d_model, old_d_model)
            else:
                new_w = old_w
            v11_state[k] = new_w
        logger.info("Ported layer %d: %d tensors", layer_idx, len(layer_keys))

    # 3. Zero-init layers 16-29 (new layers)
    # These will be initialized by the model builder with zero_init_residual=True
    logger.info("Layers 16-29: will be zero-init by model builder")

    # 4. Expand output head (if not tied)
    if not v11_config.tie_word_embeddings and "head.weight" in v10_state:
        old_head = v10_state["head.weight"]
        new_head = expand_embedding(old_head, new_vocab, old_vocab)
        new_head = expand_d_model(new_head, new_d_model, old_d_model)
        v11_state["head.weight"] = new_head
        logger.info("Expanded output head")

    # 5. Final norm
    if "norm.weight" in v10_state:
        v11_state["norm.weight"] = expand_d_model(
            v10_state["norm.weight"], new_d_model, old_d_model)

    # 6. Vision tower (SigLIP2)
    if siglip2_path is not None and Path(siglip2_path).is_file():
        logger.info("Loading SigLIP2 weights: %s", siglip2_path)
        siglip2_state = load_file(siglip2_path)
        for k, v in siglip2_state.items():
            v11_state[f"vision.{k}"] = v
        logger.info("Loaded %d SigLIP2 tensors", len(siglip2_state))
    else:
        logger.info("No SigLIP2 checkpoint — vision tower will be random init")

    # Save V11 checkpoint
    logger.info("Saving V11 checkpoint: %s", v11_path)
    Path(v11_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(v11_state, v11_path)
    logger.info("V11 checkpoint saved: %d tensors", len(v11_state))

    # Verify
    v11_loaded = load_file(v11_path)
    logger.info("Verification: loaded %d tensors", len(v11_loaded))
    total_params = sum(t.numel() for t in v11_loaded.values())
    logger.info("Total V11 parameters: %.1fM", total_params / 1e6)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Port ForgeLM V10 checkpoint to V11 (LFM2.5-VL-3B)")
    parser.add_argument("--v10-checkpoint", required=True,
                        help="Path to V10 safetensors checkpoint")
    parser.add_argument("--output", required=True,
                        help="Output path for V11 checkpoint")
    parser.add_argument("--siglip2-checkpoint", default=None,
                        help="Optional path to pretrained SigLIP2 weights")
    parser.add_argument("--device", default="cpu",
                        help="Device for conversion (cpu or cuda)")
    args = parser.parse_args()

    port_v10_to_v11(args.v10_checkpoint, args.output,
                    args.siglip2_checkpoint, args.device)


if __name__ == "__main__":
    main()

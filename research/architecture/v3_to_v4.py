"""Convert a trained V3 checkpoint (Differential Attention) to V4 (GTA).

Pipeline:
  1. Reverse DifferentialAttentionKey: average doubled q/k rows → GQA shapes,
     remove lambda_param and diff-specific rms_norm.
  2. Forward GTAKey: set v_proj = k_proj (V=K warm start), add v_mix_gate=0.
  3. Keep all other weights unchanged (BitNet qscale, TITAN, MoD, MHC, AttnRes).

The result is a V4 checkpoint that loads bit-exact into forgelm_v7 config.

Auto-detects n_heads, n_kv_heads, head_dim, and attention layers from
checkpoint shapes — no hardcoded params needed.

CLI:
  python -m research.architecture.v3_to_v4 \
    --input research/checkpoints/forgelm_v7_SFT.safetensors \
    --output research/checkpoints/forgelm_v7.safetensors

Programmatic (used by model_loader for auto-conversion at load time):
  from research.architecture.v3_to_v4 import convert_v3_to_v4_state
  state = convert_v3_to_v4_state(state)  # in-place on a state dict
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from safetensors.torch import load_file, save_file


def _detect_params(state: dict[str, torch.Tensor]) -> dict:
    """Auto-detect n_heads, n_kv_heads, head_dim, d_model from checkpoint shapes.

    V3 diff checkpoint has:
      - q_proj.weight: [2*n_heads*head_dim, d_model]
      - k_proj.weight: [2*n_kv_heads*head_dim, d_model]
      - v_proj.weight: [n_kv_heads*head_dim, d_model]  (v is not doubled in diff)

    Falls back to detecting from o_proj if q/k are ambiguous.
    """
    # Find first q_proj to get d_model and q rows
    d_model = None
    q_rows = None
    k_rows = None
    v_rows = None

    for k, v in state.items():
        if ".attn.q_proj.weight" in k:
            q_rows = v.shape[0]
            d_model = v.shape[1]
            break
    for k, v in state.items():
        if ".attn.k_proj.weight" in k:
            k_rows = v.shape[0]
            break
    for k, v in state.items():
        if ".attn.v_proj.weight" in k:
            v_rows = v.shape[0]
            break

    if d_model is None or q_rows is None:
        raise ValueError("Cannot detect params: no attn.q_proj.weight found")

    # Diff doubles q and k rows: q_rows = 2*n_heads*head_dim
    # If q_rows == d_model, it's already GQA (not diff) — head_dim = d_model / n_heads
    # If q_rows == 2*d_model, it's diff with n_heads*head_dim = d_model
    is_diff = q_rows == 2 * d_model

    if is_diff:
        # q_rows = 2 * n_heads * head_dim = 2 * d_model
        # So n_heads * head_dim = d_model
        # head_dim = d_model / n_heads — need to figure out n_heads
        # k_rows = 2 * n_kv_heads * head_dim
        # v_rows = n_kv_heads * head_dim (v not doubled in diff)
        # So k_rows / v_rows = 2 → confirms diff
        # n_kv_heads * head_dim = v_rows
        # n_heads * head_dim = d_model
        # Common configs: n_heads=32, head_dim=64, d_model=2048
        # Try to infer head_dim from v_rows and common n_kv_heads values
        if v_rows is not None and k_rows is not None:
            # n_kv_heads * head_dim = v_rows
            # Try common n_kv_heads: 2, 4, 8, 16
            for nkv in (8, 4, 2, 16, 1):
                hd = v_rows // nkv
                if hd > 0 and nkv * hd == v_rows:
                    # Check: n_heads * hd = d_model
                    nh = d_model // hd
                    if nh * hd == d_model:
                        # Verify k_rows = 2 * nkv * hd
                        if k_rows == 2 * nkv * hd:
                            return {
                                "n_heads": nh,
                                "n_kv_heads": nkv,
                                "head_dim": hd,
                                "d_model": d_model,
                                "is_diff": True,
                            }
        # Fallback: assume head_dim = d_model // n_heads with n_heads from config
        # Try head_dim=64 (most common for 2048 d_model)
        for hd in (64, 128, 96, 48, 32):
            nh = d_model // hd
            if nh * hd == d_model:
                nkv = k_rows // (2 * hd) if k_rows else 0
                if nkv > 0 and 2 * nkv * hd == k_rows:
                    return {
                        "n_heads": nh,
                        "n_kv_heads": nkv,
                        "head_dim": hd,
                        "d_model": d_model,
                        "is_diff": True,
                    }
        raise ValueError(
            f"Cannot auto-detect params from diff checkpoint: "
            f"q_rows={q_rows}, k_rows={k_rows}, v_rows={v_rows}, d_model={d_model}"
        )
    else:
        # Already GQA: q_rows = n_heads * head_dim = d_model
        # Infer from k_rows and v_rows
        if k_rows is not None and v_rows is not None:
            # k_rows = n_kv_heads * head_dim, v_rows = n_kv_heads * head_dim (GQA ties or not)
            for nkv in (8, 4, 2, 16, 1):
                hd = k_rows // nkv
                if hd > 0 and nkv * hd == k_rows:
                    nh = d_model // hd
                    if nh * hd == d_model:
                        return {
                            "n_heads": nh,
                            "n_kv_heads": nkv,
                            "head_dim": hd,
                            "d_model": d_model,
                            "is_diff": False,
                        }
        # Fallback
        hd = d_model // 32  # assume 32 heads
        return {
            "n_heads": 32,
            "n_kv_heads": k_rows // hd if k_rows else 8,
            "head_dim": hd,
            "d_model": d_model,
            "is_diff": False,
        }


def _find_attn_layers(state: dict[str, torch.Tensor]) -> list[int]:
    """Find all layer indices that have attention q_proj."""
    layers = set()
    for k in state:
        if k.startswith("blocks.") and ".attn.q_proj.weight" in k:
            layers.add(int(k.split(".")[1]))
    return sorted(layers)


def convert_v3_to_v4_state(
    state: dict[str, torch.Tensor],
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a V3 (diff attention) state dict to V4 (GTA) in-place.

    Auto-detects n_heads, n_kv_heads, head_dim from checkpoint shapes if
    not provided. Works on in-memory state dicts (no disk I/O).

    Args:
        state: V3 checkpoint state dict (modified in-place)
        n_heads, n_kv_heads, head_dim: optional explicit params (auto-detected if None)
        verbose: print conversion details

    Returns:
        The same state dict, converted to V4 (GTA) format.
    """
    # Auto-detect params if not provided
    if n_heads is None or n_kv_heads is None or head_dim is None:
        params = _detect_params(state)
        n_heads = n_heads or params["n_heads"]
        n_kv_heads = n_kv_heads or params["n_kv_heads"]
        head_dim = head_dim or params["head_dim"]
        is_diff = params["is_diff"]
        if verbose:
            print(f"  [V3→V4] Auto-detected: n_heads={n_heads}, "
                  f"n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
                  f"is_diff={is_diff}")

    attn_layers = _find_attn_layers(state)
    if verbose:
        print(f"  [V3→V4] Attention layers: {attn_layers}")

    # Step 1: Reverse DifferentialAttention (if diff checkpoint)
    keys_to_remove = []
    for layer_idx in attn_layers:
        base = f"blocks.{layer_idx}.attn"

        # Average q_proj: [2*n_heads*hd, d] → [n_heads*hd, d]
        qk = f"{base}.q_proj.weight"
        if qk in state:
            w = state[qk]
            if w.shape[0] == 2 * n_heads * head_dim:
                w_avg = (w.view(n_heads, 2, head_dim, -1).mean(dim=1)
                         .reshape(n_heads * head_dim, -1))
                state[qk] = w_avg.contiguous()
                if verbose:
                    print(f"    {base}.q_proj: {list(w.shape)} → {list(w_avg.shape)}")

        # Average k_proj: [2*n_kv*hd, d] → [n_kv*hd, d]
        kk = f"{base}.k_proj.weight"
        if kk in state:
            w = state[kk]
            if w.shape[0] == 2 * n_kv_heads * head_dim:
                w_avg = (w.view(n_kv_heads, 2, head_dim, -1).mean(dim=1)
                         .reshape(n_kv_heads * head_dim, -1))
                state[kk] = w_avg.contiguous()
                if verbose:
                    print(f"    {base}.k_proj: {list(w.shape)} → {list(w_avg.shape)}")

        # Remove lambda_param (diff-specific)
        lam_key = f"{base}.lambda_param"
        if lam_key in state:
            keys_to_remove.append(lam_key)

        # Remove diff-specific rms_norm (GTA doesn't have this)
        rms_key = f"{base}.rms_norm.weight"
        if rms_key in state:
            keys_to_remove.append(rms_key)

    for k in keys_to_remove:
        del state[k]
    if verbose and keys_to_remove:
        print(f"    Removed {len(keys_to_remove)} diff-specific keys "
              f"(lambda_param + rms_norm)")

    # Step 2: Forward GTAKey
    #   - If K and V are already tied (V=K), set v_mix_gate=0 (V=K, KV cache savings)
    #   - If K and V are separate (trained model), preserve original V_proj and
    #     set v_mix_gate to a large value so sigmoid(gate)≈1.0 → V=V_proj (bit-exact
    #     with original GQA). Training can then move gate toward 0 for KV savings.
    for layer_idx in attn_layers:
        base = f"blocks.{layer_idx}.attn"
        kk = f"{base}.k_proj.weight"
        vk = f"{base}.v_proj.weight"

        # Determine if V is already tied to K
        v_tied_to_k = False
        if kk in state and vk in state and state[kk].shape == state[vk].shape:
            v_tied_to_k = (state[kk] - state[vk]).abs().max().item() < 1e-6

        if v_tied_to_k:
            # V=K already: use gate=0 (V=K, halves KV cache BW)
            state[f"{base}.v_mix_gate"] = torch.zeros(1, dtype=torch.float32)
            if verbose:
                print(f"    {base}: V=K tied, gate=0 (KV cache savings active)")
        else:
            # V separate from K: preserve original V_proj, use large gate
            # so sigmoid(gate)≈1.0 → V=V_proj (bit-exact with GQA).
            # Training can move gate toward 0 to enable V=K KV savings.
            state[f"{base}.v_mix_gate"] = torch.tensor([100.0], dtype=torch.float32)
            if verbose:
                print(f"    {base}: V separate, gate=100 (V=V_proj, bit-exact)")

        # Copy k_proj.qscale to v_proj.qscale if BitNet and V=K
        if v_tied_to_k:
            kq = f"{base}.k_proj.qscale"
            vq = f"{base}.v_proj.qscale"
            if kq in state and vq in state:
                state[vq] = state[kq].clone()

    return state


def convert_v3_to_v4(
    input_path: str,
    output_path: str,
    n_layers: int | None = None,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
):
    """Convert V3 (diff attention) checkpoint file to V4 (GTA) checkpoint file.

    Auto-detects all params from checkpoint shapes if not provided.
    """
    print(f"  [V3→V4] Loading: {input_path}")
    t0 = time.time()
    state = load_file(input_path)
    print(f"  [V3→V4] Loaded {len(state)} keys in {time.time()-t0:.1f}s")

    convert_v3_to_v4_state(state, n_heads=n_heads, n_kv_heads=n_kv_heads,
                           verbose=True)

    print(f"  [V3→V4] Saving: {output_path}")
    t1 = time.time()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(state, output_path)
    print(f"  [V3→V4] Saved {len(state)} keys in {time.time()-t1:.1f}s")
    print(f"  [V3→V4] Done! Total: {time.time()-t0:.1f}s")

    # Verify key shapes
    print(f"\n  [V3→V4] Verification (layer {next(iter(_find_attn_layers(state)))}):")
    first_layer = _find_attn_layers(state)[0]
    for k in sorted(state.keys()):
        if f"blocks.{first_layer}.attn" in k:
            print(f"    {k}: {list(state[k].shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert V3 checkpoint (diff attention) to V4 (GTA). "
                    "Auto-detects model params from checkpoint shapes."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input V3 checkpoint (.safetensors)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output V4 checkpoint (.safetensors)")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Override n_heads (auto-detected if not specified)")
    parser.add_argument("--n-kv-heads", type=int, default=None,
                        help="Override n_kv_heads (auto-detected if not specified)")
    args = parser.parse_args()

    convert_v3_to_v4(
        input_path=args.input,
        output_path=args.output,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
    )


if __name__ == "__main__":
    main()

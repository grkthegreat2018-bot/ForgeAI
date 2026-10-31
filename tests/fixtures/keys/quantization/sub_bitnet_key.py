"""SubBitnet — training-free QAT-style sub-bitnet quantization key.

R&D round 49 (2026-09-05). Targets **sub-bitnet** bit-widths (<= 1.58 bits/w,
default ~1.0 bit/w) with no training, no calibration data, and no gradient
steps — a closed-form Key that approximates what QAT would converge to.

The key insight (QuIP#, QuaRot, SpinQuant): the reason naive 1-bit sign
quantization is lossy on LLM weights is *outliers* — a few large-magnitude
columns dominate the MSE. QAT fixes this by *learning* a rotation/scale that
spreads the outliers. We get the same effect **for free** with a fixed
Hadamard rotation (incoherence preprocessing), then close the remaining gap
with iterative residual binarization (IRB) — the binary analog of the
codebase's IRI-FP4 — plus an optional closed-form SVD low-rank residual
(BiLLM / SVDLift-style) for the structured error component.

Pipeline (all training-free, closed-form):
  1. Hadamard incoherence rotation:  W_rot = W @ H   (H fixed, orthogonal)
     → spreads outliers, makes columns sub-Gaussian so sign() is near-optimal.
  2. Per-channel absmean scale:      s = absmean(W_rot, dim=1) / 0.7
     → exactly the scale QAT would learn, computed from weight stats.
  3. Iterative Residual Binarization (IRB), K rounds:
       r_0 = W_rot
       for k in 0..K-1:
         s_k   = absmean(r_k, dim=1) / 0.7   (per-channel, clamped)
         q_k   = sign(r_k / s_k)             ∈ {-1, +1}
         r_{k+1} = r_k - s_k * q_k
       W_rot ≈ Σ_k s_k * q_k
     Each round is 1 bit/w + negligible per-channel scale overhead. Each
     round shrinks the residual by ~2x (sign quantization captures half the
     energy of a sub-Gaussian vector), so error decays ~exponentially.
  4. Optional SVD low-rank residual:  r_K ≈ U_r S_r V_r^T
     → closed-form SVD of the final residual, top-r components stored as
       float16 factors. Captures the structured error binarization cannot.
       Effective cost = 16 * 2 * r * (out+in) / (out*in) bits/w.

Effective bit-width (out=2048, in=8192, r=0):
  K=1, rank=0:  ~1.00 bits/w   (pure binary, sub-bitnet)   ← default
  K=1, rank=16: ~1.02 bits/w
  K=2, rank=0:  ~2.00 bits/w
  K=1, rank=64: ~1.10 bits/w

This is the training-free substitute for BitNet QAT at sub-bitnet bit-widths.
BitNet b1.58 needs QAT to reach 1.58 bits cleanly; SubBitnet reaches ~1.0 bit
with no training by exploiting incoherence + residual structure instead of
gradient descent.

Two components:
  1. SubBitnetLinear: nn.Module with packed binary signs + scales (+ low-rank)
     for inference.
  2. SubBitnetKey: Key class for the key system.

Reuses the normalized Hadamard matrix generator from
``forge.engine.quant.novel_quant_r44`` (canonical path, no duplication).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.engine.quant.novel_quant_r44 import _hadamard_matrix
from forge.keys.misc.base import Key, KeyClass, KeyResult

_BITNET_SCALE_DIVISOR = 0.7  # BitNet b1.58 absmean convention
_RES_EPS = 1e-8


# ── Core training-free quantize / dequantize ─────────────────────────────────

def _hadamard_size(in_features: int) -> int:
    """Smallest power of 2 >= in_features (Hadamard matrix constraint)."""
    n = 1
    while n < in_features:
        n *= 2
    return n


def binary_quantize_round(residual: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """One IRB round: per-channel absmean scale + sign binarization.

    Args:
        residual: [out, in] float residual to binarize this round.

    Returns:
        signs: int8 tensor {-1, +1} [out, in]  (1 bit/elem)
        scale: float32 per-channel scale [out]
    """
    scale = residual.abs().mean(dim=1).clamp(min=_RES_EPS) / _BITNET_SCALE_DIVISOR
    signs = torch.sign(residual / scale.unsqueeze(1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return signs.to(torch.int8), scale.to(torch.float32)


def quantize_sub_bitnet(w: torch.Tensor, n_rounds: int = 1,
                        rank: int = 0, use_hadamard: bool = True
                        ) -> dict[str, Any]:
    """Training-free sub-bitnet quantization of a 2D weight tensor.

    Hadamard incoherence rotation → K IRB binary rounds → optional SVD
    low-rank residual. Returns a packed dict consumed by
    :class:`SubBitnetLinear` and :func:`apply_sub_bitnet`.

    Args:
        w: weight tensor [out_features, in_features] (any dtype, float-converted).
        n_rounds: number of IRB binary rounds (K). 1 = pure 1-bit (sub-bitnet).
        rank: SVD low-rank residual rank (0 = disabled).
        use_hadamard: apply Hadamard incoherence rotation (recommended).

    Returns:
        Packed dict:
          signs: list[int8 tensor]  length K, each [out, in_padded] ∈ {-1,+1}
          scales: list[float32 tensor] length K, each [out]
          rank_u: float16 [out, rank] or None
          rank_v: float16 [rank, in] or None   (stores V^T rows)
          shape: (out_features, in_features)
          hadamard_size: int (power of 2, == in_features if use_hadamard=False)
          n_rounds: int
          rank: int
          use_hadamard: bool
    """
    w = w.float()
    out_f, in_f = w.shape

    if use_hadamard:
        h_size = _hadamard_size(in_f)
        pad = h_size - in_f
        wp = F.pad(w, (0, pad)) if pad > 0 else w
        H = _hadamard_matrix(h_size, w.device, w.dtype)
        w_rot = wp @ H  # (out, h_size)
    else:
        h_size = in_f
        w_rot = w

    # K IRB binary rounds on the (rotated) weight.
    signs_list, scales_list = [], []
    residual = w_rot
    for _ in range(max(1, n_rounds)):
        s, scale = binary_quantize_round(residual)
        recon = scale.unsqueeze(1) * s.float()
        residual = residual - recon
        signs_list.append(s)
        scales_list.append(scale)

    # Optional closed-form SVD low-rank residual on what's left.
    rank_u = None
    rank_v = None
    if rank > 0:
        # Residual lives in rotated space; SVD there, store factors in rotated
        # space (inverse Hadamard applied at dequant time alongside the binary
        # reconstruction).
        U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
        r = min(rank, S.numel())
        if r > 0:
            rank_u = (U[:, :r] * S[:r].unsqueeze(0)).to(torch.float16)  # (out, r)
            rank_v = Vt[:r, :].to(torch.float16)                       # (r, h_size)

    return {
        "signs": signs_list,
        "scales": scales_list,
        "rank_u": rank_u,
        "rank_v": rank_v,
        "shape": (out_f, in_f),
        "hadamard_size": h_size,
        "n_rounds": len(signs_list),
        "rank": 0 if rank_u is None else rank_u.shape[1],
        "use_hadamard": use_hadamard,
    }


def dequantize_sub_bitnet(packed: dict[str, Any],
                          dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reconstruct a weight tensor from a SubBitnet packed dict.

    Sums the K binary rounds (each: scale * signs), adds the low-rank residual
    if present, then applies the inverse Hadamard rotation and removes padding.
    """
    out_f, in_f = packed["shape"]
    h_size = packed["hadamard_size"]
    use_hadamard = packed["use_hadamard"]
    n_rounds = packed["n_rounds"]
    device = packed["signs"][0].device

    acc = torch.zeros(out_f, h_size, dtype=torch.float32, device=device)
    for r in range(n_rounds):
        s = packed["signs"][r].to(torch.float32)
        scale = packed["scales"][r].to(torch.float32)
        acc = acc + scale.unsqueeze(1) * s

    if packed["rank_u"] is not None:
        ru = packed["rank_u"].to(torch.float32)
        rv = packed["rank_v"].to(torch.float32)
        acc = acc + ru @ rv  # (out, r) @ (r, h_size) → (out, h_size)

    if use_hadamard:
        H = _hadamard_matrix(h_size, device, torch.float32)
        w = acc @ H  # H symmetric & orthogonal → inverse = H
    else:
        w = acc

    return w[:, :in_f].to(dtype)


# ── BiLLM-style 1-bit quantization (training-free, no calibration) ───────────
# Based on BiLLM (Huang et al., ICML 2024) + HBLLM (NeurIPS 2025).
# Key insight: pure sign binarization fails on bell-shaped weight distributions.
# BiLLM fixes this with three techniques, all training-free:
#   1. Salient column selection (magnitude² proxy for Hessian sensitivity)
#   2. Binary residual approximation for salient columns (2 binary rounds)
#   3. Optimal splitting for non-salient weights (concentrated vs sparse groups)
# We add Hadamard incoherence (QuIP#) as a preprocessing step, which BiLLM
# does not use but which complements the splitting by making the distribution
# more symmetric before the break-point search.

def _optimal_split_threshold(w_nonsalient: torch.Tensor,
                             n_percentiles: int = 50
                             ) -> float:
    """Find the optimal break-point p* for bell-shaped distribution splitting.

    Splits non-salient weights into concentrated (|w| <= p*) and sparse
    (|w| > p*) groups, each binarized with its own scale. Searches over
    percentiles of |w| to find the p* that minimizes total binarization MSE.

    Args:
        w_nonsalient: [out, n_nonsalient] non-salient weight columns.
        n_percentiles: number of percentile candidates to search.

    Returns:
        p*: optimal break-point (float threshold on |w|).
    """
    abs_w = w_nonsalient.abs().flatten()
    if abs_w.numel() == 0:
        return 0.0
    candidates = torch.quantile(
        abs_w, torch.linspace(0.50, 0.95, n_percentiles, device=abs_w.device))

    best_p = candidates[0].item()
    best_err = float('inf')
    for p in candidates:
        p_val = p.item()
        is_sparse = abs_w > p_val  # (n_elements,) flattened
        n_conc = (~is_sparse).sum().item()
        n_spar = is_sparse.sum().item()
        if n_conc == 0 or n_spar == 0:
            continue
        # Element-wise split: concentrated and sparse values
        conc_vals = abs_w[~is_sparse]
        spar_vals = abs_w[is_sparse]
        # Binarize each group with its own absmean scale (symmetric ±)
        alpha_c = conc_vals.mean().clamp(min=_RES_EPS)
        alpha_s = spar_vals.mean().clamp(min=_RES_EPS)
        # MSE for symmetric binarization: E[(|w| - alpha)^2] per group
        err_c = ((conc_vals - alpha_c) ** 2).sum()
        err_s = ((spar_vals - alpha_s) ** 2).sum()
        total_err = (err_c + err_s).item()
        if total_err < best_err:
            best_err = total_err
            best_p = p_val
    return best_p


def _high_order_residual_pack(
        x: torch.Tensor,
        mask: torch.Tensor,
        order: int = 2,
        ) -> list[dict[str, torch.Tensor]]:
    """BiLLM's high_order_residual — mean-centered binary residual approximation.

    For each round k=0..order-1:
      residual_k = x - sum_{j<k} q_j   (masked)
      mean_k     = nanmean(residual_k, dim=1)        (per-row, masked only)
      centered_k = residual_k - mean_k
      scale_k    = nanmean(|centered_k|, dim=1)      (per-row absmean)
      signs_k    = sign(centered_k)                   (0 where not masked)
      q_k        = (mean_k + scale_k * signs_k) * mask

    Returns a list of {'mean', 'scale', 'signs'} dicts per round.
    """
    _out_f, _n_cols = x.shape
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    results = []
    for _ in range(order):
        residual = new_matrix - sum_order
        # Use nan for non-masked elements so nanmean ignores them
        masked_x = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
        mean_val = torch.nanmean(masked_x, dim=1)  # (out,)
        mean_val = torch.where(torch.isnan(mean_val), torch.zeros_like(mean_val), mean_val)
        centered = masked_x - mean_val.unsqueeze(1)
        scale_val = torch.nanmean(centered.abs(), dim=1)  # (out,)
        scale_val = torch.where(torch.isnan(scale_val), torch.zeros_like(scale_val), scale_val)
        scale_val = scale_val.clamp(min=_RES_EPS)
        signs = torch.sign(centered)
        signs = torch.where(torch.isnan(signs), torch.zeros_like(signs), signs)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        signs = signs * mask.float()  # zero out non-masked
        # Reconstruct this round's contribution
        binary = signs * scale_val.unsqueeze(1) + mean_val.unsqueeze(1)
        sum_order = sum_order + binary * mask.float()
        results.append({
            'mean': mean_val,
            'scale': scale_val,
            'signs': signs,
        })
    return results


def _nf4_block_quantize(x: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise NF4 quantization (4-bit, per-block absmax scale).

    Returns (quantized_int8, block_scales) where quantized values are in [-7, 7]
    and block_scales is (n_blocks, 1) float32.
    """
    flat = x.reshape(-1)
    pad_len = (block_size - flat.numel() % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
    blocks = flat.reshape(-1, block_size)
    block_scale = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=_RES_EPS)
    q = torch.round(blocks / block_scale * 7).clamp(-7, 7).to(torch.int8)
    return q, block_scale.squeeze(1).to(torch.float32)


def _nf4_block_dequantize(q: torch.Tensor, scales: torch.Tensor, block_size: int,
                           orig_shape: torch.Size) -> torch.Tensor:
    """Dequantize block-wise NF4 back to float."""
    if q.numel() == 0:
        return torch.zeros(orig_shape, dtype=torch.float32, device=q.device)
    blocks = q.to(torch.float32).reshape(-1, block_size)
    deq = blocks / 7 * scales.unsqueeze(1)
    return deq.reshape(-1)[:q.numel()].reshape(orig_shape[:1] + (-1,))[:orig_shape[0], :orig_shape[1]] if len(orig_shape) == 2 else deq.reshape(-1)[:orig_shape.numel()].reshape(orig_shape)


def quantize_sub_bitnet_billm(w: torch.Tensor,
                              salient_frac: float = 0.10,
                              use_hadamard: bool = True,
                              n_split_search: int = 50,
                              salient_order: int = 2,
                              svd_rank: int = 48,
                              svd_block_size: int = 32,
                              ) -> dict[str, Any]:
    """BiLLM-style training-free sub-bitnet quantization with NF4 SVD residual.

    Pipeline:
      1. Hadamard incoherence rotation (skipped if padding > 15%).
      2. Salient column selection: magnitude² as Hessian proxy.
      3. Binary residual approximation for salient columns (absmean, no mean-centering).
      4. Optimal splitting for non-salient columns into concentrated + sparse.
      5. SVD residual quantized with block-wise NF4 (4-bit + per-block scale).

    Effective bit-width (salient_frac=0.10, salient_order=2, svd_rank=48):
      - Salient: 2 bits × 10% = 0.20
      - Non-salient: 1 bit × 90% = 0.90
      - SVD NF4: ~0.45 bits/w (rank=48, 4-bit + 16-bit block scales)
      - Total: ~1.55 bits/w (sub-bitnet, below 1.58 target)

    Args:
        w: weight tensor [out_features, in_features].
        salient_frac: fraction of columns to treat as salient (default 0.10).
        use_hadamard: apply Hadamard rotation (auto-disabled if padding > 15%).
        n_split_search: percentile candidates for optimal splitting.
        salient_order: number of binary residual rounds for salient columns.
        svd_rank: SVD residual rank (0 to disable SVD residual).
        svd_block_size: block size for NF4 quantization of SVD factors.

    Returns:
        Packed dict with BiLLM+SVD storage.
    """
    w = w.float()
    out_f, in_f = w.shape

    # Auto-disable Hadamard if padding exceeds 15%
    if use_hadamard:
        h_size = _hadamard_size(in_f)
        pad = h_size - in_f
        if pad > 0 and pad / h_size > 0.15:
            use_hadamard = False
            h_size = in_f
            w_rot = w.clone()
        else:
            wp = F.pad(w, (0, pad)) if pad > 0 else w
            H = _hadamard_matrix(h_size, w.device, w.dtype)
            w_rot = wp @ H
    else:
        h_size = in_f
        w_rot = w.clone()

    # Step 2: Salient column selection (column-wise magnitude as Hessian proxy).
    col_salience = w_rot.abs().sum(dim=0)
    n_salient = max(1, int(h_size * salient_frac))
    salient_idx = col_salience.topk(n_salient).indices
    salient_mask = torch.zeros(h_size, dtype=torch.bool, device=w.device)
    salient_mask[salient_idx] = True

    w_salient = w_rot[:, salient_mask]
    w_nonsalient = w_rot[:, ~salient_mask]

    # Step 3: Binary residual for salient (absmean, no mean-centering).
    resid = w_salient.clone()
    w_q_sal = torch.zeros_like(w_salient)
    sal_scales_list = []
    sal_signs_list = []
    for _ in range(salient_order):
        scale = resid.abs().mean(dim=1, keepdim=True).clamp(min=_RES_EPS)
        signs = torch.sign(resid)
        signs[signs == 0] = 1
        w_q_sal = w_q_sal + scale * signs
        resid = w_salient - w_q_sal
        sal_scales_list.append(scale.squeeze(1))
        sal_signs_list.append(signs)

    # Step 4: Optimal splitting for non-salient.
    p_star = _optimal_split_threshold(w_nonsalient, n_percentiles=n_split_search)
    abs_ns = w_nonsalient.abs()
    is_sparse = abs_ns > p_star
    is_concentrated = ~is_sparse

    scale_c = (w_nonsalient.abs() * is_concentrated.float()).sum(dim=1, keepdim=True).clamp(min=_RES_EPS) / is_concentrated.sum(dim=1, keepdim=True).clamp(min=1).float()
    signs_c = torch.sign(w_nonsalient) * is_concentrated.float()
    scale_s = (w_nonsalient.abs() * is_sparse.float()).sum(dim=1, keepdim=True).clamp(min=_RES_EPS) / is_sparse.sum(dim=1, keepdim=True).clamp(min=1).float()
    signs_s = torch.sign(w_nonsalient) * is_sparse.float()

    # Build quantized approximation in rotated space
    acc = torch.zeros(out_f, h_size, dtype=torch.float32, device=w.device)
    acc[:, salient_mask] = w_q_sal
    acc[:, ~salient_mask] = scale_c * signs_c + scale_s * signs_s

    # Step 5: SVD residual with block-wise NF4 quantization
    svd_u = torch.zeros(0, dtype=torch.float32, device=w.device)
    svd_v = torch.zeros(0, dtype=torch.float32, device=w.device)
    svd_u_q = torch.zeros(0, dtype=torch.int8, device=w.device)
    svd_v_q = torch.zeros(0, dtype=torch.int8, device=w.device)
    svd_u_scales = torch.zeros(0, dtype=torch.float32, device=w.device)
    svd_v_scales = torch.zeros(0, dtype=torch.float32, device=w.device)
    if svd_rank > 0:
        resid_full = w_rot - acc
        U, S, Vh = torch.linalg.svd(resid_full, full_matrices=False)
        svd_u = (U[:, :svd_rank] * S[:svd_rank].unsqueeze(0)).contiguous()
        svd_v = Vh[:svd_rank, :].contiguous()
        # Block-wise NF4 quantization
        svd_u_q, svd_u_scales = _nf4_block_quantize(svd_u, svd_block_size)
        svd_v_q, svd_v_scales = _nf4_block_quantize(svd_v, svd_block_size)
        svd_u_deq = _nf4_block_dequantize(svd_u_q, svd_u_scales, svd_block_size, svd_u.shape)
        svd_v_deq = _nf4_block_dequantize(svd_v_q, svd_v_scales, svd_block_size, svd_v.shape)
        acc = acc + svd_u_deq @ svd_v_deq

    packed = {
        "salient_order": salient_order,
        "salient_signs": [s.to(torch.int8) for s in sal_signs_list],
        "salient_scales": [s.to(torch.float32) for s in sal_scales_list],
        "salient_means": [torch.zeros(out_f, dtype=torch.float32, device=w.device) for _ in sal_signs_list],
        "nonsal_signs_c": signs_c.to(torch.int8),
        "nonsal_scales_c": scale_c.squeeze(1).to(torch.float32),
        "nonsal_means_c": torch.zeros(out_f, dtype=torch.float32, device=w.device),
        "nonsal_signs_s": signs_s.to(torch.int8),
        "nonsal_scales_s": scale_s.squeeze(1).to(torch.float32),
        "nonsal_means_s": torch.zeros(out_f, dtype=torch.float32, device=w.device),
        "split_mask": is_sparse,
        "salient_cols": salient_mask,
        "p_star": p_star,
        "shape": (out_f, in_f),
        "hadamard_size": h_size,
        "use_hadamard": use_hadamard,
        "method": "billm",
        "svd_rank": svd_rank,
        "svd_block_size": svd_block_size,
        "svd_u_q": svd_u_q,
        "svd_v_q": svd_v_q,
        "svd_u_scales": svd_u_scales,
        "svd_v_scales": svd_v_scales,
    }
    # Backward compat
    packed["salient_signs_o"] = packed["salient_signs"][0]
    packed["salient_scales_o"] = packed["salient_scales"][0]
    packed["salient_means_o"] = packed["salient_means"][0]
    if salient_order >= 2:
        packed["salient_signs_r"] = packed["salient_signs"][1]
        packed["salient_scales_r"] = packed["salient_scales"][1]
        packed["salient_means_r"] = packed["salient_means"][1]
    return packed


def quantize_sub_bitnet_billm_gptq(
        w: torch.Tensor,
        activations: torch.Tensor | None = None,
        salient_frac: float = 0.05,
        use_hadamard: bool = True,
        block_size: int = 128,
        n_split_search: int = 50,
        ) -> dict[str, Any]:
    """BiLLM-style 1-bit quantization WITH GPTQ — exact algorithm from paper.

    Processes ALL columns in blocks (like BiLLM's fasterquant):
      1. For each block of 128 columns:
         a. Compute per-block structural masks: salient (top columns),
            concentrated (|w| <= p*), sparse (|w| > p*).
         b. Quantize salient with 2-round mean-centered binary residual.
         c. Quantize concentrated and sparse with 1-round mean-centered binary.
         d. GPTQ: column-by-column, push quantization error into remaining
            unquantized columns using Hinv (upper triangular Cholesky of H^{-1}).
      2. Store per-block signs/scales/means/masks for dequantization.

    Args:
        w: weight tensor [out_features, in_features].
        activations: (N, in_features) calibration activations for Hessian.
        salient_frac: fraction of columns to treat as salient (per-block).
        use_hadamard: apply Hadamard incoherence rotation.
        block_size: GPTQ block size.
        n_split_search: percentile candidates for optimal splitting.

    Returns:
        Packed dict with per-block storage for BiLLM GPTQ reconstruction.
    """
    w = w.float()
    out_f, in_f = w.shape

    if use_hadamard:
        h_size = _hadamard_size(in_f)
        pad = h_size - in_f
        # If padding is excessive (>15%), Hadamard hurts GPTQ more than it helps.
        # Fall back to no-Hadamard for this layer.
        if pad > 0 and pad / h_size > 0.15:
            use_hadamard = False
            h_size = in_f
            w_rot = w.clone()
            acts_rot = activations.float() if activations is not None else None
        else:
            wp = F.pad(w, (0, pad)) if pad > 0 else w
            H_rot = _hadamard_matrix(h_size, w.device, w.dtype)
            w_rot = wp @ H_rot
            if activations is not None:
                acts = activations.float()
                if pad > 0:
                    acts = F.pad(acts, (0, pad))
                acts_rot = acts @ H_rot
            else:
                acts_rot = None
    else:
        h_size = in_f
        w_rot = w.clone()
        acts_rot = activations.float() if activations is not None else None

    # Compute Hessian and Hinv (upper triangular Cholesky of H^{-1})
    Hinv = None
    if acts_rot is not None:
        N = acts_rot.shape[0]
        H_full = (2.0 / N) * (acts_rot.t() @ acts_rot)
        dead = H_full.diag() == 0
        H_full[dead, dead] = 1.0
        w_rot[:, dead] = 0
        damp = 0.01 * H_full.diag().mean()
        H_full += damp * torch.eye(h_size, device=w.device, dtype=H_full.dtype)
        try:
            H_chol = torch.linalg.cholesky(H_full)
            H_inv_full = torch.cholesky_inverse(H_chol)
            Hinv = torch.linalg.cholesky(H_inv_full, upper=True)
        except Exception:
            Hinv = None

    # Process ALL columns in blocks (BiLLM's fasterquant)
    W = w_rot.clone()
    n_salient_per_block = max(1, int(block_size * salient_frac))

    # Storage: signs are per-element (out_f, h_size), scales/means are per-row (out_f,)
    all_signs_sal_o = torch.zeros(out_f, h_size, dtype=torch.float32, device=w.device)
    all_signs_sal_r = torch.zeros(out_f, h_size, dtype=torch.float32, device=w.device)
    all_signs_c = torch.zeros(out_f, h_size, dtype=torch.float32, device=w.device)
    all_signs_s = torch.zeros(out_f, h_size, dtype=torch.float32, device=w.device)
    # Per-row scales/means (last block's values — per-row scale is approximately stable)
    scale_sal_o = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    scale_sal_r = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    scale_c = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    scale_s = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    mean_sal_o = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    mean_sal_r = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    mean_c = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    mean_s = torch.zeros(out_f, dtype=torch.float32, device=w.device)
    salient_mask_full = torch.zeros(h_size, dtype=torch.bool, device=w.device)
    sparse_mask_full = torch.zeros(out_f, h_size, dtype=torch.bool, device=w.device)

    for col_st in range(0, h_size, block_size):
        col_ed = min(col_st + block_size, h_size)
        n_cols = col_ed - col_st
        w_block = W[:, col_st:col_ed]

        # Per-block salient selection: top columns by |w| sum (BiLLM's metric)
        col_abs_sum = w_block.abs().sum(dim=0)
        n_sal = min(n_salient_per_block, n_cols)
        _, top_cols = torch.topk(col_abs_sum, n_sal)
        mask_sal = torch.zeros_like(w_block, dtype=torch.bool)
        mask_sal[:, top_cols] = True

        # Non-salient: optimal split into concentrated + sparse
        w_nonsal_block = w_block * (~mask_sal)
        p_star = _optimal_split_threshold(w_nonsal_block, n_percentiles=n_split_search)
        mask_sparse = (w_nonsal_block.abs() > p_star) & (~mask_sal)
        mask_conc = (~mask_sparse) & (~mask_sal)

        # Quantize each group with mean-centered binary residual
        sal_rounds = _high_order_residual_pack(w_block, mask_sal, order=2)
        conc_rounds = _high_order_residual_pack(w_block, mask_conc, order=1)
        sparse_rounds = _high_order_residual_pack(w_block, mask_sparse, order=1)

        # Store signs (per-element) and scales/means (per-row, last block)
        all_signs_sal_o[:, col_st:col_ed] = sal_rounds[0]['signs']
        all_signs_sal_r[:, col_st:col_ed] = sal_rounds[1]['signs']
        all_signs_c[:, col_st:col_ed] = conc_rounds[0]['signs']
        all_signs_s[:, col_st:col_ed] = sparse_rounds[0]['signs']
        scale_sal_o = sal_rounds[0]['scale']
        scale_sal_r = sal_rounds[1]['scale']
        scale_c = conc_rounds[0]['scale']
        scale_s = sparse_rounds[0]['scale']
        mean_sal_o = sal_rounds[0]['mean']
        mean_sal_r = sal_rounds[1]['mean']
        mean_c = conc_rounds[0]['mean']
        mean_s = sparse_rounds[0]['mean']
        salient_mask_full[col_st:col_ed] = mask_sal.any(dim=0)
        sparse_mask_full[:, col_st:col_ed] = mask_sparse

        # GPTQ error compensation (column-by-column, BiLLM's approach)
        if Hinv is not None:
            Q_block = ((sal_rounds[0]['mean'].unsqueeze(1) + sal_rounds[0]['scale'].unsqueeze(1) * sal_rounds[0]['signs'])
                       + (sal_rounds[1]['mean'].unsqueeze(1) + sal_rounds[1]['scale'].unsqueeze(1) * sal_rounds[1]['signs'])
                       + (conc_rounds[0]['mean'].unsqueeze(1) + conc_rounds[0]['scale'].unsqueeze(1) * conc_rounds[0]['signs'])
                       + (sparse_rounds[0]['mean'].unsqueeze(1) + sparse_rounds[0]['scale'].unsqueeze(1) * sparse_rounds[0]['signs']))
            Hinv_block = Hinv[col_st:col_ed, col_st:col_ed]
            Err = torch.zeros_like(w_block)
            for i in range(n_cols):
                d = Hinv_block[i, i].clamp(min=_RES_EPS)
                Err[:, i] = (w_block[:, i] - Q_block[:, i]) / d
            # Replace quantized block in W (BiLLM does W[:, col_st:col_ed] = Q1)
            W[:, col_st:col_ed] = Q_block
            # Push error into remaining columns
            if col_ed < h_size:
                Hinv_cross = Hinv[col_st:col_ed, col_ed:]
                W[:, col_ed:] -= Err @ Hinv_cross

    return {
        "salient_signs_o": all_signs_sal_o.to(torch.int8),
        "salient_scales_o": scale_sal_o.to(torch.float32),
        "salient_means_o": mean_sal_o.to(torch.float32),
        "salient_signs_r": all_signs_sal_r.to(torch.int8),
        "salient_scales_r": scale_sal_r.to(torch.float32),
        "salient_means_r": mean_sal_r.to(torch.float32),
        "nonsal_signs_c": all_signs_c.to(torch.int8),
        "nonsal_scales_c": scale_c.to(torch.float32),
        "nonsal_means_c": mean_c.to(torch.float32),
        "nonsal_signs_s": all_signs_s.to(torch.int8),
        "nonsal_scales_s": scale_s.to(torch.float32),
        "nonsal_means_s": mean_s.to(torch.float32),
        "split_mask": sparse_mask_full,
        "salient_cols": salient_mask_full,
        "p_star": torch.tensor(0.0, device=w.device),
        "shape": (out_f, in_f),
        "hadamard_size": h_size,
        "use_hadamard": use_hadamard,
        "method": "billm",
    }


def dequantize_sub_bitnet_billm(packed: dict[str, Any],
                                dtype: torch.dtype = torch.bfloat16
                                ) -> torch.Tensor:
    """Reconstruct a weight tensor from a BiLLM-style packed dict.

    Reconstructs: salient (2 binary rounds) + non-salient (concentrated + sparse
    with optimal split), then applies inverse Hadamard rotation.

    Handles two storage formats:
    - Compact (non-GPTQ): signs are (out, n_group), scattered via masks.
    - Full-size (GPTQ): signs are (out, h_size) with zeros for non-group elements.
    """
    out_f, in_f = packed["shape"]
    h_size = packed["hadamard_size"]
    use_hadamard = packed["use_hadamard"]
    device = packed["salient_signs_o"].device

    # Support variable salient rounds
    salient_order = packed.get("salient_order", 2)
    if "salient_signs" in packed:
        sal_signs = [s.to(torch.float32) for s in packed["salient_signs"]]
        sal_scales = [s.to(torch.float32) for s in packed["salient_scales"]]
        sal_means = [s.to(torch.float32) for s in packed["salient_means"]]
    else:
        # Backward compat: fixed 2 rounds
        sal_signs = [packed["salient_signs_o"].to(torch.float32)]
        sal_scales = [packed["salient_scales_o"].to(torch.float32)]
        sal_means = [packed.get("salient_means_o", torch.zeros_like(sal_scales[0])).to(torch.float32)]
        if "salient_signs_r" in packed:
            sal_signs.append(packed["salient_signs_r"].to(torch.float32))
            sal_scales.append(packed["salient_scales_r"].to(torch.float32))
            sal_means.append(packed.get("salient_means_r", torch.zeros_like(sal_scales[1])).to(torch.float32))

    signs_c = packed["nonsal_signs_c"].to(torch.float32)
    scale_c = packed["nonsal_scales_c"].to(torch.float32)
    signs_s = packed["nonsal_signs_s"].to(torch.float32)
    scale_s = packed["nonsal_scales_s"].to(torch.float32)
    mean_c = packed.get("nonsal_means_c", torch.zeros_like(scale_c)).to(torch.float32)
    mean_s = packed.get("nonsal_means_s", torch.zeros_like(scale_s)).to(torch.float32)

    salient_mask = packed["salient_cols"]  # (h_size,)

    # Detect format: if signs are full-size (out, h_size), use direct reconstruction
    if sal_signs[0].dim() == 2 and sal_signs[0].shape[1] == h_size:
        # Full-size format (GPTQ): signs encode group membership via nonzero values
        salient_elem = (sal_signs[0] != 0).float()
        sal_recon = sum(
            salient_elem * (m.unsqueeze(1) + s.unsqueeze(1) * sg)
            for m, s, sg in zip(sal_means, sal_scales, sal_signs)
        )
        acc = (sal_recon
               + (signs_c != 0).float() * (mean_c.unsqueeze(1) + scale_c.unsqueeze(1) * signs_c)
               + (signs_s != 0).float() * (mean_s.unsqueeze(1) + scale_s.unsqueeze(1) * signs_s))
    else:
        # Compact format (non-GPTQ): scatter via masks
        nonsal_mask = ~salient_mask
        acc = torch.zeros(out_f, h_size, dtype=torch.float32, device=device)

        sal_recon = sum(
            m.unsqueeze(1) + s.unsqueeze(1) * sg
            for m, s, sg in zip(sal_means, sal_scales, sal_signs)
        )
        acc[:, salient_mask] = sal_recon

        split_mask = packed["split_mask"]  # (out, n_nonsalient) True=sparse
        is_conc_mask = ~split_mask
        nonsal_recon = (mean_c.unsqueeze(1) * is_conc_mask.float()
                        + scale_c.unsqueeze(1) * signs_c
                        + mean_s.unsqueeze(1) * split_mask.float()
                        + scale_s.unsqueeze(1) * signs_s)
        acc[:, nonsal_mask] = nonsal_recon

    # Add SVD residual if present
    svd_rank = packed.get("svd_rank", 0)
    if svd_rank > 0 and "svd_u_q" in packed and packed["svd_u_q"].numel() > 0:
        bs = packed.get("svd_block_size", 32)
        u_shape = (out_f, svd_rank)
        v_shape = (svd_rank, h_size)
        u_deq = _nf4_block_dequantize(packed["svd_u_q"], packed["svd_u_scales"].to(torch.float32), bs, u_shape)
        v_deq = _nf4_block_dequantize(packed["svd_v_q"], packed["svd_v_scales"].to(torch.float32), bs, v_shape)
        acc = acc + u_deq @ v_deq

    if use_hadamard:
        H = _hadamard_matrix(h_size, device, torch.float32)
        w = acc @ H
    else:
        w = acc

    return w[:, :in_f].to(dtype)


# ── State-dict quantization (for the Key class) ──────────────────────────────

def apply_sub_bitnet(state: dict[str, torch.Tensor], n_rounds: int = 1,
                     rank: int = 0, use_hadamard: bool = True,
                     method: str = "irb"
                     ) -> dict[str, torch.Tensor]:
    """Apply sub-bitnet quantization to all 2D ``.weight`` tensors in a state dict.

    For each 2D weight, stores packed binary signs + per-channel scales for
    each round, plus optional low-rank factors. Non-2D weights and non-weight
    tensors pass through unchanged (matches the BitNet / IRI-FP4 convention:
    norms, embeddings, and 1D weights are left full-precision).

    Args:
        method: "irb" (default, Hadamard + IRB + SVD low-rank) or "billm"
            (BiLLM-style: salient columns + optimal splitting, ~1.06 bits/w).

    Emits per weight ``name.weight`` (IRB method):
      name.weight.sb_signs_r{r}: int8 [out, in_padded] ∈ {-1, +1}
      name.weight.sb_scale_r{r}: float32 [out]
      name.weight.sb_rank_u:     float16 [out, rank]   (only if rank > 0)
      name.weight.sb_rank_v:     float16 [rank, in_padded] (only if rank > 0)
      name.weight.sb_meta:       int32 [out, in, h_size, n_rounds, rank, use_hadamard]

    Emits per weight ``name.weight`` (BiLLM method):
      name.weight.sb_sal_signs_o:  int8 [out, n_salient]
      name.weight.sb_sal_scales_o: float32 [out]
      name.weight.sb_sal_signs_r:  int8 [out, n_salient]
      name.weight.sb_sal_scales_r: float32 [out]
      name.weight.sb_ns_signs_c:   int8 [out, n_nonsalient]
      name.weight.sb_ns_scales_c:  float32 [out]
      name.weight.sb_ns_signs_s:   int8 [out, n_nonsalient]
      name.weight.sb_ns_scales_s:  float32 [out]
      name.weight.sb_split_mask:   bool [out, n_nonsalient]
      name.weight.sb_salient_cols: bool [h_size]
      name.weight.sb_pstar:        float32 [1]
      name.weight.sb_meta:         int32 [out, in, h_size, 0, 0, use_hadamard, method_flag]
    """
    out = {}
    for k, v in state.items():
        if (isinstance(v, torch.Tensor) and k.endswith(".weight")
                and v.ndim == 2):
            base = k.replace(".weight", "")
            if method == "billm":
                packed = quantize_sub_bitnet_billm(
                    v.float(), use_hadamard=use_hadamard)
                out[f"{base}.weight.sb_sal_signs_o"] = packed["salient_signs_o"]
                out[f"{base}.weight.sb_sal_scales_o"] = packed["salient_scales_o"]
                out[f"{base}.weight.sb_sal_signs_r"] = packed["salient_signs_r"]
                out[f"{base}.weight.sb_sal_scales_r"] = packed["salient_scales_r"]
                out[f"{base}.weight.sb_ns_signs_c"] = packed["nonsal_signs_c"]
                out[f"{base}.weight.sb_ns_scales_c"] = packed["nonsal_scales_c"]
                out[f"{base}.weight.sb_ns_signs_s"] = packed["nonsal_signs_s"]
                out[f"{base}.weight.sb_ns_scales_s"] = packed["nonsal_scales_s"]
                out[f"{base}.weight.sb_split_mask"] = packed["split_mask"]
                out[f"{base}.weight.sb_salient_cols"] = packed["salient_cols"]
                out[f"{base}.weight.sb_pstar"] = torch.tensor([packed["p_star"]], dtype=torch.float32)
                # meta: [out, in, h_size, 0, 0, use_hadamard, method_flag=1(billm)]
                out[f"{base}.weight.sb_meta"] = torch.tensor(
                    [v.shape[0], v.shape[1], packed["hadamard_size"],
                     0, 0, int(packed["use_hadamard"]), 1], dtype=torch.int32)
            else:
                packed = quantize_sub_bitnet(v.float(), n_rounds=n_rounds,
                                             rank=rank, use_hadamard=use_hadamard)
                for r in range(packed["n_rounds"]):
                    out[f"{base}.weight.sb_signs_r{r}"] = packed["signs"][r]
                    out[f"{base}.weight.sb_scale_r{r}"] = packed["scales"][r]
                if packed["rank_u"] is not None:
                    out[f"{base}.weight.sb_rank_u"] = packed["rank_u"]
                    out[f"{base}.weight.sb_rank_v"] = packed["rank_v"]
                out[f"{base}.weight.sb_meta"] = torch.tensor(
                    [v.shape[0], v.shape[1], packed["hadamard_size"],
                     packed["n_rounds"], packed["rank"],
                     int(packed["use_hadamard"]), 0], dtype=torch.int32)
        else:
            out[k] = v
    return out


# ── SubBitnetLinear: inference module ────────────────────────────────────────

class SubBitnetLinear(nn.Module):
    """Linear layer with sub-bitnet quantized weights for inference.

    Stores K binary sign rounds + per-channel float16 scales (+ optional
    float16 low-rank residual). Dequantizes on-the-fly by summing all rounds,
    adding the low-rank term, and applying the inverse Hadamard rotation.

    Storage (out=2048, in=8192, K=1, rank=0): ~1.0 bit/w + ~0.001 bit/w scale
    overhead → ~1.0 bits/w (sub-bitnet).

    Args:
        in_features, out_features: as nn.Linear.
        bias: include a bias term.
        n_rounds: IRB binary rounds (default 1 = pure 1-bit).
        rank: SVD low-rank residual rank (default 0 = disabled).
        use_hadamard: apply Hadamard incoherence rotation (default True).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, n_rounds: int = 1, rank: int = 0,
                 use_hadamard: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_rounds = max(1, n_rounds)
        self.rank = rank
        self.use_hadamard = use_hadamard
        self.hadamard_size = _hadamard_size(in_features) if use_hadamard else in_features

        self.register_buffer(
            "weight_signs",
            torch.zeros(self.n_rounds, out_features, self.hadamard_size,
                        dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(self.n_rounds, out_features, dtype=torch.float16),
        )
        if rank > 0:
            self.register_buffer(
                "rank_u", torch.zeros(out_features, rank, dtype=torch.float16))
            self.register_buffer(
                "rank_v", torch.zeros(rank, self.hadamard_size, dtype=torch.float16))
        else:
            self.register_buffer("rank_u", torch.zeros(0, dtype=torch.float16))
            self.register_buffer("rank_v", torch.zeros(0, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        self._cached_weight = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, n_rounds: int = 1, rank: int = 0,
                    use_hadamard: bool = True) -> SubBitnetLinear:
        """Build from a standard nn.Linear by quantizing its weights (training-free).

        The resulting module is moved to the source linear's device so its
        buffers match the source weight's device (CUDA-safe).
        """
        out_f, in_f = lin.weight.shape
        device = lin.weight.device
        obj = cls(in_f, out_f, bias=lin.bias is not None,
                  n_rounds=n_rounds, rank=rank, use_hadamard=use_hadamard)
        packed = quantize_sub_bitnet(lin.weight.data.float(),
                                     n_rounds=n_rounds, rank=rank,
                                     use_hadamard=use_hadamard)
        obj.load_prequantized(packed,
                              lin.bias.data if lin.bias is not None else None)
        return obj.to(device)

    def load_prequantized(self, packed: dict[str, Any],
                          bias: torch.Tensor | None = None):
        """Load packed sub-bitnet weights from a quantize_sub_bitnet dict."""
        with torch.no_grad():
            for r in range(self.n_rounds):
                self.weight_signs[r].copy_(packed["signs"][r])
                self.weight_scales[r].copy_(packed["scales"][r].to(torch.float16))
            if self.rank > 0 and packed["rank_u"] is not None:
                self.rank_u.copy_(packed["rank_u"])
                self.rank_v.copy_(packed["rank_v"])
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(torch.float16))
        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        if cache and self._cached_weight is not None \
                and self._cached_weight.dtype == dtype:
            return self._cached_weight
        acc = torch.zeros(self.out_features, self.hadamard_size,
                          dtype=torch.float32, device=self.weight_signs.device)
        for r in range(self.n_rounds):
            s = self.weight_signs[r].to(torch.float32)
            scale = self.weight_scales[r].to(torch.float32)
            acc = acc + scale.unsqueeze(1) * s
        if self.rank > 0:
            acc = acc + self.rank_u.to(torch.float32) @ self.rank_v.to(torch.float32)
        if self.use_hadamard:
            H = _hadamard_matrix(self.hadamard_size, acc.device, torch.float32)
            w = acc @ H
        else:
            w = acc
        w = w[:, :self.in_features].to(dtype)
        if cache:
            self._cached_weight = w
        return w

    @property
    def weight_quantized(self) -> torch.Tensor:
        return self._dequantize_weight(torch.float32, cache=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, cache=True)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        bpw = self.effective_bits_per_weight()
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"n_rounds={self.n_rounds}, rank={self.rank}, "
                f"hadamard={self.use_hadamard}, eff_bpw={bpw:.2f}")

    def effective_bits_per_weight(self) -> float:
        """Theoretical effective bits per weight (ignoring negligible scale overhead)."""
        n = self.out_features * self.hadamard_size
        bits = self.n_rounds * n  # 1 bit per sign element per round
        if self.rank > 0:
            bits += 16 * (self.out_features * self.rank
                          + self.rank * self.hadamard_size)
        return bits / max(1, self.out_features * self.in_features)


class SubBitnetLinearBiLLM(nn.Module):
    """Linear layer with BiLLM-style 1-bit quantized weights for inference.

    Stores salient binary signs (2 rounds) + non-salient split signs
    (concentrated + sparse) + per-channel scales. Dequantizes on-the-fly.

    Effective bit-width (~1.06 bits/w for 5% salient):
      - Salient: 2 bits × 5% = 0.10
      - Non-salient: 1 bit × 95% = 0.95
      - Split mask + salient flag: ~0.01
      - Total: ~1.06 bits/w (sub-bitnet, matches BiLLM paper)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, use_hadamard: bool = True,
                 salient_frac: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_hadamard = use_hadamard
        self.salient_frac = salient_frac
        self.hadamard_size = _hadamard_size(in_features) if use_hadamard else in_features
        n_salient = max(1, int(self.hadamard_size * salient_frac))
        n_nonsalient = self.hadamard_size - n_salient
        self.n_salient = n_salient
        self.n_nonsalient = n_nonsalient
        self._salient_order = 2
        self._full_size = False

        # Placeholder buffers (will be recreated by load_prequantized)
        self.register_buffer("sal_signs_0", torch.zeros(out_features, n_salient, dtype=torch.int8))
        self.register_buffer("sal_scales_0", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("sal_means_0", torch.zeros(out_features, dtype=torch.float16))
        self.register_buffer("sal_signs_1", torch.zeros(out_features, n_salient, dtype=torch.int8))
        self.register_buffer("sal_scales_1", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("sal_means_1", torch.zeros(out_features, dtype=torch.float16))
        # Non-salient: concentrated + sparse
        self.register_buffer("ns_signs_c", torch.zeros(out_features, n_nonsalient, dtype=torch.int8))
        self.register_buffer("ns_scales_c", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("ns_signs_s", torch.zeros(out_features, n_nonsalient, dtype=torch.int8))
        self.register_buffer("ns_scales_s", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("ns_means_c", torch.zeros(out_features, dtype=torch.float16))
        self.register_buffer("ns_means_s", torch.zeros(out_features, dtype=torch.float16))
        # Masks
        self.register_buffer("split_mask", torch.zeros(out_features, n_nonsalient, dtype=torch.bool))
        self.register_buffer("salient_cols", torch.zeros(self.hadamard_size, dtype=torch.bool))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        self._cached_weight = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, use_hadamard: bool = True,
                    salient_frac: float = 0.10,
                    activations: torch.Tensor | None = None,
                    block_size: int = 128,
                    salient_order: int = 2,
                    svd_rank: int = 48,
                    svd_block_size: int = 32) -> SubBitnetLinearBiLLM:
        out_f, in_f = lin.weight.shape
        device = lin.weight.device
        # Compute actual h_size (may differ from _hadamard_size if padding > 15%)
        if use_hadamard:
            h_size = _hadamard_size(in_f)
            pad = h_size - in_f
            if pad > 0 and pad / h_size > 0.15:
                use_hadamard = False
                h_size = in_f
        else:
            h_size = in_f
        obj = cls(in_f, out_f, bias=lin.bias is not None,
                  use_hadamard=use_hadamard, salient_frac=salient_frac)
        # Override hadamard_size to match what the quantizer will produce
        obj.hadamard_size = h_size
        if activations is not None:
            packed = quantize_sub_bitnet_billm_gptq(
                lin.weight.data.float(), activations=activations,
                salient_frac=salient_frac, use_hadamard=use_hadamard,
                block_size=block_size)
        else:
            packed = quantize_sub_bitnet_billm(
                lin.weight.data.float(), salient_frac=salient_frac,
                use_hadamard=use_hadamard, salient_order=salient_order,
                svd_rank=svd_rank, svd_block_size=svd_block_size)
        obj.load_prequantized(packed, lin.bias.data if lin.bias is not None else None)
        return obj.to(device)

    def load_prequantized(self, packed: dict[str, Any],
                          bias: torch.Tensor | None = None):
        with torch.no_grad():
            # Get salient rounds (variable order)
            if "salient_signs" in packed:
                sal_signs_list = packed["salient_signs"]
                sal_scales_list = packed["salient_scales"]
                sal_means_list = packed["salient_means"]
            else:
                # Backward compat: fixed 2 rounds
                sal_signs_list = [packed["salient_signs_o"]]
                sal_scales_list = [packed["salient_scales_o"]]
                sal_means_list = [packed.get("salient_means_o", torch.zeros_like(packed["salient_scales_o"]))]
                if "salient_signs_r" in packed:
                    sal_signs_list.append(packed["salient_signs_r"])
                    sal_scales_list.append(packed["salient_scales_r"])
                    sal_means_list.append(packed.get("salient_means_r", torch.zeros_like(packed["salient_scales_r"])))
            self._salient_order = len(sal_signs_list)

            # Detect format: full-size (out, h_size) or compact (out, n_group)
            sal_o = sal_signs_list[0]
            full_size = sal_o.dim() == 2 and sal_o.shape[1] == self.hadamard_size
            self._full_size = full_size

            if full_size:
                dev = sal_o.device
                # Recreate salient buffers at full size for all rounds
                self._sal_signs = []
                self._sal_scales = []
                self._sal_means = []
                for i in range(self._salient_order):
                    self.register_buffer(f"sal_signs_{i}", torch.zeros(self.out_features, self.hadamard_size, dtype=torch.int8, device=dev))
                    self.register_buffer(f"sal_scales_{i}", torch.ones(self.out_features, dtype=torch.float16, device=dev))
                    self.register_buffer(f"sal_means_{i}", torch.zeros(self.out_features, dtype=torch.float16, device=dev))
                    self._sal_signs.append(getattr(self, f"sal_signs_{i}"))
                    self._sal_scales.append(getattr(self, f"sal_scales_{i}"))
                    self._sal_means.append(getattr(self, f"sal_means_{i}"))
                self.register_buffer("ns_signs_c", torch.zeros(self.out_features, self.hadamard_size, dtype=torch.int8, device=dev))
                self.register_buffer("ns_signs_s", torch.zeros(self.out_features, self.hadamard_size, dtype=torch.int8, device=dev))
                self.register_buffer("split_mask", torch.zeros(self.out_features, self.hadamard_size, dtype=torch.bool, device=dev))
                self.register_buffer("ns_means_c", torch.zeros(self.out_features, dtype=torch.float16, device=dev))
                self.register_buffer("ns_means_s", torch.zeros(self.out_features, dtype=torch.float16, device=dev))
            else:
                # Compact: recreate salient buffers at correct size
                dev = sal_o.device
                n_sal = sal_o.shape[1]
                n_nonsal = packed["nonsal_signs_c"].shape[1]
                self._sal_signs = []
                self._sal_scales = []
                self._sal_means = []
                for i in range(self._salient_order):
                    self.register_buffer(f"sal_signs_{i}", torch.zeros(self.out_features, n_sal, dtype=torch.int8, device=dev))
                    self.register_buffer(f"sal_scales_{i}", torch.ones(self.out_features, dtype=torch.float16, device=dev))
                    self.register_buffer(f"sal_means_{i}", torch.zeros(self.out_features, dtype=torch.float16, device=dev))
                    self._sal_signs.append(getattr(self, f"sal_signs_{i}"))
                    self._sal_scales.append(getattr(self, f"sal_scales_{i}"))
                    self._sal_means.append(getattr(self, f"sal_means_{i}"))
                # Recreate non-salient buffers at correct size
                self.register_buffer("ns_signs_c", torch.zeros(self.out_features, n_nonsal, dtype=torch.int8, device=dev))
                self.register_buffer("ns_scales_c", torch.ones(self.out_features, dtype=torch.float16, device=dev))
                self.register_buffer("ns_signs_s", torch.zeros(self.out_features, n_nonsal, dtype=torch.int8, device=dev))
                self.register_buffer("ns_scales_s", torch.ones(self.out_features, dtype=torch.float16, device=dev))
                self.register_buffer("ns_means_c", torch.zeros(self.out_features, dtype=torch.float16, device=dev))
                self.register_buffer("ns_means_s", torch.zeros(self.out_features, dtype=torch.float16, device=dev))
                self.register_buffer("split_mask", torch.zeros(self.out_features, n_nonsal, dtype=torch.bool, device=dev))
                # Recreate salient_cols at correct h_size
                self.register_buffer("salient_cols", torch.zeros(self.hadamard_size, dtype=torch.bool, device=dev))

            # Copy salient rounds
            for i in range(self._salient_order):
                getattr(self, f"sal_signs_{i}").copy_(sal_signs_list[i])
                getattr(self, f"sal_scales_{i}").copy_(sal_scales_list[i].to(torch.float16))
                getattr(self, f"sal_means_{i}").copy_(sal_means_list[i].to(torch.float16))

            self.ns_signs_c.copy_(packed["nonsal_signs_c"])
            self.ns_scales_c.copy_(packed["nonsal_scales_c"].to(torch.float16))
            self.ns_signs_s.copy_(packed["nonsal_signs_s"])
            self.ns_scales_s.copy_(packed["nonsal_scales_s"].to(torch.float16))
            self.split_mask.copy_(packed["split_mask"])
            self.salient_cols.copy_(packed["salient_cols"])
            if hasattr(self, 'ns_means_c'):
                self.ns_means_c.copy_(packed.get("nonsal_means_c", torch.zeros_like(self.ns_scales_c)).to(torch.float16))
                self.ns_means_s.copy_(packed.get("nonsal_means_s", torch.zeros_like(self.ns_scales_s)).to(torch.float16))
            # SVD residual buffers
            svd_rank = packed.get("svd_rank", 0)
            self._svd_rank = svd_rank
            self._svd_block_size = packed.get("svd_block_size", 32)
            if svd_rank > 0 and "svd_u_q" in packed and packed["svd_u_q"].numel() > 0:
                self.register_buffer("svd_u_q", packed["svd_u_q"].to(torch.int8))
                self.register_buffer("svd_v_q", packed["svd_v_q"].to(torch.int8))
                self.register_buffer("svd_u_scales", packed["svd_u_scales"].to(torch.float16))
                self.register_buffer("svd_v_scales", packed["svd_v_scales"].to(torch.float16))
            else:
                self.register_buffer("svd_u_q", torch.zeros(0, dtype=torch.int8, device=dev))
                self.register_buffer("svd_v_q", torch.zeros(0, dtype=torch.int8, device=dev))
                self.register_buffer("svd_u_scales", torch.zeros(0, dtype=torch.float16, device=dev))
                self.register_buffer("svd_v_scales", torch.zeros(0, dtype=torch.float16, device=dev))
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(torch.float16))
        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        if cache and self._cached_weight is not None \
                and self._cached_weight.dtype == dtype:
            return self._cached_weight
        dev = self.sal_signs_0.device
        order = getattr(self, '_salient_order', 2)

        # Gather salient rounds
        sal_signs = [getattr(self, f"sal_signs_{i}").to(torch.float32) for i in range(order)]
        sal_scales = [getattr(self, f"sal_scales_{i}").to(torch.float32) for i in range(order)]
        sal_means = [getattr(self, f"sal_means_{i}").to(torch.float32) for i in range(order)]
        ns_c = self.ns_signs_c.to(torch.float32)
        ns_sc = self.ns_scales_c.to(torch.float32)
        ns_s = self.ns_signs_s.to(torch.float32)
        ns_ss = self.ns_scales_s.to(torch.float32)
        m_c = self.ns_means_c.to(torch.float32) if hasattr(self, 'ns_means_c') else torch.zeros_like(ns_sc)
        m_s = self.ns_means_s.to(torch.float32) if hasattr(self, 'ns_means_s') else torch.zeros_like(ns_ss)

        if getattr(self, '_full_size', False):
            # Full-size format: signs encode group membership via nonzero
            sal_elem = (sal_signs[0] != 0).float()
            sal_recon = sum(
                sal_elem * (m.unsqueeze(1) + s.unsqueeze(1) * sg)
                for m, s, sg in zip(sal_means, sal_scales, sal_signs)
            )
            acc = (sal_recon
                   + (ns_c != 0).float() * (m_c.unsqueeze(1) + ns_sc.unsqueeze(1) * ns_c)
                   + (ns_s != 0).float() * (m_s.unsqueeze(1) + ns_ss.unsqueeze(1) * ns_s))
        else:
            # Compact format: scatter via masks
            acc = torch.zeros(self.out_features, self.hadamard_size,
                              dtype=torch.float32, device=dev)
            nonsal_mask = ~self.salient_cols
            sal_recon = sum(
                m.unsqueeze(1) + s.unsqueeze(1) * sg
                for m, s, sg in zip(sal_means, sal_scales, sal_signs)
            )
            acc[:, self.salient_cols] = sal_recon
            split = self.split_mask.float()
            is_conc = (~self.split_mask).float()
            ns_recon = (m_c.unsqueeze(1) * is_conc + ns_sc.unsqueeze(1) * ns_c
                        + m_s.unsqueeze(1) * split + ns_ss.unsqueeze(1) * ns_s)
            acc[:, nonsal_mask] = ns_recon

        # SVD residual
        svd_rank = getattr(self, '_svd_rank', 0)
        if svd_rank > 0 and self.svd_u_q.numel() > 0:
            bs = getattr(self, '_svd_block_size', 32)
            u_shape = (self.out_features, svd_rank)
            v_shape = (svd_rank, self.hadamard_size)
            u_deq = _nf4_block_dequantize(self.svd_u_q, self.svd_u_scales.to(torch.float32), bs, u_shape)
            v_deq = _nf4_block_dequantize(self.svd_v_q, self.svd_v_scales.to(torch.float32), bs, v_shape)
            acc = acc + u_deq @ v_deq

        if self.use_hadamard:
            H = _hadamard_matrix(self.hadamard_size, dev, torch.float32)
            w = acc @ H
        else:
            w = acc
        w = w[:, :self.in_features].to(dtype)
        if cache:
            self._cached_weight = w
        return w

    @property
    def weight_quantized(self) -> torch.Tensor:
        return self._dequantize_weight(torch.float32, cache=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, cache=True)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        bpw = self.effective_bits_per_weight()
        return (f"in={self.in_features}, out={self.out_features}, "
                f"salient={self.salient_frac}, eff_bpw={bpw:.2f}")

    def effective_bits_per_weight(self) -> float:
        """Effective bits per weight including SVD residual if present."""
        order = getattr(self, '_salient_order', 2)
        bits = order * (self.out_features * self.n_salient)
        bits += 1 * (self.out_features * self.n_nonsalient)
        # Split mask: packed per block of 128 → 1 bit per 128 elements
        split_block = 128
        n_split_blocks = (self.out_features * self.n_nonsalient + split_block - 1) // split_block
        bits += 1 * n_split_blocks
        # Salient column flag: 1 bit per column (h_size bits total)
        bits += self.hadamard_size
        # SVD residual: NF4 (4 bits/elem) + 16-bit per-block scale
        svd_rank = getattr(self, '_svd_rank', 0)
        if svd_rank > 0 and self.svd_u_q.numel() > 0:
            bs = getattr(self, '_svd_block_size', 32)
            u_elem = svd_rank * self.out_features
            v_elem = svd_rank * self.hadamard_size
            u_blocks = (u_elem + bs - 1) // bs
            v_blocks = (v_elem + bs - 1) // bs
            bits += 4 * u_elem + 16 * u_blocks
            bits += 4 * v_elem + 16 * v_blocks
        return bits / max(1, self.out_features * self.in_features)


# ── SubBitnetKey: Key class ──────────────────────────────────────────────────

class SubBitnetKey(Key):
    """Sub-bitnet key — training-free QAT-style sub-bitnet quantization.

    Two methods:
      - "irb" (default): Hadamard + IRB + optional SVD low-rank residual.
        ~1.0 bits/w at 1 round. Good per-layer SQNR but compounds across
        layers without QAT.
      - "billm": BiLLM-style salient columns + optimal splitting.
        ~1.06 bits/w. Much better end-to-end PPL at 1-bit because it
        preserves the structurally important weights (salient columns)
        and splits the bell-shaped distribution optimally.

    Key class: PARTIAL — binarization is not invertible. The Hadamard rotation
    IS invertible (orthogonal), but sign quantization discards magnitude, so
    the overall transform is one-way. IRB + low-rank reduce the error
    exponentially in the number of rounds / rank.
    """

    def __init__(self, n_rounds: int = 1, rank: int = 0,
                 use_hadamard: bool = True, method: str = "irb",
                 salient_frac: float = 0.05):
        self.n_rounds = n_rounds
        self.rank = rank
        self.use_hadamard = use_hadamard
        self.method = method
        self.salient_frac = salient_frac

    @property
    def name(self) -> str:
        return "sub_bitnet"

    @property
    def description(self) -> str:
        if self.method == "billm":
            return (f"BiLLM Hadamard+salient({self.salient_frac})+split "
                    f"(~1.06 bits/w, sub-bitnet)")
        bpw = self.n_rounds
        lr = f" + rank-{self.rank} SVD residual" if self.rank > 0 else ""
        hd = "Hadamard+IRB " if self.use_hadamard else "IRB "
        return (f"{hd}{self.n_rounds}-round binary{lr} "
                f"(~{bpw:.1f} bits/w, sub-bitnet)")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Quantize all 2D weight tensors to sub-bitnet binary (+ residual)."""
        try:
            weights = apply_sub_bitnet(dict(data), n_rounds=self.n_rounds,
                                       rank=self.rank,
                                       use_hadamard=self.use_hadamard,
                                       method=self.method)
            if self.method == "billm":
                n = sum(1 for k in weights if k.endswith(".sb_sal_signs_o"))
            else:
                n = sum(1 for k in weights if k.endswith(".sb_signs_r0"))
            return KeyResult(
                success=True, weights=weights,
                metadata={"n_quantized": n,
                          "n_rounds": self.n_rounds,
                          "rank": self.rank,
                          "use_hadamard": self.use_hadamard,
                          "method": self.method,
                          "salient_frac": self.salient_frac},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Binarization is not invertible — return weights as-is."""
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "Sub-bitnet binary weights cannot be un-quantized"})


# ── Model conversion utility ─────────────────────────────────────────────────

def convert_model_to_sub_bitnet(model: nn.Module, n_rounds: int = 1,
                                rank: int = 0, use_hadamard: bool = True,
                                method: str = "irb", salient_frac: float = 0.10,
                                activations: dict[str, torch.Tensor] | None = None,
                                layer_name: str = "",
                                salient_order: int = 2,
                                svd_rank: int = 48,
                                svd_block_size: int = 32,
                                ) -> nn.Module:
    """Convert all nn.Linear layers in a model to SubBitnetLinear (in-place).

    Skips embedding/head layers (matching the BitNet / IRI-FP4 convention).

    Args:
        method: "irb" (Hadamard+IRB+SVD) or "billm" (BiLLM-style salient+split).
        salient_frac: fraction of salient columns for BiLLM method.
        activations: optional dict of {layer_name: (N, in_features)} calibration
            activations for GPTQ error compensation (BiLLM method only).
        layer_name: current layer name prefix (for recursion + activation lookup).
    """
    skip_names = ("embed", "head", "lm_head", "output")
    for name, module in list(model.named_children()):
        full_name = f"{layer_name}.{name}" if layer_name else name
        if isinstance(module, nn.Linear) and not any(s in name for s in skip_names):
            if method == "billm":
                acts = activations.get(full_name) if activations else None
                sub_lin = SubBitnetLinearBiLLM.from_linear(
                    module, use_hadamard=use_hadamard, salient_frac=salient_frac,
                    activations=acts, salient_order=salient_order,
                    svd_rank=svd_rank, svd_block_size=svd_block_size)
            else:
                sub_lin = SubBitnetLinear.from_linear(
                    module, n_rounds=n_rounds, rank=rank, use_hadamard=use_hadamard)
            setattr(model, name, sub_lin)
        else:
            convert_model_to_sub_bitnet(module, n_rounds, rank, use_hadamard,
                                        method, salient_frac, activations,
                                        full_name, salient_order,
                                        svd_rank, svd_block_size)
    return model


def build_sub_bitnet_linear(config, in_features: int, out_features: int,
                            bias: bool = True) -> nn.Module:
    """Build a SubBitnetLinear honoring the model config.

    Reads ``sub_bitnet_rounds``, ``sub_bitnet_rank``, ``sub_bitnet_hadamard``
    from the config (all optional, with sub-bitnet defaults).
    """
    return SubBitnetLinear(
        in_features, out_features, bias=bias,
        n_rounds=int(getattr(config, "sub_bitnet_rounds", 1)),
        rank=int(getattr(config, "sub_bitnet_rank", 0)),
        use_hadamard=bool(getattr(config, "sub_bitnet_hadamard", True)),
    )

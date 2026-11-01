"""Fused QK-Norm + RoPE Triton kernel.

Fuses two element-wise pre-attention operations into a single GPU kernel:
  1. RMSNorm on Q and K (per-head, along head_dim)
  2. RoPE (Rotary Position Embedding) on Q and K

Currently these are separate PyTorch ops with intermediate HBM reads/writes:
  q → RMSNorm → HBM → RoPE → HBM  (2 loads, 2 stores, 2 kernel launches)
  k → RMSNorm → HBM → RoPE → HBM  (2 loads, 2 stores, 2 kernel launches)

Fused kernel:
  q → RMSNorm + RoPE → HBM  (1 load, 1 store, 1 kernel launch)
  k → RMSNorm + RoPE → HBM  (1 load, 1 store, 1 kernel launch)

This halves the HBM traffic for Q/K preprocessing and reduces kernel launch
overhead. The attention computation itself remains on FlashAttention-2 (FA2)
via F.scaled_dot_product_attention — FA2 is already a fused kernel and
cannot be improved by manual Triton fusion.

The kernel processes one (batch, head, seq_pos) vector of head_dim elements
per Triton program. Since head_dim=64 for LFM2.5-1.2B, the entire vector
fits in registers — no shared memory needed.

Usage:
    from forge.decoding.fused_rope_qknorm import fused_qk_norm_rope
    q, k = fused_qk_norm_rope(q, k, q_norm_weight, k_norm_weight,
                               cos, sin, eps=1e-6)

Fallback: pure PyTorch when Triton is unavailable (CPU, or kernel compile fail).
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ─── PyTorch fallback ─────────────────────────────────────────────────

def _pytorch_qk_norm_rope(q: torch.Tensor, k: torch.Tensor,
                          q_weight: torch.Tensor, k_weight: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor,
                          eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback: separate RMSNorm then RoPE."""
    # RMSNorm on Q and K (along head_dim, last dimension)
    q_rms = q.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    q_normed = q * q_rms * q_weight
    k_rms = k.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    k_normed = k * k_rms * k_weight

    # RoPE: rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])
    def rotate_half(x):
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # cos/sin shape: (1, 1, T, dim) or (B, 1, T, dim) — broadcast with q/k
    q_out = (q_normed * cos) + (rotate_half(q_normed) * sin)
    k_out = (k_normed * cos) + (rotate_half(k_normed) * sin)
    return q_out, k_out


# ─── Triton kernel ────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_norm_rope_kernel(
        x_ptr,          # [B, n_heads, T, head_dim]
        out_ptr,        # [B, n_heads, T, head_dim]
        weight_ptr,     # [head_dim] — RMSNorm weight
        cos_ptr,        # [T, head_dim] — cos table (pre-sliced to seq positions)
        sin_ptr,        # [T, head_dim] — sin table
        eps,            # RMSNorm epsilon
        head_dim,       # D (compile-time constant via tl.constexpr)
        n_heads,        # number of heads in this tensor
        T,              # sequence length
        table_len,      # cos/sin position-table extent (row count)
        x_stride_b, x_stride_h, x_stride_t, x_stride_d,
        out_stride_b, out_stride_h, out_stride_t, out_stride_d,
        cos_stride_t, cos_stride_d,
        sin_stride_t, sin_stride_d,
        BLOCK_D: tl.constexpr,  # head_dim, power of 2
    ):
        """Fused RMSNorm + RoPE for one (batch, head, seq_pos) vector.

        Grid: (B * n_heads * T,)
        Each program handles head_dim elements.
        """
        pid = tl.program_id(0)
        # Decompose pid into (batch, head, seq_pos)
        b = pid // (n_heads * T)
        ht = pid % (n_heads * T)
        h = ht // T
        t = ht % T

        # Load x[b, h, t, :] — shape [head_dim]
        x_offset = b * x_stride_b + h * x_stride_h + t * x_stride_t
        x_ptrs = x_ptr + x_offset + tl.arange(0, BLOCK_D) * x_stride_d
        x = tl.load(x_ptrs).to(tl.float32)

        # RMSNorm: x_normed = x * weight / sqrt(mean(x^2) + eps)
        rms = tl.sqrt(tl.sum(x * x) / head_dim + eps)
        x_normed = x / rms

        # Load weight[head_dim]
        w_ptrs = weight_ptr + tl.arange(0, BLOCK_D)
        weight = tl.load(w_ptrs).to(tl.float32)
        x_normed = x_normed * weight

        # RoPE: out = x_normed * cos + rotate_half(x_normed) * sin
        # rotate_half: [x[d/2:], -x[:d/2]] → for d=64: [x[32:64], -x[0:32]]
        half = BLOCK_D // 2
        d_idx = tl.arange(0, BLOCK_D)
        # rotate_half: indices [d/2, d/2+1, ..., d-1, 0, 1, ..., d/2-1]
        rotate_idx = (d_idx + half) % BLOCK_D
        # Gather-free rotate: re-load x with shifted pointers (L2 hit) and
        # re-apply norm+weight with the same rotated weight indices.
        # tl.gather needs an `axis` kwarg on newer triton and breaks
        # inductor tracing, so avoid it entirely.
        rot_ptrs = x_ptr + x_offset + rotate_idx * x_stride_d
        x_rot_raw = tl.load(rot_ptrs).to(tl.float32)
        weight_rot = tl.load(weight_ptr + rotate_idx).to(tl.float32)
        x_rotated = (x_rot_raw / rms) * weight_rot
        # negate the first half: rotate_half(x) = [-x[d/2:], x[:d/2]]
        negate_mask = d_idx < half
        x_rotated = tl.where(negate_mask, -x_rotated, x_rotated)

        # Load cos[t, :] and sin[t, :]. t derives from x's T, not from the
        # table extent — a short table would silently read out of bounds
        # (OOB "future-token" read into whatever follows the table in HBM).
        # The wrapper asserts T <= table_len; this clamp is the in-kernel
        # bound for the same read.
        t_idx = tl.minimum(t, table_len - 1)
        cos_ptrs = cos_ptr + t_idx * cos_stride_t + tl.arange(0, BLOCK_D) * cos_stride_d
        sin_ptrs = sin_ptr + t_idx * sin_stride_t + tl.arange(0, BLOCK_D) * sin_stride_d
        cos_val = tl.load(cos_ptrs).to(tl.float32)
        sin_val = tl.load(sin_ptrs).to(tl.float32)

        # Fused output
        out = x_normed * cos_val + x_rotated * sin_val

        # Store out[b, h, t, :]
        out_offset = b * out_stride_b + h * out_stride_h + t * out_stride_t
        out_ptrs = out_ptr + out_offset + tl.arange(0, BLOCK_D) * out_stride_d
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty))


    def _triton_fused_norm_rope(x: torch.Tensor, weight: torch.Tensor,
                                cos: torch.Tensor, sin: torch.Tensor,
                                eps: float = 1e-6) -> torch.Tensor:
        """Apply fused RMSNorm + RoPE to a single tensor (Q or K).

        Args:
            x: [B, n_heads, T, head_dim]
            weight: [head_dim] — RMSNorm weight
            cos: [T, head_dim] — cos table (already sliced to seq positions)
            sin: [T, head_dim] — sin table
            eps: RMSNorm epsilon

        Returns:
            [B, n_heads, T, head_dim] — normed + rotated tensor
        """
        B, n_heads, T, head_dim = x.shape

        # head_dim must be power of 2 for Triton
        assert head_dim & (head_dim - 1) == 0, \
            f"head_dim must be power of 2, got {head_dim}"

        # Ensure contiguous (pointer args are walked by raw strides below)
        x = x.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()
        weight = weight.contiguous()
        out = torch.empty_like(x)

        # Bounds: the kernel reads the cos/sin position tables at row t for
        # t in [0, T) — a table shorter than T is an out-of-bounds read of
        # "future tokens" past the table end. weight must cover head_dim
        # and the tables must be [>=T, head_dim].
        table_len = cos.shape[0]
        assert cos.dim() == 2 and sin.dim() == 2, \
            f"cos/sin must be [T, head_dim], got {cos.shape}/{sin.shape}"
        assert T <= table_len, \
            f"T={T} exceeds cos table length {table_len}"
        assert T <= sin.shape[0], \
            f"T={T} exceeds sin table length {sin.shape[0]}"
        assert cos.shape[1] >= head_dim and sin.shape[1] >= head_dim, \
            f"cos/sin head_dim mismatch: {cos.shape}/{sin.shape} vs {head_dim}"
        assert weight.numel() >= head_dim, \
            f"weight has {weight.numel()} elems < head_dim {head_dim}"

        grid = (B * n_heads * T,)
        _fused_norm_rope_kernel[grid](
            x, out, weight, cos, sin,
            eps, head_dim, n_heads, T, table_len,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            BLOCK_D=head_dim,
        )
        return out

else:
    _triton_fused_norm_rope = None


# ─── Public API ───────────────────────────────────────────────────────

def fused_qk_norm_rope(q: torch.Tensor, k: torch.Tensor,
                       q_weight: torch.Tensor, k_weight: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor,
                       eps: float = 1e-6,
                       use_triton: bool = True
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK-Norm (RMSNorm) + RoPE for Q and K tensors.

    Applies per-head RMSNorm followed by Rotary Position Embedding to both
    Q and K in a single fused Triton kernel (one launch per tensor).

    Args:
        q: [B, n_heads, T, head_dim] — query tensor (pre-norm, pre-RoPE)
        k: [B, n_kv_heads, T, head_dim] — key tensor (pre-norm, pre-RoPE)
        q_weight: [head_dim] — RMSNorm weight for Q
        k_weight: [head_dim] — RMSNorm weight for K
        cos: [T, head_dim] or [1, T, head_dim] — cos table (sliced to positions)
        sin: [T, head_dim] or [1, T, head_dim] — sin table
        eps: RMSNorm epsilon (default 1e-6)
        use_triton: use Triton kernel if available (default True)

    Returns:
        (q_out, k_out): both [B, n_heads/kv_heads, T, head_dim], normed + rotated

    Note:
        The cos/sin tables must be pre-sliced to the correct sequence positions
        (offset : offset + T). The caller (RotaryEmbedding) normally handles this.
    """
    # Squeeze cos/sin to [T, head_dim] if needed
    if cos.dim() > 2:
        cos = cos.squeeze(0).squeeze(0) if cos.dim() == 4 else cos.squeeze(0)
    if sin.dim() > 2:
        sin = sin.squeeze(0).squeeze(0) if sin.dim() == 4 else sin.squeeze(0)

    # Try Triton path
    if use_triton and _TRITON_AVAILABLE and q.is_cuda and _triton_fused_norm_rope is not None:
        try:
            q_out = _triton_fused_norm_rope(q, q_weight, cos, sin, eps)
            k_out = _triton_fused_norm_rope(k, k_weight, cos, sin, eps)
            return q_out, k_out
        except Exception:
            # Fall back to PyTorch on any Triton error (compile fail, OOM, etc.)
            logger.debug("Triton fused QK-norm+RoPE failed, falling back to PyTorch", exc_info=True)

    # PyTorch fallback
    # Expand cos/sin for broadcasting: [1, 1, T, head_dim]
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    return _pytorch_qk_norm_rope(q, k, q_weight, k_weight, cos_b, sin_b, eps)


def fused_norm_rope_single(x: torch.Tensor, weight: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor,
                           eps: float = 1e-6,
                           use_triton: bool = True) -> torch.Tensor:
    """Fused RMSNorm + RoPE for a single tensor (Q or K).

    Convenience wrapper for applying the fused kernel to one tensor.

    Args:
        x: [B, n_heads, T, head_dim]
        weight: [head_dim] — RMSNorm weight
        cos: [T, head_dim] — cos table
        sin: [T, head_dim] — sin table
        eps: RMSNorm epsilon
        use_triton: use Triton if available

    Returns:
        [B, n_heads, T, head_dim] — normed + rotated
    """
    if cos.dim() > 2:
        cos = cos.squeeze(0).squeeze(0) if cos.dim() == 4 else cos.squeeze(0)
    if sin.dim() > 2:
        sin = sin.squeeze(0).squeeze(0) if sin.dim() == 4 else sin.squeeze(0)

    if use_triton and _TRITON_AVAILABLE and x.is_cuda and _triton_fused_norm_rope is not None:
        try:
            return _triton_fused_norm_rope(x, weight, cos, sin, eps)
        except Exception:
            logger.debug("Triton fused norm+RoPE failed, falling back to PyTorch", exc_info=True)

    # PyTorch fallback
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    x_normed = x * rms * weight

    def rotate_half(t):
        d = t.shape[-1]
        return torch.cat((-t[..., d // 2:], t[..., :d // 2]), dim=-1)

    return (x_normed * cos_b) + (rotate_half(x_normed) * sin_b)

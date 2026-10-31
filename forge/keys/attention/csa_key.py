"""Compressed Sparse Attention (CSA) — DeepSeek-V4 long-context efficiency.

DeepSeek-V4 combines CSA (for long context) and HCA (heavily compressed
attention for short context) in a hybrid architecture. CSA enables
million-token context efficiently by attending to only a SUBSET of
positions, selected by a learned or entropy-based router.

CSA mechanism:
  1. Compute a relevance score for each (query, key) pair using a lightweight
     router (e.g., dot product with a learned query embedding).
  2. Select the top-k most relevant positions per query.
  3. Attend only to those positions (sparse attention).

This reduces attention complexity from O(S^2) to O(S * k) where k << S.

Two router modes:
  - "learned": a small linear layer scores (q, k) pairs.
  - "entropy": use the entropy of the attention distribution to adaptively
    select k (high-entropy = attend to more, low-entropy = attend to fewer).

Hybrid CSA+HCA:
  - CSA for layers 0-14 (long-range, sparse).
  - HCA (existing MLA with KV compression) for layers 15-27 (short-range).
  - Config: attention_pattern="csa_hca_hybrid".

Usage:
    from forge.keys.attention.csa_key import CSAAttention

    # In a transformer block:
    csa = CSAAttention(d_model, n_heads, top_k=256)
    out = csa(q, k, v, is_causal=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAAttention(nn.Module):
    """Compressed Sparse Attention with top-k position selection.

    For each query position, selects the top-k most relevant key positions
    via a lightweight router, then attends only to those positions.

    Args:
        d_model: model dimension.
        n_heads: number of attention heads.
        top_k: number of positions to attend to per query (default 256).
        head_dim: dimension per head (default d_model // n_heads).
        router_mode: "learned" (linear router) or "dot" (q·k score).
    """

    def __init__(self, d_model: int, n_heads: int, top_k: int = 256,
                 head_dim: int | None = None, router_mode: str = "dot"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.top_k = top_k
        self.router_mode = router_mode
        self.scale = self.head_dim ** -0.5

        if router_mode == "learned":
            # Lightweight router: scores (q, k) pairs via a small linear layer.
            self.router = nn.Linear(self.head_dim * 2, 1, bias=False)
            # Identity-ish init: weight = [I; I] / 2 → score ≈ (q·w + k·w) / 2
            with torch.no_grad():
                nn.init.zeros_(self.router.weight)
                # Set first half to +1, second half to +1 → score = q + k
                # (not ideal, but it's a starting point)
                self.router.weight.data[:self.head_dim] = 1.0 / self.head_dim
                self.router.weight.data[self.head_dim:] = 1.0 / self.head_dim

    def _compute_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute relevance scores for position selection.

        Args:
            q: (B, H, Sq, Dh)
            k: (B, H, Sk, Dh)

        Returns:
            (B, H, Sq, Sk) relevance scores.
        """
        if self.router_mode == "dot":
            # Simple dot product (same as attention scores).
            return torch.matmul(q, k.transpose(-2, -1)) * self.scale
        else:
            # Learned router: score = router([q, k]) per (q, k) pair.
            B, H, Sq, Dh = q.shape
            Sk = k.shape[2]
            # Expand for pairwise: (B, H, Sq, Sk, 2*Dh)
            q_exp = q.unsqueeze(3).expand(B, H, Sq, Sk, Dh)
            k_exp = k.unsqueeze(2).expand(B, H, Sq, Sk, Dh)
            pairs = torch.cat([q_exp, k_exp], dim=-1)  # (B, H, Sq, Sk, 2*Dh)
            scores = self.router(pairs).squeeze(-1)  # (B, H, Sq, Sk)
            return scores

    def _select_topk_causal(self, scores: torch.Tensor,
                            is_causal: bool = True) -> torch.Tensor:
        """Select top-k positions per query, respecting causal masking.

        Args:
            scores: (B, H, Sq, Sk) relevance scores.
            is_causal: if True, only consider positions <= query position.

        Returns:
            (B, H, Sq, top_k) indices of the top-k positions.
        """
        _B, _H, Sq, Sk = scores.shape

        if is_causal:
            # Mask future positions with -inf before top-k.
            causal_mask = torch.triu(
                torch.ones(Sq, Sk, device=scores.device, dtype=torch.bool),
                diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0),
                                         float("-inf"))

        k = min(self.top_k, Sk)
        # top-k indices per query.
        _, indices = scores.topk(k, dim=-1, sorted=False)
        return indices

    def _gather_attention(self, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor, indices: torch.Tensor,
                          is_causal: bool = True) -> torch.Tensor:
        """Gather top-k keys/values and compute attention.

        Args:
            q: (B, H, Sq, Dh)
            k: (B, H, Sk, Dh)
            v: (B, H, Sk, Dh)
            indices: (B, H, Sq, top_k) — selected key positions per query.
            is_causal: if True, ensure no future positions are selected.

        Returns:
            (B, H, Sq, Dh) attention output.
        """
        B, H, Sq, Dh = q.shape
        Sk = k.shape[2]
        k_sel = self.top_k

        # Gather selected keys and values: (B, H, Sq, top_k, Dh)
        # indices: (B, H, Sq, top_k) -> expand for gather
        idx_exp = indices.unsqueeze(-1).expand(B, H, Sq, k_sel, Dh)
        k_gathered = k.unsqueeze(2).expand(B, H, Sq, Sk, Dh).gather(3, idx_exp)
        v_gathered = v.unsqueeze(2).expand(B, H, Sq, Sk, Dh).gather(3, idx_exp)

        # Compute attention scores on selected positions only.
        # q: (B, H, Sq, Dh) -> (B, H, Sq, 1, Dh)
        # k_gathered: (B, H, Sq, top_k, Dh)
        attn_scores = torch.einsum("bhsd,bhskd->bhsk", q, k_gathered) * self.scale

        # Causal masking within selected positions: set scores where
        # index > query position to -inf.
        if is_causal:
            # indices: (B, H, Sq, top_k) — key positions
            # query positions: 0..Sq-1
            q_pos = torch.arange(Sq, device=q.device).view(1, 1, Sq, 1)
            future_mask = indices > q_pos  # (B, H, Sq, top_k)
            attn_scores = attn_scores.masked_fill(future_mask, float("-inf"))

        # Softmax over selected positions.
        attn = F.softmax(attn_scores, dim=-1)
        # Handle all-masked rows (query 0 with causal may have no valid keys
        # if top_k > 1 and all indices are future). Set to 0.
        attn = torch.nan_to_num(attn, nan=0.0)

        # Apply attention to values: (B, H, Sq, top_k) @ (B, H, Sq, top_k, Dh)
        out = torch.einsum("bhsk,bhskd->bhsd", attn, v_gathered)
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                is_causal: bool = True) -> torch.Tensor:
        """Compressed Sparse Attention forward.

        Args:
            q: (B, n_heads, seq_q, head_dim)
            k: (B, n_heads, seq_kv, head_dim)
            v: (B, n_heads, seq_kv, head_dim)
            is_causal: apply causal masking.

        Returns:
            (B, n_heads, seq_q, head_dim) attention output.
        """
        q.shape[2]
        Sk = k.shape[2]

        # If sequence is shorter than top_k, use full attention (no sparsity).
        if Sk <= self.top_k:
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Compute relevance scores for position selection.
        scores = self._compute_scores(q, k)

        # Select top-k positions per query (causal-aware).
        with torch.no_grad():
            indices = self._select_topk_causal(scores, is_causal=is_causal)

        # Gather and attend.
        out = self._gather_attention(q, k, v, indices, is_causal=is_causal)
        return out


class CSAKey:
    """CSA configuration key (not a standard Key — no weight conversion).

    This is a configuration helper that determines which layers use CSA
    vs HCA (MLA) in the hybrid architecture.
    """

    def __init__(self, n_layers: int, csa_layers: int | None = None):
        """
        Args:
            n_layers: total number of layers.
            csa_layers: number of layers using CSA (from layer 0).
                Default: first half. Remaining layers use HCA (MLA).
        """
        self.n_layers = n_layers
        if csa_layers is None:
            csa_layers = n_layers // 2
        self.csa_layers = csa_layers

    def get_layer_pattern(self, layer_idx: int) -> str:
        """Get the attention pattern for a given layer.

        Returns:
            "csa" for long-range layers, "hca" for short-range layers.
        """
        if layer_idx < self.csa_layers:
            return "csa"
        return "hca"

    def get_config(self) -> dict:
        """Get the full configuration dict."""
        return {
            "n_layers": self.n_layers,
            "csa_layers": self.csa_layers,
            "hca_layers": self.n_layers - self.csa_layers,
            "pattern": "csa_hca_hybrid",
        }

"""Differential Attention (Diff-Transformer) — attention-noise cancellation.

Splits each query/key head into two groups, computes two softmax attention
maps, and takes their difference. Like a differential amplifier, common-mode
attention noise cancels out, producing sparse, focused patterns. Empirically
matches standard Transformer quality at ~60-65% of the parameters
(Diff Transformer, Ye et al., ICLR 2025).

Module design (config-driven, works at any scale):
  - q/k proj: d_model -> n_heads * head_dim * 2 (two groups per head)
  - v proj:   d_model -> n_heads * head_dim
  - attn_h = softmax(q1 k1^T / sqrt(d)) - lambda_h * softmax(q2 k2^T / sqrt(d))
  - per-head RMSNorm + (1 - lambda_init) * exp(lambda_init) head scaling
  - lambda: learnable per-head, initialized to the paper's layer-dependent
    lambda_init (lossless-ish warm start; duplicated GQA weights via Key).

Supports the pre-allocated KV cache path used by ConfigurableResearchLLM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys._tensor_utils import _repeat_kv as _repeat_kv_shared
from forge.keys.misc.base import Key, KeyClass, KeyResult
from forge.model_loader import RotaryEmbedding


def paper_lambda_init(n_layers: int, layer_idx: int) -> float:
    """Diff-Transformer λ_init: 0.8 - 0.6 * exp(-0.01 * (L - 1 - l))."""
    return 0.8 - 0.6 * math.exp(-0.01 * (n_layers - 1 - layer_idx))


class DifferentialAttention(nn.Module):
    """Differential attention module (config-compatible with GQA slot)."""

    def __init__(self, d_model: int = 2048, n_heads: int = 32,
                 n_kv_heads: int | None = None, max_seq_len: int = 32768,
                 base: float = 1_000_000.0, rope_scaling: dict | None = None,
                 use_qk_norm: bool = False, attn_bias: bool = False,
                 n_layers: int = 16, layer_idx: int = 0,
                 lambda_init: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.layer_idx = layer_idx

        # Two groups per head: 2x head dim on q/k. v stays GQA-shaped
        # (n_kv_heads) and is repeated to n_heads on read.
        q_dim = n_heads * self.head_dim * 2
        k_dim = self.n_kv_heads * self.head_dim * 2
        v_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, q_dim, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, k_dim, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, v_dim, bias=attn_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len,
                                    base=base, rope_scaling=rope_scaling)

        # QK-norm (LFM2.5 uses trained q/k layernorms — required for a
        # lossless GQA->diff conversion). Identity weights => skipped.
        self.use_qk_norm = use_qk_norm
        self._qk_norm_identity = True
        if use_qk_norm:
            from forge.model_loader import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Learnable per-head lambda. lambda_init=0.0 selects IDENTITY mode:
        # the second attention map is skipped entirely, so with duplicated
        # GQA rows the module computes exactly standard attention (bit-exact
        # warm start). Training moves lambda off 0 and enables the full
        # differential mechanism.
        self.lambda_init = (lambda_init if lambda_init is not None
                            else paper_lambda_init(n_layers, layer_idx))
        self.lambda_param = nn.Parameter(
            torch.full((n_heads,), self.lambda_init, dtype=torch.float32))
        self._identity = (self.lambda_init == 0.0)

        # Per-head RMSNorm + scaling factor (paper eq. 7); identity bypasses.
        self.rms_norm = nn.RMSNorm(self.head_dim)
        self.scale = (1 - self.lambda_init) * math.exp(self.lambda_init)

        # Memoized contiguous group-1 weights for the identity path.
        self._q1_cache: torch.Tensor | None = None
        self._k1_cache: torch.Tensor | None = None

    def set_identity(self, identity: bool):
        """Toggle identity (lossless) vs. full differential mode."""
        self._identity = bool(identity)
        if identity:
            self._q1_cache = None
            self._k1_cache = None

    def _group1_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Contiguous GQA-shaped q/k weights (group-1 rows of each head).

        The doubled projection is head-interleaved ([g1_h; g2_h] per head),
        so group-1 rows are strided. Extracting them once into a contiguous
        (n_heads*hd, d) tensor reproduces the ORIGINAL GQA weight exactly —
        identical GEMM shapes -> identical cuBLAS kernel -> bit-exact.
        """
        if self._q1_cache is None:
            w = self.q_proj.weight
            self._q1_cache = (w.view(self.n_heads, 2 * self.head_dim, -1)
                              [:, : self.head_dim]
                              .reshape(self.n_heads * self.head_dim, -1)
                              .contiguous())
        if self._k1_cache is None:
            w = self.k_proj.weight
            self._k1_cache = (w.view(self.n_kv_heads, 2 * self.head_dim, -1)
                              [:, : self.head_dim]
                              .reshape(self.n_kv_heads * self.head_dim, -1)
                              .contiguous())
        return self._q1_cache, self._k1_cache

    def _repeat_kv(self, x):
        return _repeat_kv_shared(x, self.n_rep)

    def forward(self, x, past_key_value=None, use_cache=False,
                preallocated_cache=None, layer_idx: int = 0,
                attention_bias: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor | None = None):
        B, T, _C = x.shape
        hd = self.head_dim

        if self._identity:
            # ── Bit-exact GQA-equivalent path ────────────────────────────
            # Use the memoized contiguous group-1 weights: identical GEMM
            # shapes to the GQA attention this module was converted from,
            # so cuBLAS selects the SAME kernel and results are bit-exact.
            qw, kw = self._group1_weights()
            q = F.linear(x, qw, self.q_proj.bias).view(
                B, T, self.n_heads, hd).transpose(1, 2)
            k = F.linear(x, kw, self.k_proj.bias).view(
                B, T, self.n_kv_heads, hd).transpose(1, 2)
            v = self.v_proj(x).view(
                B, T, self.n_kv_heads, hd).transpose(1, 2)
            if self.use_qk_norm and not self._qk_norm_identity:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if preallocated_cache is not None:
                past_len = preallocated_cache.position
                q = self.rope(q, offset=past_len, position_ids=position_ids)
                k = self.rope(k, offset=past_len, position_ids=position_ids)
                # Identity mode only ever reads group 1 (the first hd of the
                # 2*hd slot); write just that half — avoids 2 cats/layer
                # (the biggest per-token CPU overhead in decode). Full mode
                # overwrites both halves when lambda != 0.
                pos = past_len
                kc = preallocated_cache.k_caches[layer_idx]
                vc = preallocated_cache.v_caches[layer_idx]
                kc[:, :, pos:pos + T, :hd] = k
                vc[:, :, pos:pos + T, :hd] = v
                k = kc[:, :, :past_len + T, :hd].contiguous()
                v = vc[:, :, :past_len + T, :hd].contiguous()
            else:
                past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
                q = self.rope(q, offset=past_len, position_ids=position_ids)
                k = self.rope(k, offset=past_len, position_ids=position_ids)
                if past_key_value is not None:
                    k = torch.cat([past_key_value[0], k], dim=-2)
                    v = torch.cat([past_key_value[1], v], dim=-2)

            new_kv = (k, v) if use_cache else None
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
            if attention_bias is not None:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_bias)
            else:
                out = F.scaled_dot_product_attention(q, k, v,
                                                     is_causal=T > 1)
            out = out.transpose(1, 2).reshape(B, T, -1)
            return self.out_proj(out), new_kv

        # ── Full differential path ───────────────────────────────────────
        q = self.q_proj(x).view(B, T, self.n_heads, 2 * hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, 2 * hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)

        q1, q2 = q[..., :hd], q[..., hd:]
        k1, k2 = k[..., :hd], k[..., hd:]

        # QK-norm on both groups (mirrors GQA semantics; with duplicated
        # rows both groups get identical norms -> lossless conversion).
        if self.use_qk_norm and not self._qk_norm_identity:
            q1 = self.q_norm(q1)
            q2 = self.q_norm(q2)
            k1 = self.k_norm(k1)
            k2 = self.k_norm(k2)

        if preallocated_cache is not None:
            past_len = preallocated_cache.position
            q1 = self.rope(q1, offset=past_len, position_ids=position_ids)
            q2 = self.rope(q2, offset=past_len, position_ids=position_ids)
            k1 = self.rope(k1, offset=past_len, position_ids=position_ids)
            k2 = self.rope(k2, offset=past_len, position_ids=position_ids)
            # Cache slots are 2*hd wide (k stores both groups); v is stored
            # duplicated into the same layout and read back at single width.
            preallocated_cache.append(
                layer_idx, k, v.repeat_interleave(2, dim=-1))
            k = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
            v = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]
            v = v[..., :hd]
            k1 = k[..., :hd]
            k2 = k[..., hd:]
        else:
            past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            q1 = self.rope(q1, offset=past_len, position_ids=position_ids)
            q2 = self.rope(q2, offset=past_len, position_ids=position_ids)
            k1 = self.rope(k1, offset=past_len, position_ids=position_ids)
            k2 = self.rope(k2, offset=past_len, position_ids=position_ids)
            if past_key_value is not None:
                k_old, v_old = past_key_value
                k = torch.cat([k_old, k], dim=-2)
                v = torch.cat([v_old, v], dim=-2)
                k1 = k[..., :hd]
                k2 = k[..., hd:]

        new_kv = (k, v) if use_cache else None

        k1 = self._repeat_kv(k1)
        k2 = self._repeat_kv(k2)
        v = self._repeat_kv(v)

        if attention_bias is not None:
            a1 = F.scaled_dot_product_attention(
                q1, k1, v, attn_mask=attention_bias)
            a2 = F.scaled_dot_product_attention(
                q2, k2, v, attn_mask=attention_bias)
        else:
            a1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=T > 1)
            a2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=T > 1)
        lam = self.lambda_param.view(1, self.n_heads, 1, 1).to(x.dtype)
        diff = a1 - lam * a2
        # Per-head RMSNorm + head scaling (paper eq. 7).
        out = self.rms_norm(diff.float()).to(x.dtype) * self.scale

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), new_kv


# ---------------------------------------------------------------------------
# Key: GQA -> DifferentialAttention weight transform (warm start)
# ---------------------------------------------------------------------------

def _dup_rows(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Duplicate per-head row blocks: GQA (n_heads*hd, d) -> diff (2*n_heads*hd, d).

    The diff module views q/k rows as (n_heads, 2*hd): head h holds group 1
    in rows [64h : 64h+hd] and group 2 in [64h+hd : 64h+2hd]. Duplicating at
    HEAD granularity (h0,h0,h1,h1,...) puts the original rows in BOTH groups
    of every head — the lossless warm start (q1 == q2 == original).
    """
    hd = t.shape[0] // n_heads
    return (t.view(n_heads, hd, -1).repeat_interleave(2, dim=0)
            .reshape(-1, t.shape[-1]))


def _avg_rows(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of _dup_rows: average the duplicated head blocks back."""
    hd = t.shape[0] // (2 * n_heads)
    return (t.view(n_heads, 2, hd, -1).mean(dim=1)
            .reshape(n_heads * hd, -1))


class DifferentialAttentionKey(Key):
    """Transform GQA attention weights into DifferentialAttention weights.

    forward: duplicate q/k projection rows (q1=q2, k1=k2 warm start) and add
    the per-head lambda (paper init). reverse: average the duplicated rows.
    Key class: PARTIAL (lambda split is one-directional).
    """

    def __init__(self, n_layers: int = 16, n_heads: int = 32,
                 identity: bool = True):
        """identity=True: lambda=0 => lossless GQA->diff warm start."""
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.identity = identity

    @property
    def name(self) -> str:
        return "differential_attention"

    @property
    def description(self) -> str:
        mode = "identity (lossless warm start)" if self.identity else "paper init"
        return f"GQA -> Diff-Transformer ({mode})"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if k.startswith("blocks.") and ".attn.q_proj.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)
            if n_layers == 0:
                n_layers = self.n_layers

            for i in range(n_layers):
                base = f"blocks.{i}.attn"
                qk = f"{base}.q_proj.weight"
                if qk in state:
                    # head granularity: q uses n_heads, k uses n_kv_heads
                    # (inferred from row counts since hd = q_rows / n_heads).
                    hd = state[qk].shape[0] // self.n_heads
                    state[qk] = _dup_rows(state[qk], self.n_heads)
                    kk = f"{base}.k_proj.weight"
                    if kk in state:
                        n_kv = state[kk].shape[0] // hd
                        state[kk] = _dup_rows(state[kk], n_kv)
                if self.identity:
                    lam = torch.zeros(self.n_heads, dtype=torch.float32)
                else:
                    lam = torch.full(
                        (self.n_heads,), paper_lambda_init(n_layers, i),
                        dtype=torch.float32)
                state[f"{base}.lambda_param"] = lam
            return KeyResult(success=True, weights=state,
                             metadata={"n_layers": n_layers,
                                       "identity": self.identity})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(weights)
            # Infer head counts: q uses n_heads; k uses n_kv_heads
            # (n_kv = k_rows / hd, hd = q_rows / (2 * n_heads)).
            q_rows = None
            for i in range(self.n_layers):
                qk = f"blocks.{i}.attn.q_proj.weight"
                if qk in state:
                    q_rows = state[qk].shape[0] // 2
                    break
            for k in list(state.keys()):
                if ".lambda_param" in k:
                    del state[k]
                elif ".attn.q_proj.weight" in k:
                    state[k] = _avg_rows(state[k], self.n_heads)
                elif ".attn.k_proj.weight" in k and q_rows:
                    hd = q_rows // self.n_heads
                    n_kv = state[k].shape[0] // (2 * hd)
                    state[k] = _avg_rows(state[k], n_kv)
            return KeyResult(success=True, weights=state)
        except Exception as e:
            return KeyResult(success=False, error=str(e))

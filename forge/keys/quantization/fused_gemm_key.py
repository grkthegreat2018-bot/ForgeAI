"""Fused QKV and Gate-Up GEMM projections.

Fuses separate Q/K/V projections into a single GEMM and separate
W_gate/W_up FFN projections into a single GEMM. Halves kernel launches
and lets cuBLAS pick a single efficient large GEMM instead of 2-3 small ones.

For decode (batch=1, seq=1), this eliminates 2 kernel launches per attention
layer and 1 per FFN layer — on a 16-layer model with 6 attention layers,
that's 12 + 6 = 18 fewer launches per token.

Weight layout:
  Fused QKV: [(n_heads + 2*n_kv_heads) * head_dim, d_model]
    Q rows: [0 : n_heads*hd]
    K rows: [n_heads*hd : n_heads*hd + n_kv_heads*hd]
    V rows: [n_heads*hd + n_kv_heads*hd : ...]
  Fused GateUp: [2 * hidden_dim, d_model]
    Gate rows: [0 : hidden_dim]
    Up rows:   [hidden_dim : 2*hidden_dim]

Compatible with BitNet (ternary QAT) and W8A8 (INT8 tensor-core GEMM).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedQKVLinear(nn.Module):
    """Fused Q/K/V projection in a single GEMM.

    Stores weights as a single (q_dim + k_dim + v_dim, d_model) matrix.
    Splits the output into Q, K, V after the GEMM.
    """

    def __init__(self, d_model: int, q_dim: int, k_dim: int, v_dim: int,
                 bias: bool = False, g_dim: int = 0):
        super().__init__()
        self.d_model = d_model
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        # d2: LFM3 per-head output gate rides in the SAME GEMM — the gate
        # reads the block input just like q/k/v, so its n_heads rows are
        # appended after v (Qwen3-Next fused-QKV+gate pattern).
        self.g_dim = g_dim
        self.out_dim = q_dim + k_dim + v_dim + g_dim
        self.weight = nn.Parameter(torch.empty(self.out_dim, d_model))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_dim))
            nn.init.uniform_(self.bias, -1.0 / d_model ** 0.5, 1.0 / d_model ** 0.5)

    def forward(self, x: torch.Tensor):
        """Returns (q, k, v) — or (q, k, v, gate_logits) when g_dim > 0."""
        out = F.linear(x, self.weight, self.bias)
        if self.g_dim:
            return out.split([self.q_dim, self.k_dim, self.v_dim,
                              self.g_dim], dim=-1)
        return out.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)

    @property
    def q_weight(self) -> torch.Tensor:
        return self.weight[:self.q_dim]

    @q_weight.setter
    def q_weight(self, w: torch.Tensor):
        with torch.no_grad():
            self.weight[:self.q_dim] = w

    @property
    def k_weight(self) -> torch.Tensor:
        return self.weight[self.q_dim:self.q_dim + self.k_dim]

    @k_weight.setter
    def k_weight(self, w: torch.Tensor):
        with torch.no_grad():
            self.weight[self.q_dim:self.q_dim + self.k_dim] = w

    @property
    def v_weight(self) -> torch.Tensor:
        return self.weight[self.q_dim + self.k_dim:]

    @v_weight.setter
    def v_weight(self, w: torch.Tensor):
        with torch.no_grad():
            self.weight[self.q_dim + self.k_dim:] = w


class FusedGateUpLinear(nn.Module):
    """Fused W_gate + W_up FFN projection in a single GEMM.

    Stores weights as a single (2 * hidden_dim, d_model) matrix.
    Splits the output into gate and up after the GEMM.
    """

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.out_dim = 2 * hidden_dim
        self.weight = nn.Parameter(torch.empty(self.out_dim, d_model))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_dim))
            nn.init.uniform_(self.bias, -1.0 / d_model ** 0.5, 1.0 / d_model ** 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gate, up) tensors, each (B, T, hidden_dim)."""
        out = F.linear(x, self.weight, self.bias)
        gate, up = out.split([self.hidden_dim, self.hidden_dim], dim=-1)
        return gate, up

    @property
    def gate_weight(self) -> torch.Tensor:
        return self.weight[:self.hidden_dim]

    @gate_weight.setter
    def gate_weight(self, w: torch.Tensor):
        with torch.no_grad():
            self.weight[:self.hidden_dim] = w

    @property
    def up_weight(self) -> torch.Tensor:
        return self.weight[self.hidden_dim:]

    @up_weight.setter
    def up_weight(self, w: torch.Tensor):
        with torch.no_grad():
            self.weight[self.hidden_dim:] = w


def fuse_qkv_weights(q_weight: torch.Tensor, k_weight: torch.Tensor,
                     v_weight: torch.Tensor,
                     q_bias: torch.Tensor | None = None,
                     k_bias: torch.Tensor | None = None,
                     v_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Concatenate separate Q/K/V weights into a single fused weight matrix."""
    fused_w = torch.cat([q_weight, k_weight, v_weight], dim=0)
    fused_b = None
    if q_bias is not None and k_bias is not None and v_bias is not None:
        fused_b = torch.cat([q_bias, k_bias, v_bias], dim=0)
    return fused_w, fused_b


def fuse_gateup_weights(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """Concatenate separate gate/up weights into a single fused weight matrix."""
    return torch.cat([gate_weight, up_weight], dim=0)

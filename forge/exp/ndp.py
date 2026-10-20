"""E-ndp: near-data processing for tiered KV attention.

Idea: when KV lives in a slower tier (host RAM / CXL pool), do NOT page
it to the GPU. Compute the attention partial there — score max, sum-exp,
and exp-weighted V accumulate (FlashAttention online-softmax state) —
and ship only O(H*d) floats across the interconnect.  The GPU merges
the partial with its local-window attention exactly:

    out = (acc_g*e^(m_g-m) + acc_h*e^(m_h-m)) / (s_g*e^(m_g-m) + s_h*e^(m_h-m))
    m   = max(m_g, m_h)

This is EXACT softmax attention — no approximation, no KV migration.
Per-decode-step transfer = 2*H + H*d floats vs T_far*d_kv*2*2 bytes
to migrate the pages.
"""
from __future__ import annotations

import math

import torch


def attn_partial(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """Online-softmax partial for one query position.

    q: (H, d); K,V: (H, T, d) — all on the SAME device (the data's tier).
    Returns (m, s, acc): (H,), (H,), (H, d) — the only tensors that ever
    cross the interconnect.
    """
    d = q.shape[-1]
    scores = torch.einsum("hd,htd->ht", q, K) / math.sqrt(d)
    m = scores.max(dim=-1).values                       # (H,)
    p = (scores - m.unsqueeze(-1)).exp()
    s = p.sum(dim=-1)                                   # (H,)
    acc = torch.einsum("ht,htd->hd", p, V)              # (H,d)
    return m, s, acc


def ndp_combine(part_a, part_b):
    """Merge two (m, s, acc) partials on part_a's device. Exact."""
    m_a, s_a, acc_a = part_a
    dev = m_a.device
    m_b, s_b, acc_b = (part_b[0].to(dev), part_b[1].to(dev),
                       part_b[2].to(dev))
    m = torch.maximum(m_a, m_b)
    ea = (m_a - m).exp()
    eb = (m_b - m).exp()
    s = s_a * ea + s_b * eb
    out = (acc_a * ea.unsqueeze(-1) + acc_b * eb.unsqueeze(-1)) \
        / s.clamp_min(1e-12).unsqueeze(-1)
    return out


def ndp_attention(q, local_kv, far_kv):
    """Full exact attention with far pages processed at their tier.

    local_kv: (K,V) on GPU; far_kv: (K,V) on host (never leaves).
    q: (H, d) on GPU (the decode query — always resident).
    """
    K_l, V_l = local_kv
    K_f, V_f = far_kv
    ql = q.to(K_l.device)
    part_local = attn_partial(ql, K_l, V_l)
    part_far = attn_partial(q.to(K_f.device), K_f, V_f)
    return ndp_combine(part_local, part_far)


def transfer_bytes(H: int, d: int, T_far: int, H_kv: int,
                   migrate: bool = False) -> int:
    """Bytes crossing the interconnect per decode step.
    migrate=False -> NDP partials (2*H + H*d floats32).
    migrate=True  -> the far KV pages themselves (bf16 K+V, H_kv heads)."""
    if migrate:
        return T_far * H_kv * d * 2 * 2
    return (2 * H + H * d) * 4

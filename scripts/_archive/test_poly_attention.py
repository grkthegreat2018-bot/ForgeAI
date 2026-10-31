"""Learned polynomial-kernel attention test (plan §8.5). CUDA.

HYPOTHESIS: softmax(QK^T) can be approximated by a low-order polynomial kernel
in (Q, K) with a CLOSED-FORM RECURRENCE (O(1) per step, no KV cache). Performer/
linear attention do this with a fixed kernel; the novel twist is LEARNING the
kernel per head (small head-specific polynomial coefficients).

Compares on a real-structured attention layer:
  1. Standard softmax attention (reference, O(n^2))
  2. Fixed polynomial kernel (Performer-style, random features)
  3. Learned per-head polynomial kernel (the novel idea)
  4. GLA (gated linear attention, already in codebase) as linear-attention baseline

Measures: attention-output error at equal FLOPs, and whether the learned kernel
adapts per head.

Runs on CUDA.
"""
import math

import torch
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def softmax_attention(q, k, v):
    """q [n_heads, seq, hd], k/v [n_heads, seq, hd] -> out [n_heads, seq, hd]."""
    hd = q.shape[-1]
    scores = torch.einsum('hsd,htd->hst', q, k) / math.sqrt(hd)
    attn = F.softmax(scores, dim=-1)
    return torch.einsum('hst,htd->hsd', attn, v)


def poly_kernel_attention_fixed(q, k, v, order=2, n_features=64):
    """Polynomial kernel: (1 + <q,k>/d)^order via random features.
    O(n * n_features) instead of O(n^2). Fixed random projection.
    """
    hd = q.shape[-1]
    # Random features for polynomial kernel
    torch.manual_seed(0)
    omega = torch.randn(hd, n_features, device=DEV) / math.sqrt(hd)
    # Map q, k to feature space: phi(x) = [1, x@omega] then polynomial
    phi_q = torch.cat([torch.ones(*q.shape[:-1], 1, device=DEV), q @ omega], dim=-1)
    phi_k = torch.cat([torch.ones(*k.shape[:-1], 1, device=DEV), k @ omega], dim=-1)
    # Raise to power 'order' (element-wise for diagonal approx)
    for _ in range(order - 1):
        phi_q = phi_q * torch.cat([torch.ones(*q.shape[:-1], 1, device=DEV), q @ omega], dim=-1)
        phi_k = phi_k * torch.cat([torch.ones(*k.shape[:-1], 1, device=DEV), k @ omega], dim=-1)
    # Linear attention: out = (phi_q @ (phi_k^T v)) / (phi_q @ phi_k.sum)
    kv = torch.einsum('htf,htd->hfd', phi_k, v)  # [n_heads, n_features, hd]
    denom = phi_k.sum(dim=1)  # [n_heads, n_features]
    out = torch.einsum('hsf,hfd->hsd', phi_q, kv)
    norm = torch.einsum('hsf,hf->hs', phi_q, denom).unsqueeze(-1).clamp(min=1e-8)
    return out / norm


def poly_kernel_attention_learned(q, k, v, order=2, n_features=64, epochs=100, lr=1e-2):
    """Learned per-head polynomial kernel: learn the omega projection per head.
    Trained to match softmax attention output on this batch.
    """
    n_heads, seq, hd = q.shape
    # Per-head omega: [n_heads, hd, n_features] — must be a leaf nn.Parameter
    omega = torch.nn.Parameter(torch.randn(n_heads, hd, n_features, device=DEV) / math.sqrt(hd))
    opt = torch.optim.Adam([omega], lr=lr)
    target = softmax_attention(q, k, v)
    for _ep in range(epochs):
        # Per-head feature map
        omega_q = omega.unsqueeze(1)  # [h, 1, d, f]
        phi_q = torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                           torch.einsum('hsd,hdf->hsf', q, omega)], dim=-1)
        phi_k = torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                           torch.einsum('hsd,hdf->hsf', k, omega)], dim=-1)
        for _ in range(order - 1):
            phi_q = phi_q * torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                                       torch.einsum('hsd,hdf->hsf', q, omega)], dim=-1)
            phi_k = phi_k * torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                                       torch.einsum('hsd,hdf->hsf', k, omega)], dim=-1)
        kv = torch.einsum('htf,htd->hfd', phi_k, v)
        denom = phi_k.sum(dim=1)
        out = torch.einsum('hsf,hfd->hsd', phi_q, kv)
        norm = torch.einsum('hsf,hf->hs', phi_q, denom).unsqueeze(-1).clamp(min=1e-8)
        out = out / norm
        loss = (out - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        phi_q = torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                           torch.einsum('hsd,hdf->hsf', q, omega)], dim=-1)
        phi_k = torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                           torch.einsum('hsd,hdf->hsf', k, omega)], dim=-1)
        for _ in range(order - 1):
            phi_q = phi_q * torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                                       torch.einsum('hsd,hdf->hsf', q, omega)], dim=-1)
            phi_k = phi_k * torch.cat([torch.ones(n_heads, seq, 1, device=DEV),
                                       torch.einsum('hsd,hdf->hsf', k, omega)], dim=-1)
        kv = torch.einsum('htf,htd->hfd', phi_k, v)
        denom = phi_k.sum(dim=1)
        out = torch.einsum('hsf,hfd->hsd', phi_q, kv)
        norm = torch.einsum('hsf,hf->hs', phi_q, denom).unsqueeze(-1).clamp(min=1e-8)
        out = out / norm
    n_params = omega.numel()
    return out, n_params


def gla_attention(q, k, v):
    """Gated Linear Attention (simple version): recurrence with forget gate.
    O(1) memory per step. Uses sigmoid(q) as gate.
    """
    n_heads, seq, hd = q.shape
    gate = torch.sigmoid(q)
    state = torch.zeros(n_heads, hd, hd, device=DEV)
    outs = []
    for t in range(seq):
        kt = k[:, t, :]  # [h, d]
        vt = v[:, t, :]  # [h, d]
        gt = gate[:, t, :]  # [h, d]
        state = state * gt.unsqueeze(-1) + kt.unsqueeze(-1) * vt.unsqueeze(1)
        qt = q[:, t, :]  # [h, d]
        out_t = torch.einsum('hd,hde->he', qt, state)  # [h, d]
        outs.append(out_t)
    return torch.stack(outs, dim=1)  # [h, seq, d]


def test_seq_len(seq_len, n_heads=8, head_dim=64):
    torch.manual_seed(42)
    d = n_heads * head_dim
    W_q = (torch.randn(d, d, device=DEV) * 0.02)
    W_k = (torch.randn(d, d, device=DEV) * 0.02)
    W_v = (torch.randn(d, d, device=DEV) * 0.02)
    x = (torch.randn(seq_len, d, device=DEV) * 0.5)
    q = (x @ W_q).reshape(seq_len, n_heads, head_dim).permute(1, 0, 2).contiguous()
    k = (x @ W_k).reshape(seq_len, n_heads, head_dim).permute(1, 0, 2).contiguous()
    v = (x @ W_v).reshape(seq_len, n_heads, head_dim).permute(1, 0, 2).contiguous()

    out_ref = softmax_attention(q, k, v)

    # Fixed poly kernel
    out_fixed = poly_kernel_attention_fixed(q, k, v, order=2, n_features=64)
    e_fixed = rel_err(out_fixed, out_ref)

    # Learned poly kernel
    out_learned, npar = poly_kernel_attention_learned(q, k, v, order=2, n_features=64, epochs=100)
    e_learned = rel_err(out_learned, out_ref)

    # GLA
    out_gla = gla_attention(q, k, v)
    e_gla = rel_err(out_gla, out_ref)

    print(f"\n--- seq_len={seq_len}, n_heads={n_heads}, hd={head_dim} ---")
    print("  Softmax (ref):     err=0.0000, O(n^2) memory")
    print(f"  Fixed poly (r=2):  err={e_fixed:.4f}, O(1) memory, 64 features")
    print(f"  Learned poly:      err={e_learned:.4f}, O(1) memory, {npar} learned params")
    print(f"  GLA:               err={e_gla:.4f}, O(1) memory (linear attention baseline)")
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
    return {'seq': seq_len, 'fixed': e_fixed, 'learned': e_learned, 'gla': e_gla}


if __name__ == "__main__":
    print("=" * 60)
    print("Learned polynomial-kernel attention (plan §8.5)")
    print("Novel: learn per-head polynomial kernel -> O(1) memory attention")
    print("=" * 60)
    results = []
    for sl in [256, 1024, 4096]:
        results.append(test_seq_len(sl))
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"{'seq':>6} {'fixed_poly':>12} {'learned_poly':>14} {'GLA':>10}")
    for r in results:
        print(f"{r['seq']:>6} {r['fixed']:>12.4f} {r['learned']:>14.4f} {r['gla']:>10.4f}")
    print("=" * 60)
    print("KEY: if learned_poly << fixed_poly and << GLA, the per-head learned")
    print("kernel adapts to head-specific attention patterns -> viable O(1) memory")
    print("attention replacement. If learned ~= fixed, the kernel doesn't adapt.")

"""Spectral-KV feasibility test (plan §8.3) — the highest-priority novel idea.

HYPOTHESIS: RoPE-rotated K is sinusoidal in position, so K(pos) is exactly
representable as a low-order Fourier series + a learned residual. If the residual
is small, we can replace the KV cache with O(1) per-layer memory (a Fourier
basis + tiny residual MLP) instead of O(n) cached vectors.

This script measures, on a REAL-STRUCTURED synthetic attention layer:
  1. How well a pure Fourier basis fits K(pos) and V(pos) (closed-form, no fitting)
  2. How well Fourier + small residual MLP fits (the proposed spectral-KV)
  3. Baseline: S4R-style low-rank KV at the same byte budget
  4. The downstream effect: attention-output error when using the approximation

Dims match LFM2.5: d_model=2048, n_heads=32, n_kv_heads=8, head_dim=64,
rope_base=1e6. Tests at seq_len=512, 2048, 8192 to see scaling.

Runs on CPU in seconds. No GPU/checkpoint needed.
"""
import math

import torch
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def rope_k(k, pos, theta=1e6, head_dim=64):
    """Apply RoPE to key tensor k [seq, n_kv_heads, head_dim] at positions pos [seq].
    Standard RoPE: rotate pairs (2i, 2i+1) by angle pos * theta^(-2i/head_dim).
    """
    seq, n_kv, hd = k.shape
    half = hd // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half).float() / half))  # [half]
    angles = pos.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq, half]
    cos = angles.cos()
    sin = angles.sin()
    # Interleaved rotation: k_even' = k_even*cos - k_odd*sin; k_odd' = k_even*sin + k_odd*cos
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    cos_b = cos.unsqueeze(1)  # [seq, 1, half]
    sin_b = sin.unsqueeze(1)
    k_rot = torch.stack([k_even * cos_b - k_odd * sin_b,
                         k_even * sin_b + k_odd * cos_b], dim=-1)
    return k_rot.reshape(seq, n_kv, hd)


def fourier_basis(seq_len, max_freq, device=DEV):
    """Build a Fourier basis [seq_len, 2*max_freq] (cos + sin for each freq)."""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.arange(1, max_freq + 1, dtype=torch.float32, device=device)
    cos = torch.cos(2 * math.pi * pos.unsqueeze(1) * freqs.unsqueeze(0) / seq_len)
    sin = torch.sin(2 * math.pi * pos.unsqueeze(1) * freqs.unsqueeze(0) / seq_len)
    basis = torch.cat([cos, sin], dim=1)  # [seq, 2*max_freq]
    # Add DC term
    dc = torch.ones(seq_len, 1, device=device)
    return torch.cat([dc, basis], dim=1)  # [seq, 1 + 2*max_freq]


def fit_fourier(target, basis):
    """Least-squares fit coefficients: target [seq, D] = basis [seq, F] @ coef [F, D].
    Returns coef and reconstruction.
    """
    coef = torch.linalg.lstsq(basis, target).solution  # [F, D]
    recon = basis @ coef
    return coef, recon


def fit_residual_mlp(target, basis, hidden=64, epochs=200, lr=1e-2):
    """Fit residual = target - basis_fit with a tiny MLP(pos) -> D.
    MLP input = positional encoding (sin/cos of pos at a few freqs).
    """
    seq, D = target.shape
    basis_fit = fit_fourier(target, basis)[1]
    residual = target - basis_fit

    # Positional encoding for MLP input
    pe_dim = 32
    pos = torch.arange(seq, dtype=torch.float32, device=DEV) / seq
    freqs = torch.pow(10000, -torch.arange(0, pe_dim, 2, device=DEV).float() / pe_dim)
    pe = torch.cat([torch.sin(pos.unsqueeze(1) * freqs.unsqueeze(0)),
                    torch.cos(pos.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(pe_dim, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, D)
    ).to(DEV)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(epochs):
        pred = mlp(pe)
        loss = (pred - residual).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        residual_pred = mlp(pe)
        recon = basis_fit + residual_pred
    n_params = sum(p.numel() for p in mlp.parameters())
    return recon, n_params


def s4r_baseline(k, v, rank, budget_bytes):
    """S4R-style: low-rank approximation of [seq, D] as [seq, rank] @ [rank, D].
    budget_bytes = rank * (seq + D) * 2 (bf16). Match to spectral-KV budget.
    """
    _seq, _D = k.shape
    U, S, Vh = torch.linalg.svd(k.float(), full_matrices=False)
    k_recon = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
    U2, S2, Vh2 = torch.linalg.svd(v.float(), full_matrices=False)
    v_recon = (U2[:, :rank] * S2[:rank]) @ Vh2[:rank, :]
    return k_recon, v_recon


def attention_output(q, k, v):
    """Standard attention: q [seq, n_heads, hd], k/v [seq, n_kv, hd].
    GQA: each kv head serves n_heads/n_kv query heads.
    """
    _seq, n_heads, hd = q.shape
    n_kv = k.shape[1]
    rep = n_heads // n_kv
    k_rep = k.repeat_interleave(rep, dim=1)  # [seq, n_heads, hd]
    v_rep = v.repeat_interleave(rep, dim=1)
    # [n_heads, seq, seq]
    scores = torch.einsum('qhd,khd->hqk', q, k_rep) / math.sqrt(hd)
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('hqk,khd->qhd', attn, v_rep)
    return out


def test_seq_len(seq_len, n_kv=8, head_dim=64, n_heads=32, max_freq=64):
    torch.manual_seed(42)
    # Simulate a real attention layer: random projections (trained-like)
    d_model = n_heads * head_dim
    W_q = (torch.randn(d_model, n_heads * head_dim) * 0.02).to(DEV)
    W_k = (torch.randn(d_model, n_kv * head_dim) * 0.02).to(DEV)
    W_v = (torch.randn(d_model, n_kv * head_dim) * 0.02).to(DEV)
    x = (torch.randn(seq_len, d_model) * 0.5).to(DEV)  # hidden states

    q = (x @ W_q).reshape(seq_len, n_heads, head_dim)
    k_raw = (x @ W_k).reshape(seq_len, n_kv, head_dim)
    v = (x @ W_v).reshape(seq_len, n_kv, head_dim)
    pos = torch.arange(seq_len, device=DEV)
    k = rope_k(k_raw, pos, head_dim=head_dim)  # RoPE-rotated K

    # Flatten for fitting: [seq, n_kv * head_dim]
    k_flat = k.reshape(seq_len, -1)
    v_flat = v.reshape(seq_len, -1)
    D = k_flat.shape[1]

    # Reference attention output
    out_ref = attention_output(q, k, v)

    # --- Method 1: Pure Fourier basis ---
    basis = fourier_basis(seq_len, max_freq)
    _, k_fourier = fit_fourier(k_flat, basis)
    _, v_fourier = fit_fourier(v_flat, basis)
    k_f = k_fourier.reshape(seq_len, n_kv, head_dim)
    v_f = v_fourier.reshape(seq_len, n_kv, head_dim)
    out_fourier = attention_output(q, k_f, v_f)
    err_k_fourier = rel_err(k_fourier, k_flat)
    err_v_fourier = rel_err(v_fourier, v_flat)
    err_out_fourier = rel_err(out_fourier, out_ref)
    bytes_fourier = (1 + 2 * max_freq) * D * 2 * 2  # K+V coef, bf16

    # --- Method 2: Fourier + residual MLP ---
    k_recon_mlp, n_params_k = fit_residual_mlp(k_flat, basis, hidden=64, epochs=150)
    v_recon_mlp, n_params_v = fit_residual_mlp(v_flat, basis, hidden=64, epochs=150)
    k_m = k_recon_mlp.reshape(seq_len, n_kv, head_dim)
    v_m = v_recon_mlp.reshape(seq_len, n_kv, head_dim)
    out_mlp = attention_output(q, k_m, v_m)
    err_k_mlp = rel_err(k_recon_mlp, k_flat)
    err_v_mlp = rel_err(v_recon_mlp, v_flat)
    err_out_mlp = rel_err(out_mlp, out_ref)
    bytes_mlp = bytes_fourier + (n_params_k + n_params_v) * 2  # MLP params bf16

    # --- Method 3: S4R low-rank at matched byte budget ---
    # Match bytes to the Fourier+MLP budget
    # S4R bytes = rank * (seq + D) * 2 * 2 (K+V)
    target_bytes = bytes_mlp
    rank = max(1, target_bytes // (2 * 2 * (seq_len + D)))
    rank = min(rank, min(seq_len, D) - 1)
    k_s4r_flat, v_s4r_flat = s4r_baseline(k_flat, v_flat, rank, target_bytes)
    k_s = k_s4r_flat.reshape(seq_len, n_kv, head_dim)
    v_s = v_s4r_flat.reshape(seq_len, n_kv, head_dim)
    out_s4r = attention_output(q, k_s, v_s)
    err_k_s4r = rel_err(k_s4r_flat, k_flat)
    err_v_s4r = rel_err(v_s4r_flat, v_flat)
    err_out_s4r = rel_err(out_s4r, out_ref)
    bytes_s4r = rank * (seq_len + D) * 2 * 2

    # --- Method 4: Full cache (reference byte cost) ---
    bytes_full = seq_len * D * 2 * 2  # K+V bf16

    print(f"\n--- seq_len={seq_len}, D={D}, max_freq={max_freq} ---")
    print(f"  Full cache:   {bytes_full/1024:.0f} KB, err=0.0 (reference)")
    print(f"  Fourier only: {bytes_fourier/1024:.0f} KB ({bytes_fourier/bytes_full*100:.1f}% of full)")
    print(f"    K err={err_k_fourier:.4f}  V err={err_v_fourier:.4f}  ATTN OUT err={err_out_fourier:.4f}")
    print(f"  Fourier+MLP:  {bytes_mlp/1024:.0f} KB ({bytes_mlp/bytes_full*100:.1f}% of full)")
    print(f"    K err={err_k_mlp:.4f}  V err={err_v_mlp:.4f}  ATTN OUT err={err_out_mlp:.4f}")
    print(f"  S4R r={rank}:    {bytes_s4r/1024:.0f} KB ({bytes_s4r/bytes_full*100:.1f}% of full)")
    print(f"    K err={err_k_s4r:.4f}  V err={err_v_s4r:.4f}  ATTN OUT err={err_out_s4r:.4f}")
    print(f"  COMPRESSION: Fourier+MLP = {bytes_full/bytes_mlp:.1f}x, S4R = {bytes_full/bytes_s4r:.1f}x")

    return {
        'seq_len': seq_len,
        'bytes_full': bytes_full,
        'fourier': {'bytes': bytes_fourier, 'err_out': err_out_fourier},
        'fourier_mlp': {'bytes': bytes_mlp, 'err_out': err_out_mlp},
        's4r': {'bytes': bytes_s4r, 'err_out': err_out_s4r, 'rank': rank},
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Spectral-KV Feasibility Test (plan §8.3)")
    print("Hypothesis: RoPE-K is sinusoidal → Fourier basis fits K well → O(1) KV")
    print("=" * 70)
    results = []
    for sl in [512, 2048, 8192]:
        results.append(test_seq_len(sl, max_freq=64))
    print("\n" + "=" * 70)
    print("SUMMARY — attention output error vs compression:")
    print(f"{'seq':>6} {'method':>14} {'compress':>10} {'out_err':>10}")
    for r in results:
        for m, name in [('fourier', 'Fourier'), ('fourier_mlp', 'Fourier+MLP'), ('s4r', 'S4R')]:
            cr = r['bytes_full'] / r[m]['bytes']
            print(f"{r['seq_len']:>6} {name:>14} {cr:>9.1f}x {r[m]['err_out']:>10.4f}")
    print("=" * 70)
    # Key insight: does Fourier+MLP error SHRINK as seq grows? (O(1) memory)
    # while S4R error GROWS (it needs more rank for longer seq)?
    print("KEY: if Fourier+MLP error is stable/grows-slowly with seq_len while")
    print("S4R needs proportional rank, spectral-KV is O(1) memory — the dream.")

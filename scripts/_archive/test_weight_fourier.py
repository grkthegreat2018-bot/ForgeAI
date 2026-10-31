"""Fourier weight decomposition — NOVEL idea (post §14.2 finding). CUDA.

CONTEXT: §14.2 showed LFM2.5 weights are NEAR FULL-RANK (99% energy at rank
1962/2048). SVD/NLRQ cannot compress them. INR (§12.2) and VQ (§12.3) also lost
to SVD. So no known method compresses these weights well.

NOVEL HYPOTHESIS: trained weights have SPATIAL structure — adjacent neurons
(rows/columns) are correlated. A Fourier/wavelet basis over the (row, col) index
space can capture this smooth structure, while a sparse residual captures the
uncorrelated part. This is "low-frequency + sparse" decomposition, fundamentally
different from "low-rank" (SVD).

W ≈ W_lowfreq (Fourier basis, few coefficients) + W_sparse (top-k errors)

The key insight: SVD finds the best LOW-RANK approximation (global structure).
Fourier finds the best LOW-FREQUENCY approximation (local smoothness). These
are DIFFERENT structure types. A weight can be full-rank (SVD can't compress)
but low-frequency (Fourier can compress).

Test on REAL LFM2.5 weights:
  1. 2D Fourier basis over (row, col) index space
  2. Keep lowest-frequency coefficients (analogous to JPEG compression)
  3. + sparse residual for high-frequency detail
  4. Compare to SVD at matched param budget

Also tests: DCT (discrete cosine transform, JPEG-style) vs DFT.

Runs on CUDA.
"""
import math

import torch
from safetensors import safe_open

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")

CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def dct_2d(W, keep_frac):
    """2D DCT compression: keep lowest-frequency coefficients (JPEG-style).
    W [m, n] -> DCT -> keep top keep_frac of coefficients -> inverse DCT.
    """
    m, n = W.shape
    Wf = W.float()

    # 1D DCT-II basis
    def dct_matrix(N):
        k = torch.arange(N, device=DEV).float()
        n_idx = torch.arange(N, device=DEV).float()
        D = torch.cos(math.pi * (2 * n_idx.unsqueeze(0) + 1) * k.unsqueeze(1) / (2 * N))
        D[0, :] *= 1.0 / math.sqrt(N)
        D[1:, :] *= math.sqrt(2.0 / N)
        return D

    Dm = dct_matrix(m)  # [m, m]
    Dn = dct_matrix(n)  # [n, n]

    # Forward DCT: coeffs = Dm @ W @ Dn.T
    coeffs = Dm @ Wf @ Dn.T  # [m, n]

    # Keep lowest-frequency: zero out high-frequency coefficients
    # Frequency = row_idx + col_idx (Manhattan distance from DC)
    row_idx = torch.arange(m, device=DEV).unsqueeze(1).expand(m, n).float()
    col_idx = torch.arange(n, device=DEV).unsqueeze(0).expand(m, n).float()
    freq = row_idx + col_idx
    threshold = torch.quantile(freq.flatten(), keep_frac)
    mask = freq <= threshold
    coeffs_compressed = coeffs * mask

    # Inverse DCT: W_recon = Dm.T @ coeffs @ Dn
    W_recon = Dm.T @ coeffs_compressed @ Dn

    # Count nonzero coefficients (params)
    n_coeffs = mask.sum().item()
    return W_recon, n_coeffs


def dct_2d_plus_sparse(W, keep_frac, sparse_frac):
    """DCT low-freq + sparse residual for high-freq detail."""
    W_recon, n_coeffs = dct_2d(W, keep_frac)
    err = (W.float() - W_recon).abs()
    k = int(W.numel() * sparse_frac)
    flat_err = err.flatten()
    top_idx = flat_err.topk(k).indices
    W_final = W_recon.clone()
    W_final.flatten()[top_idx] = W.float().flatten()[top_idx]
    total_params = n_coeffs + k  # coefficients + sparse indices+values
    return W_final, total_params


def svd_compress(W, rank):
    m, n = W.shape
    U, S, Vh = torch.linalg.svd(W.float().to(DEV), full_matrices=False)
    r = min(rank, min(m, n))
    return (U[:, :r] * S[:r]) @ Vh[:r, :], r * (m + n)


def test_real_weight(label, W_key):
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).float()

    m, n = W.shape
    dense_params = m * n
    print(f"\n{'='*60}")
    print(f"  {label}: {W.shape} ({dense_params} params)")

    x = torch.randn(1, 64, m, device=DEV) * 0.5
    y_ref = x @ W

    print(f"  {'budget':>10} {'method':>22} {'params':>10} {'compress':>10} {'out_err':>10}")

    for target_frac in [0.05, 0.10, 0.20, 0.50]:
        target_params = int(dense_params * target_frac)

        # SVD
        rank = target_params // (m + n)
        if rank < 1:
            continue
        r_svd, a_svd = svd_compress(W, rank)
        e_svd = rel_err(x @ r_svd, y_ref)
        print(f"  {target_params:>10} {'SVD(r=%d)' % rank:>22} {a_svd:>10} "
              f"{dense_params/a_svd:>9.1f}x {e_svd:>10.4f}")

        # DCT (keep fraction of coefficients)
        # DCT params = n_coeffs. We want n_coeffs ≈ target_params
        # n_coeffs = keep_frac * m * n, so keep_frac = target / (m*n)
        keep_frac = target_params / dense_params
        if keep_frac > 0.95:
            continue
        r_dct, n_dct = dct_2d(W, keep_frac)
        e_dct = rel_err(x @ r_dct, y_ref)
        print(f"  {target_params:>10} {'DCT(keep={:.2f})'.format(keep_frac):>22} {n_dct:>10} "
              f"{dense_params/max(n_dct,1):>9.1f}x {e_dct:>10.4f}")

        # DCT + sparse (70% DCT, 30% sparse budget)
        dct_budget = int(target_params * 0.7)
        sparse_budget = target_params - dct_budget
        keep_frac2 = dct_budget / dense_params
        sparse_frac = sparse_budget / dense_params
        if keep_frac2 > 0.95 or sparse_frac > 0.1:
            pass
        else:
            r_ds, n_ds = dct_2d_plus_sparse(W, keep_frac2, sparse_frac)
            e_ds = rel_err(x @ r_ds, y_ref)
            print(f"  {target_params:>10} {'DCT+sparse({:.2f}+{:.3f})'.format(keep_frac2, sparse_frac):>22} "
                  f"{n_ds:>10} {dense_params/max(n_ds,1):>9.1f}x {e_ds:>10.4f}")

    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 70)
    print("Fourier/DCT weight decomposition — NOVEL (post §14.2)")
    print("Hypothesis: full-rank weights may be low-frequency (spatially smooth)")
    print("=" * 70)
    # Use smaller weights for speed (DCT on 8192x2048 is heavy)
    test_real_weight("Attn Q_proj [2048x2048]", "blocks.2.attn.q_proj.weight")
    test_real_weight("Attn K_proj [512x2048]", "blocks.2.attn.k_proj.weight")
    test_real_weight("Attn V_proj [512x2048]", "blocks.2.attn.v_proj.weight")
    print(f"\n{'='*70}")
    print("KEY: if DCT err < SVD err at same param budget, weights are")
    print("low-frequency (spatially smooth) even though they're full-rank.")
    print("This would be a NOVEL weight compression method that beats SVD")
    print("on near-full-rank weights — exactly the case §14.2 showed is hard.")

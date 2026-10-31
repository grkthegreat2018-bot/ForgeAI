"""INR-weights test (plan §8.1) — weights as implicit neural representations. CUDA.

HYPOTHESIS: a trained weight matrix W in R^{m x n} is a smooth function of
(row, col) indices. A tiny MLP f(i,j) -> w_ij can represent it with far fewer
params than m*n floats, giving high disk compression.

Compares at EQUAL PARAM BUDGET:
  1. SVD low-rank (the NLRQ baseline)
  2. INR: coordinate MLP f(row_norm, col_norm) -> w
  3. INR + sparse residual (top-k errors kept dense)

Uses a synthetic weight that mimics trained structure: low-rank + smooth + sparse
outliers (real trained weights have this profile).

Runs on CUDA when available (much faster MLP fitting).
"""
import math

import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def make_trained_like_weight(m, n, rank, outlier_frac=0.02, seed=0):
    torch.manual_seed(seed)
    U = torch.randn(m, rank) * 0.1
    V = torch.randn(rank, n) * 0.1
    W = U @ V
    rows = torch.arange(m).float().unsqueeze(1) / m
    cols = torch.arange(n).float().unsqueeze(0) / n
    smooth = torch.sin(rows * 3.14) @ torch.cos(cols * 6.28) * 0.05
    W = W + smooth
    n_out = int(m * outlier_frac)
    out_rows = torch.randperm(m)[:n_out]
    W[out_rows] += torch.randn(n_out, n) * 0.3
    return W.to(DEV)


def svd_compress(W, target_params):
    m, n = W.shape
    r = target_params // (m + n)
    r = min(r, min(m, n))
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    recon = (U[:, :r] * S[:r]) @ Vh[:r, :]
    actual = r * (m + n)
    return recon, actual, r


def inr_compress(W, target_params, epochs=500, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    m, n = W.shape
    pe_dim = 16
    inp = pe_dim
    h = int((- (inp + 1) + math.sqrt((inp + 1) ** 2 + 4 * target_params)) / 2)
    h = max(h, 8)

    rows = torch.arange(m, device=DEV).float() / m - 0.5
    cols = torch.arange(n, device=DEV).float() / n - 0.5
    freqs = torch.pow(10000, -torch.arange(0, pe_dim, 2, device=DEV).float() / pe_dim)
    pe_r = torch.cat([torch.sin(rows.unsqueeze(1) * freqs.unsqueeze(0)),
                      torch.cos(rows.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)
    pe_c = torch.cat([torch.sin(cols.unsqueeze(1) * freqs.unsqueeze(0)),
                      torch.cos(cols.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)
    coords = torch.cat([pe_r.unsqueeze(1).expand(m, n, inp),
                        pe_c.unsqueeze(0).expand(m, n, inp)], dim=2).reshape(m * n, inp * 2)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(inp * 2, h), torch.nn.GELU(),
        torch.nn.Linear(h, h), torch.nn.GELU(),
        torch.nn.Linear(h, 1)
    ).to(DEV)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    target = W.float().reshape(-1, 1)
    for _ep in range(epochs):
        pred = mlp(coords)
        loss = (pred - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        recon = mlp(coords).reshape(m, n)
    actual = sum(p.numel() for p in mlp.parameters())
    return recon, actual, h


def inr_plus_residual(W, target_params, residual_frac=0.05, epochs=500, lr=1e-2, seed=0):
    inr_budget = int(target_params * 0.7)
    recon_inr, n_inr, h = inr_compress(W, inr_budget, epochs=epochs, lr=lr, seed=seed)
    err = (W.float() - recon_inr).abs()
    m, n = W.shape
    k = int(m * n * residual_frac)
    flat_err = err.flatten()
    topk_idx = flat_err.topk(k).indices
    topk_vals = W.float().flatten()[topk_idx]
    recon = recon_inr.clone()
    recon.flatten()[topk_idx] = topk_vals
    actual = n_inr + k * 5
    return recon, actual, h, k


def test_weight(m, n, rank, label, budgets):
    W = make_trained_like_weight(m, n, rank)
    print(f"\n{'='*60}")
    print(f"Weight: {label} [{m}x{n}], true rank={rank}, {m*n} params dense")
    print(f"{'='*60}")
    print(f"{'budget':>10} {'method':>16} {'actual':>10} {'compress':>10} {'rel_err':>10}")
    for budget in budgets:
        r_svd, a_svd, _rk = svd_compress(W, budget)
        e_svd = rel_err(r_svd, W)
        print(f"{budget:>10} {'SVD':>16} {a_svd:>10} {m*n/a_svd:>9.1f}x {e_svd:>10.4f}")
        r_inr, a_inr, h = inr_compress(W, budget, epochs=300)
        e_inr = rel_err(r_inr, W)
        print(f"{budget:>10} {'INR(h=%d)' % h:>16} {a_inr:>10} {m*n/a_inr:>9.1f}x {e_inr:>10.4f}")
        r_ir, a_ir, _h2, k = inr_plus_residual(W, budget, residual_frac=0.03, epochs=300)
        e_ir = rel_err(r_ir, W)
        print(f"{budget:>10} {'INR+res(k=%d)' % k:>16} {a_ir:>10} {m*n/a_ir:>9.1f}x {e_ir:>10.4f}")
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("INR-weights vs SVD (plan §8.1)")
    print("Weight = coordinate MLP f(row, col) -> w_ij")
    print("=" * 60)
    # Smaller weights + fewer epochs for fast CUDA turnaround
    test_weight(128, 512, rank=32, label="FFN-like (trained-like)",
                budgets=[8192, 16384, 32768])
    test_weight(512, 128, rank=64, label="Embedding-like",
                budgets=[8192, 16384, 32768])
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- If INR err < SVD err at same param budget -> INR wins (smooth weights)")
    print("- If INR+residual < SVD -> the hybrid captures outliers SVD misses")
    print("- INR is a DISK format: materialize W at load, then normal matmul")

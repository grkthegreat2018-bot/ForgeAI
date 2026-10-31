"""VQ-weights test (plan §8.2) — vector-quantized weight rows. CUDA.

HYPOTHESIS: across a weight matrix's output neurons (rows), many are similar.
k-means clustering into K codewords + int assignments gives high compression.

Compares at EQUAL BYTE BUDGET:
  1. NLRQ (SVD + INT8 factor quant) — the current V8 FFN compression
  2. VQ: k-means on rows, store codebook [K, n] + assignments [m] int16
  3. VQ + residual: keep top-k error rows dense

Uses trained-like synthetic weights (low-rank + outlier channels).

Runs on CUDA when available.
"""
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


def nlrq_compress(W, rank, factor_bits=8):
    m, n = W.shape
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    Ur, Sr, Vhr = U[:, :rank], S[:rank], Vh[:rank, :]
    levels = 2 ** factor_bits - 1
    def quant(t):
        scale = t.abs().max(dim=-1, keepdim=True).values / (levels / 2)
        scale = scale.clamp(min=1e-8)
        return (t / scale).round().clamp(-levels/2, levels/2) * scale
    Uq = quant(Ur * Sr.sqrt().unsqueeze(0))
    Vq = quant(Sr.sqrt().unsqueeze(1) * Vhr)
    recon = Uq @ Vq
    bytes_est = rank * (m + n) * (factor_bits / 8) + rank * 2 + (m + n) * 2
    return recon, bytes_est


def vq_compress(W, K, residual_frac=0.0):
    m, n = W.shape
    Wf = W.float()
    torch.manual_seed(0)
    idx = torch.randperm(m, device=DEV)[:K]
    codebook = Wf[idx].clone()
    for _it in range(25):
        dists = torch.cdist(Wf, codebook)
        assign = dists.argmin(dim=1)
        for k in range(K):
            mask = assign == k
            if mask.any():
                codebook[k] = Wf[mask].mean(dim=0)
    recon = codebook[assign]
    bytes_vq = K * n * 2 + m * 2
    if residual_frac > 0:
        err = (Wf - recon).abs().sum(dim=1)
        k_res = int(m * residual_frac)
        top_rows = err.topk(k_res).indices
        recon[top_rows] = Wf[top_rows]
        bytes_vq += k_res * n * 2
    return recon, bytes_vq, K


def test_weight(m, n, rank, label, budgets_bytes):
    W = make_trained_like_weight(m, n, rank)
    dense_bytes = m * n * 2
    print(f"\n{'='*60}")
    print(f"Weight: {label} [{m}x{n}], true rank={rank}, dense={dense_bytes/1024:.0f} KB")
    print(f"{'='*60}")
    print(f"{'budget_KB':>10} {'method':>18} {'actual_KB':>10} {'compress':>10} {'rel_err':>10}")
    for budget in budgets_bytes:
        max_r = (budget - (m + n) * 2) // (m + n + 2)
        max_r = min(max_r, min(m, n))
        if max_r < 1: continue
        r_n, b_n = nlrq_compress(W, max_r, factor_bits=8)
        e_n = rel_err(r_n, W)
        print(f"{budget/1024:>10.0f} {'NLRQ(r=%d)' % max_r:>18} {b_n/1024:>10.0f} {dense_bytes/b_n:>9.1f}x {e_n:>10.4f}")
        max_K = (budget - m * 2) // (n * 2)
        max_K = min(max_K, m)
        if max_K < 4: continue
        r_v, b_v, K = vq_compress(W, max_K, residual_frac=0.0)
        e_v = rel_err(r_v, W)
        print(f"{budget/1024:>10.0f} {'VQ(K=%d)' % K:>18} {b_v/1024:>10.0f} {dense_bytes/b_v:>9.1f}x {e_v:>10.4f}")
        max_K2 = int((budget - m * 2) // (n * 2 * 1.03))
        max_K2 = min(max_K2, m)
        if max_K2 < 4: continue
        r_vr, b_vr, K2 = vq_compress(W, max_K2, residual_frac=0.03)
        e_vr = rel_err(r_vr, W)
        print(f"{budget/1024:>10.0f} {'VQ+res(K=%d)' % K2:>18} {b_vr/1024:>10.0f} {dense_bytes/b_vr:>9.1f}x {e_vr:>10.4f}")
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("VQ-weights vs NLRQ (plan §8.2)")
    print("Weight rows = quantized to K codewords + assignments")
    print("=" * 60)
    test_weight(2048, 512, rank=64, label="FFN-like",
                budgets_bytes=[51200, 102400, 204800, 409600])
    test_weight(4096, 512, rank=128, label="Embedding-like",
                budgets_bytes=[51200, 102400, 204800])
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- VQ wins if weight rows cluster (true for FFN gate/up which share structure)")
    print("- VQ+residual handles outlier rows that don't cluster")
    print("- VQ matmul = gather codewords then matmul (materialize at load, like INR)")

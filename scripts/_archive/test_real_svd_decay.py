"""Singular value decay of REAL LFM2.5 weights (plan §13.5). CUDA.

Answers: "what is the true effective rank of trained weights?"
- SVD each weight type (FFN gate/up/down, attention Q/K/V, embedding)
- Plot singular value decay + cumulative energy
- Find effective rank (where cumulative energy > 99.9%, 99%, 95%)
- This determines whether NLRQ rank-1024 is sufficient for V8's 16384×4096 FFN

Also compares: Monarch vs TT vs SVD at matched param budget (§13.4).

Runs on CUDA.
"""

import torch
from safetensors import safe_open

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")

CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def svd_analysis(W, label):
    """SVD a weight, report singular value decay + effective ranks."""
    m, n = W.shape
    Wf = W.float().to(DEV)
    _U, S, _Vh = torch.linalg.svd(Wf, full_matrices=False)
    total_energy = S.pow(2).sum().item()
    cum_energy = S.pow(2).cumsum(0) / total_energy

    # Effective ranks at different energy thresholds
    ranks = {}
    for thresh in [0.9999, 0.999, 0.99, 0.95, 0.90, 0.50]:
        idx = (cum_energy >= thresh).nonzero()
        r = idx[0].item() + 1 if len(idx) > 0 else len(S)
        ranks[thresh] = r

    # Top-10 singular values
    top10 = S[:10].tolist()

    # Decay rate: fit S[i] ~ C * i^(-alpha)
    log_idx = torch.arange(1, min(50, len(S)) + 1, device=DEV).float().log()
    log_S = S[:50].clamp(min=1e-10).log()
    alpha = -(log_S * log_idx).mean().item() - (log_S.mean() * log_idx.mean()).item()
    # Simple: alpha = -slope of log(S) vs log(i)
    alpha = -((log_idx - log_idx.mean()) * (log_S - log_S.mean())).sum() / \
             ((log_idx - log_idx.mean()).pow(2).sum() + 1e-12)

    print(f"\n{'='*60}")
    print(f"  {label}: {W.shape} ({m*n} params)")
    print(f"  Top-10 SVs: {['{:.3f}'.format(s) for s in top10]}")
    print(f"  SV decay exponent (alpha, first 50): {alpha:.2f}")
    print("  Effective rank (energy captured):")
    for t, r in sorted(ranks.items(), reverse=True):
        cr = (m * n) / (r * (m + n)) if r > 0 else 0
        print(f"    {t*100:.2f}%: rank={r} ({cr:.1f}x SVD compression)")
    return ranks, alpha.item() if hasattr(alpha, 'item') else alpha


def svd_compress(W, rank):
    _m, _n = W.shape
    U, S, Vh = torch.linalg.svd(W.float().to(DEV), full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank, :]


def monarch_compress(W, block_r, block_size):
    """Monarch matrix: block-diagonal L @ block-diagonal R.
    Approximates W [m,n] as M_L [m, block_r] @ M_R [block_r, n] where
    M_L and M_R are block-diagonal. Simplified: just block-diagonal SVD.
    """
    m, n = W.shape
    Wf = W.float().to(DEV)
    # Reshape into blocks and SVD each
    n_blocks_m = m // block_size
    n_blocks_n = n // block_size
    recon = torch.zeros_like(Wf)
    total_params = 0
    for i in range(n_blocks_m):
        for j in range(n_blocks_n):
            block = Wf[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if block.shape[0] == 0 or block.shape[1] == 0:
                continue
            U, S, Vh = torch.linalg.svd(block, full_matrices=False)
            r = min(block_r, len(S))
            recon[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                (U[:, :r] * S[:r]) @ Vh[:r, :]
            total_params += r * (block_size + block_size)
    return recon, total_params


def test_real_weights():
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        weights = {
            "FFN w_gate (layer 2)": f.get_tensor("blocks.2.ffn.w_gate.weight"),
            "FFN w_up (layer 2)": f.get_tensor("blocks.2.ffn.w_up.weight"),
            "FFN w_down (layer 2)": f.get_tensor("blocks.2.ffn.w_down.weight"),
            "Attn Q_proj (layer 2)": f.get_tensor("blocks.2.attn.q_proj.weight"),
            "Attn K_proj (layer 2)": f.get_tensor("blocks.2.attn.k_proj.weight"),
            "Attn V_proj (layer 2)": f.get_tensor("blocks.2.attn.v_proj.weight"),
            "Embedding": f.get_tensor("embed.weight"),
        }

    all_ranks = {}
    for label, W in weights.items():
        ranks, _alpha = svd_analysis(W, label)
        all_ranks[label] = ranks

    # --- Monarch vs SVD comparison on FFN w_gate ---
    print(f"\n{'='*60}")
    print("Monarch vs SVD on FFN w_gate [8192x2048]:")
    print(f"{'='*60}")
    W_gate = weights["FFN w_gate (layer 2)"].float().to(DEV)
    m, n = W_gate.shape
    for target_params in [50000, 200000, 500000]:
        # SVD
        r_svd = target_params // (m + n)
        r_svd = min(r_svd, min(m, n))
        recon_svd = svd_compress(W_gate, r_svd)
        e_svd = rel_err(recon_svd, W_gate)
        actual_svd = r_svd * (m + n)

        # Monarch (block_size=256, rank per block)
        block_size = 256
        r_per_block = max(1, target_params // ((m // block_size) * (n // block_size) * 2 * block_size))
        recon_mon, actual_mon = monarch_compress(W_gate, r_per_block, block_size)
        e_mon = rel_err(recon_mon, W_gate)

        print(f"  budget={target_params}: SVD(r={r_svd}) err={e_svd:.4f} ({actual_svd} params) | "
              f"Monarch(r={r_per_block},bs={block_size}) err={e_mon:.4f} ({actual_mon} params)")

    # --- Summary: is NLRQ rank-1024 enough for V8? ---
    print(f"\n{'='*60}")
    print("NLRQ RANK SUFFICIENCY ANALYSIS:")
    print(f"{'='*60}")
    ffn_ranks = all_ranks["FFN w_gate (layer 2)"]
    print("  LFM2.5 FFN w_gate [8192x2048] effective ranks:")
    for t, r in sorted(ffn_ranks.items(), reverse=True):
        print(f"    {t*100:.2f}% energy: rank={r}")
    print("  V8 FFN is [16384x4096] (2x both dims).")
    print(f"  V8 NLRQ rank=1024. If LFM2.5 99% rank ~{ffn_ranks[0.99]},")
    print(f"  V8 99% rank ~{ffn_ranks[0.99]*4} (scales with min(m,n)).")
    print("  rank=1024 captures: need to check on V8-sized weights.")
    # Extrapolate: V8 FFN is 2x wider and 2x taller, so rank scales ~2x
    v8_99_rank = ffn_ranks[0.99] * 2  # rough extrapolation
    print(f"  Estimated V8 99% rank: ~{v8_99_rank}")
    if v8_99_rank > 1024:
        print("  WARNING: rank=1024 may be INSUFFICIENT for 99% energy!")
        print(f"  Consider rank={int(v8_99_rank)} for V8-8B-B.")
    else:
        print("  rank=1024 is SUFFICIENT for 99% energy. ✓")

    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 70)
    print("SVD decay of REAL LFM2.5 weights (plan §13.5)")
    print("Question: what is the true effective rank of trained weights?")
    print("=" * 70)
    test_real_weights()

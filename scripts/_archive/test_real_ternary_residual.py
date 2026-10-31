"""BitNet+residual on REAL LFM2.5 weights (plan §13.3). CUDA.

Tests the ternary+residual idea (§12.4 winner) on REAL trained weights to find:
  1. The actual outlier distribution (are there few outliers or many?)
  2. The optimal k (residual fraction) for real weights
  3. Whether real weights behave differently from synthetic (§12.4 used synthetic)

Loads real FFN and attention weights from LFM2.5, applies ternary quantization,
measures error with and without residual at various k values.

Also tests: per-ROW residual (keep outlier rows dense) vs per-ELEMENT residual
(keep outlier individual elements dense). These are different sparsity patterns.

Runs on CUDA.
"""
import torch
from safetensors import safe_open

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")

CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def bitnet_ternary(W):
    scale = W.abs().mean().item()
    W_t = torch.round(W / (scale + 1e-8)).clamp(-1, 1)
    return W_t * scale, scale


def ternary_row_residual(W, k_frac):
    """Keep top-k% outlier ROWS dense, rest ternary."""
    W_t, _scale = bitnet_ternary(W)
    err = (W - W_t).abs().sum(dim=1)  # per-row error
    k = int(W.shape[0] * k_frac)
    if k == 0:
        return W_t, 0
    top_rows = err.topk(k).indices
    W_t[top_rows] = W[top_rows]
    return W_t, k


def ternary_elem_residual(W, k_frac):
    """Keep top-k% outlier INDIVIDUAL ELEMENTS dense, rest ternary."""
    W_t, _scale = bitnet_ternary(W)
    err = (W - W_t).abs()
    k = int(W.numel() * k_frac)
    if k == 0:
        return W_t, 0
    flat_err = err.flatten()
    top_idx = flat_err.topk(k).indices
    W_t.flatten()[top_idx] = W.flatten()[top_idx]
    return W_t, k


def ternary_col_residual(W, k_frac):
    """Keep top-k% outlier COLUMNS dense, rest ternary."""
    W_t, _scale = bitnet_ternary(W)
    err = (W - W_t).abs().sum(dim=0)  # per-column error
    k = int(W.shape[1] * k_frac)
    if k == 0:
        return W_t, 0
    top_cols = err.topk(k).indices
    W_t[:, top_cols] = W[:, top_cols]
    return W_t, k


def test_real_weight(label, W_key, x_dim=None):
    """Test ternary+residual on a real weight."""
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).float()

    m, n = W.shape
    print(f"\n{'='*60}")
    print(f"  {label}: {W.shape} ({m*n} params, {m*n*2/1024:.0f} KB)")
    print(f"  W stats: mean={W.mean():.4f}, std={W.std():.4f}, "
          f"max={W.abs().max():.4f}")

    # Outlier analysis
    W_t, _scale = bitnet_ternary(W)
    err = (W - W_t).abs()
    row_errs = err.sum(dim=1)
    col_errs = err.sum(dim=0)
    # What fraction of error is in top-1% of rows?
    top1pct_rows = row_errs.topk(max(1, m // 100)).indices
    err_concentration_rows = row_errs[top1pct_rows].sum().item() / row_errs.sum().item()
    top1pct_cols = col_errs.topk(max(1, n // 100)).indices
    err_concentration_cols = col_errs[top1pct_cols].sum().item() / col_errs.sum().item()
    print(f"  Outlier concentration: top-1% rows = {err_concentration_rows*100:.1f}% of error, "
          f"top-1% cols = {err_concentration_cols*100:.1f}%")

    # Test input
    if x_dim is None:
        x_dim = m
    x = torch.randn(1, 64, x_dim, device=DEV) * 0.5
    # Linear layer: y = x @ W.T (W is [out, in], x is [., in], y is [., out])
    y_ref = x @ W.T

    # Pure ternary
    y_t = x @ W_t.T
    e_t = rel_err(y_t, y_ref)
    print(f"  Pure ternary: err={e_t:.4f}")

    # Sweep k for each residual type
    print(f"  {'k%':>6} {'row_res':>12} {'col_res':>12} {'elem_res':>12}")
    for k_frac in [0.005, 0.01, 0.02, 0.05, 0.10]:
        # Row residual
        W_rr, _k_r = ternary_row_residual(W, k_frac)
        e_rr = rel_err(x @ W_rr.T, y_ref)
        # Col residual
        W_cr, _k_c = ternary_col_residual(W, k_frac)
        e_cr = rel_err(x @ W_cr.T, y_ref)
        # Element residual
        W_er, _k_e = ternary_elem_residual(W, k_frac)
        e_er = rel_err(x @ W_er.T, y_ref)
        print(f"  {k_frac*100:>5.1f}% {e_rr:>12.4f} {e_cr:>12.4f} {e_er:>12.4f}")

    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 70)
    print("BitNet+residual on REAL LFM2.5 weights (plan §13.3)")
    print("=" * 70)
    test_real_weight("FFN w_gate", "blocks.2.ffn.w_gate.weight", x_dim=2048)
    test_real_weight("FFN w_up", "blocks.2.ffn.w_up.weight", x_dim=2048)
    test_real_weight("FFN w_down", "blocks.2.ffn.w_down.weight", x_dim=8192)
    test_real_weight("Attn Q_proj", "blocks.2.attn.q_proj.weight", x_dim=2048)
    test_real_weight("Attn K_proj", "blocks.2.attn.k_proj.weight", x_dim=2048)
    test_real_weight("Embedding (first 2048 rows)", "embed.weight", x_dim=2048)
    print(f"\n{'='*70}")
    print("KEY: find the k% where error drops to <0.05 (acceptable).")
    print("Compare row vs col vs elem residual — which captures outliers best?")
    print("Real weights may have different outlier structure than synthetic.")

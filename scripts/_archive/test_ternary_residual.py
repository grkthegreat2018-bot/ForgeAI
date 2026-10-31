"""BitNet ternary + residual k-sweep (plan §5.2). CUDA.

HYPOTHESIS: BitNet b1.58 ternary {-1,0,+1} destroys outlier channels (0.79 error
on random weights). Keeping the top-k% highest-error channels in bf16 (dense)
recovers most quality at small memory cost.

Sweeps k = 0%, 1%, 2%, 5%, 10% on trained-like weights.
Compares: pure ternary vs ternary+residual vs INT8 vs bf16 baseline.

Runs on CUDA.
"""
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def make_trained_like_weight(m, n, rank=64, outlier_frac=0.05, seed=0):
    torch.manual_seed(seed)
    U = torch.randn(m, rank, device=DEV) * 0.05
    V = torch.randn(rank, n, device=DEV) * 0.05
    W = U @ V
    # Sparse outlier channels (the part ternary destroys)
    n_out = int(m * outlier_frac)
    out_rows = torch.randperm(m, device=DEV)[:n_out]
    W[out_rows] += torch.randn(n_out, n, device=DEV) * 0.3
    return W


def bitnet_ternary(W):
    """BitNet b1.58: ternary {-1, 0, +1} with abs-mean scaling."""
    scale = W.abs().mean().item()
    W_t = torch.round(W / (scale + 1e-8)).clamp(-1, 1)
    return W_t * scale, scale


def int8_quant(W):
    """Per-row INT8 quantization."""
    scale = W.abs().max(dim=-1, keepdim=True).values / 127.0
    scale = scale.clamp(min=1e-8)
    W_q = (W / scale).round().clamp(-128, 127)
    return W_q * scale


def ternary_plus_residual(W, k_frac):
    """Ternary on all rows, then restore top-k% error rows to dense bf16."""
    W_t, _scale = bitnet_ternary(W)
    err = (W - W_t).abs().sum(dim=1)
    k = int(W.shape[0] * k_frac)
    if k == 0:
        return W_t, 0
    top_rows = err.topk(k).indices
    W_t[top_rows] = W[top_rows]  # restore to dense
    return W_t, k


def test_weight(m, n, label):
    W = make_trained_like_weight(m, n)
    x = torch.randn(1, 64, m, device=DEV) * 0.5
    y_ref = x @ W  # [1, 64, n] (W is [m, n])

    print(f"\n{'='*60}")
    print(f"Weight: {label} [{m}x{n}], dense={m*n*2/1024:.0f} KB")
    print(f"{'='*60}")
    print(f"{'method':>22} {'bytes/param':>12} {'compress':>10} {'out_err':>10}")

    # bf16 baseline
    print(f"{'bf16 (baseline)':>22} {'2.00':>12} {'1.0x':>10} {'0.0000':>10}")

    # INT8
    W_i = int8_quant(W)
    y_i = x @ W_i
    e_i = rel_err(y_i, y_ref)
    print(f"{'INT8':>22} {'1.00':>12} {'2.0x':>10} {e_i:>10.4f}")

    # Pure ternary
    W_t, _ = bitnet_ternary(W)
    y_t = x @ W_t
    e_t = rel_err(y_t, y_ref)
    # ternary = 1 byte/param (int8 storage of {-1,0,1})
    print(f"{'BitNet ternary':>22} {'1.00':>12} {'2.0x':>10} {e_t:>10.4f}")

    # Ternary + residual sweep
    for k_frac in [0.01, 0.02, 0.05, 0.10]:
        W_tr, k = ternary_plus_residual(W, k_frac)
        y_tr = x @ W_tr
        e_tr = rel_err(y_tr, y_ref)
        # bytes = (m-k)*1 (ternary) + k*n*2 (dense rows) ... per-param avg
        bytes_total = (m - k) * 1 + k * n * 2
        bpp = bytes_total / (m * n)
        cr = m * n * 2 / bytes_total
        print(f"{'ternary+res(k=%d,%.0f%%)' % (k, k_frac*100):>22} {bpp:>12.2f} {cr:>9.1f}x {e_tr:>10.4f}")

    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("BitNet ternary + residual k-sweep (plan §5.2)")
    print("Fix: keep top-k% outlier channels dense, rest ternary")
    print("=" * 60)
    test_weight(2048, 8192, "FFN w_gate (LFM2.5 size)")
    test_weight(4096, 4096, "Attention QKV (V8 size)")
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- If ternary+res(k=2%) err << pure ternary err -> outliers are the problem")
    print("- Find the k where err approaches INT8 -> that's the sweet spot")
    print("- Memory cost of k% residual = k% * 2 bytes extra per param")

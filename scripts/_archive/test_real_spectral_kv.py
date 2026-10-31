"""Spectral-KV on REAL LFM2.5 trained weights (plan §13.1). CUDA.

Tests whether the Fourier basis (§12.1 winner) fits K/V computed from REAL
trained attention weights better or worse than random projections.

Loads ForgeLM_V2_Light.safetensors, extracts attention layer 2 (W_q, W_k,
W_v), computes K/V over synthetic hidden states, fits Fourier basis, measures
attention-output error vs full cache and vs S4R.

Also tests: does the RoPE component dominate? (fit K_raw vs K_rope separately)

Runs on CUDA.
"""
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")

CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"
THETA = 1_000_000.0
HEAD_DIM = 64
N_HEADS = 32
N_KV_HEADS = 8
D_MODEL = 2048


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def rope_k(k, pos, theta=THETA, head_dim=HEAD_DIM):
    """Apply RoPE to key tensor k [seq, n_kv_heads, head_dim]."""
    seq, n_kv, hd = k.shape
    half = hd // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=DEV).float() / half))
    angles = pos.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    k_rot = torch.stack([k_even * cos - k_odd * sin,
                         k_even * sin + k_odd * cos], dim=-1)
    return k_rot.reshape(seq, n_kv, hd)


def fourier_basis(seq_len, max_freq):
    pos = torch.arange(seq_len, dtype=torch.float32, device=DEV)
    freqs = torch.arange(1, max_freq + 1, dtype=torch.float32, device=DEV)
    cos = torch.cos(2 * math.pi * pos.unsqueeze(1) * freqs.unsqueeze(0) / seq_len)
    sin = torch.sin(2 * math.pi * pos.unsqueeze(1) * freqs.unsqueeze(0) / seq_len)
    dc = torch.ones(seq_len, 1, device=DEV)
    return torch.cat([dc, cos, sin], dim=1)


def fit_fourier(target, basis):
    coef = torch.linalg.lstsq(basis, target).solution
    return basis @ coef


def attention_output(q, k, v):
    """q [seq, n_heads, hd], k/v [seq, n_kv, hd]. GQA."""
    _seq, n_heads, hd = q.shape
    n_kv = k.shape[1]
    rep = n_heads // n_kv
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum('qhd,khd->hqk', q, k_rep) / math.sqrt(hd)
    attn = F.softmax(scores, dim=-1)
    return torch.einsum('hqk,khd->qhd', attn, v_rep)


def load_real_weights(layer_idx=2):
    """Load real attention weights from LFM2.5 checkpoint."""
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W_q = f.get_tensor(f"blocks.{layer_idx}.attn.q_proj.weight").to(DEV)
        W_k = f.get_tensor(f"blocks.{layer_idx}.attn.k_proj.weight").to(DEV)
        W_v = f.get_tensor(f"blocks.{layer_idx}.attn.v_proj.weight").to(DEV)
    return W_q.float(), W_k.float(), W_v.float()


def test_real_spectral_kv(seq_lens=None, max_freq=64):
    if seq_lens is None:
        seq_lens = [512, 2048, 8192]
    W_q, W_k, W_v = load_real_weights(layer_idx=2)
    print("Loaded real LFM2.5 layer-2 weights:")
    print(f"  W_q: {W_q.shape}, W_k: {W_k.shape}, W_v: {W_v.shape}")
    print(f"  W_k stats: mean={W_k.mean():.4f}, std={W_k.std():.4f}, "
          f"max={W_k.abs().max():.4f}")

    results = []
    for seq_len in seq_lens:
        torch.manual_seed(42)
        # Synthetic hidden states (we don't have real tokens, but the WEIGHTS are real)
        x = torch.randn(seq_len, D_MODEL, device=DEV) * 0.5

        q = (x @ W_q.T).reshape(seq_len, N_HEADS, HEAD_DIM)
        k_raw = (x @ W_k.T).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        v = (x @ W_v.T).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        pos = torch.arange(seq_len, device=DEV)
        k_rope = rope_k(k_raw, pos)

        k_flat = k_rope.reshape(seq_len, -1)  # [seq, 512]
        v_flat = v.reshape(seq_len, -1)
        D = k_flat.shape[1]

        out_ref = attention_output(q, k_rope, v)

        # --- Fourier basis ---
        basis = fourier_basis(seq_len, max_freq)
        k_fourier = fit_fourier(k_flat, basis).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        v_fourier = fit_fourier(v_flat, basis).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        out_fourier = attention_output(q, k_fourier, v_fourier)
        err_fourier = rel_err(out_fourier, out_ref)
        k_err_fourier = rel_err(fit_fourier(k_flat, basis), k_flat)
        v_err_fourier = rel_err(fit_fourier(v_flat, basis), v_flat)
        bytes_fourier = (1 + 2 * max_freq) * D * 2 * 2

        # --- S4R at matched budget ---
        target_bytes = bytes_fourier
        rank = max(1, target_bytes // (2 * 2 * (seq_len + D)))
        rank = min(rank, min(seq_len, D) - 1)
        U, S, Vh = torch.linalg.svd(k_flat.float(), full_matrices=False)
        k_s4r = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        U2, S2, Vh2 = torch.linalg.svd(v_flat.float(), full_matrices=False)
        v_s4r = (U2[:, :rank] * S2[:rank]) @ Vh2[:rank, :]
        out_s4r = attention_output(q, k_s4r.reshape(seq_len, N_KV_HEADS, HEAD_DIM),
                                   v_s4r.reshape(seq_len, N_KV_HEADS, HEAD_DIM))
        err_s4r = rel_err(out_s4r, out_ref)
        bytes_s4r = rank * (seq_len + D) * 2 * 2

        # --- Also test K_raw (pre-RoPE) Fourier fit ---
        k_raw_flat = k_raw.reshape(seq_len, -1)
        k_raw_fourier = fit_fourier(k_raw_flat, basis)
        err_k_raw = rel_err(k_raw_fourier, k_raw_flat)

        bytes_full = seq_len * D * 2 * 2
        cr_f = bytes_full / bytes_fourier
        cr_s = bytes_full / bytes_s4r

        print(f"\n--- seq_len={seq_len}, max_freq={max_freq} ---")
        print(f"  Full cache:    {bytes_full/1024:.0f} KB")
        print(f"  Fourier:       {bytes_fourier/1024:.0f} KB ({cr_f:.1f}x)")
        print(f"    K(rope) err={k_err_fourier:.4f}  K(raw) err={err_k_raw:.4f}  "
              f"V err={v_err_fourier:.4f}")
        print(f"    ATTN OUT err={err_fourier:.4f}")
        print(f"  S4R (r={rank}): {bytes_s4r/1024:.0f} KB ({cr_s:.1f}x)")
        print(f"    ATTN OUT err={err_s4r:.4f}")

        results.append({
            'seq': seq_len, 'fourier_err': err_fourier, 's4r_err': err_s4r,
            'cr_f': cr_f, 'cr_s': cr_s,
            'k_rope_err': k_err_fourier, 'k_raw_err': err_k_raw,
        })
        if DEV.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("SUMMARY — REAL LFM2.5 weights:")
    print(f"{'seq':>6} {'method':>10} {'compress':>10} {'out_err':>10} {'K_err':>10}")
    for r in results:
        print(f"{r['seq']:>6} {'Fourier':>10} {r['cr_f']:>9.1f}x {r['fourier_err']:>10.4f} {r['k_rope_err']:>10.4f}")
        print(f"{'':>6} {'S4R':>10} {r['cr_s']:>9.1f}x {r['s4r_err']:>10.4f} {'':>10}")
    print(f"{'='*70}")
    print("KEY: compare K(rope) err vs K(raw) err — if rope err << raw err,")
    print("RoPE's sinusoidal structure makes K more Fourier-friendly (validates §8.3).")
    print("Compare to §12.1 random-projection results — real weights should be")
    print("MORE compressible (trained weights are smoother/low-rank).")


if __name__ == "__main__":
    print("=" * 70)
    print("Spectral-KV on REAL LFM2.5 trained weights (plan §13.1)")
    print("=" * 70)
    test_real_spectral_kv()

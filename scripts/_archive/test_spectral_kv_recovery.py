"""Spectral-KV + parent-child recovery test (plan §13.2). CUDA.

THE DREAM SCENARIO TEST: can a learned correction network drive the spectral-KV
attention error (0.095 on real weights) to near-zero?

If yes: spectral-KV + correction = effectively lossless O(1) KV memory.
This would be the biggest win in the project.

Setup:
  1. Load real LFM2.5 attention layer 2
  2. Compute full-cache attention output (PARENT, frozen)
  3. Compute spectral-KV attention output (CHILD, 0.095 error)
  4. Train a tiny correction MLP: spectral_out -> corrected_out (target = parent)
  5. Measure: does correction drive error to <0.01?

The correction MLP is per-head, small (input=hd, hidden=hd, output=hd).
It's the "recovery" step from the compress-then-recover framework (§8.6).

Also tests: correction capacity vs size (does a bigger correction do better?)
and: does correction generalize to unseen positions (train on first half of
sequence, test on second half)?

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


def attention_output(q, k, v):
    _seq, n_heads, hd = q.shape
    n_kv = k.shape[1]
    rep = n_heads // n_kv
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum('qhd,khd->hqk', q, k_rep) / math.sqrt(hd)
    attn = F.softmax(scores, dim=-1)
    return torch.einsum('hqk,khd->qhd', attn, v_rep)


def load_real_weights(layer_idx=2):
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W_q = f.get_tensor(f"blocks.{layer_idx}.attn.q_proj.weight").to(DEV)
        W_k = f.get_tensor(f"blocks.{layer_idx}.attn.k_proj.weight").to(DEV)
        W_v = f.get_tensor(f"blocks.{layer_idx}.attn.v_proj.weight").to(DEV)
    return W_q.float(), W_k.float(), W_v.float()


def test_recovery(seq_len=2048, max_freq=64, n_batches=8, hidden_mult=1):
    """Test spectral-KV + learned correction on real weights."""
    W_q, W_k, W_v = load_real_weights(layer_idx=2)

    # Generate multiple batches of synthetic hidden states (for training correction)
    torch.manual_seed(42)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(seq_len, D_MODEL, device=DEV) * 0.5
        q = (x @ W_q.T).reshape(seq_len, N_HEADS, HEAD_DIM)
        k_raw = (x @ W_k.T).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        v = (x @ W_v.T).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        pos = torch.arange(seq_len, device=DEV)
        k_rope = rope_k(k_raw, pos)
        batches.append((q, k_rope, v, k_raw))

    # Compute parent (full cache) and child (spectral-KV) outputs for all batches
    basis = fourier_basis(seq_len, max_freq)
    parent_outs = []
    child_outs = []
    for q, k_rope, v, k_raw in batches:
        out_parent = attention_output(q, k_rope, v)  # [seq, n_heads, hd]
        k_flat = k_rope.reshape(seq_len, -1)
        v_flat = v.reshape(seq_len, -1)
        k_fourier = fit_fourier(k_flat, basis).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        v_fourier = fit_fourier(v_flat, basis).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
        out_child = attention_output(q, k_fourier, v_fourier)
        parent_outs.append(out_parent)
        child_outs.append(out_child)

    # Baseline error (no correction)
    baseline_err = sum(rel_err(c, p) for c, p in zip(child_outs, parent_outs)) / len(batches)

    # --- Correction MLP: RESIDUAL correction (child + correction(child) = parent) ---
    # KEY FIX: zero-init last layer so corrected = child at init (no noise added).
    # The MLP learns only the SMALL RESIDUAL (parent - child), not the full parent.
    hd = HEAD_DIM
    hidden = hd * hidden_mult
    correction = torch.nn.Sequential(
        torch.nn.Linear(hd, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hd)
    ).to(DEV)
    # Zero-init last layer: at init, correction(child) = 0, so corrected = child
    torch.nn.init.zeros_(correction[-1].weight)
    torch.nn.init.zeros_(correction[-1].bias)

    # Per-head correction (shared MLP, applied to each head independently)
    opt = torch.optim.Adam(correction.parameters(), lr=1e-3)

    # Train on first 75% of positions, test on last 25% (generalization test)
    split = int(seq_len * 0.75)

    for epoch in range(200):
        total_loss = 0
        for i in range(n_batches):
            child = child_outs[i][:split]  # [split, n_heads, hd]
            parent = parent_outs[i][:split]
            # RESIDUAL correction: corrected = child + correction(child)
            corrected = child + correction(child)  # [split, n_heads, hd]
            loss = (corrected - parent).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"  epoch {epoch}: loss={total_loss/n_batches:.6f}")

    # Evaluate
    with torch.no_grad():
        # Train positions
        train_errs = []
        for i in range(n_batches):
            corrected = child_outs[i][:split] + correction(child_outs[i][:split])
            train_errs.append(rel_err(corrected, parent_outs[i][:split]))
        # Test positions (unseen during training)
        test_errs = []
        for i in range(n_batches):
            corrected = child_outs[i][split:] + correction(child_outs[i][split:])
            test_errs.append(rel_err(corrected, parent_outs[i][split:]))

    train_err = sum(train_errs) / len(train_errs)
    test_err = sum(test_errs) / len(test_errs)
    n_params = sum(p.numel() for p in correction.parameters())

    print(f"\n--- seq={seq_len}, max_freq={max_freq}, hidden={hidden}, "
          f"batches={n_batches} ---")
    print(f"  Baseline (spectral-KV only):    {baseline_err:.4f}")
    print(f"  + correction (train positions): {train_err:.4f}")
    print(f"  + correction (test positions):  {test_err:.4f}")
    print(f"  Correction params: {n_params} ({n_params*2/1024:.0f} KB)")
    print(f"  Improvement: {baseline_err/max(test_err,1e-6):.1f}x error reduction")

    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
    return baseline_err, train_err, test_err, n_params


def fit_fourier(target, basis):
    coef = torch.linalg.lstsq(basis, target).solution
    return basis @ coef


if __name__ == "__main__":
    print("=" * 70)
    print("Spectral-KV + parent-child recovery (plan §13.2)")
    print("Can a learned correction drive 0.095 error to near-zero?")
    print("=" * 70)

    results = []
    # Test different correction sizes
    for hm in [1, 2, 4]:
        print(f"\n{'='*60}")
        print(f"Correction hidden = {HEAD_DIM * hm} ({hm}x head_dim)")
        print(f"{'='*60}")
        r = test_recovery(seq_len=2048, max_freq=64, n_batches=8, hidden_mult=hm)
        results.append(('hidden=%dx' % hm, r))

    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'config':>16} {'baseline':>10} {'train_err':>10} {'test_err':>10} {'improvement':>12}")
    for name, (base, tr, te, _np) in results:
        print(f"{name:>16} {base:>10.4f} {tr:>10.4f} {te:>10.4f} {base/max(te,1e-6):>11.1f}x")
    print(f"{'='*70}")
    print("KEY: if test_err < 0.01, spectral-KV + correction = effectively lossless")
    print("O(1) KV memory. The correction MLP is tiny and per-head (shared).")
    print("If test_err >> train_err, correction overfits (doesn't generalize).")

"""EXP-1 NDP: exact attention split across GPU-window + host-tier KV.

Compares per-token cost of:
  A) full attention over KV resident on GPU          (baseline ceiling)
  B) NDP: host computes far partial, GPU combines    (exact, no migration)
  C) migrate far pages H2D each step, then attend    (naive tiered)
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from forge.exp.harness import log_result
from forge.exp.ndp import ndp_attention, attn_partial, ndp_combine, \
    transfer_bytes

H, D = 14, 64          # Qwen2.5-0.5B query heads, head_dim
T_LOCAL = 1024


def bench(fn, iters=10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    torch.manual_seed(0)
    dev = "cuda"
    for T_far in (4096, 16384):
        q = torch.randn(H, D, device=dev, dtype=torch.bfloat16)
        K_l = torch.randn(H, T_LOCAL, D, device=dev, dtype=torch.bfloat16)
        V_l = torch.randn_like(K_l)
        K_f = torch.randn(H, T_far, D, dtype=torch.bfloat16)  # host
        V_f = torch.randn_like(K_f)

        # A) all-on-GPU reference (far pages resident — the ideal tier)
        K_a = torch.cat([K_l, K_f.to(dev)], dim=1)
        V_a = torch.cat([V_l, V_f.to(dev)], dim=1)

        def run_gpu():
            return attn_partial(q, K_a, V_a)

        out_ref = run_gpu()
        t_gpu = bench(run_gpu)

        # B) NDP: host partial + combine
        def run_ndp():
            return ndp_attention(q, (K_l, V_l), (K_f, V_f))

        out_ndp = run_ndp()
        t_ndp = bench(run_ndp)

        # exactness check vs reference
        m_r, s_r, acc_r = out_ref
        err = float((acc_r / s_r.unsqueeze(-1) - out_ndp).abs().max())

        # C) migrate pages then attend
        def run_migrate():
            Km = K_f.to(dev, non_blocking=True)
            Vm = V_f.to(dev, non_blocking=True)
            return attn_partial(q, torch.cat([K_l, Km], 1),
                                torch.cat([V_l, Vm], 1))

        run_migrate()
        t_mig = bench(run_migrate)

        ndp_b = transfer_bytes(H, D, T_far, 2, migrate=False)
        mig_b = transfer_bytes(H, D, T_far, 2, migrate=True)
        log_result("ndp", phase="tiered_attn", T_far=T_far,
                   t_gpu_ms=round(t_gpu * 1e3, 3),
                   t_ndp_ms=round(t_ndp * 1e3, 3),
                   t_migrate_ms=round(t_mig * 1e3, 3),
                   max_err=round(err, 5),
                   ndp_bytes=ndp_b, migrate_bytes=mig_b,
                   bytes_ratio=round(mig_b / max(ndp_b, 1), 1))
    print("[ndp] done")


if __name__ == "__main__":
    main()

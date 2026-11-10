"""Spark research probes — viability tests for the 2026-11 research topics.

Subcommands:
  prism   PRISM-lite (innovation-augmented polar / Muon family) on SIFT
          deltas vs fused AdamW, identical SIFT setup + batches.
  es      SIFT-ES: backward-free per-slice scale evolution on top of the
          Newton-initialized deltas (OpenAI-ES, antithetic, rank-shaped).
  vocab   Two-stage vocab decode viability: kmeans centroid shortlist
          recall of the true top-k logits on real hidden states.
  hdc     Hyperdimensional (random near-orthogonal) codebook vs kmeans
          codebook MSE on a real Spark weight matrix (CPU).
  assoc   Linear associative memory (delta rule) retrieval capacity (CPU).

Run: venv/Scripts/python scripts/bench_spark_research.py <probe>
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_sift as B
from forge.training import sift as sift_mod
from forge.training.optim.muon_sf_blockwise import _newton_schulz
from research.tokenizer_cache import get_tokenizer


def _setup(device, newton=0):
    tokenizer = get_tokenizer(B.CKPT)
    dataset, batches = B.build_batches(tokenizer, 64, 4, 2048, device)
    model = B.build_model(device)

    def calib_iter():
        yield from ((b[0], b[1], b[2]) for b in batches)

    st = sift_mod.setup_sift(
        model, calib_iter, lambda i, l, m: B.ce_loss(model, i, l, m),
        attn_units=2, ffn_units=4, attn_gran=128, ffn_gran=16,
        frontier="auto", calib_batches=4, newton_passes=newton)
    return model, st, batches


# ── PRISM-lite ───────────────────────────────────────────────────────────


class PrismLite(torch.optim.Optimizer):
    """Muon-family update with innovation augmentation (PRISM arXiv2602.03096).

    M <- lerp(M,g); D = g-M; U = NS([M, gamma*D]) -> take the M block.
    Innovation-heavy (high-variance) directions get spectrally suppressed.
    RMS-normalized so lr is comparable to AdamW's per-entry step.
    """

    def __init__(self, params, lr=2e-5, beta=0.95, gamma=0.5, ns_steps=5):
        super().__init__(list(params), dict(lr=lr, beta=beta, gamma=gamma,
                                            ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, beta, gamma = group["lr"], group["beta"], group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                m.lerp_(p.grad, 1 - beta)
                d = p.grad - m
                aug = torch.cat([m, gamma * d], dim=1 if m.shape[0] <= m.shape[1] else 0)
                u = _newton_schulz(aug, steps=group["ns_steps"])
                u = u[:, :m.shape[1]] if m.shape[0] <= m.shape[1] else u[:m.shape[0]]
                u = u.float()
                p.add_(-lr * (u / (u.pow(2).mean().sqrt() + 1e-8)).to(p.dtype))


def probe_prism():
    device = "cuda"
    out = {}
    for arm in ("adamw", "prism"):
        model, st, batches = _setup(device)
        opt = (torch.optim.AdamW(st.delta_params, lr=2e-5, fused=True)
               if arm == "adamw" else PrismLite(st.delta_params, lr=2e-5))
        losses = []
        B.run_steps(model, opt, batches, 2, losses)  # warmup
        losses = []
        times, vram = B.run_steps(model, opt, batches, 10, losses)
        out[arm] = dict(loss0=losses[0], lossN=losses[-1],
                        step_ms=1000 * sum(times) / len(times), vram_gb=vram)
        print(f"[prism:{arm}] {out[arm]}", flush=True)
        del model, opt, st
        torch.cuda.empty_cache()
    print("PRISM_VERDICT", out, flush=True)


# ── SIFT-ES ──────────────────────────────────────────────────────────────


def probe_es():
    device = "cuda"
    model, st, batches = _setup(device, newton=4)
    mods = [m for m in model.modules() if getattr(m, "_sift_side", None) is not None]
    base = [m.sift_delta.detach().clone() for m in mods]
    eval_batches = batches[:2]
    holdout = batches[2]

    def loss_at(theta, bs):
        with torch.no_grad():
            for m, b, t in zip(mods, base, theta):
                m.sift_delta.copy_(b * t)
            return sum(float(B.ce_loss(model, b[0], b[1], b[2])) for b in bs) / len(bs)

    n = len(mods)
    theta = torch.ones(n, device=device)
    l0 = loss_at(theta, eval_batches)
    lh0 = loss_at(theta, [holdout])
    print(f"[es] slices={n} init loss={l0:.4f} holdout={lh0:.4f}", flush=True)
    pop, sigma, lr_es, iters = 8, 0.05, 0.05, 16
    for it in range(iters):
        eps = torch.randn(pop // 2, n, device=device)
        eps = torch.cat([eps, -eps])  # antithetic
        ls = torch.tensor([loss_at((theta + sigma * e).clamp(min=0), eval_batches)
                           for e in eps], device=device)
        w = -(ls - ls.mean()) / (ls.std() + 1e-8)  # rank-free shaped weights
        theta = (theta + lr_es * (w[:, None] * eps).mean(0) / sigma).clamp(min=0)
        if it % 4 == 0 or it == iters - 1:
            print(f"[es] it={it} meanL={ls.mean():.4f} best={ls.min():.4f}", flush=True)
    lf = loss_at(theta, eval_batches)
    lhf = loss_at(theta, [holdout])
    print(f"ES_VERDICT init={l0:.4f}->{lf:.4f} holdout={lh0:.4f}->{lhf:.4f}", flush=True)


# ── two-stage vocab shortlist ────────────────────────────────────────────


def probe_vocab():
    device = "cuda"
    model = B.build_model(device)
    head = None
    for name in ("head", "lm_head", "output", "output_proj"):
        h = getattr(model, name, None)
        if isinstance(h, torch.nn.Module) and hasattr(h, "weight"):
            head = h
            break
    assert head is not None, "no head module found"
    W = head.weight  # [V, d] tied embedding/head
    V, d = W.shape
    C = 256  # centroids
    # kmeans on a row subsample, then assign all rows (one big GEMM)
    sub = W[torch.randperm(V, device=device)[:16384]].float()
    cent = sub[torch.randperm(len(sub))[:C]].clone()
    for _ in range(12):
        a = (sub @ cent.T).argmax(1)
        for c in range(C):
            m = a == c
            if m.any():
                cent[c] = sub[m].mean(0)
    assign = (W.float() @ cent.T).argmax(1)
    sizes = torch.bincount(assign, minlength=C).float()
    hs = []
    handle = head.register_forward_pre_hook(lambda mod, args: hs.append(args[0].detach()))
    tokenizer = get_tokenizer(B.CKPT)
    _, batches = B.build_batches(tokenizer, 16, 2, 1024, device)
    with torch.no_grad():
        for b in batches[:4]:
            model(b[0], attention_mask=b[2])
    handle.remove()
    h = torch.cat([x.reshape(-1, x.shape[-1]) for x in hs]).float()[:4096]
    for top_clusters in (8, 32, 64):
        recalls = []
        for i in range(0, len(h), 256):
            hh = h[i:i + 256]
            logits = hh @ W.float().T
            true_top = logits.topk(10, dim=1).indices  # [B,10]
            cs = hh @ cent.T  # [B,C]
            cand_c = cs.topk(top_clusters, dim=1).indices
            mask = torch.zeros(V, dtype=torch.bool, device=device)
            for c in cand_c.unique():
                mask |= assign == c
            cand = mask.nonzero().squeeze(1)
            sub_logits = logits[:, cand]
            est_top = cand[sub_logits.topk(10, dim=1).indices]
            hit = (est_top.unsqueeze(2) == true_top.unsqueeze(1)).any(1).float().mean()
            recalls.append(float(hit))
        print(f"VOCAB_VERDICT clusters={top_clusters}/{C} "
              f"recall@10={sum(recalls)/len(recalls):.4f} "
              f"shortlist~{int(sizes.mean()*top_clusters)}/{V}", flush=True)


# ── HDC codebook vs kmeans ───────────────────────────────────────────────


def probe_hdc():
    from safetensors import safe_open
    import glob
    files = sorted(glob.glob(str(ROOT / B.CKPT / "*.safetensors")))
    W = None
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith("out_proj.weight") and ("layers.10." in k or "blocks.10." in k):
                    W = sf.get_tensor(k).float()
                    break
        if W is not None:
            break
    assert W is not None
    torch.manual_seed(0)
    X = W.reshape(-1, 8)  # 8-dim vectors (E8-style VQ)
    sub = X[torch.randperm(len(X))[:32768]]
    K = 256
    cent = sub[torch.randperm(len(sub))[:K]].clone()
    for _ in range(10):
        a = torch.cdist(sub, cent).argmin(1)
        for c in range(K):
            m = a == c
            if m.any():
                cent[c] = sub[m].mean(0)
    rand = torch.nn.functional.normalize(torch.randn(K, 8), dim=1) * X.norm(dim=1).mean()
    mse_k = ((X - cent[torch.cdist(X, cent).argmin(1)]) ** 2).mean()
    mse_r = ((X - rand[torch.cdist(X, rand).argmin(1)]) ** 2).mean()
    var = X.pow(2).mean()
    print(f"HDC_VERDICT mse_kmeans={mse_k:.6f} mse_random={mse_r:.6f} "
          f"ratio={float(mse_r/mse_k):.3f} rel_kmeans={float(mse_k/var):.5f} "
          f"rel_random={float(mse_r/var):.5f}", flush=True)


# ── linear associative memory capacity ───────────────────────────────────


def probe_assoc():
    torch.manual_seed(0)
    d = 64
    out = []
    for n in (16, 32, 64, 96, 128, 256):
        K = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
        Vv = torch.randn(n, d)
        S = Vv.T @ K  # delta-rule linear memory write (Hebbian)
        rec = (S @ K.T).T  # [n,d] recalled values
        acc = (rec @ Vv.T).argmax(1).eq(torch.arange(n)).float().mean()
        out.append((n, round(float(acc), 3)))
    print(f"ASSOC_VERDICT d={d} recall_by_n={out} (capacity cliff near n=d)",
          flush=True)


if __name__ == "__main__":
    probe = sys.argv[1]
    dict(prism=probe_prism, es=probe_es, vocab=probe_vocab,
         hdc=probe_hdc, assoc=probe_assoc)[probe]()

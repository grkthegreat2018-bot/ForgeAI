"""probe_exact_decode_tradeoff.py — exact-output decode with fractional KV cache.

Hypothesis: decode keeps K,V for all T positions (memory ~ T*B_kv).
Exact alternative: keep K,V for a SUFFIX WINDOW of f*T positions only; the
remaining (1-f)*T positions are recomputed from token ids through the full
stack each step. Identical math (modulo GEMM reduction order), KV memory ~ f*T,
extra FLOPs ~ (1-f)*T forward per step.

Layout:  [ missing prefix: recomputed ][ kept suffix window: cached ][ new tok ]
A missing row at prefix index j attends ONLY to missing rows 0..j (kept rows
are in its future). The new token attends to everything.

Per f: peak resident KV bytes, median decode ms/tok, sampled-seq divergence
vs f=1, and max|K diff| between recomputed and originally-prefilled keys
(measures GEMM-shape bitwise fidelity).
"""
import torch, time
torch.manual_seed(0)
dev, DT = "cuda", torch.float16

D, H, DH, L, V = 512, 8, 64, 4, 32000
T_PRE, T_GEN = 2048, 128

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for n, i, o in (("wq", D, H*DH), ("wk", D, H*DH), ("wv", D, H*DH),
                        ("wo", H*DH, D), ("w1", D, 4*D), ("w2", 4*D, D)):
            setattr(self, n, torch.nn.Linear(i, o, bias=False, dtype=DT, device=dev))
        self.n1 = torch.nn.RMSNorm(D, dtype=DT, device=dev)
        self.n2 = torch.nn.RMSNorm(D, dtype=DT, device=dev)
    def qkv(self, x):
        return self.wq(x), self.wk(x), self.wv(x)
    def mix(self, a):  # a = residual after attn; returns a + ffn(n2(a))
        return a + torch.nn.functional.silu(self.w1(self.n2(a))) @ self.w2.weight.T

def sdpa_reshape(t):  # [m, H*DH] -> [1, H, m, DH]
    return t.view(-1, H, DH).transpose(0, 1).unsqueeze(0)

def attn_out(blk, x, k, v, mask=None):  # x:[m,H*DH] k,v:[n,H*DH]
    o = torch.nn.functional.scaled_dot_product_attention(
        sdpa_reshape(blk.wq(blk.n1(x))), sdpa_reshape(k), sdpa_reshape(v),
        attn_mask=mask, is_causal=mask is None)
    return blk.wo(o.transpose(1, 2).reshape(x.shape[0], H*DH))

blocks = [Block() for _ in range(L)]
emb = torch.nn.Embedding(V, D, dtype=DT, device=dev)
head = torch.nn.Linear(D, V, bias=False, dtype=DT, device=dev)
ids = torch.randint(0, V, (T_PRE,), device=dev)

def stack(h, kv_override=None, kept=None):
    """Run all layers on h [m,D]. kv_override: list of (k,v) to use as the
    full prefix (decode-time recompute+kept path). kept: list of (kc,vc)
    appended AFTER h's own kv (suffix window layout). Returns (h_out, Ks, Vs)."""
    m = h.shape[0]
    Ks, Vs = [], []
    for li, blk in enumerate(blocks):
        x1 = blk.n1(h)
        _, k, v = blk.qkv(x1)
        if kv_override is not None:
            k, v = kv_override[li]
        if kept is not None:
            kc, vc = kept[li]
            kfull = torch.cat([k, kc]); vfull = torch.cat([v, vc])
            nkc = kc.shape[0]
            # row j sees own-kv rows 0..j ; last row additionally sees kc
            bias = torch.full((m, m + nkc), float("-inf"), dtype=DT, device=dev)
            bias[:, :m] = torch.triu(
                torch.full((m, m), float("-inf"), dtype=DT, device=dev), 1)
            bias[-1, m:] = 0.0
            o = attn_out(blk, h, kfull, vfull, mask=bias.unsqueeze(0).unsqueeze(0))
        else:
            Ks.append(k); Vs.append(v)
            o = attn_out(blk, h, k, v)  # is_causal batched
        h = blk.mix(h + o)
    return h, Ks, Vs

@torch.inference_mode()
def prefill():
    h, Ks, Vs = stack(emb(ids))
    return Ks, Vs, h[T_PRE-1:T_PRE].clone()

@torch.inference_mode()
def run(frac, refK=None, forced=None):
    n_miss = T_PRE - int(T_PRE * frac)
    keep_idx = torch.arange(n_miss, T_PRE, device=dev)
    Ks, Vs, cur = prefill()
    kept = [(Ks[l][keep_idx].contiguous(), Vs[l][keep_idx].contiguous()) for l in range(L)]
    miss_ids = ids[:n_miss].clone()
    kv_bytes = int(sum(k.numel() + v.numel() for k, v in kept) * 2)
    maxdiff = None
    if refK is not None:  # does masked-recompute reproduce prefilled K bitwise?
        xb, _, _ = stack(emb(miss_ids), kv_override=None) if False else (None, None, None)
        h = emb(miss_ids); kd = 0.0
        for li, blk in enumerate(blocks):
            x1 = blk.n1(h); _, k, v = blk.qkv(x1)
            kd = max(kd, (k - refK[li][:n_miss]).abs().max().item())
            o = attn_out(blk, h, k, v)  # pure causal, same as prefill math
            h = blk.mix(h + o)
        maxdiff = kd
    toks, lat, gen_ids, logits_hist = [], [], [], []
    for step in range(T_GEN):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg = head(cur)
        logits_hist.append(lg.clone())
        nxt = int(lg.argmax(-1))
        feed = forced[step] if forced is not None else nxt
        toks.append(nxt); gen_ids.append(feed)
        # recompute set = missing prefix + generated-so-far + new token.
        # true order: missing(0..n_miss-1) < kept(n_miss..T-1) < gen < nxt.
        # kfull layout: [km = missing+gen+nxt (m rows)] ++ [kc = kept (nkc)]
        rec_ids = torch.cat([miss_ids,
                             torch.tensor(gen_ids, device=dev, dtype=torch.long)])
        xb = emb(rec_ids)
        m = xb.shape[0]
        for li, blk in enumerate(blocks):
            x1 = blk.n1(xb)
            q, km, vm = blk.qkv(x1)
            kc, vc = kept[li]
            kfull = torch.cat([km, kc]); vfull = torch.cat([vm, vc])
            nkc = kc.shape[0]
            # row j sees km[0..j]; rows j>=n_miss (gen tokens, newest) also see
            # the kept window (kept is in their past); missing rows do not.
            bias = torch.triu(torch.full((m, m), float("-inf"), dtype=DT, device=dev), 1)
            kc_mask = torch.full((m, nkc), float("-inf"), dtype=DT, device=dev)
            kc_mask[n_miss:, :] = 0.0
            bias = torch.cat([bias, kc_mask], 1)
            o = torch.nn.functional.scaled_dot_product_attention(
                sdpa_reshape(blk.wq(x1)), sdpa_reshape(kfull), sdpa_reshape(vfull),
                attn_mask=bias.view(1, 1, m, m + nkc))
            xb = blk.mix(xb + blk.wo(o.transpose(1, 2).reshape(m, H * DH)))
        cur = xb[-1:]
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1e3)
    return toks, sorted(lat)[len(lat)//2], kv_bytes, maxdiff, logits_hist

print(f"L={L} D={D} H={H} DH={DH} T={T_PRE} gen={T_GEN} fp16 cuda")
refKs, _, _ = prefill()
toks_b, ms_b, kvb_b, _, logits_b = run(1.0)
print(f"frac=1.000  KV={kvb_b/1e6:7.2f}MB  {ms_b:7.2f}ms/tok  toks[:8]={toks_b[:8]}")
for f in (0.5, 0.25, 0.125, 0.0):
    # teacher-forced: feed baseline tokens, compare logits stepwise
    toks, ms, kvb, kd, logits_f = run(f, refK=refKs, forced=toks_b)
    dmax = max((a - b).abs().max().item() for a, b in zip(logits_f, logits_b))
    agree = sum(int(a.argmax(-1) == b) for a, b in zip(logits_f, toks_b))
    print(f"frac={f:.3f}  KV={kvb/1e6:7.2f}MB  {ms:7.2f}ms/tok  "
          f"max_dlogit={dmax:.3e}  argmax_agree={agree}/{len(toks_b)}  "
          f"maxK_recomp={kd:.3e}")

# ---- fp64 CPU control: same math, no kernel-tiling noise ------------------
print("\n[fp64 cpu] shape-invariant math — true exactness control")
dev, DT = "cpu", torch.float64
D, H, DH, L, V = 256, 4, 32, 2, 1024
T_PRE, T_GEN = 192, 8
blocks = [Block() for _ in range(L)]
emb = torch.nn.Embedding(V, D, dtype=DT, device=dev)
head = torch.nn.Linear(D, V, bias=False, dtype=DT, device=dev)
ids = torch.randint(0, V, (T_PRE,), device=dev)
refKs64, _, _ = prefill()
tb, _, _, _, lb = run(1.0)
_, _, _, kd, lf = run(0.0, refK=refKs64, forced=tb)
dmax = max((a - b).abs().max().item() for a, b in zip(lf, lb))
eq = torch.equal(lf[0], lb[0])
print(f"frac=0 vs 1: max_dlogit={dmax:.3e}  bitwise_logits={eq}  maxK_recomp={kd:.3e}")

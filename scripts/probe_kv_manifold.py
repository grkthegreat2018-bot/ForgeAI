"""probe_kv_manifold.py — is the KV cache information-redundant on real weights?

Loads proposer_mla_moe (Qwen2.5-0.5B -> GLA(d_c=256)+MoE port, bf16), forwards
real text, captures per-layer (k,v) presents. Per attention layer measures:
  - bytes actually stored per token (reconstructed per-head K,V)
  - distinct-head duplication factor (GLA repeats kv->q heads before caching)
  - effective rank of K,V [T, n_h*hd] vs the d_c=256 latent the model computes
  - bf16 byte entropy -> lossless floor; temporal delta entropy -> delta bound
  - int8/int4 per-token recon rel-err
  - END-TO-END: last-position logits diff when past KV is int8/int4 quantized
    (exactly what a lossy cache would do to the real model on real text)
"""
import torch, numpy as np
from forge.config import get_config
from forge.model_loader import ModelLoader
from transformers import AutoTokenizer

CKPT = "forge/evolution/proposer_mla_moe"
T_MAX = 1024

def h_bits(b: np.ndarray) -> float:
    c = np.bincount(b.ravel(), minlength=256).astype(np.float64)
    p = c[c > 0] / b.size
    return float(-(p * np.log2(p)).sum())

def quant_err(k: torch.Tensor, bits: int) -> float:
    # per-(head,position) symmetric quant over head_dim
    qmax = 2 ** (bits - 1) - 1
    s = k.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    kq = (k / s).round().clamp(-qmax, qmax) * s
    return (kq - k).norm().item() / k.norm().item()

cfg = get_config("qwen25_05b_mla_moe", device="cuda")
print(f"loading {CKPT} (d={cfg.d_model} L={cfg.n_layers} nh={cfg.n_heads} "
      f"nkv={cfg.n_kv_heads} hd={cfg.head_dim} dc={getattr(cfg,'latent_dim',None)})")
model = ModelLoader.build_model_fast(cfg, checkpoint_path=CKPT,
                                     dtype=torch.bfloat16).eval().cuda()
tok = AutoTokenizer.from_pretrained(CKPT)
src = open("forge/model/layers.py", encoding="utf8").read()[:6000] + \
      "\n\nThe transformer cache stores key and value projections for every " \
      "prior position. If those projections lie on a low-dimensional manifold, " \
      "the cache stores far more bytes than the information requires.\n" * 4
ids = tok(src, return_tensors="pt").input_ids[:, :T_MAX].cuda()
T = ids.shape[1]

with torch.inference_mode():
    logits, _, presents = model(ids, use_cache=True)
attn = [(i, p) for i, p in enumerate(presents) if isinstance(p, tuple)]
print(f"T={T}  attn layers={len(attn)} of {len(presents)}\n")
hdr = (f"{'lyr':>3} {'B/tok':>6} {'dup':>4} {'rk99K':>6} {'rk99V':>6} "
       f"{'H(K)':>5} {'H(V)':>5} {'H(dK)':>6} {'cosK':>6} {'i8K':>6} {'i4K':>6}")
print(hdr); print("-" * len(hdr))
tot_bpt = 0.0; tot_uniq = 0.0
for li, (k, v) in attn:
    k = k[0].float(); v = v[0].float()          # [n_h, T, hd]
    nh, T_, hd = k.shape
    K2 = k.permute(1, 0, 2).reshape(T_, nh * hd)
    V2 = v.permute(1, 0, 2).reshape(T_, nh * hd)
    def r99(M):
        sv = torch.linalg.svdvals(M)
        c = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        return int(torch.searchsorted(c, torch.tensor(0.99, device=M.device))) + 1
    rkK, rkV = r99(K2), r99(V2)
    # duplication: fraction of (k,v) head pairs identical to another head
    kh = k.reshape(nh, -1); vh = v.reshape(nh, -1)
    dup = 1 - (torch.unique(kh, dim=0).shape[0] +
               torch.unique(vh, dim=0).shape[0]) / (2 * nh)
    kb = k.to(torch.bfloat16).view(torch.uint8).cpu().numpy()
    vb = v.to(torch.bfloat16).view(torch.uint8).cpu().numpy()
    dk = (k[:, 1:, :] - k[:, :-1, :]).to(torch.bfloat16).view(torch.uint8).cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(k[:, 1:, :], k[:, :-1, :], -1).mean()
    bpt = (k.numel() + v.numel()) * 2 / T_
    tot_bpt += bpt; tot_uniq += bpt * (1 - dup)
    print(f"{li:>3} {bpt:6.0f} {dup:4.2f} {rkK:>6} {rkV:>6} "
          f"{h_bits(kb):5.2f} {h_bits(vb):5.2f} {h_bits(dk):6.2f} {cos:6.3f} "
          f"{quant_err(k, 8):6.4f} {quant_err(k, 4):6.4f}")
print("-" * len(hdr))
print(f"TOT  {tot_bpt:6.0f} B/tok/layer summed={tot_bpt:,.0f}  "
      f"dedup'd={tot_uniq:,.0f} ({tot_bpt/max(tot_uniq,1):.1f}x head-dup factor)")

# ---- end-to-end: logits with quantized past vs exact past ----
last = ids[:, -1:]
with torch.inference_mode():
    ref = model(last, past_key_values=presents, use_cache=True)[0][0, -1].float()
for bits in (8, 4):
    qp = []
    for p in presents:
        if isinstance(p, tuple):
            k, v = p
            qm = 2 ** (bits - 1) - 1
            ks = k.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qm
            vs = v.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qm
            qp.append(((k / ks).round().clamp(-qm, qm) * ks,
                       (v / vs).round().clamp(-qm, qm) * vs))
        else:
            qp.append(p)
    with torch.inference_mode():
        lq = model(last, past_key_values=qp, use_cache=True)[0][0, -1].float()
    same = int(ref.argmax() == lq.argmax())
    print(f"int{bits} past-KV: max_dlogit={(ref-lq).abs().max():.3e}  "
          f"cos={torch.nn.functional.cosine_similarity(ref, lq, dim=0):.5f}  "
          f"argmax_same={same}")

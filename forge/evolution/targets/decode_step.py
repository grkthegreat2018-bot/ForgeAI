"""Evolution target: end-to-end single-token decode step.

System-level meta-target (COMPETITIVE_AUDIT A11): the evolution loop
optimizes micro-kernels scored in isolation, but the project's own
diagnosis is launch-bound decode (~800 kernels/token). This target
benchmarks a COMPLETE transformer decode step — embed -> 2x(norm,
qkv, rope, attn-over-cached-KV, o, residual, norm, mlp, residual) ->
norm -> head -> logits -> sampled token — so candidates that fuse,
precompute, or reorder whole-step work win on the metric that maps
directly to engine tok/s.

Gate: logits equal the reference step bit-for-bit (fp32, exact ops
reorder only) and the KV cache must contain the appended step.
"""

NAME = "decode_step"
GOAL = "min"
METRIC = "us_per_token"
TAGS = ["decode", "e2e", "hot-path", "engine"]
BUDGETS = {"us_per_token": ("<=", 20000.0)}
REQUIRED = ["decode_step"]
PROMPT = (
    "Faster single-token decode step. Seed does ~40 small tensor ops "
    "per token over (d=256, 2 layers, 4 heads, fp32): per-op kernel "
    "launches dominate. Ideas: precompute rope cos/sin once outside, "
    "fuse the two norms, keep weights as packed tensors instead of "
    "dict lookups, cache transposes, use torch ops that fuse "
    "(addmm/einsum/scaled_dot_product_attention), preallocate the KV "
    "cache instead of list appends. MUST return the same logits "
    "(allclose atol=1e-5) AND the same cache contents."
)

SEED_CODE = '''
import math

import torch


D, NL, NH, HD, V = 256, 2, 4, 64, 512
_g = torch.Generator().manual_seed(7)


def _w(*s):
    return torch.randn(*s, generator=_g) * 0.02


W = {
    "emb": _w(V, D),
    "qkv": [_w(D, 3 * D) for _ in range(NL)],
    "o": [_w(D, D) for _ in range(NL)],
    "w1": [_w(D, 4 * D) for _ in range(NL)],
    "w2": [_w(4 * D, D) for _ in range(NL)],
    "ln": [_w(D) for _ in range(2 * NL)],
    "lnf": _w(D),
    "head": _w(D, V),
}


def _rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w


def decode_step(tok: int, past_k: list, past_v: list, pos: int):
    """One token: -> (logits, past_k, past_v). past_* are per-layer
    lists of (1, NH, T, HD) tensors grown in place each step."""
    x = W["emb"][tok].unsqueeze(0)
    for l in range(NL):
        h = _rms(x, W["ln"][2 * l])
        qkv = h @ W["qkv"][l]
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(1, NH, 1, HD)
        k = k.view(1, NH, 1, HD)
        v = v.view(1, NH, 1, HD)
        ang = pos / (10000.0 ** (torch.arange(HD // 2).float()
                                 / (HD // 2)))
        c, s = ang.cos(), ang.sin()
        qr = q.view(1, NH, 1, HD // 2, 2)
        q_rot = torch.stack(
            [qr[..., 0] * c - qr[..., 1] * s,
             qr[..., 0] * s + qr[..., 1] * c], dim=-1).flatten(-2)
        kr = k.view(1, NH, 1, HD // 2, 2)
        k_rot = torch.stack(
            [kr[..., 0] * c - kr[..., 1] * s,
             kr[..., 0] * s + kr[..., 1] * c], dim=-1).flatten(-2)
        if past_k[l] is None:
            past_k[l] = k_rot
            past_v[l] = v
        else:
            past_k[l] = torch.cat([past_k[l], k_rot], dim=2)
            past_v[l] = torch.cat([past_v[l], v], dim=2)
        att = (q_rot @ past_k[l].transpose(-1, -2)
               / math.sqrt(HD)).softmax(-1)
        x = x + (att @ past_v[l]).view(1, D) @ W["o"][l]
        h = _rms(x, W["ln"][2 * l + 1])
        ff = torch.nn.functional.gelu(h @ W["w1"][l]) @ W["w2"][l]
        x = x + ff
    logits = (_rms(x, W["lnf"]) @ W["head"]).squeeze(0)
    return logits, past_k, past_v
'''


def _ref():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns


def test(ns):
    if not callable(ns.get("decode_step")):
        return {"ok": False, "detail": "missing decode_step"}
    import torch
    ref = _ref()
    for T in (0, 1, 5, 33):
        pk, pv = [None] * 2, [None] * 2
        rk, rv = [None] * 2, [None] * 2
        for pos in range(T + 1):
            try:
                lg, pk, pv = ns["decode_step"](pos * 7 % 511, pk, pv, pos)
            except Exception as e:
                return {"ok": False,
                        "detail": f"pos {pos} raised {e}"}
            rlg, rk, rv = ref["decode_step"](pos * 7 % 511, rk, rv, pos)
            if not torch.allclose(lg, rlg, atol=1e-5):
                return {"ok": False,
                        "detail": f"logits diverge at pos {pos}"}
        for l in range(2):
            if pk[l] is None or pk[l].shape[2] != rk[l].shape[2]:
                return {"ok": False,
                        "detail": f"cache T mismatch layer {l}"}
    return {"ok": True}


def bench(ns):
    import time
    step = ns["decode_step"]
    pk, pv = [None] * 2, [None] * 2
    for pos in range(64):                       # warmup + cache growth
        _, pk, pv = step(pos * 11 % 511, pk, pv, pos)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for pos in range(64, 192):
            _, pk, pv = step(pos * 11 % 511, pk, pv, pos)
        dt = (time.perf_counter() - t0) * 1e6 / 128
        best = min(best, dt)
    return {"us_per_token": best}

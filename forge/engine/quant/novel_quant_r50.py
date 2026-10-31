"""Novel parameter-format experiments (R&D Round 50, 2026-09-16).

Goal: replace the standard scalar/FP/minifloat weight parameter formats with
novel *storage formats* for Qwen2.5-0.5B, measuring quality (PPL/KL), memory
(bits/weight incl. metadata), and decode speed.

Novelty audit (web-checked 2026-09-16; see .devin/xparam/scratchpad notes):
  - PVQ for LLMs exists (arXiv 2410.16926)            -> not implemented
  - BiE dual-exponent BFP exists (ICML'24)            -> not implemented
  - PTQTP trit-planes = residual ternary              -> not implemented
  - DACQ CDF companding exists (arXiv 2603.00364)     -> ASH-Q kept as the
    closed-form arcsinh variant (no per-layer empirical CDF fitting)
  - PolarQuant-for-weights = Hadamard+LloydMax        -> NOT true 2-D polar;
    PPC-W here is genuinely different (radius+angle codec)
  - Entropy-coded indices exist (Huff-LLM/ECQ/ANS)    -> not implemented
  - Sigma-delta / error diffusion exists (ED arXiv 2410.11203, SDQ-LLM)
                                                    -> DPCM-W is a *predictive
    residual* codec (different mechanism, keeps causal decode chain)
  - eXmY / Q-Palette / LiftQuant do fractional bpw     -> MRQ is the plain
    scalar mixed-radix packing variant (undocumented for LLM AFAICT)

Implemented formats (encode/decode pairs, all tensor-level):
  esc_q   - in-band escape-code scalar quant: reserved code -> fp16 exceptions
  prcb    - per-row learned codebook (Lloyd-Max per row, fp16 centroids)
  ppc_w   - 2-D polar pair codec (log-radius + angle, shared block scale)
  geoq    - geometric-radix magnitude codebook (w = s * r^k, sweep r)
  ash_q   - arcsinh compander + uniform quant (closed-form, no calibration)
  mrq     - mixed-radix packed N-level scalar (5/7/9-level @ 2.33/2.8/3.2 bpw)
  dpcm_w  - leaky-predictor delta coding along input dim (correlation-gated)
  det_q   - per-block affine detrend + residual quant (correlation-gated)
  pairrot - 45-degree pair rotation + asymmetric bit split (correlation-gated)
  dctq    - row DCT-II subband bit allocation (correlation-gated)
  inr_w   - implicit neural representation: tiny coord-MLP + int-b residual
  cld_w   - cross-layer delta coding for same-suffix weight chains
  int_u   - baseline uniform symmetric int-b (reference)
  nf4     - baseline Gaussian-quantile codebook (reference)
  fp4     - baseline E2M1 minifloat (reference)

Every codec returns a payload dict; `payload_bytes` gives the exact serialized
size (bit-exact accounting) and `eff_bpw` the effective bits/weight.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _blocks(w: torch.Tensor, bs: int) -> torch.Tensor:
    """(rows, cols) -> (rows * n_blocks, bs); pads cols to multiple of bs."""
    r, c = w.shape
    nb = (c + bs - 1) // bs
    if c % bs:
        w = F.pad(w, (0, nb * bs - c))
    return w.reshape(r * nb, bs)


def _unblocks(b: torch.Tensor, rows: int, cols: int, bs: int) -> torch.Tensor:
    return b.reshape(rows, -1)[:, :cols]


def _mse_opt_scale(x: torch.Tensor, levels: torch.Tensor,
                   grid: torch.Tensor | None = None) -> torch.Tensor:
    """Per-block scale minimizing ||x - s*round_lvl(x/s)||^2 over a scale grid.

    x: (B, bs) block tensor. levels: sorted fp level set (abs values) for the
    codebook. grid: candidate scale multipliers of absmax/levels.max().
    Returns (B, 1) scales.
    """
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    base = absmax / levels.abs().max()
    if grid is None:
        grid = torch.linspace(0.5, 1.15, 14, device=x.device, dtype=x.dtype)
    best_s = base.clone()
    best_e = torch.full((x.shape[0], 1), float("inf"), device=x.device)
    lv = levels.to(x.device, x.dtype)
    for g in grid:
        s = base * g
        v = x / s
        # nearest level (signed symmetric set)
        d = (v.unsqueeze(-1) - lv).abs()
        q = lv[d.argmin(-1)]
        e = ((x - q * s) ** 2).sum(-1, keepdim=True)
        better = e < best_e
        best_e = torch.where(better, e, best_e)
        best_s = torch.where(better, s, best_s)
    return best_s


def _nearest_idx(v: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Indices of nearest level. v (...,), levels (L,) sorted."""
    d = (v.unsqueeze(-1) - levels).abs()
    return d.argmin(-1)


# ===========================================================================
# baseline codecs
# ===========================================================================

def enc_int_u(w: torch.Tensor, bits: int = 4, bs: int = 32) -> dict:
    """Uniform symmetric int-b, per-block absmax scale (reference baseline)."""
    x = _blocks(w.float(), bs)
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / (2 ** (bits - 1) - 1)
    q = torch.round(x / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1).to(torch.int8)
    return {"q": q, "s": s.half(), "bits": bits, "bs": bs}


def dec_int_u(p: dict, shape: tuple) -> torch.Tensor:
    w = p["q"].float() * p["s"].float()
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_int_u(p: dict, n: int) -> float:
    nb = p["q"].shape[0]
    return (n * p["bits"] + nb * 16) / n


def _nf4_levels() -> torch.Tensor:
    """NF4 codebook (Gaussian-quantile, from QLoRA paper)."""
    return torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0])


def enc_nf4(w: torch.Tensor, bs: int = 32) -> dict:
    lv = _nf4_levels().to(w.device)
    x = _blocks(w.float(), bs)
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = _nearest_idx(x / s, lv).to(torch.int8)
    return {"q": q, "s": s.half(), "bs": bs}


def dec_nf4(p: dict, shape: tuple) -> torch.Tensor:
    lv = _nf4_levels().to(p["q"].device)
    w = lv[p["q"].long()] * p["s"].float()
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_nf4(p: dict, n: int) -> float:
    return (n * 4 + p["q"].shape[0] * 16) / n


_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def enc_fp4(w: torch.Tensor, bs: int = 32) -> dict:
    """E2M1 minifloat, per-block scale (reference)."""
    lv = torch.cat([-_FP4.flip(0), _FP4]).to(w.device)
    x = _blocks(w.float(), bs)
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 6.0
    q = _nearest_idx(x / s, lv).to(torch.int8)
    return {"q": q, "s": s.half(), "bs": bs}


def dec_fp4(p: dict, shape: tuple) -> torch.Tensor:
    lv = torch.cat([-_FP4.flip(0), _FP4]).to(p["q"].device)
    w = lv[p["q"].long()] * p["s"].float()
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_fp4(p: dict, n: int) -> float:
    return (n * 4 + p["q"].shape[0] * 16) / n


# ===========================================================================
# F1. ESC-Q: escape-code scalar quant
# ===========================================================================

def enc_esc_q(w: torch.Tensor, bits: int = 4, bs: int = 32,
              esc_q: float = 0.995) -> dict:
    """In-band escape: code value -L is reserved; outlier weights go to a
    per-block fp16 exception list. The base grid therefore only needs to cover
    the bulk -> denser effective resolution at the same code width.

    Layout per block: b-bit code per element (range [-L, L-1], -L = escape),
    fp16 scale, and k fp16 exceptions (k variable). Encoding order: exceptions
    consumed left-to-right.
    """
    x = _blocks(w.float(), bs)
    L = 2 ** (bits - 1)
    # threshold: per-block quantile of |x|
    t = torch.quantile(x.abs(), esc_q, dim=-1, keepdim=True)
    s = (t / (L - 1)).clamp(min=1e-8)
    q = torch.round(x / s).clamp(-L, L - 1)
    esc_mask = (x.abs() / s) > (L - 0.5)  # values that would clip
    esc_mask |= q <= -L
    q = torch.where(esc_mask, torch.full_like(q, -L), q)
    # exceptions row-major per block
    exc = x[esc_mask].half()
    exc_cnt = esc_mask.sum(-1)  # (B,)
    return {"q": q.to(torch.int8), "s": s.half(), "exc": exc,
            "exc_cnt": exc_cnt.to(torch.int32), "bits": bits, "bs": bs}


def dec_esc_q(p: dict, shape: tuple) -> torch.Tensor:
    L = 2 ** (p["bits"] - 1)
    q = p["q"].float()
    w = q * p["s"].float()
    esc_mask = q <= -L
    exc = p["exc"].float()
    # scatter exceptions back in order
    idx = esc_mask.reshape(-1).nonzero().squeeze(-1)
    flat = w.reshape(-1)
    flat[idx] = exc
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_esc_q(p: dict, n: int) -> float:
    nb = p["q"].shape[0]
    return (n * p["bits"] + nb * 16 + p["exc"].numel() * 16
            + nb * 32) / n  # exc_cnt int32 per block


# ===========================================================================
# F2. PRCB: per-row learned codebook
# ===========================================================================

def _lloyd_max_1d(x: torch.Tensor, K: int, iters: int = 30,
                  row_chunk: int = 8192) -> torch.Tensor:
    """1-D Lloyd-Max on a set of rows simultaneously (batched, row-chunked
    to bound memory on large tensors).

    x: (R, N). Returns centroids (R, K) minimizing per-row MSE.
    """
    R = x.shape[0]
    outs = []
    for r0 in range(0, R, row_chunk):
        xs = x[r0:r0 + row_chunk]
        r = xs.shape[0]
        lo = torch.quantile(xs, 0.001, dim=1)
        hi = torch.quantile(xs, 0.999, dim=1)
        c = torch.linspace(0, 1, K, device=x.device).expand(r, K).clone()
        c = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * c
        for _ in range(iters):
            d = (xs.unsqueeze(-1) - c.unsqueeze(1)).abs()  # (r, N, K)
            a = d.argmin(-1)
            new = torch.zeros_like(c)
            cnt = torch.zeros_like(c)
            new.scatter_add_(1, a, xs)
            cnt.scatter_add_(1, a, torch.ones_like(xs))
            c_new = new / cnt.clamp(min=1.0)
            c_new = torch.where(cnt == 0, c, c_new)
            if torch.allclose(c_new, c):
                c = c_new
                break
            c = c_new
        outs.append(c.sort(dim=1).values)
    return torch.cat(outs, dim=0)


def enc_prcb(w: torch.Tensor, bits: int = 4, lloyd_iters: int = 30) -> dict:
    """Per-row K=2^b codebook + b-bit codes. Storage: b bpw + K*16/d_in."""
    K = 2 ** bits
    x = w.float()
    c = _lloyd_max_1d(x, K, lloyd_iters)
    qs = []
    for r0 in range(0, x.shape[0], 8192):
        xs = x[r0:r0 + 8192]
        cs = c[r0:r0 + 8192]
        d = (xs.unsqueeze(-1) - cs.unsqueeze(1)).abs()
        qs.append(d.argmin(-1).to(torch.int8))
    return {"q": torch.cat(qs, 0), "c": c.half(), "bits": bits}


def dec_prcb(p: dict, shape: tuple) -> torch.Tensor:
    c = p["c"].float()
    w = torch.gather(c, 1, p["q"].long())
    return w


def bits_prcb(p: dict, n: int) -> float:
    K = p["c"].shape[1]
    R = p["c"].shape[0]
    return (n * p["bits"] + R * K * 16) / n


# ===========================================================================
# F3. PPC-W: 2-D polar pair codec
# ===========================================================================

def enc_ppc_w(w: torch.Tensor, r_bits: int = 4, t_bits: int = 4,
              bs: int = 32) -> dict:
    """Pair adjacent input-dim elements -> (r, theta).
    r quantized in log domain over block range; theta uniform on [0, 2pi).
    Per-block fp16 scale normalizes radii (gain-shape split)."""
    x = _blocks(w.float(), bs)
    _B, _W = x.shape
    a = x[:, 0::2]; b = x[:, 1::2]
    r = torch.sqrt(a * a + b * b).clamp(min=1e-8)
    th = torch.atan2(b, a)  # [-pi, pi]
    # per-block radius scale = max radius
    rs = r.amax(dim=-1, keepdim=True).clamp(min=1e-8)
    rn = r / rs  # (0,1]
    # square-root compander on rn (Rayleigh radii concentrate near max)
    cr = rn.sqrt()
    rL = 2 ** r_bits
    rq = torch.round(cr * (rL - 1)).clamp(0, rL - 1).to(torch.int8)
    tL = 2 ** t_bits
    tq = torch.round((th / (2 * math.pi) + 0.5) * (tL - 1)).clamp(0, tL - 1).to(torch.int8)
    return {"rq": rq, "tq": tq, "rs": rs.half(), "bs": bs,
            "r_bits": r_bits, "t_bits": t_bits}


def dec_ppc_w(p: dict, shape: tuple) -> torch.Tensor:
    rL = 2 ** p["r_bits"]; tL = 2 ** p["t_bits"]
    rn = (p["rq"].float() / (rL - 1)) ** 2
    r = rn * p["rs"].float()
    th = (p["tq"].float() / (tL - 1) - 0.5) * 2 * math.pi
    a = r * torch.cos(th); b = r * torch.sin(th)
    B = a.shape[0]
    w = torch.stack([a, b], dim=-1).reshape(B, -1)
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_ppc_w(p: dict, n: int) -> float:
    nb = p["rq"].shape[0]
    per = p["r_bits"] + p["t_bits"]
    return (n / 2 * per + nb * 16) / n


# ===========================================================================
# F4. GeoQ: geometric-radix magnitude codebook
# ===========================================================================

def enc_geoq(w: torch.Tensor, bits: int = 4, r: float = math.sqrt(2),
             bs: int = 32, mse_scale: bool = True) -> dict:
    """Codebook {0} u {+-s*r^k}, k = 0..K-2, geometric ratio r.
    sign + magnitude index packed in one b-bit code."""
    K = 2 ** (bits - 1)  # magnitude slots incl. 0 (sign separate)
    mags = torch.tensor([r ** k for k in range(K - 1)], device=w.device)
    lv = torch.cat([torch.zeros(1, device=w.device), mags])
    x = _blocks(w.float(), bs)
    if mse_scale:
        s = _mse_opt_scale(x, lv)
    else:
        s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / mags.max()
    sign = (x < 0).to(torch.int8)
    mi = _nearest_idx(x.abs() / s, lv).to(torch.int8)
    # pack: code = sign*K + mi (0..2K-1)
    q = (sign * K + mi).to(torch.int8)
    return {"q": q, "s": s.half(), "r": r, "bits": bits, "bs": bs}


def dec_geoq(p: dict, shape: tuple) -> torch.Tensor:
    K = 2 ** (p["bits"] - 1)
    q = p["q"].long()
    sign = torch.where(q // K == 1, -1.0, 1.0)
    mi = q % K
    mags = torch.tensor([p["r"] ** k for k in range(K - 1)], device=q.device)
    lv = torch.cat([torch.zeros(1, device=q.device), mags])
    w = sign * lv[mi] * p["s"].float()
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_geoq(p: dict, n: int) -> float:
    nb = p["q"].shape[0]
    return (n * p["bits"] + nb * 16) / n


# ===========================================================================
# F5. ASH-Q: arcsinh compander
# ===========================================================================

def enc_ash_q(w: torch.Tensor, bits: int = 4, bs: int = 32,
              alpha_mode: str = "rms") -> dict:
    """v = asinh(w/alpha) -> uniform int-b over [vmin,vmax] -> w = a*sinh(vhat).
    alpha = per-block rms (asinh(x/rms) ~ linear near 0, log in tails).
    Closed-form compander — no per-layer empirical CDF (differs from DACQ)."""
    x = _blocks(w.float(), bs)
    if alpha_mode == "rms":
        a = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)
    else:
        a = x.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    v = torch.asinh(x / a)
    vmax = v.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    L = 2 ** (bits - 1) - 1
    q = torch.round(v / vmax * L).clamp(-L - 1, L).to(torch.int8)
    return {"q": q, "a": a.half(), "vmax": vmax.half(), "bits": bits, "bs": bs}


def dec_ash_q(p: dict, shape: tuple) -> torch.Tensor:
    L = 2 ** (p["bits"] - 1) - 1
    v = p["q"].float() / L * p["vmax"].float()
    w = p["a"].float() * torch.sinh(v)
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_ash_q(p: dict, n: int) -> float:
    nb = p["q"].shape[0]
    return (n * p["bits"] + nb * 32) / n  # a + vmax fp16


# ===========================================================================
# F6. MRQ: mixed-radix packed N-level
# ===========================================================================

_MR_PACK = {5: (3, 7), 6: (5, 13), 7: (5, 14), 9: (5, 16), 10: (6, 20),
            11: (4, 14), 13: (7, 26), 16: (1, 4)}
# N -> (group size, container bits); group packed into a single integer


def enc_mrq(w: torch.Tensor, n_levels: int = 5, bs: int = 32,
            lloyd: bool = True) -> dict:
    """Uniform/Lloyd N-level symmetric quant; groups of g elements packed into
    one integer in base N (exact mixed-radix packing — proves the rate).
    lloyd=True: per-ROW Lloyd-Max codebook (N fp16 per row, ~0 overhead)."""
    x = w.float()
    if lloyd:
        c = _lloyd_max_1d(x, n_levels, iters=25)
        qs = []
        for r0 in range(0, x.shape[0], 8192):
            xs = x[r0:r0 + 8192]
            cs = c[r0:r0 + 8192]
            d = (xs.unsqueeze(-1) - cs.unsqueeze(1)).abs()
            qs.append(d.argmin(-1).to(torch.int8))
        return {"q": torch.cat(qs, 0), "c": c.half(), "n": n_levels,
                "bs": bs, "lloyd": True}
    xb = _blocks(x, bs)
    L = (n_levels - 1) / 2
    s = xb.abs().amax(-1, keepdim=True).clamp(min=1e-8) / L
    q = (torch.round(xb / s + L)).clamp(0, n_levels - 1).to(torch.int8)
    return {"q": q, "s": s.half(), "n": n_levels, "bs": bs, "lloyd": False}


def dec_mrq(p: dict, shape: tuple) -> torch.Tensor:
    q = p["q"].long()
    if p["lloyd"]:
        return torch.gather(p["c"].float(), 1, q)
    L = (p["n"] - 1) / 2
    w = (q.float() - L) * p["s"].float()
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_mrq(p: dict, n: int) -> float:
    N = p["n"]
    g, cbits = _MR_PACK[N]
    packed = math.ceil(n / g) * cbits
    if p["lloyd"]:
        meta = p["c"].shape[0] * N * 16  # per-row codebook
    else:
        meta = p["q"].shape[0] * 16      # per-block scale
    return (packed + meta) / n


# ===========================================================================
# F7. DPCM-W: leaky-predictor delta coding along input dim
# ===========================================================================

def enc_dpcm_w(w: torch.Tensor, bits: int = 4, bs: int = 32,
               rho_mode: str = "ls") -> dict:
    """Serial predictor w_j ~ rho*w_{j-1}; quantize residuals int-b.
    rho per row least-squares on the ORIGINAL weights (not decoded).
    Residual blocks share fp16 scale per bs residuals."""
    x = w.float()
    R, C = x.shape
    if rho_mode == "ls":
        num = (x[:, 1:] * x[:, :-1]).sum(1)
        den = (x[:, :-1] ** 2).sum(1).clamp(min=1e-12)
        rho = (num / den).clamp(-0.999, 0.999)
    else:
        rho = torch.full((R,), float(rho_mode), device=x.device)
    # residuals wrt prediction on ORIGINAL (open-loop)
    res = torch.zeros_like(x)
    res[:, 0] = x[:, 0]
    res[:, 1:] = x[:, 1:] - rho.unsqueeze(1) * x[:, :-1]
    rb = _blocks(res, bs)
    L = 2 ** (bits - 1) - 1
    s = rb.abs().amax(-1, keepdim=True).clamp(min=1e-8) / L
    q = torch.round(rb / s).clamp(-L - 1, L).to(torch.int8)
    return {"q": q, "s": s.half(), "rho": rho.half(), "bits": bits, "bs": bs,
            "cols": C}


def dec_dpcm_w(p: dict, shape: tuple) -> torch.Tensor:
    rb = p["q"].float() * p["s"].float()
    res = _unblocks(rb, shape[0], shape[1], p["bs"])
    rho = p["rho"].float().unsqueeze(1)
    _R, C = res.shape
    w = torch.zeros_like(res)
    w[:, 0] = res[:, 0]
    for j in range(1, C):
        w[:, j] = res[:, j] + rho.squeeze(1) * w[:, j - 1]
    return w


def bits_dpcm_w(p: dict, n: int) -> float:
    nb = p["q"].shape[0]
    R = p["rho"].numel()
    return (n * p["bits"] + nb * 16 + R * 16) / n


# ===========================================================================
# F8. DET-Q: per-block affine detrend
# ===========================================================================

def enc_det_q(w: torch.Tensor, bits: int = 4, bs: int = 32) -> dict:
    """Per block: w_j ~ a + b*t_j (t_j in [0,1]); residual int-b.
    Storage: a,b fp16 per block + b-bit residual codes."""
    x = _blocks(w.float(), bs)
    _B, W = x.shape
    t = torch.linspace(0, 1, W, device=x.device).unsqueeze(0)
    tm = t.mean(); tt = t - tm
    b = ((x - x.mean(-1, keepdim=True)) * tt).sum(-1) / (tt ** 2).sum()
    a = x.mean(-1) - b * tm
    trend = a.unsqueeze(1) + b.unsqueeze(1) * t
    res = x - trend
    L = 2 ** (bits - 1) - 1
    s = res.abs().amax(-1, keepdim=True).clamp(min=1e-8) / L
    q = torch.round(res / s).clamp(-L - 1, L).to(torch.int8)
    return {"q": q, "s": s.half(), "a": a.half(), "b": b.half(),
            "bits": bits, "bs": bs}


def dec_det_q(p: dict, shape: tuple) -> torch.Tensor:
    res = p["q"].float() * p["s"].float()
    W = res.shape[1]
    t = torch.linspace(0, 1, W, device=res.device).unsqueeze(0)
    w = res + p["a"].float().unsqueeze(1) + p["b"].float().unsqueeze(1) * t
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_det_q(p: dict, n: int) -> float:
    nb = p["q"].shape[0]
    return (n * p["bits"] + nb * 48) / n  # a+b+s fp16


# ===========================================================================
# F9. PAIRROT: 45-degree pair rotation + asymmetric bits
# ===========================================================================

def enc_pairrot(w: torch.Tensor, bu: int = 5, bv: int = 3,
                bs: int = 32) -> dict:
    """u=(a+b)/sqrt2, v=(a-b)/sqrt2 per pair; u at bu bits, v at bv bits.
    Exploits adjacent-element correlation: if rho>0, Var(u)>>Var(v)."""
    x = _blocks(w.float(), bs)
    a = x[:, 0::2]; b = x[:, 1::2]
    u = (a + b) * (1 / math.sqrt(2)); v = (a - b) * (1 / math.sqrt(2))
    def qz(z, bits):
        L = 2 ** (bits - 1) - 1
        s = z.abs().amax(-1, keepdim=True).clamp(min=1e-8) / L
        q = torch.round(z / s).clamp(-L - 1, L).to(torch.int8)
        return q, s
    qu, su = qz(u, bu); qv, sv = qz(v, bv)
    return {"qu": qu, "su": su.half(), "qv": qv, "sv": sv.half(),
            "bu": bu, "bv": bv, "bs": bs}


def dec_pairrot(p: dict, shape: tuple) -> torch.Tensor:
    u = p["qu"].float() * p["su"].float()
    v = p["qv"].float() * p["sv"].float()
    a = (u + v) * (1 / math.sqrt(2)); b = (u - v) * (1 / math.sqrt(2))
    w = torch.stack([a, b], dim=-1).reshape(a.shape[0], -1)
    return _unblocks(w, shape[0], shape[1], p["bs"])


def bits_pairrot(p: dict, n: int) -> float:
    nb = p["qu"].shape[0]
    return (n / 2 * (p["bu"] + p["bv"]) + nb * 32) / n


# ===========================================================================
# F10. DCTQ: row DCT-II + variance bit allocation
# ===========================================================================

def enc_dctq(w: torch.Tensor, bits_hi: int = 8, bits_lo: int = 2,
             frac: float = 0.25) -> dict:
    """Row -> orthonormal DCT-II -> first `frac` coeffs at bits_hi (per-coeff
    group scale), rest at bits_lo. Requires spectral compaction to win."""
    x = w.float()
    _R, C = x.shape
    k = torch.arange(C, device=x.device, dtype=torch.float32)
    i = torch.arange(C, device=x.device, dtype=torch.float32).unsqueeze(1)
    Cm = torch.cos(math.pi / C * (k.unsqueeze(0) + 0.5) * i)
    Cm[0] *= math.sqrt(1.0 / C); Cm[1:] *= math.sqrt(2.0 / C)
    X = x @ Cm.T
    m = max(1, int(C * frac))
    hi, lo = X[:, :m], X[:, m:]
    def qz(z, bits, g=64):
        zb = _blocks(z, g)
        L = 2 ** (bits - 1) - 1
        s = zb.abs().amax(-1, keepdim=True).clamp(min=1e-8) / L
        q = torch.round(zb / s).clamp(-L - 1, L).to(torch.int8)
        return q, s.half(), zb.shape[1]
    qh, sh, g1 = qz(hi, bits_hi)
    ql, sl, g2 = qz(lo, bits_lo)
    return {"qh": qh, "sh": sh, "ql": ql, "sl": sl, "m": m,
            "bh": bits_hi, "bl": bits_lo, "g1": g1, "g2": g2}


def dec_dctq(p: dict, shape: tuple) -> torch.Tensor:
    R, C = shape
    hi = _unblocks(p["qh"].float() * p["sh"].float(), R, p["m"], p["g1"])
    lo = _unblocks(p["ql"].float() * p["sl"].float(), R, C - p["m"], p["g2"])
    X = torch.cat([hi, lo], dim=1)
    k = torch.arange(C, device=X.device, dtype=torch.float32)
    i = torch.arange(C, device=X.device, dtype=torch.float32).unsqueeze(1)
    Cm = torch.cos(math.pi / C * (k.unsqueeze(0) + 0.5) * i)
    Cm[0] *= math.sqrt(1.0 / C); Cm[1:] *= math.sqrt(2.0 / C)
    return X @ Cm  # inverse = transpose for orthonormal DCT-II


def bits_dctq(p: dict, n: int) -> float:
    nh = p["qh"].numel(); nl = p["ql"].numel()
    nbh = p["qh"].shape[0]; nbl = p["ql"].shape[0]
    return (nh * p["bh"] + nl * p["bl"] + (nbh + nbl) * 16) / n


# ===========================================================================
# F11. INR-W: implicit neural representation + residual quant
# ===========================================================================

def enc_inr_w(w: torch.Tensor, hid: int = 32, steps: int = 300,
              res_bits: int = 4, res_bs: int = 32, lr: float = 3e-3,
              device: str = "cuda") -> dict:
    """Novel flagship: store a tiny coord-MLP f(i,j)~w_ij instead of weights.
    Weights = MLP params + int-b residual. Tests whether LLM weight matrices
    have learnable low-complexity structure (expect: mostly not -> dead end
    documented honestly)."""
    R, C = w.shape
    dev = device if torch.cuda.is_available() else "cpu"
    xi = torch.linspace(-1, 1, R, device=dev)
    yj = torch.linspace(-1, 1, C, device=dev)
    gi, gj = torch.meshgrid(xi, yj, indexing="ij")
    coords = torch.stack([gi, gj], -1).reshape(-1, 2)  # (R*C, 2)
    target = w.float().to(dev).reshape(-1)
    net = nn.Sequential(
        nn.Linear(2, hid), nn.SiLU(),
        nn.Linear(hid, hid), nn.SiLU(),
        nn.Linear(hid, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = net(coords).squeeze(-1)
        loss = ((out - target) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        approx = net(coords).squeeze(-1).reshape(R, C).cpu()
    res = w.float() - approx
    pr = enc_int_u(res, res_bits, res_bs)
    # serialize MLP params
    theta = torch.cat([p.detach().flatten() for p in net.parameters()]).half()
    return {"theta": theta, "hid": hid, "res": pr, "shape": (R, C)}


def dec_inr_w(p: dict, shape: tuple) -> torch.Tensor:
    R, C = p["shape"]
    hid = p["hid"]
    net = nn.Sequential(
        nn.Linear(2, hid), nn.SiLU(),
        nn.Linear(hid, hid), nn.SiLU(),
        nn.Linear(hid, 1))
    # rebuild params
    theta = p["theta"].float()
    ofs = 0
    for pm in net.parameters():
        k = pm.numel()
        pm.data = theta[ofs:ofs + k].reshape(pm.shape).clone()
        ofs += k
    xi = torch.linspace(-1, 1, R)
    yj = torch.linspace(-1, 1, C)
    gi, gj = torch.meshgrid(xi, yj, indexing="ij")
    coords = torch.stack([gi, gj], -1).reshape(-1, 2)
    with torch.no_grad():
        approx = net(coords).squeeze(-1).reshape(R, C)
    res = dec_int_u(p["res"], shape)
    return approx + res


def bits_inr_w(p: dict, n: int) -> float:
    theta_bits = p["theta"].numel() * 16
    return (theta_bits + bits_int_u(p["res"], n) * n) / n


# ===========================================================================
# F12. CLD-W: cross-layer delta (applied at model level, not per-layer)
# ===========================================================================

def enc_cld_w(prev_w: torch.Tensor, w: torch.Tensor, bits: int = 4,
              bs: int = 32) -> dict:
    """w stored as prev_w + int-b delta. Caller decides the chain.
    prev_w is the DECODED previous weight (encoder must use it to stay
    consistent with the decoder)."""
    d = w.float() - prev_w.float()
    pd = enc_int_u(d, bits, bs)
    return {"delta": pd}


def dec_cld_w(p: dict, prev_w: torch.Tensor, shape: tuple) -> torch.Tensor:
    return prev_w.float() + dec_int_u(p["delta"], shape)


# ===========================================================================
# registry + Linear wrapper + model surgery
# ===========================================================================

CODECS = {
    "int_u": (enc_int_u, dec_int_u, bits_int_u),
    "nf4": (enc_nf4, dec_nf4, bits_nf4),
    "fp4": (enc_fp4, dec_fp4, bits_fp4),
    "esc_q": (enc_esc_q, dec_esc_q, bits_esc_q),
    "prcb": (enc_prcb, dec_prcb, bits_prcb),
    "ppc_w": (enc_ppc_w, dec_ppc_w, bits_ppc_w),
    "geoq": (enc_geoq, dec_geoq, bits_geoq),
    "ash_q": (enc_ash_q, dec_ash_q, bits_ash_q),
    "mrq": (enc_mrq, dec_mrq, bits_mrq),
    "dpcm_w": (enc_dpcm_w, dec_dpcm_w, bits_dpcm_w),
    "det_q": (enc_det_q, dec_det_q, bits_det_q),
    "pairrot": (enc_pairrot, dec_pairrot, bits_pairrot),
    "dctq": (enc_dctq, dec_dctq, bits_dctq),
    "inr_w": (enc_inr_w, dec_inr_w, bits_inr_w),
}


def encode(codec: str, w: torch.Tensor, **kw) -> dict:
    return CODECS[codec][0](w, **kw)


def decode(codec: str, p: dict, shape: tuple) -> torch.Tensor:
    return CODECS[codec][1](p, shape)


def eff_bpw(codec: str, p: dict, n: int) -> float:
    return CODECS[codec][2](p, n)


class XParamLinear(QuantizedLinearMixin):
    """Linear layer storing a novel-format payload; dequantizes on forward.

    Measures realistic generation cost: decode() runs every forward pass
    (no caching), so speed numbers include the format's decode overhead.
    """

    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear, codec: str, **kw) -> XParamLinear:
        mod = cls(lin.in_features, lin.out_features, lin.bias is not None)
        w = lin.weight.detach().float()
        payload = encode(codec, w, **kw)
        mod._codec = codec
        mod._payload = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in payload.items()}
        mod._shape = tuple(lin.weight.shape)
        mod._n = lin.weight.numel()
        if lin.bias is not None:
            mod.bias.data = lin.bias.detach().clone()
        return mod

    def _dequantize_weight(self, dtype=torch.float32) -> torch.Tensor:
        dev = self._dev()
        pl = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
              for k, v in self._payload.items()}
        return decode(self._codec, pl, self._shape).to(dtype)

    def _dev(self):
        for v in self._payload.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    def store_payload_(self):
        """Move payload tensors into registered buffers (for device moves)."""
        for k, v in list(self._payload.items()):
            if isinstance(v, torch.Tensor):
                safe = k.replace(".", "_")
                self.register_buffer(f"xp_{safe}", v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype)
        return F.linear(x, w, self.bias)


_SKIP_SUBSTR = ("norm", "ln_")
_TARGET_SUFFIX = (".weight",)


def quantize_model_xparam(model: nn.Module, codec: str, verbose: bool = False,
                          include_embed: bool = False, min_elems: int = 4096,
                          **kw) -> int:
    """Replace every nn.Linear (and optionally embed) with XParamLinear.

    Skips lm_head when it shares storage with embed_tokens (tied) unless
    include_embed — we quantize the shared tensor once via the module that
    owns it."""
    n = 0
    seen_payload = {}  # data_ptr -> payload dict (shared tensors encode once)
    # embeddings first (so a tied lm_head can share the payload)
    if include_embed:
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Embedding) and mod.weight.numel() >= min_elems:
                xp = XParamEmbedding.from_embedding(mod, codec, **kw)
                seen_payload[mod.weight.data_ptr()] = xp._payload
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, xp)
                n += 1
                if verbose:
                    logger.info(f"  {name} (emb): {tuple(mod.weight.shape)} -> {codec}")
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < min_elems:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if mod.weight.data_ptr() in seen_payload:
            # tied weight (e.g. lm_head == embed_tokens): share payload
            xp = XParamLinear(mod.in_features, mod.out_features,
                              mod.bias is not None)
            xp._codec = codec
            xp._payload = seen_payload[mod.weight.data_ptr()]
            xp._shape = tuple(mod.weight.shape)
            xp._n = mod.weight.numel()
            if mod.bias is not None:
                xp.bias.data = mod.bias.detach().clone()
            setattr(parent, attr, xp)
            n += 1
            if verbose:
                logger.info(f"  {name}: tied -> shared {codec} payload")
            continue
        xp = XParamLinear.from_linear(mod, codec, **kw)
        seen_payload[mod.weight.data_ptr()] = xp._payload
        setattr(parent, attr, xp)
        n += 1
        if verbose:
            logger.info(f"  {name}: {tuple(mod.weight.shape)} -> {codec}")
    return n


class XParamEmbedding(QuantizedLinearMixin):
    def __init__(self, num_emb: int, dim: int):
        super().__init__()
        self.num_embeddings = num_emb
        self.embedding_dim = dim
        self.in_features = dim
        self.out_features = num_emb

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, codec: str, **kw):
        mod = cls(emb.num_embeddings, emb.embedding_dim)
        mod._codec = codec
        mod._payload = encode(codec, emb.weight.detach().float(), **kw)
        mod._shape = tuple(emb.weight.shape)
        return mod

    def _dequantize_weight(self, dtype=torch.float32):
        dev = next((v.device for v in self._payload.values()
                    if isinstance(v, torch.Tensor)), torch.device("cpu"))
        pl = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
              for k, v in self._payload.items()}
        return decode(self._codec, pl, self._shape).to(dtype)

    def forward(self, x):
        w = self._dequantize_weight()
        return F.embedding(x, w)


def estimate_xparam_memory(model: nn.Module) -> dict:
    total_bits = 0.0
    total_bytes = 0
    n_params = 0
    seen_payloads = set()
    for mod in model.modules():
        if isinstance(mod, (XParamLinear, XParamEmbedding)):
            n = mod._shape[0] * mod._shape[1]
            pid = id(mod._payload)
            if pid not in seen_payloads:
                seen_payloads.add(pid)
                bpw = eff_bpw(mod._codec, mod._payload, n)
                total_bits += bpw * n
                total_bytes += int(bpw * n / 8)
            n_params += n
    # unquantized remainder (biases, norms)
    rest = 0
    for p in model.parameters():
        rest += p.numel() * p.element_size()
    return {"quant_mb": total_bytes / 1024**2,
            "rest_mb": rest / 1024**2,
            "total_mb": (total_bytes + rest) / 1024**2,
            "avg_bpw": total_bits / max(n_params, 1),
            "n_quant_params": n_params}

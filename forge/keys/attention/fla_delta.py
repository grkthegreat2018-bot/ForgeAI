"""fla-org fused kernels for the delta-rule family (COMPETITIVE_AUDIT A4).

"Vendor, don't reinvent" — the same policy plan 1.9 applies to vLLM's
Mamba kernels. fla-org ships the chunked WY-representation kernels that
Qwen3-Next (GDN) and Kimi Linear (KDA) ship upstream; the naive fp32
Python ``_scan`` loops in ``kda_key``/``gdn2_key`` are kept as the
CPU/reference fallback.

Semantics (verified against fla's naive references):
  * KDA  -> ``fla.ops.kda.chunk_kda``: S_t = Diag(a)S + βk(v − kᵀ Diag(a)S)
    — identical to ``KDALayer._scan`` (probe on the decayed state).
  * GDN2 -> ``fla.ops.generalized_delta_rule.chunk_dplr_delta_rule``:
    our S_t = Diag(a)S + k(w⊙v)ᵀ − k(a⊙b⊙k)ᵀS maps onto the DPLR form
    S_t = Diag(γ)S + k vᵀ + β(αᵀS) with a_dplr = −(a⊙e⊙k), b_dplr = k,
    v_dplr = w⊙v. Exact, not an approximation.

WINDOWS/TRITON PIN: flash-linear-attention 0.6.0 requires
``triton.experimental.gluon`` which triton-windows 3.4.0.post21 does not
ship — ``import fla`` fails at top level. Pin ``fla==0.4.2`` +
``fla-core==0.4.2`` + ``einops`` (installed --no-deps to avoid the
transformers>=5 bump). Verified importable + runnable on the RTX 5070
(SM120) box.

Both wrappers return ``None`` when the fast path can't run (fla absent,
CPU tensors, or T<2 where the 1-step naive loop is already minimal).
"""
from __future__ import annotations

import torch

try:  # fla==0.4.2 — see module docstring for the pin rationale
    from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
    from fla.ops.kda import chunk_kda
    _HAS_FLA = True
except Exception:  # pragma: no cover - fla absent on CPU-only envs
    chunk_kda = None
    chunk_dplr_delta_rule = None
    _HAS_FLA = False


def has_fla() -> bool:
    return _HAS_FLA


def kda_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             alpha: torch.Tensor, beta: torch.Tensor,
             S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Chunked KDA via fla. Shapes: q,k (B,T,H,dk); v (B,T,H,dv);
    alpha (B,T,H,dk); beta (B,T,H); S (B,H,dk,dv)."""
    if not _HAS_FLA or not q.is_cuda or q.shape[1] < 2:
        return None
    return chunk_kda(q, k, v, g=torch.log(alpha), beta=beta,
                     scale=1.0, initial_state=S,
                     output_final_state=True)


def gdn2_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              alpha: torch.Tensor, egate: torch.Tensor,
              wgate: torch.Tensor,
              S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Chunked GDN2 via fla's DPLR delta rule (exact mapping — module
    docstring). Shapes: q,k,alpha,egate (B,T,H,dk); v,wgate (B,T,H,dv);
    S (B,H,dk,dv)."""
    if not _HAS_FLA or not q.is_cuda or q.shape[1] < 2:
        return None
    return chunk_dplr_delta_rule(
        q, k, wgate * v, -(alpha * egate * k), k,
        gk=torch.log(alpha), scale=1.0, initial_state=S,
        output_final_state=True)

"""KDA — Kimi Delta Attention key (R49-2, arXiv:2510.26692).

Gated DeltaNet-family linear attention with per-key-dim decay:

    S_t = Diag(a_t) S_{t-1} + b_t k_t (v_t - (Diag(a_t) S_{t-1})^T k_t)^T
    o_t = S_t^T q_t          (q, k L2-normalized)

where a_t in (0,1)^d_k is the per-channel decay, b_t the write strength.
State is fixed-size (H x d_k x d_v) — KDA layers hold NO KV cache, which is
why a 3:1 KDA:full-attention hybrid cuts KV memory ~75% at validated
(Kimi Linear / Qwen3-Next) quality.

LOSSLESS AT INIT (ForgeHybrid pattern): the KDA branch is attached as a
side path whose output is weighted by a scalar ``gate`` initialized to 0.
At gate=0 the block output is bit-exact vs. the baseline; training opens
the gate. GDN-style parameterization for the warm start:

  - A_log ~ log(uniform(0.01, 16))     (per-key-dim decay base)
  - dt_bias = softplus^{-1}(uniform(1e-3, 0.1))
  - depthwise causal conv kernel = 4 on q/k/v

Recurrence is evaluated in fp32 (state drift in low precision when
gate~1). On CUDA it dispatches to fla-org's chunked KDA Triton kernel
(fla_delta.kda_scan — same recurrence); the sequential Python loop is
the CPU/fallback reference (rd_round_49_plan.md Phase 2, landed via
fla vendor — AUDIT A4).

Usage:
    config.use_kda = True            # side-path on every block
    config.kda_n_heads = 0           # 0 = inherit config.n_heads
    config.kda_head_dim = 0          # 0 = inherit d_model // n_heads
    config.kda_beta_gt1 = False      # N3: allow beta in (0,2) — off by default
    config.kda_decay_floor = 0.0     # optional lower bound on a_t

Key conversion (port-first):
    key = KDAKey(d_model=2560, n_heads=20, head_dim=128)
    result = key.forward(block_attn_state)   # adds kda.* zero/identity-init
    back   = key.reverse(result.weights)     # strips kda.* (lossless iff gate==0)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys.misc.base import Key, KeyClass, KeyResult


def _inv_softplus(u: torch.Tensor) -> torch.Tensor:
    """softplus^{-1}(u) = log(exp(u) - 1), numerically stable for u in (0, ~1]."""
    return torch.log(torch.expm1(u))


class _CausalDWConv(nn.Module):
    """Depthwise causal conv1d with a rolling decode-state.

    Holds its own conv kernel; the caller manages the boundary state
    (``state`` attr, (B, C, kernel-1)) so prefix-cache snapshots can
    capture/restore it alongside the recurrent state.
    """

    def __init__(self, channels: int, kernel: int = 4,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels,
                              padding=0, bias=True)
        w = self.conv.weight
        nn.init.uniform_(w, -1.0 / (kernel ** 0.5), 1.0 / (kernel ** 0.5),
                         generator=generator)
        nn.init.zeros_(self.conv.bias)
        self.state: torch.Tensor | None = None  # (B, C, kernel-1)

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                prefix: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C) with SiLU, causal.

        ``prefix`` is a stored left-pad ((B, C, k-1)) for suffix prefills on
        a prefix-cache hit; consumed by the caller (one-shot).
        """
        _B, _T, _C = x.shape
        xt = x.transpose(1, 2)  # (B, C, T)
        # Input length is always T + kernel - 1 (left context), so the
        # unpadded conv emits exactly T causal outputs in every branch.
        if prefix is not None:
            xt = torch.cat([prefix.to(xt.device, xt.dtype), xt], dim=2)
        elif use_cache and self.state is not None:
            xt = torch.cat([self.state, xt], dim=2)
        else:
            xt = F.pad(xt, (self.kernel - 1, 0))
        y = self.conv(xt)
        if use_cache and self.kernel > 1:
            # Store the last kernel-1 *inputs* (pre-conv) as decode context.
            tail = xt[:, :, -(self.kernel - 1):]
            self.state = tail.clone() if tail.shape[2] == self.kernel - 1 \
                else F.pad(tail, (self.kernel - 1 - tail.shape[2], 0)).clone()
        return F.silu(y.transpose(1, 2))


class KDALayer(nn.Module):
    """Gated DeltaNet / Kimi Delta Attention side-path layer.

    Additive branch: ``out = gate * out_proj(o * sigmoid(g(x)))`` with
    ``gate`` scalar zero-init -> contributes exactly nothing at warm start.

    Recurrent state (``self._state``, (B, H, d_k, d_v), fp32) and conv
    boundary states persist across decode steps when ``use_cache=True``.
    Reset on a new sequence via the ``_state_reset`` flag (set by the
    model's forward, same convention as ``_conv_state_reset``).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int | None = None,
                 d_k: int | None = None, d_v: int | None = None,
                 conv_kernel: int = 4, beta_gt1: bool = False,
                 decay_floor: float = 0.0, layer_idx: int = 0,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k or head_dim or (d_model // n_heads)
        self.d_v = d_v or self.d_k
        self.beta_gt1 = beta_gt1
        self.decay_floor = decay_floor
        self.layer_idx = layer_idx

        qk_dim = n_heads * self.d_k
        v_dim = n_heads * self.d_v

        self.q_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.v_proj = nn.Linear(d_model, v_dim, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)  # decay activation
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)  # write strength
        self.g_proj = nn.Linear(d_model, v_dim, bias=False)    # output gate
        self.out_proj = nn.Linear(v_dim, d_model, bias=False)

        # GDN-style decay params: A_log (per key-dim), dt_bias (per head).
        a_base = torch.empty(n_heads, self.d_k).uniform_(0.01, 16.0,
                                                         generator=generator)
        self.A_log = nn.Parameter(torch.log(a_base))
        dt_lo = torch.empty(n_heads).uniform_(1e-3, 0.1, generator=generator)
        self.dt_bias = nn.Parameter(_inv_softplus(dt_lo))

        self.conv_q = _CausalDWConv(qk_dim, conv_kernel, generator)
        self.conv_k = _CausalDWConv(qk_dim, conv_kernel, generator)
        self.conv_v = _CausalDWConv(v_dim, conv_kernel, generator)

        # Lossless knob: scalar branch gate (0 at init).
        self.gate = nn.Parameter(torch.zeros(1))

        self._state: torch.Tensor | None = None   # (B, H, d_k, d_v) fp32
        self._prefix: dict | None = None          # one-shot restore context

    # ── state management ────────────────────────────────────────────────────

    def reset_state(self) -> None:
        self._state = None
        self._prefix = None
        self.conv_q.state = self.conv_k.state = self.conv_v.state = None

    def snapshot_state(self) -> dict:
        """Capture live recurrent + conv state (for prefix/session caches)."""
        return {
            "state": self._state.clone() if self._state is not None else None,
            "conv_q": self.conv_q.state.clone()
                      if self.conv_q.state is not None else None,
            "conv_k": self.conv_k.state.clone()
                      if self.conv_k.state is not None else None,
            "conv_v": self.conv_v.state.clone()
                      if self.conv_v.state is not None else None,
        }

    def set_state_prefix(self, snap: dict) -> None:
        """Install a stored boundary state, consumed by the next forward."""
        self._prefix = snap

    # ── recurrence ──────────────────────────────────────────────────────────

    def _scan(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              alpha: torch.Tensor, beta: torch.Tensor,
              S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential KDA recurrence in fp32.

        q,k: (B,T,H,d_k) normalized; v: (B,T,H,d_v);
        alpha: (B,T,H,d_k); beta: (B,T,H); S: (B,H,d_k,d_v).
        Returns (o (B,T,H,d_v), S_final).

        Uses fla's chunked Triton kernel on CUDA when available
        (fla_delta.kda_scan — same recurrence); the Python loop below is
        the CPU/fallback reference (T==1 decode steps stay here too).
        """
        from forge.keys.attention.fla_delta import kda_scan
        fast = kda_scan(q, k, v, alpha, beta, S)
        if fast is not None:
            return fast
        T = q.shape[1]
        outs = []
        for t in range(T):
            Sa = S * alpha[:, t].unsqueeze(-1)                    # Diag(a) S
            pred = torch.einsum("bhdv,bhd->bhv", Sa, k[:, t])
            S = Sa + beta[:, t, :, None, None] * \
                k[:, t].unsqueeze(-1) * (v[:, t] - pred).unsqueeze(-2)
            outs.append(torch.einsum("bhdv,bhd->bhv", S, q[:, t]))
        return torch.stack(outs, dim=1), S

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                use_cache: bool = False) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model) — gated KDA branch output."""
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.d_k, self.d_v

        if getattr(self, "_state_reset", False):
            self.reset_state()
            self._state_reset = False
        prefix = self._prefix if (use_cache and T > 1) else None
        self._prefix = None  # one-shot

        q = self.conv_q(self.q_proj(x), use_cache=use_cache,
                        prefix=None if prefix is None else prefix.get("conv_q"))
        k = self.conv_k(self.k_proj(x), use_cache=use_cache,
                        prefix=None if prefix is None else prefix.get("conv_k"))
        v = self.conv_v(self.v_proj(x), use_cache=use_cache,
                        prefix=None if prefix is None else prefix.get("conv_v"))

        q = q.view(B, T, H, dk).float()
        k = k.view(B, T, H, dk).float()
        v = v.view(B, T, H, dv).float()
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Per-key-dim decay: a_t[h,c] = exp(-exp(A_log[h,c]) * softplus(a_t+dt_bias))
        a_act = self.a_proj(x).float()                            # (B,T,H)
        log_alpha = -self.A_log.exp() * \
            F.softplus(a_act + self.dt_bias).unsqueeze(-1)        # (B,T,H,dk)
        alpha = log_alpha.exp()
        if self.decay_floor > 0.0:
            alpha = alpha.clamp(min=self.decay_floor)

        beta = torch.sigmoid(self.b_proj(x).float())              # (B,T,H)
        if self.beta_gt1:
            beta = beta * 2.0

        if use_cache and self._state is not None and prefix is None:
            S = self._state
        elif prefix is not None and prefix.get("state") is not None:
            S = prefix["state"].to(x.device, torch.float32)
        else:
            S = torch.zeros(B, H, dk, dv, dtype=torch.float32, device=x.device)

        o, S = self._scan(q, k, v, alpha, beta, S)
        if use_cache:
            self._state = S

        o = o.reshape(B, T, H * dv).to(x.dtype)
        o = o * torch.sigmoid(self.g_proj(x))
        return self.gate * self.out_proj(o)


# ═══════════════════════════════════════════════════════════════════════════════
# KDAKey — lossless side-path conversion (KeyClass.BI)
# ═══════════════════════════════════════════════════════════════════════════════

# Parameter suffixes owned by the KDA branch inside a block.
KDA_PARAM_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "a_proj.weight", "b_proj.weight", "g_proj.weight", "out_proj.weight",
    "conv_q.conv.weight", "conv_q.conv.bias",
    "conv_k.conv.weight", "conv_k.conv.bias",
    "conv_v.conv.weight", "conv_v.conv.bias",
    "A_log", "dt_bias", "gate",
)


class KDAKey(Key):
    """Lossless port adding a KDA side-path to a layer's weight dict.

    forward(data): passes all existing tensors through unchanged and adds
    ``kda.<param>`` entries matching :class:`KDALayer`'s state dict,
    generated deterministically (seeded init, ``gate=0`` -> the ported
    checkpoint is bit-exact vs. baseline).

    reverse(weights): strips every ``kda.`` key. Lossless iff the stripped
    gates are all ~0 (still at warm start); reported in metadata.
    """

    def __init__(self, d_model: int, n_heads: int,
                 head_dim: int | None = None, d_k: int | None = None,
                 d_v: int | None = None, conv_kernel: int = 4,
                 beta_gt1: bool = False, decay_floor: float = 0.0,
                 seed: int = 0):
        self._d_model = d_model
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._d_k = d_k
        self._d_v = d_v
        self._conv_kernel = conv_kernel
        self._beta_gt1 = beta_gt1
        self._decay_floor = decay_floor
        self._seed = seed

    @property
    def name(self) -> str:
        return "kda"

    @property
    def description(self) -> str:
        return ("Adds a gated Kimi Delta Attention (Gated DeltaNet) "
                "side-path with gate=0 — bit-exact warm start. Reverse "
                "strips kda.* params (lossless while gate~0).")

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def _layer_state(self) -> dict[str, torch.Tensor]:
        """Deterministically-initialized KDALayer state dict (CPU)."""
        g = torch.Generator(device="cpu").manual_seed(self._seed)
        layer = KDALayer(self._d_model, self._n_heads,
                         head_dim=self._head_dim, d_k=self._d_k, d_v=self._d_v,
                         conv_kernel=self._conv_kernel,
                         beta_gt1=self._beta_gt1,
                         decay_floor=self._decay_floor, generator=g)
        return {f"kda.{k}": v.detach().clone()
                for k, v in layer.state_dict().items()}

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Layer weights -> same weights + deterministic zero-init kda.*."""
        if not data:
            return KeyResult(success=False, error="Empty input weight dict")
        if any(k.startswith("kda.") for k in data):
            return KeyResult(success=False,
                             error="kda.* already present — would double-port")
        result = {k: v.clone() for k, v in data.items()}
        result.update(self._layer_state())
        return KeyResult(
            success=True, weights=result,
            metadata={"conversion": "base->kda_sidepath",
                      "gate": 0.0, "lossless": True,
                      "n_heads": self._n_heads, "d_k": self._d_k,
                      "seed": self._seed})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Strip kda.* params; lossless iff all stripped gates are ~0."""
        kda_keys = [k for k in weights if k.startswith("kda.")]
        if not kda_keys:
            return KeyResult(success=False, error="No kda.* weights present")
        max_gate = max(
            weights[k].abs().max().item() for k in kda_keys
            if k.endswith("gate"))
        result = {k: v.clone() for k, v in weights.items()
                  if not k.startswith("kda.")}
        return KeyResult(
            success=True, data=result,
            metadata={"conversion": "kda_sidepath->base",
                      "dropped": len(kda_keys),
                      "max_gate": max_gate,
                      "lossless": max_gate < 1e-6})

    # ── whole-model convenience ──────────────────────────────────────────────

    def convert_model_state(self, state: dict[str, torch.Tensor],
                            block_prefix: str = "blocks",
                            indices: list[int] | None = None
                            ) -> KeyResult:
        """Apply the side-path port to ``{block_prefix}.{i}._kda.`` for every
        block index (or the given subset). Whole-checkpoint convenience
        wrapper over :meth:`forward`."""
        if indices is None:
            import re
            seen = set()
            for k in state:
                m = re.match(rf"{block_prefix}\.(\d+)\.", k)
                if m:
                    seen.add(int(m.group(1)))
            indices = sorted(seen)
        if not indices:
            return KeyResult(success=False,
                             error=f"No {block_prefix}.<i>.* keys found")
        layer_sd = {k.split("kda.", 1)[1]: v
                    for k, v in self._layer_state().items()}
        result = {k: v.clone() for k, v in state.items()}
        for i in indices:
            for suffix, w in layer_sd.items():
                key = f"{block_prefix}.{i}._kda.{suffix}"
                if key in result:
                    return KeyResult(
                        success=False,
                        error=f"{key} already present — would double-port")
                result[key] = w.clone()
        return KeyResult(
            success=True, weights=result,
            metadata={"conversion": "model->kda_sidepath",
                      "blocks": indices, "lossless": True})

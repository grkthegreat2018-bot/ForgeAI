"""GDN2 — Gated DeltaNet-2 key (R60-1, arXiv:2605.22791, NVlabs).

Decoupled erase/write gated delta rule — a strict generalization of KDA:

    S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(a_t) S_{t-1} + k_t (w_t ⊙ v_t)^T
    o_t = S_t^T q_t          (q, k L2-normalized)

where a_t in (0,1)^d_k is the per-channel decay (as KDA), b_t in (0,1)^d_k
is the CHANNEL-WISE erase gate (which key-side coordinates of the decayed
state are read and removed) and w_t in (0,1)^d_v is the CHANNEL-WISE write
gate (which value-side coordinates of the new content are committed).
KDA/GDN tie erase and write to a single scalar beta_t; decoupling them is
the paper's win — strongest at multi-key retrieval.

COLLAPSE RULES (used for lossless ports):
  - b_t = w_t = beta_t * 1  ->  exactly KDA. A KDA-keyed checkpoint
    converts to GDN2 FUNCTION-EXACTLY (any gate value, not just gate=0):
    e_proj/w_proj rows are tiled copies of the scalar b_proj row, so
    sigmoid emits beta_h broadcast across channels.
  - b_t = w_t -> scalar a_t -> Gated DeltaNet.

LOSSLESS AT INIT (same ForgeHybrid/KDA pattern): the branch output is
weighted by scalar ``gate`` initialized to 0 — bit-exact vs. baseline.

Recurrence is fp32 sequential (same as KDALayer) — the paper's chunkwise
WY kernel is a follow-up; fla-org has upstream impls to crib from.

Usage:
    config.use_gdn2 = True           # side-path on every block
    config.gdn2_n_heads = 0          # 0 = inherit config.n_heads
    config.gdn2_head_dim = 0         # 0 = inherit d_model // n_heads
    config.gdn2_decay_floor = 0.0    # optional lower bound on a_t
    # Mutually exclusive with use_kda (GDN2 supersedes it — KDA is the
    # b=w collapsed special case; running both is redundant).

Key conversion (port-first):
    key = GDN2Key(d_model=2560, n_heads=20, head_dim=128)
    result = key.forward(block_state)    # adds gdn2.* (or upgrades kda.*)
    back   = key.reverse(result.weights) # strips gdn2.* (lossless iff gate~0)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys.attention.kda_key import _CausalDWConv, _inv_softplus
from forge.keys.misc.base import Key, KeyClass, KeyResult


def _tile_rows(w: torch.Tensor, repeats: int) -> torch.Tensor:
    """(H, D) -> (H*repeats, D) tiling each row contiguously.

    Row order [h0 x repeats, h1 x repeats, ...] matches a (B,T,H,r) view of
    a (B,T,H*r) projection output — channel c of head h reads row h*r+c.
    """
    return w.repeat_interleave(repeats, dim=0)


def kda_state_to_gdn2(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a KDALayer state dict to a GDN2Layer state dict.

    Function-exact for ANY trained weights: the scalar write gate
    ``b_proj`` (H, d_model) becomes the channel-wise erase ``e_proj``
    (H*d_k, d_model) and write ``w_proj`` (H*d_v, d_model) by tiling each
    head's row — sigmoid then emits beta_h on every channel, which is
    precisely KDA's coupled gate. Requires the state's head count to be
    recoverable from ``b_proj`` (H rows) — d_k/d_v follow from the conv
    channels (H*d_k, H*d_v).
    """
    if "b_proj.weight" not in sd:
        raise KeyError("kda state dict missing b_proj.weight")
    H = sd["b_proj.weight"].shape[0]
    dk = sd["conv_q.conv.weight"].shape[0] // H
    dv = sd["conv_v.conv.weight"].shape[0] // H
    out = {}
    for k, v in sd.items():
        if k == "b_proj.weight":
            continue
        out[k] = v.clone()
    out["e_proj.weight"] = _tile_rows(sd["b_proj.weight"], dk)
    out["w_proj.weight"] = _tile_rows(sd["b_proj.weight"], dv)
    return out


class GDN2Layer(nn.Module):
    """Gated DeltaNet-2 side-path layer (decoupled channel-wise gates).

    Additive branch: ``out = gate * out_proj(o * sigmoid(g(x)))`` with
    ``gate`` scalar zero-init -> contributes exactly nothing at warm start.

    Same state conventions as :class:`KDALayer`: recurrent ``_state``
    (B, H, d_k, d_v) fp32 + conv boundary states persist across decode
    steps under ``use_cache``; ``_state_reset`` flag triggers a new-
    sequence reset; ``set_state_prefix`` installs a prefix-cache restore.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int | None = None,
                 d_k: int | None = None, d_v: int | None = None,
                 conv_kernel: int = 4, decay_floor: float = 0.0,
                 layer_idx: int = 0,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k or head_dim or (d_model // n_heads)
        self.d_v = d_v or self.d_k
        self.decay_floor = decay_floor
        self.layer_idx = layer_idx

        qk_dim = n_heads * self.d_k
        v_dim = n_heads * self.d_v

        self.q_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.v_proj = nn.Linear(d_model, v_dim, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)   # decay activation
        self.e_proj = nn.Linear(d_model, qk_dim, bias=False)    # erase gate (per d_k)
        self.w_proj = nn.Linear(d_model, v_dim, bias=False)     # write gate (per d_v)
        self.g_proj = nn.Linear(d_model, v_dim, bias=False)     # output gate
        self.out_proj = nn.Linear(v_dim, d_model, bias=False)

        # KDA-style decay params: A_log (per key-dim), dt_bias (per head).
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

    def restore_state(self, snap: dict) -> None:
        """Direct state install — journal/revert counterpart to
        :meth:`snapshot_state`. Unlike ``set_state_prefix`` this takes
        effect immediately (no use_cache/T>1 consumption rule)."""
        st = snap.get("state")
        self._state = None if st is None else st.clone()
        for name in ("conv_q", "conv_k", "conv_v"):
            conv = getattr(self, name, None)
            val = snap.get(name)
            if conv is not None:
                conv.state = None if val is None else val.clone()
        self._prefix = None

    # ── recurrence ──────────────────────────────────────────────────────────

    def _scan(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              alpha: torch.Tensor, egate: torch.Tensor, wgate: torch.Tensor,
              S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Naive sequential GDN2 recurrence in fp32.

        q,k: (B,T,H,d_k) normalized; v: (B,T,H,d_v);
        alpha: (B,T,H,d_k); egate: (B,T,H,d_k); wgate: (B,T,H,d_v);
        S: (B,H,d_k,d_v). Returns (o (B,T,H,d_v), S_final).
        """
        T = q.shape[1]
        outs = []
        for t in range(T):
            Sa = S * alpha[:, t].unsqueeze(-1)                    # Diag(a) S
            # (b ⊙ k)^T (Diag(a) S): which decayed content this key erases.
            erase = torch.einsum("bhdv,bhd->bhv", Sa,
                                 egate[:, t] * k[:, t])
            # Write: channel-gated value minus the erased read.
            S = Sa + k[:, t].unsqueeze(-1) * \
                (wgate[:, t] * v[:, t] - erase).unsqueeze(-2)
            outs.append(torch.einsum("bhdv,bhd->bhv", S, q[:, t]))
        return torch.stack(outs, dim=1), S

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                use_cache: bool = False) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model) — gated GDN2 branch output."""
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

        # Per-key-dim decay (identical to KDA): a_t = exp(-exp(A_log) * sp(act))
        a_act = self.a_proj(x).float()                            # (B,T,H)
        log_alpha = -self.A_log.exp() * \
            F.softplus(a_act + self.dt_bias).unsqueeze(-1)        # (B,T,H,dk)
        alpha = log_alpha.exp()
        if self.decay_floor > 0.0:
            alpha = alpha.clamp(min=self.decay_floor)

        # Decoupled channel-wise gates.
        egate = torch.sigmoid(self.e_proj(x).float()).view(B, T, H, dk)
        wgate = torch.sigmoid(self.w_proj(x).float()).view(B, T, H, dv)

        if use_cache and self._state is not None and prefix is None:
            S = self._state
        elif prefix is not None and prefix.get("state") is not None:
            S = prefix["state"].to(x.device, torch.float32)
        else:
            S = torch.zeros(B, H, dk, dv, dtype=torch.float32, device=x.device)

        o, S = self._scan(q, k, v, alpha, egate, wgate, S)
        if use_cache:
            self._state = S

        o = o.reshape(B, T, H * dv).to(x.dtype)
        o = o * torch.sigmoid(self.g_proj(x))
        return self.gate * self.out_proj(o)


# ═══════════════════════════════════════════════════════════════════════════════
# GDN2Key — lossless side-path conversion (KeyClass.BI)
# ═══════════════════════════════════════════════════════════════════════════════

# Parameter suffixes owned by the GDN2 branch inside a block.
GDN2_PARAM_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "a_proj.weight", "e_proj.weight", "w_proj.weight",
    "g_proj.weight", "out_proj.weight",
    "conv_q.conv.weight", "conv_q.conv.bias",
    "conv_k.conv.weight", "conv_k.conv.bias",
    "conv_v.conv.weight", "conv_v.conv.bias",
    "A_log", "dt_bias", "gate",
)


class GDN2Key(Key):
    """Lossless port adding a GDN2 side-path to a layer's weight dict.

    forward(data): three cases —
      * ``kda.*`` present: upgrades it to ``gdn2.*`` via
        :func:`kda_state_to_gdn2` — FUNCTION-EXACT at any gate value
        (tiled scalar gate = channel-wise broadcast).
      * neither present: adds seeded zero-init ``gdn2.*`` (gate=0 ->
        bit-exact warm start).
      * ``gdn2.*`` present: fails (double-port guard).

    reverse(weights): strips every ``gdn2.`` key. Lossless iff the
    stripped gate is ~0; also reports ``kda_collapsible`` — whether e/w
    gate rows are per-head uniform, i.e. the branch is still exactly a
    KDA layer (reversible to kda.* without loss).
    """

    def __init__(self, d_model: int, n_heads: int,
                 head_dim: int | None = None, d_k: int | None = None,
                 d_v: int | None = None, conv_kernel: int = 4,
                 decay_floor: float = 0.0, seed: int = 0):
        self._d_model = d_model
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._d_k = d_k
        self._d_v = d_v
        self._conv_kernel = conv_kernel
        self._decay_floor = decay_floor
        self._seed = seed

    @property
    def name(self) -> str:
        return "gdn2"

    @property
    def description(self) -> str:
        return ("Adds a Gated DeltaNet-2 (decoupled erase/write) "
                "side-path with gate=0 — bit-exact warm start; upgrades "
                "existing kda.* function-exactly. Reverse strips gdn2.* "
                "(lossless while gate~0).")

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def _layer_state(self) -> dict[str, torch.Tensor]:
        """Deterministically-initialized GDN2Layer state dict (CPU)."""
        g = torch.Generator(device="cpu").manual_seed(self._seed)
        layer = GDN2Layer(self._d_model, self._n_heads,
                          head_dim=self._head_dim, d_k=self._d_k,
                          d_v=self._d_v, conv_kernel=self._conv_kernel,
                          decay_floor=self._decay_floor, generator=g)
        return {f"gdn2.{k}": v.detach().clone()
                for k, v in layer.state_dict().items()}

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Layer weights -> same weights + gdn2.* (fresh or kda-upgraded)."""
        if not data:
            return KeyResult(success=False, error="Empty input weight dict")
        if any(k.startswith("gdn2.") for k in data):
            return KeyResult(success=False,
                             error="gdn2.* already present — would double-port")
        result = {k: v.clone() for k, v in data.items()}
        kda_keys = [k for k in data if k.startswith("kda.")]
        if kda_keys:
            # Function-exact upgrade: KDA's coupled scalar gate becomes
            # tiled channel-wise erase/write gates.
            kda_sd = {k[len("kda."):]: data[k] for k in kda_keys}
            gdn2_sd = kda_state_to_gdn2(kda_sd)
            for k in kda_keys:
                del result[k]
            result.update({f"gdn2.{k}": v for k, v in gdn2_sd.items()})
            return KeyResult(
                success=True, weights=result,
                metadata={"conversion": "kda->gdn2",
                          "lossless": True,
                          "note": "function-exact (tiled scalar gates)"})
        result.update(self._layer_state())
        return KeyResult(
            success=True, weights=result,
            metadata={"conversion": "base->gdn2_sidepath",
                      "gate": 0.0, "lossless": True,
                      "n_heads": self._n_heads, "d_k": self._d_k,
                      "seed": self._seed})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Strip gdn2.* params; lossless iff the stripped gate is ~0."""
        gdn2_keys = [k for k in weights if k.startswith("gdn2.")]
        if not gdn2_keys:
            return KeyResult(success=False, error="No gdn2.* weights present")
        max_gate = max(
            weights[k].abs().max().item() for k in gdn2_keys
            if k.endswith("gate"))
        collapsible = self._kda_collapsible(weights)
        result = {k: v.clone() for k, v in weights.items()
                  if not k.startswith("gdn2.")}
        return KeyResult(
            success=True, data=result,
            metadata={"conversion": "gdn2_sidepath->base",
                      "dropped": len(gdn2_keys),
                      "max_gate": max_gate,
                      "kda_collapsible": collapsible,
                      "lossless": max_gate < 1e-6})

    @staticmethod
    def _kda_collapsible(weights: dict[str, torch.Tensor]) -> bool:
        """True iff e/w gate rows are constant within each head —
        i.e. the branch is exactly a KDA layer (reverse to kda.* safe)."""
        e = weights.get("gdn2.e_proj.weight")
        w = weights.get("gdn2.w_proj.weight")
        a = weights.get("gdn2.a_proj.weight")
        if e is None or w is None or a is None:
            return False
        H = a.shape[0]
        dk = e.shape[0] // H
        dv = w.shape[0] // H
        e_rows = e.view(H, dk, -1)
        w_rows = w.view(H, dv, -1)
        return bool((e_rows == e_rows[:, :1].expand_as(e_rows)).all()
                    and (w_rows == w_rows[:, :1].expand_as(w_rows)).all())

    # ── whole-model convenience ──────────────────────────────────────────────

    def convert_model_state(self, state: dict[str, torch.Tensor],
                            block_prefix: str = "blocks",
                            indices: list[int] | None = None
                            ) -> KeyResult:
        """Apply the side-path port to ``{block_prefix}.{i}._gdn2.`` for
        every block index (or the given subset). Whole-checkpoint
        convenience wrapper over :meth:`forward` — handles both fresh
        zero-init ports and in-place ``_kda.`` -> ``_gdn2.`` upgrades."""
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
        result = {k: v.clone() for k, v in state.items()}
        fresh_sd = None
        upgraded = 0
        for i in indices:
            dst = f"{block_prefix}.{i}._gdn2."
            src = f"{block_prefix}.{i}._kda."
            if any(k.startswith(dst) for k in result):
                return KeyResult(
                    success=False,
                    error=f"{dst}* already present — would double-port")
            kda_sd = {k[len(src):]: v for k, v in result.items()
                      if k.startswith(src)}
            if kda_sd:
                gdn2_sd = kda_state_to_gdn2(kda_sd)
                for k in [k for k in result if k.startswith(src)]:
                    del result[k]
                result.update({dst + k: v for k, v in gdn2_sd.items()})
                upgraded += 1
            else:
                if fresh_sd is None:
                    fresh_sd = {k.split("gdn2.", 1)[1]: v
                                for k, v in self._layer_state().items()}
                for suffix, w in fresh_sd.items():
                    result[dst + suffix] = w.clone()
        return KeyResult(
            success=True, weights=result,
            metadata={"conversion": "model->gdn2_sidepath",
                      "blocks": indices, "kda_upgraded": upgraded,
                      "lossless": True})

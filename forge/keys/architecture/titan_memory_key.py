"""TITAN Neural Memory — test-time training long-term memory (v2).

TITAN (Behrouz et al., ICLR 2025) augments short-range attention with a
trainable long-term memory module whose weights update *at test time* via
a "surprise" signal — context is compressed into memory weights instead
of an ever-growing KV tensor, keeping VRAM flat for long sequences.

v2 upgrade (LiveT R&D): the original module was a Hebbian auto-associator
(x outer x — the input doubled as key and value). Real Titans memory is an
associative recall store M : key -> value where the key/value/query
*formats are learned parameters* and only the content matrix is live:

    read:   o_t = o_proj( normalize(q_t) @ M )            gated, residual
    write:  err_t = normalize(k_t) @ M - v_t            (recall error)
    inner:  G = Σ_t s_t k_t ⊗ err_t / N                 (surprise-weighted)
            fmom ← μ·fmom + G ;  M ← γ·M − η·fmom       (forget+momentum)

s_t is a per-token surprise weight supplied by the caller (LiveSession
uses clipped normalized next-token CE); when omitted it falls back to
the memory's own normalized recall error.

State surfaces (journaled by LiveSession, all registered buffers):
    memory   (d, d)        rank=0 full fast weight
    u (d, r) / v (r, d)    rank>0 factorized fast weight
    *_mom                  momentum buffers matching the fast weights

Lossless-at-init conventions (unchanged): gate zero-init AND o_proj
zero-init — a ported checkpoint produces identical outputs until the
memory is opened (by bootstrap training or LiveSession.open_gates).
`memory`/`u`/`v` are buffers, not parameters: outer-loop optimizers never
touch live state; only the inner loop and snapshot/revert do.

Checkpoint compat: legacy ckpts carry `memory.{memory|u,v},gate` only.
The new projections are params with fresh init — missing keys fall back
to init values under strict=False loading; gate==0 keeps the module a
no-op, so the port remains function-lossless.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys.misc.base import Key, KeyClass, KeyResult


class TitanMemory(nn.Module):
    """Gated neural long-term memory with surprise-gated inner-loop updates.

    Args:
        d_model: hidden size.
        rank: 0 = full fast weight ``memory`` (d, d); >0 = factorized
            ``u`` (d, r) @ ``v`` (r, d).
        update_lr: inner-loop learning rate η.
        forget: per-update decay γ on the fast weight (0.99 ≈ ~100-chunk
            half-life-ish horizon; 1.0 disables forgetting).
        momentum: inner-loop momentum μ on the surprise gradient.
        surprise_cap: clip applied to per-token surprise weights.
    """

    def __init__(self, d_model: int = 2048, rank: int = 0,
                 update_lr: float = 0.5, forget: float = 0.99,
                 momentum: float = 0.9, surprise_cap: float = 4.0,
                 solver: str = "gd", ridge: float = 1.0,
                 cg_iters: int = 8):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.update_lr = update_lr
        self.forget = forget
        self.momentum = momentum
        self.surprise_cap = surprise_cap
        # Inner-loop solver for the chunkwise ridge objective
        #   min_M  Σ_t s_t‖k_t M − v_t‖² + λ‖M − γ·M_prev‖²
        # "gd"    = 1 momentum step (DeltaNet/LMS class)
        # "cg"    = cg_iters conjugate-gradient steps, warm-started at the
        #           decayed prior (MesaNet class — near-optimal per chunk)
        # "solve" = dense solve of the normal equations (exact; full rank only)
        self.solver = solver
        self.ridge = ridge
        self.cg_iters = cg_iters

        # Learned write/read format (outer-loop params; frozen live).
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            # Symmetric addressing at init: read query = write key.
            self.q_proj.weight.copy_(self.k_proj.weight)
            # Pass-through readout: memory stores residual-space content
            # (successor embeddings) — identity decode is correct at init
            # and bootstrap refines it. Still lossless: memory=0, gate=0.
            nn.init.eye_(self.o_proj.weight)

        # Fast-weight state — buffers (live surface, journaled).
        if rank > 0:
            self.register_buffer("u", torch.zeros(d_model, rank))
            self.register_buffer("v", torch.zeros(rank, d_model))
            self.register_buffer("u_mom", torch.zeros(d_model, rank))
            self.register_buffer("v_mom", torch.zeros(rank, d_model))
        else:
            self.register_buffer("memory", torch.zeros(d_model, d_model))
            self.register_buffer("memory_mom",
                                 torch.zeros(d_model, d_model))

        # Zero-init gate => lossless at start.
        self.gate = nn.Parameter(torch.zeros(1))

    # ── state management (LiveSession journal) ──────────────────────────────

    def snapshot_state(self) -> dict[str, torch.Tensor]:
        if self.rank > 0:
            return {"u": self.u.clone(), "v": self.v.clone(),
                    "u_mom": self.u_mom.clone(), "v_mom": self.v_mom.clone()}
        return {"memory": self.memory.clone(),
                "memory_mom": self.memory_mom.clone()}

    def restore_state(self, snap: dict[str, torch.Tensor]) -> None:
        for k, val in snap.items():
            buf = getattr(self, k, None)
            if buf is not None:
                buf.copy_(val.to(buf.device, buf.dtype))

    def reset_state(self) -> None:
        """Zero the fast weights (full reset — long-term memory wiped)."""
        for name in ("memory", "memory_mom", "u", "v", "u_mom", "v_mom"):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.zero_()

    # ── read path ───────────────────────────────────────────────────────────

    def _read(self, x: torch.Tensor) -> torch.Tensor:
        """o = o_proj( normalize(q(x)) @ M )."""
        q = F.normalize(self.q_proj(x).float(), dim=-1)
        if self.rank > 0:
            o = (q @ self.u.float()) @ self.v.float()
        else:
            o = q @ self.memory.float()
        return self.o_proj(o.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read memory and gate it. Lossless while gate == 0 or o_proj == 0."""
        return self.gate * self._read(x)

    # ── write path (inner loop — test-time training) ────────────────────────

    def update(self, x: torch.Tensor,
               surprise: torch.Tensor | None = None,
               target: torch.Tensor | None = None,
               proj_lr: float = 0.0, proj_clip: float = 0.0) -> dict:
        """One chunkwise inner-loop step on the fast weight.

        x: (B, T, d_model) hidden inputs at this layer (detached by caller
        or detached here). surprise: optional (B, T) or (N,) per-token
        write weights; default = normalized recall error. target: optional
        (N, d) content written under each key — LiveSession passes the
        *successor token's embedding* (context → next-token associative
        store); None falls back to v_proj(x) (classic Titans self-target).
        When proj_lr > 0 the k/v/q projections also take a clipped inner
        step (``live_projections`` mode — journaled by the caller).

        Returns stats {mean_err, mean_surprise, grad_norm}.
        """
        if self.update_lr == 0.0 and proj_lr == 0.0:
            return {"mean_err": 0.0, "mean_surprise": 0.0, "grad_norm": 0.0}
        xf = x.detach().float().reshape(-1, self.d_model)
        N = max(xf.shape[0], 1)

        with torch.no_grad():
            k = F.normalize(self.k_proj(xf), dim=-1)
            v = (target.detach().float().reshape(-1, self.d_model)
                 if target is not None else self.v_proj(xf))
            if self.rank > 0:
                pred = (k @ self.u.float()) @ self.v.float()
            else:
                pred = k @ self.memory.float()
            err = pred - v                                # (N, d)

        if surprise is None:
            s = err.norm(dim=-1)
            s = s / s.mean().clamp_min(1e-8)
        else:
            s = surprise.detach().float().reshape(-1).to(xf.device)
        s = s.clamp(min=0.0, max=self.surprise_cap)

        if self.solver != "gd" and self.rank == 0:
            gn = self._solve_step(k, v, s)
        else:
            grad = (k * s[:, None]).T @ err / N           # dL/dM
            gn = float(grad.norm())
            with torch.no_grad():
                if self.rank > 0:
                    # Factorized inner step: grads through M = u @ v.
                    gu = grad @ self.v.float().T
                    gv = self.u.float().T @ grad
                    self.u_mom.mul_(self.momentum).add_(gu)
                    self.v_mom.mul_(self.momentum).add_(gv)
                    self.u.mul_(self.forget).add_(self.u_mom,
                                                  alpha=-self.update_lr)
                    self.v.mul_(self.forget).add_(self.v_mom,
                                                  alpha=-self.update_lr)
                else:
                    self.memory_mom.mul_(self.momentum).add_(grad)
                    self.memory.mul_(self.forget).add_(self.memory_mom,
                                                       alpha=-self.update_lr)

        # Optional: bounded online drift of the format projections.
        # Off by default — the literature's warning is that these need
        # outer-loop shaping; enable via LiveConfig.live_projections.
        if proj_lr > 0.0:
            self._step_projections(xf, s, proj_lr, proj_clip)

        return {"mean_err": float(err.norm(dim=-1).mean()),
                "mean_surprise": float(s.mean()), "grad_norm": gn}

    def _solve_step(self, k: torch.Tensor, v: torch.Tensor,
                    s: torch.Tensor) -> float:
        """Ridge-regularized solve for the fast weight (MesaNet class):

            M* = argmin Σ_t s_t‖k_t M − v_t‖² + λ‖M − γ·M_prev‖²
               = (KᵀSK + λI)⁻¹ (KᵀSV + λγ M_prev)

        The λ-prior term IS the forgetting mechanism: old content decays
        toward the solve center γ·M_prev while new data pulls M to fit.
        "cg" warm-starts at γ·M_prev and runs cg_iters iterations
        (parallel, near-optimal); "solve" does the dense solve (exact).
        """
        with torch.no_grad():
            d = self.d_model
            lam = self.ridge
            prev = self.memory.float()
            center = self.forget * prev                       # γ·M_prev
            sk = k * s[:, None]
            if self.solver == "solve":
                A = sk.T @ k + lam * torch.eye(d, device=k.device)
                B = sk.T @ v + lam * center
                self.memory.copy_(torch.linalg.solve(A, B))
            else:  # "cg" — conjugate gradient on A X = B, warm start
                X = center.clone()
                R = sk.T @ (v - k @ center)                   # residual
                P = R.clone()
                rs = float((R * R).sum())
                for _ in range(max(int(self.cg_iters), 1)):
                    AP = sk.T @ (k @ P) + lam * P
                    alpha = rs / max(float((P * AP).sum()), 1e-12)
                    X.add_(P, alpha=alpha)
                    R.add_(AP, alpha=-alpha)
                    rs_new = float((R * R).sum())
                    if rs_new < 1e-12:
                        break
                    P.mul_(rs_new / rs).add_(R)
                    rs = rs_new
                self.memory.copy_(X)
            return float((self.memory - prev).norm())

    def _step_projections(self, xf: torch.Tensor, s: torch.Tensor,
                          lr: float, clip: float) -> None:
        """Clipped inner SGD on k/v/q projections (journaled upstream)."""
        params = [self.k_proj.weight, self.v_proj.weight,
                  self.q_proj.weight]
        with torch.enable_grad():
            for p in params:
                p.requires_grad_(True)
            k = F.normalize(self.k_proj(xf), dim=-1)
            v = self.v_proj(xf)
            with torch.no_grad():
                if self.rank > 0:
                    pred = (k @ self.u.float()) @ self.v.float()
                else:
                    pred = k @ self.memory.float()
            loss = (s[:, None] * (pred - v) ** 2).sum() / xf.shape[0]
            grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is None:
                    continue
                if clip > 0.0:
                    g = g.clamp(min=-clip, max=clip)
                p.add_(g, alpha=-lr)


class TitanMemoryKey(Key):
    """Adds/removes the TITAN memory parameters on a state dict.

    forward: inject memory module params into every block —
    seeded k/q/v projections, zero o_proj, zero fast weights, gate=0
    (function-lossless port: gate AND empty memory both give zero output).
    reverse: strip the memory keys (back to a plain transformer state).
    Key class: PARTIAL.
    """

    def __init__(self, d_model: int = 2048, rank: int = 0, seed: int = 0):
        self.d_model = d_model
        self.rank = rank
        self._seed = seed

    @property
    def name(self) -> str:
        return "titan_memory"

    @property
    def description(self) -> str:
        return f"TITAN neural memory v2 (d={self.d_model}, rank={self.rank})"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def _layer_state(self) -> dict[str, torch.Tensor]:
        """Deterministically-initialized TitanMemory state dict (CPU)."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed)
            layer = TitanMemory(self.d_model, rank=self.rank)
        return dict(layer.state_dict())

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if k.startswith("blocks.") and ".ln1.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)
            added = 0
            layer_sd = self._layer_state()
            for i in range(n_layers):
                base = f"blocks.{i}.memory"
                for k, w in layer_sd.items():
                    state[f"{base}.{k}"] = w.clone()
                    added += 1
            return KeyResult(success=True, weights=state,
                             metadata={"n_layers": n_layers,
                                       "keys_added": added})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(weights)
            removed = 0
            for k in list(state.keys()):
                if ".memory." in k:
                    del state[k]
                    removed += 1
            return KeyResult(success=True, weights=state,
                             metadata={"keys_removed": removed})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

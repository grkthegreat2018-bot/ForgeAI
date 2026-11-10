"""SIFT — Structured Importance-ranked Fine-Tuning.

A minimal-compute alternative to LoRA-SFT that trains *real weight slices*
instead of adapter parameters. Three mechanisms, each cutting cost:

1. **Structured unit selection (Gauss–Southwell style).** A single
   calibration forward+backward over a few batches yields the true gradient
   block-norms ``||dL/dW[unit]||_F`` for every candidate unit. Units are
   *functional* parameter groups, not random entries:

   - attention sub-head block u (granularity ``attn_gran``): the matching
     row-slice of the fused QKV projection (q- and gate-sections, plus k/v
     when ``include_kv``) **and** the matching column-slice of out_proj —
     i.e. every edge of that sub-head;
   - FFN neuron-block j (granularity ``ffn_gran``): rows ``j*g..j*g+g`` of
     w_gate **and** w_up (or both halves of a fused gate_up GEMM) **and**
     cols ``j*g..j*g+g`` of w_down — i.e. every edge of that neuron block.

   Only the top-``attn_units``/``ffn_units`` units per active block train.

2. **Backward-frontier truncation.** The frontier F is the shallowest block
   covering ``frontier_mass`` of total gradient mass. Blocks below F carry
   zero trainable tensors, so autograd builds no graph through them at all:
   backward cost drops ~proportionally to (N-F)/N and their activations are
   never saved. No detach hooks needed — requires_grad=False does it all.

3. **Slice updates that merge back.** Each selected slice is a zero-init
   ``sift_delta`` Parameter applied as ``y[...,sel] += x @ Δᵀ`` (row side)
   or ``y += x[...,sel] @ Δᵀ`` (col side) — a one-sided LoRA on real
   weights. ``merge_sift`` folds Δ into W (and mirrors fused QKV/gate_up
   rows back into the split checkpoint weights). Periodic re-selection
   merges, re-calibrates, and re-picks units — ReLoRA-style compounding
   coverage while each round's touched set stays small.

Optional **prefix cache**: hidden states at the frontier input are a pure
function of the frozen prefix weights + token ids, so they can be cached
per sequence row and re-injected (``_sift_inject_hidden`` /
``_sift_start_layer`` hooks in ConfigurableResearchLLM._forward_impl).
Epoch ≥2 skips the entire prefix forward; the fill pass *is* the prefix
compute, so epoch 1 pays no extra cost either.

Adapter artifact: ``save_sift_adapter`` writes only ``{sel, delta, side}``
per touched module — ~tens of MB instead of a full checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def _is_fused_qkv(mod: nn.Module) -> bool:
    return all(hasattr(mod, a) for a in ("q_dim", "k_dim", "v_dim")) and \
        isinstance(getattr(mod, "weight", None), nn.Parameter)


def _is_fused_gateup(mod: nn.Module) -> bool:
    return hasattr(mod, "hidden_dim") and hasattr(mod, "out_dim") and \
        isinstance(getattr(mod, "weight", None), nn.Parameter) and \
        mod.weight.shape[0] == 2 * mod.hidden_dim


def _is_plain_linear(mod: nn.Module) -> bool:
    return isinstance(mod, nn.Linear) or (
        hasattr(mod, "in_features") and hasattr(mod, "out_features")
        and isinstance(getattr(mod, "weight", None), nn.Parameter))


@dataclass
class UnitSpan:
    """One weight-slice member of a unit: ``dim`` of ``module`` rows/cols."""
    module: str          # dotted path relative to model
    side: str            # "row" (out-rows) | "col" (in-cols)
    start: int
    length: int


@dataclass
class Unit:
    kind: str            # "attn" | "ffn"
    members: list[UnitSpan]
    score: float = 0.0


@dataclass
class _Cands:
    # module path -> nn.Module (dense, grad-capable weight)
    mods: dict
    # block_idx -> {"attn": [Unit...], "ffn": [Unit...]}
    units: dict
    # module path -> span granularity used for that module's blocks
    gran: dict
    head_dim: int
    hidden: int


def _mod(model: nn.Module, path: str) -> nn.Module:
    m = model
    for p in path.split("."):
        m = getattr(m, p)
    return m


def discover_units(model: nn.Module, attn_gran: int, ffn_gran: int,
                   include_kv: bool = False) -> _Cands:
    """Walk ``model.blocks`` and build the per-block unit table.

    Duck-typed on attribute names so it works for fused and unfused
    layouts (and any block whose attn/ffn carries the canonical
    qkv_proj/out_proj/w_gate/w_up/w_down names). Blocks with conv/mamba/
    MoE content simply yield no units and zero mass — the frontier
    ignores them automatically.
    """
    cands: dict[str, nn.Module] = {}
    units: dict[int, dict[str, list[Unit]]] = {}
    gran_map: dict[str, int] = {}
    head_dim = int(getattr(model.config, "head_dim", 0) or 0)
    hidden = int(getattr(model.config, "intermediate_size", 0) or 0)
    n_heads = int(getattr(model.config, "n_heads", 0) or 0)
    n_kv = int(getattr(model.config, "n_kv_heads", 0) or 0)

    for bi, block in enumerate(model.blocks):
        bu = {"attn": [], "ffn": []}
        attn = getattr(block, "attn", None)
        ffn = getattr(block, "ffn", None)

        # ── attention units ──
        if attn is not None:
            qkv = getattr(attn, "qkv_proj", None)
            out_p = getattr(attn, "out_proj", None)
            hd = head_dim
            if _is_fused_qkv(qkv):
                q_dim, k_dim, v_dim = qkv.q_dim, qkv.k_dim, qkv.v_dim
                g_dim = int(getattr(qkv, "g_dim", 0) or 0)
                if not hd:
                    hd = q_dim // max(1, n_heads)
                qkv_path = f"blocks.{bi}.attn.qkv_proj"
                out_path = f"blocks.{bi}.attn.out_proj"
                cands[qkv_path] = qkv
                gran_map[qkv_path] = attn_gran
                if _is_plain_linear(out_p):
                    cands[out_path] = out_p
                    gran_map[out_path] = attn_gran
                n_units = q_dim // attn_gran if attn_gran > 0 else 0
                g_base = q_dim + k_dim + v_dim
                # map a q-section unit index -> (head, offset inside head)
                for u in range(n_units):
                    s = u * attn_gran
                    members = [UnitSpan(qkv_path, "row", s, attn_gran)]
                    if g_dim == q_dim:
                        # LFM3-style gate: gate rows mirror q rows 1:1
                        members.append(UnitSpan(qkv_path, "row",
                                                g_base + s, attn_gran))
                    elif g_dim and n_heads and g_dim == n_heads \
                            and (attn_gran == hd or (hd and hd % attn_gran == 0
                                                     and s % hd == 0)):
                        # Spark-style scalar per-head gate: one row per head.
                        # Owned by the head's first sub-unit when attn_gran
                        # subdivides a head.
                        members.append(UnitSpan(qkv_path, "row",
                                                g_base + s // max(1, hd or attn_gran), 1))
                    if out_path in cands:
                        members.append(UnitSpan(out_path, "col", s, attn_gran))
                    if include_kv and hd and hd % attn_gran == 0 and n_kv:
                        h = s // hd
                        kvh = min(h * n_kv // max(1, n_heads), n_kv - 1)
                        off = s - h * hd
                        members.append(UnitSpan(qkv_path, "row",
                                                q_dim + kvh * hd + off, attn_gran))
                        members.append(UnitSpan(qkv_path, "row",
                                                q_dim + k_dim + kvh * hd + off, attn_gran))
                    bu["attn"].append(Unit("attn", members))
            else:
                # Unfused path: q_proj rows (+ gate_proj rows) + out_proj cols.
                q_p = getattr(attn, "q_proj", None)
                g_p = getattr(attn, "gate_proj", None)
                if _is_plain_linear(q_p) and _is_plain_linear(out_p):
                    if not hd:
                        hd = q_p.out_features // max(1, n_heads)
                    q_path = f"blocks.{bi}.attn.q_proj"
                    g_path = f"blocks.{bi}.attn.gate_proj"
                    out_path = f"blocks.{bi}.attn.out_proj"
                    cands[q_path] = q_p
                    cands[out_path] = out_p
                    gran_map[q_path] = attn_gran
                    gran_map[out_path] = attn_gran
                    if _is_plain_linear(g_p) and g_p.out_features == q_p.out_features:
                        cands[g_path] = g_p
                        gran_map[g_path] = attn_gran
                    for u in range(q_p.out_features // attn_gran):
                        s = u * attn_gran
                        members = [UnitSpan(q_path, "row", s, attn_gran),
                                   UnitSpan(out_path, "col", s, attn_gran)]
                        if g_path in cands:
                            members.append(UnitSpan(g_path, "row", s, attn_gran))
                        bu["attn"].append(Unit("attn", members))

        # ── FFN neuron-block units ──
        if ffn is not None:
            gu = getattr(ffn, "gate_up_proj", None)
            wg = getattr(ffn, "w_gate", None)
            wu = getattr(ffn, "w_up", None)
            wd = getattr(ffn, "w_down", None)
            down_path = f"blocks.{bi}.ffn.w_down"
            down_ok = _is_plain_linear(wd)
            if down_ok:
                cands[down_path] = wd
                gran_map[down_path] = ffn_gran
            h_dim = hidden or (wd.in_features if down_ok else 0)
            if getattr(ffn, "_fused_gate_up", False) and _is_fused_gateup(gu):
                h_dim = gu.hidden_dim
                gu_path = f"blocks.{bi}.ffn.gate_up_proj"
                cands[gu_path] = gu
                gran_map[gu_path] = ffn_gran
                if down_ok:
                    for u in range(h_dim // ffn_gran):
                        s = u * ffn_gran
                        bu["ffn"].append(Unit("ffn", [
                            UnitSpan(gu_path, "row", s, ffn_gran),
                            UnitSpan(gu_path, "row", h_dim + s, ffn_gran),
                            UnitSpan(down_path, "col", s, ffn_gran)]))
            elif _is_plain_linear(wg) and _is_plain_linear(wu) and down_ok:
                wg_path = f"blocks.{bi}.ffn.w_gate"
                wu_path = f"blocks.{bi}.ffn.w_up"
                cands[wg_path] = wg
                cands[wu_path] = wu
                gran_map[wg_path] = ffn_gran
                gran_map[wu_path] = ffn_gran
                h_dim = h_dim or wg.out_features
                for u in range(h_dim // ffn_gran):
                    s = u * ffn_gran
                    bu["ffn"].append(Unit("ffn", [
                        UnitSpan(wg_path, "row", s, ffn_gran),
                        UnitSpan(wu_path, "row", s, ffn_gran),
                        UnitSpan(down_path, "col", s, ffn_gran)]))
        units[bi] = bu

    return _Cands(cands, units, gran_map, head_dim, hidden)


# ---------------------------------------------------------------------------
# Calibration — true block-gradient norms, no dW materialized twice
# ---------------------------------------------------------------------------


def _sq_norms(grad: torch.Tensor, side: str) -> torch.Tensor:
    """Per-row (side="row") or per-col (side="col") squared grad norms.

    Span Frobenius norms are then ``sqrt(sq[start:start+len].sum())`` —
    granularity-free, so arbitrary units (head blocks, 1-row scalar gates,
    neuron spans) all index the same vector."""
    return grad.pow(2).sum(dim=1 if side == "row" else 0)


def calibrate_importance(model: nn.Module, cands: _Cands,
                         calib_iter, loss_fn, n_batches: int,
                         score_mode: str = "fro") -> dict[int, float]:
    """Run ``n_batches`` of fwd+bwd with candidate weights un-frozen and
    accumulate per-unit gradient block-norms into ``Unit.score``.

    ``score_mode``:
    - ``"fro"`` (default): expected Frobenius block-norm ``Σ_b‖g_b‖_F`` —
      Gauss–Southwell block rule (Nutini'15).
    - ``"smean"``: norm of the *mean* gradient ``‖ḡ‖_F`` — gradients that
      cancel across batches score ~0 (noise-robust G-S).
    - ``"obs"``: variance-normalized saliency ``Σ_i ḡ_i²/(Var[g_i]+ε)`` —
      Optimal-Brain-Surgeon saliency used as a *selector*: the diagonal
      empirical Fisher `E[g²]` is a free Hessian-diag proxy, so each
      element's score ≈ expected Newton-step descent gain.

    Cost ≈ n_batches normal steps; no optimizer step is taken.
    Returns per-block total importance mass.
    """
    cand_paths = list(cands.mods)
    cand_weights = {id(_mod(model, p).weight) for p in cand_paths}
    # Only candidate weights may receive grads: leaving embed/head/norms
    # trainable during calibration would allocate ~1B+ of dead .grad
    # buffers (the 131k-vocab head alone wants a 512MB CE grad buffer)
    # and build autograd state we never read.
    prev_req: dict[int, bool] = {}
    for p in model.parameters():
        prev_req[id(p)] = p.requires_grad
        p.requires_grad_(id(p) in cand_weights)
    # Precompute per-unit member refs: (module, side, start, length)
    unit_refs: list[tuple[int, int, Unit, list]] = []
    for bi, units in cands.units.items():
        for k, u in enumerate(units["attn"] + units["ffn"]):
            refs = [(sp.module, sp.side, sp.start, sp.length)
                    for sp in u.members]
            unit_refs.append((bi, k, u, refs))
    # Element-wise accumulators for smean/obs modes (CPU, ~3-6GB total for
    # a 1.7B-class candidate set — fp16 for the sum, fp32 for the squares).
    gsum: dict[str, torch.Tensor] = {} if score_mode in ("smean", "obs") else None
    gssq: dict[str, torch.Tensor] = {} if score_mode == "obs" else None
    try:
        model.zero_grad(set_to_none=True)
        it = calib_iter()
        for _ in range(n_batches):
            try:
                ids, labels, mask = next(it)
            except StopIteration:
                it = calib_iter()
                ids, labels, mask = next(it)
            loss = loss_fn(ids, labels, mask)
            (loss / n_batches).backward()
            if gsum is not None:
                for p in cand_paths:
                    g = _mod(model, p).weight.grad
                    if g is None:
                        continue
                    if p not in gsum:
                        gsum[p] = torch.zeros_like(g, dtype=torch.float16,
                                                   device="cpu")
                        if gssq is not None:
                            gssq[p] = torch.zeros_like(g, dtype=torch.float32,
                                                       device="cpu")
                    gsum[p] += g.to(torch.float16).cpu()
                    if gssq is not None:
                        gssq[p] += g.float().pow(2).cpu()
            if score_mode == "fro":
                # One vectorized row/col sq-norm pass per module per side,
                # then slice by unit span.
                sq_cache: dict[tuple[str, str], torch.Tensor] = {}
                for _, _, u, refs in unit_refs:
                    s = 0.0
                    for path, side, start, length in refs:
                        key = (path, side)
                        if key not in sq_cache:
                            g = _mod(model, path).weight.grad
                            sq_cache[key] = (None if g is None
                                             else _sq_norms(g.float(), side).cpu())
                        sq = sq_cache[key]
                        if sq is not None and start + length <= sq.numel():
                            s += float(sq[start:start + length].sum().sqrt())
                    u.score += s
            model.zero_grad(set_to_none=True)
        if score_mode != "fro":
            B = float(n_batches)
            # Per-module elementwise score matrices, computed once.
            # smean: M = ḡ²  → span score = sqrt(Σ M[span]) = ‖ḡ_span‖_F
            # obs:   M = ḡ²/(Var+mean(Var)) → span score = Σ M[span]
            #        (additive OBS energy — expected descent under a
            #        diagonal-Newton step on that slice).
            sal_cache: dict[str, torch.Tensor] = {}
            for p in cand_paths:
                if p not in (gsum or {}):
                    continue
                m = gsum[p].float() / B
                if score_mode == "smean":
                    sal_cache[p] = m * m
                else:
                    var = gssq[p] / B - m * m
                    eps = float(var.mean().clamp(min=1e-12))
                    sal_cache[p] = m * m / var.clamp(min=0).add_(eps)
            for _, _, u, refs in unit_refs:
                s = 0.0
                for path, side, start, length in refs:
                    S = sal_cache.get(path)
                    if S is None:
                        continue
                    if side == "row":
                        v = float(S[start:start + length].sum())
                    else:
                        v = float(S[:, start:start + length].sum())
                    s += v ** 0.5 if score_mode == "smean" else v
                u.score = s
    finally:
        for p in model.parameters():
            if id(p) in prev_req:
                p.requires_grad_(prev_req[id(p)])
        model.zero_grad(set_to_none=True)

    block_mass: dict[int, float] = {}
    for bi, units in cands.units.items():
        block_mass[bi] = sum(u.score for u in units["attn"] + units["ffn"])
    return block_mass


def pick_frontier(block_mass: dict[int, float], mass_frac: float,
                  n_blocks: int) -> int:
    """Shallowest block F such that blocks ≥ F hold ``mass_frac`` of the
    total importance mass."""
    total = sum(block_mass.values())
    if total <= 0:
        return max(0, n_blocks // 2)
    # sort blocks deep->shallow, accumulate until coverage hit
    order = sorted(block_mass, key=lambda b: -b)
    acc = 0.0
    frontier = n_blocks - 1
    for bi in order:
        acc += block_mass[bi]
        frontier = min(frontier, bi)
        if acc >= mass_frac * total:
            break
    return frontier


# ---------------------------------------------------------------------------
# Slice attach / detach / merge
# ---------------------------------------------------------------------------


def _attach_slice(mod: nn.Module, side: str, sel: torch.Tensor) -> nn.Parameter:
    """Attach a zero-init slice delta to ``mod`` and wrap its forward."""
    W = mod.weight
    sel = sel.to(W.device)
    if side == "row":
        delta = torch.zeros(sel.numel(), W.shape[1], device=W.device, dtype=W.dtype)
    else:
        delta = torch.zeros(W.shape[0], sel.numel(), device=W.device, dtype=W.dtype)
    mod.sift_delta = nn.Parameter(delta)
    mod.register_buffer("sift_sel", sel)
    mod.register_buffer("sift_side", torch.tensor(
        [0 if side == "row" else 1], dtype=torch.uint8, device=W.device))
    mod._sift_side = side
    mod._sift_orig_forward = mod.forward

    if _is_fused_qkv(mod):
        dims = [mod.q_dim, mod.k_dim, mod.v_dim] + ([mod.g_dim] if mod.g_dim else [])

        def fused_fwd(x, _mod=mod, _dims=dims):
            out = F.linear(x, _mod.weight, _mod.bias)
            out = out.index_add(-1, _mod.sift_sel, x @ _mod.sift_delta.T)
            return out.split(_dims, dim=-1)
        mod.forward = fused_fwd
    elif _is_fused_gateup(mod):
        def gu_fwd(x, _mod=mod):
            out = F.linear(x, _mod.weight, _mod.bias)
            out = out.index_add(-1, _mod.sift_sel, x @ _mod.sift_delta.T)
            return out.split([_mod.hidden_dim, _mod.hidden_dim], dim=-1)
        mod.forward = gu_fwd
    elif side == "row":
        def row_fwd(x, _mod=mod, _orig=mod.forward):
            return _orig(x).index_add(-1, _mod.sift_sel, x @ _mod.sift_delta.T)
        mod.forward = row_fwd
    else:
        def col_fwd(x, _mod=mod, _orig=mod.forward):
            return _orig(x) + (x[..., _mod.sift_sel] @ _mod.sift_delta.T)
        mod.forward = col_fwd
    return mod.sift_delta


def _detach_slice(mod: nn.Module) -> None:
    """Restore the original forward and remove delta/sel (after merge)."""
    if getattr(mod, "_sift_orig_forward", None) is not None:
        mod.forward = mod._sift_orig_forward
        del mod._sift_orig_forward
    for a in ("sift_delta", "sift_sel", "sift_side"):
        if hasattr(mod, a):
            delattr(mod, a)
    if hasattr(mod, "_sift_side"):
        del mod._sift_side
    if hasattr(mod, "_sift_mirror"):
        del mod._sift_mirror


def _register_mirrors(model: nn.Module) -> None:
    """Record mirror-writers on fused modules so merge can update the split
    checkpoint weights (q/k/v/gate_proj, w_gate/w_up) that loaders re-pack
    the fused GEMMs from."""
    for bi, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if attn is not None:
            qkv = getattr(attn, "qkv_proj", None)
            if _is_fused_qkv(qkv):
                qd, kd, vd = qkv.q_dim, qkv.k_dim, qkv.v_dim
                gd = int(getattr(qkv, "g_dim", 0) or 0)

                def mirror_qkv(sel, delta, _a=attn, _qd=qd, _kd=kd, _vd=vd, _gd=gd):
                    parts = [("q_proj", 0, _qd), ("k_proj", _qd, _qd + _kd),
                             ("v_proj", _qd + _kd, _qd + _kd + _vd),
                             ("gate_proj", _qd + _kd + _vd, _qd + _kd + _vd + _gd)]
                    with torch.no_grad():
                        for name, lo, hi in parts:
                            lin = getattr(_a, name, None)
                            if lin is None or hi <= lo:
                                continue
                            m = (sel >= lo) & (sel < hi)
                            if not m.any():
                                continue
                            lin.weight[sel[m] - lo] += delta[m].to(lin.weight.dtype)
                qkv._sift_mirror = mirror_qkv
        ffn = getattr(block, "ffn", None)
        if ffn is not None:
            gu = getattr(ffn, "gate_up_proj", None)
            if _is_fused_gateup(gu):
                hd = gu.hidden_dim

                def mirror_gu(sel, delta, _f=ffn, _h=hd):
                    with torch.no_grad():
                        g = getattr(_f, "w_gate", None)
                        u = getattr(_f, "w_up", None)
                        mg = sel < _h
                        if g is not None and mg.any():
                            g.weight[sel[mg]] += delta[mg].to(g.weight.dtype)
                        mu = ~mg
                        if u is not None and mu.any():
                            u.weight[sel[mu] - _h] += delta[mu].to(u.weight.dtype)
                gu._sift_mirror = mirror_gu


def merge_sift(model: nn.Module, zero: bool = True) -> int:
    """Fold every ``sift_delta`` into its base weight slice (and the split
    mirror weights). ``zero=True`` keeps slices attached but resets them
    (continued training); use ``detach_sift`` to remove them entirely."""
    n = 0
    for mod in model.modules():
        if getattr(mod, "_sift_side", None) is None:
            continue
        with torch.no_grad():
            d = mod.sift_delta
            if mod._sift_side == "row":
                mod.weight[mod.sift_sel] += d.to(mod.weight.dtype)
            else:
                mod.weight[:, mod.sift_sel] += d.to(mod.weight.dtype)
            if getattr(mod, "_sift_mirror", None) is not None:
                mod._sift_mirror(mod.sift_sel, d)
            if zero:
                d.zero_()
        n += 1
    return n


def detach_sift(model: nn.Module, merge: bool = True) -> int:
    """Merge (optional) then remove all slice machinery — restores a plain
    model whose state_dict has no sift_* keys. Use before final save/export."""
    n = merge_sift(model, zero=False) if merge else 0
    for mod in model.modules():
        if getattr(mod, "_sift_side", None) is not None or \
                getattr(mod, "_sift_orig_forward", None) is not None:
            _detach_slice(mod)
    return n


def apply_selection(model: nn.Module, sel: dict[str, tuple[str, torch.Tensor]]) -> list[nn.Parameter]:
    """Attach deltas per ``{module_path: (side, sel_idx)}``. Returns params."""
    params = []
    for path, (side, idx) in sel.items():
        if idx.numel() == 0:
            continue
        params.append(_attach_slice(_mod(model, path), side, idx))
    return params


def build_selection(cands: _Cands, frontier: int,
                    attn_units: int, ffn_units: int) -> dict[str, tuple[str, torch.Tensor]]:
    """Top-k units per active block -> per-module sorted index sets."""
    sel: dict[str, list] = {}
    for bi, units in cands.units.items():
        if bi < frontier:
            continue
        for group, k in (("attn", attn_units), ("ffn", ffn_units)):
            us = sorted(units[group], key=lambda u: -u.score)[:k]
            for u in us:
                if u.score <= 0:
                    continue
                for sp in u.members:
                    sel.setdefault((sp.module, sp.side), set()).update(
                        range(sp.start, sp.start + sp.length))
    return {p: (s, torch.tensor(sorted(idxs), dtype=torch.long))
            for (p, s), idxs in sel.items()}


# ---------------------------------------------------------------------------
# Newton init — one-shot diagonal Gauss–Newton on the attached slices
# ---------------------------------------------------------------------------


def newton_init(model: nn.Module, calib_iter, loss_fn, n_passes: int,
                lr: float = 1.0, damp: float = 1e-2,
                trust: float = 0.25) -> dict:
    """Preconditioned warm-start for attached deltas.

    For slice delta Δ the local quadratic surrogate is
    ``L(Δ) ≈ L0 + gᵀΔ + ½ Σ_j h_j‖Δ[:,j]‖²`` with input-factor curvature
    ``h_j = E[x_j²]`` — the input-only half of K-FAC (output curvature is
    folded into ``lr``).  The exact minimizer of that surrogate is
    ``Δ* = -ḡ/(E[x²]+damp)``, applied here in one shot:

    - ``ḡ``: mean ``∂L/∂Δ`` — read directly off ``delta.grad`` (free: the
      delta is already the trainable leaf, no weight-grad buffers needed).
    - ``E[x²]``: per-input-channel second moment of the delta's input,
      captured by forward-pre-hooks (fires on grad-ckpt recompute too, so
      token counts are accumulated alongside and the ratio stays exact).
    - col-side slices index ``E[x²]`` by ``sift_sel`` (their delta input is
      ``x[..., sel]``).

    The raw diagonal step has no global scale guarantee on non-quadratic
    losses (CE output curvature ≠ 1), so the init is protected two ways:
    per-slice trust region ``‖Δ‖_F ≤ 0.25‖W_sel‖_F`` and an Armijo-style
    backtracking line search over the calibration batches — the accepted
    point never has higher calibration loss than the zero-init base, and
    if nothing helps the deltas stay zero (init is then a no-op).

    Memory: one E[x²] vector per sliced module (~KBs); ``delta.grad`` plus
    one direction copy = 2× the touched-parameter count.
    """
    mods = [m for m in model.modules()
            if getattr(m, "_sift_side", None) is not None]
    if not mods:
        return {"passes": 0}
    x2: dict[int, torch.Tensor] = {}
    ntok: dict[int, int] = {}
    hooks = []

    def _mk(m):
        def _pre(mod, args):
            x = args[0].detach()
            x2[id(m)] = x2.get(id(m), torch.zeros(
                x.shape[-1], dtype=torch.float32, device=x.device)) + \
                x.float().pow(2).sum(dim=tuple(range(x.dim() - 1)))
            ntok[id(m)] = ntok.get(id(m), 0) + x.numel() // x.shape[-1]
        return m.register_forward_pre_hook(_pre)

    for m in mods:
        hooks.append(_mk(m))
    dirs: dict[int, torch.Tensor] = {}
    l0 = 0.0
    try:
        model.zero_grad(set_to_none=True)
        it = calib_iter()
        for _ in range(n_passes):
            try:
                ids, labels, mask = next(it)
            except StopIteration:
                it = calib_iter()
                ids, labels, mask = next(it)
            loss = loss_fn(ids, labels, mask)
            l0 += float(loss.detach()) / n_passes
            (loss / n_passes).backward()
        with torch.no_grad():
            for m in mods:
                d = m.sift_delta
                if d.grad is None:
                    continue
                ex2 = x2[id(m)] / max(ntok[id(m)], 1)
                dmp = float(ex2.mean()) * damp + 1e-12
                if m._sift_side == "col":
                    ex2 = ex2[m.sift_sel]
                step = -lr * d.grad.float() / (ex2.float() + dmp)
                step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
                # trust region: never move a slice more than 25% of its base
                base = (m.weight[m.sift_sel] if m._sift_side == "row"
                        else m.weight[:, m.sift_sel])
                cap = trust * float(base.float().norm()) + 1e-12
                n = float(step.norm())
                if n > cap:
                    step *= cap / n
                dirs[id(m)] = step
        model.zero_grad(set_to_none=True)
    finally:
        for h in hooks:
            h.remove()
        model.zero_grad(set_to_none=True)

    if not dirs:
        return {"passes": n_passes, "accepted": False}

    # Armijo backtracking on the same calibration data: accept the largest
    # scale whose forward-only loss beats the zero-init baseline l0.
    def _eval() -> float:
        tot, it2 = 0.0, calib_iter()
        with torch.no_grad():
            for _ in range(n_passes):
                try:
                    ids, labels, mask = next(it2)
                except StopIteration:
                    it2 = calib_iter()
                    ids, labels, mask = next(it2)
                tot += float(loss_fn(ids, labels, mask))
        return tot / n_passes

    best_s, best_l = 0.0, l0
    s = 1.0
    for _ in range(7):
        with torch.no_grad():
            for m in mods:
                if id(m) in dirs:
                    m.sift_delta.copy_(dirs[id(m)] * s)
        ls = _eval()
        if ls < best_l - 1e-6:
            best_s, best_l = s, ls
            break
        s *= 0.5
    with torch.no_grad():
        for m in mods:
            if id(m) in dirs:
                m.sift_delta.copy_(dirs[id(m)] * best_s)
    return {"passes": n_passes, "accepted": best_s > 0,
            "scale": best_s, "loss0": l0, "loss_newton": best_l}


def es_scale_polish(model: nn.Module, calib_iter, loss_fn, iters: int = 16,
                    pop: int = 8, sigma: float = 0.05, lr: float = 0.05,
                    eval_batches: int = 2) -> dict:
    """Backward-free per-slice scale evolution (SIFT-ES).

    One scalar gain per attached slice, evolved with antithetic OpenAI-ES
    on a few fixed calibration batches; gains are clamped >= 0 (sign flips
    diverge — measured) and folded into the deltas in-place at the end, so
    the forward/artifact format is unchanged. Zero backward passes, zero
    optimizer state. Measured on Spark: holdout 0.945 -> 0.848 in 16 iters
    (256 forward evals). A *polish* stage, not a gradient replacement.
    """
    mods = [m for m in model.modules()
            if getattr(m, "_sift_side", None) is not None]
    if not mods:
        return {"iters": 0}
    base = [m.sift_delta.detach().clone() for m in mods]
    fixed: list = []
    it = calib_iter()
    for _ in range(eval_batches):
        try:
            fixed.append(next(it))
        except StopIteration:
            break

    def loss_at(theta):
        with torch.no_grad():
            for m, b, t in zip(mods, base, theta):
                m.sift_delta.copy_(b * t)
            return sum(float(loss_fn(*b)) for b in fixed) / len(fixed)

    n = len(mods)
    device = mods[0].sift_delta.device
    theta = torch.ones(n, device=device)
    l0 = loss_at(theta)
    for _ in range(iters):
        eps = torch.randn(pop // 2, n, device=device)
        eps = torch.cat([eps, -eps])
        ls = torch.tensor([loss_at((theta + sigma * e).clamp(min=0))
                           for e in eps], device=device)
        w = -(ls - ls.mean()) / (ls.std() + 1e-8)
        theta = (theta + lr * (w[:, None] * eps).mean(0) / sigma).clamp(min=0)
    lf = loss_at(theta)
    if lf > l0:  # never leave the model worse than the pre-polish point
        loss_at(torch.ones(n, device=device))
        return {"iters": iters, "accepted": False, "loss0": l0, "loss_es": l0}
    return {"iters": iters, "accepted": True, "loss0": l0, "loss_es": lf}


# ---------------------------------------------------------------------------
# Adapter save / load — the compact artifact
# ---------------------------------------------------------------------------


def save_sift_adapter(model: nn.Module, path: str, meta: dict | None = None) -> str:
    """Write only the touched slices: per module ``sift_sel`` (int64),
    ``sift_delta`` (bf16), ``sift_side`` (uint8 row=0/col=1)."""
    from safetensors.torch import save_file
    state = {}
    touched = 0
    for name, mod in model.named_modules():
        if getattr(mod, "_sift_side", None) is None:
            continue
        state[f"{name}.sift_sel"] = mod.sift_sel.cpu()
        state[f"{name}.sift_delta"] = mod.sift_delta.detach().cpu()
        state[f"{name}.sift_side"] = torch.tensor(
            [0 if mod._sift_side == "row" else 1], dtype=torch.uint8)
        touched += mod.sift_delta.numel()
    m = dict(meta or {})
    m["sift_adapter"] = True
    m["sift_only"] = True  # resume-detection flag, mirrors lora_only
    m["touched_params"] = touched
    save_file(state, path, metadata={k: str(v) for k, v in m.items()})
    side = f"{path}.meta.json"
    try:
        with open(side, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
    except OSError:
        pass
    return path


def load_sift_adapter(model: nn.Module, path: str) -> int:
    """Attach deltas exactly as recorded in an adapter file (no
    recalibration) — the resume/eval-apply path. Returns n slices."""
    from safetensors.torch import load_file
    sd = load_file(path)
    bases = sorted({k.rsplit(".sift_", 1)[0] for k in sd})
    n = 0
    for b in bases:
        mod = _mod(model, b)
        sel = sd[f"{b}.sift_sel"].to(mod.weight.device)
        side = "row" if int(sd[f"{b}.sift_side"][0]) == 0 else "col"
        p = _attach_slice(mod, side, sel)
        with torch.no_grad():
            p.copy_(sd[f"{b}.sift_delta"].to(p.device, p.dtype))
        n += 1
    _register_mirrors(model)
    return n


def freeze_non_sift(model: nn.Module) -> int:
    """requires_grad=False for every param that is not a sift_delta."""
    n = 0
    for name, p in model.named_parameters():
        if name.endswith("sift_delta"):
            continue
        p.requires_grad_(False)
        p._forge_frozen = True
        n += 1
    return n


def resume_has_sift(path) -> bool:
    """True when a checkpoint/adapter file carries sift_* tensors — checked
    on the file itself (meta flags may be absent on emergency saves)."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            return any(".sift_" in k for k in f.keys())
    except Exception:
        return False


def collect_selection(model: nn.Module) -> dict:
    """{module_path: (side, sel)} for every attached slice."""
    return {name: (mod._sift_side, mod.sift_sel.detach().clone())
            for name, mod in model.named_modules()
            if getattr(mod, "_sift_side", None) is not None}


# ---------------------------------------------------------------------------
# Prefix cache — skip the frozen prefix entirely on repeat views
# ---------------------------------------------------------------------------


def _row_key(ids: torch.Tensor) -> str:
    return hashlib.blake2b(ids.detach().to("cpu", torch.int32)
                           .numpy().tobytes(), digest_size=16).hexdigest()


class SiftPrefixCache:
    """Per-row cache of the frontier-input hidden state.

    Keys are content hashes of the *non-pad* token ids — left-pad depth
    varies with batch composition, but causal attention makes the hidden at
    real positions a pure function of the real tokens only. Hidden rows are
    stored content-length and right-aligned on inject (pad positions get
    zeros — masked downstream, loss-ignored anyway). Entries live in
    pinned CPU memory; hit rows pay only an H2D copy.
    """

    def __init__(self, model: nn.Module, frontier: int, max_gb: float = 6.0):
        self.model = model
        self.frontier = frontier
        self.max_bytes = int(max_gb * 1e9)
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.fills = 0

    @staticmethod
    def _content_len(attn_mask_row: torch.Tensor | None, T: int) -> int:
        return int(attn_mask_row.sum().item()) if attn_mask_row is not None else T

    def _store(self, key: str, h: torch.Tensor) -> None:
        h = h.to("cpu", copy=True).pin_memory() if h.is_cuda else h.cpu()
        b = h.numel() * h.element_size()
        while self.bytes + b > self.max_bytes and self.cache:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.cache[key] = h
        self.bytes += b

    def prepare(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None,
                device: torch.device) -> None:
        """Ensure every row's prefix hidden is cached, then set the model's
        inject/skip hooks for the next forward."""
        B, T = input_ids.shape
        keys = []
        for i in range(B):
            n = self._content_len(
                attn_mask[i] if attn_mask is not None else None, T)
            keys.append(_row_key(input_ids[i, T - n:]))
        miss_rows = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss_rows:
            mid = input_ids[miss_rows]
            mm = attn_mask[miss_rows] if attn_mask is not None else None
            m = self.model
            try:
                m._sift_capture_at = self.frontier
                with torch.no_grad():
                    m(mid, attention_mask=mm)
                cap = m._sift_captured  # [M, T, d]
            finally:
                m._sift_capture_at = None
                m._sift_captured = None
            for r, i in enumerate(miss_rows):
                n = self._content_len(
                    attn_mask[i] if attn_mask is not None else None, T)
                self._store(keys[i], cap[r, T - n:])  # content rows only
            self.fills += len(miss_rows)
        self.hits += B - len(miss_rows)
        c0 = self.cache[keys[0]]
        h = torch.zeros((B, T, c0.shape[-1]), dtype=c0.dtype, device=device)
        for i, k in enumerate(keys):
            c = self.cache[k].to(device, non_blocking=True)
            h[i, T - c.shape[0]:] = c
        m = self.model
        m._sift_inject_hidden = h
        m._sift_start_layer = self.frontier

    def clear(self) -> None:
        m = self.model
        m._sift_inject_hidden = None
        m._sift_start_layer = 0

    def invalidate(self) -> None:
        self.cache.clear()
        self.bytes = 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class SiftState:
    model: nn.Module
    frontier: int
    selection: dict
    delta_params: list
    block_mass: dict
    n_touched: int
    args: dict
    prefix_cache: SiftPrefixCache | None = None
    calib_iter: object = None
    loss_fn: object = None
    cands: _Cands | None = None
    n_reselects: int = 0

    def reselect(self) -> dict:
        """Merge deltas, re-calibrate, re-pick units. Caller rebuilds the
        optimizer afterwards (new Parameter objects)."""
        merge_sift(self.model, zero=False)
        for mod in self.model.modules():
            if getattr(mod, "_sift_side", None) is not None or \
                    getattr(mod, "_sift_orig_forward", None) is not None:
                _detach_slice(mod)
        st = setup_sift(self.model, self.calib_iter, self.loss_fn, **self.args)
        self.frontier = st.frontier
        self.selection = st.selection
        self.delta_params = st.delta_params
        self.block_mass = st.block_mass
        self.n_touched = st.n_touched
        self.cands = st.cands
        self.n_reselects += 1
        if self.prefix_cache is not None and st.frontier != self.prefix_cache.frontier:
            self.prefix_cache.invalidate()
            self.prefix_cache.frontier = st.frontier
        return {"frontier": self.frontier, "n_touched": self.n_touched}

    def prepare_prefix(self, input_ids, attn_mask, device) -> None:
        if self.prefix_cache is not None:
            self.prefix_cache.prepare(input_ids, attn_mask, device)

    def clear_prefix(self) -> None:
        if self.prefix_cache is not None:
            self.prefix_cache.clear()


def resume_state(model: nn.Module, calib_iter, loss_fn, **args) -> SiftState:
    """Build the driver state for slices that were attached by
    ``load_sift_adapter`` (resume path — skips calibration entirely)."""
    sel = collect_selection(model)
    params = [m.sift_delta for m in model.modules()
              if getattr(m, "_sift_side", None) is not None]
    n_touched = sum(p.numel() for p in params)
    frontiers = [int(n.split(".")[1]) for n in sel if n.startswith("blocks.")]
    frontier = min(frontiers) if frontiers else 0
    pc = SiftPrefixCache(model, frontier, args.get("prefix_cache_gb", 0.0)) \
        if args.get("prefix_cache_gb", 0.0) > 0 else None
    return SiftState(model=model, frontier=frontier, selection=sel,
                     delta_params=params, block_mass={}, n_touched=n_touched,
                     args=args, prefix_cache=pc, calib_iter=calib_iter,
                     loss_fn=loss_fn, cands=None)


def setup_sift(model: nn.Module, calib_iter, loss_fn, *,
               attn_units: int = 4, ffn_units: int = 8,
               attn_gran: int = 128, ffn_gran: int = 32,
               include_kv: bool = False,
               frontier="auto", frontier_mass: float = 0.75,
               calib_batches: int = 4,
               score_mode: str = "fro",
               newton_passes: int = 0, newton_lr: float = 1.0,
               newton_damp: float = 1e-2,
               prefix_cache_gb: float = 0.0) -> SiftState:
    """Calibrate -> pick frontier+units -> attach deltas -> freeze rest.

    ``calib_iter`` is a callable returning a fresh iterator of
    ``(input_ids, labels, attn_mask)``; ``loss_fn(ids, labels, mask)``
    returns the scalar CE loss. Both are sft_train callables.
    """
    model.train()
    # Calibration backprops through every candidate block — make sure
    # gradient checkpointing is active NOW (sft_train's explicit
    # enable_gradient_checkpointing() runs after setup; on the ForgeEngine
    # load path cfg alone does not engage it).
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    cands = discover_units(model, attn_gran, ffn_gran, include_kv)
    block_mass = calibrate_importance(
        model, cands, calib_iter, loss_fn, calib_batches, score_mode)
    if frontier == "auto":
        F_ = pick_frontier(block_mass, frontier_mass, len(model.blocks))
    else:
        F_ = int(frontier)
    sel = build_selection(cands, F_, attn_units, ffn_units)
    params = apply_selection(model, sel)
    _register_mirrors(model)
    n_frozen = freeze_non_sift(model)
    n_touched = sum(p.numel() for p in params)
    if newton_passes > 0:
        newton_init(model, calib_iter, loss_fn, newton_passes,
                    lr=newton_lr, damp=newton_damp)
    pc = SiftPrefixCache(model, F_, max_gb=prefix_cache_gb) \
        if prefix_cache_gb > 0 else None
    return SiftState(
        model=model, frontier=F_, selection=sel, delta_params=params,
        block_mass=block_mass, n_touched=n_touched,
        args=dict(attn_units=attn_units, ffn_units=ffn_units,
                  attn_gran=attn_gran, ffn_gran=ffn_gran,
                  include_kv=include_kv, frontier=frontier,
                  frontier_mass=frontier_mass, calib_batches=calib_batches,
                  score_mode=score_mode,
                  newton_passes=newton_passes, newton_lr=newton_lr,
                  newton_damp=newton_damp,
                  prefix_cache_gb=prefix_cache_gb),
        prefix_cache=pc, calib_iter=calib_iter, loss_fn=loss_fn, cands=cands)

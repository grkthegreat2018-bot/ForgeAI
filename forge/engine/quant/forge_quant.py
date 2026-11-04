"""ForgeQuant — calibration-free INT4 + INT8 row-mixed weight quant (R32-6).

Format: every output row of a Linear is stored at INT4 (group-128
symmetric absmax), EXCEPT the top `sparse_ratio` rows by weight L2 norm,
which are stored at INT8 (per-row absmax). The INT4 copy of outlier rows
is zeroed but still packed (~0.4 bits/param of dead storage). Decode uses
a fused Triton GEMV (packed nibbles + int8 outlier matvec + index_add);
prefill/M>1 falls back to dequantize-then-matmul. No calibration, no
Hessian, no activation statistics — pure RTN.

Prior art (this is a recombination, not a new format):
  - OWQ (arXiv 2306.02272) — dense 3-4b + high-precision outlier columns
  - LLM.int8() (arXiv 2208.07339) — magnitude-selected outlier channels
  - SpQR (arXiv 2306.03078) — dense low-bit + sparse high-prec residual
  - GRINQH (arXiv 2606.23419) — per-channel precision assignment WITH
    activation statistics + calibration (the stronger version of this idea)
  - SharQ (arXiv 2606.26587) — ACTIVATION-side sparse-dense FP4, not a
    weight format; cited in R32 as inspiration but not "inverted"

ForgeQuant's differentiators are choices, not novelty: weight-norm
selection (weaker than activation-aware), whole-row granularity, INT8
outliers, zero calibration. Activation-aware or error-aware row selection
would likely dominate the weight-norm criterion for free.

Bit budget (gs=128, sparse_ratio=0.10) — ~4.9 bits/param, NOT the 3.6
claimed in the original R32 notes:
  - INT4 dense: 4.0 bits/param (all rows incl. zeroed outlier rows)
  - fp16 group scales: 0.125; INT8 outlier rows: ~0.8 + indices
  → compare quality against W5 baselines, not W4.  VRAM for V10 (1.2B):
  ~700-740MB vs ~2.4GB bf16.

Measured (.devin/scratchpad.md + .devin/exp_results.jsonl):
  - qwen25_coder_3b_mla_moe_fq: ~41 tok/s fused GEMV decode, 4.5GB VRAM
    (vs torchao int4 59.7 tok/s — FQ is slower but BCG-capturable);
    unfused prequant ~25 tok/s.
  - Qwen2.5-0.5B: dNLL +0.47 vs bf16 — RTN-class quality, not SOTA.

R66-quant upgrades (opt-in):
  - ``use_rotation=True`` — QuaRot/SpinQuant-style block-diagonal
    Hadamard rotation on the input dim at pack time: R = D·H_gs where
    D is an RNG-seeded ±1 diagonal and H_gs is the normalized
    Walsh-Hadamard per group (gs must be a power of 2). Stored dense
    weights are quantized in the ROTATED basis — group absmax shrinks
    toward RMS as outliers spread across the group, cutting quant
    error. The rotation folds into the activation at forward
    (out = (xR)(WR)^T = xW^T exactly); the FWHT costs O(K·log gs) vs
    O(M·K·N) for the matmul. The INT8 outlier rows stay in the
    ORIGINAL basis (unchanged param layout, in_f recovery intact).
    Packed ckpts flag it via forge_convert.json {"fq_rotation": true}.
  - Fused batched Triton path: ``_fq_int4_gemm_kernel`` extends the
    int4 GEMV to M in 2..32 (tl.dot, per-N-row-block program_id), so
    small-batch decode/prefill no longer pays the full dequant+cuBLAS
    unpack on every forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin

try:  # fused decode GEMV — optional, CUDA only
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only envs
    _HAS_TRITON = False


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform on the last dim (power-of-2 size).

    Unnormalized — H H = B·I, so _fwht(_fwht(x)) == B * x. Butterfly
    structure, O(B log B)."""
    B = x.shape[-1]
    y = x.clone().reshape(-1, B)
    h = 1
    while h < B:
        y = y.reshape(-1, B // (2 * h), 2, h)
        a = y[..., 0, :].clone()
        y[..., 0, :] = a + y[..., 1, :]
        y[..., 1, :] = a - y[..., 1, :]
        h *= 2
    return y.reshape(x.shape)


if _HAS_TRITON:
    @triton.jit
    def _fq_int4_gemv_kernel(
        x_ptr,           # (K_padded,) input, any float dtype
        packed_ptr,      # (N, n_groups*GS//2) uint8 packed nibbles
        scales_ptr,      # (N, n_groups) fp16 group scales
        out_ptr,         # (N,) fp32 output
        N, n_groups, stride_p,
        GS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[r] = sum_i x[i] * ((u[r,i]-8) * scale[r, i//GS] / 7).

        One packed byte = two adjacent in-features (lo nibble even,
        hi nibble odd). GS//2 packed bytes == one group, so the group
        scale is a per-row scalar per inner iteration.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        BK: tl.constexpr = GS // 2
        for g in range(0, n_groups):
            cols = g * BK + tl.arange(0, BK)
            b = tl.load(packed_ptr + rows[:, None] * stride_p
                        + cols[None, :],
                        mask=rmask[:, None], other=0)
            lo = (b & 0x0F).to(tl.float32) - 8.0
            hi = (b >> 4).to(tl.float32) - 8.0
            w = tl.interleave(lo, hi)                    # (BN, GS)
            xoff = g * GS + tl.arange(0, GS)
            xv = tl.load(x_ptr + xoff).to(tl.float32)
            sc = tl.load(scales_ptr + rows * n_groups + g,
                         mask=rmask, other=0.0).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1) * (sc * 0.14285714285714285)
        tl.store(out_ptr + rows, acc, mask=rmask)

    @triton.jit
    def _fq_int4_gemm_kernel(
        x_ptr,           # (M, K_pad) input rows — rotation pre-applied
        packed_ptr,      # (N, n_groups*GS//2) uint8 packed nibbles
        scales_ptr,      # (N, n_groups) fp16 group scales
        out_ptr,         # (M, N) fp32 output
        M, N, n_groups, stride_p, stride_x,
        GS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Batched sibling of _fq_int4_gemv_kernel: out[m,r] for a
        (BLOCK_N x BLOCK_M) tile per program. Same nibble layout; the
        per-group inner product uses tl.dot (ieee — the GEMV's fp32 FMA
        accumulates exactly, TF32 would silently degrade 4-bit math).
        BLOCK_M is padded up to the 16-row tl.dot floor and masked, so
        M in 2..32 maps to BLOCK_M 16 or 32 on a per-N-block grid.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        ms = tl.arange(0, BLOCK_M)
        mmask = ms < M
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        BK: tl.constexpr = GS // 2
        for g in range(0, n_groups):
            cols = g * BK + tl.arange(0, BK)
            b = tl.load(packed_ptr + rows[:, None] * stride_p
                        + cols[None, :],
                        mask=rmask[:, None], other=0)
            lo = (b & 0x0F).to(tl.float32) - 8.0
            hi = (b >> 4).to(tl.float32) - 8.0
            w = tl.interleave(lo, hi)                    # (BN, GS)
            xoff = g * GS + tl.arange(0, GS)
            xv = tl.load(x_ptr + ms[:, None] * stride_x + xoff[None, :],
                         mask=mmask[:, None], other=0.0).to(tl.float32)
            sc = tl.load(scales_ptr + rows * n_groups + g,
                         mask=rmask, other=0.0).to(tl.float32)
            acc += tl.dot(w, tl.trans(xv),
                          input_precision="ieee") * \
                (sc * 0.14285714285714285)[:, None]
        tl.store(out_ptr + ms[:, None] * N + rows[None, :],
                 tl.trans(acc),
                 mask=mmask[:, None] & rmask[None, :])

    @triton.jit
    def _nf4_gemv_kernel(
        x_ptr,           # (K_padded,) input, any float dtype
        packed_ptr,      # (N, n_groups*GS//2) uint8 packed nibbles
        scales_ptr,      # (N, n_groups) fp16 group absmax scales
        levels_ptr,      # (16,) fp32 NF4 codebook
        out_ptr,         # (N,) fp32 output
        N, n_groups, stride_p,
        GS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[r] = sum_i x[i] * nf4[u[r,i]] * scale[r, i//GS].

        Same packed nibble layout as _fq_int4_gemv_kernel but the code
        book is the NF4 LUT instead of signed int4 — the gather is over
        a 16-entry table that stays L1-resident.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        BK: tl.constexpr = GS // 2
        for g in range(0, n_groups):
            cols = g * BK + tl.arange(0, BK)
            b = tl.load(packed_ptr + rows[:, None] * stride_p
                        + cols[None, :],
                        mask=rmask[:, None], other=0)
            lo_idx = (b & 0x0F).to(tl.int32)
            hi_idx = (b >> 4).to(tl.int32)
            idx = tl.interleave(lo_idx, hi_idx)          # (BN, GS)
            w = tl.load(levels_ptr + idx)                # LUT gather
            xoff = g * GS + tl.arange(0, GS)
            xv = tl.load(x_ptr + xoff).to(tl.float32)
            sc = tl.load(scales_ptr + rows * n_groups + g,
                         mask=rmask, other=0.0).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1) * sc
        tl.store(out_ptr + rows, acc, mask=rmask)


class ForgeQuantLinear(QuantizedLinearMixin):
    """ForgeQuant quantized linear: INT4 dense + INT8 outlier-row split.

    The dense path uses INT4 with per-group absmax scaling (group_size=128).
    The sparse path stores the top-k outlier rows (by L2 norm) at INT8 with
    their row indices; the dense copies of those rows are zeroed but still
    packed. At inference the int4 GEMV output is corrected by an int8 row
    matvec scattered via index_add. The Triton GEMV is a scalar dequant+FMA
    kernel — it does NOT use tensor cores (mma.sync); the SM120 benefit is
    memory bandwidth (~0.6 B/weight read), not INT4 tensor-core throughput.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128, sparse_ratio: float = 0.10,
                 use_rotation: bool = False, rotation_seed: int = 0,
                 residual_rank: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.group_size = group_size
        self.sparse_ratio = sparse_ratio
        # Opt-in Hadamard rotation (R66): dense weights are packed in the
        # rotated basis R = D·H_gs; forward rotates the activation first.
        # Deterministic from rotation_seed — no checkpoint params added.
        self.use_rotation = bool(use_rotation)
        self.rotation_seed = int(rotation_seed)
        if self.use_rotation and (group_size & (group_size - 1)):
            raise ValueError(
                "use_rotation requires power-of-2 group_size "
                f"(got {group_size})")
        n_sparse = max(1, int(out_features * sparse_ratio))
        self.n_sparse = n_sparse
        n_groups = (in_features + group_size - 1) // group_size
        self._k_pad = n_groups * group_size      # group-padded in width

        # Dense: INT4 packed (2 per byte) → (out, n_groups*gs // 2).
        # load_from_weight pads to whole groups before packing, so the
        # stored width is the padded width, not (in+1)//2.
        self.dense_packed = nn.Parameter(
            torch.zeros(out_features, n_groups * group_size // 2,
                        dtype=torch.uint8),
            requires_grad=False)
        # Dense scales: (out, n_groups) float16
        self.dense_scales = nn.Parameter(
            torch.zeros(out_features, n_groups, dtype=torch.float16),
            requires_grad=False)

        # Sparse: INT8 for top-k outlier channels (high precision where it matters)
        # Stored as: (n_sparse, in_features) int8 + (n_sparse,) indices + (n_sparse, 1) scales
        self.sparse_packed = nn.Parameter(
            torch.zeros(n_sparse, in_features, dtype=torch.int8),
            requires_grad=False)
        self.sparse_indices = nn.Parameter(
            torch.zeros(n_sparse, dtype=torch.long),
            requires_grad=False)
        self.sparse_scales = nn.Parameter(
            torch.zeros(n_sparse, 1, dtype=torch.float16),
            requires_grad=False)

        self._cached_weight: torch.Tensor | None = None
        self._cached_weight_rot: torch.Tensor | None = None
        # Prequant checkpoints on VRAM-tight models set this False — caching
        # every dequantized weight is a second full bf16 model in VRAM.
        self._cache_w = True
        # Fused int4 GEMV (Triton) for single-token decode on CUDA — reads
        # packed weights only, no bf16 weight materialization. Falls back
        # to the dequantize+linear path for prefill (M>max_m) or CPU tensors.
        self._fused_gemv = True
        # Batched fused path (_fq_int4_gemm_kernel): M <= this uses the
        # packed Triton GEMM instead of dequant+cuBLAS.
        self._fused_gemm_max_m = 32
        self._rot_cache: dict = {}               # device -> ±1 diag (K_pad,)
        self.lora_adapter: nn.Module | None = None
        # FP16 low-rank residual patch (ForgeQ-R): W ≈ deq(Q) + res_a @ res_b,
        # fitted at pack time by SVD of the quantization error (LoftQ-style).
        # Stored fp16, applied in fp32 at forward — rank-r GEMV is ~free at
        # decode. NOT folded into _dequantize_weight or merge_lora: it stays
        # a separate additive correction so the int4 surface remains pure.
        if residual_rank > 0:
            self.residual_rank = residual_rank
            self.res_a = nn.Parameter(
                torch.zeros(out_features, residual_rank,
                            dtype=torch.float16),
                requires_grad=False)
            self.res_b = nn.Parameter(
                torch.zeros(residual_rank, in_features,
                            dtype=torch.float16),
                requires_grad=False)
        else:
            self.residual_rank = 0
            self.res_a = None
            self.res_b = None

    # ── Hadamard rotation helpers (use_rotation only) ──────────────────

    def _rot_diag(self, device) -> torch.Tensor:
        """Seeded ±1 diagonal over the group-padded width, cached per
        device. Deterministic — regenerated identically at pack and load."""
        d = self._rot_cache.get(device)
        if d is None:
            g = torch.Generator().manual_seed(self.rotation_seed)
            d = (torch.randint(0, 2, (self._k_pad,), generator=g,
                               dtype=torch.int8).float() * 2 - 1).to(device)
            self._rot_cache[device] = d
        return d

    def _rotate_x(self, t: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, K_pad): apply R = D·H_blockdiag. Fp32 math, one
        orthogonal transform — xR then (WR)^T equals xW^T exactly."""
        gs = self.group_size
        t = t.float()
        if t.shape[-1] < self._k_pad:
            t = F.pad(t, (0, self._k_pad - t.shape[-1]))
        d = self._rot_diag(t.device)
        tg = (t * d).reshape(*t.shape[:-1], self._k_pad // gs, gs)
        return _fwht(tg).reshape(*t.shape) * (gs ** -0.5)

    def _unrotate_w(self, w_rot: torch.Tensor) -> torch.Tensor:
        """Inverse of _rotate_x on packed-width rows: u = vR ⇒ v = H(u)⊙d
        (H is symmetric orthonormal; D and H do not commute — order matters)."""
        gs = self.group_size
        d = self._rot_diag(w_rot.device)
        ug = w_rot.float().reshape(*w_rot.shape[:-1], self._k_pad // gs, gs)
        return (_fwht(ug) * (gs ** -0.5) * d.reshape(-1, gs)).reshape(
            w_rot.shape)

    # ── fused Triton decode paths ──────────────────────────────────────

    def _fused_gemv_forward(self, x: torch.Tensor,
                            x_sparse: torch.Tensor | None = None
                            ) -> torch.Tensor:
        """x: (in_features,) CUDA float vector -> (out_features,) same dtype.
        x_sparse: original-basis vector for the INT8 outlier correction
        (identical to x unless use_rotation, where the dense side consumes
        the rotated input instead)."""
        dev = x.device
        packed = self.dense_packed.data                      # (N, Kp) uint8
        scales = self.dense_scales.data                      # (N, G) fp16
        N, n_groups = scales.shape
        K_pad = packed.shape[1] * 2                          # padded in-features
        if self.use_rotation:
            xv = self._rotate_x(x.reshape(-1))
        else:
            xv = x.reshape(-1)
            if xv.numel() < K_pad:
                xv = F.pad(xv, (0, K_pad - xv.numel()))
        out = torch.empty(N, device=dev, dtype=torch.float32)
        if N >= 2048:
            BLOCK_N, num_warps = 128, 8      # measured 139 GB/s @2048
        else:
            BLOCK_N, num_warps = 32, 4       # parallelism on small N
        _fq_int4_gemv_kernel[((N + BLOCK_N - 1) // BLOCK_N,)](
            xv, packed, scales, out, N, n_groups, packed.shape[1],
            GS=self.group_size, BLOCK_N=BLOCK_N, num_warps=num_warps)
        # INT8 sparse outlier correction — always ORIGINAL basis
        # (n_sparse x in_features), even under use_rotation.
        xs = x if x_sparse is None else x_sparse
        sp = self.sparse_packed.data.to(x.dtype) @ xs[: self.in_features]
        sp = sp * self.sparse_scales.data.squeeze(-1).to(sp.dtype)
        out.index_add_(0, self.sparse_indices.data.to(dev),
                       sp.to(torch.float32))
        if self.bias is not None:
            out += self.bias.data.to(dev, torch.float32)
        return out.to(x.dtype)

    def _fused_gemm_forward(self, x2d: torch.Tensor) -> torch.Tensor:
        """Batched fused path: x2d (M, in_features) CUDA, 2 <= M <=
        _fused_gemm_max_m -> (M, out_features). Reads packed weights
        only; the INT8 sparse correction stays a small torch matvec."""
        dev = x2d.device
        M = x2d.shape[0]
        packed = self.dense_packed.data
        scales = self.dense_scales.data
        N, n_groups = scales.shape
        K_pad = packed.shape[1] * 2
        if self.use_rotation:
            xv = self._rotate_x(x2d)
        else:
            xv = x2d
            if xv.shape[1] < K_pad:
                xv = F.pad(xv, (0, K_pad - xv.shape[1]))
        xv = xv.contiguous()
        out = torch.empty(M, N, device=dev, dtype=torch.float32)
        if N >= 2048:
            BLOCK_N, num_warps = 128, 8
        else:
            BLOCK_N, num_warps = 32, 4
        BLOCK_M = 16 if M <= 16 else 32
        if BLOCK_M == 32 and BLOCK_N == 128:
            # (128N x 128K) fp32 w-tile + (32M x 128K) x-tile + dot
            # staging > 99KB smem on SM120 — halve the row block.
            BLOCK_N = 64
        _fq_int4_gemm_kernel[((N + BLOCK_N - 1) // BLOCK_N,)](
            xv, packed, scales, out, M, N, n_groups,
            packed.shape[1], xv.stride(0),
            GS=self.group_size, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
            num_warps=num_warps)
        # INT8 sparse outlier correction — original basis
        sp = self.sparse_packed.data.to(x2d.dtype) @ x2d.T  # (n_sparse, M)
        sp = sp * self.sparse_scales.data.to(sp.dtype)      # (n_sparse, 1)
        out.index_add_(1, self.sparse_indices.data.to(dev),
                       sp.T.to(torch.float32))
        if self.bias is not None:
            out += self.bias.data.to(dev, torch.float32)
        return out.to(x2d.dtype)

    @torch.no_grad()
    def load_from_weight(self, w: torch.Tensor):
        """Quantize a float weight tensor into ForgeQuant format."""
        assert w.shape == (self.out_features, self.in_features)
        gs = self.group_size
        n_groups = (self.in_features + gs - 1) // gs

        # Dense storage basis: original weight, or the Hadamard-rotated
        # (group-padded) one. Rotation is orthogonal ⇒ per-row L2 norms
        # are identical, so the outlier-row selection below is unchanged.
        if self.use_rotation:
            w_eff = self._rotate_x(
                F.pad(w.float(), (0, self._k_pad - self.in_features)))
        else:
            w_eff = w

        # 1. Identify outlier channels by per-row L2 norm
        row_norms = w_eff.norm(dim=1)  # (out_features,)
        _, top_indices = row_norms.topk(self.n_sparse)
        top_indices = top_indices.sort().values  # keep sorted for cache
        self.sparse_indices.data = top_indices.to('cpu')

        # 2. Extract sparse outlier rows — ALWAYS the original basis, so
        # the correction contracts with un-rotated activations at forward.
        w_sparse = w[top_indices]  # (n_sparse, in_features)
        sparse_scales = w_sparse.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.sparse_scales.data = sparse_scales.to(torch.float16).to('cpu')

        # Quantize sparse to INT8 (high precision for outliers)
        w_sparse_norm = (w_sparse / sparse_scales).clamp(-128, 127)
        sparse_q = torch.round(w_sparse_norm).clamp(-128, 127).to(torch.int8)
        self.sparse_packed.data = sparse_q.to('cpu')

        # 3. Quantize dense in the STORAGE basis: zero the outlier rows
        # (zero rows stay zero under rotation), then pad/quantize.
        w_dense = w_eff.clone()
        w_dense[top_indices] = 0.0

        # INT4 quantization with per-group scale
        pad_g = n_groups * gs - w_dense.shape[-1]
        if pad_g > 0:
            w_dense = F.pad(w_dense, (0, pad_g))
        w_grouped = w_dense.reshape(self.out_features, n_groups, gs)
        dense_scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = (w_grouped / dense_scales).clamp(-1, 1)
        # INT4: 16 levels from -7 to 7 (or -8 to 7)
        int4_vals = w_norm * 7  # scale to [-7, 7]
        int4_q = torch.round(int4_vals).clamp(-8, 7).to(torch.int8)  # (out, n_groups, gs)
        # Convert to unsigned 4-bit: add 8 → [0, 15]
        int4_u = (int4_q + 8).to(torch.uint8)
        # Pack 2 per byte
        int4_flat = int4_u.reshape(self.out_features, -1)
        packed_dense = int4_flat[:, 0::2] | (int4_flat[:, 1::2] << 4)
        self.dense_packed.data = packed_dense.to('cpu')
        self.dense_scales.data = dense_scales.squeeze(-1).to(torch.float16).to('cpu')
        self._cached_weight = None
        self._cached_weight_rot = None

    def _dequantize_dense(self, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize INT4 dense weights — in the STORAGE basis (rotated
        + full padded width when use_rotation, else cropped to in_features)."""
        packed = self.dense_packed.data
        scales = self.dense_scales.data.to(torch.float32)
        low = (packed & 0x0F).long()
        high = (packed >> 4).long()
        idx = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)
        # Packed stream is padded to n_groups*group_size — do NOT crop to
        # in_features here or the group reshape breaks when
        # in_features % group_size != 0 (final line crops instead).
        # Convert back: unsigned → signed
        int4_q = idx.to(torch.float32) - 8  # [-8, 7]
        w_norm = int4_q / 7.0  # [-1, 1] approx
        # Apply per-group scales
        n_groups = scales.shape[1]
        w_grouped = w_norm.reshape(self.out_features, n_groups, -1)
        w_scaled = w_grouped * scales.unsqueeze(-1)
        w = w_scaled.reshape(self.out_features, -1)
        if not self.use_rotation:
            w = w[:, :self.in_features]
        return w.to(dtype)

    def _dequantize_sparse(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize INT8 sparse weights. Returns (values, indices)."""
        packed = self.sparse_packed.data.to(torch.float32)
        scales = self.sparse_scales.data.to(torch.float32)
        w_scaled = packed * scales
        return w_scaled.to(dtype), self.sparse_indices.data

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        """Approximate reconstruction in the ORIGINAL basis (out, in) —
        the compat surface for merge_lora/tests. Under use_rotation the
        dense half is un-rotated; the INT8 sparse half is already original."""
        if self._cached_weight is not None:
            return self._cached_weight.to(dtype)
        w = self._dequantize_dense(dtype)
        if self.use_rotation:
            w = self._unrotate_w(w)[:, :self.in_features]
        # Add sparse outlier correction
        w_sparse, sparse_idx = self._dequantize_sparse(dtype)
        w[sparse_idx] += w_sparse
        if cache:
            self._cached_weight = w.to(torch.bfloat16)
        return w

    def _rotated_dense(self, dtype: torch.dtype) -> torch.Tensor:
        """Rotated-basis dense weight (out, K_pad) for the rotation
        forward path, with the same _cache_w semantics as _cached_weight."""
        if self._cached_weight_rot is not None:
            return self._cached_weight_rot.to(dtype)
        w = self._dequantize_dense(dtype)
        if self._cache_w:
            self._cached_weight_rot = w.to(torch.bfloat16)
        return w

    def _residual_apply(self, x: torch.Tensor, out: torch.Tensor
                        ) -> torch.Tensor:
        """Add the FP16 low-rank quant-error patch: out += x @ res_b.T @ res_a.T.

        Computed in fp32 (it exists to recover precision), then cast to
        out's dtype. Cost is a rank-r matvec — negligible at decode.
        """
        if self.res_a is None:
            return out
        flat = x.reshape(-1, x.shape[-1]) if x.dim() > 1 \
            else x.reshape(1, -1)
        corr = (flat.float() @ self.res_b.data.t().float()
                ) @ self.res_a.data.t().float()
        return out + corr.reshape(out.shape).to(out.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]) if x.dim() > 1 \
            else x.reshape(1, -1)
        M = flat.shape[0]
        if (self._fused_gemv and _HAS_TRITON and x.is_cuda
                and self.dense_packed.is_cuda):
            if M == 1:
                v = flat.reshape(-1)
                out = self._fused_gemv_forward(v, x_sparse=v)
                out = out.view(*x.shape[:-1], self.out_features)
            elif M <= self._fused_gemm_max_m:
                out = self._fused_gemm_forward(flat)
                out = out.view(*x.shape[:-1], self.out_features)
            else:
                out = None
            if out is not None:
                if self.lora_adapter is not None:
                    out = out + self.lora_adapter(x)
                return self._residual_apply(x, out)
        if self.use_rotation:
            # out = (xR)(WR)^T + W_s·x — dense contracts in the rotated
            # basis, the INT8 outlier rows contract in the original one.
            w_rot = self._rotated_dense(x.dtype)
            out = F.linear(self._rotate_x(flat).to(x.dtype), w_rot)
            w_sparse, sparse_idx = self._dequantize_sparse(x.dtype)
            out.index_add_(1, sparse_idx.to(out.device),
                           F.linear(flat, w_sparse))
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            out = out.view(*x.shape[:-1], self.out_features)
        else:
            w = self._dequantize_weight(x.dtype, cache=self._cache_w)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            out = F.linear(x, w, bias)
        if self.lora_adapter is not None:
            out = out + self.lora_adapter(x)
        return self._residual_apply(x, out)

    @torch.no_grad()
    def merge_lora(self) -> bool:
        """Merge LoRA adapter into ForgeQuant weights (QLoRA merge)."""
        if self.lora_adapter is None:
            return False
        lora = self.lora_adapter
        w = self._dequantize_weight(torch.float32, cache=False)
        delta = lora.scale * (lora.lora_B @ lora.lora_A)
        w = w + delta.to(torch.float32)
        self.load_from_weight(w)
        self.lora_adapter = None
        self._cached_weight = None
        self._cached_weight_rot = None
        # Refit the ForgeQ-R residual to the merged weight's fresh quant
        # error — the old patch was fitted to the pre-merge error and is
        # stale (LoRA merge is rare; the SVD cost is fine).
        if self.res_a is not None and self.residual_rank > 0:
            resid = w - self._dequantize_weight(torch.float32)
            try:
                U, S, Vh = torch.linalg.svd(resid, full_matrices=False)
                r = self.residual_rank
                sq = S[:r].clamp_min(0).sqrt()
                self.res_a.data.copy_((U[:, :r] * sq.unsqueeze(0)).half())
                self.res_b.data.copy_((sq.unsqueeze(1) * Vh[:r]).half())
            except RuntimeError:
                # SVD failure (rare on CPU, NaN weights) -> zero patch is
                # safer than a wrong one.
                self.res_a.data.zero_()
                self.res_b.data.zero_()
        return True

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"group_size={self.group_size}, "
                f"sparse_ratio={self.sparse_ratio}, "
                f"use_rotation={self.use_rotation}, "
                f"residual_rank={self.residual_rank}, "
                f"forge_quant=True")


def quantize_model_forge_quant(model: nn.Module, group_size: int = 128,
                               sparse_ratio: float = 0.10,
                               target_modules: list[str] | None = None,
                               min_size: int = 64,
                               use_rotation: bool = False) -> int:
    """Replace nn.Linear with ForgeQuantLinear.

    Args:
        model: The model to quantize.
        group_size: INT4 group size for dense path.
        sparse_ratio: Fraction of output rows to store at INT8 (outlier rows).
        target_modules: List of module name substrings to target. None = all Linear.
        min_size: Skip layers smaller than this.
        use_rotation: Hadamard-rotate weights at pack time (QuaRot-style
            outlier spreading; requires power-of-2 group_size).

    Returns:
        Number of layers quantized.
    """
    n_quantized = 0

    def convert(module, prefix=""):
        nonlocal n_quantized
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            is_target = True
            if target_modules is not None:
                is_target = any(t in full_name for t in target_modules)
            if isinstance(child, nn.Linear) and is_target:
                if child.in_features >= min_size and child.out_features >= min_size:
                    fq = ForgeQuantLinear(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        group_size=group_size,
                        sparse_ratio=sparse_ratio,
                        use_rotation=use_rotation)
                    fq.load_from_weight(child.weight.data)
                    if child.bias is not None:
                        fq.bias.data.copy_(child.bias.data)
                    fq = fq.to(child.weight.device)
                    setattr(module, name, fq)
                    n_quantized += 1
            else:
                convert(child, full_name)

    convert(model)
    return n_quantized


# Standard NF4 codebook (QLoRA paper, Dettmers et al. 2023) -- identical
# values to forge.training.bitnet_lora._NF4_LEVELS, redefined here so the
# engine stays free of training-side imports.
_NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986832, -0.5250730514526367, -0.39491748809814453,
    -0.28444141149520874, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170048713684,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


class NF4QuantLinear(QuantizedLinearMixin):
    """NF4 4-bit linear -- engine-side counterpart of training NF4Linear.

    Same checkpoint layout (`weight_packed` uint8 2-nibbles/byte lo=even,
    `weight_scales` fp16 per-group absmax, group_size=64) so a
    QLoRA-NF4 checkpoint loads natively -- no bf16/int4 transcode and no
    double-quantization noise. Decode uses the fused NF4 GEMV kernel
    (reads packed weights only); prefill/M>1 falls back to dequant.

    No sparse-outlier path (unlike ForgeQuantLinear) -- NF4 checkpoints
    carry exactly packed+scales+bias.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, group_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.group_size = group_size
        n_groups = (in_features + group_size - 1) // group_size
        # Stored width is group-padded (load_from_weight pads before pack)
        self.weight_packed = nn.Parameter(
            torch.zeros(out_features, n_groups * group_size // 2,
                        dtype=torch.uint8), requires_grad=False)
        self.weight_scales = nn.Parameter(
            torch.zeros(out_features, n_groups, dtype=torch.float16),
            requires_grad=False)
        self._cached_weight: torch.Tensor | None = None
        self._cache_w = True
        self._fused_gemv = True
        self.lora_adapter: nn.Module | None = None

    def _fused_gemv_forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        packed = self.weight_packed.data
        scales = self.weight_scales.data
        N, n_groups = scales.shape
        K_pad = packed.shape[1] * 2
        xv = x.reshape(-1)
        if xv.numel() < K_pad:
            xv = F.pad(xv, (0, K_pad - xv.numel()))
        out = torch.empty(N, device=dev, dtype=torch.float32)
        levels = _NF4_LEVELS.to(dev)
        if N >= 2048:
            BLOCK_N, num_warps = 128, 8
        else:
            BLOCK_N, num_warps = 32, 4
        _nf4_gemv_kernel[((N + BLOCK_N - 1) // BLOCK_N,)](
            xv, packed, scales, levels, out, N, n_groups,
            packed.shape[1], GS=self.group_size, BLOCK_N=BLOCK_N,
            num_warps=num_warps)
        if self.bias is not None:
            out += self.bias.data.to(dev, torch.float32)
        return out.to(x.dtype)

    @torch.no_grad()
    def load_from_weight(self, w: torch.Tensor):
        """Quantize a float weight into NF4 packed format."""
        assert w.shape == (self.out_features, self.in_features)
        gs = self.group_size
        n_groups = (self.in_features + gs - 1) // gs
        pad = n_groups * gs - self.in_features
        if pad > 0:
            w = F.pad(w, (0, pad))
        w_grouped = w.reshape(self.out_features, n_groups, gs)
        scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = (w_grouped / scales).unsqueeze(-1)
        dist = (w_norm - _NF4_LEVELS.to(w.device).view(1, 1, 1, -1)).abs()
        idx = dist.argmin(dim=-1)
        idx_flat = idx.reshape(self.out_features, -1).to(torch.uint8)
        packed = idx_flat[:, ::2] | (idx_flat[:, 1::2] << 4)
        self.weight_packed.data = packed.to('cpu')
        self.weight_scales.data = scales.squeeze(-1).to(torch.float16).to('cpu')
        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        if self._cached_weight is not None:
            return self._cached_weight.to(dtype)
        packed = self.weight_packed.data
        scales = self.weight_scales.data.to(torch.float32)
        low = (packed & 0x0F).long()
        high = (packed >> 4).long()
        idx = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)
        w_norm = _NF4_LEVELS.to(idx.device)[idx]
        n_groups = scales.shape[1]
        w = (w_norm.reshape(self.out_features, n_groups, -1)
             * scales.unsqueeze(-1)).reshape(self.out_features, -1)
        w = w[:, :self.in_features]
        if cache:
            self._cached_weight = w.to(torch.bfloat16)
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x
        if (self._fused_gemv and _HAS_TRITON and x.is_cuda
                and self.weight_packed.is_cuda and flat.shape[0] == 1):
            out = self._fused_gemv_forward(x.reshape(-1))
            out = out.view(*x.shape[:-1], self.out_features)
            if self.lora_adapter is not None:
                out = out + self.lora_adapter(x)
            return out
        w = self._dequantize_weight(x.dtype, cache=self._cache_w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, bias)
        if self.lora_adapter is not None:
            out = out + self.lora_adapter(x)
        return out

    @torch.no_grad()
    def merge_lora(self) -> bool:
        if self.lora_adapter is None:
            return False
        lora = self.lora_adapter
        w = self._dequantize_weight(torch.float32, cache=False)
        delta = lora.scale * (lora.lora_B @ lora.lora_A)
        w = w + delta.to(torch.float32)
        self.load_from_weight(w)
        self.lora_adapter = None
        self._cached_weight = None
        return True

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"group_size={self.group_size}, nf4=True")

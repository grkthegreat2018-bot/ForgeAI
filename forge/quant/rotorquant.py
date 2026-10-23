"""RotorQuant — KV cache compression via block-diagonal rotations.

Implements the RotorQuant family of rotation-based KV cache quantizers from
the paper https://www.scrya.com/rotorquant.pdf (scrya-com/rotorquant).

Core idea: instead of a dense d×d random orthogonal transform (TurboQuant's
Walsh-Hadamard butterfly, O(d log d)), apply small block-diagonal rotations
that decorrelate KV cache vectors before scalar Lloyd-Max quantization.

This module ships the simplest production variant — **PlanarQuant** — which
uses 2D Givens rotations (SO(2)) per adjacent pair of dimensions:
    - O(d) per vector, fully parallelizable, no inter-element dependencies
    - 1 angle (cos θ, sin θ) per 2D pair — data-oblivious, fixed at init
    - Pure PyTorch (no Triton kernels needed)

Pipeline (per KV vector):
    1. Normalize  → store L2 norm separately
    2. Rotate     → fixed Givens rotation per 2D pair breaks coordinate correlations
    3. Quantize   → each coordinate mapped to nearest Lloyd-Max centroid (2^bits levels)
    4. Inverse rotate on decompress → rescale by stored norm

Deferred quantization (from the paper): the K cache is kept as FP16 during
prefill (zero error compounding, zero rotation overhead) and quantized
lazily on the first decode-token insertion. This gives ~3x better PPL than
roundtrip quantization and makes decode faster than the FP16 baseline.

Usage:
    from forge.quant.rotorquant import RotorQuantCache

    cache = RotorQuantCache(d_model=4096, n_heads=32, head_dim=128,
                            bits=4, max_seq_len=2048)
    # prefill (stored as FP16, deferred)
    cache.append(prefill_k, prefill_v)
    # decode (triggers quantization of prefill buffer + new token)
    cache.append(decode_k, decode_v)
    k, v = cache.get()  # approximate K/V for attention
"""
import math
import time

import torch

# ── Lloyd-Max scalar quantizer (pure PyTorch, no scipy) ─────────────


def _solve_lloyd_max_torch(d: int, bits: int, n_grid: int = 4096,
                           max_iter: int = 150, tol: float = 1e-9) -> torch.Tensor:
    """Solve Lloyd-Max optimal centroids for N(0, 1/d) coordinate distribution.

    After a random rotation of a d-dim unit vector, each coordinate is
    approximately Gaussian N(0, 1/d) for d >= 64. We run continuous 1-D
    k-means (Lloyd-Max) on a fine grid to find MSE-optimal centroids.

    Returns: (2^bits,) sorted float32 centroids.
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -5.0 * sigma, 5.0 * sigma
    x = torch.linspace(lo, hi, n_grid, dtype=torch.float64)
    pdf = torch.exp(-x * x / (2 * sigma * sigma))
    pdf = pdf / pdf.sum()

    # Uniform initialization in [-3σ, 3σ].
    centroids = torch.linspace(-3 * sigma, 3 * sigma, n_levels, dtype=torch.float64)
    for _ in range(max_iter):
        # Assign each grid point to nearest centroid.
        diffs = (x.unsqueeze(-1) - centroids).abs()  # (n_grid, n_levels)
        assign = diffs.argmin(dim=-1)  # (n_grid,)
        new_centroids = centroids.clone()
        for i in range(n_levels):
            mask = assign == i
            w = pdf[mask]
            if w.sum() > 0:
                new_centroids[i] = (x[mask] * w).sum() / w.sum()
        shift = (new_centroids - centroids).abs().max()
        centroids = new_centroids
        if shift < tol:
            break
    return centroids.to(torch.float32)


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given dimension and bit-width."""

    def __init__(self, d: int, bits: int):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.centroids = _solve_lloyd_max_torch(d, bits)
        # Boundaries = midpoints between adjacent sorted centroids.
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map values to nearest centroid indices via bucketize (fast)."""
        idx = torch.bucketize(x, self.boundaries)
        return idx.to(torch.int32)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(indices.device)[indices]


# ── 2D Givens rotation (PlanarQuant) ────────────────────────────────


def make_givens_rotations(n_groups: int, seed: int = 42,
                          device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate fixed random 2D rotation parameters (cos θ, sin θ) per group.

    Data-oblivious: angles are sampled once at init and never updated.
    Returns: (n_groups, 2) as [cos θ, sin θ].
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    angles = torch.rand(n_groups, generator=gen) * (2 * math.pi)
    return torch.stack([angles.cos(), angles.sin()], dim=-1).to(device=device, dtype=dtype)


def rot2_apply(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply 2D Givens rotation to pairs.

    cs: (..., 2) as [cos θ, sin θ]   (broadcastable)
    v:  (..., 2) as [v0, v1]
    Returns: (..., 2)  [c·v0 − s·v1, s·v0 + c·v1]
    """
    c = cs[..., 0]
    s = cs[..., 1]
    return torch.stack([c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1]], dim=-1)


def rot2_inverse(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply inverse 2D Givens rotation (transpose: negate sin).

    cs: (..., 2) as [cos θ, sin θ]
    v:  (..., 2) as [v0, v1]
    Returns: (..., 2)  [c·v0 + s·v1, −s·v0 + c·v1]
    """
    c = cs[..., 0]
    s = cs[..., 1]
    return torch.stack([c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1]], dim=-1)


# ── RotorQuantCache ─────────────────────────────────────────────────


class RotorQuantCache:
    """KV cache compression via PlanarQuant (2D Givens) + Lloyd-Max quantization.

    Drop-in replacement for ``KVQuantCache`` / ``CompressedKVCache`` in
    ``research.kv_compress.py``.

    Pipeline per KV vector:
        normalize → fixed Givens rotation per 2D pair → Lloyd-Max scalar
        quantization → (decompress: inverse rotate → rescale by norm)

    Deferred quantization: K/V are stored as FP16 during prefill and
    quantized lazily on the first decode-token insertion, matching the
    paper's deferred-K-cache design (3x better PPL than roundtrip).

    Works with both GQA (n_kv_heads < n_heads) and MLA: the cache only
    ever stores the KV-head tensors it receives via ``append``.

    Args:
        d_model:   model hidden dimension (kept for interface compatibility)
        n_heads:   number of query heads (kept for interface compatibility)
        head_dim:  dimension per head (the vector length that gets rotated/quantized)
        bits:      bits per coordinate for Lloyd-Max (3 or 4)
        max_seq_len: maximum sequence length (preallocated where possible)
        n_kv_heads: number of KV heads (defaults to n_heads; set smaller for GQA)
        attention_type: "gqa" or "mla" (affects nothing mathematically —
                        rotation/quantization is per-vector — but stored for
                        downstream consumers)
        seed:      random seed for the fixed Givens angles
        device:    torch device
        dtype:     compute dtype for rotations (float32 recommended for accuracy)
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 bits: int = 4, max_seq_len: int = 2048,
                 n_kv_heads: int | None = None,
                 attention_type: str = "gqa",
                 seed: int = 42, device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 bits_v: int | None = None,
                 layer_idx: int | None = None,
                 n_layers: int | None = None,
                 boundary_v_bits: int | None = None,
                 boundary_layers: int = 2):
        """TurboQuant+ extensions over the base PlanarQuant pipeline.

        Three drop-in upgrades from the TurboQuant+ fork findings
        (TheTom/turboquant_plus, vLLM PR#38479 merged):

          Asymmetric K/V (``bits_v``): K precision dominates error —
          quantizing K to 3 bits is catastrophic while V at 3 bits is
          nearly free, so ``bits`` applies to K and ``bits_v`` (default
          = ``bits``) to V.  E.g. bits=4, bits_v=3.

          Boundary-V (``layer_idx`` + ``n_layers`` + ``boundary_v_bits``):
          the first and last ``boundary_layers`` layers (default 2) keep
          V at a higher precision — the boundary layers carry outsized V
          error sensitivity; recovers most of the quality gap at ~no cost.
          Only applies when this cache is instantiated per layer.

          Sparse-V dequant: see ``get_sparse_v`` — positions whose
          attention weight is below a threshold skip V dequant entirely.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bits = bits
        self.bits_v = self._resolve_v_bits(
            bits, bits_v, layer_idx, n_layers,
            boundary_v_bits, boundary_layers)
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.attention_type = attention_type
        self.device = device
        self.dtype = dtype
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.boundary_v = (self.bits_v != (bits_v if bits_v is not None
                                           else bits))

        # Sparse-V dequant accounting (get_sparse_v)
        self._sparse_kept = 0
        self._sparse_skipped = 0

        # 2D pairs (pad head_dim to even if needed).
        self.n_groups = (head_dim + 1) // 2
        self.d_padded = self.n_groups * 2
        self.pad = self.d_padded - head_dim

        # Fixed data-oblivious Givens rotations: (n_groups, 2)
        self.rot2 = make_givens_rotations(self.n_groups, seed=seed,
                                          device=device, dtype=dtype)

        # Lloyd-Max codebooks — asymmetric K/V bits get separate tables.
        self.codebook_k = LloydMaxCodebook(d=head_dim, bits=bits)
        self._centroids_k = self.codebook_k.centroids.to(
            device=device, dtype=dtype)
        self._boundaries_k = self.codebook_k.boundaries.to(
            device=device, dtype=dtype)
        if self.bits_v == bits:
            self.codebook_v = self.codebook_k
            self._centroids_v = self._centroids_k
            self._boundaries_v = self._boundaries_k
        else:
            self.codebook_v = LloydMaxCodebook(d=head_dim, bits=self.bits_v)
            self._centroids_v = self.codebook_v.centroids.to(
                device=device, dtype=dtype)
            self._boundaries_v = self.codebook_v.boundaries.to(
                device=device, dtype=dtype)

        # --- Storage ---
        # Deferred prefill buffers (FP16). Non-None while in prefill mode.
        self._prefill_k: torch.Tensor | None = None  # (B, Hkv, T, D) fp16
        self._prefill_v: torch.Tensor | None = None

        # Quantized storage (after deferred conversion / decode insertion).
        # norms:   (B, Hkv, T)        fp16  — per-vector L2 norm
        # indices: (B, Hkv, T, D)     int32 — centroid index per coordinate
        self._k_norms: torch.Tensor | None = None
        self._k_idx: torch.Tensor | None = None
        self._v_norms: torch.Tensor | None = None
        self._v_idx: torch.Tensor | None = None

        self.current_length = 0
        self._quantized = False  # True once prefill buffer has been flushed to quantized

    @staticmethod
    def _resolve_v_bits(bits: int, bits_v: int | None,
                        layer_idx: int | None, n_layers: int | None,
                        boundary_v_bits: int | None,
                        boundary_layers: int) -> int:
        """Effective V bit-width: boundary layers get boundary_v_bits."""
        base = bits if bits_v is None else bits_v
        if (boundary_v_bits is not None and layer_idx is not None
                and n_layers is not None):
            if (layer_idx < boundary_layers
                    or layer_idx >= n_layers - boundary_layers):
                return boundary_v_bits
        return base

    # ── low-level quantize / dequantize for a single tensor ──

    def _quantize_tensor(self, x: torch.Tensor,
                         boundaries: torch.Tensor | None = None,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate + Lloyd-Max quantize a KV tensor.

        x: (B, Hkv, T, head_dim) float
        boundaries: codebook boundaries to quantize against (K or V table)
        Returns:
            norms:   (B, Hkv, T)        fp16
            indices: (B, Hkv, T, head_dim) int32
        """
        if boundaries is None:
            boundaries = self._boundaries_k
        x = x.to(self.dtype)
        norms = x.norm(dim=-1).clamp(min=1e-8)  # (B, Hkv, T)
        x_unit = x / norms.unsqueeze(-1)

        if self.pad > 0:
            x_unit = torch.nn.functional.pad(x_unit, (0, self.pad))
        v = x_unit.reshape(*x_unit.shape[:-1], self.n_groups, 2)  # (..., n_groups, 2)

        # Rotate each pair with the fixed Givens angle.
        v_rot = rot2_apply(self.rot2, v)  # (..., n_groups, 2)
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)  # (..., d_padded)
        flat = flat[..., :self.head_dim] if self.pad > 0 else flat

        # Scalar Lloyd-Max quantization per coordinate.
        idx = torch.bucketize(flat, boundaries).to(torch.int32)
        return norms.to(torch.float16), idx

    def _dequantize_tensor(self, norms: torch.Tensor, idx: torch.Tensor,
                           centroids: torch.Tensor | None = None,
                           ) -> torch.Tensor:
        """Inverse-rotate + rescale to reconstruct a KV tensor.

        norms:   (B, Hkv, T)        float
        idx:     (B, Hkv, T, head_dim) int
        centroids: codebook centroids to reconstruct from (K or V table)
        Returns: (B, Hkv, T, head_dim) float
        """
        if centroids is None:
            centroids = self._centroids_k
        flat = centroids.to(idx.device)[idx]  # (B, Hkv, T, D)

        if self.pad > 0:
            flat = torch.nn.functional.pad(flat, (0, self.pad))
        v = flat.reshape(*flat.shape[:-1], self.n_groups, 2)
        v_recon = rot2_inverse(self.rot2.to(v.device), v)
        out = v_recon.reshape(*v_recon.shape[:-2], -1)[..., :self.head_dim]
        return (out * norms.to(out.dtype).unsqueeze(-1)).to(torch.float16)

    # ── public API ──

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K, V tensors.

        k, v: (B, n_kv_heads, T_new, head_dim) float.

        Prefill chunks (T_new > 1) are buffered as FP16 (deferred quantization).
        Decode tokens (T_new == 1) flush the prefill buffer to quantized form
        and are appended in quantized form.
        """
        B, Hkv, T_new, D = k.shape
        assert self.head_dim == D, f"head_dim mismatch: got {D}, expected {self.head_dim}"
        assert Hkv == self.n_kv_heads, f"n_kv_heads mismatch: got {Hkv}, expected {self.n_kv_heads}"

        if T_new > 1:
            # Prefill: defer quantization, accumulate FP16.
            k_fp = k.to(torch.float16)
            v_fp = v.to(torch.float16)
            if self._prefill_k is None:
                self._prefill_k = k_fp
                self._prefill_v = v_fp
            else:
                self._prefill_k = torch.cat([self._prefill_k, k_fp], dim=2)
                self._prefill_v = torch.cat([self._prefill_v, v_fp], dim=2)
            self.current_length += T_new
            return

        # Decode token (T_new == 1).
        if not self._quantized and self._prefill_k is not None:
            # Flush deferred prefill buffer → quantized storage.
            kn, ki = self._quantize_tensor(self._prefill_k)
            vn, vi = self._quantize_tensor(self._prefill_v,
                                           self._boundaries_v)
            self._k_norms, self._k_idx = kn, ki
            self._v_norms, self._v_idx = vn, vi
            self._prefill_k = None
            self._prefill_v = None
            self._quantized = True

        kn, ki = self._quantize_tensor(k)
        vn, vi = self._quantize_tensor(v, self._boundaries_v)

        if self._quantized:
            self._k_norms = torch.cat([self._k_norms, kn], dim=2)
            self._k_idx = torch.cat([self._k_idx, ki], dim=2)
            self._v_norms = torch.cat([self._v_norms, vn], dim=2)
            self._v_idx = torch.cat([self._v_idx, vi], dim=2)
        else:
            # No prefill buffer — first append was a decode token.
            self._k_norms, self._k_idx = kn, ki
            self._v_norms, self._v_idx = vn, vi
            self._quantized = True

        self.current_length += T_new

    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return approximate (K, V) tensors for attention computation.

        Returns:
            k, v: (B, n_kv_heads, T, head_dim) float16, or (None, None) if empty.
        """
        if self._prefill_k is not None:
            return self._prefill_k, self._prefill_v
        if self._k_idx is None:
            return None, None
        k = self._dequantize_tensor(self._k_norms, self._k_idx)
        v = self._dequantize_tensor(self._v_norms, self._v_idx,
                                    self._centroids_v)
        return k, v

    def get_sparse_v(self, attn_weights: torch.Tensor,
                     threshold: float = 1e-6,
                     ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Dequantize V only at positions with attention mass ≥ threshold.

        Sparse-V (TurboQuant+): in a long cache ~90% of positions receive
        attention weight < 1e-6 — their contribution to ``attn @ V`` is
        numerically nil, so dequantizing their V rows is wasted work.
        Positions below ``threshold`` are returned as zero rows, which
        keeps the downstream weighted sum unchanged to within
        ``threshold * ||v||`` per position.

        Args:
            attn_weights: (..., T) post-softmax attention weights over the
                cached positions — the max over leading dims decides keep.
            threshold: skip positions whose max weight is below this.

        Returns:
            (v_hat, keep_mask): v_hat (B, n_kv, T, head_dim) with zero
            rows at skipped positions; keep_mask (T,) bool.
        """
        if self._prefill_v is not None:
            v_full = self._prefill_v
            T = v_full.shape[2]
            keep = torch.ones(T, dtype=torch.bool,
                              device=v_full.device)
            if attn_weights is not None:
                w = attn_weights.reshape(-1, attn_weights.shape[-1])
                keep = w.amax(dim=0)[:T] >= threshold
            self._sparse_kept += int(keep.sum())
            self._sparse_skipped += int((~keep).sum())
            if not bool(keep.all()):
                v_full = v_full.clone()
                v_full[:, :, ~keep] = 0
            return v_full, keep
        if self._v_idx is None:
            return None, torch.zeros(0, dtype=torch.bool,
                                     device=self.device)
        T = self._v_idx.shape[2]
        if attn_weights is None:
            keep = torch.ones(T, dtype=torch.bool, device=self._v_idx.device)
        else:
            w = attn_weights.reshape(-1, attn_weights.shape[-1])
            keep = (w.amax(dim=0)[:T] >= threshold).to(self._v_idx.device)
        kept_n = int(keep.sum())
        self._sparse_kept += kept_n
        self._sparse_skipped += T - kept_n
        if kept_n == T:
            # Nothing skipped — dequantize everything.
            return (self._dequantize_tensor(
                        self._v_norms, self._v_idx, self._centroids_v),
                    keep)
        # The actual win: skipped positions never touch the inverse
        # rotation / codebook gather — only surviving rows are dequantized.
        v_hat = torch.zeros(
            *self._v_idx.shape[:2], T, self.head_dim,
            dtype=torch.float16, device=self._v_idx.device)
        if kept_n:
            v_sel = self._dequantize_tensor(
                self._v_norms[:, :, keep],
                self._v_idx[:, :, keep],
                self._centroids_v)
            v_hat[:, :, keep] = v_sel
        return v_hat, keep

    def compress(self, k: torch.Tensor, v: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stateless compression of full K/V tensors.

        k, v: (B, n_kv_heads, T, head_dim) float
        Returns dict with norms + centroid indices for K and V.
        """
        kn, ki = self._quantize_tensor(k)
        vn, vi = self._quantize_tensor(v, self._boundaries_v)
        return {
            "k_norms": kn, "k_idx": ki,
            "v_norms": vn, "v_idx": vi,
            "bits": self.bits,
            "bits_v": self.bits_v,
            "head_dim": self.head_dim,
            "n_kv_heads": self.n_kv_heads,
        }

    def decompress(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return approximate K/V from internal quantized storage (alias of get)."""
        return self.get()

    def decompress_dict(self, comp: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress a dict produced by ``compress``."""
        k = self._dequantize_tensor(comp["k_norms"], comp["k_idx"])
        v = self._dequantize_tensor(comp["v_norms"], comp["v_idx"],
                                    self._centroids_v)
        return k, v

    def clear(self):
        """Reset the cache to empty (drops prefill buffer + quantized store)."""
        self._prefill_k = None
        self._prefill_v = None
        self._k_norms = None
        self._k_idx = None
        self._v_norms = None
        self._v_idx = None
        self.current_length = 0
        self._quantized = False

    def __len__(self) -> int:
        return self.current_length

    # ── memory / compression accounting ──

    def memory_usage_mb(self) -> float:
        """Estimate current memory usage in MB."""
        total = 0.0
        if self._prefill_k is not None:
            total += self._prefill_k.numel() * 2 + self._prefill_v.numel() * 2
        if self._k_idx is not None:
            # norms: fp16 (2 bytes); indices: int32 (4 bytes) — could pack to bits/coord
            total += self._k_norms.numel() * 2 + self._k_idx.numel() * 4
            total += self._v_norms.numel() * 2 + self._v_idx.numel() * 4
        return total / (1024 * 1024)

    def compression_ratio(self) -> float:
        """Compression ratio vs FP16 KV cache (theoretical, bit-packed)."""
        if self.current_length == 0:
            return 1.0
        # FP16 baseline: 2 bytes * 2 (K+V) * Hkv * D * T
        fp16_bytes = 2 * 2 * self.n_kv_heads * self.head_dim * self.current_length
        # Compressed: bits/coord packed + 2 bytes norm per vector.
        # Asymmetric K/V (TurboQuant+): K uses self.bits, V self.bits_v.
        packed_bytes = ((self.bits + self.bits_v)
                        * self.n_kv_heads * self.head_dim
                        * self.current_length) / 8
        norm_bytes = 2 * 2 * self.n_kv_heads * self.current_length
        actual = packed_bytes + norm_bytes
        return fp16_bytes / max(actual, 1)

    def info(self) -> dict:
        return {"type": "rotorquant", "seq_len": self.current_length,
                "bits_k": self.bits, "bits_v": self.bits_v,
                "boundary_v": self.boundary_v,
                "sparse_v_kept": self._sparse_kept,
                "sparse_v_skipped": self._sparse_skipped,
                "compression": self.compression_ratio()}

    def is_quantized(self) -> bool:
        """True if the cache is in quantized mode (prefill flushed)."""
        return self._quantized


# ── Benchmark ───────────────────────────────────────────────────────


def benchmark(d_model: int = 4096, n_heads: int = 32, head_dim: int = 128,
              bits: int = 4, prefill_len: int = 512, decode_steps: int = 128,
              device: str = "cpu", n_kv_heads: int | None = None):
    """Benchmark RotorQuant compression ratio, speed, and reconstruction error.

    Prints:
      - compression ratio (theoretical bit-packed vs FP16)
      - prefill append time (FP16 deferred — should be near-zero overhead)
      - decode append time per token (includes quantization)
      - compress + decompress roundtrip time
      - reconstruction MSE / relative error
    """
    n_kv_heads = n_kv_heads or n_heads
    torch.manual_seed(0)
    k_full = torch.randn(1, n_kv_heads, prefill_len, head_dim, device=device)
    v_full = torch.randn(1, n_kv_heads, prefill_len, head_dim, device=device)

    cache = RotorQuantCache(d_model=d_model, n_heads=n_heads, head_dim=head_dim,
                            bits=bits, max_seq_len=prefill_len + decode_steps,
                            n_kv_heads=n_kv_heads, device=device)

    # Prefill (deferred FP16).
    t0 = time.perf_counter()
    cache.append(k_full, v_full)
    prefill_time = time.perf_counter() - t0

    # Decode tokens.
    decode_k = torch.randn(1, n_kv_heads, 1, head_dim, device=device)
    decode_v = torch.randn(1, n_kv_heads, 1, head_dim, device=device)
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        cache.append(decode_k, decode_v)
    decode_time = time.perf_counter() - t0

    # Roundtrip compress/decompress.
    t0 = time.perf_counter()
    comp = cache.compress(k_full, v_full)
    compress_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    k_hat, v_hat = cache.decompress_dict(comp)
    decompress_time = time.perf_counter() - t0

    # Error metrics.
    k_err = (k_full.to(k_hat.device) - k_hat).pow(2).mean().item()
    k_rel = k_err / k_full.pow(2).mean().item()
    v_err = (v_full.to(v_hat.device) - v_hat).pow(2).mean().item()
    v_rel = v_err / v_full.pow(2).mean().item()

    print("=" * 64)
    print("RotorQuant (PlanarQuant) benchmark")
    print("=" * 64)
    print(f"  config:        head_dim={head_dim}, bits={bits}, "
          f"n_kv_heads={n_kv_heads}, prefill={prefill_len}, decode={decode_steps}")
    print(f"  device:        {device}")
    print(f"  cache length:  {len(cache)}")
    print(f"  quantized:     {cache.is_quantized()}")
    print(f"  compression:   {cache.compression_ratio():.2f}x (theoretical, vs FP16)")
    print(f"  memory:        {cache.memory_usage_mb():.3f} MB")
    print(f"  prefill time:  {prefill_time*1000:.2f} ms (deferred FP16, {prefill_len} tokens)")
    print(f"  decode time:   {decode_time*1000:.2f} ms total, "
          f"{decode_time/decode_steps*1000:.3f} ms/token")
    print(f"  compress:      {compress_time*1000:.2f} ms")
    print(f"  decompress:    {decompress_time*1000:.2f} ms")
    print(f"  K MSE:         {k_err:.6f}  (relative {k_rel*100:.2f}%)")
    print(f"  V MSE:         {v_err:.6f}  (relative {v_rel*100:.2f}%)")
    print("=" * 64)
    return {
        "compression_ratio": cache.compression_ratio(),
        "prefill_ms": prefill_time * 1000,
        "decode_ms_per_token": decode_time / decode_steps * 1000,
        "k_relative_error": k_rel,
        "v_relative_error": v_rel,
    }


if __name__ == "__main__":
    # CPU benchmark (small) + GPU benchmark if available.
    benchmark(d_model=4096, n_heads=32, head_dim=128, bits=4,
              prefill_len=512, decode_steps=64, device="cpu")
    if torch.cuda.is_available():
        benchmark(d_model=4096, n_heads=32, head_dim=128, bits=4,
                  prefill_len=512, decode_steps=128, device="cuda",
                  n_kv_heads=8)  # GQA: 32 query heads, 8 KV heads

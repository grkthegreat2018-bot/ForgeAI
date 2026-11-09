"""StreamHead ring internals: int4 shard codec + pinned-CPU shard cache.

Split from airllm_streamer.py — these are the self-contained codec/cache
primitives; the streamer keeps orchestration (plan/issue/bind/generate).

- ``_pack_i4`` / ``_dequant_i4``: g32 minmax int4 packing for shard files
  (opt-in via FORGE_AIRLLM_INT4=1 — bf16 wins on PCIe 5.0, see streamer doc).
- ``_PinnedCache``: whole-shard pinned-CPU cache — each layer shard is read
  from disk once, then every decode step streams from pinned memory
  (~53 GB/s measured vs ~13 GB/s pageable — the core fix).
"""
from __future__ import annotations

import torch

_INT4_GROUP = 32             # minmax group size for int4 shards


def _pack_i4(t: torch.Tensor, g: int = _INT4_GROUP):
    """g32 minmax int4 pack -> (q u8[ng*g/2], mn bf16[ng], mx bf16[ng], tail bf16)."""
    w = t.float().reshape(-1)
    n = w.numel() // g * g
    wg = w[:n].reshape(-1, g)
    mn = wg.min(1, keepdim=True).values
    mx = wg.max(1, keepdim=True).values
    q = ((wg - mn) / (mx - mn).clamp(min=1e-8) * 15).round().clamp(0, 15).to(torch.uint8)
    q = q[:, 0::2] | (q[:, 1::2] << 4)
    tail = w[n:].to(torch.bfloat16)
    return (q.reshape(-1), mn.reshape(-1).to(torch.bfloat16),
            mx.reshape(-1).to(torch.bfloat16), tail, t.shape, n)


def _dequant_i4_core(q, mn, mx, g: int):
    """Fused dequant body — interleaved-nibble unpack, one elementwise chain,
    emits bf16 directly. torch.compile fuses this to ~1-2 passes."""
    ng = mn.numel()
    packed = q[: ng * g // 2].reshape(ng, g // 2)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=2).reshape(ng, g)
    return (mn.float().unsqueeze(1)
            + codes.float() / 15.0
            * (mx.float() - mn.float()).unsqueeze(1)).to(torch.bfloat16)


_dequant_i4_fused = None
_dequant_compile_failed = False


def _dequant_i4(q, mn, mx, tail, shape, n, device, g=_INT4_GROUP):
    """GPU dequant of _pack_i4 output -> bf16 tensor shaped `shape` (or flat
    when shape is None). Uses a compiled fused kernel on CUDA when possible."""
    global _dequant_i4_fused, _dequant_compile_failed
    if isinstance(device, str):
        device = torch.device(device)
    ng = mn.numel()
    w = torch.empty(0, dtype=torch.float32, device=device)
    if ng:
        if device.type == "cuda" and not _dequant_compile_failed:
            try:
                if _dequant_i4_fused is None:
                    _dequant_i4_fused = torch.compile(_dequant_i4_core, dynamic=False)
                w = _dequant_i4_fused(q, mn, mx, g).reshape(-1)
            except Exception:
                _dequant_compile_failed = True
        if not ng or w.numel() == 0:
            w = _dequant_i4_core(q, mn, mx, g).reshape(-1)
    if tail.numel():
        w = torch.cat([w, tail])
    w = w[:n]
    out = w.reshape(shape) if shape is not None else w
    return out


class _PinnedCache:
    """Whole-shard pinned-CPU cache: each layer shard is read from disk once,
    then every decode step streams from pinned memory (~53 GB/s measured vs
    ~13 GB/s pageable — the core fix)."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.cache: dict[int, dict[str, torch.Tensor]] = {}

    def get(self, li: int) -> dict[str, torch.Tensor]:
        if li in self.cache:
            return self.cache[li]
        from safetensors.torch import load_file
        state = load_file(str(self.paths[li]))
        if torch.cuda.is_available():
            state = {k: v.pin_memory() for k, v in state.items()}
        self.cache[li] = state
        return state


def _shard_sort_key(p) -> int:
    """Numeric sort for shard_N.safetensors / shard_N.i4.safetensors —
    lexicographic order scrambles layers at N>=10 (HF #32023 class of bug)."""
    return int(p.stem.split("_")[1].split(".")[0])

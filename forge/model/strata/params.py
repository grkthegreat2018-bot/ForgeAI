"""ParamSet - the IQ layer (.iq).

Named tensors only - the small fixed compute machinery that turns
cell votes into logits (mixer weights, fingerprint table, optional
dense channel).  Seeded tensors (e.g. the _R randproj fingerprint
bank) are stored as {"seed": ...} derivations, not bytes - a 4 MB
table costs 8 bytes on disk and regenerates bit-exact.

Sub-bit storage: learned tensors can be packed at 1/2/4 bits per
element (``set_packed``) - group-wise fp16 scales + int32-packed
codes.  bits=1 is binary sign coding (scale = group mean|w|),
bits=2 covers int2/ternary (codes -2..1, {-1,0,1} is a subset),
bits=4 = symmetric int4.  Scales make this quantization, not
truncation: relative error stays bounded per group.

Behavior rule: params decide HOW info is used, never WHAT is stored.
Swapping the .iq file upgrades the brain without touching memory.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .io import save_container, load_container


# ---- sub-bit packing ------------------------------------------------------

def _pack_vals(q: torch.Tensor, bits: int) -> torch.Tensor:
    """uint codes [0, 2**bits) -> int32 words (32/bits per word)."""
    vpw = 32 // bits
    q = q.reshape(-1)
    pad = (-q.numel()) % vpw
    if pad:
        q = torch.cat([q, q.new_zeros(pad)])
    q = q.reshape(-1, vpw).long()
    shifts = torch.arange(vpw, device=q.device) * bits
    return (q << shifts).sum(1).to(torch.int32)


def _unpack_vals(w: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """int32 words -> n uint codes (inverse of _pack_vals)."""
    vpw = 32 // bits
    shifts = torch.arange(vpw, device=w.device) * bits
    q = ((w.long()[:, None] >> shifts) & ((1 << bits) - 1))
    return q.reshape(-1)[:n].to(torch.int32)


# E2M1 levels (NVFP4-native, in-tree via use_nvfp4): 4-bit float,
# denser near zero -> fits gaussian weights better than uniform int4.
_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_BOUNDS = ((_FP4[:-1] + _FP4[1:]) / 2)    # decision midpoints


def quantize_param(t: torch.Tensor, bits: int = 4, group: int = 64):
    """-> (words int32, scales fp16 [n_groups], meta).

    bits=1: binary sign code, scale = group mean|w| (BitNet-style).
    bits=2: symmetric int2 {-2..1} (ternary {-1,0,1} is a subset).
    bits=4: FP4 E2M1 magnitude code + sign bit (default - matches the
            NVFP4 storage format used elsewhere in the codebase)."""
    flat = t.detach().float().reshape(-1)
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.reshape(-1, group)
    if bits == 1:
        s = g.abs().mean(1).clamp_min(1e-8)
        code = (g >= 0).to(torch.int32)
    elif bits == 2:
        # BitNet-b1.58 ternary: scale by mean|w|, round, clamp to
        # {-1,0,1} -> codes {0,1,2}.  Symmetric int2 (-2..1) wastes a
        # code and scales by absmax - ~2x worse MSE on gaussian.
        s = g.abs().mean(1).clamp_min(1e-8)
        code = (torch.round(g / s[:, None]).clamp(-1, 1) + 1) \
            .to(torch.int32)
    elif bits == 4:
        s = (g.abs().amax(1) / 6.0).clamp_min(1e-8)   # E2M1 max = 6
        mag = (g / s[:, None]).abs()
        idx = torch.bucketize(mag, _FP4_BOUNDS.to(g.device))
        code = ((g < 0).to(torch.int32) << 3) | idx.to(torch.int32)
    else:
        qmax = (1 << (bits - 1)) - 1          # int2: 1
        s = (g.abs().amax(1) / qmax).clamp_min(1e-8)
        code = (torch.round(g / s[:, None])
                .clamp(-qmax - 1, qmax) + qmax + 1).to(torch.int32)
    meta = {"bits": bits, "group": group,
            "shape": list(t.shape), "numel": t.numel()}
    return _pack_vals(code, bits), s.to(torch.float16), meta


def dequantize_param(words: torch.Tensor, scales: torch.Tensor,
                     meta: dict) -> torch.Tensor:
    """Inverse of quantize_param -> fp32 tensor of meta["shape"]."""
    bits, group = meta["bits"], meta["group"]
    n_codes = words.numel() * (32 // bits)
    q = _unpack_vals(words, bits, n_codes).reshape(-1, group).float()
    if bits == 1:
        w = (q * 2 - 1) * scales.float()[:, None]
    elif bits == 2:
        w = (q - 1) * scales.float()[:, None]
    elif bits == 4:
        lut = _FP4.to(words.device)
        w = (torch.where(q >= 8, -1.0, 1.0)
             * lut[(q.long() & 7).long()] * scales.float()[:, None])
    else:
        qmax = (1 << (bits - 1)) - 1
        w = (q - (qmax + 1)) * scales.float()[:, None]
    return w.reshape(-1)[:meta["numel"]].reshape(meta["shape"])



class ParamSet:
    def __init__(self):
        self.tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.derived: dict = {}   # name -> {"seed":int,"shape":[],
                                  #         "dtype":str}
        self.packed: dict = {}    # name -> quantize_param meta; the
                                  # bytes live in tensors as
                                  # "name::w" (int32) / "name::s" (fp16)
        self.version: int = 1

    def set(self, name: str, t: torch.Tensor) -> None:
        self.tensors[name] = t.detach()

    def set_packed(self, name: str, t: torch.Tensor, bits: int = 4,
                   group: int = 64, residual: int = 1) -> dict:
        """Store a learned tensor at sub-byte width.  ``residual``>1
        stacks correction levels: W ~= sum of dequantized levels, each
        packed from the previous level's error (NLRQ residual trick).
        Error compounds down ~multiplicatively per level: fp4 x2 lands
        ~1% rel err at 1B/param; fp4 x3 ~0.1%.  Returns meta."""
        res = t.detach().float()
        for i in range(residual):
            w, s, m = quantize_param(res, bits, group)
            self.tensors[f"{name}::w{i}"] = w
            self.tensors[f"{name}::s{i}"] = s
            if i + 1 < residual:
                res = res - dequantize_param(w, s, m)
        self.packed[name] = {"bits": bits, "group": group,
                             "shape": list(t.shape),
                             "numel": t.numel(), "residual": residual}
        return self.packed[name]

    def set_seeded(self, name: str, shape, seed: int,
                   dtype: str = "float32") -> torch.Tensor:
        self.derived[name] = {"seed": int(seed), "shape": list(shape),
                              "dtype": dtype}
        return self._derive(name)

    def _derive(self, name: str) -> torch.Tensor:
        d = self.derived[name]
        g = torch.Generator().manual_seed(d["seed"])
        t = torch.randn(d["shape"], generator=g)
        if d["dtype"] == "int8":
            t = (t * 24).round().clamp(-127, 127).to(torch.int8)
        return t

    def get(self, name: str, device=None) -> torch.Tensor:
        if name in self.packed:
            meta = self.packed[name]
            t = dequantize_param(self.tensors[name + "::w0"],
                                 self.tensors[name + "::s0"], meta)
            for i in range(1, meta.get("residual", 1)):
                t = t + dequantize_param(self.tensors[f"{name}::w{i}"],
                                         self.tensors[f"{name}::s{i}"],
                                         meta)
        elif name in self.tensors:
            t = self.tensors[name]
        elif name in self.derived:
            t = self._derive(name)
        else:
            raise KeyError(name)
        return t.to(device) if device is not None else t

    def names(self) -> list:
        base = [n for n in self.tensors if "::" not in n]
        return (base + list(self.packed)
                + [n for n in self.derived if n not in self.tensors])

    def stored_bytes(self) -> int:
        """Total slab bytes on disk (packed tensors counted packed)."""
        return sum(t.numel() * t.element_size()
                   for t in self.tensors.values())

    # -- io -----------------------------------------------------------------

    def save(self, path) -> int:
        meta = {"kind": "params", "version": self.version,
                "derived": self.derived, "packed": self.packed}
        return save_container(path, self.tensors, meta)

    @classmethod
    def load(cls, path, device: str | None = None) -> "ParamSet":
        meta, tensors = load_container(path, device=device)
        p = cls()
        p.version = meta.get("version", 1)
        p.derived = meta.get("derived", {})
        p.packed = meta.get("packed", {})
        p.tensors = OrderedDict(tensors)
        return p

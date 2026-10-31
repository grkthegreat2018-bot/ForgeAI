"""Evolution target: grinqh._dequantize_4bit.

INT4 (sign|3-bit-mag) weight dequantizer on the forward hot path. Seed
burns an int64 n_padded scratch, per-call _4BIT_MAGNITUDES.to(device),
and stack/reshape for nibble unpack. Goal: numerically identical
output (fp noise ok), lower ms_per_call.
"""

NAME = "dequantize_4bit"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["quant", "kernel", "hot-path"]
BUDGETS = {}
REQUIRED = ["_dequantize_4bit"]
PROMPT = (
    "Faster 4-bit weight dequantizer. Format: packed uint8, 2 codes/byte "
    "(lo nibble first); code = sign<<3 | mag_idx (0..6); w = sign*mag*scale "
    "per group of group_size. Seed allocates int64 scratch, stack+reshape "
    "to unpack nibbles, and calls mags.to(device) every call. Ideas: "
    "uint8/int8 arith instead of int64, single index_select/view for "
    "unpack, cache the magnitude table per-device, fuse sign into table "
    "lookup (16-entry signed table indexed by raw 4-bit code). Output "
    "must match seed to fp tolerance."
)

SEED_CODE = '''
import torch
import torch.nn.functional as F

_4BIT_MAGNITUDES = torch.tensor(
    [1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7, 1.0],
    dtype=torch.float32,
)


def _dequantize_4bit(packed: torch.Tensor, scales: torch.Tensor,
                     n: int, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    n_padded = scales.numel() * group_size
    device = packed.device
    code_flat = torch.zeros(n_padded, dtype=torch.int64, device=device)
    n_even = (n_padded // 2) * 2
    if n_even > 0:
        p = packed[:n_even // 2].to(torch.int64)
        c0 = p & 0x0F
        c1 = (p >> 4) & 0x0F
        codes = torch.stack([c0, c1], dim=1).reshape(-1)
        code_flat[:n_even] = codes
    if n_padded % 2 == 1:
        code_flat[-1] = int(packed[-1]) & 0x0F

    sign_bit = (code_flat >> 3) & 1
    mag_idx = code_flat & 0x07
    mags = _4BIT_MAGNITUDES.to(device)
    magnitude = mags[mag_idx.clamp(0, 6)]
    w_norm = torch.where(sign_bit.bool(), -magnitude, magnitude)
    n_groups = scales.numel()
    w_norm = w_norm.view(n_groups, group_size)
    w = w_norm * scales.to(torch.float32).unsqueeze(1)
    w = w.view(-1)[:n].to(dtype)
    return w
'''


def _inputs():
    import torch, math
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    cases = []
    for n, gs in ((8192, 128), (8192, 64), (4097, 128), (1536, 32),
                  (20480, 128)):
        n_groups = math.ceil(n / gs)
        n_packed = (n_groups * gs + 1) // 2
        packed = torch.randint(0, 256, (n_packed,), dtype=torch.uint8,
                               generator=g).to(dev)
        scales = (torch.rand(n_groups, generator=g) * 0.5 +
                  0.05).to(torch.float16).to(dev)
        for dt in (torch.float32, torch.bfloat16):
            cases.append((packed, scales, n, gs, dt))
    return cases


def _ref():
    import torch

    MAG = torch.tensor(
        [1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7, 1.0],
        dtype=torch.float32)

    def deq_ref(packed, scales, n, group_size, dtype):
        n_padded = scales.numel() * group_size
        device = packed.device
        code_flat = torch.zeros(n_padded, dtype=torch.int64, device=device)
        n_even = (n_padded // 2) * 2
        if n_even > 0:
            p = packed[:n_even // 2].to(torch.int64)
            codes = torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=1).reshape(-1)
            code_flat[:n_even] = codes
        if n_padded % 2 == 1:
            code_flat[-1] = int(packed[-1]) & 0x0F
        sign_bit = (code_flat >> 3) & 1
        mag_idx = code_flat & 0x07
        magnitude = MAG.to(device)[mag_idx.clamp(0, 6)]
        w_norm = torch.where(sign_bit.bool(), -magnitude, magnitude)
        w_norm = w_norm.view(scales.numel(), group_size)
        w = w_norm * scales.to(torch.float32).unsqueeze(1)
        return w.view(-1)[:n].to(dtype)

    return deq_ref


def test(ns):
    import torch
    ref = _ref()
    fn = ns.get("_dequantize_4bit")
    if not callable(fn):
        return {"ok": False, "detail": "no _dequantize_4bit"}
    for i, (pk, sc, n, gs, dt) in enumerate(_inputs()):
        try:
            y, y0 = fn(pk, sc, n, gs, dt), ref(pk, sc, n, gs, dt)
        except Exception as e:
            return {"ok": False,
                    "detail": f"input {i} raised {type(e).__name__}: {e}"}
        if y.shape != y0.shape:
            return {"ok": False, "detail": f"input {i} shape {y.shape}"}
        if not torch.isfinite(y).all():
            return {"ok": False, "detail": f"input {i} nonfinite"}
        if not torch.allclose(y.float(), y0.float(), atol=2e-3, rtol=2e-3):
            return {"ok": False, "detail": f"input {i} mismatch"}
    return {"ok": True}


def bench(ns):
    import time, torch
    fn = ns["_dequantize_4bit"]
    cases = _inputs()
    for pk, sc, n, gs, dt in cases:
        fn(pk, sc, n, gs, dt)
    dev = cases[0][0].device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        for pk, sc, n, gs, dt in cases:
            fn(pk, sc, n, gs, dt)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return {"ms_per_call": (time.perf_counter() - t0) / (reps * len(cases)) * 1000}

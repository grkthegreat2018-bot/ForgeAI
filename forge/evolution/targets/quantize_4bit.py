"""Evolution target: grinqh._quantize_4bit.

Per-row INT4 (sign|3-bit mag) packer — runs on every weight row at
quantize time. Seed builds a boundaries tensor per call, searchsorted
+ per-element Python fallback for odd tails, int64 dance on pack.
Gate: packed bytes AND scales must match the reference bit-for-bit
(int paths are exact; scales fp16 -> allclose with 0 tol = equal).
"""

import random
NAME = "quantize_4bit"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["quant", "kernel", "pack"]
BUDGETS = {}
REQUIRED = ["_quantize_4bit"]
PROMPT = (
    "Faster INT4 symmetric quantizer for a 1D weight row. Format: "
    "code = sign<<3 | mag_idx(0..6) over boundaries of |w|/absmax at "
    "k/7; pack 2 nibbles/byte (lo first); odd tail -> last byte lo "
    "nibble. Seed: per-call boundaries tensor + searchsorted + int64 "
    "stack/shift pack + python int() tail. Ideas: compare directly "
    "against absmax*k (skip the division), bit-pack via "
    "uint8 view tricks, vectorized tail. packed bytes must equal ref "
    "EXACTLY, scales to fp tolerance."
)

SEED_CODE = '''
import torch
import torch.nn.functional as F


def _quantize_4bit(w_row: torch.Tensor, group_size: int):
    n = w_row.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        w_row = F.pad(w_row, (0, pad))
    n_padded = w_row.numel()
    n_groups = n_padded // group_size
    wg = w_row.view(n_groups, group_size)

    absmax = wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax
    w_norm = wg / scale.clamp(min=1e-12)

    abs_norm = w_norm.abs()
    boundaries = torch.tensor(
        [0.5 / 7, 1.5 / 7, 2.5 / 7, 3.5 / 7, 4.5 / 7, 5.5 / 7, 6.5 / 7],
        dtype=torch.float32, device=w_row.device,
    )
    mag_idx = torch.searchsorted(boundaries, abs_norm).clamp(0, 6)
    sign_bit = (w_norm < 0).long()
    code = (sign_bit << 3) | mag_idx

    code_flat = code.view(-1).to(torch.int64)
    n_packed = (n_padded + 1) // 2
    packed = torch.zeros(n_packed, dtype=torch.uint8, device=w_row.device)
    n_even = (n_padded // 2) * 2
    if n_even > 0:
        c2 = code_flat[:n_even].view(-1, 2)
        packed[:n_even // 2] = ((c2[:, 0] & 0x0F) | ((c2[:, 1] << 4) & 0xF0)).to(torch.uint8)
    if n_padded % 2 == 1:
        packed[-1] = int(code_flat[-1]) & 0x0F

    scales = scale.squeeze(1).to(torch.float16)
    return packed, scales
'''


def _inputs():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(random.SystemRandom().getrandbits(63))
    cases = []
    for n, gs in ((8192, 128), (8192, 64), (4097, 128), (1536, 32),
                  (20480, 128), (300, 32)):
        w = torch.randn(n, generator=g).to(dev) * 0.05
        cases.append((w, gs))
    return cases


def _ref():
    import torch
    import torch.nn.functional as F

    def q_ref(w_row, group_size):
        n = w_row.numel()
        pad = (group_size - n % group_size) % group_size
        if pad > 0:
            w_row = F.pad(w_row, (0, pad))
        n_padded = w_row.numel()
        n_groups = n_padded // group_size
        wg = w_row.view(n_groups, group_size)
        absmax = wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = absmax
        w_norm = wg / scale.clamp(min=1e-12)
        abs_norm = w_norm.abs()
        boundaries = torch.tensor(
            [0.5 / 7, 1.5 / 7, 2.5 / 7, 3.5 / 7, 4.5 / 7, 5.5 / 7,
             6.5 / 7], dtype=torch.float32, device=w_row.device)
        mag_idx = torch.searchsorted(boundaries, abs_norm).clamp(0, 6)
        sign_bit = (w_norm < 0).long()
        code = (sign_bit << 3) | mag_idx
        code_flat = code.view(-1).to(torch.int64)
        n_packed = (n_padded + 1) // 2
        packed = torch.zeros(n_packed, dtype=torch.uint8,
                             device=w_row.device)
        n_even = (n_padded // 2) * 2
        if n_even > 0:
            c2 = code_flat[:n_even].view(-1, 2)
            packed[:n_even // 2] = ((c2[:, 0] & 0x0F)
                                    | ((c2[:, 1] << 4) & 0xF0)).to(
                                        torch.uint8)
        if n_padded % 2 == 1:
            packed[-1] = int(code_flat[-1]) & 0x0F
        return packed, scale.squeeze(1).to(torch.float16)

    return q_ref


def test(ns):
    import torch
    ref = _ref()
    fn = ns.get("_quantize_4bit")
    if not callable(fn):
        return {"ok": False, "detail": "no _quantize_4bit"}
    for i, (w, gs) in enumerate(_inputs()):
        try:
            pk, sc = fn(w, gs)
            pk0, sc0 = ref(w, gs)
        except Exception as e:
            return {"ok": False,
                    "detail": f"input {i} raised {type(e).__name__}: {e}"}
        if not torch.equal(pk, pk0):
            return {"ok": False, "detail": f"input {i} packed mismatch"}
        if not torch.allclose(sc.float(), sc0.float(),
                              atol=1e-3, rtol=1e-3):
            return {"ok": False, "detail": f"input {i} scales mismatch"}
    return {"ok": True}


def bench(ns):
    import time, torch
    fn = ns["_quantize_4bit"]
    cases = _inputs()
    for w, gs in cases:
        fn(w, gs)
    dev = cases[0][0].device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        for w, gs in cases:
            fn(w, gs)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return {"ms_per_call": (time.perf_counter() - t0) / (reps * len(cases)) * 1000}

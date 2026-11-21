"""int4 PreAllocatedKVCache: pack/unpack, running per-channel K scale,
dequant-aware kv_slice (which also covers the fixed int8 read path)."""
import torch

from forge.model.kv_cache import PreAllocatedKVCache

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _mk(q="int4", cap=64, hd=64):
    return PreAllocatedKVCache(
        2, 1, 2, cap, hd, torch.bfloat16, torch.device(DEV), quantize=q)


def test_pack_unpack_roundtrip():
    q = torch.randint(-8, 8, (1, 2, 5, 64), dtype=torch.int8, device=DEV)
    p = PreAllocatedKVCache._pack4(q)
    assert p.shape == (1, 2, 5, 32) and p.dtype == torch.uint8
    assert torch.equal(PreAllocatedKVCache._unpack4(p, 64), q)


def test_int4_bounded_error():
    torch.manual_seed(0)
    c = _mk()
    k = torch.randn(1, 2, 40, 64, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 40, 64, device=DEV, dtype=torch.bfloat16)
    c.append(0, k, v)
    c.advance(40)
    kr, vr = c.kv_slice(0, 40)
    s_k = k.abs().amax(2, keepdim=True) / 7.0
    s_v = v.abs().amax(-1, keepdim=True) / 7.0
    tol_k = s_k / 2 + k.abs() * 0.004 + 1e-4  # bf16 output rounding
    tol_v = s_v / 2 + v.abs() * 0.004 + 1e-4
    assert bool(((kr - k).abs() <= tol_k).all())
    assert bool(((vr - v).abs() <= tol_v).all())


def test_int4_memory_ratio():
    c4 = _mk("int4")
    c0 = _mk("none")
    assert c4.cache_memory_mb() < c0.cache_memory_mb() / 3.5


def test_int4_running_scale_requant():
    # Second append with much larger magnitudes grows the channel absmax:
    # stored entries must be requantized, not corrupted.
    torch.manual_seed(1)
    c = _mk()
    k1 = torch.randn(1, 2, 20, 64, device=DEV, dtype=torch.bfloat16) * 0.1
    c.append(0, k1, torch.zeros(1, 2, 20, 64, device=DEV))
    c.advance(20)
    k2 = torch.randn(1, 2, 5, 64, device=DEV, dtype=torch.bfloat16) * 10
    c.append(0, k2, torch.zeros(1, 2, 5, 64, device=DEV))
    c.advance(5)
    kr, _ = c.kv_slice(0, 25)
    s1 = k1.abs().amax(2, keepdim=True) / 7.0
    s2 = torch.maximum(s1,
                       k2.abs().amax(2, keepdim=True) / 7.0)
    assert bool(((kr[:, :, :20] - k1).abs() <= s2 + 1e-3).all())


def test_int8_kv_slice_read_path():
    torch.manual_seed(2)
    c = _mk("int8")
    k = torch.randn(1, 2, 30, 64, device=DEV, dtype=torch.bfloat16)
    c.append(0, k, k)
    c.advance(30)
    kr, _ = c.kv_slice(0, 30)
    assert (kr - k).abs().max().item() < 0.2  # dequantized, not raw int8


def test_pos_t_rejected_when_quantized():
    c = _mk()
    try:
        c.enable_tensor_pos()
    except ValueError:
        return
    raise AssertionError("pos_t must reject quantized caches")

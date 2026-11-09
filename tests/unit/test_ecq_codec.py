"""Tests for forge.quant.ecq_codec — range coder roundtrip + ECQ pack format."""

import numpy as np
import pytest

from forge.quant.ecq_codec import (
    ecq_codebook,
    ecq_pack,
    ecq_unpack,
    minmax_int4_baseline,
    pack_bf16_lossless,
    rc_decode,
    rc_encode,
    scale_hist,
    sqnr_db,
    unpack_bf16_lossless,
)


class TestRangeCoder:
    def test_roundtrip_uniform(self):
        rng = np.random.RandomState(0)
        syms = rng.randint(0, 16, 200_000).astype(np.uint16)
        counts = scale_hist(syms, 16)
        blob = rc_encode(syms, counts, int(counts.sum()))
        dec = rc_decode(blob, len(syms), counts, int(counts.sum()))
        np.testing.assert_array_equal(dec, syms)

    def test_roundtrip_skewed(self):
        # gaussian-usage code histogram (the ECQ case)
        rng = np.random.RandomState(1)
        p = np.exp(-0.5 * ((np.arange(16) - 7.5) / 2.5) ** 2)
        p /= p.sum()
        syms = rng.choice(16, 500_000, p=p).astype(np.uint16)
        counts = scale_hist(syms, 16)
        blob = rc_encode(syms, counts, int(counts.sum()))
        dec = rc_decode(blob, len(syms), counts, int(counts.sum()))
        np.testing.assert_array_equal(dec, syms)
        # compressed size must be near entropy
        h = float(-(np.log2(p) * p).sum())
        assert len(blob) * 8 <= len(syms) * h * 1.05 + 64

    def test_roundtrip_edge_cases(self):
        for n in (1, 2, 17, 255):
            syms = np.zeros(n, dtype=np.uint16)
            counts = scale_hist(syms, 4)
            blob = rc_encode(syms, counts, int(counts.sum()))
            np.testing.assert_array_equal(rc_decode(blob, n, counts, int(counts.sum())), syms)

    def test_scale_hist_sums(self):
        rng = np.random.RandomState(2)
        syms = rng.randint(0, 32, 10_000).astype(np.uint16)
        counts = scale_hist(syms, 32)
        assert counts.sum() == 32768
        present = np.bincount(syms, minlength=32) > 0
        assert (counts[present] >= 1).all()


class TestECQPack:
    def test_pack_roundtrip_exact(self):
        rng = np.random.RandomState(3)
        w = rng.randn(64, 32).astype(np.float32) * 0.02
        blob = ecq_pack(w, k=16)
        out = ecq_unpack(blob, dtype=np.float32)
        # indices must decode identically -> same reconstructed values
        # (codebook is stored as f32 inside the pack, so compare through f32)
        from forge.quant.ecq_codec import ecq_assign
        cb = ecq_codebook(w.astype(np.float64).ravel(), k=16)
        idx = ecq_assign(w.astype(np.float64).ravel(), cb)
        expect = cb.astype(np.float32).astype(np.float64)[idx]
        np.testing.assert_allclose(out.ravel(), expect, rtol=0, atol=0)

    def test_pack_smaller_than_raw(self):
        rng = np.random.RandomState(4)
        w = rng.randn(200_000).astype(np.float32) * 0.02
        blob = ecq_pack(w, k=16)
        # bf16 raw = 2 bytes/param; packed must be well under
        assert len(blob) < w.size  # <1 byte/param => <4 b/p feasible for gaussian

    def test_rate_matched_beats_uniform_and_g32(self):
        # measured anchors (iid gaussian): ECQ-16 20.1dB@3.85 vs uniform-int4
        # 15.5dB@~4.05; ECQ-32 24.7dB@4.93 vs g32-minmax-int4 22.1dB@4.9
        rng = np.random.RandomState(5)
        w = rng.randn(400_000).astype(np.float64) * 0.02
        b16 = ecq_pack(w, k=16)
        sq16 = sqnr_db(w, ecq_unpack(b16))
        q = np.round((w - w.min()) / (w.max() - w.min()) * 15)
        sq_uni = sqnr_db(w, w.min() + q / 15 * (w.max() - w.min()))
        assert sq16 > sq_uni + 2.0 and len(b16) * 8 / w.size < 4.05
        b32 = ecq_pack(w, k=32)
        sq32 = sqnr_db(w, ecq_unpack(b32))
        sq_g32, bits_g32 = minmax_int4_baseline(w)
        assert sq32 > sq_g32 and len(b32) * 8 / w.size <= bits_g32 + 0.3


class TestBF16Lossless:
    def test_bitexact_roundtrip(self):
        rng = np.random.RandomState(6)
        u = rng.randint(0, 65536, 257 * 389, dtype=np.int32).astype(np.uint16).reshape(257, 389)
        blob = pack_bf16_lossless(u)
        out = unpack_bf16_lossless(blob)
        np.testing.assert_array_equal(out, u)

    def test_real_weight_ratio(self):
        rng = np.random.RandomState(7)
        # emulate bf16 weight byte distribution: clustered exponent
        hi = np.clip(rng.randn(1_000_000) * 4 + 62, 0, 255).astype(np.uint8)
        lo = rng.randint(0, 256, 1_000_000).astype(np.uint8)
        u = (hi.astype(np.uint16) << 8) | lo
        blob = pack_bf16_lossless(u)
        assert len(blob) * 8 / u.size < 16.0
        np.testing.assert_array_equal(unpack_bf16_lossless(blob), u)

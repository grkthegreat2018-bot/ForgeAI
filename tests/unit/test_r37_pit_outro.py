"""Tests for R37-3 (PIT Key) and R37-4 (OutRo Key).

PIT: construction, forward (polar decomposition), reverse (reconstruction),
orthonormality check (M @ M.T ≈ I), round-trip identity.
OutRo: construction, sink detection, non-causal mask generation, forward
(apply mask), partial key class.

All tests run on CPU with small shapes for speed.
"""

import torch

from forge.keys.architecture.pit_tying_key import PITKey
from forge.keys.attention.outro_key import OutRoKey
from forge.keys.misc.base import KeyClass

# Small shapes for fast CPU tests.
V = 128
D = 64
N_HEADS = 4
SEQ_LEN = 32


# ── R37-3: PIT Key ──────────────────────────────────────────────────────────


class TestPITKey:
    def test_construction(self):
        key = PITKey()
        assert key.name == "pit_tying"
        assert key.key_class() == KeyClass.BI
        assert "Pseudo-Inverse Tying" in key.description

    def test_forward_polar_decomposition(self):
        """forward produces shared orthonormal memory M + projections."""
        torch.manual_seed(0)
        # Construct W_emb and W_unemb that share a common d-dim subspace so
        # the polar factor is well-defined. Use a low-rank-ish product.
        W_emb = torch.randn(V, D)
        M_true = torch.linalg.qr(torch.randn(D, D).T).Q.T  # random orthonormal
        P_emb_true = torch.randn(V, D)
        P_unemb_true = torch.randn(D, V)
        W_emb = P_emb_true @ M_true
        W_unemb = M_true @ P_unemb_true

        key = PITKey()
        res = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert res.success, f"forward failed: {res.error}"
        assert set(res.weights.keys()) == {"M", "P_emb", "P_unemb"}
        M = res.weights["M"]
        assert M.shape == (D, D)
        assert res.weights["P_emb"].shape == (V, D)
        assert res.weights["P_unemb"].shape == (D, V)

    def test_forward_missing_keys(self):
        key = PITKey()
        res = key.forward({"W_emb": torch.randn(V, D)})
        assert not res.success
        assert "W_unemb" in res.error

    def test_forward_shape_mismatch(self):
        key = PITKey()
        res = key.forward(
            {
                "W_emb": torch.randn(V, D),
                "W_unemb": torch.randn(D + 1, V),  # wrong d
            }
        )
        assert not res.success
        assert "Shape mismatch" in res.error

    def test_orthonormality(self):
        """M @ M.T ≈ I (shared memory is orthonormal)."""
        torch.manual_seed(1)
        W_emb = torch.randn(V, D)
        W_unemb = torch.randn(D, V)
        key = PITKey()
        res = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert res.success
        M = res.weights["M"]
        ident = torch.eye(D, dtype=M.dtype, device=M.device)
        assert torch.allclose(M @ M.T, ident, atol=1e-5), "M is not orthonormal: M @ M.T deviates from I"

    def test_reverse_reconstruction(self):
        """reverse reconstructs W_emb and W_unemb from M + projections."""
        torch.manual_seed(2)
        W_emb = torch.randn(V, D)
        W_unemb = torch.randn(D, V)
        key = PITKey()
        fwd = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success, f"reverse failed: {rev.error}"
        assert set(rev.data.keys()) == {"W_emb", "W_unemb"}
        assert rev.data["W_emb"].shape == (V, D)
        assert rev.data["W_unemb"].shape == (D, V)

    def test_reverse_missing_keys(self):
        key = PITKey()
        res = key.reverse({"M": torch.eye(D)})
        assert not res.success
        assert "P_emb" in res.error or "P_unemb" in res.error

    def test_round_trip(self):
        """forward then reverse reproduces the projections exactly.

        Because M is orthonormal, P_emb @ M == W_emb and M @ P_unemb ==
        W_unemb hold to numerical precision (the lstsq solves are exact for
        an orthonormal system).
        """
        torch.manual_seed(3)
        # Use a structured pair so the polar factor captures the shared space.
        M_true = torch.linalg.qr(torch.randn(D, D).T).Q.T
        P_emb = torch.randn(V, D)
        P_unemb = torch.randn(D, V)
        W_emb = P_emb @ M_true
        W_unemb = M_true @ P_unemb

        key = PITKey()
        fwd = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success

        # The reconstructed weights equal the originals (M is orthonormal so
        # the projection equations hold exactly up to solve precision).
        assert torch.allclose(rev.data["W_emb"], fwd.weights["P_emb"] @ fwd.weights["M"], atol=1e-4)
        assert torch.allclose(rev.data["W_unemb"], fwd.weights["M"] @ fwd.weights["P_unemb"], atol=1e-4)

    def test_bi_key_class(self):
        """PITKey reports BI (both directions, round-trip identity)."""
        assert PITKey().key_class() == KeyClass.BI


# ── R37-4: OutRo Key ────────────────────────────────────────────────────────


class TestOutRoKey:
    def test_construction(self):
        key = OutRoKey()
        assert key.name == "outro"
        assert key.key_class() == KeyClass.PARTIAL
        assert key.sink_threshold == 0.5
        assert "OutRo" in key.description

    def test_construction_custom_threshold(self):
        key = OutRoKey(sink_threshold=0.3, align_strength=0.1)
        assert key.sink_threshold == 0.3
        assert key.align_strength == 0.1

    def _make_attn(self, seed: int = 0, sink_pos: int | None = 0):
        """Build attention weights with a strong sink at position 0."""
        torch.manual_seed(seed)
        # (n_heads, seq_q, seq_k) — softmax-normalized over seq_k.
        logits = torch.randn(N_HEADS, SEQ_LEN, SEQ_LEN) * 0.5
        attn = torch.softmax(logits, dim=-1)
        if sink_pos is not None:
            # Force several query positions to attend strongly to position 0.
            attn[..., 0, :] = 0.0
            attn[..., 0, 0] = 1.0  # position 0 is a sink
            attn[..., 5, :] = 0.0
            attn[..., 5, 0] = 0.9  # position 5 is a sink
            attn[..., 10, :] = attn[..., 10, :] * 0.2
            attn[..., 10, 0] = 0.8  # position 10 is a sink
            # Renormalize rows that we modified partially.
            attn[..., 10, :] = attn[..., 10, :] / attn[..., 10, :].sum(dim=-1, keepdim=True)
        return attn

    def test_sink_detection(self):
        """Tokens with attention to position 0 > threshold are sinks."""
        attn = self._make_attn()
        key = OutRoKey(sink_threshold=0.5)
        sink_mask = key._detect_sinks(attn)  # (n_heads, seq_q)
        assert sink_mask.shape == (N_HEADS, SEQ_LEN)
        # Positions 0, 5, 10 were forced to be sinks.
        for pos in (0, 5, 10):
            assert sink_mask[..., pos].all(), f"position {pos} should be a sink"
        # A typical random position should not be a sink.
        assert not sink_mask[..., 1].all()

    def test_non_causal_mask_generation(self):
        """Sink rows allow attending to all positions; others stay causal."""
        attn = self._make_attn()
        key = OutRoKey(sink_threshold=0.5)
        fwd = key.forward({"attn_weights": attn, "d_model": D})
        assert fwd.success, f"forward failed: {fwd.error}"
        mask = fwd.weights["attention_mask"]  # (n_heads, seq_q, seq_k)
        sink_mask = fwd.weights["sink_mask"]
        assert mask.shape == (N_HEADS, SEQ_LEN, SEQ_LEN)
        assert mask.dtype == torch.bool

        # For a sink query row, every position is allowed (non-causal).
        sink_q = 0
        assert sink_mask[..., sink_q].all()
        assert mask[..., sink_q, :].all(), "sink row should be fully allowed"

        # For a non-sink query row, future positions are masked (causal).
        non_sink_q = 1
        assert not sink_mask[..., non_sink_q].all()
        # Position non_sink_q can attend to itself and earlier, not later.
        assert mask[..., non_sink_q, non_sink_q].all()  # self
        if non_sink_q + 1 < SEQ_LEN:
            assert not mask[..., non_sink_q, non_sink_q + 1].any()  # future

    def test_forward_apply_mask(self):
        """forward produces mask + alignment suitable for applying to attention."""
        attn = self._make_attn()
        key = OutRoKey()
        res = key.forward({"attn_weights": attn, "d_model": D})
        assert res.success
        assert set(res.weights.keys()) == {"attention_mask", "sink_mask", "alignment"}
        align = res.weights["alignment"]
        assert align.shape == (D, D)
        # Alignment is identity at start (lossless).
        assert torch.allclose(align, torch.eye(D))
        # Metadata reports the number of sinks.
        assert res.metadata["n_sinks"] > 0

    def test_forward_missing_attn(self):
        key = OutRoKey()
        res = key.forward({"d_model": D})
        assert not res.success
        assert "attn_weights" in res.error

    def test_reverse_extracts_sink_mask_and_alignment(self):
        """reverse pulls out the sink mask + alignment parameters."""
        attn = self._make_attn()
        key = OutRoKey()
        fwd = key.forward({"attn_weights": attn, "d_model": D})
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success, f"reverse failed: {rev.error}"
        assert set(rev.data.keys()) == {"sink_mask", "alignment", "n_sinks"}
        assert rev.data["n_sinks"] > 0
        # Sink mask round-trips exactly.
        assert torch.equal(rev.data["sink_mask"], fwd.weights["sink_mask"])
        # Alignment is identity.
        assert rev.metadata["alignment_is_identity"] is True

    def test_reverse_missing_keys(self):
        key = OutRoKey()
        res = key.reverse({"sink_mask": torch.ones(SEQ_LEN, dtype=torch.bool)})
        assert not res.success
        assert "alignment" in res.error

    def test_partial_key_class(self):
        """OutRoKey reports PARTIAL (forward only — modifies attention)."""
        assert OutRoKey().key_class() == KeyClass.PARTIAL

    def test_no_sinks_when_threshold_high(self):
        """With a very high threshold, no tokens are sinks → fully causal mask."""
        attn = self._make_attn()
        key = OutRoKey(sink_threshold=2.0)  # impossible to exceed
        fwd = key.forward({"attn_weights": attn, "d_model": D})
        assert fwd.success
        assert fwd.metadata["n_sinks"] == 0
        mask = fwd.weights["attention_mask"]
        # Should be exactly the causal mask everywhere.
        causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
        assert torch.equal(mask, causal.expand(N_HEADS, SEQ_LEN, SEQ_LEN))

    def test_all_sinks_when_threshold_low(self):
        """With a near-zero threshold, every token is a sink → fully allowed."""
        attn = self._make_attn(sink_pos=None)  # random attention, no forced sink
        key = OutRoKey(sink_threshold=0.0)
        fwd = key.forward({"attn_weights": attn, "d_model": D})
        assert fwd.success
        # Every position has attn[..., 0] > 0 after softmax (strictly positive).
        assert fwd.metadata["n_sinks"] == N_HEADS * SEQ_LEN
        mask = fwd.weights["attention_mask"]
        assert mask.all()  # fully allowed (non-causal everywhere)

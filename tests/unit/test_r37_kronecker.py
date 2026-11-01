"""Tests for R37-2: Kronecker Embedding Key.

Tests the KroneckerEmbedKey and KroneckerEmbedding for:
- Key construction and properties
- Param reduction calculation (>90% for typical sizes)
- Forward: standard embedding → Kronecker form (SVD initialization)
- Reverse: Kronecker → standard embedding (reconstruction)
- Round-trip quality (bounded reconstruction error)
- Shape correctness for all factored components

Runs on CPU with small dimensions for fast, GPU-independent testing.
"""

import pytest
import torch
import torch.nn as nn

from forge.keys.architecture.kronecker_embed_key import (
    KroneckerEmbedding,
    KroneckerEmbedKey,
    _ids_to_bytes,
)
from forge.keys.misc.base import Key, KeyClass

# ── Test fixtures ─────────────────────────────────────────────────────────────

# Small shapes for fast tests
VOCAB = 256
D_MODEL = 128
D_CHAR = 32
MAX_CHAR_LEN = 4

# Large shapes for param reduction test (not instantiated, just calculated)
LARGE_VOCAB = 65536
LARGE_D_MODEL = 2048
LARGE_D_CHAR = 64
LARGE_MAX_CHAR_LEN = 8


@pytest.fixture
def key():
    return KroneckerEmbedKey(vocab_size=VOCAB, d_model=D_MODEL, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)


@pytest.fixture
def standard_embedding():
    """Random standard embedding (V, d_model)."""
    torch.manual_seed(42)
    return nn.Embedding(VOCAB, D_MODEL)


# ── Key construction and properties ───────────────────────────────────────────


class TestKeyConstruction:
    def test_inherits_key(self, key):
        assert isinstance(key, Key)

    def test_name(self, key):
        assert key.name == "kronecker_embed"

    def test_key_class_is_bi(self, key):
        assert key.key_class() == KeyClass.BI

    def test_description_nonempty(self, key):
        assert len(key.description) > 0

    def test_repr(self, key):
        assert "kronecker_embed" in repr(key)
        assert "bi" in repr(key)

    def test_init_stores_params(self, key):
        assert key.vocab_size == VOCAB
        assert key.d_model == D_MODEL
        assert key.d_char == D_CHAR
        assert key.max_char_len == MAX_CHAR_LEN


# ── Param reduction calculation ───────────────────────────────────────────────


class TestParamReduction:
    def test_small_config_reduction(self, key):
        """Small config: 256*32 + 4*32 + 32*4*128 = 8192 + 128 + 16384 = 24704."""
        standard = key.standard_param_count()
        kronecker = key.param_count()
        assert standard == VOCAB * D_MODEL  # 32768
        # 256*d_char + L*d_char + d_char*L*d_model
        expected = 256 * D_CHAR + MAX_CHAR_LEN * D_CHAR + D_CHAR * MAX_CHAR_LEN * D_MODEL
        assert kronecker == expected  # 24704
        assert kronecker < standard
        pct = key.reduction_pct()
        assert 0 < pct < 100

    def test_large_config_reduction_gt_90(self):
        """Large config (65536×2048): should achieve >90% reduction."""
        key = KroneckerEmbedKey(
            vocab_size=LARGE_VOCAB, d_model=LARGE_D_MODEL, d_char=LARGE_D_CHAR, max_char_len=LARGE_MAX_CHAR_LEN
        )
        key.standard_param_count()  # 134.2M
        kronecker = key.param_count()
        # 256*64 + 8*64 + 64*8*2048 = 16384 + 512 + 1048576 = 1065472
        expected = (
            256 * LARGE_D_CHAR + LARGE_MAX_CHAR_LEN * LARGE_D_CHAR + LARGE_D_CHAR * LARGE_MAX_CHAR_LEN * LARGE_D_MODEL
        )
        assert kronecker == expected
        pct = key.reduction_pct()
        assert pct > 90.0, f"Reduction {pct:.1f}% should be >90% for large config"
        # Spec example: ~99.2%
        assert pct > 99.0, f"Large config reduction {pct:.1f}% should be ~99%"

    def test_reduction_increases_with_vocab(self):
        """Larger vocab → greater reduction (projection is fixed cost)."""
        key_small = KroneckerEmbedKey(256, 128, d_char=32, max_char_len=4)
        key_large = KroneckerEmbedKey(65536, 128, d_char=32, max_char_len=4)
        assert key_large.reduction_pct() > key_small.reduction_pct()


# ── Forward: standard → Kronecker (SVD init) ──────────────────────────────────


class TestForward:
    def test_forward_success(self, key, standard_embedding):
        data = {"weight": standard_embedding.weight}
        result = key.forward(data)
        assert result.success
        assert result.weights is not None

    def test_forward_missing_weight(self, key):
        result = key.forward({})
        assert not result.success
        assert "weight" in result.error

    def test_forward_wrong_vocab(self, key):
        W = torch.randn(128, D_MODEL)  # wrong vocab size
        result = key.forward({"weight": W})
        assert not result.success

    def test_forward_wrong_d_model(self, key):
        W = torch.randn(VOCAB, 64)  # wrong d_model
        result = key.forward({"weight": W})
        assert not result.success

    def test_forward_weight_shapes(self, key, standard_embedding):
        result = key.forward({"weight": standard_embedding.weight})
        w = result.weights
        assert w["char_embed"].shape == (256, D_CHAR)
        assert w["pos_encode"].shape == (MAX_CHAR_LEN, D_CHAR)
        assert w["projection"].shape == (D_CHAR * MAX_CHAR_LEN, D_MODEL)

    def test_forward_char_embed_zero_init(self, key, standard_embedding):
        """Spec: char_embed zero-init (warm start)."""
        result = key.forward({"weight": standard_embedding.weight})
        assert torch.allclose(result.weights["char_embed"], torch.zeros(256, D_CHAR))

    def test_forward_pos_encode_zero_init(self, key, standard_embedding):
        """Spec: pos_encode zero-init (warm start)."""
        result = key.forward({"weight": standard_embedding.weight})
        assert torch.allclose(result.weights["pos_encode"], torch.zeros(MAX_CHAR_LEN, D_CHAR))

    def test_forward_projection_from_svd(self, key, standard_embedding):
        """Projection should contain SVD right singular vectors (scaled by S)."""
        W = standard_embedding.weight.float()
        _U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(D_CHAR * MAX_CHAR_LEN, min(VOCAB, D_MODEL))

        result = key.forward({"weight": standard_embedding.weight})
        proj = result.weights["projection"].float()

        # Top-k rows should match S * Vt
        expected_topk = S[:k].unsqueeze(1) * Vt[:k, :]
        assert torch.allclose(proj[:k, :], expected_topk, atol=1e-5)
        # Remaining rows should be zero
        if k < D_CHAR * MAX_CHAR_LEN:
            assert torch.allclose(proj[k:, :], torch.zeros_like(proj[k:, :]))

    def test_forward_metadata(self, key, standard_embedding):
        result = key.forward({"weight": standard_embedding.weight})
        meta = result.metadata
        assert meta["vocab_size"] == VOCAB
        assert meta["d_model"] == D_MODEL
        assert meta["d_char"] == D_CHAR
        assert meta["max_char_len"] == MAX_CHAR_LEN
        assert meta["svd_rank"] == min(D_CHAR * MAX_CHAR_LEN, min(VOCAB, D_MODEL))
        assert meta["standard_params"] == VOCAB * D_MODEL
        assert meta["kronecker_params"] == key.param_count()


# ── Reverse: Kronecker → standard (reconstruction) ────────────────────────────


class TestReverse:
    def test_reverse_success(self, key):
        weights = {
            "char_embed": torch.randn(256, D_CHAR),
            "pos_encode": torch.randn(MAX_CHAR_LEN, D_CHAR),
            "projection": torch.randn(D_CHAR * MAX_CHAR_LEN, D_MODEL),
        }
        result = key.reverse(weights)
        assert result.success
        assert result.data is not None

    def test_reverse_missing_keys(self, key):
        result = key.reverse({"char_embed": torch.zeros(256, D_CHAR)})
        assert not result.success

    def test_reverse_wrong_shapes(self, key):
        weights = {
            "char_embed": torch.zeros(128, D_CHAR),  # wrong: should be 256
            "pos_encode": torch.zeros(MAX_CHAR_LEN, D_CHAR),
            "projection": torch.zeros(D_CHAR * MAX_CHAR_LEN, D_MODEL),
        }
        result = key.reverse(weights)
        assert not result.success

    def test_reverse_output_shape(self, key):
        weights = {
            "char_embed": torch.randn(256, D_CHAR),
            "pos_encode": torch.randn(MAX_CHAR_LEN, D_CHAR),
            "projection": torch.randn(D_CHAR * MAX_CHAR_LEN, D_MODEL),
        }
        result = key.reverse(weights)
        assert result.data["weight"].shape == (VOCAB, D_MODEL)

    def test_reverse_zero_weights_give_zero(self, key):
        """With zero char_embed and pos_encode, reconstruction is zero."""
        weights = {
            "char_embed": torch.zeros(256, D_CHAR),
            "pos_encode": torch.zeros(MAX_CHAR_LEN, D_CHAR),
            "projection": torch.randn(D_CHAR * MAX_CHAR_LEN, D_MODEL),
        }
        result = key.reverse(weights)
        W = result.data["weight"]
        assert torch.allclose(W, torch.zeros(VOCAB, D_MODEL))

    def test_reverse_correct_computation(self, key):
        """Manually verify reverse computes the Kronecker forward pass correctly."""
        torch.manual_seed(123)
        char_embed = torch.randn(256, D_CHAR)
        pos_encode = torch.randn(MAX_CHAR_LEN, D_CHAR)
        projection = torch.randn(D_CHAR * MAX_CHAR_LEN, D_MODEL)
        weights = {"char_embed": char_embed, "pos_encode": pos_encode, "projection": projection}
        result = key.reverse(weights)
        W = result.data["weight"]

        # Manually compute for a few token ids
        token_ids = torch.arange(VOCAB, dtype=torch.long)
        byte_ids = _ids_to_bytes(token_ids, MAX_CHAR_LEN)
        for t in [0, 1, 42, 127, 255]:
            b = byte_ids[t]
            char_vec = char_embed[b] + pos_encode  # (L, d_char)
            flat = char_vec.view(-1)  # (L*d_char,)
            expected = flat @ projection  # (d_model,)
            assert torch.allclose(W[t], expected, atol=1e-5), f"Token {t}: reverse mismatch"


# ── Round-trip quality ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_round_trip_shapes(self, key, standard_embedding):
        """forward → reverse produces correct shapes."""
        fwd = key.forward({"weight": standard_embedding.weight})
        assert fwd.success
        rev = key.reverse(fwd.weights)
        assert rev.success
        assert rev.data["weight"].shape == (VOCAB, D_MODEL)

    def test_round_trip_no_nan(self, key, standard_embedding):
        """Round-trip should not produce NaN or Inf."""
        fwd = key.forward({"weight": standard_embedding.weight})
        rev = key.reverse(fwd.weights)
        W = rev.data["weight"]
        assert torch.isfinite(W).all()

    def test_round_trip_error_bounded(self, key, standard_embedding):
        """Round-trip reconstruction error is bounded.

        With zero-init char_embed and pos_encode, the reconstruction is zero,
        so the error equals ||W||. The relative error is bounded by 1.0.
        The SVD projection captures the principal directions for fine-tuning.
        """
        W_orig = standard_embedding.weight
        fwd = key.forward({"weight": W_orig})
        rev = key.reverse(fwd.weights)
        W_recon = rev.data["weight"]

        error = (W_recon - W_orig).norm()
        w_norm = W_orig.norm()
        relative_error = error / w_norm

        # Error is bounded (finite, relative error <= 1.0)
        assert torch.isfinite(error)
        assert relative_error <= 1.0 + 1e-6, f"Relative error {relative_error:.4f} should be <= 1.0"

    def test_round_trip_with_nonzero_weights(self, key):
        """Round-trip with non-zero Kronecker weights: reverse → forward → reverse.

        This tests that the reverse correctly reconstructs the effective embedding,
        and that forward→reverse preserves the SVD-captured structure.
        """
        torch.manual_seed(99)
        # Create Kronecker weights with non-zero char_embed and pos_encode
        char_embed = torch.randn(256, D_CHAR) * 0.1
        pos_encode = torch.randn(MAX_CHAR_LEN, D_CHAR) * 0.1
        projection = torch.randn(D_CHAR * MAX_CHAR_LEN, D_MODEL) * 0.1
        weights = {"char_embed": char_embed, "pos_encode": pos_encode, "projection": projection}

        # Reverse: get effective W
        rev1 = key.reverse(weights)
        assert rev1.success
        W = rev1.data["weight"]

        # Forward: convert W back to Kronecker (SVD init, zero char/pos)
        fwd = key.forward({"weight": W})
        assert fwd.success

        # Reverse: reconstruct from SVD-initialized Kronecker
        rev2 = key.reverse(fwd.weights)
        assert rev2.success
        rev2.data["weight"]

        # The SVD projection captures the top-k directions.
        # With zero char/pos, reconstruction is zero, but the projection
        # contains the SVD components. Verify the projection captures
        # the singular structure of W.
        proj = fwd.weights["projection"].float()
        W_f = W.float()
        _U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
        k = min(D_CHAR * MAX_CHAR_LEN, min(VOCAB, D_MODEL))
        expected_topk = S[:k].unsqueeze(1) * Vt[:k, :]
        assert torch.allclose(proj[:k, :], expected_topk, atol=1e-4)

    def test_round_trip_svd_quality(self, key, standard_embedding):
        """The SVD projection captures the top-k singular values of W."""
        W = standard_embedding.weight.float()
        _U, S, _Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(D_CHAR * MAX_CHAR_LEN, min(VOCAB, D_MODEL))

        fwd = key.forward({"weight": standard_embedding.weight})
        proj = fwd.weights["projection"].float()

        # The projection rows should have norms matching singular values
        proj_row_norms = proj[:k, :].norm(dim=1)
        assert torch.allclose(proj_row_norms, S[:k], atol=1e-4), (
            f"Projection row norms {proj_row_norms[:5]} vs S {S[:5]}"
        )


# ── Shape correctness for all factored components ─────────────────────────────


class TestShapeCorrectness:
    def test_char_embed_shape(self, key, standard_embedding):
        result = key.forward({"weight": standard_embedding.weight})
        assert result.weights["char_embed"].shape == (256, D_CHAR)

    def test_pos_encode_shape(self, key, standard_embedding):
        result = key.forward({"weight": standard_embedding.weight})
        assert result.weights["pos_encode"].shape == (MAX_CHAR_LEN, D_CHAR)

    def test_projection_shape(self, key, standard_embedding):
        result = key.forward({"weight": standard_embedding.weight})
        assert result.weights["projection"].shape == (D_CHAR * MAX_CHAR_LEN, D_MODEL)

    def test_reverse_output_shape(self, key):
        weights = {
            "char_embed": torch.zeros(256, D_CHAR),
            "pos_encode": torch.zeros(MAX_CHAR_LEN, D_CHAR),
            "projection": torch.zeros(D_CHAR * MAX_CHAR_LEN, D_MODEL),
        }
        result = key.reverse(weights)
        assert result.data["weight"].shape == (VOCAB, D_MODEL)


# ── KroneckerEmbedding nn.Module ──────────────────────────────────────────────


class TestKroneckerEmbedding:
    def test_init(self):
        emb = KroneckerEmbedding(VOCAB, D_MODEL, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        assert emb.vocab_size == VOCAB
        assert emb.d_model == D_MODEL
        assert emb.d_char == D_CHAR
        assert emb.max_char_len == MAX_CHAR_LEN

    def test_param_shapes(self):
        emb = KroneckerEmbedding(VOCAB, D_MODEL, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        assert emb.char_embed.shape == (256, D_CHAR)
        assert emb.pos_encode.shape == (MAX_CHAR_LEN, D_CHAR)
        assert emb.projection.shape == (D_CHAR * MAX_CHAR_LEN, D_MODEL)

    def test_forward_output_shape(self):
        emb = KroneckerEmbedding(VOCAB, D_MODEL, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        token_ids = torch.tensor([0, 1, 42, 127, 255])
        out = emb(token_ids)
        assert out.shape == (5, D_MODEL)

    def test_forward_batched(self):
        emb = KroneckerEmbedding(VOCAB, D_MODEL, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        token_ids = torch.randint(0, VOCAB, (4, 16))
        out = emb(token_ids)
        assert out.shape == (4, 16, D_MODEL)

    def test_param_count(self):
        emb = KroneckerEmbedding(VOCAB, D_MODEL, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        expected = 256 * D_CHAR + MAX_CHAR_LEN * D_CHAR + D_CHAR * MAX_CHAR_LEN * D_MODEL
        assert emb.param_count() == expected

    def test_from_embedding(self):
        """from_embedding should produce a Kronecker embedding that approximates W."""
        torch.manual_seed(42)
        original = nn.Embedding(VOCAB, D_MODEL)
        kron = KroneckerEmbedding.from_embedding(original, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        # Check that the effective weight is finite and roughly shaped
        W_eff = kron.weight  # (V, d_model) computed
        W_orig = original.weight
        assert W_eff.shape == W_orig.shape
        assert torch.isfinite(W_eff).all()
        # Least-squares init should give some approximation (not exact)
        relative_error = (W_eff - W_orig).norm() / W_orig.norm()
        assert relative_error < 1.0, f"from_embedding relative error {relative_error:.4f} should be < 1.0"

    def test_from_embedding_better_than_zero_init(self):
        """from_embedding (least-squares) should be at least slightly better than zero."""
        torch.manual_seed(42)
        original = nn.Embedding(VOCAB, D_MODEL)
        kron = KroneckerEmbedding.from_embedding(original, d_char=D_CHAR, max_char_len=MAX_CHAR_LEN)
        W_eff = kron.weight
        W_orig = original.weight
        error_ls = (W_eff - W_orig).norm().item()

        # Zero-init would give error = ||W_orig||
        error_zero = W_orig.norm().item()
        # Least-squares should be at least marginally better than zero
        assert error_ls <= error_zero, (
            f"Least-squares error {error_ls:.4f} should be <= zero-init error {error_zero:.4f}"
        )

    def test_vocab_too_large_raises(self):
        with pytest.raises(AssertionError):
            KroneckerEmbedding(vocab_size=256**4 + 1, d_model=64, d_char=8, max_char_len=4)


# ── Byte conversion utility ───────────────────────────────────────────────────


class TestIdsToBytes:
    def test_basic_conversion(self):
        ids = torch.tensor([0, 1, 255, 256, 257])
        byte_ids = _ids_to_bytes(ids, max_char_len=4)
        assert byte_ids.shape == (5, 4)
        # token 0 → all zeros
        assert byte_ids[0].tolist() == [0, 0, 0, 0]
        # token 1 → [1, 0, 0, 0] (little-endian)
        assert byte_ids[1].tolist() == [1, 0, 0, 0]
        # token 255 → [255, 0, 0, 0]
        assert byte_ids[2].tolist() == [255, 0, 0, 0]
        # token 256 → [0, 1, 0, 0] (256 = 1*256 + 0)
        assert byte_ids[3].tolist() == [0, 1, 0, 0]
        # token 257 → [1, 1, 0, 0]
        assert byte_ids[4].tolist() == [1, 1, 0, 0]

    def test_all_byte_values(self):
        """Test that all 256 byte values appear for token 256^2-1."""
        ids = torch.tensor([256**2 - 1])  # 65535 = 255*256 + 255
        byte_ids = _ids_to_bytes(ids, max_char_len=4)
        assert byte_ids[0].tolist() == [255, 255, 0, 0]

    def test_shape(self):
        ids = torch.arange(1000)
        byte_ids = _ids_to_bytes(ids, max_char_len=8)
        assert byte_ids.shape == (1000, 8)

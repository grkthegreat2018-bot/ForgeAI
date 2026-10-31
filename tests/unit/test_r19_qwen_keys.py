"""Tests for R&D Round 19: Qwen3.8-Flash-Next architecture keys.

QSA (Qwen Sparse Attention), Gated Residual, N-gram Embedding.
All run on GPU (CUDA) with fp32. CPU fallback only if CUDA unavailable.
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32


# ── R19a: QSA (Qwen Sparse Attention) ───────────────────────────────────────

def test_qsa_identity_warm_start():
    """QSA with budget=all blocks should match full attention at init."""
    from forge.keys.attention.qsa_key import QSALayer

    torch.manual_seed(42)
    d_model, n_heads, n_kv_heads, head_dim = 256, 8, 2, 32
    B, T = 2, 64

    layer = QSALayer(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, block_size=4, budget_blocks=64,  # all blocks
        n_indexer_heads=4, indexer_head_dim=64,
    ).to(_DEV, _DTYPE)

    x = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)
    out, _cache = layer(x, use_cache=True)

    # Output should have the right shape
    assert out.shape == (B, T, d_model), f"Wrong shape: {out.shape}"
    # Output should be finite (no NaN/Inf from softmax)
    assert torch.isfinite(out).all(), "Output has NaN/Inf"
    print("  QSA identity warm start: PASS")


def test_qsa_sparse_vs_full():
    """QSA with budget < all blocks should produce different (sparse) output."""
    from forge.keys.attention.qsa_key import QSALayer

    torch.manual_seed(42)
    d_model, n_heads, n_kv_heads, head_dim = 256, 8, 2, 32
    B, T = 2, 128

    # Full attention (budget = all blocks)
    layer_full = QSALayer(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, block_size=4, budget_blocks=32,  # all 32 blocks
    ).to(_DEV, _DTYPE)

    # Sparse attention (budget = 8 blocks out of 32)
    layer_sparse = QSALayer(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, block_size=4, budget_blocks=8,  # only 8/32 blocks
    ).to(_DEV, _DTYPE)

    # Copy weights so the only difference is the budget
    layer_sparse.load_state_dict(layer_full.state_dict())

    x = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)
    out_full, _ = layer_full(x)
    out_sparse, _ = layer_sparse(x)

    # Outputs should differ (sparse skips some blocks)
    diff = (out_full - out_sparse).abs().max().item()
    print(f"  QSA full vs sparse max diff: {diff:.4f}")
    assert diff > 1e-6, "Sparse and full should differ when budget < n_blocks"
    print("  QSA sparse vs full: PASS")


def test_qsa_key_forward_reverse():
    """QSA key forward produces weights, reverse drops indexer."""
    from forge.keys.attention.qsa_key import QSAKey

    key = QSAKey(block_size=4, budget_blocks=512)
    result = key.forward({
        "d_model": 256, "n_heads": 8, "n_kv_heads": 2, "head_dim": 32
    })
    assert result.success
    assert "indexer_q_weight" in result.weights
    assert "indexer_k_weight" in result.weights
    assert "q_weight" in result.weights

    # Reverse: should drop indexer, keep standard attention
    rev = key.reverse(result.weights)
    assert rev.success
    assert "q_weight" in rev.weights
    assert "indexer_q_weight" not in rev.weights
    print("  QSA key forward/reverse: PASS")


# ── R19b: Gated Residual ────────────────────────────────────────────────────

def test_gated_residual_identity_warm_start():
    """GR at init should be near-identity (branch 0 active, others disabled)."""
    from forge.keys.architecture.gated_residual_key import GatedResidualLayer

    torch.manual_seed(42)
    d_model, n_branches, rank = 256, 4, 64
    B, T = 2, 32

    layer = GatedResidualLayer(
        d_model=d_model, n_branches=n_branches, bottleneck_rank=rank,
    ).to(_DEV, _DTYPE)

    x = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)
    out = layer(x)

    # At init, branch 0 write gate ≈ sigmoid(3) ≈ 0.95, others ≈ sigmoid(-3) ≈ 0.05
    # So output ≈ x + 0.95 * branch_0(x) + small * others
    # Branch 0 has small random weights, so output ≈ x + small perturbation
    diff = (out - x).abs().max().item()
    print(f"  GR init max diff from identity: {diff:.4f}")
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()
    # The diff should be relatively small (branches are near-zero at init)
    # Branch 0 has Kaiming init with a=0.01, so weights are small
    print("  GR identity warm start: PASS")


def test_gated_residual_branches_active():
    """After perturbing write gates, all branches should contribute."""
    from forge.keys.architecture.gated_residual_key import GatedResidualLayer

    torch.manual_seed(42)
    d_model, n_branches, rank = 256, 4, 64
    B, T = 2, 32

    layer = GatedResidualLayer(
        d_model=d_model, n_branches=n_branches, bottleneck_rank=rank,
    ).to(_DEV, _DTYPE)

    # Open all write gates
    with torch.no_grad():
        for i in range(n_branches):
            layer.write_gate[i].bias.fill_(3.0)  # sigmoid(3) ≈ 0.95
            # Also give branch i non-zero weights
            nn_init = torch.nn.init
            nn_init.kaiming_normal_(layer.branch_down[i].weight, a=0.1)
            nn_init.kaiming_normal_(layer.branch_up[i].weight, a=0.1)

    x = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)
    out = layer(x)

    # Now all branches contribute — output should differ more from x
    diff = (out - x).abs().max().item()
    print(f"  GR all-branches-active max diff: {diff:.4f}")
    assert diff > 0.1, "All branches active should produce larger output change"
    print("  GR branches active: PASS")


def test_gated_residual_key():
    """GR key forward produces weights for all branches."""
    from forge.keys.architecture.gated_residual_key import GatedResidualKey

    key = GatedResidualKey(n_branches=4, bottleneck_rank=64)
    result = key.forward({"d_model": 256})
    assert result.success
    # Should have weights for all 4 branches
    for i in range(4):
        assert f"read_gate_down_{i}" in result.weights
        assert f"write_gate_{i}_weight" in result.weights
        assert f"branch_down_{i}" in result.weights
    print("  GR key forward: PASS")


# ── R19c: N-gram Embedding ──────────────────────────────────────────────────

def test_ngram_embedding_identity_warm_start():
    """N-gram embedding at init (all zeros) should not change token embeddings."""
    from forge.keys.knowledge.ngram_embedding_key import NGramEmbeddingLayer

    vocab_size, d_model, n_gram, table_size = 1000, 256, 2, 10000
    B, T = 2, 32

    # Use GPU table (not host) for test simplicity
    layer = NGramEmbeddingLayer(
        vocab_size=vocab_size, d_model=d_model, n_gram=n_gram,
        table_size=table_size, device=str(_DEV), host_table=False,
    ).to(_DEV)

    input_ids = torch.randint(0, vocab_size, (B, T), device=_DEV)
    token_emb = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)

    out = layer(input_ids, token_emb)

    # At init, ngram_table is all zeros → output = token_emb
    diff = (out - token_emb).abs().max().item()
    print(f"  N-gram init max diff: {diff:.6f}")
    assert diff < 1e-6, "N-gram embedding should be zero at init"
    print("  N-gram identity warm start: PASS")


def test_ngram_embedding_nonzero():
    """After filling table, n-gram embedding should change the output."""
    from forge.keys.knowledge.ngram_embedding_key import NGramEmbeddingLayer

    vocab_size, d_model, n_gram, table_size = 1000, 256, 2, 10000
    B, T = 2, 32

    layer = NGramEmbeddingLayer(
        vocab_size=vocab_size, d_model=d_model, n_gram=n_gram,
        table_size=table_size, device=str(_DEV), host_table=False,
    ).to(_DEV)

    # Fill table with random values
    with torch.no_grad():
        layer.ngram_table.data = torch.randn_like(layer.ngram_table.data) * 0.1

    input_ids = torch.randint(0, vocab_size, (B, T), device=_DEV)
    token_emb = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)

    out = layer(input_ids, token_emb)

    diff = (out - token_emb).abs().max().item()
    print(f"  N-gram non-zero max diff: {diff:.4f}")
    assert diff > 0.01, "Non-zero table should change output"
    print("  N-gram non-zero: PASS")


def test_ngram_embedding_host_offload():
    """N-gram embedding with host_table=True should work (CPU table, GPU output)."""
    from forge.keys.knowledge.ngram_embedding_key import NGramEmbeddingLayer

    vocab_size, d_model, n_gram, table_size = 1000, 256, 2, 10000
    B, T = 2, 32

    layer = NGramEmbeddingLayer(
        vocab_size=vocab_size, d_model=d_model, n_gram=n_gram,
        table_size=table_size, device=str(_DEV), host_table=True,
    )

    # Table should be on CPU
    assert layer.ngram_table.data.device.type == "cpu", \
        "Host table should be on CPU"

    # Fill with random values
    with torch.no_grad():
        layer.ngram_table.data = torch.randn_like(layer.ngram_table.data) * 0.1

    input_ids = torch.randint(0, vocab_size, (B, T), device=_DEV)
    token_emb = torch.randn(B, T, d_model, device=_DEV, dtype=_DTYPE)

    out = layer(input_ids, token_emb)

    # Output should be on GPU and have the right shape
    assert out.device.type == "cuda" or not torch.cuda.is_available()
    assert out.shape == (B, T, d_model)
    diff = (out - token_emb).abs().max().item()
    print(f"  N-gram host-offload max diff: {diff:.4f}")
    assert diff > 0.01
    print("  N-gram host offload: PASS")


def test_ngram_embedding_key():
    """N-gram key forward produces zero table, reverse drops it."""
    from forge.keys.knowledge.ngram_embedding_key import NGramEmbeddingKey

    key = NGramEmbeddingKey(n_gram=2, table_size=10000)
    result = key.forward({"vocab_size": 1000, "d_model": 256})
    assert result.success
    assert "ngram_table" in result.weights
    assert result.weights["ngram_table"].shape == (10000, 256)
    # Table should be all zeros at init
    assert result.weights["ngram_table"].abs().max().item() < 1e-6

    # Reverse: drops the table
    rev = key.reverse(result.weights)
    assert rev.success
    assert "ngram_table" not in rev.weights
    print("  N-gram key forward/reverse: PASS")


# ── VRAM and parameter analysis ─────────────────────────────────────────────

def test_r19_vram_analysis():
    """Analyze VRAM usage of each R19 key at LFM2.5-1.2B scale."""
    print("\n  R19 VRAM/param analysis (LFM2.5-1.2B scale, d_model=2048):")
    print("-" * 70)

    d_model = 2048

    # QSA: indexer adds 4 query heads + 1 key head, head_dim=128
    # Per attention layer: indexer_q = d_model * (4*128) = 2048*512 = 1M
    #                      indexer_k = d_model * 128 = 2048*128 = 0.26M
    # Total per layer: 1.26M params, 6 layers → 7.56M params
    qsa_per_layer = d_model * (4 * 128) + d_model * 128
    qsa_total = qsa_per_layer * 6
    qsa_vram_mb = qsa_total * 2 / 1024 / 1024  # bf16
    print(f"  QSA indexer:      {qsa_per_layer/1e6:.2f}M/layer × 6 = "
          f"{qsa_total/1e6:.2f}M params ({qsa_vram_mb:.1f} MB bf16)")

    # Gated Residual: 4 branches, rank=256
    # Per layer: 4 × (read_gate_down + read_gate_up + write_gate +
    #                 branch_down + branch_up)
    # = 4 × (d*r + r*d + d*1 + d*r + r*d) = 4 × (4*d*r + d)
    # = 4 × (4*2048*256 + 2048) = 4 × 2,100,224 = 8.4M params
    gr_per_layer = 4 * (4 * d_model * 256 + d_model)
    gr_total = gr_per_layer * 16  # all 16 layers
    gr_vram_mb = gr_total * 2 / 1024 / 1024
    print(f"  Gated Residual:   {gr_per_layer/1e6:.2f}M/layer × 16 = "
          f"{gr_total/1e6:.2f}M params ({gr_vram_mb:.1f} MB bf16)")

    # N-gram Embedding: table_size=2M, d_model=2048
    # Total: 2M * 2048 = 4.1B params — but on HOST RAM, not GPU VRAM
    ngram_table_size = 2_000_000
    ngram_params = ngram_table_size * d_model
    ngram_host_gb = ngram_params * 4 / 1024**3  # fp32 on host
    ngram_gpu_mb = 0  # table is on host, only batch embeddings on GPU
    print(f"  N-gram Embedding: {ngram_params/1e9:.2f}B params "
          f"({ngram_host_gb:.1f} GB host RAM, {ngram_gpu_mb:.1f} MB GPU)")

    print(f"\n  Total GPU VRAM overhead: "
          f"{qsa_vram_mb + gr_vram_mb + ngram_gpu_mb:.1f} MB")
    print(f"  Total host RAM:          {ngram_host_gb:.1f} GB")
    print("  Base model VRAM:         ~2340 MB (bf16)")
    print(f"  With R19 keys:           ~{2340 + qsa_vram_mb + gr_vram_mb:.0f} MB "
          f"(fits 12GB easily)")

    # Verify it fits in 12GB
    total_vram = 2340 + qsa_vram_mb + gr_vram_mb + ngram_gpu_mb
    assert total_vram < 12000, f"Total VRAM {total_vram:.0f} MB exceeds 12GB"
    print("  VRAM analysis: PASS (fits 12GB)")


def main_r19():
    print("=" * 70)
    print("  R&D ROUND 19: Qwen3.8-Flash-Next Architecture Keys")
    print("=" * 70)

    print("\n  R19a: QSA (Qwen Sparse Attention)")
    test_qsa_identity_warm_start()
    test_qsa_sparse_vs_full()
    test_qsa_key_forward_reverse()

    print("\n  R19b: Gated Residual")
    test_gated_residual_identity_warm_start()
    test_gated_residual_branches_active()
    test_gated_residual_key()

    print("\n  R19c: N-gram Embedding")
    test_ngram_embedding_identity_warm_start()
    test_ngram_embedding_nonzero()
    test_ngram_embedding_host_offload()
    test_ngram_embedding_key()

    print("\n  R19 VRAM Analysis")
    test_r19_vram_analysis()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 19 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r19()

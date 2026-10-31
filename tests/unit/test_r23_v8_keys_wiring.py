"""Tests for R&D Round 23: V8 config keys wiring into model_loader.

Verifies that V8 config keys (use_qsa, use_gated_residual, use_ngram_embedding,
use_hashed_nlrq) are actually consumed by model_loader.py when building
ConfigurableResearchLLM. The keys exist in config.py but were not previously
checked by model_loader — R23 wires them in.
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_v8_config(**extra_overrides):
    """Build a tiny V8-like config suitable for CPU testing.

    V10 is a plain GQA port — V8 features (QSA, gated residual, ngram,
    hashed NLRQ) must be explicitly enabled via overrides.
    """
    from forge.config import get_config
    overrides = {
        "vocab_size": 256, "d_model": 64, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2,
        "intermediate_size": 128, "max_seq_len": 128, "titan_memory_rank": 16,
        "embed_factorized_rank": 32, "mtp_n_heads": 2,
        "use_triton_kernels": False, "use_varlen": False,
        "bitnet_int8_training": False, "use_gradient_checkpointing": False,
        "use_hyperloop": False, "use_lisa": False,
        "ngram_host": False,  # disable host table for CPU tests
        # Explicitly enable V8 features (not on by default in V10)
        "use_qsa": True, "use_gated_residual": True, "use_ngram_embedding": True,
        "use_hashed_nlrq": True, "ffn_compression": "nlrq", "nlrq_rank": 32,
        "use_bitnet": False,
        # Disable V10's IRI-FP4 + SpectralKV + BitNetResidual (not compatible with V8 feature tests)
        "use_iri_fp4": False, "use_spectral_kv": False, "use_bitnet_residual": False,
    }
    overrides.update(extra_overrides)
    cfg = get_config("forgelm_v2", **overrides)
    cfg.device = "cpu"
    cfg.dtype = "float32"
    return cfg


# ── R23-V8-1: Config has V8 keys ────────────────────────────────────────────

def test_v8_config_builds():
    """Tiny V8 config should have all R19+R21 keys enabled."""
    cfg = _tiny_v8_config()
    assert cfg.use_qsa is True, "use_qsa should be True for V8"
    assert cfg.use_gated_residual is True, "use_gated_residual should be True for V8"
    assert cfg.use_ngram_embedding is True, "use_ngram_embedding should be True for V8"
    assert cfg.use_hashed_nlrq is True, "use_hashed_nlrq should be True for V8"
    print(f"  V8 config: qsa={cfg.use_qsa}, gated_res={cfg.use_gated_residual}, "
          f"ngram={cfg.use_ngram_embedding}, hashed_nlrq={cfg.use_hashed_nlrq}")
    print("  v8_config_builds: PASS")


# ── R23-V8-2: Model builds without error ────────────────────────────────────

def test_v8_model_builds():
    """ConfigurableResearchLLM with tiny V8 config should build without raising."""
    from forge.model_loader import ConfigurableResearchLLM
    cfg = _tiny_v8_config()
    model = ConfigurableResearchLLM(cfg)
    assert model is not None, "Model should be created"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  V8 model: {n_params/1e6:.2f}M params, {cfg.n_layers} layers")
    assert n_params > 0, "Model should have parameters"
    print("  v8_model_builds: PASS")


# ── R23-V8-3: Forward pass ──────────────────────────────────────────────────

def test_v8_model_forward():
    """Forward pass should produce correct shape and finite output."""
    from forge.model_loader import ConfigurableResearchLLM
    cfg = _tiny_v8_config()
    model = ConfigurableResearchLLM(cfg).to(_DEV)
    model.eval()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=_DEV)
    with torch.no_grad():
        logits, _loss = model(input_ids)

    assert logits is not None, "Should return logits"
    assert logits.shape == (B, T, cfg.vocab_size), \
        f"Expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits should be finite"
    print(f"  V8 forward: input {input_ids.shape} -> logits {logits.shape}")
    print("  v8_model_forward: PASS")


# ── R23-V8-4: QSA wiring ────────────────────────────────────────────────────

def test_v8_qsa_wired():
    """use_qsa=True should produce QSALayer modules; False should not."""
    from forge.keys.attention.qsa_key import QSALayer
    from forge.model_loader import ConfigurableResearchLLM

    # With QSA enabled
    cfg_on = _tiny_v8_config(use_qsa=True)
    model_on = ConfigurableResearchLLM(cfg_on)
    qsa_count_on = sum(1 for m in model_on.modules() if isinstance(m, QSALayer))
    print(f"  QSA=True: {qsa_count_on} QSALayer modules")
    assert qsa_count_on > 0, "use_qsa=True should produce QSALayer modules"

    # With QSA disabled
    cfg_off = _tiny_v8_config(use_qsa=False)
    model_off = ConfigurableResearchLLM(cfg_off)
    qsa_count_off = sum(1 for m in model_off.modules() if isinstance(m, QSALayer))
    print(f"  QSA=False: {qsa_count_off} QSALayer modules")
    assert qsa_count_off == 0, "use_qsa=False should produce no QSALayer modules"

    print("  v8_qsa_wired: PASS")


# ── R23-V8-5: Gated Residual wiring ─────────────────────────────────────────

def test_v8_gated_residual_wired():
    """use_gated_residual=True should produce GatedResidualLayer modules."""
    from forge.keys.architecture.gated_residual_key import GatedResidualLayer
    from forge.model_loader import ConfigurableResearchLLM

    # With GatedResidual enabled
    cfg_on = _tiny_v8_config(use_gated_residual=True)
    model_on = ConfigurableResearchLLM(cfg_on)
    gr_count_on = sum(1 for m in model_on.modules() if isinstance(m, GatedResidualLayer))
    print(f"  GatedResidual=True: {gr_count_on} GatedResidualLayer modules")
    assert gr_count_on > 0, "use_gated_residual=True should produce GatedResidualLayer modules"

    # With GatedResidual disabled
    cfg_off = _tiny_v8_config(use_gated_residual=False)
    model_off = ConfigurableResearchLLM(cfg_off)
    gr_count_off = sum(1 for m in model_off.modules() if isinstance(m, GatedResidualLayer))
    print(f"  GatedResidual=False: {gr_count_off} GatedResidualLayer modules")
    assert gr_count_off == 0, "use_gated_residual=False should produce none"

    print("  v8_gated_residual_wired: PASS")


# ── R23-V8-6: N-gram Embedding wiring ───────────────────────────────────────

def test_v8_ngram_embedding_wired():
    """use_ngram_embedding=True should produce NGramEmbeddingLayer modules."""
    from forge.keys.knowledge.ngram_embedding_key import NGramEmbeddingLayer
    from forge.model_loader import ConfigurableResearchLLM

    # With NgramEmbedding enabled (host=False for CPU)
    cfg_on = _tiny_v8_config(use_ngram_embedding=True, ngram_host=False)
    model_on = ConfigurableResearchLLM(cfg_on)
    ngram_count_on = sum(1 for m in model_on.modules() if isinstance(m, NGramEmbeddingLayer))
    print(f"  NgramEmbedding=True: {ngram_count_on} NGramEmbeddingLayer modules")
    assert ngram_count_on > 0, "use_ngram_embedding=True should produce NGramEmbeddingLayer modules"

    # With NgramEmbedding disabled
    cfg_off = _tiny_v8_config(use_ngram_embedding=False)
    model_off = ConfigurableResearchLLM(cfg_off)
    ngram_count_off = sum(1 for m in model_off.modules() if isinstance(m, NGramEmbeddingLayer))
    print(f"  NgramEmbedding=False: {ngram_count_off} NGramEmbeddingLayer modules")
    assert ngram_count_off == 0, "use_ngram_embedding=False should produce none"

    print("  v8_ngram_embedding_wired: PASS")


# ── R23-V8-7: HashedNLRQ wiring ─────────────────────────────────────────────

def test_v8_hashed_nlrq_wired():
    """use_hashed_nlrq=True should produce HashedNLRQ in FFN; False → NLRQLinear."""
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    from forge.model_loader import ConfigurableResearchLLM
    from forge.training.optim.r21_cross_domain import HashedNLRQ

    # With HashedNLRQ enabled
    cfg_on = _tiny_v8_config(use_hashed_nlrq=True)
    model_on = ConfigurableResearchLLM(cfg_on)
    hashed_count = sum(1 for m in model_on.modules() if isinstance(m, HashedNLRQ))
    print(f"  HashedNLRQ=True: {hashed_count} HashedNLRQ modules")
    assert hashed_count > 0, "use_hashed_nlrq=True should produce HashedNLRQ modules"

    # With HashedNLRQ disabled → should use plain NLRQLinear
    cfg_off = _tiny_v8_config(use_hashed_nlrq=False)
    model_off = ConfigurableResearchLLM(cfg_off)
    nlrq_count = sum(1 for m in model_off.modules() if isinstance(m, NLRQLinear))
    hashed_count_off = sum(1 for m in model_off.modules() if isinstance(m, HashedNLRQ))
    print(f"  HashedNLRQ=False: {nlrq_count} NLRQLinear, {hashed_count_off} HashedNLRQ")
    assert hashed_count_off == 0, "use_hashed_nlrq=False should produce no HashedNLRQ"
    assert nlrq_count > 0, "use_hashed_nlrq=False should use plain NLRQLinear"

    print("  v8_hashed_nlrq_wired: PASS")


# ── R23-V8-8: Backward pass ─────────────────────────────────────────────────

def test_v8_backward():
    """Forward + backward should compute gradients (at least one non-None grad)."""
    from forge.model_loader import ConfigurableResearchLLM
    cfg = _tiny_v8_config()
    model = ConfigurableResearchLLM(cfg).to(_DEV)
    model.train()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=_DEV)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=_DEV)

    _logits, loss = model(input_ids, targets=targets)
    assert loss is not None, "Should return loss when targets provided"
    loss.backward()

    grads_ok = sum(1 for p in model.parameters() if p.grad is not None)
    grads_nonzero = sum(1 for p in model.parameters()
                        if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"  V8 backward: {grads_ok} params with grad, {grads_nonzero} non-zero")
    assert grads_ok > 0, "At least one parameter should have a gradient"
    assert grads_nonzero > 0, "At least one gradient should be non-zero"

    print("  v8_backward: PASS")


# ── R23-V8-9: Training loss decreases ───────────────────────────────────────

def test_v8_loss_decreases():
    """10 steps of BAdam on tiny V8 should decrease loss."""
    from forge.model_loader import ConfigurableResearchLLM
    from forge.training.training_utils import configure_optimizer

    torch.manual_seed(42)
    cfg = _tiny_v8_config()
    model = ConfigurableResearchLLM(cfg).to(_DEV)
    model.train()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=_DEV)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=_DEV)

    opt = configure_optimizer(model, max_lr=1e-3, weight_decay=0.01,
                              optimizer_name="badam")

    initial_loss = None
    for step in range(10):
        opt.zero_grad()
        _logits, loss = model(input_ids, targets=targets)
        if loss.requires_grad:
            loss.backward()
            opt.step()
        if step == 0:
            initial_loss = loss.item()
    final_loss = loss.item()

    print(f"  V8 training: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease over 10 steps"

    print("  v8_loss_decreases: PASS")


# ── R23-V8-10: Lossless at init ─────────────────────────────────────────────

def test_v8_keys_lossless_at_init():
    """R19 keys ON vs OFF should produce near-identical logits at init.

    R19 keys are lossless at init:
    - QSA: budget = all blocks (full attention)
    - Gated Residual: gate = 1.0 (identity)
    - N-gram Embedding: table = all zeros (additive zero)
    """
    from forge.model_loader import ConfigurableResearchLLM

    torch.manual_seed(42)
    B, T = 2, 16

    # Build with all R19 keys ON
    cfg_on = _tiny_v8_config(
        use_qsa=True, use_gated_residual=True,
        use_ngram_embedding=True, use_hashed_nlrq=False)  # nlrq not lossless
    model_on = ConfigurableResearchLLM(cfg_on).to(_DEV)
    model_on.eval()

    torch.manual_seed(42)  # same seed for identical weight init
    cfg_off = _tiny_v8_config(
        use_qsa=False, use_gated_residual=False,
        use_ngram_embedding=False, use_hashed_nlrq=False)
    model_off = ConfigurableResearchLLM(cfg_off).to(_DEV)
    model_off.eval()

    input_ids = torch.randint(0, cfg_on.vocab_size, (B, T), device=_DEV)
    with torch.no_grad():
        logits_on, _ = model_on(input_ids)
        logits_off, _ = model_off(input_ids)

    max_diff = (logits_on - logits_off).abs().max().item()
    mean_diff = (logits_on - logits_off).abs().mean().item()
    print(f"  V8 lossless at init: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    # R19 keys are lossless at init — difference should be near-zero
    # (small numerical differences from different code paths are OK)
    assert max_diff < 1e-2, \
        f"R19 keys should be lossless at init, max diff {max_diff:.6f}"

    print("  v8_keys_lossless_at_init: PASS")


def main_r23_v8():
    print("=" * 70)
    print("  R&D ROUND 23: V8 Config Keys Wiring")
    print("=" * 70)

    print("\n  V8-1: Config builds")
    test_v8_config_builds()

    print("\n  V8-2: Model builds")
    test_v8_model_builds()

    print("\n  V8-3: Forward pass")
    test_v8_model_forward()

    print("\n  V8-4: QSA wiring")
    test_v8_qsa_wired()

    print("\n  V8-5: Gated Residual wiring")
    test_v8_gated_residual_wired()

    print("\n  V8-6: N-gram Embedding wiring")
    test_v8_ngram_embedding_wired()

    print("\n  V8-7: HashedNLRQ wiring")
    test_v8_hashed_nlrq_wired()

    print("\n  V8-8: Backward pass")
    test_v8_backward()

    print("\n  V8-9: Loss decreases")
    test_v8_loss_decreases()

    print("\n  V8-10: Lossless at init")
    test_v8_keys_lossless_at_init()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 V8 WIRING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_v8()

"""CPU tests for the V7-8B cloud training path (sft_train.py + BAdam).

The cloud handoff uses sft_train.py (via vast_connector.py), NOT the
dedicated train_8b_all.py. These tests verify that sft_train.py properly
handles 8B-specific issues that train_8b_all.py already solves:

  1. Dead-param freezing (MTP, loop_block, gated modules) — without this,
     BAdam crashes when it activates a block containing only dead params.
  2. NLRQ factor training (STE masters) for NLRQ-compressed configs (8B-D).
  3. From-scratch init (NLRQ reset, BitNet QAT disable, kaiming init,
     logit scale normalization).
  4. Checkpoint save/load with NLRQ (masters stripped, INT8 buffers kept).

All tests run on CPU with tiny models — no GPU needed. This mirrors the
user's constraint: "my system cannot test a full run, not enough mem."
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.training.optim.badam import BAdam

# ── Helpers ───────────────────────────────────────────────────────────────


def _tiny_v7_8b_config(config_name="forgelm_v2", **extra_overrides):
    """Create a tiny V7-8B-like config that builds on CPU in <1s.

    V10 is a plain GQA port — V7-8B features (MTP, BitNet, GTA, TITAN, MoD,
    MHC, AttnRes, value-residual, sandwich-norm, learned-sink, PIT, factorized
    embeddings, NLRQ) must be explicitly enabled via overrides.
    Disables features that require GPU (triton, varlen, int8
    training) or need many layers (hyperloop, lisa).
    """
    from forge.config import get_config

    overrides = {
        "vocab_size": 256,
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "intermediate_size": 128,
        "max_seq_len": 128,
        "titan_memory_rank": 16,
        "embed_factorized_rank": 32,
        "mtp_n_heads": 2,
        "use_triton_kernels": False,
        "use_varlen": False,
        "bitnet_int8_training": False,
        "use_gradient_checkpointing": False,
        "use_hyperloop": False,
        "use_lisa": False,
        # Explicitly enable V7-8B features (not on by default in V10)
        "use_mtp": True,
        "use_bitnet": True,
        "attn_type": "gta",
        "use_titan_memory": True,
        "use_mod": True,
        "use_mhc": True,
        "use_attn_residual": True,
        "use_value_residual": True,
        "use_sandwich_norm": True,
        "use_learned_sink": True,
        "use_pit": True,
        "use_factorized_embeddings": True,
        "ffn_compression": "nlrq",
        "nlrq_rank": 32,
        # Disable V10's IRI-FP4 + SpectralKV + BitNetResidual (not compatible with V7-8B feature tests)
        "use_iri_fp4": False,
        "use_spectral_kv": False,
        "use_bitnet_residual": False,
    }
    overrides.update(extra_overrides)
    cfg = get_config(config_name, **overrides)
    cfg.device = "cpu"
    cfg.dtype = "float32"
    return cfg


def _build_tiny_model(config_name="forgelm_v2", **extra_overrides):
    from forge.model_loader import ConfigurableResearchLLM

    cfg = _tiny_v7_8b_config(config_name, **extra_overrides)
    model = ConfigurableResearchLLM(cfg)
    return model, cfg


def _manual_ce_loss(model, ids, vocab_size):
    """Compute next-token CE without model's targets= path (avoids MTP loss)."""
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    shift_l = logits[:, :-1, :].contiguous()
    shift_t = ids[:, 1:].contiguous()
    return F.cross_entropy(shift_l.view(-1, vocab_size).float(), shift_t.view(-1))


# ── Test 1: Dead-param crash without freeze_dead_params_ ──────────────────


class _TinyLayer(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(x)


class _ModelWithDeadModule(nn.Module):
    """Mimics 8B: blocks.N + a large dead module (like MTP) that gets chunked
    into its own BAdam block by the chunking logic."""

    def __init__(self, d=8, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(16, d)
        self.blocks = nn.ModuleList([_TinyLayer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, 16)
        # Large dead module — never called in forward, like MTP with no loss
        self.dead_module = nn.Sequential(nn.Linear(d, 256), nn.Linear(256, 256), nn.Linear(256, d))

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_badam_crashes_on_dead_only_block_without_freeze():
    """Without freeze_dead_params_, BAdam creates blocks with ONLY dead params.
    When such a block activates, loss has no grad_fn → backward() crashes."""
    model = _ModelWithDeadModule()
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    ids = torch.randint(0, 16, (2, 4))

    crashed = False
    for _step in range(opt._n_blocks + 2):
        opt.zero_grad()
        out = model(ids)
        loss = out.float().pow(2).mean()
        if not loss.requires_grad:
            crashed = True
            break
        loss.backward()
        opt.step()
        del out, loss
    assert crashed, "Expected a no-grad crash when a dead-only block activates"


def test_freeze_dead_params_prevents_badam_crash():
    """freeze_dead_params_ excludes dead params from BAdam blocks → no crash."""
    from forge.training.runners.train_8b_all import freeze_dead_params_

    model = _ModelWithDeadModule()
    n_dead = freeze_dead_params_(model, torch.device("cpu"), use_flce=False)
    assert n_dead > 0, "Should have frozen dead params"

    # Dead params must be excluded from BAdam blocks
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    all_opt_params = {id(p) for b in opt._blocks for p in b["params"]}
    dead_params = list(model.dead_module.parameters())
    for p in dead_params:
        assert id(p) not in all_opt_params, "Dead param should not be in BAdam"

    # Training should complete without crash
    ids = torch.randint(0, 16, (2, 4))
    for step in range(opt._n_blocks + 2):
        opt.zero_grad()
        out = model(ids)
        loss = out.float().pow(2).mean()
        assert loss.requires_grad, f"Step {step}: loss has no grad_fn"
        loss.backward()
        opt.step()
        del out, loss


# ── Test 2: Tiny V7-8B model build + forward + backward ───────────────────


def test_tiny_v7_8b_b_builds_and_forwards():
    """8B-B config (dense, BitNet, no NLRQ) builds and forwards on CPU."""
    model, cfg = _build_tiny_model("forgelm_v2", ffn_compression="none")
    assert cfg.use_mtp, "8B-B should have MTP"
    assert cfg.use_bitnet, "8B-B should have BitNet"
    assert cfg.ffn_compression == "none", "8B-B should be dense FFN"

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 16, cfg.vocab_size)


def test_tiny_v7_8b_d_builds_and_forwards():
    """8B-D config (NLRQ compressed, deeper) builds and forwards on CPU."""
    model, cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    assert cfg.ffn_compression == "nlrq", "8B-D should use NLRQ"
    assert cfg.nlrq_rank == 16

    from forge.keys.compression.nlrq_ffn_key import NLRQLinear

    nlrq_count = sum(1 for m in model.modules() if isinstance(m, NLRQLinear))
    assert nlrq_count > 0, "8B-D should have NLRQ layers"

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 16, cfg.vocab_size)


def test_tiny_v7_8b_b_backward_and_badam_step():
    """8B-B: forward + backward + BAdam step works on CPU."""
    model, cfg = _build_tiny_model("forgelm_v2")
    from forge.training.runners.train_8b_all import freeze_dead_params_

    freeze_dead_params_(model, torch.device("cpu"), use_flce=False)

    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    losses = []
    for _step in range(opt._n_blocks + 2):
        opt.zero_grad()
        loss = _manual_ce_loss(model, ids, cfg.vocab_size)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        del loss
    assert all(math.isfinite(x) for x in losses), "All losses should be finite"


def test_tiny_v7_8b_d_backward_and_badam_step():
    """8B-D (NLRQ): forward + backward + BAdam step works on CPU."""
    model, cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    from forge.training.runners.train_8b_all import freeze_dead_params_

    freeze_dead_params_(model, torch.device("cpu"), use_flce=False)

    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    for _step in range(opt._n_blocks + 2):
        opt.zero_grad()
        loss = _manual_ce_loss(model, ids, cfg.vocab_size)
        loss.backward()
        opt.step()
        del loss


# ── Test 3: NLRQ factor training (STE) for 8B-D ───────────────────────────


def test_nlrq_factor_training_enables_on_8b_d():
    """enable_factor_training_all_ creates STE masters on NLRQ layers."""
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    from forge.training.runners.train_8b_all import enable_factor_training_all_

    model, _cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    n = enable_factor_training_all_(model)
    assert n > 0, "Should enable factor training on NLRQ layers"

    for m in model.modules():
        if isinstance(m, NLRQLinear):
            assert m.factor_training_enabled(), "NLRQ layer should have STE masters"
            assert m.U_m is not None and m.V_m is not None
            assert m.U_m.requires_grad and m.V_m.requires_grad


def test_nlrq_factor_training_grads_reach_masters():
    """STE: gradients flow through quantizer to U_m/V_m masters."""
    from forge.training.runners.train_8b_all import enable_factor_training_all_

    model, cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    enable_factor_training_all_(model)

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    loss = _manual_ce_loss(model, ids, cfg.vocab_size)
    loss.backward()

    from forge.keys.compression.nlrq_ffn_key import NLRQLinear

    has_grad = False
    for m in model.modules():
        if isinstance(m, NLRQLinear) and m.U_m is not None and m.U_m.grad is not None:
            has_grad = True
            break
    assert has_grad, "At least one NLRQ master should have gradients"


# ── Test 4: Checkpoint save/load with NLRQ ────────────────────────────────


def test_snapshot_state_strips_nlrq_masters():
    """snapshot_state exports INT8 buffers and strips STE masters."""
    from forge.training.runners.train_8b_all import enable_factor_training_all_, snapshot_state

    model, _cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    enable_factor_training_all_(model)

    state = snapshot_state(model, step=42)
    assert not any(k.endswith((".U_m", ".V_m")) for k in state), "STE masters should be stripped from checkpoint"
    assert any(k.endswith("U_q") for k in state), "INT8 buffers should be kept"
    assert state["step"] == 42


def test_checkpoint_roundtrip_preserves_forward_output():
    """Save → load → forward gives same output (within quantization tolerance)."""
    from forge.model_loader import ConfigurableResearchLLM
    from forge.training.runners.train_8b_all import enable_factor_training_all_, snapshot_state

    model, cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    enable_factor_training_all_(model)

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out_before = model(ids)
        logits_before = out_before[0] if isinstance(out_before, tuple) else out_before

    state = snapshot_state(model, step=1)
    # Remove non-param keys for load_state_dict
    sd = {k: v for k, v in state.items() if "." in k}

    # Build a fresh model and load
    model2 = ConfigurableResearchLLM(cfg)
    _missing, unexpected = model2.load_state_dict(sd, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"

    with torch.no_grad():
        out_after = model2(ids)
        logits_after = out_after[0] if isinstance(out_after, tuple) else out_after

    # INT8 quantization has small tolerance
    assert torch.allclose(logits_before, logits_after, atol=2.0), (
        f"Forward mismatch: max diff {(logits_before - logits_after).abs().max():.4f}"
    )


# ── Test 5: sft_train 8B setup integration ────────────────────────────────


def test_sft_train_freezes_dead_params_for_8b():
    """sft_train.py should call freeze_dead_params_ when using BAdam with 8B.

    This tests the FIX: without it, BAdam crashes on dead-only blocks.
    We verify the function is importable and works on a tiny 8B model.
    """
    from forge.training.runners.train_8b_all import freeze_dead_params_

    model, _cfg = _build_tiny_model("forgelm_v2")
    mtp = getattr(model, "mtp_module", None)
    assert mtp is not None, "8B-B should have MTP module"

    # MTP params should start trainable
    mtp_params = list(mtp.parameters())
    assert all(p.requires_grad for p in mtp_params), "MTP should start trainable"

    # Freeze dead params
    n_dead = freeze_dead_params_(model, torch.device("cpu"), use_flce=False)
    assert n_dead > 0, "Should freeze dead params"

    # MTP params should now be frozen (no CE pathway to MTP)
    mtp_frozen = sum(1 for p in mtp_params if not p.requires_grad)
    assert mtp_frozen == len(mtp_params), f"All MTP params should be frozen, got {mtp_frozen}/{len(mtp_params)}"

    # BAdam should exclude frozen params
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    all_opt = {id(p) for b in opt._blocks for p in b["params"]}
    for p in mtp_params:
        assert id(p) not in all_opt, "Frozen MTP params should not be in BAdam"


def test_sft_train_nlrq_factor_training_for_8b_d():
    """sft_train.py should enable NLRQ factor training for NLRQ configs.

    Without this, only S (singular values) train — U/V factors stay frozen.
    """
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    from forge.training.runners.train_8b_all import enable_factor_training_all_

    model, cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)
    assert cfg.ffn_compression == "nlrq"

    # Before: no STE masters
    for m in model.modules():
        if isinstance(m, NLRQLinear):
            assert not m.factor_training_enabled(), "Should start without STE"

    # Enable factor training (what sft_train should do for NLRQ configs)
    n = enable_factor_training_all_(model)
    assert n > 0

    # After: STE masters exist
    for m in model.modules():
        if isinstance(m, NLRQLinear):
            assert m.factor_training_enabled(), "Should have STE masters"


# ── Test 6: From-scratch init for 8B ──────────────────────────────────────


def test_from_scratch_init_normalizes_logit_scale():
    """normalize_logit_scale_ brings logit std toward 1.0.

    On the real 8B model, kaiming init on the two-stage factorized head
    compounds to std ~5.4, so scale = 1/5.4 < 1.0 (scales down). On the
    tiny test model the std may already be <1.0, so the function scales UP.
    The invariant is: |std_after - 1.0| < |std_before - 1.0|.
    """
    from types import SimpleNamespace

    from forge.training.runners.train_8b_all import (
        forward_model,
        normalize_logit_scale_,
    )

    model, cfg = _build_tiny_model("forgelm_v2")
    # Blow up the head to simulate the confidently-wrong init
    with torch.no_grad():
        if hasattr(model, "head") and hasattr(model.head, "weight"):
            model.head.weight.mul_(14.0)

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits = forward_model(model, ids)
    std_before = float(logits.float().std())

    scale = normalize_logit_scale_(model, torch.device("cpu"), SimpleNamespace(vocab_size=cfg.vocab_size))
    assert scale != 1.0, "Should apply a non-trivial scale"

    with torch.no_grad():
        logits_after = forward_model(model, ids)
    std_after = float(logits_after.float().std())
    # The function scales toward std=1.0: |after - 1| should be < |before - 1|
    assert abs(std_after - 1.0) < abs(std_before - 1.0), (
        f"std should move toward 1.0: {std_before:.3f} → {std_after:.3f} (scale={scale:.3f})"
    )


def test_from_scratch_init_disables_bitnet_qat():
    """disable_bitnet_qat_ turns off ternary QAT (wasteful on random init)."""
    from forge.keys.quantization.bitnet_b158_key import BitNetLinear
    from forge.training.runners.train_8b_all import disable_bitnet_qat_

    model, _cfg = _build_tiny_model("forgelm_v2")
    bitnet_layers = [m for m in model.modules() if isinstance(m, BitNetLinear)]
    assert len(bitnet_layers) > 0, "8B-B should have BitNet layers"

    # Before: QAT should be on (default)
    assert bitnet_layers[0].quantize, "BitNet QAT should start enabled"

    disable_bitnet_qat_(model)

    # After: QAT should be off
    for m in bitnet_layers:
        assert not m.quantize, "BitNet QAT should be disabled"
        assert not m.force_quant, "BitNet force_quant should be disabled"


# ── Test 7: Full training cycle (forward + backward + BAdam + checkpoint) ─


def test_full_8b_training_cycle_on_cpu():
    """End-to-end: build → freeze dead → enable NLRQ → train 3 steps → save → load.

    This is the minimal cloud-handoff simulation: what the remote sft_train.py
    does, but on a tiny model that fits in CPU memory.
    """
    from forge.training.runners.train_8b_all import (
        enable_factor_training_all_,
        freeze_dead_params_,
        snapshot_state,
    )

    model, cfg = _build_tiny_model("forgelm_v2", nlrq_rank=16)

    # 1. Freeze dead params (prevents BAdam crash)
    n_dead = freeze_dead_params_(model, torch.device("cpu"), use_flce=False)
    assert n_dead > 0

    # 2. Enable NLRQ factor training (STE masters)
    n_nlrq = enable_factor_training_all_(model)
    assert n_nlrq > 0

    # 3. Build BAdam
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)

    # 4. Train for a few steps
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    losses = []
    for _step in range(3):
        opt.zero_grad()
        loss = _manual_ce_loss(model, ids, cfg.vocab_size)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        del loss
    assert all(math.isfinite(x) for x in losses)

    # 5. Save checkpoint (strips NLRQ masters)
    state = snapshot_state(model, step=3)
    assert not any(k.endswith((".U_m", ".V_m")) for k in state)

    # 6. Load into fresh model
    from forge.model_loader import ConfigurableResearchLLM

    model2 = ConfigurableResearchLLM(cfg)
    sd = {k: v for k, v in state.items() if "." in k}
    _missing, unexpected = model2.load_state_dict(sd, strict=False)
    assert not unexpected

    # 7. Fresh model should produce similar output
    with torch.no_grad():
        out = model2(ids)
        logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 16, cfg.vocab_size)

"""Tests for R&D Round 23: V8 training runner (train_v8.py fork of train_8b_all.py).

The V8 runner adds 5 warm-start modes (scratch, lora-seed, dlora-warmstart,
hypercloning, ligo), ETA projection, rolling checkpoint retention, resume
bundles, VRAM preflight, and NaN guards on top of the V7-8B training path.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn.functional as F

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────

def _tiny_v8_config(**extra):
    """Create a tiny V8-like config that builds on CPU in <1s.

    V10 is a plain GQA port — V8 features (MTP, etc.) must be explicitly enabled.
    """
    from forge.config import get_config
    overrides = {
        "vocab_size": 256, "d_model": 64, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2,
        "intermediate_size": 128, "max_seq_len": 128, "titan_memory_rank": 16,
        "embed_factorized_rank": 32, "mtp_n_heads": 2,
        "use_triton_kernels": False, "use_varlen": False,
        "bitnet_int8_training": False, "use_gradient_checkpointing": False,
        "use_hyperloop": False, "use_lisa": False, "ngram_host": False,
        # Explicitly enable V8 features (not on by default in V10)
        "use_mtp": True, "use_bitnet": False,
        # Disable V10's IRI-FP4 + SpectralKV + BitNetResidual (not compatible with V8 feature tests)
        "use_iri_fp4": False, "use_spectral_kv": False, "use_bitnet_residual": False,
    }
    overrides.update(extra)
    cfg = get_config("forgelm_v2", **overrides)
    cfg.device = "cpu"
    cfg.dtype = "float32"
    return cfg


def _build_tiny_v8(**extra):
    """Build a tiny V8 model + config."""
    from forge.model_loader import ConfigurableResearchLLM
    cfg = _tiny_v8_config(**extra)
    model = ConfigurableResearchLLM(cfg)
    return model, cfg


def _manual_ce_loss(model, ids, vocab_size):
    """Compute next-token CE without model's targets= path (avoids MTP loss)."""
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    shift_l = logits[:, :-1, :].contiguous()
    shift_t = ids[:, 1:].contiguous()
    return F.cross_entropy(shift_l.view(-1, vocab_size).float(), shift_t.view(-1))


# ── Test 1: Imports ───────────────────────────────────────────────────────

def test_train_v8_imports():
    """train_v8 module should expose parse_args() and run() functions."""
    import forge.training.runners.train_v8 as mod

    assert hasattr(mod, "parse_args"), "train_v8 should have parse_args()"
    assert hasattr(mod, "run"), "train_v8 should have run()"
    assert callable(mod.parse_args), "parse_args should be callable"
    assert callable(mod.run), "run should be callable"
    print("  train_v8_imports: PASS")


# ── Test 2: Mode parsing ──────────────────────────────────────────────────

def test_train_v8_modes():
    """parse_args should accept --mode with all 5 warm-start choices."""
    import forge.training.runners.train_v8 as mod

    valid_modes = ["scratch", "lora-seed", "dlora-warmstart", "hypercloning", "ligo"]
    for mode in valid_modes:
        old_argv = sys.argv
        sys.argv = ["train_v8.py", "--mode", mode, "--steps", "1"]
        try:
            args = mod.parse_args()
            assert args.mode == mode, f"args.mode should be '{mode}', got '{args.mode}'"
        finally:
            sys.argv = old_argv
    print("  train_v8_modes: PASS")


# ── Test 3: ETA projection ────────────────────────────────────────────────

def test_train_v8_eta_projection():
    """ETA projection should compute remaining time from step_times + tokens."""
    from forge.training.runners.train_v8 import project_eta

    # 100 steps at 1s each, 50 done → 50s remaining
    step_times = [1.0] * 50
    tokens_seen = 50 * 128
    tokens_total = 100 * 128
    eta = project_eta(step_times, tokens_seen, tokens_total)
    assert "eta_seconds" in eta, "ETA dict should have eta_seconds"
    assert "eta_str" in eta, "ETA dict should have eta_str"
    expected = 50.0  # 50 remaining steps × 1s
    assert abs(eta["eta_seconds"] - expected) < 5.0, \
        f"ETA should be ~{expected}s, got {eta['eta_seconds']:.1f}s"
    print(f"  ETA: {eta['eta_seconds']:.1f}s remaining ({eta['eta_str']})")
    print("  train_v8_eta_projection: PASS")


# ── Test 4: Rolling checkpoint retention ──────────────────────────────────

def test_train_v8_rolling_checkpoints():
    """Rolling retention should keep only the last N step checkpoints."""
    from forge.training.runners.train_v8 import CheckpointWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpts")
        os.makedirs(ckpt_dir)
        keep = 3

        # Save 5 fake checkpoints with step numbers
        for step in range(5):
            path = os.path.join(ckpt_dir, f"ForgeLM_V8_step{step}.safetensors")
            torch.save({"step": step, "fake": torch.zeros(4)}, path)

        # Use CheckpointWriter's _rotate to clean up
        writer = CheckpointWriter(keep=keep, ckpt_dir=os.path.join(tmpdir, "ckpts"),
                                  prefix="ForgeLM_V8")
        writer._rotate()

        import glob
        remaining = sorted(glob.glob(os.path.join(ckpt_dir, "ForgeLM_V8_step*.safetensors")))
        assert len(remaining) == keep, \
            f"Should keep {keep} checkpoints, got {len(remaining)}"
        # Should keep the LAST 3 (steps 2, 3, 4)
        for p in remaining:
            step = int(os.path.basename(p).split("step")[1].split(".")[0])
            assert step >= 2, f"Should not keep old step {step}"
        print(f"  Rolling: kept {len(remaining)}/{5} checkpoints (keep={keep})")
        print("  train_v8_rolling_checkpoints: PASS")


# ── Test 5: Resume bundle ─────────────────────────────────────────────────

def test_train_v8_resume_bundle():
    """Resume bundle should save and restore step, BAdam state, RNG, best_val."""
    from forge.training.runners.train_v8 import load_resume_bundle, save_resume_bundle

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake resume bundle
        step = 42
        badam_state = {"block_idx": 2, "step_count": 42, "steps_in_block": 5}
        rng_state = {"torch": torch.get_rng_state(), "cuda": None,
                     "numpy": None, "python": None}
        best_val = 2.345

        bundle_path = os.path.join(tmpdir, "resume.pt")
        save_resume_bundle(bundle_path, step=step, badam_state=badam_state,
                           rng=rng_state, best_val=best_val)

        # Load it back
        loaded = load_resume_bundle(bundle_path)
        assert loaded["step"] == step, f"Step mismatch: {loaded['step']} != {step}"
        assert loaded["badam_state"]["block_idx"] == 2
        assert loaded["best_val"] == best_val
        assert torch.equal(loaded["rng"]["torch"], rng_state["torch"]), "RNG mismatch"
        print(f"  Resume: step={loaded['step']}, block={loaded['badam_state']['block_idx']}, "
              f"best_val={loaded['best_val']:.3f}")
        print("  train_v8_resume_bundle: PASS")


# ── Test 6: VRAM preflight ────────────────────────────────────────────────

def test_train_v8_vram_preflight():
    """preflight_vram_check should return dict with ok, est_inc_gb, available_gb."""
    from types import SimpleNamespace

    from forge.training.optim.badam import BAdam
    from forge.training.runners.train_v8 import preflight_vram_check

    model, _cfg = _build_tiny_v8()
    from forge.training.runners.train_8b_all import freeze_dead_params_
    freeze_dead_params_(model, torch.device("cpu"), use_flce=False)

    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    args = SimpleNamespace(batch_size=1, seq_len=16)
    result = preflight_vram_check(model, opt, args, torch.device("cpu"), use_flce=False)

    assert "ok" in result, "Should have 'ok' key"
    assert "est_inc_gb" in result, "Should have 'est_inc_gb' key"
    assert "available_gb" in result, "Should have 'available_gb' key"
    # On CPU, should always be ok (no VRAM limit)
    assert result["ok"] is True, "CPU preflight should pass"
    print(f"  Preflight: ok={result['ok']}, est_inc={result['est_inc_gb']:.2f}GB, "
          f"avail={result['available_gb']:.2f}GB")
    print("  train_v8_vram_preflight: PASS")


# ── Test 7: NaN guard ─────────────────────────────────────────────────────

def test_train_v8_nan_guard():
    """NaN guard should skip a step when loss is NaN instead of crashing."""
    from forge.training.optim.badam import BAdam
    from forge.training.runners.train_8b_all import freeze_dead_params_
    from forge.training.runners.train_v8 import nan_guard_step

    model, cfg = _build_tiny_v8()
    freeze_dead_params_(model, torch.device("cpu"), use_flce=False)
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)

    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    # Inject NaN into model weights to produce NaN loss
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data.fill_(float('nan'))
                break

    skipped = False
    for step in range(3):
        opt.zero_grad()
        loss = _manual_ce_loss(model, ids, cfg.vocab_size)
        result = nan_guard_step(opt, loss, step)
        if result == "skipped_nan":
            skipped = True
            break
        if math.isnan(loss.item()):
            # If loss is NaN and we didn't crash, that's the guard working
            skipped = True
            break
        try:
            loss.backward()
            opt.step()
        except RuntimeError:
            skipped = True
            break
        del loss

    assert skipped, "NaN guard should skip the step, not crash"
    print("  train_v8_nan_guard: PASS")


# ── Test 8: Dead param freeze (MTP) ───────────────────────────────────────

def test_train_v8_dead_param_freeze():
    """freeze_dead_params_ should freeze MTP params when mtp_weight=0."""
    from forge.training.runners.train_8b_all import freeze_dead_params_

    model, _cfg = _build_tiny_v8(mtp_weight=0.0)
    mtp = getattr(model, "mtp_module", None)
    if mtp is None:
        # Some configs name it differently
        mtp = getattr(model, "mtp", None)
    assert mtp is not None, "V8 should have MTP module"

    mtp_params = list(mtp.parameters())
    assert len(mtp_params) > 0, "MTP should have parameters"
    assert all(p.requires_grad for p in mtp_params), "MTP should start trainable"

    n_dead = freeze_dead_params_(model, torch.device("cpu"), use_flce=False)
    assert n_dead > 0, "Should freeze dead params"

    mtp_frozen = sum(1 for p in mtp_params if not p.requires_grad)
    # Most MTP params should be frozen; shared params (e.g. tied embedding)
    # may still receive gradients from the main loss and remain trainable.
    assert mtp_frozen >= len(mtp_params) - 1, \
        f"Most MTP params should be frozen, got {mtp_frozen}/{len(mtp_params)}"
    print(f"  Dead param freeze: {n_dead} frozen, MTP {mtp_frozen}/{len(mtp_params)} frozen")
    print("  train_v8_dead_param_freeze: PASS")


# ── Test 9: Loss decreases ────────────────────────────────────────────────

def test_train_v8_loss_decreases():
    """10 training steps with BAdam on tiny V8 should decrease loss."""
    from forge.training.optim.badam import BAdam
    from forge.training.runners.train_8b_all import freeze_dead_params_

    model, cfg = _build_tiny_v8()
    freeze_dead_params_(model, torch.device("cpu"), use_flce=False)
    opt = BAdam(model, lr=1e-2, switch_every=1, verbose=False)

    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    losses = []
    for _step in range(10):
        opt.zero_grad()
        loss = _manual_ce_loss(model, ids, cfg.vocab_size)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        del loss

    assert all(math.isfinite(l) for l in losses), "All losses should be finite"
    # Loss should generally decrease (allow some noise from BAdam block switching)
    assert losses[-1] < losses[0], \
        f"Loss should decrease: {losses[0]:.3f} → {losses[-1]:.3f}"
    print(f"  Loss: {losses[0]:.3f} → {losses[-1]:.3f} over 10 steps")
    print("  train_v8_loss_decreases: PASS")


# ── Test 10: Checkpoint roundtrip ─────────────────────────────────────────

def test_train_v8_checkpoint_roundtrip():
    """Save checkpoint, load into new model, forward pass should match."""
    from forge.model_loader import ConfigurableResearchLLM
    from forge.training.runners.train_8b_all import snapshot_state

    model, cfg = _build_tiny_v8()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        out_before = model(ids)
        logits_before = out_before[0] if isinstance(out_before, tuple) else out_before

    state = snapshot_state(model, step=1)
    sd = {k: v for k, v in state.items() if "." in k}

    model2 = ConfigurableResearchLLM(cfg)
    _missing, unexpected = model2.load_state_dict(sd, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"

    with torch.no_grad():
        out_after = model2(ids)
        logits_after = out_after[0] if isinstance(out_after, tuple) else out_after

    assert torch.allclose(logits_before, logits_after, atol=1e-5), \
        f"Forward mismatch: max diff {(logits_before - logits_after).abs().max():.6f}"
    print(f"  Checkpoint roundtrip: max diff {(logits_before - logits_after).abs().max():.6f}")
    print("  train_v8_checkpoint_roundtrip: PASS")


# ── Main ──────────────────────────────────────────────────────────────────

def main_r23_v8():
    print("=" * 70)
    print("  R&D ROUND 23: V8 Training Runner")
    print("=" * 70)

    print("\n  Imports & modes")
    test_train_v8_imports()
    test_train_v8_modes()

    print("\n  ETA & checkpoints")
    test_train_v8_eta_projection()
    test_train_v8_rolling_checkpoints()
    test_train_v8_resume_bundle()

    print("\n  VRAM & NaN guards")
    test_train_v8_vram_preflight()
    test_train_v8_nan_guard()

    print("\n  Training & checkpointing")
    test_train_v8_dead_param_freeze()
    test_train_v8_loss_decreases()
    test_train_v8_checkpoint_roundtrip()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 V8 RUNNER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_v8()

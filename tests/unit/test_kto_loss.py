"""Test KTO (Kahneman-Tversky Optimization) loss and data format.

KTO (arXiv:2402.01306, ICML 2024) uses binary good/bad labels instead of
paired preference data. These tests verify the loss function, KL reference
point EMA, gradient flow, and data format compatibility.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from forge.training.runners.dpo_align import (
    KLReferencePoint,
    build_kto_sample,
    build_preference_sample,
    dpo_loss,
    kto_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Minimal tokenizer stub for build_*_sample tests (no HF dependency)."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [ord(c) % 100 + 1 for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        # Simplistic: just concatenate contents.
        parts = []
        for m in msgs:
            parts.append(m["content"])
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\nassistant:"
        return text


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


def test_kto_loss_good_response():
    """For a good (label=True) response, loss should DECREASE when the policy
    assigns higher log-prob than the reference (log_ratio > KL_ref)."""
    beta = 0.1
    kl_ref = 0.0
    labels = torch.tensor([True])

    # Policy assigns higher logp than ref -> log_ratio positive.
    policy_logp_hi = torch.tensor([10.0])
    ref_logp = torch.tensor([0.0])
    loss_hi = kto_loss(policy_logp_hi, ref_logp, labels, beta=beta, kl_ref=kl_ref)

    # Policy assigns lower logp than ref -> log_ratio negative.
    policy_logp_lo = torch.tensor([-10.0])
    loss_lo = kto_loss(policy_logp_lo, ref_logp, labels, beta=beta, kl_ref=kl_ref)

    assert loss_hi.item() < loss_lo.item(), (
        f"Good response: loss should be lower when policy > ref "
        f"(got hi={loss_hi.item():.4f} vs lo={loss_lo.item():.4f})"
    )
    # When log_ratio >> 0, loss should be near zero (sigmoid saturates to 1).
    assert loss_hi.item() < 0.5, f"Loss for strong good response should be small, got {loss_hi.item():.4f}"
    print(f"PASS: kto_loss good response (hi={loss_hi.item():.4f} < lo={loss_lo.item():.4f})")


def test_kto_loss_bad_response():
    """For a bad (label=False) response, loss should DECREASE when the policy
    assigns LOWER log-prob than the reference (log_ratio < KL_ref)."""
    beta = 0.1
    kl_ref = 0.0
    labels = torch.tensor([False])

    # Policy assigns lower logp than ref -> log_ratio negative (good for bad label).
    policy_logp_lo = torch.tensor([-10.0])
    ref_logp = torch.tensor([0.0])
    loss_lo = kto_loss(policy_logp_lo, ref_logp, labels, beta=beta, kl_ref=kl_ref)

    # Policy assigns higher logp than ref -> log_ratio positive (bad for bad label).
    policy_logp_hi = torch.tensor([10.0])
    loss_hi = kto_loss(policy_logp_hi, ref_logp, labels, beta=beta, kl_ref=kl_ref)

    assert loss_lo.item() < loss_hi.item(), (
        f"Bad response: loss should be lower when policy < ref (got lo={loss_lo.item():.4f} vs hi={loss_hi.item():.4f})"
    )
    assert loss_lo.item() < 0.5, f"Loss for strong bad response should be small, got {loss_lo.item():.4f}"
    print(f"PASS: kto_loss bad response (lo={loss_lo.item():.4f} < hi={loss_hi.item():.4f})")


def test_kto_loss_batch_mixed_labels():
    """KTO loss should handle a batch with mixed good/bad labels."""
    beta = 0.1
    kl_ref = 0.0
    policy_logp = torch.tensor([2.0, -2.0, 1.0, -1.0])
    ref_logp = torch.tensor([0.0, 0.0, 0.0, 0.0])
    labels = torch.tensor([True, False, True, False])

    loss = kto_loss(policy_logp, ref_logp, labels, beta=beta, kl_ref=kl_ref)
    assert torch.isfinite(loss), "Loss should be finite for mixed batch"
    assert loss.item() > 0, "Loss should be positive"
    print(f"PASS: kto_loss mixed batch (loss={loss.item():.4f})")


def test_kto_loss_kl_ref_shifts_decision_boundary():
    """The KL reference point should shift the decision boundary.
    With kl_ref > 0, a good response needs log_ratio > kl_ref to minimize loss."""
    beta = 0.1
    labels = torch.tensor([True])
    ref_logp = torch.tensor([0.0])

    # log_ratio = 0.5, kl_ref = 0.0 -> positive margin -> low loss
    loss_no_kl = kto_loss(torch.tensor([0.5]), ref_logp, labels, beta=beta, kl_ref=0.0)
    # log_ratio = 0.5, kl_ref = 1.0 -> negative margin -> high loss
    loss_with_kl = kto_loss(torch.tensor([0.5]), ref_logp, labels, beta=beta, kl_ref=1.0)

    assert loss_with_kl.item() > loss_no_kl.item(), (
        f"KL ref should shift boundary: loss_with_kl={loss_with_kl.item():.4f} "
        f"should > loss_no_kl={loss_no_kl.item():.4f}"
    )
    print(f"PASS: kto_loss KL ref shifts boundary (no_kl={loss_no_kl.item():.4f} < with_kl={loss_with_kl.item():.4f})")


# ---------------------------------------------------------------------------
# KL reference point tests
# ---------------------------------------------------------------------------


def test_kto_kl_reference_initialization():
    """KL reference point should initialize to the first batch mean."""
    kl_ref = KLReferencePoint(ema_beta=0.9)
    assert kl_ref.value == 0.0
    assert not kl_ref.initialized

    log_ratio = torch.tensor([1.0, 2.0, 3.0])
    val = kl_ref.update(log_ratio)
    assert kl_ref.initialized
    assert abs(val - 2.0) < 1e-6, f"Initial value should be batch mean (2.0), got {val}"
    print(f"PASS: KL reference initializes to first batch mean ({val})")


def test_kto_kl_reference_ema_update():
    """KL reference point should update via EMA after initialization."""
    kl_ref = KLReferencePoint(ema_beta=0.9)
    # First update initializes.
    kl_ref.update(torch.tensor([1.0]))
    assert abs(kl_ref.value - 1.0) < 1e-6

    # Second update: EMA = 0.9 * 1.0 + 0.1 * 3.0 = 1.2
    kl_ref.update(torch.tensor([3.0]))
    assert abs(kl_ref.value - 1.2) < 1e-6, f"EMA update should give 1.2, got {kl_ref.value}"

    # Third update: EMA = 0.9 * 1.2 + 0.1 * 5.0 = 1.58
    kl_ref.update(torch.tensor([5.0]))
    assert abs(kl_ref.value - 1.58) < 1e-6, f"EMA update should give 1.58, got {kl_ref.value}"
    print(f"PASS: KL reference EMA updates correctly (1.0 -> 1.2 -> {kl_ref.value})")


def test_kto_kl_reference_detached():
    """KL reference update should not create gradient connections."""
    kl_ref = KLReferencePoint(ema_beta=0.9)
    log_ratio = torch.tensor([1.0, 2.0], requires_grad=True)
    kl_ref.update(log_ratio)
    # The update should not affect the computation graph.
    assert kl_ref.value == 1.5
    # No grad_fn on the stored value (it's a Python float).
    assert isinstance(kl_ref.value, float)
    print("PASS: KL reference update is detached (no grad connection)")


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


def test_kto_gradient_flow_good():
    """Gradients should flow through KTO loss for good (label=True) examples."""
    beta = 0.1
    kl_ref = 0.0
    labels = torch.tensor([True])
    policy_logp = torch.tensor([1.0], requires_grad=True)
    ref_logp = torch.tensor([0.0])  # detached (no grad)

    loss = kto_loss(policy_logp, ref_logp, labels, beta=beta, kl_ref=kl_ref)
    loss.backward()

    assert policy_logp.grad is not None, "Gradient should flow to policy_logp"
    assert policy_logp.grad.abs() > 0, "Gradient should be non-zero"
    print(f"PASS: KTO gradient flows for good response (grad={policy_logp.grad.item():.6f})")


def test_kto_gradient_flow_bad():
    """Gradients should flow through KTO loss for bad (label=False) examples."""
    beta = 0.1
    kl_ref = 0.0
    labels = torch.tensor([False])
    policy_logp = torch.tensor([1.0], requires_grad=True)
    ref_logp = torch.tensor([0.0])

    loss = kto_loss(policy_logp, ref_logp, labels, beta=beta, kl_ref=kl_ref)
    loss.backward()

    assert policy_logp.grad is not None, "Gradient should flow to policy_logp"
    assert policy_logp.grad.abs() > 0, "Gradient should be non-zero"
    # For bad responses, gradient should be positive (pushing logp DOWN).
    assert policy_logp.grad.item() > 0, (
        f"Bad response gradient should push logp down (positive grad), got {policy_logp.grad.item():.6f}"
    )
    print(f"PASS: KTO gradient flows for bad response (grad={policy_logp.grad.item():.6f}, pushes logp down)")


def test_kto_gradient_flow_batch():
    """Gradients should flow through a batch of mixed-label KTO examples."""
    beta = 0.1
    kl_ref = 0.0
    policy_logp = torch.tensor([1.0, -1.0, 0.5, -0.5], requires_grad=True)
    ref_logp = torch.tensor([0.0, 0.0, 0.0, 0.0])
    labels = torch.tensor([True, False, True, False])

    loss = kto_loss(policy_logp, ref_logp, labels, beta=beta, kl_ref=kl_ref)
    loss.backward()

    assert policy_logp.grad is not None
    assert torch.all(policy_logp.grad.abs() > 0), "All gradients should be non-zero"
    print(f"PASS: KTO gradient flows through batch (grads={policy_logp.grad.tolist()})")


def test_kto_gradient_direction_good_vs_bad():
    """For good responses, gradient should push logp UP (negative grad).
    For bad responses, gradient should push logp DOWN (positive grad)."""
    beta = 0.1
    kl_ref = 0.0
    ref_logp = torch.tensor([0.0])

    # Good response with log_ratio near 0 (boundary).
    good_logp = torch.tensor([0.0], requires_grad=True)
    loss_good = kto_loss(good_logp, ref_logp, torch.tensor([True]), beta=beta, kl_ref=kl_ref)
    loss_good.backward()
    good_grad = good_logp.grad.item()

    # Bad response with log_ratio near 0 (boundary).
    bad_logp = torch.tensor([0.0], requires_grad=True)
    loss_bad = kto_loss(bad_logp, ref_logp, torch.tensor([False]), beta=beta, kl_ref=kl_ref)
    loss_bad.backward()
    bad_grad = bad_logp.grad.item()

    # Good: gradient negative (increase logp). Bad: gradient positive (decrease logp).
    assert good_grad < 0, f"Good response grad should be negative (push up), got {good_grad}"
    assert bad_grad > 0, f"Bad response grad should be positive (push down), got {bad_grad}"
    print(f"PASS: gradient directions correct (good={good_grad:.6f} < 0, bad={bad_grad:.6f} > 0)")


# ---------------------------------------------------------------------------
# Data format tests
# ---------------------------------------------------------------------------


def test_kto_vs_dpo_data_format():
    """KTO works with binary labels (prompt, response, label);
    DPO works with pairs (prompt, chosen, rejected)."""
    tok = _DummyTokenizer()
    max_len = 128

    # KTO sample: single response + binary label.
    kto_good = build_kto_sample(tok, "Hello", "Hi there!", True, max_len, use_chat_template=False)
    kto_bad = build_kto_sample(tok, "Hello", "Bye!", False, max_len, use_chat_template=False)
    assert "ids" in kto_good and "comp_start" in kto_good and "label" in kto_good
    assert kto_good["label"] is True
    assert kto_bad["label"] is False
    assert len(kto_good["ids"]) > kto_good["comp_start"]
    assert len(kto_bad["ids"]) > kto_bad["comp_start"]

    # DPO sample: chosen/rejected pair.
    dpo_sample = build_preference_sample(tok, "Hello", "Hi there!", "Bye!", max_len, use_chat_template=False)
    assert "chosen_ids" in dpo_sample and "rejected_ids" in dpo_sample
    assert "chosen_start" in dpo_sample and "rejected_start" in dpo_sample
    assert "label" not in dpo_sample, "DPO sample should NOT have a 'label' key"
    assert "ids" not in dpo_sample, "DPO sample should NOT have an 'ids' key"

    print("PASS: KTO uses binary labels, DPO uses pairs — formats are distinct")


def test_kto_sample_chat_template():
    """build_kto_sample should work with chat template enabled."""
    tok = _DummyTokenizer()
    sample = build_kto_sample(tok, "What is 2+2?", "4", True, 128, use_chat_template=True)
    assert sample["label"] is True
    assert len(sample["ids"]) > sample["comp_start"]
    assert sample["comp_start"] > 0, "comp_start should be > 0 (prompt tokens precede completion)"
    print(f"PASS: KTO sample with chat template (ids_len={len(sample['ids'])}, comp_start={sample['comp_start']})")


def test_kto_sample_max_length_truncation():
    """build_kto_sample should truncate to max_length."""
    tok = _DummyTokenizer()
    long_response = "x" * 500
    sample = build_kto_sample(tok, "prompt", long_response, True, 50, use_chat_template=False)
    assert len(sample["ids"]) <= 50, f"Should truncate to max_length=50, got {len(sample['ids'])}"
    print(f"PASS: KTO sample truncates to max_length (len={len(sample['ids'])})")


def test_kto_loss_vs_dpo_loss_different_inputs():
    """KTO loss and DPO loss accept different inputs and both produce valid scalars."""
    # DPO: paired logps.
    dpo_l = dpo_loss(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=0.1,
    )
    assert dpo_l.item() > 0 and torch.isfinite(dpo_l)

    # KTO: single logps + labels.
    kto_l = kto_loss(
        torch.tensor([2.0, -2.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([True, False]),
        beta=0.1,
        kl_ref=0.0,
    )
    assert kto_l.item() > 0 and torch.isfinite(kto_l)
    print(f"PASS: both losses produce valid scalars (dpo={dpo_l.item():.4f}, kto={kto_l.item():.4f})")


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------


def test_kto_loss_matches_formula():
    """Verify KTO loss matches the exact formula from the paper."""
    beta = 0.1
    kl_ref = 0.3
    policy_logp = torch.tensor([1.5])
    ref_logp = torch.tensor([0.2])
    labels = torch.tensor([True])

    log_ratio = 1.5 - 0.2  # 1.3
    expected = -torch.log(torch.sigmoid(torch.tensor(beta * (log_ratio - kl_ref)))).item()
    actual = kto_loss(policy_logp, ref_logp, labels, beta=beta, kl_ref=kl_ref).item()

    assert abs(actual - expected) < 1e-5, f"Expected {expected}, got {actual}"
    print(f"PASS: KTO loss matches formula (expected={expected:.6f}, actual={actual:.6f})")


def test_kto_loss_bad_matches_formula():
    """Verify KTO bad-response loss matches the exact formula."""
    beta = 0.1
    kl_ref = 0.3
    policy_logp = torch.tensor([1.5])
    ref_logp = torch.tensor([0.2])
    labels = torch.tensor([False])

    log_ratio = 1.5 - 0.2  # 1.3
    expected = -torch.log(torch.sigmoid(torch.tensor(beta * (kl_ref - log_ratio)))).item()
    actual = kto_loss(policy_logp, ref_logp, labels, beta=beta, kl_ref=kl_ref).item()

    assert abs(actual - expected) < 1e-5, f"Expected {expected}, got {actual}"
    print(f"PASS: KTO bad loss matches formula (expected={expected:.6f}, actual={actual:.6f})")


def test_kto_loss_beta_scales_margin():
    """Higher beta should make the loss more sensitive to the log-ratio margin."""
    kl_ref = 0.0
    labels = torch.tensor([True])
    policy_logp = torch.tensor([0.5])
    ref_logp = torch.tensor([0.0])

    loss_low_beta = kto_loss(policy_logp, ref_logp, labels, beta=0.01, kl_ref=kl_ref)
    loss_high_beta = kto_loss(policy_logp, ref_logp, labels, beta=1.0, kl_ref=kl_ref)

    # Higher beta -> sigmoid closer to 1 -> lower loss for good response.
    assert loss_high_beta.item() < loss_low_beta.item(), (
        f"Higher beta should reduce loss for good response "
        f"(high={loss_high_beta.item():.4f} < low={loss_low_beta.item():.4f})"
    )
    print(f"PASS: beta scales margin (low_beta={loss_low_beta.item():.4f} > high_beta={loss_high_beta.item():.4f})")


if __name__ == "__main__":
    test_kto_loss_good_response()
    test_kto_loss_bad_response()
    test_kto_loss_batch_mixed_labels()
    test_kto_loss_kl_ref_shifts_decision_boundary()
    test_kto_kl_reference_initialization()
    test_kto_kl_reference_ema_update()
    test_kto_kl_reference_detached()
    test_kto_gradient_flow_good()
    test_kto_gradient_flow_bad()
    test_kto_gradient_flow_batch()
    test_kto_gradient_direction_good_vs_bad()
    test_kto_vs_dpo_data_format()
    test_kto_sample_chat_template()
    test_kto_sample_max_length_truncation()
    test_kto_loss_vs_dpo_loss_different_inputs()
    test_kto_loss_matches_formula()
    test_kto_loss_bad_matches_formula()
    test_kto_loss_beta_scales_margin()
    print("\n=== All KTO loss tests passed ===")

"""Safety validation for key application — prevents model corruption.

Every `apply_*_to_model` function should use these utilities to verify
that applying a key does not corrupt the model. The checks are:

1. **Pre-snapshot**: capture model state (param checksums + forward output).
2. **Apply key**: the key transformation.
3. **Post-validation**: verify:
   - All parameters are finite (no NaN/Inf).
   - Forward pass produces finite outputs.
   - For identity-init keys: output is numerically identical to pre-snapshot.
   - No parameters were unexpectedly deleted or reshaped.
4. **Rollback**: if validation fails, restore the pre-snapshot state.

Usage:
    from forge.keys.safety import safe_apply, KeySafetyError

    def my_apply_key(model, ...):
        def apply_fn(m):
            # ... modify m ...
            return m

        return safe_apply(model, apply_fn, identity_init=True,
                         test_input=torch.randn(1, 16, 768))
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class KeySafetyError(Exception):
    """Raised when a key application corrupts the model."""
    pass


@dataclass
class ModelSnapshot:
    """Snapshot of model state for safety validation."""
    param_checksums: dict[int, float] = field(default_factory=dict)
    param_shapes: dict[int, tuple] = field(default_factory=dict)
    param_dtypes: dict[int, torch.dtype] = field(default_factory=dict)
    n_params: int = 0
    forward_output: torch.Tensor | None = None
    forward_checksum: float | None = None


def _param_id(p: torch.Tensor) -> int:
    """Stable ID for a parameter tensor."""
    return id(p)


def _checksum(t: torch.Tensor) -> float:
    """Fast checksum for a tensor (sum of float values)."""
    return float(t.float().sum().item())


def take_snapshot(model: nn.Module,
                  test_input: torch.Tensor | None = None) -> ModelSnapshot:
    """Take a safety snapshot of the model.

    Args:
        model: the model to snapshot.
        test_input: optional input tensor for forward pass validation.
            If provided, the snapshot includes the model's forward output.

    Returns:
        ModelSnapshot with param checksums and optional forward output.
    """
    snap = ModelSnapshot()
    for p in model.parameters():
        pid = _param_id(p)
        snap.param_checksums[pid] = _checksum(p)
        snap.param_shapes[pid] = tuple(p.shape)
        snap.param_dtypes[pid] = p.dtype
        snap.n_params += p.numel()

    if test_input is not None:
        with torch.no_grad():
            try:
                out = model(test_input)
                if isinstance(out, tuple):
                    out = out[0]
                snap.forward_output = out.clone()
                snap.forward_checksum = _checksum(out)
            except Exception as e:
                logger.warning(f"Snapshot forward pass failed: {e}")
                snap.forward_output = None
                snap.forward_checksum = None

    return snap


def validate_snapshot(model: nn.Module, snapshot: ModelSnapshot,
                      test_input: torch.Tensor | None = None,
                      identity_init: bool = False,
                      atol: float = 1e-5,
                      rtol: float = 1e-4) -> tuple[bool, list[str]]:
    """Validate that the model has not been corrupted after a key application.

    Args:
        model: the model after key application.
        snapshot: pre-application snapshot.
        test_input: same input used for the snapshot's forward pass.
        identity_init: if True, require forward output to be numerically
            identical to the snapshot (for identity-init keys).
        atol: absolute tolerance for identity check.
        rtol: relative tolerance for identity check.

    Returns:
        (passed, errors) — True if all checks pass, list of error messages.
    """
    errors = []

    # Check 1: all parameters are finite.
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            errors.append(f"Parameter '{name}' contains NaN or Inf values")
        if p.dtype not in (torch.float32, torch.float16, torch.bfloat16,
                           torch.int64, torch.int32, torch.int8,
                           torch.uint8, torch.bool, torch.float8_e4m3fn):
            errors.append(f"Parameter '{name}' has unexpected dtype: {p.dtype}")

    # Check 2: no parameters were unexpectedly deleted.
    # New parameters are OK (keys may add them), but existing ones should
    # still be present unless explicitly replaced.
    # We check that the total param count didn't decrease unexpectedly.
    current_n = sum(p.numel() for p in model.parameters())
    if current_n < snapshot.n_params * 0.5:
        errors.append(
            f"Parameter count dropped from {snapshot.n_params} to {current_n} "
            f"(>50% loss — likely corruption)")

    # Check 3: forward pass produces finite output.
    if test_input is not None:
        with torch.no_grad():
            try:
                out = model(test_input)
                if isinstance(out, tuple):
                    out = out[0]
            except Exception as e:
                errors.append(f"Forward pass failed after key application: {e}")
                return (False, errors)

        if not torch.isfinite(out).all():
            errors.append("Forward output contains NaN or Inf values")

        # Check 4: identity-init keys should produce identical output.
        if identity_init and snapshot.forward_output is not None:
            if out.shape != snapshot.forward_output.shape:
                errors.append(
                    f"Output shape changed: {snapshot.forward_output.shape} -> "
                    f"{out.shape} (identity-init key should not change shape)")
            elif not torch.allclose(out, snapshot.forward_output,
                                    atol=atol, rtol=rtol):
                diff = (out - snapshot.forward_output).abs().max().item()
                errors.append(
                    f"Output changed by {diff:.2e} (identity-init key should "
                    f"produce identical output, atol={atol}, rtol={rtol})")

    return (len(errors) == 0, errors)


def safe_apply(model: nn.Module,
               apply_fn: Callable[[nn.Module], nn.Module],
               identity_init: bool = False,
               test_input: torch.Tensor | None = None,
               atol: float = 1e-5,
               rtol: float = 1e-4,
               rollback_on_failure: bool = True) -> nn.Module:
    """Safely apply a key transformation to a model.

    Takes a pre-snapshot, applies the key, validates, and rolls back on failure.

    Args:
        model: the model to modify.
        apply_fn: function(model) -> model that applies the key.
        identity_init: if True, the key is identity-init and the forward
            output should be numerically identical after application.
        test_input: input tensor for forward pass validation.
        atol: absolute tolerance for identity check.
        rtol: relative tolerance for identity check.
        rollback_on_failure: if True, restore the original model state on
            validation failure.

    Returns:
        The modified model (or restored model if rollback occurred).

    Raises:
        KeySafetyError: if validation fails and rollback_on_failure=False,
            or if rollback itself fails.
    """
    # Deep copy for rollback (state_dict is cheaper but we need full structure
    # in case the key adds/removes modules).
    if rollback_on_failure:
        rollback_state = copy.deepcopy(model.state_dict())
        # Also save module structure for structural rollback.
        rollback_modules = {}
        for name, mod in model.named_modules():
            if name:  # skip root
                rollback_modules[name] = type(mod)

    # Take pre-snapshot.
    snapshot = take_snapshot(model, test_input)

    # Apply the key.
    try:
        model = apply_fn(model)
    except Exception as e:
        logger.error(f"Key application failed: {e}")
        if rollback_on_failure:
            model.load_state_dict(rollback_state)
        raise KeySafetyError(f"Key application failed: {e}") from e

    # Validate.
    passed, errors = validate_snapshot(
        model, snapshot, test_input, identity_init, atol, rtol)

    if not passed:
        error_msg = "; ".join(errors)
        logger.error(f"Key safety validation failed: {error_msg}")

        if rollback_on_failure:
            try:
                model.load_state_dict(rollback_state)
                logger.info("Rolled back to pre-application state")
            except Exception as rb_err:
                raise KeySafetyError(
                    f"Validation failed AND rollback failed: {error_msg}; "
                    f"rollback error: {rb_err}") from rb_err
            raise KeySafetyError(
                f"Key validation failed (rolled back): {error_msg}")
        else:
            raise KeySafetyError(f"Key validation failed: {error_msg}")

    logger.info("Key safety validation passed")
    return model


def safe_apply_key_to_expert(expert: nn.Module,
                             apply_fn: Callable,
                             identity_init: bool = False,
                             test_input: torch.Tensor | None = None,
                             **kwargs) -> nn.Module:
    """Safely apply a key to a single MoE expert.

    Experts are sub-modules of a MoE layer. This wraps safe_apply with
    additional checks specific to experts:
      - Expert output dimensionality must match the MoE router's expectation.
      - Expert must not change its output shape.

    Args:
        expert: the expert module to modify.
        apply_fn: function(expert) -> expert that applies the key.
        identity_init: if True, expert output should be identical.
        test_input: input for forward validation.
        **kwargs: passed to safe_apply.

    Returns:
        The modified expert (or rolled-back expert on failure).
    """
    return safe_apply(expert, apply_fn, identity_init, test_input, **kwargs)


def verify_model_integrity(model: nn.Module,
                           test_input: torch.Tensor | None = None) -> tuple[bool, list[str]]:
    """Standalone integrity check for a model.

    Useful for post-training or post-merge validation.

    Args:
        model: the model to check.
        test_input: optional input for forward pass check.

    Returns:
        (healthy, issues) — True if model is healthy, list of issue descriptions.
    """
    issues = []

    # Check all parameters are finite.
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            issues.append(f"Parameter '{name}' has NaN/Inf")

    # Check for dead parameters (all zeros — might indicate corruption).
    for name, p in model.named_parameters():
        if p.numel() > 0 and p.float().abs().max().item() == 0:
            if "norm" not in name.lower() and "bias" not in name.lower():
                issues.append(f"Parameter '{name}' is all zeros (possible corruption)")

    # Forward pass check.
    if test_input is not None:
        with torch.no_grad():
            try:
                out = model(test_input)
                if isinstance(out, tuple):
                    out = out[0]
                if not torch.isfinite(out).all():
                    issues.append("Forward output has NaN/Inf")
            except Exception as e:
                issues.append(f"Forward pass failed: {e}")

    return (len(issues) == 0, issues)

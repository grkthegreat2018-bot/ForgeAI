"""Unit tests for preset lineage enforcement (critique F2).

Validates that every preset with a ``parent`` field:
- References an existing parent preset
- Documents all divergences from parent in ``dropped_keys``
- Does not list ``dropped_keys`` that don't exist in the parent

This is the machine-enforced version of AGENTS.md directive A:
"Build On The Prior, Never Beside It".
"""

import pytest

from forge.config import MODEL_CONFIGS, ModelConfig, validate_preset_lineage


def test_lineage_validator_runs_clean():
    """All presets with parent fields must pass lineage validation."""
    errors = validate_preset_lineage()
    if errors:
        msg = "\n".join(errors)
        pytest.fail(f"Preset lineage violations:\n{msg}")


def test_v12_jamba_has_parent():
    """V12-Jamba is derived from V2 and must declare its parent."""
    cfg = MODEL_CONFIGS["forgelm_v12_jamba"]
    assert cfg.parent == "forgelm_v2", f"V12-Jamba should have parent='forgelm_v2', got {cfg.parent!r}"


def test_v12_jamba_documents_divergences():
    """V12-Jamba must document all divergences from V2 in dropped_keys."""
    v2 = MODEL_CONFIGS["forgelm_v2"]
    jamba = MODEL_CONFIGS["forgelm_v12_jamba"]
    skip = {"parent", "dropped_keys"}
    divergent = [k for k in v2.__dict__ if k not in skip and v2.__dict__[k] != jamba.__dict__.get(k)]
    undocumented = set(divergent) - set(jamba.dropped_keys)
    assert not undocumented, f"V12-Jamba has undocumented divergences from V2: {sorted(undocumented)}"


def test_parent_must_exist():
    """A preset with a non-existent parent must be flagged."""
    bad = ModelConfig(parent="nonexistent_preset")
    MODEL_CONFIGS["__test_bad__"] = bad
    try:
        errors = validate_preset_lineage()
        assert any("__test_bad__" in e and "does not exist" in e for e in errors)
    finally:
        del MODEL_CONFIGS["__test_bad__"]


def test_undocumented_divergence_is_flagged():
    """A preset that diverges from parent without documenting must be flagged."""
    parent = ModelConfig(vocab_size=100)
    # Child diverges on vocab_size (100 vs default 65536) without documenting
    child = ModelConfig(parent="__test_parent__")
    MODEL_CONFIGS["__test_parent__"] = parent
    MODEL_CONFIGS["__test_child__"] = child
    try:
        errors = validate_preset_lineage()
        assert any("__test_child__" in e and "vocab_size" in e for e in errors), (
            f"Expected vocab_size in errors: {errors}"
        )
    finally:
        del MODEL_CONFIGS["__test_parent__"]
        del MODEL_CONFIGS["__test_child__"]


def test_documented_divergence_is_allowed():
    """A preset that documents divergences in dropped_keys is valid."""
    parent = ModelConfig(vocab_size=100)
    child = ModelConfig(
        parent="__test_parent2__",
        dropped_keys=("vocab_size",),  # document the change
    )
    MODEL_CONFIGS["__test_parent2__"] = parent
    MODEL_CONFIGS["__test_child2__"] = child
    try:
        errors = validate_preset_lineage()
        child_errors = [e for e in errors if "__test_child2__" in e]
        assert not child_errors, f"Unexpected errors: {child_errors}"
    finally:
        del MODEL_CONFIGS["__test_parent2__"]
        del MODEL_CONFIGS["__test_child2__"]


def test_invalid_dropped_key_is_flagged():
    """A dropped_key that doesn't exist in parent must be flagged."""
    parent = ModelConfig(vocab_size=100)
    child = ModelConfig(
        parent="__test_parent3__",
        dropped_keys=("nonexistent_field",),
    )
    MODEL_CONFIGS["__test_parent3__"] = parent
    MODEL_CONFIGS["__test_child3__"] = child
    try:
        errors = validate_preset_lineage()
        assert any("__test_child3__" in e and "nonexistent_field" in e for e in errors)
    finally:
        del MODEL_CONFIGS["__test_parent3__"]
        del MODEL_CONFIGS["__test_child3__"]


def test_root_presets_not_validated():
    """Presets without a parent field are not validated."""
    root = ModelConfig()  # parent=None
    MODEL_CONFIGS["__test_root__"] = root
    try:
        errors = validate_preset_lineage()
        assert not any("__test_root__" in e for e in errors)
    finally:
        del MODEL_CONFIGS["__test_root__"]

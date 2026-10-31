"""Unit tests for forge_gui.api.activation_catalog.

Verifies that the catalog mirrors ActivationConfig exactly (no drift),
that presets resolve, and that normalize/validate helpers work.
"""
import dataclasses

from forge_gui.api import activation_catalog as cat

# ── ActivationConfig mirror ────────────────────────────────────────────

def _activation_config_fields():
    """Get ActivationConfig field names (import lazily — pulls in torch)."""
    from forge.engine.activation import ActivationConfig
    return {f.name for f in dataclasses.fields(ActivationConfig)}


ARCH_OVERRIDE_CATEGORY = "Architecture Overrides"


def _model_config_fields():
    """Get ModelConfig field names (import lazily — pulls in torch)."""
    from forge.config import ModelConfig
    return {f.name for f in dataclasses.fields(ModelConfig)}


def test_catalog_covers_all_activation_config_fields():
    """Every ActivationConfig field must appear in the catalog exactly once.

    The "Architecture Overrides" category is exempt: those fields are
    construction-time ModelConfig overrides (passed via
    ``from_checkpoint(config_overrides=...)``), not activate() kwargs.
    They are checked against ModelConfig by
    ``test_arch_overrides_are_model_config_fields`` instead.
    """
    cfg_fields = _activation_config_fields()
    catalog_names = {f.name for f in cat.FIELDS
                     if f.category != ARCH_OVERRIDE_CATEGORY}
    missing = cfg_fields - catalog_names
    extra = catalog_names - cfg_fields
    assert not missing, f"ActivationConfig fields missing from catalog: {missing}"
    assert not extra, f"Catalog has fields not in ActivationConfig: {extra}"


def test_arch_overrides_are_model_config_fields():
    """Architecture Override catalog entries must name real ModelConfig
    fields — they are applied via config_overrides at model construction."""
    model_fields = _model_config_fields()
    arch_names = {f.name for f in cat.FIELDS
                  if f.category == ARCH_OVERRIDE_CATEGORY}
    assert arch_names, "expected at least one Architecture Overrides field"
    bogus = arch_names - model_fields
    assert not bogus, f"Arch overrides not in ModelConfig: {bogus}"


def test_no_duplicate_field_names():
    names = [f.name for f in cat.FIELDS]
    assert len(names) == len(set(names)), "duplicate field names in catalog"


def test_all_fields_have_valid_category():
    for f in cat.FIELDS:
        assert f.category in cat.CATEGORIES, \
            f"field {f.name} has unknown category {f.category!r}"


def test_combo_fields_have_options():
    for f in cat.FIELDS:
        if f.kind == "combo":
            assert len(f.options) >= 2, \
                f"combo field {f.name} has fewer than 2 options"


def test_fields_by_category_groups_all():
    grouped = cat.fields_by_category()
    total = sum(len(fields) for _, fields in grouped)
    assert total == len(cat.FIELDS)
    # every category in CATEGORIES should appear (even if empty)
    cats_in_grouped = {c for c, _ in grouped}
    for c in cat.CATEGORIES:
        assert c in cats_in_grouped, f"category {c!r} missing from grouping"


# ── defaults ───────────────────────────────────────────────────────────

def test_default_config_has_all_fields():
    defaults = cat.default_config()
    assert set(defaults.keys()) == {f.name for f in cat.FIELDS}


def test_default_config_matches_field_defaults():
    defaults = cat.default_config()
    for f in cat.FIELDS:
        assert defaults[f.name] == f.default, \
            f"default mismatch for {f.name}: {defaults[f.name]} vs {f.default}"


# ── presets ────────────────────────────────────────────────────────────

def test_preset_names_unique():
    names = [p.name for p in cat.PRESETS]
    assert len(names) == len(set(names)), "duplicate preset names"


def test_preset_config_returns_dict():
    cfg = cat.preset_config("optimal")
    assert cfg is not None
    assert isinstance(cfg, dict)
    # preset config should only contain known field names
    valid_names = {f.name for f in cat.FIELDS}
    for k in cfg:
        assert k in valid_names, f"preset has unknown field {k!r}"


def test_preset_config_unknown_returns_none():
    assert cat.preset_config("nonexistent_preset") is None


# ── normalize_value ────────────────────────────────────────────────────

def test_normalize_combo_none():
    f = cat.spec("kv_cache")
    assert f is not None
    assert cat.normalize_value(f, "none") is None
    assert cat.normalize_value(f, "") is None
    assert cat.normalize_value(f, None) is None


def test_normalize_combo_value():
    f = cat.spec("kv_cache")
    assert cat.normalize_value(f, "rotorquant") == "rotorquant"


def test_normalize_bool():
    f = next(f for f in cat.FIELDS if f.kind == "bool")
    assert cat.normalize_value(f, True) is True
    assert cat.normalize_value(f, False) is False


# ── validate ───────────────────────────────────────────────────────────

def test_validate_default_config_clean():
    errors = cat.validate(cat.default_config())
    assert errors == [], f"default config has validation errors: {errors}"


def test_validate_bad_combo_value():
    cfg = cat.default_config()
    cfg["kv_cache"] = "nonexistent_strategy"
    errors = cat.validate(cfg)
    # error message uses the field label ("KV cache"), not the field name
    assert any("KV cache" in e for e in errors)


# ── active_diff ────────────────────────────────────────────────────────

def test_active_diff_no_current():
    diff = cat.active_diff(None, cat.default_config())
    assert diff == []


def test_active_diff_identical():
    cfg = cat.default_config()
    diff = cat.active_diff(cfg, cfg)
    assert diff == []


def test_active_diff_changed():
    current = cat.default_config()
    desired = dict(current)
    # kv_cache default is "paged" — change to something else
    desired["kv_cache"] = "rotorquant"
    diff = cat.active_diff(current, desired)
    assert len(diff) == 1
    assert "rotorquant" in diff[0]


# ── spec lookup ────────────────────────────────────────────────────────

def test_spec_known():
    f = cat.spec("kv_cache")
    assert f is not None
    assert f.name == "kv_cache"


def test_spec_unknown():
    assert cat.spec("nonexistent_field") is None

"""Unit tests for forge_gui.api.lora_store category system + LoraHarness.

Tests category parsing, MODE_CATEGORY_MAP, and LoraHarness logic
(mocked runtime — no actual engine loading).
"""

from unittest.mock import MagicMock

import pytest

from forge_gui.api.lora_store import (
    CATEGORY_INFO,
    LORA_CATEGORIES,
    MODE_CATEGORY_MAP,
    LoRAEntry,
    LoraHarness,
    LoraManager,
    parse_category,
)

# ── Category parsing ───────────────────────────────────────────────────


class TestParseCategory:
    def test_coding_category(self):
        assert parse_category("ForgeLM_V11_coding_R32_lora.safetensors") == "coding"

    def test_agentic_category(self):
        assert parse_category("ForgeLM_V11_agentic_R64_lora.safetensors") == "agentic"

    def test_self_play_category(self):
        assert parse_category("ForgeLM_V11_self_play_R16_lora.safetensors") == "self_play"

    def test_chat_assist_category(self):
        assert parse_category("ForgeLM_V11_chat_assist_R32_lora.safetensors") == "chat_assist"

    def test_uncategorized(self):
        assert parse_category("random_adapter_lora.safetensors") == "uncategorized"

    def test_case_insensitive(self):
        assert parse_category("ForgeLM_V11_CODING_R32_lora.safetensors") == "coding"

    def test_category_in_name_anywhere(self):
        assert parse_category("my_coding_adapter.safetensors") == "coding"


# ── Category constants ─────────────────────────────────────────────────


class TestCategoryConstants:
    def test_all_categories_have_info(self):
        for cat in LORA_CATEGORIES:
            assert cat in CATEGORY_INFO
            label, desc, icon = CATEGORY_INFO[cat]
            assert label
            assert desc
            assert icon

    def test_all_modes_have_categories(self):
        for _mode, cats in MODE_CATEGORY_MAP.items():
            assert isinstance(cats, list)
            assert len(cats) > 0
            for cat in cats:
                assert cat in LORA_CATEGORIES

    def test_chat_mode_prefers_chat_assist(self):
        assert MODE_CATEGORY_MAP["chat"][0] == "chat_assist"

    def test_agent_mode_prefers_agentic(self):
        assert MODE_CATEGORY_MAP["agent"][0] == "agentic"

    def test_self_play_mode_prefers_self_play(self):
        assert MODE_CATEGORY_MAP["self_play"][0] == "self_play"


# ── LoRAEntry with category ────────────────────────────────────────────


class TestLoRAEntryCategory:
    def test_default_category(self):
        e = LoRAEntry(name="test.safetensors", path="test")
        assert e.category == "uncategorized"

    def test_category_field(self):
        e = LoRAEntry(name="test.safetensors", path="test", category="coding")
        assert e.category == "coding"


# ── LoraHarness ────────────────────────────────────────────────────────


class TestLoraHarness:
    @pytest.fixture()
    def mock_runtime(self):
        return MagicMock()

    @pytest.fixture()
    def manager(self):
        return LoraManager()

    @pytest.fixture()
    def harness(self, manager, mock_runtime):
        h = LoraHarness(manager, mock_runtime)
        # mock _load to avoid starting QThreads in unit tests
        h._load = MagicMock()
        return h

    def test_initial_mode(self, harness):
        assert harness.mode == "chat"

    def test_set_mode(self, harness):
        harness.set_mode("agent")
        assert harness.mode == "agent"

    def test_set_unknown_mode_ignored(self, harness):
        harness.set_mode("nonexistent")
        assert harness.mode == "chat"

    def test_pin_adapter(self, harness):
        harness.pin_adapter("some/path.safetensors")
        assert harness.pinned_adapter == "some/path.safetensors"

    def test_unpin(self, harness):
        harness.pin_adapter("some/path.safetensors")
        harness.unpin()
        assert harness.pinned_adapter is None

    def test_adapters_by_category(self, harness, monkeypatch):
        # mock scan to return test entries
        test_entries = [
            LoRAEntry(name="coding_lora.safetensors", path="a", category="coding"),
            LoRAEntry(name="math_lora.safetensors", path="b", category="math"),
        ]
        monkeypatch.setattr("forge_gui.api.lora_store.scan_lora_adapters", lambda: test_entries)
        by_cat = harness.adapters_by_category()
        assert len(by_cat["coding"]) == 1
        assert len(by_cat["math"]) == 1

    def test_recommend_for_mode(self, harness, monkeypatch):
        test_entries = [
            LoRAEntry(name="coding_lora.safetensors", path="a", category="coding", modified=100.0),
            LoRAEntry(name="agentic_lora.safetensors", path="b", category="agentic", modified=200.0),
        ]
        monkeypatch.setattr("forge_gui.api.lora_store.scan_lora_adapters", lambda: test_entries)
        # agent mode prefers agentic
        rec = harness.recommend_for_mode("agent")
        assert rec is not None
        assert rec.category == "agentic"

    def test_recommend_no_match(self, harness, monkeypatch):
        test_entries = [
            LoRAEntry(name="vision_lora.safetensors", path="a", category="vision"),
        ]
        monkeypatch.setattr("forge_gui.api.lora_store.scan_lora_adapters", lambda: test_entries)
        # chat mode prefers chat_assist, reasoning, coding — none available
        rec = harness.recommend_for_mode("chat")
        assert rec is None

"""Unit tests for forge_gui.api.lorebook.

Tests entry CRUD, keyword matching, hybrid injection, search, and
MemoryTools dispatch. Uses tmp_path for isolation.
"""
import pytest

from forge_gui.api.lorebook import (
    Lorebook,
    LoreEntry,
    MemoryTools,
    memory_tool_defs,
)


@pytest.fixture()
def lore(tmp_path):
    return Lorebook(root=tmp_path)


# ── LoreEntry ──────────────────────────────────────────────────────────

class TestLoreEntry:
    def test_matches_keyword(self):
        e = LoreEntry(id="1", keys=["python", "naming"], content="use snake_case")
        assert e.matches("I prefer python naming")
        assert e.matches("PYTHON is great")
        assert not e.matches("java conventions")

    def test_matches_empty_keys(self):
        e = LoreEntry(id="1", keys=[], content="test")
        assert not e.matches("anything")

    def test_to_dict_roundtrip(self):
        e = LoreEntry(id="1", keys=["a"], content="b", constant=True,
                      priority=50, category="user_pref")
        d = e.to_dict()
        e2 = LoreEntry.from_dict(d)
        assert e2.id == "1"
        assert e2.keys == ["a"]
        assert e2.content == "b"
        assert e2.constant is True
        assert e2.priority == 50
        assert e2.category == "user_pref"


# ── Lorebook CRUD ──────────────────────────────────────────────────────

class TestLorebookCRUD:
    def test_empty_lorebook(self, lore):
        assert lore.entries == []
        assert lore.stats()["total"] == 0

    def test_add_entry(self, lore):
        e = lore.add(["python"], "use snake_case", category="user_pref")
        assert e.id in [x.id for x in lore.entries]
        assert e.category == "user_pref"
        # persisted
        assert lore._path.is_file()

    def test_add_persists_across_instances(self, lore, tmp_path):
        lore.add(["test"], "remember this")
        lore2 = Lorebook(root=tmp_path)
        assert len(lore2.entries) == 1
        assert lore2.entries[0].content == "remember this"

    def test_get_entry(self, lore):
        e = lore.add(["x"], "content")
        assert lore.get(e.id) is not None
        assert lore.get("nonexistent") is None

    def test_update_entry(self, lore):
        e = lore.add(["x"], "old")
        lore.update(e.id, content="new", priority=50)
        assert lore.get(e.id).content == "new"
        assert lore.get(e.id).priority == 50

    def test_delete_entry(self, lore):
        e = lore.add(["x"], "content")
        assert lore.delete(e.id) is True
        assert lore.get(e.id) is None

    def test_delete_nonexistent(self, lore):
        assert lore.delete("nonexistent") is False

    def test_forget_by_key(self, lore):
        lore.add(["python"], "use snake_case")
        lore.add(["java"], "use camelCase")
        n = lore.forget(["python"])
        assert n == 1
        assert len(lore.entries) == 1
        assert lore.entries[0].keys == ["java"]


# ── Search ─────────────────────────────────────────────────────────────

class TestLorebookSearch:
    def test_search_by_key(self, lore):
        lore.add(["python", "naming"], "use snake_case")
        lore.add(["java", "naming"], "use camelCase")
        results = lore.search("python")
        assert len(results) == 1
        assert "snake_case" in results[0].content

    def test_search_by_content(self, lore):
        lore.add(["x"], "the model should use bitnet quantization")
        results = lore.search("bitnet")
        assert len(results) == 1

    def test_search_no_match(self, lore):
        lore.add(["xyzabc"], "content")
        assert lore.search("nonexistent") == []

    def test_search_limit(self, lore):
        for i in range(5):
            lore.add([f"key{i}"], f"content{i}")
        results = lore.search("key", limit=2)
        assert len(results) == 2

    def test_search_disabled_excluded(self, lore):
        e = lore.add(["python"], "content")
        lore.update(e.id, enabled=False)
        assert lore.search("python") == []


# ── Injection ──────────────────────────────────────────────────────────

class TestLorebookInjection:
    def test_empty_injection(self, lore):
        assert lore.inject([]) == ""

    def test_constant_injection(self, lore):
        lore.add(["rule"], "always be helpful", constant=True)
        text = lore.inject([])
        assert "always be helpful" in text
        assert "=== Memory ===" in text

    def test_keyword_triggered_injection(self, lore):
        lore.add(["python"], "use snake_case for variables")
        msgs = [{"role": "user", "content": "help me with python code"}]
        text = lore.inject(msgs)
        assert "snake_case" in text

    def test_no_trigger_no_injection(self, lore):
        lore.add(["python"], "use snake_case")
        msgs = [{"role": "user", "content": "help me with java"}]
        text = lore.inject(msgs)
        assert text == ""

    def test_priority_ordering(self, lore):
        lore.add(["a"], "low priority", priority=200)
        lore.add(["b"], "high priority", priority=10)
        lore.add(["c"], "medium priority", priority=100)
        # all constant so they all fire
        lore.update(lore.entries[0].id, constant=True)
        lore.update(lore.entries[1].id, constant=True)
        lore.update(lore.entries[2].id, constant=True)
        text = lore.inject([])
        # high priority (10) should appear before low (200)
        assert text.index("high priority") < text.index("low priority")

    def test_token_budget_truncation(self, lore):
        lore.add(["x"], "A" * 1000, constant=True)
        text = lore.inject([], token_budget=10)  # 40 chars budget
        assert len(text) < 200  # should be truncated

    def test_trigger_count_increments(self, lore):
        lore.add(["python"], "content")
        msgs = [{"role": "user", "content": "python help"}]
        lore.inject(msgs)
        assert lore.entries[0].trigger_count == 1


# ── Stats ──────────────────────────────────────────────────────────────

class TestLorebookStats:
    def test_stats(self, lore):
        lore.add(["a"], "x", constant=True, category="user_pref")
        lore.add(["b"], "y", category="project")
        stats = lore.stats()
        assert stats["total"] == 2
        assert stats["enabled"] == 2
        assert stats["constant"] == 1
        assert stats["by_category"]["user_pref"] == 1
        assert stats["by_category"]["project"] == 1


# ── MemoryTools ────────────────────────────────────────────────────────

class TestMemoryTools:
    def test_tool_defs(self):
        defs = memory_tool_defs()
        names = [d["function"]["name"] for d in defs]
        assert "remember" in names
        assert "recall_memory" in names
        assert "forget" in names

    def test_remember(self, lore):
        mt = MemoryTools(lore)
        result = mt.execute("remember", {
            "keys": ["python"], "content": "use snake_case",
            "category": "user_pref"})
        assert result["ok"] is True
        assert "id" in result["result"]

    def test_remember_missing_fields(self, lore):
        mt = MemoryTools(lore)
        result = mt.execute("remember", {"keys": []})
        assert result["ok"] is False

    def test_recall_memory(self, lore):
        lore.add(["python"], "use snake_case")
        mt = MemoryTools(lore)
        result = mt.execute("recall_memory", {"query": "python"})
        assert result["ok"] is True
        assert result["result"]["count"] == 1

    def test_forget(self, lore):
        lore.add(["python"], "content")
        mt = MemoryTools(lore)
        result = mt.execute("forget", {"keys": ["python"]})
        assert result["ok"] is True
        assert result["result"]["deleted"] == 1

    def test_unknown_tool(self, lore):
        mt = MemoryTools(lore)
        result = mt.execute("nonexistent", {})
        assert result["ok"] is False

"""Tests for SFT data loader silent-corruption fix (critique F13).

Validates that ``load_examples``:
  - logs per-row errors with file path, line number, and exception class
  - tracks skips by error class in the summary
  - raises on the first malformed row when ``strict_data=True``
  - still loads valid rows alongside bad ones when ``strict_data=False``
"""
import json
import logging
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import pytest

from forge.training.runners.sft_train import load_examples


@pytest.fixture
def jsonl_file(tmp_path):
    """Write a JSONL file with the given lines and return its path."""
    def _write(lines: list[str]) -> str:
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return str(p)
    return _write


class TestLoadExamplesLogging:
    """Per-row error logging (F13)."""

    def test_valid_rows_loaded(self, jsonl_file):
        lines = [
            json.dumps({"prompt": "hello", "response": "world"}),
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        ]
        path = jsonl_file(lines)
        examples = load_examples([path])
        assert len(examples) == 2

    def test_bad_json_logged_not_silent(self, jsonl_file, caplog):
        """A malformed JSON line should be logged at WARNING, not silently skipped."""
        lines = [
            json.dumps({"prompt": "good", "response": "row"}),
            "this is not json {{{",
            json.dumps({"prompt": "another", "response": "good"}),
        ]
        path = jsonl_file(lines)
        with caplog.at_level(logging.WARNING, logger="forge.training.runners.sft_train"):
            examples = load_examples([path])
        # Two valid rows loaded
        assert len(examples) == 2
        # The bad line should appear in the logs
        assert any("JSON parse error" in rec.message for rec in caplog.records)
        # Line number should be in the log
        assert any(":2" in rec.message or "line 2" in rec.message.lower()
                    for rec in caplog.records)

    def test_schema_mismatch_logged(self, jsonl_file, caplog):
        """A row missing both 'messages' and 'prompt' should be logged."""
        lines = [
            json.dumps({"prompt": "good", "response": "row"}),
            json.dumps({"unrelated": "data"}),
        ]
        path = jsonl_file(lines)
        with caplog.at_level(logging.WARNING, logger="forge.training.runners.sft_train"):
            examples = load_examples([path])
        assert len(examples) == 1
        assert any("Schema mismatch" in rec.message for rec in caplog.records)

    def test_skip_summary_by_class(self, jsonl_file, caplog):
        """The per-file summary should break down skips by error class."""
        lines = [
            "not json 1",
            "not json 2",
            json.dumps({"prompt": "good", "response": "row"}),
        ]
        path = jsonl_file(lines)
        with caplog.at_level(logging.WARNING, logger="forge.training.runners.sft_train"):
            load_examples([path])
        # Summary should mention JSONDecodeError count
        assert any("JSONDecodeError=2" in rec.message for rec in caplog.records)


class TestStrictData:
    """--strict-data flag behavior (F13)."""

    def test_strict_raises_on_bad_json(self, jsonl_file):
        lines = [
            json.dumps({"prompt": "good", "response": "row"}),
            "not json at all",
        ]
        path = jsonl_file(lines)
        with pytest.raises(json.JSONDecodeError):
            load_examples([path], strict_data=True)

    def test_strict_raises_on_schema_mismatch(self, jsonl_file):
        lines = [
            json.dumps({"prompt": "good", "response": "row"}),
            json.dumps({"unrelated": "data"}),
        ]
        path = jsonl_file(lines)
        with pytest.raises(ValueError, match="Schema mismatch"):
            load_examples([path], strict_data=True)

    def test_strict_passes_when_all_valid(self, jsonl_file):
        lines = [
            json.dumps({"prompt": "good", "response": "row"}),
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        ]
        path = jsonl_file(lines)
        examples = load_examples([path], strict_data=True)
        assert len(examples) == 2

    def test_non_strict_skips_bad_rows(self, jsonl_file):
        """Default (strict_data=False) should skip bad rows, not raise."""
        lines = [
            "bad json",
            json.dumps({"prompt": "good", "response": "row"}),
            json.dumps({"unrelated": "data"}),
        ]
        path = jsonl_file(lines)
        examples = load_examples([path], strict_data=False)
        assert len(examples) == 1


class TestMissingFile:
    """Missing files should warn, not crash."""

    def test_missing_file_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="forge.training.runners.sft_train"):
            examples = load_examples(["/nonexistent/path/file.jsonl"])
        assert len(examples) == 0
        assert any("not found" in rec.message for rec in caplog.records)

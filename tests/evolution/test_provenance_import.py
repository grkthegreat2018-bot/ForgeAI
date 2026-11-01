"""Quick import + smoke test for DB provenance columns + CurriculumFineTuner.

Run:
  set PYTHONPATH=<repo-root>
  python tests\\evolution\\test_provenance_import.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge.evolution.curriculum_finetuner import CurriculumFineTuner
from forge.evolution.database import FindingsDB


def test_schema_migration():
    """Create a fresh DB, verify provenance columns + schema_meta exist."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        # Check schema_meta
        cols = db._table_columns("schema_meta")
        assert "key" in cols and "value" in cols, "schema_meta missing cols"
        # Check discoveries provenance columns
        dcols = db._table_columns("discoveries")
        for expected in [
            "script_text",
            "input_text",
            "output_text",
            "expected_text",
            "gen_model_size",
            "gen_model_version",
            "scoring_hash",
            "content_sha256",
        ]:
            assert expected in dcols, f"discoveries missing {expected}"
        # Check gen_models table
        gmcols = db._table_columns("gen_models")
        for expected in [
            "id",
            "version",
            "config_json",
            "weights_blob",
            "param_count",
            "performance_score",
            "timestamp",
        ]:
            assert expected in gmcols, f"gen_models missing {expected}"
        # Check gen_model_performance table
        gmpcols = db._table_columns("gen_model_performance")
        for expected in ["id", "version", "domain", "round", "score", "param_count", "timestamp"]:
            assert expected in gmpcols, f"gen_model_performance missing {expected}"
        # Check schema_version recorded
        c = db.conn.cursor()
        c.execute("SELECT value FROM schema_meta WHERE key='schema_version'")
        row = c.fetchone()
        assert row and row[0] == db.SCHEMA_VERSION, \
            f"schema_version wrong: {row}"
        db.close()
        print("[PASS] schema_migration")
    finally:
        os.unlink(db_path)


def test_idempotent_migration():
    """Open the same DB twice — migration should be safe."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db1 = FindingsDB(db_path)
        db1.close()
        db2 = FindingsDB(db_path)  # second open triggers migration again
        dcols = db2._table_columns("discoveries")
        assert "input_text" in dcols
        db2.close()
        print("[PASS] idempotent_migration")
    finally:
        os.unlink(db_path)


def test_save_discoveries_with_provenance():
    """Save a discovery with provenance fields, query it back."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        db.save_run("run1", "test_domain", {"foo": 1}, {"best_score": 5.0}, 0.0)
        discoveries = [
            {
                "config": {"x": 0.5},
                "score": 5.0,
                "generation": 1,
                "input_text": "What is 2+2?",
                "output_text": "4",
                "expected_text": "4",
                "script_text": "assert ans == 4",
                "gen_model_size": 1200000000,
                "gen_model_version": "forgelm_v2",
                "scoring_hash": "abc123def456",
            }
        ]
        n_saved, _n_upd = db.save_discoveries("run1", "test_domain", discoveries)
        assert n_saved == 1, f"expected 1 saved, got {n_saved}"
        # Query back
        results = db.query_discoveries("test_domain", min_score=0.0)
        assert len(results) == 1
        r = results[0]
        assert r["input_text"] == "What is 2+2?"
        assert r["output_text"] == "4"
        assert r["expected_text"] == "4"
        assert r["scoring_hash"] == "abc123def456"
        assert r["gen_model_version"] == "forgelm_v2"
        db.close()
        print("[PASS] save_discoveries_with_provenance")
    finally:
        os.unlink(db_path)


def test_backward_compat():
    """Save without provenance fields — should still work."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        db.save_run("run1", "test_domain", {}, {"best_score": 1.0}, 0.0)
        discoveries = [{"config": {"x": 0.3}, "score": 1.0, "generation": 0}]
        n_saved, _ = db.save_discoveries("run1", "test_domain", discoveries)
        assert n_saved == 1
        db.close()
        print("[PASS] backward_compat")
    finally:
        os.unlink(db_path)


def test_curriculum_data():
    """Test get_curriculum_data returns provenance rows sorted ASC."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        db.save_run("run1", "d", {}, {"best_score": 10}, 0.0)
        disc = [
            {"config": {"x": 0.1}, "score": 10.0, "input_text": "q1", "output_text": "a1"},
            {"config": {"x": 0.2}, "score": 2.0, "input_text": "q2", "output_text": "a2"},
            {"config": {"x": 0.3}, "score": 5.0, "input_text": "q3", "output_text": "a3"},
            {"config": {"x": 0.4}, "score": 1.0},  # no provenance
        ]
        db.save_discoveries("run1", "d", disc)
        curr = db.get_curriculum_data(min_score=0.0, limit=100)
        assert len(curr) == 3, f"expected 3, got {len(curr)}"
        # Sorted ASC by score
        assert curr[0]["score"] == 2.0
        assert curr[1]["score"] == 5.0
        assert curr[2]["score"] == 10.0
        db.close()
        print("[PASS] curriculum_data")
    finally:
        os.unlink(db_path)


def test_scoring_hash_guard():
    """Test scoring hash save/get/check."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        # No hash stored → incompatible
        assert db.check_scoring_compatibility("d", "h1") is False
        db.save_scoring_hash("d", "h1")
        # Same hash → compatible
        assert db.check_scoring_compatibility("d", "h1") is True
        # Different hash → incompatible
        assert db.check_scoring_compatibility("d", "h2") is False
        assert db.get_scoring_hash("d") == "h1"
        db.close()
        print("[PASS] scoring_hash_guard")
    finally:
        os.unlink(db_path)


def test_gen_model_storage():
    """Test save/load gen model + performance history."""
    import torch

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        # Save a gen model (fake state_dict)
        sd = {"layer.weight": torch.zeros(4, 4)}
        db.save_gen_model("v1", {"arch": "test"}, sd, 1200000, -0.5)
        # Load it back
        loaded = db.load_gen_model("v1")
        assert loaded is not None
        assert loaded["version"] == "v1"
        assert loaded["param_count"] == 1200000
        assert loaded["performance_score"] == -0.5
        # Latest
        latest = db.get_latest_gen_model()
        assert latest is not None and latest["version"] == "v1"
        # Performance history
        db.record_gen_model_performance("v1", "d1", 0, 5.0, 1200000)
        db.record_gen_model_performance("v1", "d2", 1, 7.0, 1200000)
        hist = db.get_gen_model_performance_history("v1")
        assert len(hist) == 2
        assert hist[0]["domain"] == "d1"
        db.close()
        print("[PASS] gen_model_storage")
    finally:
        os.unlink(db_path)


def test_curriculum_finetuner_init():
    """Test CurriculumFineTuner can resolve a plain nn.Module."""
    import torch.nn as nn

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = FindingsDB(db_path)
        # Plain nn.Module (no .model attr) → should resolve directly
        model = nn.Linear(10, 10)
        tuner = CurriculumFineTuner(db, model, tokenizer=_FakeTok())
        assert tuner._nn_module is model
        stats = tuner.get_curriculum_stats()
        assert stats["total"] == 0
        result = tuner.fine_tune()
        assert result["n_examples"] == 0
        db.close()
        print("[PASS] curriculum_finetuner_init")
    finally:
        os.unlink(db_path)


class _FakeTok:
    """Minimal tokenizer stub for testing (avoids HF dependency)."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 100 for c in text[:50]]


if __name__ == "__main__":
    test_schema_migration()
    test_idempotent_migration()
    test_save_discoveries_with_provenance()
    test_backward_compat()
    test_curriculum_data()
    test_scoring_hash_guard()
    test_gen_model_storage()
    test_curriculum_finetuner_init()
    print("\nALL TESTS PASSED")

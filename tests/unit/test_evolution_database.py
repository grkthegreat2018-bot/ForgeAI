"""Tests for forge.evolution.database — FindingsDB SQLite persistence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sqlite3

import pytest
import torch
import torch.nn as nn

from forge.evolution.database import FindingsDB

# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class _Cfg:
    """Minimal cfg object for BatchedGenerator."""

    def __init__(self, noise_dim=4, context_dim=3, hidden_dim=8, output_dim=2, n_generators=2):
        self.noise_dim = noise_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_generators = n_generators


class MockBatchedGenerator(nn.Module):
    """Minimal BatchedGenerator mock with named_parameters, fitness_ema, cfg."""

    def __init__(self, n_generators=2, noise_dim=4, context_dim=3, hidden_dim=8, output_dim=2):
        super().__init__()
        self.cfg = _Cfg(noise_dim, context_dim, hidden_dim, output_dim, n_generators)
        in_dim = noise_dim + context_dim
        self.W1 = nn.Parameter(torch.randn(n_generators, in_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.randn(n_generators, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(n_generators, hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.randn(n_generators, output_dim))
        self.register_buffer("fitness_ema", torch.randn(n_generators, dtype=torch.float32))

    def load_state_dict(self, sd, strict=True):
        super().load_state_dict(sd, strict=strict)


class MockSurrogateNet(nn.Module):
    """Minimal MLP net used by MockSurrogate."""

    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)


class MockSurrogate:
    """Minimal Surrogate mock with mode, net, n_trained, input_dim, hidden_dim,
    and an `ensemble` list of nn.Module (per task spec)."""

    def __init__(self, mode="mlp", input_dim=4, hidden_dim=8, n_trained=0):
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_trained = n_trained
        self.net = MockSurrogateNet(input_dim, hidden_dim)
        self.ensemble = [self.net]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """A FindingsDB instance backed by tmp_path / test.db."""
    instance = FindingsDB(tmp_path / "test.db")
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindingsDBInit:
    """Initialization creates the DB file and all tables."""

    def test_db_file_created(self, tmp_path):
        path = tmp_path / "test.db"
        assert not path.exists()
        d = FindingsDB(path)
        assert path.exists()
        d.close()

    def test_tables_exist(self, db):
        c = db.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        names = {r[0] for r in c.fetchall()}
        assert "runs" in names
        assert "discoveries" in names
        assert "evaluations" in names
        assert "generators" in names
        assert "surrogate" in names


class TestSaveAndListRuns:
    """save_run + list_runs round-trip and domain filtering."""

    def test_save_and_list_run(self, db):
        cfg = {"lr": 0.01, "gens": 5}
        results = {
            "generations": 5,
            "total_evaluations": 50,
            "discoveries": 3,
            "best_score": 1.5,
            "best_config": {"x": 1},
            "device": "cpu",
        }
        db.save_run("run1", "quant", cfg, results, start_time=100.0)

        runs = db.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run1"
        assert runs[0]["domain"] == "quant"
        assert runs[0]["generations"] == 5
        assert runs[0]["total_evals"] == 50
        assert runs[0]["discoveries"] == 3
        assert runs[0]["best_score"] == 1.5
        assert runs[0]["device"] == "cpu"

    def test_list_runs_filter_by_domain(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_run("r2", "mesh", {}, {"generations": 1}, 2.0)

        quant_runs = db.list_runs(domain="quant")
        assert len(quant_runs) == 1
        assert quant_runs[0]["run_id"] == "r1"

        mesh_runs = db.list_runs(domain="mesh")
        assert len(mesh_runs) == 1
        assert mesh_runs[0]["run_id"] == "r2"

        all_runs = db.list_runs()
        assert len(all_runs) == 2


class TestSaveAndQueryDiscoveries:
    """save_discoveries + query_discoveries filtering by domain and min_score."""

    def test_save_and_query_discoveries(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        discoveries = [
            {"generation": 0, "config": {"a": 1}, "score": 0.5, "behavioral": {"b": 1}, "metadata": {"k": "v"}},
            {"generation": 1, "config": {"a": 2}, "score": 1.5, "behavioral": {"b": 2}, "metadata": {}},
        ]
        db.save_discoveries("r1", "quant", discoveries)

        results = db.query_discoveries("quant")
        assert len(results) == 2
        # sorted by score descending
        assert results[0]["score"] == 1.5
        assert results[1]["score"] == 0.5
        assert results[0]["config"] == {"a": 2}
        assert results[0]["run_id"] == "r1"

    def test_query_discoveries_min_score(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        discoveries = [
            {"generation": 0, "config": {"a": 1}, "score": 0.1},
            {"generation": 1, "config": {"a": 2}, "score": 1.0},
            {"generation": 2, "config": {"a": 3}, "score": 2.0},
        ]
        db.save_discoveries("r1", "quant", discoveries)

        results = db.query_discoveries("quant", min_score=1.0)
        assert len(results) == 2
        assert all(r["score"] >= 1.0 for r in results)
        assert results[0]["score"] == 2.0

    def test_query_discoveries_domain_filter(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_run("r2", "mesh", {}, {"generations": 1}, 2.0)
        db.save_discoveries("r1", "quant", [{"config": {}, "score": 1.0}])
        db.save_discoveries("r2", "mesh", [{"config": {}, "score": 5.0}])

        quant = db.query_discoveries("quant")
        assert len(quant) == 1
        assert quant[0]["score"] == 1.0

    def test_save_discoveries_dedup_same_score(self, db):
        """Identical config with same score is skipped (true duplicate)."""
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        discoveries = [{"config": {"a": 1}, "score": 5.0}]
        n_saved, n_skipped = db.save_discoveries("r1", "quant", discoveries)
        assert n_saved == 1
        assert n_skipped == 0

        # Same config, same score → should be skipped
        discoveries2 = [{"config": {"a": 1}, "score": 5.0}]
        n_saved, n_skipped = db.save_discoveries("r2", "quant", discoveries2)
        assert n_saved == 0
        assert n_skipped == 1

        # Verify only 1 row in DB
        results = db.query_discoveries("quant")
        assert len(results) == 1

    def test_save_discoveries_dedup_lower_score(self, db):
        """Identical config with lower score is skipped (worse duplicate)."""
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_discoveries("r1", "quant", [{"config": {"a": 1}, "score": 10.0}])
        # Same config, lower score → skipped
        n_saved, n_skipped = db.save_discoveries("r2", "quant", [{"config": {"a": 1}, "score": 5.0}])
        assert n_saved == 0
        assert n_skipped == 1

        results = db.query_discoveries("quant")
        assert len(results) == 1
        assert results[0]["score"] == 10.0  # kept the better one

    def test_save_discoveries_dedup_higher_score_updates(self, db):
        """Identical config with higher score UPDATES the existing row
        (no duplicate inserted, just the score is updated)."""
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_discoveries("r1", "quant", [{"config": {"a": 1}, "score": 5.0}])
        # Same config, higher score → updates existing row, no new row
        n_saved, n_skipped = db.save_discoveries("r2", "quant", [{"config": {"a": 1}, "score": 15.0}])
        assert n_saved == 0  # no new row inserted
        assert n_skipped == 1  # 1 updated (counted as skipped/deduped)

        results = db.query_discoveries("quant")
        assert len(results) == 1  # still only 1 row (updated, not duplicated)
        assert results[0]["score"] == 15.0  # score was updated to the better one


class TestSaveEvaluation:
    """save_evaluation stores a single evaluation row."""

    def test_save_evaluation(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_evaluation("r1", "quant", 0, {"x": 1}, 0.7, {"meta": "ok"})

        c = db.conn.cursor()
        c.execute("SELECT run_id, domain, generation, config_json, score, metadata_json FROM evaluations")
        row = c.fetchone()
        assert row is not None
        assert row[0] == "r1"
        assert row[1] == "quant"
        assert row[2] == 0
        assert '"x"' in row[3]
        assert row[4] == 0.7
        assert "ok" in row[5]


class TestSaveLoadGenerators:
    """save_generators + load_generators round-trip; missing run returns False."""

    def test_save_and_load_generators(self, db):
        gen = MockBatchedGenerator()
        # snapshot original weights
        orig = {n: p.clone() for n, p in gen.named_parameters()}
        orig_fit = gen.fitness_ema.clone()

        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_generators("r1", "quant", gen)

        # mutate weights so we can detect load
        with torch.no_grad():
            for p in gen.parameters():
                p.zero_()
            gen.fitness_ema.zero_()

        loaded = db.load_generators("r1", gen)
        assert loaded is True

        for n, p in gen.named_parameters():
            assert torch.allclose(p, orig[n]), f"param {n} mismatch"
        assert torch.allclose(gen.fitness_ema, orig_fit)

    def test_load_generators_nonexistent_run(self, db):
        gen = MockBatchedGenerator()
        loaded = db.load_generators("nope", gen)
        assert loaded is False


class TestSaveLoadSurrogate:
    """save_surrogate + load_surrogate round-trip; non-MLP returns early."""

    def test_save_and_load_surrogate(self, db):
        surr = MockSurrogate(mode="mlp", input_dim=4, hidden_dim=8, n_trained=0)
        orig = {n: p.clone() for n, p in surr.net.named_parameters()}

        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        db.save_surrogate("r1", "quant", surr)

        # mutate
        with torch.no_grad():
            for p in surr.net.parameters():
                p.zero_()

        loaded = db.load_surrogate("r1", surr)
        assert loaded is True
        assert surr.n_trained == 0

        for n, p in surr.net.named_parameters():
            assert torch.allclose(p, orig[n]), f"param {n} mismatch"

    def test_save_surrogate_non_mlp_returns_early(self, db):
        surr = MockSurrogate(mode="cnn")
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        # should return without writing
        db.save_surrogate("r1", "quant", surr)

        c = db.conn.cursor()
        c.execute("SELECT COUNT(*) FROM surrogate")
        assert c.fetchone()[0] == 0

    def test_load_surrogate_non_mlp_returns_false(self, db):
        surr = MockSurrogate(mode="cnn")
        loaded = db.load_surrogate("r1", surr)
        assert loaded is False

    def test_load_surrogate_nonexistent_run(self, db):
        surr = MockSurrogate(mode="mlp")
        loaded = db.load_surrogate("nope", surr)
        assert loaded is False


class TestQueryBestConfigs:
    """query_best_configs returns sorted by score descending."""

    def test_query_best_configs_sorted(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        discoveries = [
            {"config": {"a": 1}, "score": 0.3},
            {"config": {"a": 2}, "score": 2.0},
            {"config": {"a": 3}, "score": 1.0},
        ]
        db.save_discoveries("r1", "quant", discoveries)

        best = db.query_best_configs("quant", limit=10)
        scores = [d["score"] for d in best]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 2.0

    def test_query_best_configs_limit(self, db):
        db.save_run("r1", "quant", {}, {"generations": 1}, 1.0)
        discoveries = [{"config": {"a": i}, "score": float(i)} for i in range(5)]
        db.save_discoveries("r1", "quant", discoveries)

        best = db.query_best_configs("quant", limit=2)
        assert len(best) == 2
        assert best[0]["score"] == 4.0
        assert best[1]["score"] == 3.0


class TestContextManager:
    """`with FindingsDB(path) as db:` works and closes after."""

    def test_context_manager(self, tmp_path):
        path = tmp_path / "ctx.db"
        with FindingsDB(path) as d:
            c = d.conn.cursor()
            c.execute("SELECT 1")
            assert c.fetchone()[0] == 1
        # after exit, connection should be closed -> using it raises
        with pytest.raises(sqlite3.ProgrammingError):
            d.conn.cursor().execute("SELECT 1")


class TestClose:
    """close() can be called without error; double-close is safe."""

    def test_close(self, tmp_path):
        d = FindingsDB(tmp_path / "c.db")
        d.close()
        # using a closed connection should raise
        with pytest.raises(sqlite3.ProgrammingError):
            d.conn.cursor().execute("SELECT 1")

    def test_double_close_is_safe(self, tmp_path):
        d = FindingsDB(tmp_path / "c2.db")
        d.close()
        # second close should not raise (sqlite3.close on closed conn is no-op)
        d.close()


class TestSeedFromPast:
    """seed_from_past returns 0 when no past data exists."""

    def test_seed_from_past_empty(self, db):
        gen = MockBatchedGenerator()
        surr = MockSurrogate(mode="mlp")
        n = db.seed_from_past("quant", gen, surr, n_seed=5)
        assert n == 0


# ---------------------------------------------------------------------------
# Canonical generators/surrogate tests (permanent knowledge layer)
# ---------------------------------------------------------------------------


class TestCanonicalGenerators:
    """save_canonical_generators / load_canonical_generators round-trip."""

    def test_save_and_load_canonical(self, db):
        gen = MockBatchedGenerator()
        # Save original weights
        orig = {n: p.clone() for n, p in gen.named_parameters()}
        updated = db.save_canonical_generators("quant", gen, best_score=5.0, run_id="r1")
        assert updated is True

        # Mutate the generator
        with torch.no_grad():
            for p in gen.parameters():
                p.add_(torch.randn_like(p) * 10)

        # Load canonical — should restore original weights
        loaded = db.load_canonical_generators("quant", gen)
        assert loaded is True
        for n, p in gen.named_parameters():
            assert torch.allclose(p, orig[n]), f"param {n} not restored"

    def test_save_canonical_only_updates_if_better(self, db):
        gen = MockBatchedGenerator()
        db.save_canonical_generators("quant", gen, best_score=10.0, run_id="r1")

        # Try to save with lower score — should NOT update
        updated = db.save_canonical_generators("quant", gen, best_score=5.0, run_id="r2")
        assert updated is False

        # Try with higher score — should update
        updated = db.save_canonical_generators("quant", gen, best_score=15.0, run_id="r3")
        assert updated is True

    def test_load_canonical_nonexistent_returns_false(self, db):
        gen = MockBatchedGenerator()
        loaded = db.load_canonical_generators("nonexistent", gen)
        assert loaded is False

    def test_get_canonical_best_score(self, db):
        gen = MockBatchedGenerator()
        assert db.get_canonical_best_score("quant") is None
        db.save_canonical_generators("quant", gen, best_score=7.5, run_id="r1")
        assert db.get_canonical_best_score("quant") == 7.5

    def test_canonical_persists_across_db_reopen(self, tmp_path):
        """Canonical data survives DB close/reopen (permanent knowledge)."""
        path = tmp_path / "persist.db"
        gen = MockBatchedGenerator()
        orig = {n: p.clone() for n, p in gen.named_parameters()}

        with FindingsDB(path) as db:
            db.save_canonical_generators("quant", gen, best_score=3.0, run_id="r1")

        # Reopen
        with FindingsDB(path) as db2:
            # Mutate gen
            with torch.no_grad():
                for p in gen.parameters():
                    p.add_(torch.randn_like(p) * 10)
            loaded = db2.load_canonical_generators("quant", gen)
            assert loaded is True
            for n, p in gen.named_parameters():
                assert torch.allclose(p, orig[n])
            assert db2.get_canonical_best_score("quant") == 3.0


class TestCanonicalSurrogate:
    """save_canonical_surrogate / load_canonical_surrogate round-trip."""

    def test_save_and_load_canonical_surrogate(self, db):
        surr = MockSurrogate(mode="mlp", n_trained=42)
        orig = {n: p.clone() for n, p in surr.net.named_parameters()}
        updated = db.save_canonical_surrogate("quant", surr, best_score=5.0, run_id="r1")
        assert updated is True

        # Mutate
        with torch.no_grad():
            for p in surr.net.parameters():
                p.add_(torch.randn_like(p) * 10)
        surr.n_trained = 0

        loaded = db.load_canonical_surrogate("quant", surr)
        assert loaded is True
        for n, p in surr.net.named_parameters():
            assert torch.allclose(p, orig[n])
        assert surr.n_trained == 42

    def test_save_canonical_surrogate_only_if_better(self, db):
        surr = MockSurrogate(mode="mlp")
        db.save_canonical_surrogate("quant", surr, best_score=10.0, run_id="r1")
        updated = db.save_canonical_surrogate("quant", surr, best_score=5.0, run_id="r2")
        assert updated is False
        updated = db.save_canonical_surrogate("quant", surr, best_score=20.0, run_id="r3")
        assert updated is True

    def test_canonical_surrogate_non_mlp_returns_false(self, db):
        surr = MockSurrogate(mode="gp")
        updated = db.save_canonical_surrogate("quant", surr, best_score=1.0, run_id="r1")
        assert updated is False
        loaded = db.load_canonical_surrogate("quant", surr)
        assert loaded is False


class TestCanonicalTablesExist:
    """The canonical tables are created on init."""

    def test_canonical_tables_exist(self, db):
        c = db.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        names = {r[0] for r in c.fetchall()}
        assert "canonical_generators" in names
        assert "canonical_surrogate" in names

"""Unit tests for MapElitesArchive — MAP-Elites quality-diversity archive.

Tests grid binning, add/replace logic, Pareto front, UCB sampling,
history capping, and edge cases (empty archive, boundary values).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from forge.evolution.archive import ArchiveEntry, MapElitesArchive

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_archive():
    """2D archive: (compression, error) with 5x4 grid."""
    return MapElitesArchive(
        dims=[("compression", 5, 0.0, 10.0), ("error", 4, 0.0, 1.0)],
        max_history=100,
    )


@pytest.fixture
def filled_archive(small_archive):
    """Archive with several cells filled."""
    entries = [
        ({"block_size": 32}, 5.0, (2.0, 0.3), 0),
        ({"block_size": 64}, 7.0, (4.0, 0.1), 1),
        ({"block_size": 16}, 3.0, (1.0, 0.5), 2),
        ({"block_size": 128}, 8.0, (8.0, 0.05), 3),
    ]
    for config, score, beh, gen in entries:
        small_archive.add(config, score, beh, gen)
    return small_archive


# ── Grid initialization ───────────────────────────────────────────────────


class TestArchiveInit:
    def test_grid_has_correct_shape(self, small_archive):
        assert len(small_archive.grid) == 5 * 4  # 20 cells

    def test_all_cells_init_none(self, small_archive):
        assert all(v is None for v in small_archive.grid.values())

    def test_coverage_zero_on_empty(self, small_archive):
        assert small_archive.coverage() == 0.0

    def test_best_score_neg_inf_on_empty(self, small_archive):
        assert small_archive.best_score == -float("inf")

    def test_best_entry_none_on_empty(self, small_archive):
        assert small_archive.best_entry is None

    def test_history_is_deque(self, small_archive):
        import collections

        assert isinstance(small_archive.history, collections.deque)
        assert small_archive.history.maxlen == 100


# ── Binning logic ─────────────────────────────────────────────────────────


class TestArchiveBinning:
    def test_bin_clamps_low(self, small_archive):
        idx = small_archive._to_bin((-1.0, -1.0))
        assert idx == (0, 0)

    def test_bin_clamps_high(self, small_archive):
        idx = small_archive._to_bin((100.0, 100.0))
        assert idx == (4, 3)  # max bins

    def test_bin_midpoint(self, small_archive):
        idx = small_archive._to_bin((5.0, 0.5))
        # 5.0 / 10.0 * 5 = 2.5 → 2; 0.5 / 1.0 * 4 = 2.0 → 2
        assert idx == (2, 2)


# ── Add logic ─────────────────────────────────────────────────────────────


class TestArchiveAdd:
    def test_add_to_empty_cell_accepted(self, small_archive):
        ok = small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        assert ok is True

    def test_add_increments_n_filled(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        assert small_archive.n_filled == 1

    def test_add_better_score_replaces(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        ok = small_archive.add({"a": 2}, 7.0, (2.0, 0.3), 1)
        assert ok is True
        entry = small_archive.grid[small_archive._to_bin((2.0, 0.3))]
        assert entry.score == 7.0
        assert entry.config == {"a": 2}

    def test_add_worse_score_rejected(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        ok = small_archive.add({"a": 2}, 3.0, (2.0, 0.3), 1)
        assert ok is False
        entry = small_archive.grid[small_archive._to_bin((2.0, 0.3))]
        assert entry.score == 5.0  # original kept

    def test_add_equal_score_rejected(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        ok = small_archive.add({"a": 2}, 5.0, (2.0, 0.3), 1)
        assert ok is False

    def test_add_does_not_increment_n_filled_on_replace(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        assert small_archive.n_filled == 1
        small_archive.add({"a": 2}, 7.0, (2.0, 0.3), 1)
        assert small_archive.n_filled == 1  # still 1, not 2

    def test_add_updates_best_score(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        assert small_archive.best_score == 5.0
        small_archive.add({"a": 2}, 8.0, (4.0, 0.1), 1)
        assert small_archive.best_score == 8.0
        assert small_archive.best_entry.score == 8.0

    def test_add_invalidates_pareto_cache(self, filled_archive):
        # Force cache build
        filled_archive.get_pareto_front()
        assert filled_archive._pareto_cache is not None
        # Add new entry → cache should be invalidated
        filled_archive.add({"new": True}, 9.0, (9.0, 0.02), 4)
        assert filled_archive._pareto_cache is None

    def test_add_appends_to_history(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        assert len(small_archive.history) == 1

    def test_history_capped_at_max(self, small_archive):
        # max_history=100, add 150 entries
        for i in range(150):
            small_archive.add({"i": i}, float(i), (float(i % 10), float(i % 10) / 10.0), 0)
        assert len(small_archive.history) == 100  # capped by deque maxlen


# ── Sampling ──────────────────────────────────────────────────────────────


class TestArchiveSampling:
    def test_sample_elite_empty_returns_none(self, small_archive):
        assert small_archive.sample_elite() is None

    def test_sample_elite_returns_entry(self, filled_archive):
        entry = filled_archive.sample_elite()
        assert isinstance(entry, ArchiveEntry)

    def test_sample_elite_ucb_empty_returns_none(self, small_archive):
        assert small_archive.sample_elite_ucb() is None

    def test_sample_elite_ucb_returns_entry(self, filled_archive):
        np.random.seed(42)
        entry = filled_archive.sample_elite_ucb(current_gen=5)
        assert isinstance(entry, ArchiveEntry)

    def test_sample_elite_ucb_increments_pulls(self, filled_archive):
        np.random.seed(42)
        filled_archive.sample_elite_ucb()
        total_pulls = sum(filled_archive._cell_pulls.values())
        assert total_pulls == 1


# ── Pareto front ──────────────────────────────────────────────────────────


class TestArchivePareto:
    def test_pareto_empty_returns_empty(self, small_archive):
        assert small_archive.get_pareto_front() == []

    def test_pareto_single_entry_is_pareto(self, small_archive):
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        pareto = small_archive.get_pareto_front()
        assert len(pareto) == 1

    def test_pareto_dominated_excluded(self, small_archive):
        # Entry A: (comp=2, err=0.3) — dominated by B
        small_archive.add({"a": 1}, 5.0, (2.0, 0.3), 0)
        # Entry B: (comp=4, err=0.1) — dominates A (more comp, less err)
        small_archive.add({"b": 1}, 7.0, (4.0, 0.1), 1)
        pareto = small_archive.get_pareto_front()
        assert len(pareto) == 1
        assert pareto[0].config == {"b": 1}

    def test_pareto_non_dominated_both_kept(self, small_archive):
        # A: (comp=8, err=0.5) — high comp, high err
        small_archive.add({"a": 1}, 8.0, (8.0, 0.5), 0)
        # B: (comp=2, err=0.1) — low comp, low err
        small_archive.add({"b": 1}, 6.0, (2.0, 0.1), 1)
        pareto = small_archive.get_pareto_front()
        assert len(pareto) == 2  # neither dominates the other

    def test_pareto_cached(self, filled_archive):
        pareto1 = filled_archive.get_pareto_front()
        pareto2 = filled_archive.get_pareto_front()
        assert pareto1 is pareto2  # same object (cached)


# ── Coverage and summary ──────────────────────────────────────────────────


class TestArchiveCoverage:
    def test_coverage_filled(self, filled_archive):
        # 4 entries in 20-cell grid
        assert filled_archive.coverage() == 4 / 20

    def test_get_all_elites_count(self, filled_archive):
        elites = filled_archive.get_all_elites()
        assert len(elites) == 4

    def test_summary_string(self, filled_archive):
        s = filled_archive.summary()
        assert "Archive:" in s
        assert "cells" in s
        assert "best=" in s

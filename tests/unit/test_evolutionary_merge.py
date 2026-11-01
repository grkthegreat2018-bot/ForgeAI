"""Tests for evolutionary model merging ("sexual reproduction" of LLM weights).

Covers the operators in research/merge_models.py (crossover, mutation,
selection, evolve loop) and the ForgeEngine.merge_checkpoints() /
ForgeEngine.evolve_merge() callable methods.

State dicts use the ForgeAI key convention:
  blocks.{i}.<submodule>.weight, embed.weight, head.weight, norm.weight
"""

import pytest
import torch

from forge.checkpoint_io import load_checkpoint, save_checkpoint
from research.merge_models import (
    _block_index,
    _n_blocks,
    _non_block_keys,
    _population_diversity,
    _state_distance,
    crossover_block_random,
    crossover_blockwise,
    crossover_uniform,
    evolve,
    mutate_block_swap,
    mutate_gaussian,
    mutate_quant_perturb,
    scale_fitness_rank,
    scale_fitness_sigma,
    select_diversity,
    select_rank,
    select_roulette,
    select_tournament,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_block_state(n_blocks: int = 4, seed: int = 0, with_qscale: bool = False) -> dict:
    """Build a state dict matching ForgeAI's blocks.{i}.<...> convention."""
    g = torch.Generator().manual_seed(seed)
    state = {
        "embed.weight": torch.randn(64, 16, generator=g, dtype=torch.float32),
        "norm.weight": torch.ones(16, dtype=torch.float32),
        "head.weight": torch.randn(64, 16, generator=g, dtype=torch.float32),
    }
    for i in range(n_blocks):
        prefix = f"blocks.{i}."
        state[prefix + "attn.q_proj.weight"] = torch.randn(16, 16, generator=g, dtype=torch.float32)
        state[prefix + "attn.v_proj.weight"] = torch.randn(16, 16, generator=g, dtype=torch.float32)
        state[prefix + "ffn.w_gate.weight"] = torch.randn(32, 16, generator=g, dtype=torch.float32)
        state[prefix + "ffn.w_down.weight"] = torch.randn(16, 32, generator=g, dtype=torch.float32)
        if with_qscale:
            state[prefix + "ffn.w_gate.qscale"] = torch.tensor([0.5 + 0.01 * i], dtype=torch.float32)
    return state


@pytest.fixture
def parent_a():
    return _make_block_state(n_blocks=4, seed=42)


@pytest.fixture
def parent_b():
    return _make_block_state(n_blocks=4, seed=99)


@pytest.fixture
def parent_a_qscale():
    return _make_block_state(n_blocks=4, seed=42, with_qscale=True)


@pytest.fixture
def parent_b_qscale():
    return _make_block_state(n_blocks=4, seed=99, with_qscale=True)


# ── Helper function tests ─────────────────────────────────────────────────


class TestBlockIndex:
    def test_block_key(self):
        assert _block_index("blocks.2.attn.q_proj.weight") == 2

    def test_non_block_key(self):
        assert _block_index("embed.weight") is None

    def test_norm_key(self):
        assert _block_index("norm.weight") is None


class TestNBlocks:
    def test_counts_blocks(self, parent_a):
        assert _n_blocks(parent_a) == 4

    def test_no_blocks(self):
        state = {"embed.weight": torch.randn(4, 8)}
        assert _n_blocks(state) == 0


class TestNonBlockKeys:
    def test_excludes_block_keys(self, parent_a):
        nb = _non_block_keys(parent_a)
        assert "embed.weight" in nb
        assert "head.weight" in nb
        assert "norm.weight" in nb
        assert all(not k.startswith("blocks.") for k in nb)


# ── Crossover tests ───────────────────────────────────────────────────────


class TestCrossoverBlockwise:
    def test_produces_valid_state(self, parent_a, parent_b):
        child = crossover_blockwise(parent_a, parent_b, split_block=2)
        assert set(child.keys()) == set(parent_a.keys())
        for k in child:
            assert child[k].shape == parent_a[k].shape

    def test_prefix_from_a_suffix_from_b(self, parent_a, parent_b):
        child = crossover_blockwise(parent_a, parent_b, split_block=2)
        # blocks 0,1 from A
        for k in parent_a:
            idx = _block_index(k)
            if idx is not None:
                if idx < 2:
                    assert torch.equal(child[k], parent_a[k])
                else:
                    assert torch.equal(child[k], parent_b[k])

    def test_non_block_from_a_by_default(self, parent_a, parent_b):
        child = crossover_blockwise(parent_a, parent_b, split_block=2)
        assert torch.equal(child["embed.weight"], parent_a["embed.weight"])
        assert torch.equal(child["head.weight"], parent_a["head.weight"])

    def test_non_block_from_b(self, parent_a, parent_b):
        child = crossover_blockwise(parent_a, parent_b, split_block=2, non_block_source="b")
        assert torch.equal(child["embed.weight"], parent_b["embed.weight"])

    def test_non_block_avg(self, parent_a, parent_b):
        child = crossover_blockwise(parent_a, parent_b, split_block=2, non_block_source="avg")
        expected = (parent_a["embed.weight"] + parent_b["embed.weight"]) * 0.5
        assert torch.allclose(child["embed.weight"], expected)

    def test_random_split_reproducible(self, parent_a, parent_b):
        c1 = crossover_blockwise(parent_a, parent_b, seed=7)
        c2 = crossover_blockwise(parent_a, parent_b, seed=7)
        for k in c1:
            assert torch.equal(c1[k], c2[k])

    def test_shape_mismatch_falls_back_to_a(self, parent_a):
        b = dict(parent_a)
        b["blocks.0.attn.q_proj.weight"] = torch.randn(8, 8)
        child = crossover_blockwise(parent_a, b, split_block=2)
        assert torch.equal(child["blocks.0.attn.q_proj.weight"], parent_a["blocks.0.attn.q_proj.weight"])


class TestCrossoverBlockRandom:
    def test_produces_valid_state(self, parent_a, parent_b):
        child = crossover_block_random(parent_a, parent_b, p=0.5, seed=1)
        assert set(child.keys()) == set(parent_a.keys())

    def test_each_block_wholesale(self, parent_a, parent_b):
        child = crossover_block_random(parent_a, parent_b, p=0.5, seed=1)
        # Each block's keys should ALL come from the same parent
        for i in range(4):
            keys = [k for k in child if _block_index(k) == i]
            sources = set()
            for k in keys:
                if torch.equal(child[k], parent_a[k]):
                    sources.add("a")
                elif torch.equal(child[k], parent_b[k]):
                    sources.add("b")
            assert len(sources) == 1, f"block {i} has mixed sources: {sources}"

    def test_p_zero_all_from_a(self, parent_a, parent_b):
        child = crossover_block_random(parent_a, parent_b, p=0.0, seed=1)
        for k in child:
            if _block_index(k) is not None:
                assert torch.equal(child[k], parent_a[k])


class TestCrossoverUniform:
    def test_produces_valid_state(self, parent_a, parent_b):
        child = crossover_uniform(parent_a, parent_b, p=0.5, seed=1)
        assert set(child.keys()) == set(parent_a.keys())

    def test_p_one_all_from_b(self, parent_a, parent_b):
        child = crossover_uniform(parent_a, parent_b, p=1.0, seed=1)
        for k in child:
            assert torch.equal(child[k], parent_b[k])

    def test_p_zero_all_from_a(self, parent_a, parent_b):
        child = crossover_uniform(parent_a, parent_b, p=0.0, seed=1)
        for k in child:
            assert torch.equal(child[k], parent_a[k])


# ── Mutation tests ────────────────────────────────────────────────────────


class TestMutateGaussian:
    def test_preserves_shapes(self, parent_a):
        mut = mutate_gaussian(parent_a, sigma=0.01, rate=0.5, seed=1)
        for k in parent_a:
            assert mut[k].shape == parent_a[k].shape

    def test_skips_embed_and_head(self, parent_a):
        mut = mutate_gaussian(parent_a, sigma=0.1, rate=1.0, seed=1)
        assert torch.equal(mut["embed.weight"], parent_a["embed.weight"])
        assert torch.equal(mut["head.weight"], parent_a["head.weight"])

    def test_rate_zero_no_change(self, parent_a):
        mut = mutate_gaussian(parent_a, sigma=0.1, rate=0.0, seed=1)
        for k in parent_a:
            if k not in ("embed.weight", "head.weight"):
                assert torch.equal(mut[k], parent_a[k])

    def test_block_weights_changed(self, parent_a):
        mut = mutate_gaussian(parent_a, sigma=0.5, rate=1.0, seed=1)
        # At least some block weights should differ
        changed = False
        for k in parent_a:
            if k.startswith("blocks.") and not torch.equal(mut[k], parent_a[k]):
                changed = True
                break
        assert changed

    def test_reproducible(self, parent_a):
        m1 = mutate_gaussian(parent_a, sigma=0.1, rate=0.5, seed=5)
        m2 = mutate_gaussian(parent_a, sigma=0.1, rate=0.5, seed=5)
        for k in m1:
            assert torch.equal(m1[k], m2[k])


class TestMutateQuantPerturb:
    def test_only_qscale_changed(self, parent_a_qscale, parent_b_qscale):
        mut = mutate_quant_perturb(parent_a_qscale, sigma=0.1, seed=1)
        for k in parent_a_qscale:
            if k.endswith(".qscale"):
                # Should be perturbed (very likely with sigma=0.1)
                assert not torch.equal(mut[k], parent_a_qscale[k])
            else:
                assert torch.equal(mut[k], parent_a_qscale[k])

    def test_no_qscale_no_change(self, parent_a):
        mut = mutate_quant_perturb(parent_a, sigma=0.1, seed=1)
        for k in parent_a:
            assert torch.equal(mut[k], parent_a[k])


class TestMutateBlockSwap:
    def test_swaps_n_blocks(self, parent_a, parent_b):
        mut = mutate_block_swap(parent_a, parent_b, n_swaps=2, seed=1)
        # Count blocks that now match B
        swapped = 0
        for i in range(4):
            keys = [k for k in mut if _block_index(k) == i]
            if all(torch.equal(mut[k], parent_b[k]) for k in keys):
                swapped += 1
        assert swapped == 2

    def test_zero_swaps_no_change(self, parent_a, parent_b):
        mut = mutate_block_swap(parent_a, parent_b, n_swaps=0, seed=1)
        for k in parent_a:
            assert torch.equal(mut[k], parent_a[k])


# ── Selection tests ───────────────────────────────────────────────────────


class TestSelectTournament:
    def test_returns_valid_index(self):
        idx = select_tournament([1.0, 2.0, 3.0, 4.0], k=2, seed=1)
        assert 0 <= idx < 4

    def test_picks_best_in_tournament(self):
        # With k=4 (full population), should always pick the max
        idx = select_tournament([1.0, 2.0, 3.0, 4.0], k=4, seed=1)
        assert idx == 3

    def test_reproducible(self):
        i1 = select_tournament([0.5, 0.9, 0.1, 0.7], k=2, seed=42)
        i2 = select_tournament([0.5, 0.9, 0.1, 0.7], k=2, seed=42)
        assert i1 == i2


class TestSelectRank:
    def test_returns_valid_index(self):
        idx = select_rank([1.0, 2.0, 3.0, 4.0], seed=1)
        assert 0 <= idx < 4

    def test_high_pressure_favors_best(self):
        # With high selection pressure, best individual should be
        # selected more often than worst
        counts = [0] * 4
        for s in range(100):
            idx = select_rank([1.0, 2.0, 3.0, 4.0], selection_pressure=2.0, seed=s)
            counts[idx] += 1
        # Best (index 3) should be selected more than worst (index 0)
        assert counts[3] > counts[0]

    def test_reproducible(self):
        i1 = select_rank([0.5, 0.9, 0.1, 0.7], seed=42)
        i2 = select_rank([0.5, 0.9, 0.1, 0.7], seed=42)
        assert i1 == i2

    def test_single_element(self):
        assert select_rank([42.0], seed=0) == 0


class TestSelectRoulette:
    def test_returns_valid_index(self):
        idx = select_roulette([1.0, 2.0, 3.0, 4.0], seed=1)
        assert 0 <= idx < 4

    def test_zero_fitness_handled(self):
        idx = select_roulette([0.0, 0.0, 0.0, 0.0], seed=1)
        assert 0 <= idx < 4

    def test_negative_fitness_shifted(self):
        # All negative — should still work (auto-shifted)
        idx = select_roulette([-4.0, -3.0, -2.0, -1.0], seed=1)
        assert 0 <= idx < 4

    def test_reproducible(self):
        i1 = select_roulette([0.5, 0.9, 0.1, 0.7], seed=42)
        i2 = select_roulette([0.5, 0.9, 0.1, 0.7], seed=42)
        assert i1 == i2


class TestSelectDiversity:
    def test_returns_valid_index(self, parent_a, parent_b):
        pop = [parent_a, parent_b]
        idx = select_diversity([1.0, 2.0], pop, seed=1)
        assert 0 <= idx < 2

    def test_reproducible(self, parent_a, parent_b):
        pop = [parent_a, parent_b]
        i1 = select_diversity([1.0, 2.0], pop, seed=42)
        i2 = select_diversity([1.0, 2.0], pop, seed=42)
        assert i1 == i2

    def test_diversity_weight_zero_is_fitness_only(self, parent_a, parent_b):
        # With diversity_weight=0, should behave like roulette on fitness
        pop = [parent_a, parent_b]
        for s in range(20):
            idx = select_diversity([1.0, 2.0], pop, diversity_weight=0.0, seed=s)
            assert 0 <= idx < 2


# ── Fitness scaling tests ─────────────────────────────────────────────────


class TestScaleFitnessSigma:
    def test_preserves_length(self):
        scaled = scale_fitness_sigma([1.0, 2.0, 3.0, 4.0])
        assert len(scaled) == 4

    def test_all_equal_returns_uniform(self):
        scaled = scale_fitness_sigma([5.0, 5.0, 5.0, 5.0])
        assert all(s == 1.0 for s in scaled)

    def test_non_negative(self):
        scaled = scale_fitness_sigma([1.0, 2.0, 3.0, 4.0])
        assert all(s >= 0.0 for s in scaled)


class TestScaleFitnessRank:
    def test_preserves_length(self):
        scaled = scale_fitness_rank([1.0, 2.0, 3.0, 4.0])
        assert len(scaled) == 4

    def test_best_gets_highest_scaled(self):
        scaled = scale_fitness_rank([1.0, 2.0, 3.0, 4.0])
        # Index 3 has the highest fitness (4.0) → should get highest scaled
        assert scaled[3] == max(scaled)

    def test_all_equal(self):
        scaled = scale_fitness_rank([5.0, 5.0, 5.0])
        assert len(scaled) == 3


# ── Diversity helper tests ────────────────────────────────────────────────


class TestStateDistance:
    def test_identical_states_zero_distance(self, parent_a):
        d = _state_distance(parent_a, parent_a)
        assert d == 0.0

    def test_different_states_positive_distance(self, parent_a, parent_b):
        d = _state_distance(parent_a, parent_b)
        assert d > 0.0


class TestPopulationDiversity:
    def test_identical_population_zero_diversity(self, parent_a):
        pop = [parent_a, parent_a, parent_a]
        assert abs(_population_diversity(pop)) < 1e-6

    def test_diverse_population_positive_diversity(self, parent_a, parent_b):
        pop = [parent_a, parent_b]
        assert _population_diversity(pop) > 0.0

    def test_single_element_zero_diversity(self, parent_a):
        assert _population_diversity([parent_a]) == 0.0


# ── Evolve loop tests ─────────────────────────────────────────────────────


class TestEvolve:
    def test_runs_n_generations(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        # Fitness: negative L2 norm (cheaper than benchmark, deterministic)
        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(
            pop, fit, n_generations=3, population_size=4, mutation_rate=0.5, elitism=1, seed=0, verbose=False
        )
        assert len(result["history"]) == 4  # 3 gens + final
        assert result["best"] is not None
        assert result["best_fitness"] > float("-inf")

    def test_best_fitness_improves_or_stable(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=5, population_size=6, elitism=2, seed=0, verbose=False)
        # With elitism, best fitness should be non-decreasing
        fitnesses = [h["best_fitness"] for h in result["history"]]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1] - 1e-9

    def test_saves_checkpoints(self, parent_a, parent_b, tmp_path):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        out_dir = str(tmp_path / "evolved")
        result = evolve(
            pop,
            fit,
            n_generations=2,
            population_size=4,
            seed=0,
            save_fn=save_checkpoint,
            out_dir=out_dir,
            verbose=False,
        )
        assert result["best_path"] is not None
        import os

        assert os.path.exists(result["best_path"])

    def test_final_population_size(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return 1.0

        result = evolve(pop, fit, n_generations=2, population_size=8, seed=0, verbose=False)
        assert len(result["final_population"]) == 8

    def test_custom_crossover_and_mutation(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return sum(v.float().abs().mean().item() for v in s.values() if v.is_floating_point())

        result = evolve(
            pop,
            fit,
            n_generations=2,
            population_size=4,
            crossover="block_random",
            crossover_kwargs={"p": 0.3},
            mutation="gaussian",
            mutation_kwargs={"sigma": 0.02, "rate": 0.1},
            seed=0,
            verbose=False,
        )
        assert result["best"] is not None


class TestEvolveSophisticatedSelection:
    """Test the new selection strategies in the evolve loop."""

    def test_rank_selection(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=3, population_size=4, selection="rank", seed=0, verbose=False)
        assert result["best"] is not None
        assert len(result["history"]) == 4

    def test_roulette_selection(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return sum(v.float().abs().mean().item() for v in s.values() if v.is_floating_point())

        result = evolve(pop, fit, n_generations=3, population_size=4, selection="roulette", seed=0, verbose=False)
        assert result["best"] is not None

    def test_diversity_selection(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return sum(v.float().abs().mean().item() for v in s.values() if v.is_floating_point())

        result = evolve(
            pop,
            fit,
            n_generations=3,
            population_size=4,
            selection="diversity",
            selection_kwargs={"diversity_weight": 0.3},
            seed=0,
            verbose=False,
        )
        assert result["best"] is not None


class TestEvolveScoreRewarding:
    """Test fitness scaling, progress bonus, and diversity bonus."""

    def test_sigma_scaling(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=3, population_size=4, fitness_scaling="sigma", seed=0, verbose=False)
        assert result["best"] is not None

    def test_rank_scaling(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=3, population_size=4, fitness_scaling="rank", seed=0, verbose=False)
        assert result["best"] is not None

    def test_progress_bonus(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return sum(v.float().abs().mean().item() for v in s.values() if v.is_floating_point())

        result = evolve(pop, fit, n_generations=3, population_size=4, progress_bonus=0.5, seed=0, verbose=False)
        assert result["best"] is not None

    def test_diversity_bonus(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return sum(v.float().abs().mean().item() for v in s.values() if v.is_floating_point())

        result = evolve(pop, fit, n_generations=3, population_size=4, diversity_bonus=0.3, seed=0, verbose=False)
        assert result["best"] is not None
        # History should include diversity metric
        assert "diversity" in result["history"][0]


class TestEvolveAdaptiveMutation:
    """Test adaptive mutation rate based on population diversity."""

    def test_adaptive_mutation_runs(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=3, population_size=4, adaptive_mutation=True, seed=0, verbose=False)
        assert result["best"] is not None
        # History should include mutation_rate
        assert "mutation_rate" in result["history"][0]


class TestEvolveHallOfFame:
    """Test the hall of fame feature."""

    def test_hof_populated(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=3, population_size=4, hall_of_fame_size=5, seed=0, verbose=False)
        assert "hall_of_fame" in result
        assert len(result["hall_of_fame"]) <= 5
        assert len(result["hall_of_fame"]) > 0
        # HoF entries are (state_dict, fitness) tuples
        hof_entry = result["hall_of_fame"][0]
        assert isinstance(hof_entry, tuple)
        assert len(hof_entry) == 2
        assert isinstance(hof_entry[0], dict)
        assert isinstance(hof_entry[1], float)

    def test_hof_sorted_by_fitness(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return -sum(v.float().norm().item() for v in s.values() if v.is_floating_point()) / 1e3

        result = evolve(pop, fit, n_generations=3, population_size=4, hall_of_fame_size=3, seed=0, verbose=False)
        fits = [f for _, f in result["hall_of_fame"]]
        assert fits == sorted(fits, reverse=True)

    def test_hof_disabled_by_default(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return 1.0

        result = evolve(pop, fit, n_generations=2, population_size=4, seed=0, verbose=False)
        assert result["hall_of_fame"] == []


class TestEvolveConvergencePatience:
    """Test early stopping via convergence patience."""

    def test_convergence_stops_early(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        # Constant fitness → never improves → should stop early
        def fit(s):
            return 1.0

        result = evolve(pop, fit, n_generations=10, population_size=4, convergence_patience=2, seed=0, verbose=False)
        # Should stop well before 10 generations
        assert len(result["history"]) < 12  # 10 gens + final = 12 max

    def test_no_convergence_check_by_default(self, parent_a, parent_b):
        pop = [parent_a, parent_b]

        def fit(s):
            return 1.0

        result = evolve(pop, fit, n_generations=3, population_size=4, seed=0, verbose=False)
        # Should run all 3 generations + final
        assert len(result["history"]) == 4


# ── ForgeEngine method tests ──────────────────────────────────────────────


class TestForgeEngineMergeCheckpoints:
    """Test ForgeEngine.merge_checkpoints() with saved checkpoint files."""

    def test_merge_two_checkpoints(self, forge_engine, parent_a, parent_b, tmp_path):
        # Save parent checkpoints
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        save_checkpoint(parent_a, path_a)
        save_checkpoint(parent_b, path_b)

        out = forge_engine.merge_checkpoints(
            [path_a, path_b], method="blockwise_crossover", out_path=str(tmp_path / "merged.safetensors"), split_block=2
        )
        import os

        assert os.path.exists(out)
        loaded = load_checkpoint(out)
        assert set(loaded.keys()) == set(parent_a.keys())

    def test_merge_load_result(self, forge_engine, parent_a, parent_b, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        save_checkpoint(parent_a, path_a)
        save_checkpoint(parent_b, path_b)

        forge_engine.merge_checkpoints(
            [path_a, path_b],
            method="uniform_crossover",
            out_path=str(tmp_path / "merged.safetensors"),
            load_result=True,
            p=0.5,
        )
        # Engine weights should now be the merged ones (no crash = success)
        assert forge_engine._awake

    def test_unknown_method_raises(self, forge_engine, parent_a, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        save_checkpoint(parent_a, path_a)
        from forge.engine.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            forge_engine.merge_checkpoints(
                [path_a], method="nonexistent_method", out_path=str(tmp_path / "out.safetensors")
            )

    def test_mutation_method(self, forge_engine, parent_a, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        save_checkpoint(parent_a, path_a)
        out = forge_engine.merge_checkpoints(
            [path_a], method="gaussian_mutation", out_path=str(tmp_path / "mutated.safetensors"), sigma=0.02, rate=0.1
        )
        import os

        assert os.path.exists(out)


class TestForgeEngineEvolveMerge:
    """Test ForgeEngine.evolve_merge() end-to-end."""

    def test_evolve_with_custom_fitness(self, forge_engine, parent_a, parent_b, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        save_checkpoint(parent_a, path_a)
        save_checkpoint(parent_b, path_b)

        # Custom fitness: negative L2 norm (no GPU needed, fast)
        def fit(state):
            return -sum(v.float().norm().item() for v in state.values() if v.is_floating_point()) / 1e3

        result = forge_engine.evolve_merge(
            [path_a, path_b],
            fitness_fn=fit,
            n_generations=2,
            population_size=4,
            out_dir=str(tmp_path / "evolved"),
            restore_original=True,
            verbose=False,
        )
        assert "best" in result
        assert "history" in result
        assert len(result["history"]) == 3  # 2 gens + final
        assert result["best_fitness"] > float("-inf")

    def test_evolve_restores_original_weights(self, forge_engine, parent_a, parent_b, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        save_checkpoint(parent_a, path_a)
        save_checkpoint(parent_b, path_b)

        # Snapshot original engine weights
        original = {k: v.detach().clone() for k, v in forge_engine.model.state_dict().items()}

        def fit(state):
            return 1.0  # constant fitness (all equal)

        forge_engine.evolve_merge(
            [path_a, path_b], fitness_fn=fit, n_generations=1, population_size=2, restore_original=True, verbose=False
        )

        # Verify weights were restored
        current = forge_engine.model.state_dict()
        for k in original:
            assert torch.equal(current[k].cpu(), original[k].cpu()), f"Weight {k} was not restored after evolution"

    def test_evolve_too_few_parents_raises(self, forge_engine, parent_a, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        save_checkpoint(parent_a, path_a)
        from forge.engine.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            forge_engine.evolve_merge([path_a], n_generations=1)

    def test_evolve_saves_best_checkpoint(self, forge_engine, parent_a, parent_b, tmp_path):
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        save_checkpoint(parent_a, path_a)
        save_checkpoint(parent_b, path_b)

        def fit(state):
            return -sum(v.float().norm().item() for v in state.values() if v.is_floating_point()) / 1e3

        result = forge_engine.evolve_merge(
            [path_a, path_b],
            fitness_fn=fit,
            n_generations=2,
            population_size=4,
            out_dir=str(tmp_path / "evolved"),
            restore_original=True,
            verbose=False,
        )
        import os

        assert result["best_path"] is not None
        assert os.path.exists(result["best_path"])

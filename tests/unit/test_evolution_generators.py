"""Tests for forge.evolution.generators — BatchedGenerator, TemplateGenerator, GeneratorPopulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from forge.evolution.generators import (
    BatchedGenerator,
    GeneratorConfig,
    GeneratorPopulation,
    TemplateGenerator,
)


@pytest.fixture
def gen_cfg():
    """Small GeneratorConfig for fast CPU tests."""
    return GeneratorConfig(
        n_generators=10,
        noise_dim=4,
        context_dim=8,
        hidden_dim=16,
        output_dim=4,
    )


@pytest.fixture
def population(gen_cfg):
    """GeneratorPopulation with small config."""
    return GeneratorPopulation(gen_cfg, device=torch.device("cpu"))


class TestBatchedGeneratorInit:
    """BatchedGenerator construction and parameter layout."""

    def test_has_all_weight_params(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        names = dict(gen.named_parameters())
        # 4-layer network: W0,b0,W1,b1,W2,b2,W3,b3 + LayerNorm params
        for key in ("W0", "b0", "W1", "b1", "W2", "b2", "W3", "b3"):
            assert key in names, f"missing parameter {key}"

    def test_has_fitness_ema_buffer(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        buffers = dict(gen.named_buffers())
        assert "fitness_ema" in buffers
        assert buffers["fitness_ema"].shape == (gen_cfg.n_generators,)

    def test_weight_shapes(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        n = gen_cfg.n_generators
        in_dim = gen_cfg.noise_dim + gen_cfg.context_dim
        h = gen_cfg.hidden_dim
        out = gen_cfg.output_dim
        assert gen.W0.shape == (n, in_dim, h)
        assert gen.b0.shape == (n, h)
        assert gen.W1.shape == (n, h, h)
        assert gen.b1.shape == (n, h)
        # 4-layer: W2 is hidden→hidden, W3 is hidden→out
        assert gen.W2.shape == (n, h, h)
        assert gen.b2.shape == (n, h)
        assert gen.W3.shape == (n, h, out)
        assert gen.b3.shape == (n, out)
        # LayerNorm params
        assert gen.ln_gamma.shape == (n, h)
        assert gen.ln_beta.shape == (n, h)


class TestBatchedGeneratorForward:
    """Batched forward pass output shape and range."""

    def test_forward_shape_and_range(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        noise = torch.randn(gen_cfg.n_generators, gen_cfg.noise_dim)
        context = torch.randn(gen_cfg.context_dim)
        out = gen(noise, context)
        assert out.shape == (gen_cfg.n_generators, gen_cfg.output_dim)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_forward_batch_consistency(self, gen_cfg):
        """forward with N=5 returns 5 rows."""
        cfg = GeneratorConfig(
            n_generators=5,
            noise_dim=4,
            context_dim=8,
            hidden_dim=16,
            output_dim=4,
        )
        gen = BatchedGenerator(cfg, device=torch.device("cpu"))
        noise = torch.randn(5, cfg.noise_dim)
        context = torch.randn(cfg.context_dim)
        out = gen(noise, context)
        assert out.shape[0] == 5
        assert out.shape == (5, cfg.output_dim)


class TestBatchedGeneratorForwardSingle:
    """Single-generator forward pass."""

    def test_forward_single_shape_and_range(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        noise = torch.randn(gen_cfg.noise_dim)
        context = torch.randn(gen_cfg.context_dim)
        out = gen.forward_single(noise, context, gen_idx=0)
        assert out.shape == (gen_cfg.output_dim,)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


class TestBatchedGeneratorMutate:
    """mutate_generator copies parent + adds noise."""

    def test_mutate_changes_weights(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        parent_w0 = gen.W0[0].clone()
        gen.mutate_generator(idx=1, parent_idx=0, rate=0.5)
        assert not torch.allclose(gen.W0[1], parent_w0)

    def test_mutate_none_rate_uses_default(self, gen_cfg):
        """rate=None should use cfg.mutation_rate (default behavior)."""
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        parent_w0 = gen.W0[0].clone()
        gen.mutate_generator(idx=2, parent_idx=0, rate=None)
        # With default mutation_rate, weights should differ (noise added)
        assert not torch.allclose(gen.W0[2], parent_w0)

    def test_mutate_resets_fitness_and_age(self, gen_cfg):
        """mutate_generator should reset fitness_ema to 0.5*parent and age to 0."""
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        # Set non-zero fitness and age on parent
        gen.fitness_ema[0] = 0.8
        gen.age[0] = 5
        gen.mutate_generator(idx=3, parent_idx=0, rate=0.1)
        assert abs(gen.fitness_ema[3].item() - 0.4) < 1e-6  # 0.5 * 0.8
        assert gen.age[3].item() == 0


class TestBatchedGeneratorNParams:
    """n_params returns a positive int."""

    def test_n_params_positive_int(self, gen_cfg):
        gen = BatchedGenerator(gen_cfg, device=torch.device("cpu"))
        n = gen.n_params()
        assert isinstance(n, int)
        assert n > 0


class TestTemplateGeneratorInit:
    """TemplateGenerator construction."""

    def test_init_with_choices(self):
        choices = {"block_size": [16, 32, 64], "lr": [1e-3, 1e-4]}
        tg = TemplateGenerator(choices)
        assert tg.choices == choices
        assert set(tg._keys) == set(choices.keys())


class TestTemplateGeneratorSample:
    """sample returns exhaustive-then-random configs."""

    def test_first_samples_exhaustive(self):
        choices = {"a": [1, 2], "b": [10, 20]}
        tg = TemplateGenerator(choices)
        configs = tg.sample(4)  # 2x2 = 4 exhaustive combos
        # First 4 should cover all combinations (odometer order)
        combos = {(c["a"], c["b"]) for c in configs}
        assert combos == {(1, 10), (1, 20), (2, 10), (2, 20)}

    def test_sample_returns_list_of_dicts(self):
        tg = TemplateGenerator({"x": [1, 2, 3]})
        configs = tg.sample(2)
        assert isinstance(configs, list)
        assert all(isinstance(c, dict) for c in configs)

    def test_sample_random_after_exhausted(self):
        choices = {"a": [1, 2]}
        tg = TemplateGenerator(choices)
        tg.sample(2)  # exhausts 2 combos
        extra = tg.sample(3)
        assert len(extra) == 3
        for c in extra:
            assert c["a"] in [1, 2]

    def test_sample_count_at_most_n(self):
        tg = TemplateGenerator({"a": [1, 2, 3], "b": [4, 5]})
        configs = tg.sample(2)
        assert len(configs) == 2


class TestTemplateGeneratorReset:
    """reset restarts enumeration."""

    def test_reset_restarts_enumeration(self):
        choices = {"a": [1, 2], "b": [10, 20]}
        tg = TemplateGenerator(choices)
        first = tg.sample(4)  # exhaust
        tg.reset()
        second = tg.sample(4)
        assert first == second


class TestGeneratorPopulationGenerate:
    """generate returns candidate dicts."""

    def test_generate_keys(self, population, gen_cfg):
        candidates = population.generate(n_per_gen=1)
        assert len(candidates) >= gen_cfg.n_generators
        for c in candidates:
            assert "params" in c
            assert "generator_idx" in c
            assert "noise" in c

    def test_generate_with_template(self, gen_cfg):
        template = TemplateGenerator({"block_size": [16, 32]})
        pop = GeneratorPopulation(gen_cfg, template=template, device=torch.device("cpu"))
        candidates = pop.generate(n_per_gen=1)
        template_candidates = [c for c in candidates if c["generator_idx"] == -1]
        assert len(template_candidates) > 0
        for c in template_candidates:
            assert "template_config" in c


class TestGeneratorPopulationEvolve:
    """evolve kills bottom performers and clones top."""

    def test_evolve_returns_mutated_indices(self, population, gen_cfg):
        candidates = population.generate(n_per_gen=1)
        scores = [float(i) for i in range(len(candidates))]
        indices = [c["generator_idx"] for c in candidates]
        mutated = population.evolve(scores, indices)
        assert isinstance(mutated, list)
        assert all(isinstance(m, int) for m in mutated)

    def test_evolve_kills_bottom_performers(self, population, gen_cfg):
        candidates = population.generate(n_per_gen=1)
        scores = [float(i) for i in range(len(candidates))]
        indices = [c["generator_idx"] for c in candidates]
        population.batched_gen.fitness_ema.clone()
        mutated = population.evolve(scores, indices)
        # n_kill = n_generators // 5 = 2 for n=10
        assert len(mutated) >= gen_cfg.n_generators // 5

    def test_evolve_changes_fitness_ema(self, population, gen_cfg):
        candidates = population.generate(n_per_gen=1)
        scores = [1.0] * len(candidates)
        indices = [c["generator_idx"] for c in candidates]
        before = population.batched_gen.fitness_ema.clone()
        population.evolve(scores, indices)
        after = population.batched_gen.fitness_ema
        # At least some generators should have updated fitness
        assert not torch.allclose(before, after)


class TestGeneratorPopulationUpdateContext:
    """update_context pads/truncates to context_dim."""

    def test_update_context_pads_short(self, population, gen_cfg):
        short = torch.ones(2)
        population.update_context(short)
        assert population.context.shape == (gen_cfg.context_dim,)
        assert torch.all(population.context[:2] == 1.0)
        assert torch.all(population.context[2:] == 0.0)

    def test_update_context_truncates_long(self, population, gen_cfg):
        long_vec = torch.arange(20.0)
        population.update_context(long_vec)
        assert population.context.shape == (gen_cfg.context_dim,)
        assert torch.allclose(population.context, long_vec[: gen_cfg.context_dim])

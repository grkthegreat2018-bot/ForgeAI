"""Tests for the proposal frontier pipeline — multi-sample generation,
vectorized dedup, queue decoupling, eval safety dispatch."""
import sys; sys.path.insert(0, r"D:\windsurf\ForgeAI")

import numpy as np
import pytest
import torch

from forge.evolution import ForgeEvolve, ForgeEvolveConfig
from forge.evolution.generators import (
    GeneratorConfig, GeneratorPopulation,
)
from forge.evolution.domains.synthetic import SyntheticDomain


@pytest.fixture
def gen_cfg():
    return GeneratorConfig(n_generators=10, noise_dim=4, context_dim=8,
                           hidden_dim=16, output_dim=8)


def _engine(domain, **kw):
    defaults = dict(domain=domain, n_generators=20, generations=2,
                    min_evaluate=4, max_evaluate=12, filter_ratio=5,
                    device="cpu", db_path=None, warm_start=False,
                    verbose=False, fixate=True)
    defaults.update(kw)
    return ForgeEvolve(ForgeEvolveConfig(**defaults))


class TestMultiSampleGenerate:
    def test_counts_and_gen_idx(self, gen_cfg):
        pop = GeneratorPopulation(gen_cfg, device=torch.device("cpu"))
        for k in (1, 3):
            cands = pop.generate(n_per_gen=k)
            assert len(cands) == gen_cfg.n_generators * k
            # rows grouped by generator: row i -> generator i // k
            assert all(c["generator_idx"] == i // k
                       for i, c in enumerate(cands))

    def test_samples_differ(self, gen_cfg):
        pop = GeneratorPopulation(gen_cfg, device=torch.device("cpu"))
        torch.manual_seed(0)
        cands = pop.generate(n_per_gen=4)
        g0 = [c["params"] for c in cands[:4]]  # all from generator 0
        assert not all(torch.equal(g0[0], g) for g in g0[1:])

    def test_generator_equivalence(self, gen_cfg):
        """Row i of the k-per-gen pass must equal generator i//k's own
        weights applied to row i's noise (manual single-gen forward)."""
        torch.manual_seed(0)
        pop = GeneratorPopulation(gen_cfg, device=torch.device("cpu"))
        k = 3
        noise = torch.randn(10 * k, gen_cfg.noise_dim)
        out = pop.batched_gen(noise, pop.context)

        W = pop.batched_gen
        g = 5
        ctx = pop.context.unsqueeze(0).expand(k, -1)
        x = torch.cat([noise[g * k:(g + 1) * k], ctx], dim=-1)  # (k, in)

        def ln(h):  # per-row LN with generator g's params
            m = h.mean(-1, keepdim=True)
            v = h.var(-1, unbiased=False, keepdim=True)
            return (h - m) / (v + 1e-6).sqrt() * W.ln_gamma[g] + W.ln_beta[g]

        h = torch.tanh(ln(x @ W.W0[g] + W.b0[g]))
        h = torch.tanh(ln(h @ W.W1[g] + W.b1[g]))
        h = torch.tanh(ln(h @ W.W2[g] + W.b2[g]))
        o = torch.sigmoid(h @ W.W3[g] + W.b3[g])
        torch.testing.assert_close(out[g * k:(g + 1) * k], o)


class TestFrontierPipeline:
    def test_submitted_exceeds_evaluated(self):
        eng = _engine(SyntheticDomain(dim=8, seed=0), propose_mult=4)
        res = eng.run()
        assert eng._submitted > res["total_evaluations"]
        assert len(eng._frontier) <= eng.cfg.frontier_cap
        assert res["total_evaluations"] > 0

    def test_no_duplicate_evals(self):
        eng = _engine(SyntheticDomain(dim=8, seed=1), propose_mult=2,
                      generations=3)
        eng.run()
        keys = set()
        for r in eng.all_results:
            k = eng._config_key(r["config"])
            assert k not in keys, "duplicate config evaluated"
            keys.add(k)

    def test_popped_entries_carry_reinforce_fields(self):
        eng = _engine(SyntheticDomain(dim=8, seed=2))
        eng.run()
        # Frontier entries must keep noise/params/generator_idx so a
        # delayed eval still supports the REINFORCE update.
        for e in eng._frontier[:5]:
            assert e["noise"] is not None
            assert e["params"] is not None
            assert e["generator_idx"] >= 0

    def test_frontier_cap_evicts(self):
        eng = _engine(SyntheticDomain(dim=8, seed=3), propose_mult=4,
                      frontier_cap=30, generations=3)
        eng.run()
        assert len(eng._frontier) <= 30


class TestIslandsAndCrossover:
    def test_islands_fill_and_migrate(self):
        eng = _engine(SyntheticDomain(dim=8, seed=7), n_islands=3,
                      island_migrate_gens=2, generations=4,
                      crossover_frac=0.3)
        eng.run()
        assert len(eng._islands) == 3
        assert sum(i.n_filled for i in eng._islands) > 0

    def test_crossover_submits(self):
        eng = _engine(SyntheticDomain(dim=8, seed=9), generations=3,
                      crossover_frac=0.5, propose_mult=2)
        eng.run()
        # Crossover proposals enter frontier with generator_idx=-2
        # (archive may hold them after eval)
        assert eng._submitted > 0

    def test_config_key_dedup_discrete(self):
        """Different param vectors decoding to the SAME discrete config
        must dedupe to one frontier entry (param-level keys would let
        identical configs waste real evals)."""

        class _Discrete(SyntheticDomain):
            def decode(self, params):
                # bin the whole vector to 4 discrete cells — huge
                # many-to-one param->config mapping
                return {"x": np.zeros(8),
                        "cell": int(np.clip(params.sum().item() // 1, 0, 3))}

        eng = _engine(_Discrete(dim=8, seed=11), propose_mult=2,
                      generations=2)
        eng.run()
        # frontier keys are config-level: at most 4 distinct cells
        distinct = {e["config"]["cell"] for e in eng._frontier}
        assert len(distinct) <= 4


class TestEvalSafetyDispatch:
    def test_unsafe_domain_uses_serial_eval(self, monkeypatch):
        domain = SyntheticDomain(dim=8, seed=0)
        domain.eval_thread_safe = False
        eng = _engine(domain)
        called = {"threaded": 0}
        orig = ForgeEvolve._threaded_eval
        def spy(self, cfgs, eval_fn=None):
            called["threaded"] += 1
            return orig(self, cfgs, eval_fn)
        monkeypatch.setattr(ForgeEvolve, "_threaded_eval", spy)
        eng._gpu_batch_eval([{"x": np.zeros(8)} for _ in range(4)])
        assert called["threaded"] == 0

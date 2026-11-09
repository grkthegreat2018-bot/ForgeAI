"""Tests for LoRA-ES zeroth-order training (R71/B3)."""
import math

import torch
import torch.nn as nn

from forge.training.lora_es import LoRAES, LowRankBypass, attach_lora


def _quad_loss_factory(params, target):
    def loss_fn():
        return float(sum((p.float() - t).pow(2).sum()
                         for p, t in zip(params, target)))
    return loss_fn


class TestZODirection:
    def test_estimate_aligns_with_true_gradient(self):
        # ZO theory: expected cos ≈ sqrt(n_dirs/dim). d=2048:
        # 32 dirs -> ~0.12; 256 dirs -> ~0.35. Check both the floor and
        # that the estimate scales with n_dirs (proves it's not noise).
        torch.manual_seed(0)
        params = [torch.randn(64, 32)]
        target = [torch.randn(64, 32)]
        true_g = 2 * (params[0] - target[0])

        def cos_after(n_dirs):
            es = LoRAES(params, eps=1e-3, lr=0.0, n_dirs=n_dirs, seed=1)
            es.step(_quad_loss_factory(params, target))
            return float(torch.nn.functional.cosine_similarity(
                es._vel[0].flatten(), true_g.flatten(), dim=0))

        c32, c256 = cos_after(32), cos_after(256)
        assert c32 > 0.03          # >> random (0) but inherently low-dim
        assert c256 > c32          # estimator quality scales with dirs
        assert c256 > 0.2

    def test_params_restored_after_perturbation(self):
        params = [torch.randn(16, 16)]
        before = params[0].clone()
        es = LoRAES(params, eps=1e-2, lr=0.0, n_dirs=4, seed=0)
        es.step(_quad_loss_factory(params, [torch.zeros(16, 16)]))
        # lr=0 → params must be bit-identical to before (no residual eps*u)
        torch.testing.assert_close(params[0], before)

    def test_block_mode_perturbs_subset(self):
        ps = [torch.randn(8, 8) for _ in range(6)]
        es = LoRAES(ps, eps=1.0, lr=0.0, n_dirs=1, block_size=2,
                    perturb_mode="block", seed=3)
        touched = set()

        def spy():
            # any param at distance ~eps from origin counts as perturbed;
            # easier: count params whose |.| distribution changed — instead
            # verify via internal call: patch _perturb_subset
            return 1.0

        orig = es._perturb_subset
        def wrapped(idx, us, sign):
            if sign > 0:
                touched.update(int(i) for i in idx)
            return orig(idx, us, sign)
        es._perturb_subset = wrapped
        es.step(spy)
        assert len(touched) <= 2

    def test_rademacher_directions(self):
        p = [torch.zeros(10)]
        es = LoRAES(p, distribution="rademacher", n_dirs=1, lr=0.0)
        u = es._sample_dir(p)[0]
        assert set(u.unique().tolist()) == {-1.0, 1.0}


class TestConvergence:
    def test_loss_decreases_on_regression(self):
        torch.manual_seed(0)
        lin = nn.Linear(8, 4, bias=False)
        x, y = torch.randn(64, 8), torch.randn(64, 4)
        es = LoRAES(list(lin.parameters()), eps=5e-3, lr=1e-2,
                    n_dirs=8, momentum=0.0, seed=0)

        def loss():
            return float(((lin(x) - y) ** 2).mean())

        l0 = loss()
        for _ in range(40):
            es.step(loss)
        assert loss() < l0

    def test_runs_under_inference_mode_no_grads(self):
        p = [torch.randn(8, 8)]
        es = LoRAES(p, n_dirs=2, lr=1e-2)
        with torch.inference_mode():
            # inference_mode tensors can't accumulate grad — if the impl
            # needed autograd this would raise
            es.step(lambda: float(p[0].pow(2).sum()))


class TestLowRankBypass:
    def test_delta_shape_and_scale(self):
        lin = nn.Linear(8, 4, bias=False)
        wrap = LowRankBypass(lin, rank=2, scale=2.0)
        assert wrap.A.shape == (2, 8) and wrap.B.shape == (4, 2)
        wrap.B.data.fill_(1.0)
        x = torch.randn(3, 8)
        delta = (x @ wrap.A.t() @ wrap.B.t()) * 2.0
        torch.testing.assert_close(wrap(x) - lin(x), delta)

    def test_attach_lora_collects_trainables(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        train = attach_lora(model, rank=2)
        assert len(train) == 4  # A,B for each of 2 linears
        assert isinstance(model[0], LowRankBypass)
        assert isinstance(model[2], LowRankBypass)
        x = torch.randn(5, 8)
        assert model(x).shape == (5, 4)

    def test_name_filter(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        train = attach_lora(model, rank=2, name_filter=lambda n: n == "1")
        assert len(train) == 2
        assert not isinstance(model[0], LowRankBypass)
        assert isinstance(model[1], LowRankBypass)


class TestFisherMode:
    def test_fisher_scale_shrinks_step(self):
        p = [torch.zeros(64, 64)]
        scale = {"p0": torch.zeros(64, 64)}  # scale 0 → no perturbation
        es = LoRAES(p, names=["p0"], eps=0.5, lr=0.0, n_dirs=1,
                    perturb_mode="fisher", fisher_scale=scale)
        es.step(lambda: 1.0)
        torch.testing.assert_close(p[0], torch.zeros(64, 64))

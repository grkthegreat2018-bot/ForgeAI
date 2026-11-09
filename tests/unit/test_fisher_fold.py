"""Tests for FisherFold — merge_fisher formula + fisher_diag (R71/B4)."""
import torch
import torch.nn as nn

from research.merge_models import fisher_diag, merge_fisher


class TestMergeFisher:
    def _sds(self):
        base = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([5.0, 6.0]),
                "c": torch.tensor([0.0, 0.0])}
        new = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([1.0, 1.0]),
               "c": torch.tensor([9.0, 9.0])}
        return base, new

    def test_gate_keeps_new_where_f_new_high(self):
        base, new = self._sds()
        fb = {"a": torch.zeros(2)}
        fn = {"a": torch.ones(2) * 10}
        out = merge_fisher(base, new, fb, fn)
        torch.testing.assert_close(out["a"], new["a"])

    def test_gate_reverts_where_f_base_high(self):
        base, new = self._sds()
        fb = {"b": torch.ones(2) * 10}
        fn = {"b": torch.zeros(2)}
        out = merge_fisher(base, new, fb, fn)
        torch.testing.assert_close(out["b"], base["b"])

    def test_gate_defaults_to_base_when_no_fisher(self):
        base, new = self._sds()
        out = merge_fisher(base, new, {}, {})  # unused drift -> base
        torch.testing.assert_close(out["c"], base["c"])
        torch.testing.assert_close(out["a"], base["a"])

    def test_interp_weights_both(self):
        base, new = self._sds()
        fb = {"a": torch.ones(2)}
        fn = {"a": torch.ones(2)}
        out = merge_fisher(base, new, fb, fn, mode="interp")
        torch.testing.assert_close(out["a"], (base["a"] + new["a"]) / 2)

    def test_interp_falls_back_to_base(self):
        base, new = self._sds()
        out = merge_fisher(base, new, {}, {}, mode="interp")
        torch.testing.assert_close(out["a"], base["a"])

    def test_shape_mismatch_keeps_base(self):
        base = {"x": torch.ones(4)}
        new = {"x": torch.ones(8)}
        out = merge_fisher(base, new, {}, {})
        assert out["x"].shape == (4,)


class TestFisherDiag:
    def test_computes_grad_squared(self):
        torch.manual_seed(0)
        lin = nn.Linear(4, 1, bias=False)
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)

        def loss_fn(m, b):
            return ((m(b[0]) - b[1]) ** 2).mean()

        f = fisher_diag(lin, loss_fn, iter([(x, y)] * 4), 4)
        assert "weight" in f
        assert f["weight"].shape == lin.weight.shape
        # manual check: fisher == E[grad^2]
        lin.zero_grad()
        loss = loss_fn(lin, (x, y))
        loss.backward()
        torch.testing.assert_close(f["weight"], lin.weight.grad.detach() ** 2)

    def test_requires_grad_filter(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
        m[0].weight.requires_grad_(False)

        def loss_fn(mm, b):
            return mm(b).sum()

        f = fisher_diag(m, loss_fn, iter([torch.ones(4)] * 2), 2)
        assert "0.weight" not in f
        assert "1.weight" in f

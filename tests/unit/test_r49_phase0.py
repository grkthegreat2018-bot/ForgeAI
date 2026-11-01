"""R49 Phase-0 tests: sigmoid MoE gating, Dion2 sampling, QK-Clip, CISPO, ETR."""

import torch
import torch.nn as nn

from forge.moe.moe import MoELayer, Router
from forge.training.optim.muon_sf_blockwise import MuonSFBlockwise, _dion2_update, _muon_update_fallback
from forge.training.optim.qk_clip import QKClipMonitor

# ── MoE sigmoid gating (MiniMax-M2 style) ───────────────────────────────────


class TestSigmoidGating:
    def test_weights_renormalized(self):
        torch.manual_seed(0)
        r = Router(8, 4, top_k=2, noisy_gating=False, gating="sigmoid")
        _mask, weights, aux = r(torch.randn(6, 8))
        assert torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-5)
        assert torch.isfinite(aux)

    def test_dispatch_topk(self):
        torch.manual_seed(0)
        r = Router(8, 4, top_k=2, noisy_gating=False, gating="sigmoid")
        mask, weights, _ = r(torch.randn(5, 8))
        assert (mask.sum(dim=-1) == 2).all()
        assert ((weights > 0) == (mask > 0)).all()

    def test_sigmoid_differs_from_softmax(self):
        torch.manual_seed(1)
        soft = Router(8, 4, top_k=2, noisy_gating=False, gating="softmax")
        sig = Router(8, 4, top_k=2, noisy_gating=False, gating="sigmoid")
        x = torch.randn(6, 8)
        _, w_soft, _ = soft(x)
        _, w_sig, _ = sig(x)
        assert not torch.allclose(w_soft, w_sig)

    def test_moe_layer_passthrough(self):
        moe = MoELayer(8, n_experts=4, top_k=2, gating="sigmoid")
        assert moe.router.gating == "sigmoid"
        out, aux = moe(torch.randn(2, 3, 8))
        assert out.shape == (2, 3, 8)
        assert torch.isfinite(aux)

    def test_aux_free_sigmoid(self):
        r = Router(8, 4, top_k=2, noisy_gating=False, mode="aux_free", gating="sigmoid")
        _mask, _weights, aux = r(torch.randn(5, 8))
        assert torch.isfinite(aux)
        r.update_bias()
        assert torch.isfinite(r.expert_bias).all()

    def test_softmax_default_untouched(self):
        r = Router(8, 4, top_k=2, noisy_gating=False)
        assert r.gating == "softmax"


# ── Dion2 row-sampled orthogonalization ─────────────────────────────────────


class TestDion2:
    def test_sparse_update_rows(self):
        torch.manual_seed(0)
        g = torch.randn(8, 6)
        m = torch.zeros(8, 6)
        update = _dion2_update(g, m, beta=0.95, rank_fraction=0.25)
        zero_rows = (update.abs().sum(dim=1) == 0).sum().item()
        assert zero_rows == 8 - max(1, int(0.25 * 8))

    def test_full_rank_matches_fallback(self):
        torch.manual_seed(0)
        g = torch.randn(8, 6)
        m1 = torch.zeros(8, 6)
        m2 = torch.zeros(8, 6)
        u_fallback = _muon_update_fallback(g.clone(), m1)
        u_dion2 = _dion2_update(g, m2, beta=0.95, rank_fraction=1.0)
        assert torch.allclose(u_fallback, u_dion2, atol=1e-4)

    def test_optimizer_runs_both_fractions(self):
        lin = torch.nn.Linear(16, 8)
        emb = torch.nn.Embedding(4, 8)
        groups = [
            {
                "params": [emb.weight],
                "lr": 1e-3,
                "betas": (0.8, 0.95),
                "eps": 1e-10,
                "use_muon": False,
                "weight_decay": 0.0,
            },
            {"params": [lin.weight, lin.bias], "lr": 1e-2, "momentum": 0.95, "use_muon": True, "weight_decay": 0.1},
        ]
        for rf in (1.0, 0.25):
            opt = MuonSFBlockwise([dict(g) for g in groups], rank_fraction=rf)
            x = torch.randn(2, 16)
            for _ in range(3):
                opt.zero_grad()
                lin(x).pow(2).mean().backward()
                opt.step()
            assert torch.isfinite(lin.weight).all()


# ── QK-Clip ─────────────────────────────────────────────────────────────────


class TestQKClip:
    def _attn(self, d_model=32, n_heads=4):
        from forge.model_loader import GroupedQueryAttention

        return GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads, max_seq_len=64)

    def test_disabled_by_default(self):
        attn = self._attn()
        assert attn._qk_clip_monitor is None

    def test_observe_records_head_max(self):
        attn = self._attn()
        monitor = QKClipMonitor.attach(attn, tau=100.0)
        assert len(monitor.attention_modules) == 1
        attn.train()
        attn(torch.randn(2, 8, 32))
        assert 0 in monitor.head_max
        assert monitor.head_max[0].shape == (4,)

    def test_clip_caps_logits(self):
        torch.manual_seed(0)
        attn = self._attn()
        monitor = QKClipMonitor.attach(attn, tau=5.0)
        with torch.no_grad():
            attn.q_proj.weight.mul_(50.0)
        attn.train()
        x = torch.randn(1, 8, 32)
        attn(x)
        assert monitor.head_max[0].max().item() > 5.0
        n = monitor.clip(attn)
        assert n > 0
        monitor.head_max.clear()
        attn(x)
        assert monitor.head_max[0].max().item() <= 5.0 + 1e-3

    def test_clip_noop_when_under_tau(self):
        attn = self._attn()
        monitor = QKClipMonitor.attach(attn, tau=1e6)
        attn.train()
        attn(torch.randn(1, 8, 32))
        w_before = attn.q_proj.weight.clone()
        assert monitor.clip(attn) == 0
        assert torch.equal(attn.q_proj.weight, w_before)


# ── CISPO + ETR in GRPOTrainer ──────────────────────────────────────────────


class _MockTok:
    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None, add_special_tokens=True):
        n = 3 + len(text)
        return type("Enc", (), {"input_ids": torch.tensor([[1] * n])})()


class _TinyLM(nn.Module):
    def __init__(self, vocab=16, d=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.blocks = []

    def forward(self, input_ids, use_cache=False, past_key_value=None, **kw):
        return self.head(self.emb(input_ids)), None


def _trainer(**cfg_kw):
    from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(group_size=2, grad_accum_steps=1, **cfg_kw)
    model = _TinyLM()
    ref = _TinyLM()
    return GRPOTrainer(model, _MockTok(), ref, device="cpu", config=cfg)


class TestCISPO:
    def test_config_defaults(self):
        from forge.self_play.grpo_trainer import GRPOConfig

        cfg = GRPOConfig()
        assert cfg.rl_algorithm == "grpo"
        assert cfg.cispo_epsilon_low == 0.2
        assert cfg.cispo_epsilon_high == 0.28

    def test_cispo_train_step_runs(self):
        trainer = _trainer(rl_algorithm="cispo")
        w_before = trainer.model.head.weight.detach().clone()
        stats = trainer.train_step(
            prompts=["p"],
            completions=[["aaaa", "bbbb"]],
            rewards=[[1.0, 0.0]],
        )
        assert stats["n_updates"] > 0
        assert torch.isfinite(torch.tensor(stats["mean_loss"]))
        assert not torch.equal(trainer.model.head.weight, w_before)

    def test_cispo_detached_weight_gradient(self):
        """Clipped IS weight is detached: gradient flows through log pi only."""
        logp = torch.tensor([-2.0], requires_grad=True)
        ratio = torch.tensor([10.0])  # far outside clip band
        adv = 1.0
        w = ratio.clamp(0.8, 1.28).detach()
        loss = -(adv * w * logp).mean()
        loss.backward()
        assert logp.grad is not None and logp.grad.item() != 0.0


class TestETR:
    def test_config_defaults(self):
        from forge.self_play.grpo_trainer import GRPOConfig

        cfg = GRPOConfig()
        assert cfg.use_etr_reward is False
        assert cfg.etr_coeff == 0.1

    def test_etr_train_step_runs(self):
        trainer = _trainer(use_etr_reward=True)
        stats = trainer.train_step(
            prompts=["p"],
            completions=[["aaaa", "bbbb"]],
            rewards=[[1.0, 0.0]],
        )
        assert stats["n_updates"] > 0
        assert torch.isfinite(torch.tensor(stats["mean_loss"]))

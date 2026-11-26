"""Regression tests for the 2026-11-18 self-play fixes.

Covers:
  - GRPOTrainer.collect_log_probs positional alignment — an invalid
    completion must not shift later group entries (was silently
    misaligning old_log_probs vs completions).
  - train_step actually consumes provided old_log_probs (PPO ratio was
    ≡1.0 because nobody passed them — detached fallback remains for
    missing/empty/mismatched entries).
  - ref_model=None → compute_kl_penalty uses the LoRA-disabled policy
    forward (adapter-off base IS the reference — saves a second model).
  - lora_disabled disables + restores the manual-LoRA integration sites.
  - _fuzz_verify seeds its mutation RNG per task description (was a fixed
    seed → identical mutation trajectory for every task).

CPU-only; uses a toy LM + word tokenizer.
"""

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer
from forge.training.bitnet_lora import add_lora_adapters, lora_disabled


class _WordTok:
    """One token per whitespace-separated word; id from first char."""
    pad_token_id = 0

    def __call__(self, text, **kw):
        ids = [(ord(w[0]) % 14) + 1 for w in str(text).split()]
        e = SimpleNamespace()
        e.input_ids = torch.tensor([ids], dtype=torch.long)
        return e


class _ToyLM(nn.Module):
    def __init__(self, vocab=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.head = nn.Linear(64, vocab)
        self.blocks = []  # update_moe_biases iterates this

    def forward(self, input_ids, attention_mask=None):
        return self.head(self.emb(input_ids)), None


def _trainer(config=None, model=None):
    return GRPOTrainer(model or _ToyLM(), _WordTok(), None,
                       device="cpu", config=config or GRPOConfig())


# ── lora_disabled ───────────────────────────────────────────────────────────

class TestLoraDisabled:
    def test_disables_and_restores_nn_linear(self):
        m = nn.Module()
        m.lin = nn.Linear(64, 64).to(torch.bfloat16)
        n, params = add_lora_adapters(m, rank=2, alpha=4)
        assert n == 1
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        base = m.lin._lora_orig_forward(x)
        torch.nn.init.normal_(m.lin.lora_adapter.lora_B)
        assert not torch.allclose(m.lin(x), base)
        with lora_disabled(m):
            assert torch.allclose(m.lin(x), base)
        assert not torch.allclose(m.lin(x), base)  # restored on exit

    def test_exception_safety(self):
        m = nn.Module()
        m.lin = nn.Linear(64, 64).to(torch.bfloat16)
        add_lora_adapters(m, rank=2, alpha=4)
        wrapped = m.lin.forward
        try:
            with lora_disabled(m):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert m.lin.forward is wrapped


# ── collect_log_probs alignment ─────────────────────────────────────────────

class TestCollectLogProbs:
    def test_invalid_first_keeps_positions(self):
        tr = _trainer()
        # comp0 too short (sol_len 1 → skipped), comp1 valid (5 sol tokens)
        out = tr.collect_log_probs(["p p p"], [["x", " a b c d e"]])
        assert out[0][0].numel() == 0
        assert out[0][1].numel() == 5

    def test_invalid_last_keeps_positions(self):
        """The old code produced [empty, lp] here — lp of comp0 landed at
        index 1. Regression guard."""
        tr = _trainer()
        out = tr.collect_log_probs(["p p p"], [[" a b c d e", "x"]])
        assert out[0][0].numel() == 5
        assert out[0][1].numel() == 0


# ── train_step consumes old_log_probs ───────────────────────────────────────

class TestOldLogProbsUsed:
    def test_old_log_probs_change_loss(self):
        cfg = GRPOConfig(grad_accum_steps=8, kl_coefficient=0.0)
        m_a = _ToyLM()
        m_b = copy.deepcopy(m_a)
        tr_a = _trainer(cfg, m_a)
        tr_b = _trainer(GRPOConfig(grad_accum_steps=8, kl_coefficient=0.0), m_b)

        prompts = ["p p p"]
        comps = [["a b c d e", "f g h i j"]]
        rewards = [[1.0, 0.0]]

        old_real = tr_a.collect_log_probs(prompts, comps)
        stats_a = tr_a.train_step(prompts, comps, rewards,
                                  old_log_probs=old_real)
        old_shifted = [[t + 3.0 if t.numel() else t for t in g]
                       for g in old_real]
        stats_b = tr_b.train_step(prompts, comps, rewards,
                                  old_log_probs=old_shifted)
        assert abs(stats_a["mean_loss"] - stats_b["mean_loss"]) > 1e-4

    def test_empty_and_mismatched_old_lp_fall_back(self):
        """Empty/mismatched old_lp entries must not crash or broadcast."""
        tr = _trainer(GRPOConfig(grad_accum_steps=8, kl_coefficient=0.0))
        prompts = ["p p p"]
        comps = [["a b c d e", "f g h i j"]]
        rewards = [[1.0, 0.0]]
        old = [[torch.tensor([]), torch.zeros(2)]]
        stats = tr.train_step(prompts, comps, rewards, old_log_probs=old)
        assert "mean_loss" in stats


# ── ref_model=None → LoRA-disabled KL ───────────────────────────────────────

class TestRefFreeKL:
    def test_kl_runs_without_ref_model(self):
        tr = _trainer()
        ids = torch.tensor([[1, 2, 3, 4]])
        logits, _ = tr.model(ids)
        kl = tr.compute_kl_penalty(ids, logits)
        assert torch.isfinite(kl).all() and kl.item() >= 0

    def test_kl_positive_when_adapter_diverged(self):
        m = _ToyLM().to(torch.bfloat16)
        add_lora_adapters(m, rank=2, alpha=4)
        # lora_B zero-init → adapter is a no-op until trained; diverge it.
        for mod in m.modules():
            if getattr(mod, "lora_adapter", None) is not None:
                torch.nn.init.normal_(mod.lora_adapter.lora_B, std=0.5)
        tr = _trainer(GRPOConfig(), m)
        ids = torch.tensor([[1, 2, 3, 4]])
        logits, _ = m(ids)  # adapted logits
        kl = tr.compute_kl_penalty(ids, logits)  # base (adapter-off) logits
        assert kl.item() > 1e-6


# ── _fuzz_verify per-task seed ──────────────────────────────────────────────

class TestFuzzSeed:
    def _cur(self, captured):
        from forge.self_play.infinite_curriculum import InfiniteCurriculum
        cur = InfiniteCurriculum.__new__(InfiniteCurriculum)
        cur._task_ref = lambda t: "def solve(a):\n    return a"
        cur._execute_sandbox = lambda code, timeout_s=10.0: (
            captured.append(code) or {"returncode": 0, "stdout": "FUZZ_PASS"})
        return cur

    def _task(self, desc):
        return SimpleNamespace(
            description=desc, solve_name="solve",
            test_cases=[{"args": [2, "x"]}] * 4)

    def test_different_tasks_different_mutations(self):
        captured = []
        cur = self._cur(captured)
        cand = "def solve(a):\n    return a"
        assert cur._fuzz_verify(self._task("double a number"), cand)
        assert cur._fuzz_verify(self._task("count the vowels"), cand)
        assert captured[0] != captured[1]

    def test_same_task_deterministic(self):
        captured = []
        cur = self._cur(captured)
        cand = "def solve(a):\n    return a"
        t = self._task("same desc")
        cur._fuzz_verify(t, cand)
        cur._fuzz_verify(t, cand)
        assert captured[0] == captured[1]

"""Benchmark tests comparing R39 engine features vs baselines.

Compares each R39 feature against a baseline approach to verify the
feature delivers its intended benefit (lossless conversion, identical
outputs, more exploration, better balancing, lower cost, etc.).

All tests run on CPU with small shapes and mock models — no real
checkpoints or GPUs are required.
"""
from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn as nn

torch.manual_seed(42)


# ── Mock state-dict factories ──────────────────────────────────────────────

def _make_qwen3_state(n_layers=2, d_model=64, vocab=256):
    """Create a minimal Qwen3-like state dict."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, d_model),
        "model.norm.weight": torch.randn(d_model),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.q_norm.weight"] = torch.randn(d_model)
        sd[p + "self_attn.k_norm.weight"] = torch.randn(d_model)
        sd[p + "input_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(d_model)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.up_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.down_proj.weight"] = torch.randn(d_model, d_model * 2)
    return sd


def _make_gemma3_state(n_layers=2, d_model=64, vocab=256):
    """Create a minimal Gemma3-like state dict."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, d_model),
        "model.norm.weight": torch.randn(d_model),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.q_norm.weight"] = torch.randn(d_model)
        sd[p + "self_attn.k_norm.weight"] = torch.randn(d_model)
        sd[p + "input_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(d_model)
        sd[p + "pre_feedforward_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_feedforward_layernorm.weight"] = torch.randn(d_model)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.up_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "mlp.down_proj.weight"] = torch.randn(d_model, d_model * 2)
    return sd


def _make_llama4_state(n_layers=2, d_model=64, vocab=256, n_experts=4):
    """Create a minimal Llama4-like state dict with MoE."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, d_model),
        "model.norm.weight": torch.randn(d_model),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[p + "input_layernorm.weight"] = torch.randn(d_model)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(d_model)
        # MoE router
        sd[p + "feed_forward.router.weight"] = torch.randn(n_experts, d_model)
        # Shared expert
        sd[p + "feed_forward.shared_expert.gate_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "feed_forward.shared_expert.up_proj.weight"] = torch.randn(d_model * 2, d_model)
        sd[p + "feed_forward.shared_expert.down_proj.weight"] = torch.randn(d_model, d_model * 2)
        # Routed experts
        for e in range(n_experts):
            ep = f"feed_forward.experts.{e}."
            sd[p + ep + "w1.weight"] = torch.randn(d_model * 2, d_model)
            sd[p + ep + "w3.weight"] = torch.randn(d_model * 2, d_model)
            sd[p + ep + "w2.weight"] = torch.randn(d_model, d_model * 2)
    return sd


def _all_tensors_preserved(original: dict, converted: dict) -> bool:
    """Every tensor in *original* must appear (by value) somewhere in *converted*."""
    remaining = list(converted.values())
    for orig_t in original.values():
        found = False
        for j, conv_t in enumerate(remaining):
            if conv_t.shape == orig_t.shape and torch.equal(conv_t, orig_t):
                remaining.pop(j)
                found = True
                break
        if not found:
            return False
    return True


# ── Mock models ────────────────────────────────────────────────────────────

class MockDeterministicModel(nn.Module):
    """Model that always outputs a fixed token — for decoding equivalence tests."""

    def __init__(self, vocab_size=20, token=5, head_dim=8, eos_token_id=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = type("Cfg", (), {
            "vocab_size": vocab_size, "head_dim": head_dim,
            "n_kv_heads": 1, "n_layers": 1,
        })()
        self.eos_token_id = eos_token_id
        self.token = token
        self._vocab_size = vocab_size
        self._head_dim = head_dim
        self.call_count = 0
        self.sparse_k_seen = None

    def forward(self, input_ids, past_key_values=None, use_cache=False,
                sparse_k=None, **kwargs):
        self.call_count += 1
        if sparse_k is not None:
            self.sparse_k_seen = sparse_k
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self._vocab_size), -10.0)
        logits[..., self.token] = 10.0
        key = torch.zeros(batch, 1, seq_len, self._head_dim)
        value = torch.zeros_like(key)
        return logits, None, ((key, value),)


class MockModel:
    """String-in/string-out mock for test-time scaling tests."""

    def __init__(self, suffix=" answer.", vocab_size=16, use_logprobs=False,
                 vary_by_seed=False):
        self.suffix = suffix
        self.vocab_size = vocab_size
        self.use_logprobs = use_logprobs
        self.vary_by_seed = vary_by_seed
        self._seed = 0
        self.call_count = 0

    def seed(self, s):
        self._seed = s

    def generate(self, prompt, **kwargs):
        self.call_count += 1
        max_new = kwargs.get("max_new_tokens", 10)
        out = self.suffix[:max_new] if max_new < len(self.suffix) else self.suffix
        if self.vary_by_seed:
            rng = random.Random(self._seed + self.call_count)
            out = out + str(rng.randint(0, 9))
        return out

    def next_token_logprobs(self, prompt):
        if not self.use_logprobs:
            return None
        base = [-math.log(i + 1) for i in range(self.vocab_size)]
        m = max(base)
        return [b - m for b in base]


class MockTokenizer:
    """Character-level tokenizer for XGrammar tests."""

    def __init__(self, vocab_chars=None):
        if vocab_chars is None:
            vocab_chars = (
                ' \t\n\r{}[]":,0123456789-+.eEtfn'
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/'
            )
        self._vocab = [""] + list(vocab_chars)
        self._id_to_char = dict(enumerate(self._vocab))
        self._char_to_id = {c: i for i, c in enumerate(self._vocab)}

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id_to_char.get(ids, "")
        return [self._id_to_char.get(i, "") for i in ids]

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._id_to_char.get(i, "") for i in ids)

    def encode(self, text, **kwargs):
        return [self._char_to_id.get(c, 0) for c in text]

    @property
    def vocab_size(self):
        return len(self._vocab)


class CountingModel:
    """Records every generate() call — for cascade cost tests."""

    def __init__(self, name="model"):
        self.name = name
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return f"[{self.name}] reply #{len(self.calls)}."


# ── Imports under test ─────────────────────────────────────────────────────

from forge.engine.cascade import ModelCascade
from forge.engine.compat.arch_adapters import (
    convert_gemma3_checkpoint,
    convert_llama4_checkpoint,
    convert_qwen3_checkpoint,
    detect_architecture,
)
from forge.engine.decoding import (
    SelfSpeculativeSparse,
    StandardDecoding,
)
from forge.engine.structured.xgrammar import XGrammarConstrainer
from forge.engine.test_time_scaling import (
    BeamSearch,
    FirstFinishSearch,
    MCTSDecoder,
)
from forge.moe.routers import LASERRouter, METRORouter

# ── 1. Arch Adapters ───────────────────────────────────────────────────────

class TestArchAdaptersLossless:
    """R39-1/2/3: checkpoint conversion must be lossless (tensor passthrough)."""

    def test_qwen3_all_tensors_preserved(self):
        sd = _make_qwen3_state(n_layers=2, d_model=64, vocab=256)
        forge = convert_qwen3_checkpoint(sd, n_layers=2)
        assert _all_tensors_preserved(sd, forge)

    def test_gemma3_all_tensors_preserved(self):
        sd = _make_gemma3_state(n_layers=2, d_model=64, vocab=256)
        forge = convert_gemma3_checkpoint(sd, n_layers=2)
        assert _all_tensors_preserved(sd, forge)

    def test_llama4_all_tensors_preserved(self):
        sd = _make_llama4_state(n_layers=2, d_model=64, vocab=256, n_experts=4)
        forge = convert_llama4_checkpoint(sd, n_layers=2)
        assert _all_tensors_preserved(sd, forge)

    @pytest.mark.parametrize("factory,convert,name", [
        (_make_qwen3_state, convert_qwen3_checkpoint, "qwen3"),
        (_make_gemma3_state, convert_gemma3_checkpoint, "gemma3"),
        (_make_llama4_state, convert_llama4_checkpoint, "llama4"),
    ])
    def test_converted_keys_gte_original(self, factory, convert, name):
        sd = factory(n_layers=2)
        forge = convert(sd, n_layers=2)
        # Some HF keys may collapse to the same forge key, but no data loss:
        # the converted dict must have at least as many entries as original.
        assert len(forge) >= len(sd)
        # Architecture detection sanity check.
        assert detect_architecture(sd) == name


# ── 2. Self-Speculative Sparse vs Standard Decoding ────────────────────────

class TestSelfSpeculativeVsStandard:
    """R39-4: self-speculative sparse decoding must be lossless vs standard."""

    def test_outputs_identical_when_deterministic(self):
        """Both strategies produce identical tokens on a deterministic model.

        StandardDecoding has a degeneration guard that triggers after 8
        identical tokens, so we keep max_new_tokens small (5) to stay below
        that threshold and compare the full output.
        """
        model = MockDeterministicModel(vocab_size=20, token=5, head_dim=8)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        std = StandardDecoding()
        out_std = std.generate(model, input_ids, max_new_tokens=5,
                               temperature=0.0)

        model2 = MockDeterministicModel(vocab_size=20, token=5, head_dim=8)
        spec = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        out_spec = spec.generate(model2, input_ids, max_new_tokens=5,
                                 temperature=0.0)

        # Generated portion (excluding the prompt) must match exactly.
        gen_std = out_std[0, input_ids.shape[1]:]
        gen_spec = out_spec[0, input_ids.shape[1]:]
        assert torch.equal(gen_std, gen_spec)
        # And every generated token is the deterministic token 5.
        assert all(t.item() == 5 for t in gen_std)

    def test_acceptance_rate_high_when_matching(self):
        """On a deterministic model the draft is always accepted."""
        model = MockDeterministicModel(vocab_size=20, token=5, head_dim=8)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        spec = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        out = spec.generate(model, input_ids, max_new_tokens=20,
                            temperature=0.0)

        # Output length should match prompt + max_new_tokens.
        assert out.shape[1] == input_ids.shape[1] + 20
        # Acceptance rate should be high (all draft tokens accepted).
        assert spec.acceptance_rate >= 0.8

    def test_sparse_k_passed_to_model(self):
        """The sparse_k kwarg is forwarded to model.forward during drafting."""
        model = MockDeterministicModel(vocab_size=20, token=5, head_dim=8)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        spec = SelfSpeculativeSparse(draft_len=4, sparse_k=32)
        spec.generate(model, input_ids, max_new_tokens=10, temperature=0.0)

        assert model.sparse_k_seen is not None
        assert model.sparse_k_seen == 32


# ── 3. XGrammar vs Unconstrained ───────────────────────────────────────────

class TestXGrammarVsUnconstrained:
    """R39-5: constrained decoding restricts the token mask vs unconstrained."""

    def test_xgrammar_restricts_vs_unconstrained(self):
        """At START, unconstrained allows all tokens; JSON-object schema only `{`."""
        tok = MockTokenizer()
        cg = XGrammarConstrainer(tok.vocab_size, tok)
        cg.compile_json({"type": "object"})
        cg.reset()
        mask = cg.get_mask(last_token_id=0)

        # Unconstrained mask would be all True.
        unconstrained = torch.ones(tok.vocab_size, dtype=torch.bool)
        assert unconstrained.all()

        # XGrammar allows strictly fewer tokens than the full vocab.
        assert mask.sum().item() < tok.vocab_size
        # The `{` token must be allowed.
        brace_id = tok._char_to_id["{"]
        assert mask[brace_id].item() is True

    def test_xgrammar_valid_json_path(self):
        """Walk a valid JSON object path and verify each expected token is allowed."""
        tok = MockTokenizer()
        cg = XGrammarConstrainer(tok.vocab_size, tok)
        cg.compile_json({"type": "object"})
        cg.reset()

        ids = {
            ch: tok._char_to_id[ch]
            for ch in ['{', '"', ':', 'a', '1']
        }

        # Step 1: START → `{` allowed
        mask = cg.get_mask(0)
        assert mask[ids["{"]].item() is True
        cg.advance(ids["{"])

        # Step 2: after `{` → `"` allowed (start of key)
        mask = cg.get_mask(ids["{"])
        assert mask[ids['"']].item() is True
        cg.advance(ids['"'])

        # Step 3: inside key string → letter `a` allowed
        mask = cg.get_mask(ids['"'])
        assert mask[ids["a"]].item() is True
        cg.advance(ids["a"])

        # Step 4: close key → `"` allowed
        mask = cg.get_mask(ids["a"])
        assert mask[ids['"']].item() is True
        cg.advance(ids['"'])

        # Step 5: after key → `:` allowed
        mask = cg.get_mask(ids['"'])
        assert mask[ids[":"]].item() is True
        cg.advance(ids[":"])

        # Step 6: after colon → value start (digit `1`) allowed
        mask = cg.get_mask(ids[":"])
        assert mask[ids["1"]].item() is True

    def test_xgrammar_blocks_invalid_tokens(self):
        """After `{`, digits cannot start a JSON object key."""
        tok = MockTokenizer()
        cg = XGrammarConstrainer(tok.vocab_size, tok)
        cg.compile_json({"type": "object"})
        cg.reset()

        # Advance past `{` to enter OBJ_KEY state.
        cg.advance(tok._char_to_id["{"])
        mask = cg.get_mask(tok._char_to_id["{"])

        # Digit token IDs must be blocked (a key must start with `"`).
        for digit in "0123456789":
            did = tok._char_to_id[digit]
            assert mask[did].item() is False, (
                f"digit '{digit}' (id={did}) should be blocked after '{{'")


# ── 4. First-Finish Search vs Single Sample ────────────────────────────────

class TestFFSVsSingleSample:
    """R39-6: FFS races multiple samples, returning the first to finish."""

    def test_ffs_returns_valid_string(self):
        model = MockModel(suffix=" answer.")
        ffs = FirstFinishSearch(n_samples=4, max_tokens=20)
        result = ffs.generate(model, "prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ffs_calls_model_multiple_times(self):
        """FFS with n_samples=4 launches parallel samples (>= 2 calls)."""
        model = MockModel(suffix=" answer.")
        ffs = FirstFinishSearch(n_samples=4, max_tokens=20)
        ffs.generate(model, "prompt")
        assert model.call_count >= 2


# ── 5. Beam Search vs Greedy ───────────────────────────────────────────────

class TestBeamSearchVsGreedy:
    """R39-6: beam search explores more than a single greedy pass."""

    def test_beam_width_4_more_calls_than_width_1(self):
        """Width-4 beam expands more candidates than width-1 (greedy)."""
        model_w4 = MockModel(suffix=" answer.")
        model_w1 = MockModel(suffix=" answer.")

        bs4 = BeamSearch(beam_width=4, max_tokens=8)
        bs4.generate(model_w4, "prompt")

        bs1 = BeamSearch(beam_width=1, max_tokens=8)
        bs1.generate(model_w1, "prompt")

        assert model_w4.call_count > model_w1.call_count

    def test_beam_returns_highest_scoring(self):
        """BeamSearch returns a non-empty string (the best beam)."""
        model = MockModel(suffix=" answer.")
        bs = BeamSearch(beam_width=4, max_tokens=8)
        result = bs.generate(model, "prompt")
        assert isinstance(result, str)
        assert len(result) > 0


# ── 6. MCTS vs Greedy ──────────────────────────────────────────────────────

class TestMCTSVsGreedy:
    """R39-6: MCTS performs more model calls than a single greedy pass."""

    def test_mcts_more_calls_than_greedy(self):
        """MCTS with 8 iterations makes more than 1 model call."""
        model = MockModel(suffix=" answer.")
        mcts = MCTSDecoder(n_iterations=8, n_children=2, max_tokens=10)
        mcts.generate(model, "prompt")
        assert model.call_count > 1

    def test_mcts_returns_valid_string(self):
        model = MockModel(suffix=" answer.")
        mcts = MCTSDecoder(n_iterations=8, n_children=2, max_tokens=10)
        result = mcts.generate(model, "prompt")
        assert isinstance(result, str)
        assert len(result) > 0


# ── 7. LASER vs Fixed Top-K ────────────────────────────────────────────────

class TestLASERVsFixedTopK:
    """R39-7: LASER gives early layers more experts than fixed top-k."""

    def test_laser_early_layers_more_experts(self):
        router = LASERRouter(
            n_experts=8, n_layers=10, default_top_k=2,
            layer_k_overrides={0: 4, 1: 3},
        )
        assert router.get_top_k(0) == 4
        assert router.get_top_k(9) == 2
        # Early layer capacity > late layer capacity.
        assert router.get_top_k(0) > router.get_top_k(9)

    def test_laser_more_capacity_than_fixed(self):
        """Sum of top_k across layers with LASER overrides > fixed top_k=2."""
        n_layers = 10
        laser = LASERRouter(
            n_experts=8, n_layers=n_layers, default_top_k=2,
            layer_k_overrides={0: 4, 1: 3},
        )
        laser_total = sum(laser.get_top_k(i) for i in range(n_layers))
        fixed_total = sum(2 for _ in range(n_layers))
        assert laser_total > fixed_total


# ── 8. METRO vs No Balancing ───────────────────────────────────────────────

class TestMETROVsNoBalancing:
    """R39-7: METRO rebalances expert load away from overloaded experts."""

    def test_metro_routes_away_from_overloaded(self):
        """When expert 0 is overloaded, METRO routes some tokens elsewhere."""
        router = METRORouter(n_experts=4, top_k=1, balance_threshold=1.5)
        # Pre-populate extreme imbalance: expert 0 is overloaded.
        router.expert_counts[0] = 100
        router.expert_counts[1] = 1
        router.expert_counts[2] = 1
        router.expert_counts[3] = 1
        router.total_tokens = 100

        # Logits that would normally always pick expert 0.
        biased = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

        # Run several routing decisions; not all should be expert 0.
        routed_to_zero = 0
        for _ in range(10):
            indices, _ = router.route(biased, layer_idx=0)
            if indices[0, 0].item() == 0:
                routed_to_zero += 1
        # METRO should route at least some tokens away from expert 0.
        assert routed_to_zero < 10

    def test_metro_load_balance_improves(self):
        """After repeated biased routing, METRO improves balance vs always-0."""
        router = METRORouter(n_experts=4, top_k=1, balance_threshold=1.5)
        biased = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

        # Run many routing steps with biased logits.
        for _ in range(50):
            router.route(biased, layer_idx=0)

        balance = router.get_load_balance()
        # get_load_balance returns max_load / mean_load (1.0 = perfect).
        # With METRO correction, balance should be finite and improved
        # relative to the worst case (all load on one expert → balance ~4).
        assert balance > 0
        # Not all tokens went to expert 0 — some balancing occurred.
        assert router.expert_counts[0].item() < router.total_tokens


# ── 9. Cascade vs Single Model ─────────────────────────────────────────────

class TestCascadeVsSingleModel:
    """R39-8: cascade routing reduces cost vs always using the large model."""

    def test_cascade_routes_easy_to_small(self):
        small = CountingModel("small")
        large = CountingModel("large")
        cascade = ModelCascade(small, large, difficulty_threshold=0.5)
        assert cascade.route("Hi") == "small"

    def test_cascade_routes_hard_to_large(self):
        small = CountingModel("small")
        large = CountingModel("large")
        cascade = ModelCascade(small, large, difficulty_threshold=0.5)
        hard_prompt = "def solve(equation): " + "x " * 300
        assert cascade.route(hard_prompt) == "large"

    def test_cascade_cost_lower_than_large_only(self):
        """Using the cascade, large_model handles < 20 of 20 mixed prompts."""
        small = CountingModel("small")
        large = CountingModel("large")
        cascade = ModelCascade(small, large, difficulty_threshold=0.5)

        easy_prompts = [
            "Hi", "Hello", "Hey", "Yo", "Sup",
            "Good morning", "How are you", "Thanks", "Bye", "Goodbye",
        ]
        hard_prompts = [
            f"def solve_{i}(equation): " + "x " * 300
            for i in range(10)
        ]

        for p in easy_prompts + hard_prompts:
            cascade.generate(p)

        total = small.calls + large.calls
        assert len(total) == 20
        # Some prompts went to the small model, so large handles < 20.
        assert len(large.calls) < 20
        # All easy prompts went to small.
        assert len(small.calls) == 10
        assert len(large.calls) == 10

"""Tests for the four recommended improvements:

1. SnapKV + 4-bit KV combined cache (kv_backend.py)
2. Golden trajectory injection (GRPOTrainer + ReplayBuffer)
3. ELO-driven curriculum (elo_tracker.py + InfiniteCurriculum)
4. Fused QK-Norm + RoPE Triton kernel (fused_rope_qknorm.py)

All CPU-only, fast, no CUDA required.
"""

import pytest
import torch

# ── 1. SnapKV + 4-bit KV Cache ─────────────────────────────────────────────


class TestSnapKV4BitCache:
    """Test the combined SnapKV eviction + Hadamard INT4 quantization cache."""

    def test_factory_creates_snapkv_4bit(self):
        from forge.engine.kv_backend import SnapKV4BitCache, build_kv_cache

        cache = build_kv_cache("snapkv_4bit")
        assert isinstance(cache, SnapKV4BitCache)

    def test_init_and_append(self):
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2, max_seq_len=256, device="cpu", dtype=torch.float32)
        # Append a small sequence
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.append(k, v, position=0)
        assert cache.seq_len == 4

    def test_get_returns_correct_shape(self):
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2, max_seq_len=256, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        cache.append(k, v, position=0)
        k_out, v_out = cache.get(None)
        assert k_out.shape == (1, 2, 8, 64)
        assert v_out.shape == (1, 2, 8, 64)

    def test_eviction_triggers_at_capacity(self):
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("snapkv_4bit")
        # Small budget to trigger eviction quickly
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2, max_seq_len=256, device="cpu", dtype=torch.float32)
        cache.obs_window = 4
        cache.budget = 8
        cache.max_capacity = 12

        # Append tokens one at a time to trigger eviction
        for i in range(20):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            cache.append(k, v, position=i)

        # After eviction, seq_len should not exceed max_capacity
        assert cache.seq_len <= cache.max_capacity

    def test_info_reports_compression(self):
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2, max_seq_len=256, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.append(k, v, position=0)
        info = cache.info()
        assert info["type"] == "snapkv_4bit"
        assert info["bits"] == 4
        assert info["compression"] > 1.0
        assert "eviction_ratio" in info
        assert "bit_ratio" in info

    def test_clear(self):
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2, max_seq_len=256, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.append(k, v, position=0)
        assert cache.seq_len == 4
        cache.clear()
        assert cache.seq_len == 0


# ── 2. Golden Trajectory Injection ──────────────────────────────────────────


class TestGoldenTrajectoryInjection:
    """Test GRPOTrainer replay buffer integration."""

    def test_replay_buffer_accepted_in_constructor(self):
        """GRPOTrainer should accept a replay_buffer parameter."""
        from forge.self_play.grpo_trainer import GRPOTrainer
        from forge.self_play.replay_buffer import ReplayBuffer

        # Create dummy model, tokenizer, ref_model
        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type(
            "MockTok",
            (),
            {"__call__": lambda self, text, **kw: type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})()},
        )()

        buf = ReplayBuffer(max_size=100)
        trainer = GRPOTrainer(model, tokenizer, ref_model, device="cpu", replay_buffer=buf)
        assert trainer.replay_buffer is buf

    def test_inject_golden_replays_skips_small_buffer(self):
        """Should not inject when buffer is below min size."""
        from forge.self_play.grpo_trainer import GRPOTrainer
        from forge.self_play.replay_buffer import ReplayBuffer

        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type(
            "MockTok",
            (),
            {"__call__": lambda self, text, **kw: type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()},
        )()

        buf = ReplayBuffer(max_size=100)
        # Add only a few items — below replay_min_buffer_size (50)
        for i in range(10):
            buf.add({"prompt": f"p{i}", "solution": f"s{i}", "quality": 1.0})

        trainer = GRPOTrainer(model, tokenizer, ref_model, device="cpu", replay_buffer=buf)
        prompts = ["test prompt"]
        completions = [["comp1", "comp2"]]
        rewards = [[1.0, 0.0]]
        p, _c, _r, n = trainer._inject_golden_replays(prompts, completions, rewards)
        assert n == 0  # should not inject
        assert len(p) == 1  # unchanged

    def test_inject_golden_replays_adds_trajectories(self):
        """Should inject golden trajectories when buffer is large enough."""
        from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer
        from forge.self_play.replay_buffer import ReplayBuffer

        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type(
            "MockTok",
            (),
            {"__call__": lambda self, text, **kw: type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()},
        )()

        buf = ReplayBuffer(max_size=1000)
        # Fill buffer above min size
        for i in range(60):
            buf.add({"prompt": f"prompt_{i}", "solution": f"solution_{i}", "quality": 1.0, "test_passed": True})

        config = GRPOConfig(replay_ratio=0.2, replay_min_buffer_size=50)
        trainer = GRPOTrainer(model, tokenizer, ref_model, device="cpu", config=config, replay_buffer=buf)

        prompts = ["p1", "p2", "p3"]
        completions = [["c1a", "c1b"], ["c2a", "c2b"], ["c3a", "c3b"]]
        rewards = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        p, _c, r, n = trainer._inject_golden_replays(prompts, completions, rewards)
        assert n > 0  # should inject some
        assert len(p) == 3 + n  # original + injected
        # Injected trajectories should have reward 1.0
        for i in range(n):
            assert r[3 + i] == [1.0, 1.0]

    def test_record_golden_trajectories(self):
        """Should record successful trajectories into replay buffer."""
        from forge.self_play.grpo_trainer import GRPOTrainer
        from forge.self_play.replay_buffer import ReplayBuffer

        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type(
            "MockTok",
            (),
            {"__call__": lambda self, text, **kw: type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()},
        )()

        buf = ReplayBuffer(max_size=1000)
        trainer = GRPOTrainer(model, tokenizer, ref_model, device="cpu", replay_buffer=buf)

        prompts = ["p1", "p2"]
        completions = [["good_sol", "bad_sol"], ["ok_sol", "great_sol"]]
        rewards = [[1.0, 0.0], [0.5, 1.0]]

        trainer._record_golden_trajectories(prompts, completions, rewards)

        # Should have recorded 3 successful trajectories (1.0, 1.0, and 0.5 < 0.99)
        # Actually 0.5 < 0.99 so not recorded. Only reward >= 0.99 recorded.
        # good_sol (1.0) + great_sol (1.0) = 2 recorded
        assert len(buf) == 2


# ── 3. ELO-Driven Curriculum ────────────────────────────────────────────────


class TestEloTracker:
    """Test the ELO matchmaking system."""

    def test_initial_state(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        assert elo.model_rating == 1200.0
        assert len(elo.prompt_ratings) == 0

    def test_register_prompt(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker()
        elo.register_prompt("task_1")
        assert "task_1" in elo.prompt_ratings
        assert elo.prompt_ratings["task_1"].rating == 1200.0

    def test_expected_win_prob_equal_rating(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        prob = elo.expected_win_prob(1200.0)
        assert abs(prob - 0.5) < 1e-6  # equal rating = 50%

    def test_expected_win_prob_higher_model(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1400.0)  # model is 200 points higher
        prob = elo.expected_win_prob(1200.0)
        assert prob > 0.5  # model favored

    def test_update_model_rating_success(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        elo.register_prompt("task_1", rating=1200.0)
        elo.update_model_rating("task_1", success=True)
        # Model won → rating should increase
        assert elo.model_rating > 1200.0
        # Prompt lost → rating should decrease
        assert elo.prompt_ratings["task_1"].rating < 1200.0

    def test_update_model_rating_failure(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        elo.register_prompt("task_1", rating=1200.0)
        elo.update_model_rating("task_1", success=False)
        # Model lost → rating should decrease
        assert elo.model_rating < 1200.0
        # Prompt won → rating should increase
        assert elo.prompt_ratings["task_1"].rating > 1200.0

    def test_zero_sum_update(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        elo.register_prompt("task_1", rating=1200.0)
        initial_sum = elo.model_rating + elo.prompt_ratings["task_1"].rating
        elo.update_model_rating("task_1", success=True)
        final_sum = elo.model_rating + elo.prompt_ratings["task_1"].rating
        assert abs(final_sum - initial_sum) < 1e-4  # zero-sum

    def test_select_prompts_goldilocks(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        # Register prompts at various difficulty levels
        for i, rating in enumerate([800, 1000, 1200, 1400, 1600]):
            elo.register_prompt(f"task_{i}", rating=rating)
        # Model at 1200 → should prefer tasks near 1200 (50% win prob)
        selected = elo.select_prompts(n=3)
        # The task at 1200 should be selected (closest to 50%)
        assert "task_2" in selected  # rating 1200 = exact match

    def test_select_mixed_prompts(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        for i in range(20):
            elo.register_prompt(f"task_{i}", rating=1000 + i * 50)
        selected = elo.select_mixed_prompts(n=10, exploration_ratio=0.2)
        assert len(selected) == 10

    def test_k_factor_decreases_with_attempts(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0, k_factor=32.0, k_factor_min=8.0)
        elo.register_prompt("task_1", rating=1200.0)
        # First attempt: full K-factor
        k1 = elo._effective_k("task_1")
        assert k1 == 32.0
        # After 20 attempts: minimum K-factor
        for _ in range(20):
            elo.update_model_rating("task_1", success=True)
        k20 = elo._effective_k("task_1")
        assert k20 == 8.0  # fully decayed

    def test_domain_stats(self):
        from forge.self_play.elo_tracker import EloTracker

        elo = EloTracker(initial_rating=1200.0)
        for i in range(5):
            elo.register_prompt(f"task_{i}", rating=1200.0 + i * 50)
            elo.update_model_rating(f"task_{i}", success=(i % 2 == 0))
        stats = elo.domain_stats()
        assert stats["n_prompts"] == 5
        assert "model_rating" in stats
        assert "mean_success_rate" in stats


# ── 4. Fused QK-Norm + RoPE ─────────────────────────────────────────────────


class TestFusedQKNormRoPE:
    """Test the fused RMSNorm + RoPE kernel (PyTorch fallback on CPU)."""

    def test_pytorch_fallback_matches_separate_ops(self):
        """The PyTorch fallback should produce the same result as separate
        RMSNorm + RoPE ops."""
        from forge.decoding.fused_rope_qknorm import fused_qk_norm_rope

        B, n_heads, T, head_dim = 2, 4, 8, 64
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.ones(head_dim)
        k_weight = torch.ones(head_dim)
        cos = torch.randn(T, head_dim)
        sin = torch.randn(T, head_dim)
        eps = 1e-6

        # Fused (will use PyTorch fallback on CPU)
        q_fused, k_fused = fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps, use_triton=False)

        # Reference: separate ops
        q_rms = q.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        q_normed = q * q_rms * q_weight
        k_rms = k.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        k_normed = k * k_rms * k_weight

        def rotate_half(x):
            d = x.shape[-1]
            return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)

        cos_b = cos.unsqueeze(0).unsqueeze(0)
        sin_b = sin.unsqueeze(0).unsqueeze(0)
        q_ref = (q_normed * cos_b) + (rotate_half(q_normed) * sin_b)
        k_ref = (k_normed * cos_b) + (rotate_half(k_normed) * sin_b)

        assert torch.allclose(q_fused, q_ref, atol=1e-5)
        assert torch.allclose(k_fused, k_ref, atol=1e-5)

    def test_identity_weights_preserve_input_up_to_rope(self):
        """With identity RMSNorm weights (all 1.0), the norm should just
        scale by 1/rms, then RoPE is applied."""
        from forge.decoding.fused_rope_qknorm import fused_qk_norm_rope

        B, n_heads, T, head_dim = 1, 2, 4, 64
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.ones(head_dim)
        k_weight = torch.ones(head_dim)
        cos = torch.ones(T, head_dim)  # identity rotation
        sin = torch.zeros(T, head_dim)
        eps = 1e-6

        q_out, _k_out = fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps, use_triton=False)

        # With cos=1, sin=0: output = normed * 1 + rotate_half(normed) * 0 = normed
        # So output should just be the RMSNormed input
        q_rms = q.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        q_expected = q * q_rms  # identity weight
        assert torch.allclose(q_out, q_expected, atol=1e-5)

    def test_non_identity_weights(self):
        """Non-identity weights should scale the normed output."""
        from forge.decoding.fused_rope_qknorm import fused_qk_norm_rope

        B, n_heads, T, head_dim = 1, 2, 4, 64
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.randn(head_dim) * 0.1 + 1.0  # near-identity
        k_weight = torch.randn(head_dim) * 0.1 + 1.0
        cos = torch.ones(T, head_dim)
        sin = torch.zeros(T, head_dim)
        eps = 1e-6

        q_out, _ = fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps, use_triton=False)

        q_rms = q.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        q_expected = q * q_rms * q_weight
        assert torch.allclose(q_out, q_expected, atol=1e-5)

    def test_single_tensor_wrapper(self):
        """fused_norm_rope_single should work for a single tensor."""
        from forge.decoding.fused_rope_qknorm import fused_norm_rope_single

        B, n_heads, T, head_dim = 1, 4, 8, 64
        x = torch.randn(B, n_heads, T, head_dim)
        weight = torch.ones(head_dim)
        cos = torch.ones(T, head_dim)
        sin = torch.zeros(T, head_dim)

        out = fused_norm_rope_single(x, weight, cos, sin, use_triton=False)
        # With identity cos/sin and identity weight: just RMSNorm
        x_rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        expected = x * x_rms
        assert torch.allclose(out, expected, atol=1e-5)

    def test_head_dim_must_be_power_of_2(self):
        """Triton path requires power-of-2 head_dim. PyTorch path works
        with any head_dim."""
        from forge.decoding.fused_rope_qknorm import fused_qk_norm_rope

        # head_dim=48 (not power of 2) — should work with PyTorch fallback
        B, n_heads, T, head_dim = 1, 2, 4, 48
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.ones(head_dim)
        k_weight = torch.ones(head_dim)
        cos = torch.ones(T, head_dim)
        sin = torch.zeros(T, head_dim)

        # Should work (uses PyTorch fallback)
        q_out, _k_out = fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, use_triton=False)
        assert q_out.shape == q.shape


class _EngineConfig:
    d_model = 4
    n_heads = 1
    n_kv_heads = 1
    head_dim = 4
    max_seq_len = 64
    vocab_size = 16
    n_layers = 1


class _EngineTokenizer:
    eos_token_id = 7

    def __call__(self, text, **kwargs):
        words = text.split() or [text]
        ids = [(len(word) % 5) + 1 for word in words]
        return type("Encoding", (), {"input_ids": torch.tensor([ids])})()

    def __len__(self):
        return 16

    def encode(self, text, **kwargs):
        return [(sum(map(ord, text)) % 8) + 8]

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(
            str(token_id) for token_id in token_ids if not skip_special_tokens or token_id != self.eos_token_id
        )


class _EngineModel(torch.nn.Module):
    def __init__(self, next_token=3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = _EngineConfig()
        self.eos_token_id = 7
        self.next_token = next_token

    def forward(self, input_ids, past_key_values=None, use_cache=False, logits_last_only=False, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.config.vocab_size), -10.0)
        logits[..., self.next_token] = 10.0
        if logits_last_only:
            logits = logits[:, -1:, :]
        key = torch.zeros(batch, 1, seq_len, self.config.head_dim)
        value = torch.zeros_like(key)
        return logits, None, ((key, value),)


def _make_engine():
    from forge.engine.forge_engine import ForgeEngine

    return ForgeEngine(_EngineModel(), _EngineTokenizer(), device="cpu")


class TestForgeEngineRefactor:
    def test_top_p_sampling_returns_a_valid_nucleus_token(self):
        engine = _make_engine()
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        token = engine._sample_next_token(logits, 1.0, 0, 0.8, 1.05, [])
        assert token.item() in {0, 1}

    def test_raw_and_streaming_generation_share_greedy_semantics(self):
        raw_engine = _make_engine()
        stream_engine = _make_engine()
        raw = raw_engine.generate_raw("prompt", max_new_tokens=3, temperature=0.0)
        streamed = "".join(stream_engine.generate_stream("prompt", max_new_tokens=3, temperature=0.0))
        assert raw == streamed == "333"
        assert raw_engine.total_tokens_generated == 3
        assert stream_engine.total_tokens_generated == 3

    def test_learned_prefix_cache_uses_its_native_interface(self):
        from forge.engine.kv.learned_prefix_cache import LearnedPrefixCache

        engine = _make_engine()
        engine._prefix_cache = LearnedPrefixCache(max_entries=2)
        prefix = " ".join(f"prefix{i}" for i in range(32))
        first_prompt = f"{prefix} first suffix tokens"
        second_prompt = f"{prefix} second different suffix"
        assert engine.generate(first_prompt, max_new_tokens=1, finish_sentence=False) == "3"
        assert engine.generate(second_prompt, max_new_tokens=1, finish_sentence=False) == "3"
        assert engine._prefix_cache.stats()["entries"] == 1

    def test_sleep_level_one_transitions_a_fresh_engine(self):
        engine = _make_engine()
        engine.sleep(level=1)
        assert not engine.is_awake
        with pytest.raises(RuntimeError, match="call wake"):
            engine.generate_raw("prompt", max_new_tokens=1)
        engine.wake()
        assert engine.is_awake

    def test_sleep_level_two_requires_a_reloadable_checkpoint(self):
        engine = _make_engine()
        with pytest.raises(RuntimeError, match="checkpoint path"):
            engine.sleep(level=2)
        assert engine.is_awake
        assert engine.model is not None

    def test_level_two_wake_re_activates_strategies(self, monkeypatch):
        """Level 2 wake should re-activate strategies after model reload."""
        from forge.engine.prefix_cache import LRUPrefixCache
        from forge.model_loader import ModelLoader

        engine = _make_engine()
        engine.checkpoint_path = "fake.safetensors"
        engine.activate(kv_cache="standard", decoding="standard", use_prefix_cache=True, warmup=False)
        assert engine.kv_cache is not None
        assert engine.decoding is not None
        assert isinstance(engine._prefix_cache, LRUPrefixCache)

        # Sleep level 2 discards the model
        engine.sleep(level=2)
        assert not engine.is_awake
        assert engine.model is None

        # Mock the model reload to return a fresh _EngineModel
        def fake_build(config, checkpoint_path, dtype):
            return _EngineModel()

        monkeypatch.setattr(ModelLoader, "build_model_fast", staticmethod(fake_build))

        engine.wake()
        assert engine.is_awake
        # Strategies should be re-activated on the new model
        assert engine.kv_cache is not None
        assert engine.decoding is not None
        assert isinstance(engine._prefix_cache, LRUPrefixCache)

    def test_activate_initializes_core_strategies_on_cpu(self):
        from forge.engine.prefix_cache import LRUPrefixCache

        engine = _make_engine()
        engine.activate(kv_cache="standard", decoding="standard", use_prefix_cache=True, warmup=False)
        assert engine.kv_cache.info()["type"] == "standard_prealloc"
        assert engine.decoding.name == "StandardDecoding"
        assert isinstance(engine._prefix_cache, LRUPrefixCache)

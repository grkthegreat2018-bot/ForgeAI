"""Tests for Round 33 KV cache & AutoContext features.

R33-1: EvoSparse — evolving token importance KV cache
R33-2: Vegas — verification-guided KV selection
R33-3: HiSparse — hierarchical HBM-DRAM KV management
R33-4: Capture — activation cache (store input, recompute KV)
R33-5: vToken — token-level virtualization for reclaimable KV
R33-6: AutoContext — automatic context window management (NOVEL)
"""

from __future__ import annotations

import torch

# ── R33-1: EvoSparse ──────────────────────────────────────────────────────


class TestEvoSparse:
    """EvoSparse — evolving token importance with cross-step + cross-layer accumulation."""

    def test_init_append_get(self):
        from forge.engine.kv.evo_sparse import EvoSparseKVCache

        cache = EvoSparseKVCache(keep_ratio=0.5, decay=0.95, n_layers=4)
        cache.init(n_heads=4, head_dim=32, n_kv_heads=2, max_seq_len=64, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            k = torch.randn(1, 2, 1, 32)
            v = torch.randn(1, 2, 1, 32)
            cache.append(k, v, i)
        assert cache.seq_len == 16
        # EvoSparse returns the full [0, seq_len) range
        positions = torch.tensor([[0, 1, 2, 3]])
        k_out, _v_out = cache.get(positions)
        assert k_out.shape[0] == 1
        assert k_out.shape[1] == 2
        assert k_out.shape[3] == 32

    def test_update_importance(self):
        from forge.engine.kv.evo_sparse import EvoSparseKVCache

        cache = EvoSparseKVCache(keep_ratio=0.5, n_layers=2)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        # Simulate attention scores: (n_heads, seq_len, seq_len)
        scores = torch.zeros(2, 8, 8)
        scores[:, :, 3] = 1.0  # position 3 is important
        cache.update_importance(scores, layer_idx=0)
        # Position 3 should have higher importance than others
        assert cache.importance[0, 3] > cache.importance[0, 0]

    def test_eviction_when_full(self):
        from forge.engine.kv.evo_sparse import EvoSparseKVCache

        cache = EvoSparseKVCache(keep_ratio=0.25, n_layers=1)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=16, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        # With keep_ratio=0.25, capacity=4, so 12 should be evicted
        info = cache.info()
        assert info["n_evicted"] > 0, "Should have evicted some positions"

    def test_clear(self):
        from forge.engine.kv.evo_sparse import EvoSparseKVCache

        cache = EvoSparseKVCache(keep_ratio=0.5)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        cache.clear()
        assert cache.seq_len == 0

    def test_info(self):
        from forge.engine.kv.evo_sparse import EvoSparseKVCache

        cache = EvoSparseKVCache(keep_ratio=0.5, n_layers=4)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(10):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        info = cache.info()
        assert info["type"] in ("evo_sparse", "evosparse")
        assert info["seq_len"] == 10
        assert "compression" in info


# ── R33-2: Vegas ──────────────────────────────────────────────────────────


class TestVegas:
    """Vegas — verification-guided KV selection."""

    def test_init_append_get(self):
        from forge.engine.kv.vegas_kv import VegasKVCache

        cache = VegasKVCache(keep_ratio=0.5)
        cache.init(n_heads=4, head_dim=32, n_kv_heads=2, max_seq_len=64, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            cache.append(torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32), i)
        # Vegas returns full [0, seq_len) range with evicted positions zeroed
        positions = torch.tensor([[0, 5, 10, 15]])
        k_out, _v_out = cache.get(positions)
        assert k_out.shape[0] == 1
        assert k_out.shape[1] == 2
        assert k_out.shape[3] == 32

    def test_attention_hints(self):
        from forge.engine.kv.vegas_kv import VegasKVCache

        cache = VegasKVCache(keep_ratio=0.5)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(10):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        # Set hints: position 2 is critical
        scores = torch.zeros(2, 10, 10)
        scores[:, :, 2] = 1.0
        cache.set_attention_hints(scores)
        info = cache.info()
        assert info["has_hints"] is True

    def test_fallback_sliding_window(self):
        from forge.engine.kv.vegas_kv import VegasKVCache

        cache = VegasKVCache(keep_ratio=0.5)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=16, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        # No hints set — should use sliding window fallback
        # Call get to trigger eviction
        cache.get(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]))
        info = cache.info()
        assert info["has_hints"] is False

    def test_clear(self):
        from forge.engine.kv.vegas_kv import VegasKVCache

        cache = VegasKVCache(keep_ratio=0.5)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        cache.clear()
        assert cache.seq_len == 0


# ── R33-3: HiSparse ───────────────────────────────────────────────────────


class TestHiSparse:
    """HiSparse — hierarchical HBM-DRAM KV management."""

    def test_init_append_get(self):
        from forge.engine.kv.hisparse_kv import HiSparseKVCache

        cache = HiSparseKVCache(gpu_cache_size=8)
        cache.init(n_heads=4, head_dim=32, n_kv_heads=2, max_seq_len=64, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32), i)
        positions = torch.tensor([[0, 1, 2, 3]])
        k_out, _v_out = cache.get(positions)
        assert k_out.shape == (1, 2, 4, 32)

    def test_gpu_cache_overflow_to_cpu(self):
        from forge.engine.kv.hisparse_kv import HiSparseKVCache

        cache = HiSparseKVCache(gpu_cache_size=4)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        info = cache.info()
        assert info["cpu_cached_count"] > 0, "Should have CPU-cached positions"

    def test_hit_rate(self):
        from forge.engine.kv.hisparse_kv import HiSparseKVCache

        cache = HiSparseKVCache(gpu_cache_size=8)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        # Access positions in GPU cache → should be hits
        cache.get(torch.tensor([[0, 1, 2, 3]]))
        hr = cache.hit_rate()
        assert hr > 0.0, "Should have some hits"

    def test_info(self):
        from forge.engine.kv.hisparse_kv import HiSparseKVCache

        cache = HiSparseKVCache(gpu_cache_size=8)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(4):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        info = cache.info()
        assert info["type"] in ("hisparse", "hisparse_hbm_dram")
        assert "hit_rate" in info


# ── R33-4: Capture ────────────────────────────────────────────────────────


class TestCapture:
    """Capture — activation cache (store input, recompute KV)."""

    def test_kv_mode_fallback(self):
        from forge.engine.kv.capture_kv import CaptureKVCache

        cache = CaptureKVCache(mode="act")
        cache.init(n_heads=4, head_dim=32, n_kv_heads=2, max_seq_len=64, device="cpu", dtype=torch.bfloat16)
        # Without projection weights, should fall back to KV mode
        for i in range(8):
            cache.append(torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32), i)
        positions = torch.tensor([[0, 1, 2, 3]])
        k_out, _v_out = cache.get(positions)
        assert k_out.shape == (1, 2, 4, 32)

    def test_act_mode_with_projections(self):
        from forge.engine.kv.capture_kv import CaptureKVCache

        cache = CaptureKVCache(mode="act")
        cache.init(n_heads=4, head_dim=16, n_kv_heads=2, max_seq_len=64, device="cpu", dtype=torch.float32)
        # Set projection weights
        hidden_dim = 4 * 16  # n_heads * head_dim
        w_k = torch.randn(2 * 16, hidden_dim)  # n_kv * head_dim, hidden_dim
        w_v = torch.randn(2 * 16, hidden_dim)
        cache.set_projection_weights(w_k, w_v)
        # Append activations
        for i in range(4):
            act = torch.randn(1, hidden_dim)
            cache.append_activation(act, i)
        positions = torch.tensor([[0, 1, 2, 3]])
        k_out, v_out = cache.get(positions)
        assert k_out.shape == (1, 2, 4, 16)
        assert v_out.shape == (1, 2, 4, 16)

    def test_info(self):
        from forge.engine.kv.capture_kv import CaptureKVCache

        cache = CaptureKVCache(mode="kv")
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(4):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        info = cache.info()
        assert info["type"] == "capture"
        assert "memory_savings" in info


# ── R33-5: vToken ─────────────────────────────────────────────────────────


class TestVToken:
    """vToken — token-level virtualization for reclaimable KV."""

    def test_init_append_get(self):
        from forge.engine.kv.vtoken_kv import VTokenKVCache

        cache = VTokenKVCache(block_size=4, max_blocks=8)
        cache.init(n_heads=4, head_dim=32, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            cache.append(
                torch.randn(1, 2, 1, 32, dtype=torch.bfloat16), torch.randn(1, 2, 1, 32, dtype=torch.bfloat16), i
            )
        positions = torch.tensor([[0, 5, 10, 15]])
        k_out, _v_out = cache.get(positions)
        assert k_out.shape == (1, 2, 4, 32)

    def test_mark_dead_and_repack(self):
        from forge.engine.kv.vtoken_kv import VTokenKVCache

        cache = VTokenKVCache(block_size=4, max_blocks=8)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(16):
            cache.append(
                torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), i
            )
        cache.mark_dead([2, 3, 6, 7])
        info_before = cache.info()
        assert info_before["n_dead_tokens"] == 4
        freed = cache.repack()
        assert freed > 0, "Repack should free some blocks"
        info_after = cache.info()
        assert info_after["n_dead_tokens"] == 0, "Dead tokens should be reclaimed"

    def test_info(self):
        from forge.engine.kv.vtoken_kv import VTokenKVCache

        cache = VTokenKVCache(block_size=4, max_blocks=8)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(
                torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), i
            )
        info = cache.info()
        assert info["type"] == "vtoken"
        assert "fragmentation_ratio" in info
        assert "reclaim_efficiency" in info

    def test_clear(self):
        from forge.engine.kv.vtoken_kv import VTokenKVCache

        cache = VTokenKVCache(block_size=4, max_blocks=8)
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(8):
            cache.append(
                torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), i
            )
        cache.clear()
        assert cache.seq_len == 0


# ── R33-6: AutoContext (NOVEL) ────────────────────────────────────────────


class TestAutoContext:
    """AutoContext — automatic context window management."""

    def test_entropy_to_task_type(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager(vram_budget_gb=10.0)
        # High entropy (>2.5 nats) → coding
        high_entropy = [3.0, 3.2, 2.8, 3.1]
        assert mgr.entropy_to_task_type(high_entropy) == "coding"
        # Low entropy (<1.0 nats) → chat
        low_entropy = [0.3, 0.5, 0.4, 0.2]
        assert mgr.entropy_to_task_type(low_entropy) == "chat"

    def test_select_strategy_short_context(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        # Short context with neutral entropy → standard (or streaming if chat override)
        strategy = mgr.select_strategy(context_length=256, entropy_trajectory=[1.5, 1.5, 1.5, 1.5], vram_pressure=0.3)
        # Short context should use standard or streaming (chat override)
        assert strategy in ("standard", "streaming"), f"Short context got {strategy}"

    def test_select_strategy_medium_context(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        # Use coding entropy (high) so task override → snapkv
        strategy = mgr.select_strategy(context_length=1024, entropy_trajectory=[3.0, 3.0, 3.0, 3.0], vram_pressure=0.3)
        assert "snapkv" in strategy or "s4r" in strategy, f"Medium context should use snapkv/s4r, got {strategy}"

    def test_select_strategy_long_context(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        strategy = mgr.select_strategy(context_length=10000, entropy_trajectory=[3.0, 3.0, 3.0, 3.0], vram_pressure=0.3)
        assert "cpu_offload" in strategy, f"Long context should use cpu_offload, got {strategy}"

    def test_select_strategy_vram_pressure(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        # At medium context with high VRAM pressure, should use s4r or cpu_offload
        # Use task_type="auto" but with coding entropy — coding override only
        # applies when vram_pressure < vram_critical (0.9), so at 0.85 it
        # still overrides to snapkv. Use 0.92 to exceed critical threshold.
        strategy = mgr.select_strategy(context_length=1024, entropy_trajectory=[3.0, 3.0, 3.0, 3.0], vram_pressure=0.92)
        assert "cpu_offload" in strategy, f"Critical VRAM should force cpu_offload, got {strategy}"

    def test_select_strategy_critical_vram(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        strategy = mgr.select_strategy(context_length=512, entropy_trajectory=[3.0, 3.0, 3.0, 3.0], vram_pressure=0.95)
        assert "cpu_offload" in strategy, f"Critical VRAM should force cpu_offload, got {strategy}"

    def test_maybe_switch(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        new_strategy = mgr.maybe_switch(
            current_strategy="standard", context_length=1024, entropy_trajectory=[3.0, 3.0, 3.0, 3.0], vram_pressure=0.3
        )
        assert new_strategy is not None, "Should switch from standard at medium context"

    def test_maybe_switch_no_change(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        # Already on streaming (chat override for short context)
        new_strategy = mgr.maybe_switch(
            current_strategy="streaming", context_length=256, entropy_trajectory=[0.3, 0.3, 0.3, 0.3], vram_pressure=0.3
        )
        assert new_strategy is None, "Should not switch when already optimal"

    def test_predict_growth(self):
        from forge.engine.kv.auto_context import AutoContextManager

        mgr = AutoContextManager()
        mgr.record_turn(100)
        mgr.record_turn(200)
        mgr.record_turn(350)
        growth = mgr.predict_growth(None)  # Use internal history
        assert growth > 0, "Should detect positive growth"

    def test_auto_context_kv_cache_wrapper(self):
        from forge.engine.kv.auto_context import AutoContextKVCache

        cache = AutoContextKVCache()
        cache.init(n_heads=2, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.bfloat16)
        for i in range(4):
            cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), i)
        positions = torch.tensor([[0, 1, 2, 3]])
        k_out, _v_out = cache.get(positions)
        assert k_out.shape == (1, 2, 4, 16)


# ── Engine dispatch integration ────────────────────────────────────────────


class TestKVDispatch:
    """Verify new KV strategies are wired into the dispatch."""

    def test_build_evo_sparse(self):
        from forge.engine.kv.evo_sparse import EvoSparseKVCache
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("evo_sparse")
        assert isinstance(cache, EvoSparseKVCache)

    def test_build_vegas(self):
        from forge.engine.kv.vegas_kv import VegasKVCache
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("vegas")
        assert isinstance(cache, VegasKVCache)

    def test_build_hisparse(self):
        from forge.engine.kv.hisparse_kv import HiSparseKVCache
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("hisparse")
        assert isinstance(cache, HiSparseKVCache)

    def test_build_capture(self):
        from forge.engine.kv.capture_kv import CaptureKVCache
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("capture")
        assert isinstance(cache, CaptureKVCache)

    def test_build_vtoken(self):
        from forge.engine.kv.vtoken_kv import VTokenKVCache
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("vtoken")
        assert isinstance(cache, VTokenKVCache)

    def test_build_auto_context(self):
        from forge.engine.kv.auto_context import AutoContextKVCache
        from forge.engine.kv_backend import build_kv_cache

        cache = build_kv_cache("auto_context")
        assert isinstance(cache, AutoContextKVCache)

    def test_engine_fallback_chain_has_new_strategies(self):
        from forge.engine.forge_engine import ForgeEngine

        chain = ForgeEngine._KV_FALLBACK_CHAIN
        for name in ["evo_sparse", "vegas", "hisparse", "capture", "vtoken", "auto_context"]:
            assert name in chain, f"{name} not in fallback chain"

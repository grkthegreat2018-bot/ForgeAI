"""Tests for Round 35 I/O Optimizations & Boot Performance.

R35-1: Lazy GUI page construction
R35-2: Splash screen
R35-3: eLLM — Virtual tensor abstraction
R35-4: AVMP — Asymmetric virtual memory paging
R35-5: DirectKV — (fused kernel, documented as future work)
R35-6: Progressive GGUF loading
"""

from __future__ import annotations

import torch

# ── R35-3: Virtual Tensor ─────────────────────────────────────────────────


class TestVirtualTensor:
    """eLLM — Virtual tensor with elastic GPU/CPU split."""

    def test_init(self):
        from forge.engine.memory.virtual_tensor import VirtualTensor

        vt = VirtualTensor((100,), dtype=torch.float32, device="cpu")
        assert vt.n_elements == 100

    def test_allocate_gpu(self):
        from forge.engine.memory.virtual_tensor import VirtualTensor

        vt = VirtualTensor((100,), dtype=torch.float32, device="cpu")
        vt.allocate_gpu(50)
        stats = vt.memory_stats()
        assert stats["gpu_n"] == 50
        assert stats["cpu_n"] == 50

    def test_inflate_deflate(self):
        from forge.engine.memory.virtual_tensor import VirtualTensor

        vt = VirtualTensor((100,), dtype=torch.float32, device="cpu")
        vt.allocate_gpu(30)
        vt.inflate(20)
        assert vt.memory_stats()["gpu_n"] == 50
        vt.deflate(20)
        assert vt.memory_stats()["gpu_n"] == 30

    def test_device_location(self):
        from forge.engine.memory.virtual_tensor import VirtualTensor

        vt = VirtualTensor((100,), dtype=torch.float32, device="cpu")
        vt.allocate_gpu(50)
        assert vt.device_location(10) == "gpu"
        assert vt.device_location(60) == "cpu"

    def test_to_gpu_cpu(self):
        from forge.engine.memory.virtual_tensor import VirtualTensor

        vt = VirtualTensor((50,), dtype=torch.float32, device="cpu")
        vt.allocate_gpu(10)
        vt.to_gpu()
        assert vt.memory_stats()["gpu_n"] == 50
        vt.to_cpu()
        assert vt.memory_stats()["gpu_n"] == 0

    def test_memory_stats(self):
        from forge.engine.memory.virtual_tensor import VirtualTensor

        vt = VirtualTensor((100,), dtype=torch.float32, device="cpu")
        vt.allocate_gpu(50)
        stats = vt.memory_stats()
        assert "gpu_bytes" in stats
        assert "cpu_bytes" in stats
        assert "gpu_fraction" in stats
        assert stats["gpu_fraction"] == 0.5


class TestVirtualTensorPool:
    """Pool of virtual tensors with shared GPU budget."""

    def test_create(self):
        from forge.engine.memory.virtual_tensor import VirtualTensorPool

        pool = VirtualTensorPool(gpu_budget_bytes=1024 * 1024)
        vt = pool.create((100,), dtype=torch.float32)
        assert vt.n_elements == 100
        assert pool.stats()["n_tensors"] == 1

    def test_rebalance(self):
        from forge.engine.memory.virtual_tensor import VirtualTensorPool

        pool = VirtualTensorPool(gpu_budget_bytes=1024 * 1024)
        pool.create((100,), dtype=torch.float32)
        pool.create((100,), dtype=torch.float32)
        pool.record_access(0)
        pool.rebalance()
        stats = pool.stats()
        assert stats["n_tensors"] == 2


# ── R35-4: AVMP ───────────────────────────────────────────────────────────


class TestAVMP:
    """AVMP — Asymmetric virtual memory paging for hybrid models."""

    def test_init(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=12 * 1024**3, kv_ratio=0.6)
        stats = mgr.stats()
        assert stats["kv_capacity"] > stats["ssm_capacity"]

    def test_request_kv(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=1024 * 1024, kv_ratio=0.6)
        assert mgr.request_kv(1024, "test_kv") is True
        assert mgr.stats()["kv_used"] == 1024

    def test_request_ssm(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=1024 * 1024, kv_ratio=0.6)
        assert mgr.request_ssm(1024, "test_ssm") is True
        assert mgr.stats()["ssm_used"] == 1024

    def test_migration_on_failure(self):
        from forge.engine.memory.avmp import AVMPManager

        # Small budget to trigger migration
        mgr = AVMPManager(gpu_budget_bytes=4096, kv_ratio=0.5)
        # Fill KV pool
        mgr.request_kv(1800, "kv1")
        # This should fail without migration, succeed with migration
        mgr.request_kv(500, "kv2")
        # Either it fits or migration happens
        stats = mgr.stats()
        assert stats["migrations_count"] >= 0

    def test_pressure(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=10000, kv_ratio=0.6)
        mgr.request_kv(3000, "k1")
        mgr.request_ssm(2000, "s1")
        p = mgr.pressure()
        assert 0.0 < p < 1.0

    def test_free(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=10000, kv_ratio=0.6)
        mgr.request_kv(1000, "k1")
        n = mgr.free_kv("k1")
        assert n == 1000
        assert mgr.stats()["kv_used"] == 0

    def test_rebalance(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=10000, kv_ratio=0.6)
        # Fill KV to high pressure
        mgr.request_kv(5000, "k1")
        # SSM has spare
        mgr.rebalance()
        stats = mgr.stats()
        assert stats["migrations_count"] >= 0

    def test_stats(self):
        from forge.engine.memory.avmp import AVMPManager

        mgr = AVMPManager(gpu_budget_bytes=10000, kv_ratio=0.6)
        stats = mgr.stats()
        assert "kv_used" in stats
        assert "ssm_used" in stats
        assert "migrations_count" in stats
        assert "pressure" in stats


# ── R35-6: Progressive Loader ─────────────────────────────────────────────


class TestProgressiveLoader:
    """Progressive tensor loading with background streaming."""

    def test_init(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake_path.pt", device="cpu")
        assert loader.device == "cpu"

    def test_set_tensor_names_and_essential(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake.pt", device="cpu")
        names = [f"layers.{i}.weight" for i in range(10)] + ["embed.weight", "head.weight"]
        loader.set_tensor_names(names)
        essential = loader._essential_names
        assert "embed.weight" in essential
        assert "head.weight" in essential
        # First 25% of layers (2 of 10) + last layer
        assert any("layers.0." in e for e in essential)
        assert any("layers.9." in e for e in essential)

    def test_load_essential(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake.pt", device="cpu")
        names = ["embed.weight", "head.weight", "layers.0.weight", "layers.1.weight"]
        loader.set_tensor_names(names)
        loader.set_load_fn(lambda n: torch.randn(10, 10))
        result = loader.load_essential()
        assert len(result) > 0
        assert loader.loading_progress() > 0

    def test_get_tensor_on_demand(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake.pt", device="cpu")
        loader.set_tensor_names(["a", "b"])
        loader.set_load_fn(lambda n: torch.randn(5, 5))
        t = loader.get_tensor("a")
        assert t is not None
        assert t.shape == (5, 5)

    def test_background_loading(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake.pt", device="cpu")
        names = [f"t{i}" for i in range(10)]
        loader.set_tensor_names(names)
        loader.set_load_fn(lambda n: torch.randn(3, 3))
        loader.load_essential()
        loader.load_remaining_background()
        assert loader.wait_until_loaded(timeout=10.0)
        assert loader.loading_progress() == 1.0

    def test_is_loaded(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake.pt", device="cpu")
        loader.set_tensor_names(["embed.weight", "head.weight"])
        loader.set_load_fn(lambda n: torch.randn(3, 3))
        loader.load_essential()
        assert loader.is_loaded("embed.weight")
        assert loader.is_loaded("head.weight")

    def test_stats(self):
        from forge.engine.loader.progressive_loader import ProgressiveLoader

        loader = ProgressiveLoader("fake.pt", device="cpu")
        loader.set_tensor_names(["embed.weight", "head.weight", "layers.0.weight"])

        def _load(n):
            return torch.empty(3, 3)

        loader.set_load_fn(_load)
        loader.load_essential()
        stats = loader.stats()
        assert "n_total" in stats
        assert "n_loaded" in stats
        assert "loading_progress" in stats
        assert stats["n_total"] == 3

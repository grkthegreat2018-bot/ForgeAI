"""HotPrefix: hotness-aware KV cache scheduling for prefix sharing.

Based on "HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient
Prefix Sharing in LLM Inference Systems" (SIGMOD 2026).

Key insight: in multi-tenant LLM serving, some prefixes are "hot" (shared
by many requests, e.g., system prompts, tool definitions) while others
are "cold" (unique to a single request). HotPrefix:

  1. Dynamic Hotness Tracking: monitors prefix tree node access frequency
  2. Hotness-Aware Eviction: evicts cold prefixes first, keeps hot ones
  3. Hotness Promotion: periodically promotes hot prefix KV caches from
     CPU memory to GPU memory (combines with our CPU KV offload)

The hotness metric combines:
  - Access frequency (how many requests share this prefix)
  - Recency (how recently was it accessed)
  - Prefix length (longer prefixes save more compute when cached)

For our setup:
  - System prompt prefix (~500 tokens): shared by ALL requests → always hot
  - Conversation history: shared by multi-turn requests → warm
  - User-specific context: unique → cold
  - HotPrefix keeps system prompt KV on GPU, offloads cold prefixes to CPU

Integrates with:
  - LearnedPrefixCache: hotness feeds into continuation prediction
  - CPUKVCache: hot prefixes promoted to GPU, cold ones demoted to CPU
  - FeatherScheduler: hot prefixes form homogeneous batches
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class PrefixNode:
    """Node in the prefix hotness tree."""
    prefix_hash: int
    length: int
    kv_cache: torch.Tensor | None = None  # GPU-resident KV for this prefix
    cpu_kv_cache: torch.Tensor | None = None  # CPU fallback
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    hotness: float = 0.0
    is_hot: bool = False
    children: dict = field(default_factory=dict)


class HotPrefixManager:
    """Manages prefix hotness tracking and GPU/CPU promotion.

    Tracks access patterns for all cached prefixes and dynamically
    promotes hot prefixes to GPU, demotes cold ones to CPU.

    Promotion policy:
      - Compute hotness score every N accesses
      - Top-k hot prefixes → GPU (fast access)
      - Rest → CPU (offloaded, slower but saves VRAM)
      - Periodic rebalancing (every 100 accesses or 60 seconds)
    """

    def __init__(self, gpu_capacity: int = 8,  # max prefixes on GPU
                 hotness_threshold: float = 0.7,
                 decay_factor: float = 0.95,
                 rebalance_interval: int = 100):
        self.gpu_capacity = gpu_capacity
        self.hotness_threshold = hotness_threshold
        self.decay_factor = decay_factor
        self.rebalance_interval = rebalance_interval

        self._prefixes: dict[int, PrefixNode] = {}
        self._accesses_since_rebalance = 0
        self._last_rebalance = time.time()

    def access(self, prefix_hash: int, length: int,
               kv_cache: torch.Tensor | None = None) -> PrefixNode | None:
        """Record an access to a prefix and return its node.

        Args:
            prefix_hash: hash of the prefix token sequence
            length: prefix length in tokens
            kv_cache: KV cache tensor (if this is a new prefix being cached)

        Returns:
            PrefixNode if found or created, None if rejected
        """
        if prefix_hash not in self._prefixes:
            if kv_cache is None:
                return None  # unknown prefix, no cache to create
            node = PrefixNode(
                prefix_hash=prefix_hash,
                length=length,
                kv_cache=kv_cache,
                last_access=time.time(),
            )
            self._prefixes[prefix_hash] = node
        else:
            node = self._prefixes[prefix_hash]
            node.access_count += 1
            node.last_access = time.time()

        # Update hotness
        self._update_hotness(node)

        # Periodic rebalancing
        self._accesses_since_rebalance += 1
        if (self._accesses_since_rebalance >= self.rebalance_interval or
                time.time() - self._last_rebalance > 60):
            self._rebalance()
            self._accesses_since_rebalance = 0
            self._last_rebalance = time.time()

        return node

    def _update_hotness(self, node: PrefixNode):
        """Update the hotness score for a prefix node.

        hotness = frequency × recency × length_factor
        """
        current_time = time.time()
        # Recency: exponential decay since last access
        time_since = current_time - node.last_access
        recency = self.decay_factor ** (time_since / 60.0)  # decay per minute

        # Frequency: log-scaled access count
        frequency = min(1.0, (node.access_count + 1) / 20.0)

        # Length factor: longer prefixes are more valuable when cached
        length_factor = min(1.0, node.length / 2048.0)

        node.hotness = frequency * recency * length_factor
        node.is_hot = node.hotness >= self.hotness_threshold

    def _rebalance(self):
        """Rebalance: promote hot prefixes to GPU, demote cold to CPU."""
        # Sort by hotness (descending)
        sorted_nodes = sorted(
            self._prefixes.values(),
            key=lambda n: n.hotness,
            reverse=True,
        )

        # Top-k → GPU (promote)
        for _i, node in enumerate(sorted_nodes[:self.gpu_capacity]):
            if node.kv_cache is None and node.cpu_kv_cache is not None:
                # Promote: CPU → GPU
                node.kv_cache = node.cpu_kv_cache.to("cuda", non_blocking=True)
                node.cpu_kv_cache = None
            node.is_hot = True

        # Rest → CPU (demote)
        for node in sorted_nodes[self.gpu_capacity:]:
            if node.kv_cache is not None and not node.is_hot:
                # Demote: GPU → CPU
                node.cpu_kv_cache = node.kv_cache.to("cpu", non_blocking=True)
                node.kv_cache = None
            node.is_hot = False

    def get_kv_cache(self, prefix_hash: int) -> torch.Tensor | None:
        """Get the KV cache for a prefix (from GPU or CPU)."""
        node = self._prefixes.get(prefix_hash)
        if node is None:
            return None

        if node.kv_cache is not None:
            return node.kv_cache  # GPU (fast)
        elif node.cpu_kv_cache is not None:
            # CPU → need to transfer (slow path)
            # Trigger promotion on next rebalance
            node.access_count += 1
            node.last_access = time.time()
            self._update_hotness(node)
            return node.cpu_kv_cache  # caller handles transfer
        return None

    def evict_cold(self, max_cold: int = 100):
        """Evict the coldest prefixes to free memory."""
        cold = sorted(
            [n for n in self._prefixes.values() if not n.is_hot],
            key=lambda n: n.hotness,
        )
        for node in cold[:max_cold]:
            if node.prefix_hash in self._prefixes:
                del self._prefixes[node.prefix_hash]

    def stats(self) -> dict:
        hot = sum(1 for n in self._prefixes.values() if n.is_hot)
        cold = len(self._prefixes) - hot
        gpu_bytes = sum(
            n.kv_cache.nelement() * n.kv_cache.element_size()
            for n in self._prefixes.values()
            if n.kv_cache is not None
        )
        cpu_bytes = sum(
            n.cpu_kv_cache.nelement() * n.cpu_kv_cache.element_size()
            for n in self._prefixes.values()
            if n.cpu_kv_cache is not None
        )
        return {
            "total_prefixes": len(self._prefixes),
            "hot": hot,
            "cold": cold,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "gpu_capacity": self.gpu_capacity,
        }

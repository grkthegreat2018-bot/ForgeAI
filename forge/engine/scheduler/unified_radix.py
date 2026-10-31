"""Unified Radix Cache: one tree for hybrid model prefix caching.

Based on SGLang's Unified Radix Cache (LMSYS blog 2026-08-11) and HiCache
design (sglang docs).

Problem: hybrid models (full attention + sliding window + recurrent states)
have different reuse semantics for each component. A single reuse boundary
across all components either discards valid reuse or permits invalid reuse.

Unified Radix Cache solution:
  1. Single token-keyed radix topology → canonical coordinate for each prefix
  2. Components (full-attn KV, SWA KV, recurrent state) attach as plugins
  3. Each component controls its own matching, splitting, insertion, eviction
  4. HiCache: 3-level hierarchy (GPU L1, Host L2, External L3)

For our model (LFM2.5-1.2B: 10 conv + 6 GQA attention):
  - Conv layers: recurrent state (checkpoint at exact prefix boundary)
  - GQA layers: full attention KV (reusable across entire matched prefix)
  - Unified tree: one structure, two component types
  - HiCache: GPU L1 (active) + Host L2 (offloaded) for long sessions
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class CacheNode:
    """A node in the radix tree."""
    token_ids: tuple[int, ...]  # tokens stored at this node
    children: dict[int, CacheNode] = field(default_factory=dict)
    parent: CacheNode | None = None
    ref_count: int = 0  # how many active requests reference this node
    last_access: float = field(default_factory=time.time)
    # Component-specific data
    kv_cache: torch.Tensor | None = None  # full-attn KV
    state_cache: torch.Tensor | None = None  # recurrent state
    kv_location: str = "gpu"  # "gpu", "host", "external"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_tokens(self) -> int:
        return len(self.token_ids) + sum(c.total_tokens() for c in self.children.values())


class TreeComponent:
    """Base class for cache components (full-attn, SWA, recurrent)."""

    def match(self, node: CacheNode, tokens: tuple[int, ...],
              offset: int) -> int:
        """Return how many tokens match from offset."""
        raise NotImplementedError

    def split(self, node: CacheNode, split_pos: int):
        """Split a node at the given position."""
        raise NotImplementedError

    def insert(self, node: CacheNode, data: any):
        """Insert cache data into a node."""
        raise NotImplementedError

    def evict(self, node: CacheNode):
        """Evict cache data from a node."""
        raise NotImplementedError


class FullAttentionComponent(TreeComponent):
    """Full attention KV cache component.

    KV is reusable across the entire matched prefix.
    """

    def match(self, node: CacheNode, tokens: tuple[int, ...],
              offset: int) -> int:
        node_tokens = node.token_ids
        match_len = 0
        for i in range(min(len(node_tokens), len(tokens) - offset)):
            if node_tokens[i] == tokens[offset + i]:
                match_len += 1
            else:
                break
        return match_len

    def split(self, node: CacheNode, split_pos: int):
        """Split node's tokens and KV cache at split_pos."""
        old_tokens = node.token_ids
        old_kv = node.kv_cache

        # Create child with remaining tokens
        child = CacheNode(
            token_ids=old_tokens[split_pos:],
            parent=node,
            kv_cache=old_kv[:, :, split_pos:] if old_kv is not None else None,
            kv_location=node.kv_location,
        )

        # Update parent
        node.token_ids = old_tokens[:split_pos]
        node.kv_cache = old_kv[:, :, :split_pos] if old_kv is not None else None
        node.children = {child.token_ids[0]: child}

    def insert(self, node: CacheNode, data: torch.Tensor):
        node.kv_cache = data
        node.last_access = time.time()

    def evict(self, node: CacheNode):
        node.kv_cache = None
        if node.kv_location != "gpu":
            node.kv_location = "evicted"


class RecurrentStateComponent(TreeComponent):
    """Recurrent state component (conv/SSM layers).

    State is valid ONLY at an exact prefix checkpoint.
    """

    def match(self, node: CacheNode, tokens: tuple[int, ...],
              offset: int) -> int:
        # Recurrent state: must match ALL tokens in the node
        node_tokens = node.token_ids
        if len(tokens) - offset < len(node_tokens):
            return 0
        for i in range(len(node_tokens)):
            if node_tokens[i] != tokens[offset + i]:
                return 0
        return len(node_tokens)

    def split(self, node: CacheNode, split_pos: int):
        # Split: state is only valid at exact boundary
        old_tokens = node.token_ids

        child = CacheNode(
            token_ids=old_tokens[split_pos:],
            parent=node,
            state_cache=None,  # no state for the child (needs recompute)
        )

        node.token_ids = old_tokens[:split_pos]
        # State remains valid for the prefix
        node.children = {child.token_ids[0]: child}

    def insert(self, node: CacheNode, data: torch.Tensor):
        node.state_cache = data
        node.last_access = time.time()

    def evict(self, node: CacheNode):
        node.state_cache = None


class UnifiedRadixCache:
    """Unified Radix Cache for hybrid models.

    One tree structure, multiple component types.
    Components attach as plugins with their own reuse semantics.
    """

    def __init__(self, max_gpu_tokens: int = 32768,
                 max_host_tokens: int = 131072,
                 eviction_policy: str = "lru"):
        self.root = CacheNode(token_ids=())
        self.max_gpu_tokens = max_gpu_tokens
        self.max_host_tokens = max_host_tokens
        self.eviction_policy = eviction_policy

        self._gpu_tokens = 0
        self._host_tokens = 0

        # Component registry
        self.components: dict[str, TreeComponent] = {
            "full_attn": FullAttentionComponent(),
            "recurrent": RecurrentStateComponent(),
        }

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def match_prefix(self, token_ids: list[int]) -> tuple[CacheNode, int, int]:
        """Find the longest matching prefix in the tree.

        Returns:
            matched_node: the node where matching stopped
            matched_tokens: number of tokens matched
            total_prefix_len: total prefix length (for KV reuse)
        """
        node = self.root
        tokens = tuple(token_ids)
        offset = 0
        total_matched = 0

        while offset < len(tokens):
            first_token = tokens[offset]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            # Match using the most restrictive component
            match_len = len(child.token_ids)
            for component in self.components.values():
                comp_match = component.match(child, tokens, offset)
                match_len = min(match_len, comp_match)

            if match_len == 0:
                break

            if match_len < len(child.token_ids):
                # Partial match → split the node
                self.components["full_attn"].split(child, match_len)
                node = child
                total_matched += match_len
                offset += match_len
                break
            else:
                # Full match → descend
                node = child
                total_matched += match_len
                offset += match_len

        if total_matched > 0:
            self._hits += 1
        else:
            self._misses += 1

        return node, total_matched, total_matched

    def insert_prefix(self, token_ids: list[int],
                      kv_cache: torch.Tensor | None = None,
                      state_cache: torch.Tensor | None = None):
        """Insert a prefix into the tree with its cache data."""
        node, matched, _total = self.match_prefix(token_ids)

        tokens = tuple(token_ids)
        offset = matched

        # Insert remaining tokens as new nodes
        while offset < len(tokens):
            # Create new node
            remaining = tokens[offset:offset + 256]  # chunk size
            new_node = CacheNode(
                token_ids=remaining,
                parent=node,
                ref_count=1,
            )
            node.children[remaining[0]] = new_node
            node = new_node
            offset += len(remaining)

        # Attach cache data to the final node
        if kv_cache is not None:
            self.components["full_attn"].insert(node, kv_cache)
            self._gpu_tokens += kv_cache.shape[2] if kv_cache.dim() == 4 else 0
        if state_cache is not None:
            self.components["recurrent"].insert(node, state_cache)

    def evict_lru(self, target_tokens: int):
        """Evict least-recently-used nodes until under target."""
        # Collect all leaf nodes sorted by last access time
        nodes = []
        self._collect_nodes(self.root, nodes)
        nodes.sort(key=lambda n: n.last_access)

        evicted = 0
        for node in nodes:
            if evicted >= target_tokens:
                break
            if node.ref_count == 0:
                if node.kv_cache is not None:
                    tokens = node.kv_cache.shape[2] if node.kv_cache.dim() == 4 else 0
                    self.components["full_attn"].evict(node)
                    self._gpu_tokens -= tokens
                    evicted += tokens
                    self._evictions += 1
                if node.state_cache is not None:
                    self.components["recurrent"].evict(node)
                    self._evictions += 1

    def _collect_nodes(self, node: CacheNode, nodes: list):
        """Collect all nodes in the tree."""
        if node.token_ids:
            nodes.append(node)
        for child in node.children.values():
            self._collect_nodes(child, nodes)

    def offload_to_host(self, node: CacheNode):
        """Offload a node's KV cache from GPU to host (HiCache L2)."""
        if node.kv_cache is not None and node.kv_location == "gpu":
            node.kv_cache = node.kv_cache.cpu()
            node.kv_location = "host"
            tokens = node.kv_cache.shape[2] if node.kv_cache.dim() == 4 else 0
            self._gpu_tokens -= tokens
            self._host_tokens += tokens

    def reload_to_gpu(self, node: CacheNode, device: str = "cuda"):
        """Reload a node's KV cache from host to GPU (HiCache L2→L1)."""
        if node.kv_cache is not None and node.kv_location == "host":
            node.kv_cache = node.kv_cache.to(device)
            node.kv_location = "gpu"
            tokens = node.kv_cache.shape[2] if node.kv_cache.dim() == 4 else 0
            self._gpu_tokens += tokens
            self._host_tokens -= tokens

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
            "evictions": self._evictions,
            "gpu_tokens": self._gpu_tokens,
            "host_tokens": self._host_tokens,
            "max_gpu_tokens": self.max_gpu_tokens,
            "total_nodes": len(list(self._collect_all())),
        }

    def _collect_all(self) -> list:
        nodes = []
        self._collect_nodes(self.root, nodes)
        return nodes

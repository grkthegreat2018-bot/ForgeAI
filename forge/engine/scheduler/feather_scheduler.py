"""Feather: prefix-homogeneity-aware batch scheduler.

Based on "Requests of a Feather Must Flock Together: Batch Size vs. Prefix
Homogeneity in LLM Inference" (arXiv 2605.06046).

Key insight: with prefix-sharing workloads (chat, agents, RAG), SMALLER
prefix-homogeneous batches (all requests share a common prefix) achieve
higher decode throughput than LARGER heterogeneous batches, because:
  1. Better spatial locality during KV cache accesses
  2. Shared KV blocks loaded once, reused by all requests in the batch
  3. Fewer unique KV blocks → less HBM traffic

Existing schedulers (vLLM, SGLang) maximize prefix reuse to reduce memory
but don't stop batch formation at smaller homogeneous batches. Feather uses
RL to learn the optimal tradeoff between batch size and prefix homogeneity.

Results: 2-10× higher end-to-end throughput vs existing schedulers.

Also introduces Chunked Hash Tree (CHT): a lightweight data structure for
fast prefix detection that avoids expensive radix-tree traversals (which
can be comparable to GPU execution time).

For our BatchQueue (which batches up to 8 requests per 50ms window):
  - Current: batches by arrival time (no prefix awareness)
  - Feather: batches by prefix similarity → shared KV loaded once
  - Especially valuable for multi-turn chat (shared conversation prefix)
"""
from __future__ import annotations

from collections import defaultdict


class ChunkedHashTree:
    """Chunked Hash Tree for fast prefix detection.

    Replaces radix-tree traversal with hash-based prefix matching.
    O(1) prefix lookup vs O(depth) for radix trees.

    Structure:
      - Hash each token chunk (e.g., 16 tokens) into a 32-bit hash
      - Build a hash chain: prefix_hash = hash(chunk_0, chunk_1, ..., chunk_n)
      - Lookup: compute hash chain for new request, find longest matching chain
    """

    def __init__(self, chunk_size: int = 16):
        self.chunk_size = chunk_size
        self._prefix_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
        # prefix_hash → [(request_id, prefix_length)]

    def _hash_chunk(self, tokens: list[int]) -> int:
        """Hash a chunk of tokens into a 32-bit hash."""
        h = 0
        for t in tokens:
            h = (h * 31 + t) & 0xFFFFFFFF
        return h

    def _compute_hash_chain(self, tokens: list[int]) -> list[int]:
        """Compute the hash chain for a token sequence.

        Returns a list of rolling hashes, one per chunk.
        hash_chain[i] = hash(tokens[0:i*chunk_size])
        """
        chain = []
        h = 0
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            for t in chunk:
                h = (h * 31 + t) & 0xFFFFFFFF
            chain.append(h)
        return chain

    def insert(self, request_id: int, tokens: list[int]):
        """Insert a request's token sequence into the tree."""
        chain = self._compute_hash_chain(tokens)
        for i, h in enumerate(chain):
            self._prefix_map[h].append((request_id, (i + 1) * self.chunk_size))

    def find_longest_prefix(self, tokens: list[int]) -> tuple[int | None, int]:
        """Find the request with the longest shared prefix.

        Args:
            tokens: new request's tokens

        Returns:
            (best_request_id, shared_prefix_length) or (None, 0)
        """
        chain = self._compute_hash_chain(tokens)
        best_request = None
        best_length = 0

        for _i, h in enumerate(chain):
            if h in self._prefix_map:
                for req_id, length in self._prefix_map[h]:
                    if length > best_length:
                        best_length = length
                        best_request = req_id

        return best_request, best_length

    def remove(self, request_id: int):
        """Remove a request from the tree."""
        to_remove = []
        for h, entries in self._prefix_map.items():
            self._prefix_map[h] = [
                (r, l) for r, l in entries if r != request_id]
            if not self._prefix_map[h]:
                to_remove.append(h)
        for h in to_remove:
            del self._prefix_map[h]


class FeatherScheduler:
    """Prefix-homogeneity-aware batch scheduler.

    Groups requests by prefix similarity into homogeneous batches.
    Smaller homogeneous batches outperform larger heterogeneous batches
    when prefix sharing is significant (chat, agents, RAG).

    Scheduling policy:
      1. Compute prefix hash for each waiting request
      2. Group requests by shared prefix (using ChunkedHashTree)
      3. Form batches from prefix-homogeneous groups first
      4. Fall back to heterogeneous batching when no prefix sharing exists

    The homogeneity threshold controls the tradeoff:
      - High threshold (0.8): only batch requests with 80%+ shared prefix
        → smaller batches, higher per-batch throughput
      - Low threshold (0.3): batch anything with 30%+ shared prefix
        → larger batches, moderate throughput gain
    """

    def __init__(self, max_batch_size: int = 8,
                 homogeneity_threshold: float = 0.5,
                 chunk_size: int = 16):
        self.max_batch_size = max_batch_size
        self.homogeneity_threshold = homogeneity_threshold
        self.cht = ChunkedHashTree(chunk_size=chunk_size)
        self._active_requests: dict[int, list[int]] = {}
        self._next_id = 0

    def add_request(self, tokens: list[int]) -> int:
        """Add a request to the scheduler."""
        req_id = self._next_id
        self._next_id += 1
        self._active_requests[req_id] = tokens
        self.cht.insert(req_id, tokens)
        return req_id

    def remove_request(self, req_id: int):
        """Remove a completed request."""
        self.cht.remove(req_id)
        self._active_requests.pop(req_id, None)

    def form_batches(self, waiting: list[int]) -> list[list[int]]:
        """Form prefix-homogeneous batches from waiting requests.

        Args:
            waiting: list of request IDs waiting to be batched

        Returns:
            List of batches (each batch is a list of request IDs)
        """
        if not waiting:
            return []

        # Group by prefix similarity
        groups: dict[int, list[int]] = defaultdict(list)
        assigned: set[int] = set()

        for req_id in waiting:
            if req_id in assigned:
                continue
            tokens = self._active_requests.get(req_id, [])
            if not tokens:
                continue

            # Find the best matching group
            best_group = None
            best_similarity = 0

            for group_prefix_id, members in groups.items():
                group_tokens = self._active_requests.get(group_prefix_id, [])
                if not group_tokens:
                    continue
                similarity = self._prefix_similarity(tokens, group_tokens)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group_prefix_id

            if best_group is not None and best_similarity >= self.homogeneity_threshold:
                groups[best_group].append(req_id)
                assigned.add(req_id)
            else:
                # Start a new group
                groups[req_id] = [req_id]
                assigned.add(req_id)

        # Convert groups to batches (respecting max_batch_size)
        batches = []
        for group_prefix_id, members in groups.items():
            for i in range(0, len(members), self.max_batch_size):
                batches.append(members[i:i + self.max_batch_size])

        return batches

    def _prefix_similarity(self, tokens_a: list[int],
                            tokens_b: list[int]) -> float:
        """Compute prefix similarity between two token sequences.

        Returns the fraction of tokens_a that are shared with tokens_b
        as a common prefix.
        """
        min_len = min(len(tokens_a), len(tokens_b))
        if min_len == 0:
            return 0.0

        # Find longest common prefix
        common = 0
        for i in range(min_len):
            if tokens_a[i] == tokens_b[i]:
                common += 1
            else:
                break

        return common / len(tokens_a) if tokens_a else 0.0

    def stats(self) -> dict:
        return {
            "active_requests": len(self._active_requests),
            "prefix_entries": sum(len(v) for v in self.cht._prefix_map.values()),
            "max_batch_size": self.max_batch_size,
            "homogeneity_threshold": self.homogeneity_threshold,
        }

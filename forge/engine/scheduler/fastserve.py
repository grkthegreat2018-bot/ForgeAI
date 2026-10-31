"""FastServe: iteration-level preemptive scheduling with skip-join MLFQ.

Based on "FastServe: Iteration-Level Preemptive Scheduling for Large Language
Model Inference" (USENIX NSDI 2026).

Problem: existing LLM serving uses run-to-completion processing, suffering
from head-of-line blocking and long latency for short requests stuck behind
long ones.

FastServe solution:
  1. Preemption at token granularity (not request granularity)
  2. Skip-join Multi-Level Feedback Queue (MLFQ) scheduler:
     - Uses input length to assign initial queue (skip lower-priority queues)
     - Preempts long requests to serve short ones
     - Reduces demotions for appropriately-sized requests
  3. Proactive GPU memory management: offload/upload intermediate state
     between GPU and host memory

Results: up to 6.1× throughput improvement over vLLM.

For our serving setup (forge_server.py with SessionManager + BatchQueue):
  - Current: FCFS batching with 50ms window
  - FastServe: skip-join MLFQ with token-level preemption
  - Short requests served immediately, long requests preempted
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class ServeRequest:
    """A serving request with priority tracking."""
    id: int
    input_ids: torch.Tensor
    max_new_tokens: int
    arrival_time: float
    input_length: int = 0
    decoded_tokens: int = 0
    queue_level: int = 0  # MLFQ level (0 = highest priority)
    last_service_time: float = 0.0
    preempted: bool = False
    kv_cache: torch.Tensor | None = None  # offloaded KV cache

    def __post_init__(self):
        if self.input_length == 0:
            self.input_length = self.input_ids.shape[1]
        self.last_service_time = self.arrival_time


class SkipJoinMLFQ:
    """Skip-join Multi-Level Feedback Queue scheduler.

    Multiple priority queues. New requests join an appropriate queue based
    on input length (skip lower-priority queues). Requests are demoted to
    lower-priority queues after receiving service time.

    Skip-join: instead of always joining the highest queue, a request joins
    the queue appropriate for its expected total length. This reduces
    unnecessary demotions.
    """

    def __init__(self, n_levels: int = 4,
                 time_quanta: list[float] | None = None,
                 length_thresholds: list[int] | None = None):
        """
        Args:
            n_levels: number of priority levels
            time_quanta: time slice per level (seconds)
            length_thresholds: input length thresholds for skip-join
        """
        self.n_levels = n_levels
        self.queues = [deque() for _ in range(n_levels)]
        self.time_quanta = time_quanta or [0.05, 0.1, 0.2, 0.5]
        self.length_thresholds = length_thresholds or [128, 512, 2048, 8192]

    def add_request(self, req: ServeRequest):
        """Add a new request using skip-join."""
        # Determine initial queue based on input length
        level = 0
        for i, threshold in enumerate(self.length_thresholds):
            if req.input_length > threshold:
                level = i + 1
            else:
                break
        level = min(level, self.n_levels - 1)
        req.queue_level = level
        self.queues[level].append(req)

    def get_next_request(self) -> ServeRequest | None:
        """Get the highest-priority request."""
        for level in range(self.n_levels):
            if self.queues[level]:
                return self.queues[level].popleft()
        return None

    def preempt_lowest(self) -> ServeRequest | None:
        """Preempt the lowest-priority active request (for memory pressure)."""
        for level in range(self.n_levels - 1, -1, -1):
            if self.queues[level]:
                req = self.queues[level].popleft()
                req.preempted = True
                return req
        return None

    def demote(self, req: ServeRequest):
        """Demote a request to the next lower priority queue."""
        req.queue_level = min(req.queue_level + 1, self.n_levels - 1)
        self.queues[req.queue_level].append(req)

    def requeue(self, req: ServeRequest):
        """Re-queue a request at its current level (after preemption)."""
        self.queues[req.queue_level].append(req)

    def total_requests(self) -> int:
        return sum(len(q) for q in self.queues)

    def stats(self) -> dict:
        return {
            "total": self.total_requests(),
            "per_level": [len(q) for q in self.queues],
            "n_levels": self.n_levels,
        }


class FastServeScheduler:
    """FastServe: iteration-level preemptive scheduling.

    Combines skip-join MLFQ with:
      - Token-level preemption (can preempt between any two decode steps)
      - Proactive GPU↔host memory offloading for preempted requests
      - Dynamic batch formation from highest-priority requests
    """

    def __init__(self, max_batch_size: int = 8,
                 max_gpu_memory_tokens: int = 32768,
                 n_levels: int = 4):
        self.max_batch_size = max_batch_size
        self.max_gpu_tokens = max_gpu_memory_tokens
        self.mlfq = SkipJoinMLFQ(n_levels=n_levels)
        self._active_batch: list[ServeRequest] = []
        self._next_id = 0
        self._gpu_tokens_used = 0

    def submit(self, input_ids: torch.Tensor,
               max_new_tokens: int = 256) -> ServeRequest:
        """Submit a new request."""
        req = ServeRequest(
            id=self._next_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            arrival_time=time.time(),
        )
        self._next_id += 1
        self.mlfq.add_request(req)
        return req

    def form_batch(self) -> list[ServeRequest]:
        """Form a batch from the highest-priority requests.

        Preempts lower-priority requests if memory is needed.
        """
        # Free completed requests from active batch
        self._active_batch = [r for r in self._active_batch
                              if r.decoded_tokens < r.max_new_tokens]

        # Fill batch from MLFQ
        while len(self._active_batch) < self.max_batch_size:
            req = self.mlfq.get_next_request()
            if req is None:
                break

            # Check memory budget
            req_tokens = req.input_length + req.decoded_tokens
            if self._gpu_tokens_used + req_tokens > self.max_gpu_tokens:
                # Need to preempt something
                if self._active_batch:
                    # Preempt lowest-priority active request
                    victim = max(self._active_batch, key=lambda r: r.queue_level)
                    self._offload_request(victim)
                    self._active_batch.remove(victim)
                    self.mlfq.requeue(victim)
                    self._gpu_tokens_used -= victim.input_length + victim.decoded_tokens
                else:
                    # Can't fit even alone — offload to host and wait
                    self.mlfq.requeue(req)
                    continue

            self._active_batch.append(req)
            if req.preempted:
                self._reload_request(req)
                req.preempted = False
            self._gpu_tokens_used += req_tokens

        return self._active_batch

    def _offload_request(self, req: ServeRequest):
        """Offload request's KV cache to host memory."""
        if req.kv_cache is not None:
            req.kv_cache = req.kv_cache.cpu()
        # In practice, this would use pinned memory + async copy

    def _reload_request(self, req: ServeRequest):
        """Reload request's KV cache from host to GPU."""
        if req.kv_cache is not None:
            device = next(iter(self._active_batch)).input_ids.device if self._active_batch else torch.device('cuda')
            req.kv_cache = req.kv_cache.to(device)

    def mark_decoded(self, req: ServeRequest, n_tokens: int = 1):
        """Mark that a request has decoded n tokens."""
        req.decoded_tokens += n_tokens
        req.last_service_time = time.time()

        # Check if request should be demoted (exceeded time quantum)
        service_time = time.time() - req.last_service_time
        if service_time > self.mlfq.time_quanta[req.queue_level]:
            self.mlfq.demote(req)

    def stats(self) -> dict:
        return {
            "mlfq": self.mlfq.stats(),
            "active_batch": len(self._active_batch),
            "gpu_tokens_used": self._gpu_tokens_used,
            "gpu_tokens_max": self.max_gpu_tokens,
        }

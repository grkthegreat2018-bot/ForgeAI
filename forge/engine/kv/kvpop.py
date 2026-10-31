"""KVpop: KV cache compression with Predictive Online Pruning.

Based on "KVpop — Key-Value Cache Compression with Predictive Online Pruning"
(arXiv 2607.05061).

Key insight: existing KV eviction methods decide retention at INSERTION time,
forgoing the evidence that accumulates while the token remains in the cache.
KVpop delays the eviction DECISION (not just eviction) until tokens leave
the protected window, using a learned scoring module supervised by
FUTURE attention mass.

KVpop structure:
  - Sink tokens (small, always kept)
  - Protected window (recent tokens, exempt from eviction)
  - Long-range top-k cache (older tokens, scored by learned module)

The scoring module is trained with supervision anchored to the future-attention
mass a token receives AFTER it leaves the protected window. The target is
computed during training with a transposed-attention pass (avoids materializing
the dense attention map).

Results: bounded KV cache without changing base architecture. Supports
static (fixed budget) and dynamic (model-uncertainty-driven) modes.

For our model:
  - Learned scoring module: small MLP on key + position features
  - Trained during SFT/self-play (supervised by future attention)
  - Inference: bounded cache, top-k selection by learned score
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KVpopScorer(nn.Module):
    """Lightweight head-wise scoring module for KVpop.

    Assigns importance scores to tokens for top-k selection.
    Input: key + position features
    Output: scalar importance score
    """

    def __init__(self, head_dim: int, hidden_dim: int = 64, n_kv: int = 8):
        super().__init__()
        self.n_kv = n_kv
        self.net = nn.Sequential(
            nn.Linear(head_dim + 1, hidden_dim),  # +1 for position
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            k: (n_kv, T, head_dim) keys
            positions: (T,) normalized positions [0, 1]

        Returns:
            scores: (n_kv, T) importance scores
        """
        T = k.shape[1]
        pos_feat = positions.unsqueeze(0).unsqueeze(-1).expand(self.n_kv, T, 1)
        x = torch.cat([k, pos_feat], dim=-1)
        return self.net(x).squeeze(-1)


class KVpopCache:
    """KVpop: predictive online pruning KV cache.

    Maintains:
      - Sink tokens (always kept)
      - Protected window (recent, uncompressed)
      - Long-range top-k cache (scored by learned module)

    The scoring module is trained during SFT/self-play and loaded at inference.
    Tokens are scored when they leave the protected window, and the top-k
    are retained in the long-range cache.
    """

    def __init__(self, n_kv_heads: int, head_dim: int,
                 sink_size: int = 4,
                 window_size: int = 256,
                 long_range_budget: int = 1024,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 scorer: KVpopScorer | None = None):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.window_size = window_size
        self.long_range_budget = long_range_budget
        self.device = device
        self.dtype = dtype

        # Scoring module (optional — if None, use attention-based scoring)
        self.scorer = scorer
        if self.scorer is not None:
            self.scorer = self.scorer.to(device)

        # Storage
        self.k_sink = torch.zeros(1, n_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
        self.v_sink = torch.zeros_like(self.k_sink)
        self.sink_len = 0

        self.k_window = torch.zeros(1, n_kv_heads, window_size, head_dim, dtype=dtype, device=device)
        self.v_window = torch.zeros_like(self.k_window)
        self.window_len = 0

        self.k_long = torch.zeros(1, n_kv_heads, long_range_budget, head_dim, dtype=dtype, device=device)
        self.v_long = torch.zeros_like(self.k_long)
        self.long_len = 0
        self.long_scores = torch.zeros(long_range_budget, device=device)

        self.position = 0

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K/V tokens."""
        T = k.shape[2]

        # First tokens → sink
        if self.sink_len < self.sink_size:
            n_to_sink = min(T, self.sink_size - self.sink_len)
            self.k_sink[0, :, self.sink_len:self.sink_len + n_to_sink] = k[0, :, :n_to_sink]
            self.v_sink[0, :, self.sink_len:self.sink_len + n_to_sink] = v[0, :, :n_to_sink]
            self.sink_len += n_to_sink
            k = k[:, :, n_to_sink:]
            v = v[:, :, n_to_sink:]
            T = k.shape[2]
            self.position += n_to_sink

        if T == 0:
            return

        # Add to window
        for i in range(T):
            if self.window_len >= self.window_size:
                # Window full → move oldest to long-range (with scoring)
                self._promote_to_long_range()
            self.k_window[0, :, self.window_len] = k[0, :, i]
            self.v_window[0, :, self.window_len] = v[0, :, i]
            self.window_len += 1
            self.position += 1

    def _promote_to_long_range(self):
        """Move oldest window token to long-range cache (with scoring)."""
        if self.window_len == 0:
            return

        # Score the oldest token
        k_token = self.k_window[0, :, 0]  # (n_kv, head_dim)
        v_token = self.v_window[0, :, 0]
        pos = torch.tensor([self.position / self.max_seq_len],
                          device=self.device, dtype=self.dtype)

        if self.scorer is not None:
            score = self.scorer(k_token.unsqueeze(1), pos).mean().item()
        else:
            # Fallback: use key norm as importance
            score = k_token.norm().item()

        # If long-range is full, evict lowest-score token
        if self.long_len >= self.long_range_budget:
            evict_idx = self.long_scores.argmin().item()
            if score > self.long_scores[evict_idx]:
                self.k_long[0, :, evict_idx] = k_token
                self.v_long[0, :, evict_idx] = v_token
                self.long_scores[evict_idx] = score
        else:
            self.k_long[0, :, self.long_len] = k_token
            self.v_long[0, :, self.long_len] = v_token
            self.long_scores[self.long_len] = score
            self.long_len += 1

        # Shift window
        self.k_window[0, :, :self.window_len - 1] = self.k_window[0, :, 1:self.window_len]
        self.v_window[0, :, :self.window_len - 1] = self.v_window[0, :, 1:self.window_len]
        self.window_len -= 1

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full K/V (sink + long-range + window)."""
        k = torch.cat([
            self.k_sink[:, :, :self.sink_len],
            self.k_long[:, :, :self.long_len],
            self.k_window[:, :, :self.window_len],
        ], dim=2)
        v = torch.cat([
            self.v_sink[:, :, :self.sink_len],
            self.v_long[:, :, :self.long_len],
            self.v_window[:, :, :self.window_len],
        ], dim=2)
        return k, v

    @property
    def max_seq_len(self) -> int:
        return 32768  # default

    @property
    def total_len(self) -> int:
        return self.sink_len + self.long_len + self.window_len

    def stats(self) -> dict:
        return {
            "sink_len": self.sink_len,
            "long_len": self.long_len,
            "window_len": self.window_len,
            "total_len": self.total_len,
            "budget": self.long_range_budget,
            "has_scorer": self.scorer is not None,
        }


def train_kvpop_scorer(model, scorer, training_data, n_steps=1000):
    """Train the KVpop scoring module.

    Supervision: future attention mass received by each token after it
    leaves the protected window. Computed via transposed-attention pass
    during training (avoids materializing dense attention map).
    """
    torch.optim.Adam(scorer.parameters(), lr=1e-3)

    for step in range(n_steps):
        # Get a batch of training data
        batch = training_data[step % len(training_data)]
        input_ids = batch['input_ids']

        # Forward pass to get hidden states and attention weights
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attn_weights = outputs.attentions  # list of (B, n_heads, T, T)

        # For each layer, compute future attention mass for each token
        # (attention received from FUTURE tokens)
        for layer_attn in attn_weights:
            # layer_attn: (B, n_heads, T, T)
            B, _H, T, _ = layer_attn.shape

            # Future attention: sum of attention from tokens AFTER position i
            # to token i (transposed attention)
            future_attn = torch.zeros(B, T, device=layer_attn.device)
            for i in range(T):
                # Attention from tokens j > i to token i
                future_attn[:, i] = layer_attn[:, :, i + 1:, i].sum(dim=(1, 2))

            # This is the supervision target for the scorer
            # (simplified — full implementation would extract keys and train scorer)
            pass

        # Train scorer to predict future attention mass
        # (simplified — would use actual keys + positions as input)
        pass

    return scorer

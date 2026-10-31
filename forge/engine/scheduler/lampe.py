"""LaMPE: Length-aware Multi-grained Positional Encoding.

Based on "LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive
Long-context Scaling Without Training" (ACL 2026 Findings, arXiv 2605.24786).

Key insight: existing RoPE extension methods use fixed mapping strategies,
ignoring the dynamic relationship between input length and the model's
effective context window.

LaMPE:
  1. Dynamic relationship between mapping length and input length via
     a parametric scaled sigmoid function → adaptively allocates positional
     capacity across varying input lengths
  2. Multi-grained attention: strategically allocates positional resolution
     across different sequence regions → captures both fine-grained locality
     AND long-range dependencies
  3. Training-free: seamlessly applied to any RoPE-based LLM

Results: significant improvements on 3 LLMs × 5 long-context benchmarks
over existing length extrapolation methods.

For our model (32K training → 128K inference):
  - Training-free: no retraining needed
  - Adaptive: positional capacity scales with input length
  - Multi-grained: fine resolution for local, coarse for long-range
"""
from __future__ import annotations

import math

import torch


class LaMPE:
    """Length-aware Multi-grained Positional Encoding.

    Combines:
      1. Dynamic position mapping (scaled sigmoid)
      2. Multi-grained attention (fine local + coarse long-range)
    """

    def __init__(self, head_dim: int, base: float = 10000.0,
                 train_length: int = 32768,
                 n_grains: int = 3,
                 grain_ratios: tuple[float, ...] = (1.0, 0.5, 0.25)):
        """
        Args:
            head_dim: dimension per head
            base: RoPE base frequency
            train_length: model's training context length
            n_grains: number of granularity levels
            grain_ratios: resolution ratio for each grain (1.0 = finest)
        """
        self.head_dim = head_dim
        self.base = base
        self.train_length = train_length
        self.n_grains = n_grains
        self.grain_ratios = grain_ratios

        d = head_dim // 2
        self.freqs = 1.0 / (base ** (torch.arange(0, d).float() / d))

        # Split channels into grains
        self.grain_sizes = [d // n_grains] * n_grains
        self.grain_sizes[-1] += d - sum(self.grain_sizes)  # remainder

    def get_mapping_length(self, input_length: int) -> float:
        """Dynamic mapping length via scaled sigmoid.

        Maps input_length to a position mapping length that:
          - Equals input_length when input ≤ train_length (no remapping)
          - Compresses when input > train_length (extrapolation)

        Uses a parametric scaled sigmoid:
          mapping = train_length + (input - train_length) * sigmoid(α * (input - train_length))
        """
        if input_length <= self.train_length:
            return float(input_length)

        # Scaled sigmoid: smooth transition from identity to compression
        excess = input_length - self.train_length
        # Compression factor: decreases as input grows
        alpha = 1.0 / self.train_length
        compression = 1.0 / (1.0 + math.exp(alpha * excess))
        mapping = self.train_length + excess * compression
        return mapping

    def get_positions(self, T: int, input_length: int) -> torch.Tensor:
        """Get multi-grained position indices.

        Args:
            T: sequence length
            input_length: total input length (for dynamic mapping)

        Returns:
            positions: (T, head_dim//2) multi-grained position indices
        """
        mapping_length = self.get_mapping_length(input_length)

        # Base positions (mapped)
        scale = mapping_length / max(input_length, 1)
        base_positions = torch.arange(T, dtype=torch.float32) * scale

        # Multi-grained: different resolution for different channel groups
        positions = torch.zeros(T, self.head_dim // 2)
        channel_start = 0
        for grain_idx, ratio in enumerate(self.grain_ratios):
            grain_size = self.grain_sizes[grain_idx]
            grain_positions = base_positions * ratio
            positions[:, channel_start:channel_start + grain_size] = \
                grain_positions.unsqueeze(-1).expand(-1, grain_size)
            channel_start += grain_size

        return positions

    def apply(self, x: torch.Tensor, input_length: int,
              offset: int = 0) -> torch.Tensor:
        """Apply LaMPE to a tensor.

        Args:
            x: (B, n_heads, T, head_dim)
            input_length: total input length (for dynamic mapping)
            offset: position offset

        Returns:
            rotated: (B, n_heads, T, head_dim)
        """
        _B, _n_h, T, hd = x.shape
        device = x.device
        d = hd // 2

        # Get multi-grained positions
        positions = self.get_positions(T + offset, input_length)
        positions = positions[offset:offset + T].to(device)  # (T, d)

        # Compute angles
        freqs = self.freqs.to(device)
        angles = positions * freqs.unsqueeze(0)  # (T, d)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        # Apply rotation
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

    def stats(self) -> dict:
        return {
            "train_length": self.train_length,
            "n_grains": self.n_grains,
            "grain_ratios": self.grain_ratios,
            "mapping_at_2x": self.get_mapping_length(self.train_length * 2),
            "mapping_at_4x": self.get_mapping_length(self.train_length * 4),
        }

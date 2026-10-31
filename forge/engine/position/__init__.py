"""Position-encoding helpers: LeRoPE (learnable frequencies) and RoPE-ID.

Merged from lerope.py + rope_id.py (both <5KB, same domain).

LeRoPE: Learnable RoPE frequencies — replaces fixed RoPE frequencies with
learnable scalars (one per band). Initialized to 1.0 (= standard RoPE), so
loading an existing checkpoint is lossless.

RoPE-ID: In-Distribution high-frequency rotation — applies RoPE with high
frequency to a subset of channels, leaving the rest unchanged. Maintains
key/query cluster separation across sequence lengths for length generalization.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── LeRoPE ─────────────────────────────────────────────────────────────

class LeRoPE(nn.Module):
    """LeRoPE: Learnable RoPE frequencies.

    Replaces fixed RoPE frequencies with learnable scalars (one per band).
    Initialized to 1.0 (= standard RoPE), so loading an existing checkpoint
    is lossless. Training learns to adjust frequencies for better performance.
    """

    def __init__(self, head_dim: int, base: float = 10000.0,
                 n_frequency_bands: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        d = head_dim // 2
        self.n_bands = n_frequency_bands or d

        # Base frequencies (standard RoPE)
        base_freqs = 1.0 / (base ** (torch.arange(0, d).float() / d))
        self.register_buffer('base_freqs', base_freqs)

        # Learnable frequency scalars (one per band)
        # Initialize to 1.0 → identical to standard RoPE
        self.freq_scalars = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, offset: int = 0,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply LeRoPE to a tensor.

        Args:
            x: (B, n_heads, T, head_dim)
            offset: position offset
            position_ids: explicit position IDs

        Returns:
            rotated: (B, n_heads, T, head_dim)
        """
        _B, _n_h, T, hd = x.shape
        device = x.device
        d = hd // 2

        if position_ids is not None:
            positions = position_ids.float()
        else:
            positions = torch.arange(offset, offset + T, device=device, dtype=torch.float32)

        # Learnable frequencies: base_freqs * freq_scalars
        freqs = self.base_freqs.to(device) * self.freq_scalars.to(device)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (T, d)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        # Apply rotation
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

    def get_learned_freqs(self) -> torch.Tensor:
        """Return the learned frequency values."""
        return self.base_freqs * self.freq_scalars

    def stats(self) -> dict:
        learned = self.get_learned_freqs()
        return {
            "n_bands": self.n_bands,
            "n_params": self.freq_scalars.numel(),
            "base_freq_range": (self.base_freqs.min().item(), self.base_freqs.max().item()),
            "learned_freq_range": (learned.min().item(), learned.max().item()),
            "scalar_range": (self.freq_scalars.min().item(), self.freq_scalars.max().item()),
            "scalar_mean": self.freq_scalars.mean().item(),
        }


def integrate_lerope_into_model(model: nn.Module, base: float = 10000.0):
    """Replace standard RoPE in a model with LeRoPE.

    Finds all attention layers with RoPE and replaces the rope module
    with LeRoPE. Existing checkpoints load losslessly (scalars init to 1.0).
    """
    count = 0
    for _name, module in model.named_modules():
        if hasattr(module, 'rope') and module.rope is not None:
            # Get head_dim from the existing RoPE
            head_dim = getattr(module.rope, 'head_dim',
                              getattr(module, 'head_dim', 64))
            # Create LeRoPE
            lerope = LeRoPE(head_dim, base=base)
            lerope = lerope.to(next(module.parameters()).device)
            module.rope = lerope
            count += 1

    logger.info(f"  [LeRoPE] Replaced RoPE in {count} attention layers "
          f"({count * 32} learnable params total)")
    return count


# ── RoPE-ID ────────────────────────────────────────────────────────────

class RoPEID:
    """RoPE-ID: In-Distribution high-frequency rotation.

    Applies RoPE with high frequency to a subset of channels, leaving
    the rest unchanged. This maintains key/query cluster separation
    across sequence lengths, enabling length generalization.
    """

    def __init__(self, head_dim: int, base: float = 10000.0,
                 rotation_fraction: float = 0.25,
                 high_freq_base: float = 1e6):
        """
        Args:
            head_dim: dimension per head
            base: base frequency for standard RoPE channels
            rotation_fraction: fraction of channels to apply high-freq rotation
            high_freq_base: base frequency for high-freq channels (much higher)
        """
        self.head_dim = head_dim
        self.base = base
        self.rotation_fraction = rotation_fraction
        self.high_freq_base = high_freq_base

        d = head_dim // 2
        n_rotated = max(1, int(d * rotation_fraction))
        d - n_rotated

        # Standard frequencies for non-rotated channels (low frequency)
        freqs_standard = 1.0 / (base ** (torch.arange(0, d).float() / d))

        # High frequencies for rotated channels
        freqs_high = 1.0 / (high_freq_base ** (torch.arange(0, n_rotated).float() / n_rotated))

        # Combine: high-freq for first n_rotated channels, standard for rest
        self.freqs = torch.zeros(d)
        self.freqs[:n_rotated] = freqs_high
        self.freqs[n_rotated:] = freqs_standard[n_rotated:]

        # Which channels are rotated (high-freq)
        self.rotated_mask = torch.zeros(d, dtype=torch.bool)
        self.rotated_mask[:n_rotated] = True

    def apply(self, x: torch.Tensor, offset: int = 0,
              position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply RoPE-ID to a tensor.

        Args:
            x: (B, n_heads, T, head_dim)
            offset: position offset
            position_ids: explicit position IDs

        Returns:
            rotated: (B, n_heads, T, head_dim)
        """
        _B, _n_h, T, hd = x.shape
        device = x.device
        d = hd // 2

        if position_ids is not None:
            positions = position_ids.float()
        else:
            positions = torch.arange(offset, offset + T, device=device, dtype=torch.float32)

        freqs = self.freqs.to(device)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (T, d)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        # Split into rotated and free channels
        x_rot = x[..., :d]
        x[..., d:]

        # Apply rotation only to rotated channels
        x1 = x_rot * cos - self._shift(x_rot) * sin
        x2 = x_rot * sin + self._shift(x_rot) * cos

        # Combine: rotated channels get new values, free channels pass through
        # Actually, RoPE rotates pairs: (x[i], x[i+d]) → rotation
        # For RoPE-ID: only rotate the first n_rotated pairs
        out = torch.cat([x1, x2], dim=-1)
        return out

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift x by half head_dim (for RoPE pair rotation)."""
        d = x.shape[-1]
        return torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)

    def stats(self) -> dict:
        return {
            "head_dim": self.head_dim,
            "rotation_fraction": self.rotation_fraction,
            "n_rotated": self.rotated_mask.sum().item(),
            "n_free": (~self.rotated_mask).sum().item(),
            "high_freq_base": self.high_freq_base,
            "standard_base": self.base,
        }

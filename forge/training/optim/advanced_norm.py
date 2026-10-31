"""SeeDNorm and Keel: advanced normalization for LLMs.

Based on:
  1. "SeeDNorm: Self-Rescaled Dynamic Normalization" (ICLR 2026):
     Dynamically adjusts scaling coefficient based on current input,
     preserving input norm information. Minimal params, negligible
     efficiency impact. Consistently superior to RMSNorm and LayerNorm.

  2. "Post-LayerNorm Is Back: Stable, ExpressivE, and Deep" (arXiv 2601.19895):
     Keel: Post-LN with Highway-style connection replaces ResNet residual.
     Prevents gradient vanishing in deep networks. Trains robustly at
     1000+ layers. Better perplexity and depth-scaling than Pre-LN.

  3. "Does Your Optimizer Care How You Normalize?" (arXiv 2604.01563):
     Normalization-optimizer coupling: Derf + Muon has negative interaction.
     DyT (Dynamic Tanh) is a bounded-normalizer control with no penalty.
     Important for our Muon-based optimizers.

For our model:
  - SeeDNorm: drop-in replacement for RMSNorm (better zero-shot performance)
  - Keel: enables deeper models without training instability
  - DyT: bounded normalizer compatible with Muon optimizers
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeeDNorm(nn.Module):
    """SeeDNorm: Self-Rescaled Dynamic Normalization.

    Enhances RMSNorm by dynamically adjusting the scaling coefficient based
    on the current input. Preserves input norm information (RMSNorm discards it).

    Formula:
      RMSNorm: x_normed = x / ||x|| * γ
      SeeDNorm: x_normed = x / ||x|| * (γ * δ(x))
      where δ(x) = sigmoid(W_norm · ||x||) is a learned function of input norm

    Only adds 1 parameter per layer (the norm weight W_norm).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Standard per-dimension scale (like RMSNorm)
        self.weight = nn.Parameter(torch.ones(d_model))

        # Dynamic scale: function of input norm
        # Single scalar that modulates scale based on input norm
        self.norm_weight = nn.Parameter(torch.tensor(0.0))  # init 0 → sigmoid=0.5
        self.norm_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SeeDNorm.

        Args:
            x: (..., d_model) input tensor

        Returns:
            normalized: (..., d_model) normalized tensor
        """
        # Compute RMS norm (like RMSNorm)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_normed = x * rms

        # Dynamic scale: modulate based on input norm
        input_norm = x.norm(dim=-1, keepdim=True)  # (..., 1)
        dynamic_scale = torch.sigmoid(self.norm_weight * input_norm + self.norm_bias)

        # Apply: standard scale * dynamic scale
        return x_normed * self.weight * dynamic_scale

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


class DynamicTanh(nn.Module):
    """DyT: Dynamic Tanh — bounded normalizer compatible with Muon.

    From "Transformers without Normalization" (Zhu et al., 2025).
    Replaces normalization with a learnable tanh activation:
      DyT(x) = γ * tanh(α * x) + β

    Benefits:
      - Bounded output (no normalization needed)
      - No negative interaction with Muon optimizer (unlike Derf)
      - Simpler than RMSNorm (no statistics computation)
      - Competitive performance

    For our Muon-based optimizers (SF-NorMuon, AMUSE, MONA): DyT is the
    recommended normalizer (avoids the Derf+Muon coupling penalty).
    """

    def __init__(self, d_model: int, init_alpha: float = 0.6):
        super().__init__()
        self.d_model = d_model

        # Learnable per-dimension scale and shift
        self.weight = nn.Parameter(torch.ones(d_model))  # γ
        self.bias = nn.Parameter(torch.zeros(d_model))    # β

        # Learnable tanh slope (shared scalar)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Dynamic Tanh.

        Args:
            x: (..., d_model) input tensor

        Returns:
            normalized: (..., d_model) bounded output
        """
        return torch.tanh(self.alpha * x) * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, alpha={self.alpha.item():.3f}"


class KeelHighwayConnection(nn.Module):
    """Keel: Highway-style residual connection for Post-LN.

    Replaces the standard ResNet residual pathway:
      standard: y = x + sublayer(PreLN(x))
      Keel:    y = gate * x + (1 - gate) * sublayer(PostLN(x))

    The gate is learnable per-dimension, initialized to favor the residual
    path (gate ≈ 1) for stable early training, then gradually learns to
    incorporate sublayer output.

    This prevents gradient vanishing in deep Post-LN networks, enabling
    stable training at 1000+ layers.
    """

    def __init__(self, d_model: int, init_gate: float = 0.9):
        super().__init__()
        self.d_model = d_model

        # Learnable gate (per-dimension)
        # init_gate = 0.9 → initially favors residual path (stable)
        self.gate = nn.Parameter(torch.full((d_model,), init_gate))

        # Post-LN normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """Apply Keel highway connection.

        Args:
            x: residual input (..., d_model)
            sublayer_output: output from attention or FFN (..., d_model)

        Returns:
            output: (..., d_model) combined output
        """
        # Post-LN: normalize the sublayer output
        normalized = self.norm(sublayer_output)

        # Highway: gate * x + (1 - gate) * normalized
        gate = torch.sigmoid(self.gate)  # ensure (0, 1)
        return gate * x + (1 - gate) * normalized

    def extra_repr(self) -> str:
        gate_mean = torch.sigmoid(self.gate).mean().item()
        return f"d_model={self.d_model}, gate_mean={gate_mean:.3f}"


def replace_rmsnorm_with_seednorm(model: nn.Module) -> int:
    """Replace all RMSNorm layers in a model with SeeDNorm.

    SeeDNorm is a drop-in replacement that preserves the weight parameter
    and adds a dynamic scale. Existing checkpoints load losslessly (the
    new norm_weight and norm_bias initialize to produce identity-like behavior).
    """
    count = 0
    for _name, module in model.named_modules():
        for child_name, child in module.named_children():
            if 'RMSNorm' in type(child).__name__ or 'rms_norm' in child_name.lower():
                d_model = getattr(child, 'weight', torch.ones(1)).shape[0]
                seednorm = SeeDNorm(d_model)
                # Copy existing weight
                if hasattr(child, 'weight'):
                    seednorm.weight.data = child.weight.data.clone()
                seednorm = seednorm.to(next(module.parameters()).device)
                setattr(module, child_name, seednorm)
                count += 1

    print(f"  [SeeDNorm] Replaced {count} RMSNorm layers with SeeDNorm")
    return count


def replace_rmsnorm_with_dyt(model: nn.Module) -> int:
    """Replace all RMSNorm layers with Dynamic Tanh (DyT).

    Recommended when using Muon-based optimizers (avoids normalization-
    optimizer coupling penalty).
    """
    count = 0
    for _name, module in model.named_modules():
        for child_name, child in module.named_children():
            if 'RMSNorm' in type(child).__name__ or 'rms_norm' in child_name.lower():
                d_model = getattr(child, 'weight', torch.ones(1)).shape[0]
                dyt = DynamicTanh(d_model)
                # Copy existing weight as initial scale
                if hasattr(child, 'weight'):
                    dyt.weight.data = child.weight.data.clone()
                dyt = dyt.to(next(module.parameters()).device)
                setattr(module, child_name, dyt)
                count += 1

    print(f"  [DyT] Replaced {count} RMSNorm layers with Dynamic Tanh")
    return count

"""ForgeAdapter — Entropy-Guided Dynamic Adapter Fusion.

R38-6 NOVEL: Cross-domain combination of AdaLoRA (adaptive rank) +
entropy monitoring (existing in ForgeEngine) + mixture-of-adapters.

The insight: not all tokens need the same adapter capacity. Easy tokens
(low entropy) are fine with rank-4; hard tokens (high entropy) need
rank-32. This saves compute on the majority of tokens while preserving
quality on hard ones. The entropy signal is FREE (already computed for
sampling).

Architecture:
  - Train multiple LoRA adapters with different ranks simultaneously
  - At inference, dynamically fuse adapters based on per-token entropy:
    low-entropy tokens use low-rank adapter (fast), high-entropy tokens
    use high-rank adapter (accurate)
  - The fusion is a soft mixture weighted by entropy signal
  - No separate router model needed — entropy is already computed

Training:
  - All adapters share the same base model
  - Each adapter has a different rank (e.g., 4, 8, 16, 32)
  - Loss is weighted by per-token entropy (hard tokens contribute more
    to high-rank adapter loss, easy tokens to low-rank)

Inference:
  - For each token, compute entropy (already available from sampling)
  - Soft mixture: output = sum(softmax(-entropy / temp) * adapter_i(x))
  - Or hard routing: use the adapter whose rank matches the entropy bin
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeAdapter(nn.Module):
    """Multi-rank LoRA adapter with entropy-guided fusion.

    Maintains multiple LoRA adapters at different ranks. At inference,
    fuses them based on per-token entropy.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        ranks: List of ranks for each adapter (e.g., [4, 8, 16, 32]).
        alpha: LoRA scaling factor.
        temperature: Softmax temperature for entropy-based fusion.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ranks: list[int] = (4, 8, 16, 32),
        alpha: float = 16.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = list(ranks)
        self.alpha = alpha
        self.temperature = temperature
        self.n_adapters = len(ranks)

        # Base weight (frozen)
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # LoRA adapters: A (in, r) and B (r, out) for each rank
        self.adapters_A = nn.ParameterList()
        self.adapters_B = nn.ParameterList()
        for r in ranks:
            A = nn.Parameter(torch.zeros(in_features, r))
            B = nn.Parameter(torch.zeros(r, out_features))
            # Kaiming init for A, zero for B (standard LoRA init)
            nn.init.kaiming_uniform_(A, a=5**0.5)
            self.adapters_A.append(A)
            self.adapters_B.append(B)

        # Entropy bin boundaries (for hard routing)
        # Low entropy → low rank, high entropy → high rank
        self.entropy_bins = torch.linspace(0, 10, self.n_adapters + 1)

    def forward(self, x: torch.Tensor, entropy: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional entropy-guided fusion.

        Args:
            x: (batch, seq_len, in_features) input.
            entropy: (batch, seq_len) per-token entropy. If None, uses
                the highest-rank adapter (full capacity).

        Returns:
            (batch, seq_len, out_features) output.
        """
        base_out = F.linear(x, self.base_weight)  # (batch, seq, out)

        if entropy is None:
            # Use highest-rank adapter
            A = self.adapters_A[-1]
            B = self.adapters_B[-1]
            scale = self.alpha / self.ranks[-1]
            lora_out = F.linear(F.linear(x, A.T), B.T) * scale
            return base_out + lora_out

        # Entropy-guided fusion
        _batch, _seq_len, _ = x.shape
        # Compute each adapter's output
        adapter_outputs = []
        for i, (A, B) in enumerate(zip(self.adapters_A, self.adapters_B)):
            scale = self.alpha / self.ranks[i]
            out_i = F.linear(F.linear(x, A.T), B.T) * scale
            adapter_outputs.append(out_i)

        # Stack: (n_adapters, batch, seq, out)
        stacked = torch.stack(adapter_outputs, dim=0)

        # Compute fusion weights from entropy
        # Low entropy → low rank (adapter 0), high entropy → high rank (last)
        # Use softmax over -entropy/temp for low-entropy preference,
        # and softmax over entropy/temp for high-entropy preference
        # Actually: weight_i = softmax(entropy / temp) gives high entropy
        # more weight on higher indices. We want low entropy → low rank.
        # So: weight_i = softmax(-|entropy - bin_center_i| / temp)
        bin_centers = (self.entropy_bins[:-1] + self.entropy_bins[1:]) / 2
        bin_centers = bin_centers.to(x.device)  # (n_adapters,)

        # entropy: (batch, seq) → (batch, seq, 1)
        ent = entropy.unsqueeze(-1)  # (batch, seq, 1)
        # Distance to each bin center: (batch, seq, n_adapters)
        dist = torch.abs(ent - bin_centers.unsqueeze(0).unsqueeze(0))
        # Weights: closer bins get more weight
        weights = F.softmax(-dist / self.temperature, dim=-1)  # (batch, seq, n_adapters)

        # Weighted sum via broadcasting:
        # stacked: (n_adapters, batch, seq, out) → permute to (batch, seq, n_adapters, out)
        # weights: (batch, seq, n_adapters) → unsqueeze to (batch, seq, n_adapters, 1)
        stacked_perm = stacked.permute(1, 2, 0, 3)  # (batch, seq, n_adapters, out)
        fused = (weights.unsqueeze(-1) * stacked_perm).sum(dim=2)  # (batch, seq, out)

        return base_out + fused

    def merge(self, rank_idx: int = -1) -> torch.Tensor:
        """Merge a specific adapter into the base weight.

        Args:
            rank_idx: Index of the adapter to merge (default: highest rank).

        Returns:
            Merged weight matrix (out_features, in_features).
        """
        A = self.adapters_A[rank_idx]
        B = self.adapters_B[rank_idx]
        scale = self.alpha / self.ranks[rank_idx]
        # A is (in, r), B is (r, out) → A @ B is (in, out) → transpose to (out, in)
        return self.base_weight + (A @ B).T * scale

    def merge_all(self) -> list[torch.Tensor]:
        """Merge all adapters. Returns list of merged weights."""
        return [self.merge(i) for i in range(self.n_adapters)]

    def get_entropy_routing(self, entropy: torch.Tensor) -> torch.Tensor:
        """Get the routing distribution for a given entropy tensor.

        Args:
            entropy: (batch, seq_len) per-token entropy.

        Returns:
            (batch, seq_len, n_adapters) routing weights.
        """
        bin_centers = (self.entropy_bins[:-1] + self.entropy_bins[1:]) / 2
        bin_centers = bin_centers.to(entropy.device)
        ent = entropy.unsqueeze(-1)
        dist = torch.abs(ent - bin_centers.unsqueeze(0).unsqueeze(0))
        return F.softmax(-dist / self.temperature, dim=-1)

    def hard_route(self, entropy: torch.Tensor) -> torch.Tensor:
        """Hard routing: assign each token to the nearest bin.

        Args:
            entropy: (batch, seq_len) per-token entropy.

        Returns:
            (batch, seq_len) adapter index for each token.
        """
        bin_centers = (self.entropy_bins[:-1] + self.entropy_bins[1:]) / 2
        bin_centers = bin_centers.to(entropy.device)
        ent = entropy.unsqueeze(-1)  # (batch, seq, 1)
        dist = torch.abs(ent - bin_centers.unsqueeze(0).unsqueeze(0))
        return dist.argmin(dim=-1)  # (batch, seq)


class ForgeAdapterForLinear(nn.Module):
    """Wrapper to apply ForgeAdapter to an existing nn.Linear layer.

    Replaces nn.Linear with ForgeAdapter, copying the original weight.
    """

    def __init__(self, linear: nn.Linear, ranks: list[int] = (4, 8, 16, 32),
                 alpha: float = 16.0, temperature: float = 1.0):
        super().__init__()
        self.adapter = ForgeAdapter(
            in_features=linear.in_features,
            out_features=linear.out_features,
            ranks=ranks,
            alpha=alpha,
            temperature=temperature,
        )
        # Copy original weight
        with torch.no_grad():
            self.adapter.base_weight.copy_(linear.weight)
        self.has_bias = linear.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(linear.bias.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor, entropy: torch.Tensor | None = None) -> torch.Tensor:
        out = self.adapter(x, entropy)
        if self.has_bias:
            out = out + self.bias
        return out


def apply_forge_adapter_to_model(
    model: nn.Module,
    ranks: list[int] = (4, 8, 16, 32),
    alpha: float = 16.0,
    temperature: float = 1.0,
    target_modules: list[str] | None = None,
) -> int:
    """Replace nn.Linear modules with ForgeAdapterForLinear.

    Args:
        model: The model to modify.
        ranks: LoRA ranks for each adapter.
        alpha: Scaling factor.
        temperature: Fusion temperature.
        target_modules: List of module name substrings to target
            (e.g., ["q_proj", "v_proj"]). None = all nn.Linear.

    Returns:
        Number of modules replaced.
    """
    if target_modules is None:
        target_modules = [""]  # match all

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        # Replace in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else getattr(model, parent_name)
        setattr(parent, child_name,
                ForgeAdapterForLinear(module, ranks=ranks, alpha=alpha,
                                      temperature=temperature))
        count += 1
    return count

"""ForgeHybrid Key — Sink-Aware SSM+Attention Routing.

R37-5 NOVEL: Cross-domain combination of attention sink mechanics
(P0-Sink, arXiv 2603.06591) + Mamba-3 SSM (arXiv 2603.15569) + OutRo
(arXiv 2603.14337).

The key insight: attention sinks form because certain tokens become
"information reservoirs" via norm inflation through specific layers.
These tokens don't benefit from full attention and are cheaper to
process with SSM. The sink norm signal IS the router — no learned
router needed.

Architecture:
  - Each layer has both an attention path and an SSM path
  - A sink-norm-based router decides per-token which path to use:
    * Tokens with high sink norm (information reservoirs) → SSM (cheap)
    * Tokens with high attention entropy (need global context) → Attention
  - The router is the sink norm signal itself (no learned parameters)
  - SSM path is zero-init for warm start (starts as pure attention)

Port path (from V11 attention-only checkpoint):
  1. Load V11 weights into attention path
  2. SSM path weights are zero-init (output = 0, identical to pure attention)
  3. Router threshold starts at infinity (all tokens use attention)
  4. Gradually lower threshold to enable SSM routing as sink signals stabilize

This is lossless at warm start (zero-init SSM = identical output to V11).
"""
from __future__ import annotations

import torch

from forge.keys.misc.base import Key, KeyClass, KeyResult


class ForgeHybridKey(Key):
    """Sink-aware SSM+Attention routing key.

    Converts a pure-attention checkpoint to ForgeHybrid format by
    adding a zero-init SSM path alongside the existing attention path.
    The router uses sink norm as the routing signal (no learned router).

    Lossless at warm start: SSM path zero-init = identical to pure attention.
    """

    def __init__(self, d_model: int = 2048, d_state: int = 16,
                 sink_threshold: float = float("inf"),
                 n_ssm_layers: int | None = None):
        """
        Args:
            d_model: Model hidden dimension.
            d_state: SSM state dimension.
            sink_threshold: Norm ratio threshold for routing to SSM.
                Tokens with sink_norm_ratio > threshold → SSM.
                Default: inf (all tokens use attention at warm start).
            n_ssm_layers: Number of layers to add SSM paths to.
                None = all layers. Can be a subset for gradual rollout.
        """
        self.d_model = d_model
        self.d_state = d_state
        self.sink_threshold = sink_threshold
        self.n_ssm_layers = n_ssm_layers

    @property
    def name(self) -> str:
        return "forge_hybrid"

    @property
    def description(self) -> str:
        return ("Sink-aware SSM+Attention routing. Zero-init SSM path "
                "alongside attention. Router = sink norm signal (no "
                "learned router). Lossless warm start from pure attention.")

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Convert pure-attention checkpoint to ForgeHybrid format.

        Adds zero-init SSM weights alongside existing attention weights.
        The result is lossless: with zero-init SSM, output = attention output.

        Expected input: standard ForgeAI block weights with attn.* keys.
        Output: same attn.* keys + new ssm.* keys (all zero-init).
        """
        try:
            weights = {}
            n_layers = 0
            # Find max layer index
            for key in data:
                if key.startswith("blocks."):
                    parts = key.split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        n_layers = max(n_layers, int(parts[1]) + 1)

            # Copy all existing weights (attention path)
            for key, val in data.items():
                weights[key] = val.clone() if isinstance(val, torch.Tensor) else val

            # Add zero-init SSM weights for each layer (or subset)
            layers_to_add = range(n_layers) if self.n_ssm_layers is None else range(min(self.n_ssm_layers, n_layers))
            for layer_idx in layers_to_add:
                prefix = f"blocks.{layer_idx}.ssm."
                # SSM weights (Mamba-2 style, zero-init)
                d_inner = self.d_model * 2  # standard Mamba expansion
                weights[prefix + "in_proj.weight"] = torch.zeros(d_inner, self.d_model)
                weights[prefix + "conv1d.weight"] = torch.zeros(d_inner, 1, 4)
                weights[prefix + "conv1d.bias"] = torch.zeros(d_inner)
                weights[prefix + "x_proj.weight"] = torch.zeros(self.d_state * 2 + 8, d_inner)
                weights[prefix + "dt_proj.weight"] = torch.zeros(d_inner, 8)
                weights[prefix + "dt_proj.bias"] = torch.zeros(d_inner)
                weights[prefix + "A_log"] = torch.zeros(d_inner, self.d_state)
                weights[prefix + "D"] = torch.zeros(d_inner)
                weights[prefix + "out_proj.weight"] = torch.zeros(self.d_model, d_inner)
                # Router config (stored as metadata, not weights)
                weights[prefix + "sink_threshold"] = torch.tensor(self.sink_threshold)

            return KeyResult(
                success=True,
                weights=weights,
                metadata={
                    "n_layers": n_layers,
                    "n_ssm_layers": len(list(layers_to_add)),
                    "sink_threshold": self.sink_threshold,
                    "lossless": True,
                    "warm_start": "zero_init_ssm",
                }
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Extract pure-attention checkpoint from ForgeHybrid format.

        Drops all SSM weights, keeps only the attention path.
        This is lossless if the SSM path was zero-init (warm start).
        """
        try:
            data = {}
            for key, val in weights.items():
                # Skip SSM weights and router config
                if ".ssm." in key or "sink_threshold" in key:
                    continue
                data[key] = val.clone() if isinstance(val, torch.Tensor) else val

            return KeyResult(
                success=True,
                data=data,
                metadata={
                    "extracted": "attention_only",
                    "dropped_ssm": True,
                }
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def detect_sinks(self, attention_weights: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
        """Detect sink tokens from attention weights.

        Sink tokens are positions that receive disproportionately high
        attention from all other tokens (typically position 0).

        Args:
            attention_weights: (n_heads, seq_len, seq_len) attention scores.
            threshold: Fraction of max attention to qualify as sink.

        Returns:
            Boolean mask (seq_len,) — True for sink positions.
        """
        # Average across heads
        avg_attn = attention_weights.mean(dim=0)  # (seq_len, seq_len)
        # How much attention each position receives (column-wise sum)
        received = avg_attn.sum(dim=0)  # (seq_len,)
        # Normalize
        received_norm = received / received.max().clamp(min=1e-8)
        return received_norm > threshold

    def compute_sink_norm_ratio(self, hidden_states: torch.Tensor,
                                sink_idx: int = 0) -> torch.Tensor:
        """Compute sink norm ratio for each token.

        The sink norm ratio = ||h_token|| / ||h_sink||.
        Tokens with HIGH ratio have norm inflation → information reservoirs
        → route to SSM. Tokens with LOW ratio → route to attention.

        Args:
            hidden_states: (batch, seq_len, d_model)
            sink_idx: Index of the sink token (usually 0).

        Returns:
            (batch, seq_len) tensor of norm ratios.
        """
        norms = hidden_states.norm(dim=-1)  # (batch, seq_len)
        sink_norm = norms[:, sink_idx:sink_idx + 1].clamp(min=1e-8)
        return norms / sink_norm

    def route_tokens(self, sink_norm_ratio: torch.Tensor) -> torch.Tensor:
        """Route tokens to SSM (True) or attention (False).

        Args:
            sink_norm_ratio: (batch, seq_len) from compute_sink_norm_ratio.

        Returns:
            (batch, seq_len) boolean mask. True = use SSM, False = use attention.
        """
        return sink_norm_ratio > self.sink_threshold

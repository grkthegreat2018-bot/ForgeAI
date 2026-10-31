"""V3 Expert Tying — tie expert weights across consecutive layer pairs.

Research basis: arxiv 2606.16825
  - Tie (share) expert weights across consecutive layer pairs
  - 2x FFN VRAM reduction (layers 0+1 share, 2+3 share, etc.)
  - Near-lossless: consecutive layers learn similar features
  - Implementation: one pointer assignment per expert pair

The method:
  For layer pairs (0,1), (2,3), (4,5), ...:
    - Layer N+1's experts point to Layer N's experts
    - Only even layers hold unique weights
    - Odd layers are aliases (zero extra VRAM)

  This works because consecutive transformer layers often learn
  similar transformations (especially in deep models with residual
  connections). The residual stream carries information forward,
  so adjacent layers refine rather than reinvent.

Key class: PARTIAL — weight sharing, needs fine-tuning to recover quality.
  Near-lossless for models with many layers (28 layers = 14 unique).

Usage:
    from forge.keys.expert_tying_key import ExpertTyingKey
    key = ExpertTyingKey()
    key.apply(model)  # tie experts in-place
    # Or as a state dict transform:
    result = key.forward({"state": state, "n_layers": 28})
"""

import torch
import torch.nn as nn

from forge.keys.misc.base import Key, KeyClass, KeyResult


class ExpertTyingKey(Key):
    """Expert Tying — share expert weights across consecutive layer pairs.

    Ties MoE expert weights between layer pairs (0,1), (2,3), etc.
    Odd layers become aliases to even layers (zero extra VRAM).
    2x FFN VRAM reduction, near-lossless with fine-tuning.

    Key class: PARTIAL — weight sharing, needs fine-tuning.
    """

    def __init__(self, tie_ratio: float = 0.5):
        self.tie_ratio = tie_ratio  # fraction of layers that are aliases

    @property
    def name(self) -> str:
        return "expert_tying"

    @property
    def description(self) -> str:
        return ("Tie MoE expert weights across consecutive layer pairs "
                "(2x FFN VRAM, near-lossless, arxiv 2606.16825)")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Transform state dict to tie expert weights."""
        try:
            state = dict(data.get("state", data))
            n_layers = data["n_layers"]

            tied_layers, saved = self._tie_state_dict(state, n_layers)

            print(f"  [ExpertTying] Tied {len(tied_layers)} layer pairs "
                  f"(layers {tied_layers})")
            print(f"    Saved {saved / 1024 / 1024:.1f} MB of expert weights")

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "tied_layers": tied_layers,
                    "n_tied_pairs": len(tied_layers),
                    "saved_mb": saved / 1024 / 1024,
                    "lossy": True,
                    "near_lossless": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def apply(self, model: nn.Module, similarity_threshold: float = 0.0) -> list[tuple[int, int]]:
        """Apply expert tying to a live model (in-place pointer assignment).

        For each consecutive layer pair (N, N+1):
            - Measure cosine similarity between expert weights
            - If similarity > threshold, tie them (share pointers)
            - If similarity < threshold, skip (keep separate for quality)

        Args:
            model: the model with MoE layers
            similarity_threshold: only tie if cos_sim > threshold (0.0 = tie all)

        Returns:
            List of (even_layer, odd_layer) pairs that were tied
        """
        # Find all MoE layers
        moe_layers = {}
        for name, module in model.named_modules():
            if hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
                parts = name.split('.')
                layer_idx = None
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                if layer_idx is not None:
                    moe_layers[layer_idx] = (name, module)

        if not moe_layers:
            print("  [ExpertTying] No MoE layers found")
            return []

        max_layer = max(moe_layers.keys())
        tied_pairs = []
        skipped_pairs = []
        saved_params = 0

        for even_idx in range(0, max_layer, 2):
            odd_idx = even_idx + 1
            if odd_idx not in moe_layers or even_idx not in moe_layers:
                continue

            _even_name, even_moe = moe_layers[even_idx]
            _odd_name, odd_moe = moe_layers[odd_idx]

            # Measure similarity between expert weights
            if similarity_threshold > 0:
                sims = []
                for i in range(min(len(even_moe.experts), len(odd_moe.experts))):
                    w_even = even_moe.experts[i].w1.weight.flatten()
                    w_odd = odd_moe.experts[i].w1.weight.flatten()
                    sim = torch.nn.functional.cosine_similarity(
                        w_even.unsqueeze(0), w_odd.unsqueeze(0), dim=-1
                    ).item()
                    sims.append(sim)
                avg_sim = sum(sims) / len(sims)

                if avg_sim < similarity_threshold:
                    skipped_pairs.append((even_idx, odd_idx, avg_sim))
                    continue

            # Count params being saved
            for expert in odd_moe.experts:
                for p in expert.parameters():
                    saved_params += p.numel()

            # Tie experts
            tied_experts = nn.ModuleList([
                even_moe.experts[i] for i in range(len(even_moe.experts))
            ])
            odd_moe.experts = tied_experts

            if hasattr(even_moe, 'shared') and hasattr(odd_moe, 'shared'):
                odd_moe.shared = even_moe.shared

            tied_pairs.append((even_idx, odd_idx))

        saved_mb = saved_params * 2 / 1024 / 1024
        print(f"  [ExpertTying] Tied {len(tied_pairs)} layer pairs: {tied_pairs}")
        if skipped_pairs:
            print(f"    Skipped {len(skipped_pairs)} pairs (low similarity): "
                  f"{[(p[0], p[1], f'{p[2]:.3f}') for p in skipped_pairs]}")
        print(f"    Saved ~{saved_mb:.1f} MB VRAM (expert weights)")

        return tied_pairs

    def _tie_state_dict(self, state: dict[str, torch.Tensor],
                        n_layers: int) -> tuple[list[tuple[int, int]], int]:
        """Tie expert weights in a state dict (for checkpoint saving)."""
        tied_pairs = []
        saved_bytes = 0

        for even_idx in range(0, n_layers, 2):
            odd_idx = even_idx + 1
            if odd_idx >= n_layers:
                continue

            # Find all expert keys for the odd layer
            odd_prefix = f"blocks.{odd_idx}.ffn.experts."
            even_prefix = f"blocks.{even_idx}.ffn.experts."
            shared_odd = f"blocks.{odd_idx}.ffn.shared."
            shared_even = f"blocks.{even_idx}.ffn.shared."

            keys_to_remove = []
            for key in list(state.keys()):
                if key.startswith(odd_prefix):
                    # Replace with even layer's key
                    even_key = key.replace(odd_prefix, even_prefix, 1)
                    if even_key in state:
                        saved_bytes += state[key].numel() * state[key].element_size()
                        keys_to_remove.append(key)
                elif key.startswith(shared_odd):
                    even_key = key.replace(shared_odd, shared_even, 1)
                    if even_key in state:
                        saved_bytes += state[key].numel() * state[key].element_size()
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del state[key]

            tied_pairs.append((even_idx, odd_idx))

        return tied_pairs, saved_bytes

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Not supported — untying requires retraining to recover unique weights."""
        return KeyResult(
            success=False,
            error="ExpertTyingKey.reverse not supported: "
                  "cannot recover unique weights after tying.",
        )

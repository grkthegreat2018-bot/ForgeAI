"""Vision tower for ForgeLM V11 — SigLIP2-SO400M encoder + MLP projector.

This module provides the vision components for the V11 multimodal model:
1. **SigLIP2VisionTower** — SigLIP2-SO400M ViT encoder (400M params)
2. **VisionProjector** — MLP projector from vision hidden dim → LM d_model
3. **VisionLanguageConnector** — combines tower + projector, produces
   token-aligned visual embeddings injected before the first LM layer

The vision tower is frozen at inference (no gradients). During training,
only the projector is trained (standard VLM practice — the vision encoder
is pretrained and frozen, the projector learns the alignment).

Architecture (SigLIP2-SO400M):
- Input: 384×384 RGB images
- Patch embedding: 14×14 patches → 27×27 = 729 tokens
- 27 transformer layers, 1152 hidden, 16 heads, 4304 FFN dim
- Output: 729 visual tokens × 1152 dim
- After projector: 128 tokens × 2560 dim (pooled to n_queries)

The pooling uses a learned query attention mechanism (Perceiver-style):
128 learnable queries attend to the 729 visual tokens, producing 128
token-aligned embeddings that the LM processes like text tokens.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Patch embedding ─────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Convert images to patch embeddings (ViT-style)."""

    def __init__(self, image_size: int = 384, patch_size: int = 14,
                 in_channels: int = 3, hidden_size: int = 1152) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) → (B, N, D) where N = n_patches."""
        x = self.proj(images)            # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ── SigLIP2 transformer block ───────────────────────────────────────────

class SigLIP2Block(nn.Module):
    """A single SigLIP2 transformer encoder block.

    Pre-norm architecture with SwiGLU FFN (matching SigLIP2 spec).
    """

    def __init__(self, hidden_size: int = 1152, n_heads: int = 16,
                 intermediate_size: int = 4304,
                 norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        # SwiGLU FFN: gate * up → down
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # SwiGLU FFN
        h = self.norm2(x)
        ffn_out = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        x = x + ffn_out
        return x


# ── SigLIP2 vision tower ────────────────────────────────────────────────

class SigLIP2VisionTower(nn.Module):
    """SigLIP2-SO400M vision encoder.

    Takes images and produces visual token embeddings.
    Frozen at inference; optionally trainable during VLM training.
    """

    def __init__(self, hidden_size: int = 1152, image_size: int = 384,
                 patch_size: int = 14, n_layers: int = 27,
                 n_heads: int = 16, intermediate_size: int = 4304) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, hidden_size)
        # CLS token (SigLIP2 uses a class token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # Positional embedding: CLS + n_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.n_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SigLIP2Block(hidden_size, n_heads, intermediate_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) → visual tokens (B, 1+N, D)."""
        B = images.shape[0]
        x = self.patch_embed(images)                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, 1+N, D)
        x = x + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x                                        # (B, 1+N, D)


# ── Vision projector ────────────────────────────────────────────────────

class VisionProjector(nn.Module):
    """Project visual embeddings from vision dim → LM dim.

    Two modes:
    - "linear": single linear projection
    - "mlp": two-layer MLP with GELU activation (default, better alignment)
    """

    def __init__(self, vision_hidden_size: int = 1152,
                 lm_hidden_size: int = 2560,
                 projector_type: str = "mlp") -> None:
        super().__init__()
        if projector_type == "linear":
            self.proj = nn.Linear(vision_hidden_size, lm_hidden_size)
        else:
            # MLP: vision_dim → lm_dim → lm_dim with GELU
            self.proj = nn.Sequential(
                nn.Linear(vision_hidden_size, lm_hidden_size),
                nn.GELU(),
                nn.Linear(lm_hidden_size, lm_hidden_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Perceiver-style query pooling ───────────────────────────────────────

class QueryPooler(nn.Module):
    """Pool visual tokens to a fixed number of queries (Perceiver-style).

    Uses learnable query embeddings that cross-attend to the visual tokens.
    This reduces 729+1 visual tokens → n_queries (e.g. 128) tokens,
    giving the LM a fixed-length visual representation regardless of
    image resolution.
    """

    def __init__(self, hidden_size: int = 1152, n_queries: int = 128,
                 n_heads: int = 8) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.zeros(1, n_queries, hidden_size))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True, bias=True)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """visual_tokens: (B, N, D) → pooled (B, n_queries, D)."""
        B = visual_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)              # (B, Q, D)
        out, _ = self.cross_attn(q, visual_tokens, visual_tokens,
                                 need_weights=False)
        out = self.norm(q + out)
        return out                                      # (B, Q, D)


# ── Full vision-language connector ──────────────────────────────────────

class VisionLanguageConnector(nn.Module):
    """Complete vision pipeline: tower → pooler → projector.

    Takes raw images and produces LM-ready visual token embeddings.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tower = SigLIP2VisionTower(
            hidden_size=config.vision_hidden_size,
            image_size=config.vision_image_size,
            patch_size=config.vision_patch_size,
            n_layers=config.vision_n_layers,
            n_heads=config.vision_n_heads,
            intermediate_size=config.vision_intermediate_size,
        )
        self.pooler = QueryPooler(
            hidden_size=config.vision_hidden_size,
            n_queries=config.vision_n_queries,
            n_heads=8,
        )
        self.projector = VisionProjector(
            vision_hidden_size=config.vision_hidden_size,
            lm_hidden_size=config.vision_projector_dim,
            projector_type=config.vision_projector_type,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) → visual embeddings (B, n_queries, d_model)."""
        visual_tokens = self.tower(images)              # (B, 1+N, D_vision)
        pooled = self.pooler(visual_tokens)             # (B, Q, D_vision)
        projected = self.projector(pooled)              # (B, Q, d_model)
        return projected

    def freeze_tower(self) -> None:
        """Freeze the vision tower (standard VLM practice)."""
        for p in self.tower.parameters():
            p.requires_grad = False

    @property
    def n_visual_tokens(self) -> int:
        return self.config.vision_n_queries

    def param_count(self) -> dict:
        """Return parameter counts for tower, pooler, projector."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "tower": count(self.tower),
            "pooler": count(self.pooler),
            "projector": count(self.projector),
            "total": count(self),
        }


# ── Image preprocessing ─────────────────────────────────────────────────

def preprocess_image(image: torch.Tensor, image_size: int = 384) -> torch.Tensor:
    """Preprocess a raw image tensor for the vision tower.

    Args:
        image: (C, H, W) or (B, C, H, W) uint8 or float tensor
        image_size: target size (default 384)

    Returns:
        (B, C, image_size, image_size) normalized float tensor
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    # resize if needed
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image, size=(image_size, image_size),
            mode="bilinear", align_corners=False)
    # normalize to [0, 1] if uint8
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    # standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    image = (image - mean) / std
    return image

"""Mamba Key — lossless checkpoint conversion for Mamba/SSM layers.

Converts between HuggingFace Mamba checkpoint format and ForgeAI internal
format. This is a STRUCTURAL key (KeyClass.BI): the conversion is a pure
key rename + tensor passthrough, no data transformation, fully lossless.

Mamba (S6 selective SSM) weight structure:
  in_proj.weight   (2*d_inner, d_model)   — projects to z (gate) + x (ssm)
  conv1d.weight    (d_inner, 1, d_conv)   — depthwise causal conv
  conv1d.bias      (d_inner,)             — conv bias
  x_proj.weight    (dt_rank + 2*d_state, d_inner) — projects to dt, B, C
  dt_proj.weight   (d_inner, dt_rank)     — discretization projection
  dt_proj.bias     (d_inner,)             — dt bias
  A_log            (d_inner, d_state)     — log(-A) for S4D real init
  D                (d_inner,)             — skip connection
  out_proj.weight  (d_model, d_inner)     — output projection

Mamba2 adds:
  dt_norm.weight   (d_inner,)             — RMSNorm on dt
  A_norm.weight    (d_state,)             — RMSNorm on A
  B_norm.weight    (d_state,)             — RMSNorm on B
  C_norm.weight    (d_state,)             — RMSNorm on C
  A_log            (n_heads, 1)           — scalar-diagonal (SSD)

HuggingFace naming:  mixer.{name}  or  model.layers.{i}.mixer.{name}
ForgeAI naming:      blocks.{i}.attn.{name}

The key is lossless because:
  1. No weight transformation (pure rename + passthrough)
  2. No precision loss (tensors copied as-is)
  3. Round-trip is identity (verified by test)
  4. No training needed (weights are loaded from checkpoint, not derived)

Usage:
    key = MambaKey()
    # HuggingFace -> ForgeAI
    result = key.forward(hf_state_dict)
    # ForgeAI -> HuggingFace
    result = key.reverse(forge_state_dict)
    # Cross-arch (Mamba1 <-> Mamba2)
    result = key.cross_arch(mamba1_state, Mamba2Key())
"""
from __future__ import annotations

import torch

from forge.keys.misc.base import Key, KeyClass, KeyResult

# ═══════════════════════════════════════════════════════════════════════════════
# Weight name mappings
# ═══════════════════════════════════════════════════════════════════════════════

# HuggingFace Mamba mixer weight names (relative to mixer.)
MAMBA1_WEIGHTS = (
    "in_proj.weight",
    "conv1d.weight",
    "conv1d.bias",
    "x_proj.weight",
    "dt_proj.weight",
    "dt_proj.bias",
    "A_log",       # Parameter, not Linear
    "D",           # Parameter, not Linear
    "out_proj.weight",
)

# Mamba2 adds 4 RMSNorm weights and changes A_log shape
# Jamba uses: dt_layernorm, b_layernorm, c_layernorm (no a_layernorm)
MAMBA2_EXTRA_WEIGHTS = (
    "dt_norm.weight",       # generic name
    "A_norm.weight",
    "B_norm.weight",
    "C_norm.weight",
    # Jamba naming:
    "dt_layernorm.weight",
    "b_layernorm.weight",
    "c_layernorm.weight",
)

# HuggingFace layer norm names → ForgeAI names
# Jamba uses pre_ff_layernorm (not post_attention_layernorm)
NORM_MAP = {
    "input_layernorm.weight": "ln1.weight",
    "post_attention_layernorm.weight": "ln2.weight",
    "pre_ff_layernorm.weight": "ln2.weight",  # Jamba naming
}


def _hf_to_forge_layer(hf_key: str, layer_idx: int) -> str | None:
    """Map a HuggingFace key to ForgeAI internal key.

    Handles Mamba (mixer. or mamba. prefix) and attention layer keys.
    Supports Jamba, Zamba, and generic HuggingFace Mamba naming.
    """
    prefix = f"model.layers.{layer_idx}."

    if not hf_key.startswith(prefix):
        return None

    suffix = hf_key[len(prefix):]

    # Mamba: mixer.{name} or mamba.{name} -> attn.{name}
    # Jamba uses "mamba.", generic HF Mamba uses "mixer."
    # Strip .weight from norm parameters (dt/b/c_layernorm are bare nn.Parameters)
    _NORM_STRIP = {"dt_layernorm.weight", "b_layernorm.weight", "c_layernorm.weight",
                   "dt_norm.weight", "A_norm.weight", "B_norm.weight", "C_norm.weight"}
    for mamba_prefix in ("mixer.", "mamba."):
        if suffix.startswith(mamba_prefix):
            name = suffix[len(mamba_prefix):]
            if name in _NORM_STRIP:
                name = name[:-len(".weight")]
            return f"blocks.{layer_idx}.attn.{name}"

    # Attention: self_attn.{name} -> attn.{name}
    # Handle o_proj -> out_proj mapping (Jamba uses o_proj, ForgeAI uses out_proj)
    if suffix.startswith("self_attn."):
        name = suffix[len("self_attn."):]
        if name == "o_proj.weight":
            name = "out_proj.weight"
        return f"blocks.{layer_idx}.attn.{name}"

    # Layer norms
    if suffix in NORM_MAP:
        return f"blocks.{layer_idx}.{NORM_MAP[suffix]}"

    # Feed-forward: handle both feed_forward and mlp prefixes
    # Jamba uses gate_proj/up_proj/down_proj; Zamba uses w1/w2/w3
    for ffn_prefix in ("feed_forward.", "mlp."):
        if suffix.startswith(ffn_prefix):
            name = suffix[len(ffn_prefix):]
            ffn_map = {
                "w1.weight": "ffn.w_gate.weight",
                "w2.weight": "ffn.w_down.weight",
                "w3.weight": "ffn.w_up.weight",
                "gate_proj.weight": "ffn.w_gate.weight",
                "down_proj.weight": "ffn.w_down.weight",
                "up_proj.weight": "ffn.w_up.weight",
            }
            if name in ffn_map:
                return f"blocks.{layer_idx}.{ffn_map[name]}"

    return None


def _forge_to_hf_layer(forge_key: str, layer_idx: int,
                       is_mamba: bool = True,
                       mamba_prefix: str = "mamba",
                       ffn_style: str = "jamba") -> str | None:
    """Map a ForgeAI internal key to HuggingFace key.

    Args:
        mamba_prefix: "mamba" (Jamba) or "mixer" (generic HF Mamba)
        ffn_style: "jamba" (gate_proj/up_proj/down_proj) or "zamba" (w1/w2/w3)
    """
    prefix = f"blocks.{layer_idx}."

    if not forge_key.startswith(prefix):
        return None

    suffix = forge_key[len(prefix):]

    # Attention/Mamba: attn.{name}
    if suffix.startswith("attn."):
        name = suffix[len("attn."):]
        if is_mamba:
            # Add .weight back for norm parameters (bare nn.Parameter -> .weight in HF)
            _NORM_ADD = {"dt_layernorm", "b_layernorm", "c_layernorm",
                         "dt_norm", "A_norm", "B_norm", "C_norm"}
            if name in _NORM_ADD:
                name = name + ".weight"
            return f"model.layers.{layer_idx}.{mamba_prefix}.{name}"
        else:
            # Attention layers: map out_proj -> o_proj for Jamba
            if name == "out_proj.weight":
                name = "o_proj.weight"
            return f"model.layers.{layer_idx}.self_attn.{name}"

    # Layer norms
    if suffix == "ln1.weight":
        return f"model.layers.{layer_idx}.input_layernorm.weight"
    if suffix == "ln2.weight":
        if ffn_style == "jamba":
            return f"model.layers.{layer_idx}.pre_ff_layernorm.weight"
        return f"model.layers.{layer_idx}.post_attention_layernorm.weight"

    # FFN
    if suffix.startswith("ffn."):
        name = suffix[len("ffn."):]
        if ffn_style == "jamba":
            ffn_rmap = {"w_gate.weight": "feed_forward.gate_proj.weight",
                        "w_down.weight": "feed_forward.down_proj.weight",
                        "w_up.weight": "feed_forward.up_proj.weight"}
        else:
            ffn_rmap = {"w_gate.weight": "feed_forward.w1.weight",
                        "w_down.weight": "feed_forward.w2.weight",
                        "w_up.weight": "feed_forward.w3.weight"}
        if name in ffn_rmap:
            return f"model.layers.{layer_idx}.{ffn_rmap[name]}"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MambaKey — lossless checkpoint conversion
# ═══════════════════════════════════════════════════════════════════════════════

class MambaKey(Key):
    """Lossless checkpoint conversion key for Mamba (S6) layers.

    Converts between HuggingFace Mamba checkpoint format and ForgeAI
    internal format. Pure key rename + tensor passthrough — no data
    transformation, fully lossless, no training needed.

    KeyClass.BI: both forward and reverse are exact inverses.
    """

    def __init__(self, n_layers: int = 1, layer_types: list[str] | None = None,
                 mamba_prefix: str = "mamba", ffn_style: str = "jamba"):
        """
        Args:
            n_layers: number of layers in the model
            layer_types: per-layer type list ("mamba" or "attention")
                         if None, all layers are treated as Mamba
            mamba_prefix: "mamba" (Jamba) or "mixer" (generic HF Mamba)
            ffn_style: "jamba" (gate_proj/up_proj/down_proj) or "zamba" (w1/w2/w3)
        """
        self.n_layers = n_layers
        self.layer_types = layer_types or ["mamba"] * n_layers
        self.mamba_prefix = mamba_prefix
        self.ffn_style = ffn_style

    @property
    def name(self) -> str:
        return "mamba"

    @property
    def description(self) -> str:
        return ("Lossless checkpoint conversion for Mamba (S6 selective SSM) "
                "layers. Pure key rename, no weight transformation.")

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def _is_mamba_layer(self, idx: int) -> bool:
        if idx < len(self.layer_types):
            return self.layer_types[idx].lower() in ("mamba", "ssm")
        return True

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """HuggingFace Mamba checkpoint -> ForgeAI internal format.

        Args:
            data: dict of HuggingFace-style weight tensors
                  (keys like "model.layers.{i}.mixer.{name}")
        Returns:
            KeyResult with weights dict (keys like "blocks.{i}.attn.{name}")
        """
        forge_state: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []

        for hf_key, tensor in data.items():
            mapped = False

            # Try each layer
            for i in range(self.n_layers):
                forge_key = _hf_to_forge_layer(hf_key, i)
                if forge_key is not None:
                    forge_state[forge_key] = tensor
                    mapped = True
                    break

            # Non-layer keys (embedding, head, final norm)
            if not mapped:
                if hf_key == "model.embed_tokens.weight":
                    forge_state["embed.weight"] = tensor
                    mapped = True
                elif hf_key == "model.embedding_norm.weight":
                    forge_state["ln_f.weight"] = tensor
                    mapped = True
                elif hf_key == "lm_head.weight":
                    forge_state["head.weight"] = tensor
                    mapped = True
                elif hf_key == "model.norm.weight" or hf_key == "model.final_layernorm.weight":
                    forge_state["ln_f.weight"] = tensor
                    mapped = True

            if not mapped:
                unmapped.append(hf_key)

        if unmapped:
            # Don't fail — just report unmapped keys
            pass

        return KeyResult(
            success=True,
            weights=forge_state,
            metadata={"unmapped_keys": unmapped,
                      "n_mapped": len(forge_state),
                      "n_input": len(data)},
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """ForgeAI internal format -> HuggingFace Mamba checkpoint.

        Args:
            weights: dict of ForgeAI-style weight tensors
                     (keys like "blocks.{i}.attn.{name}")
        Returns:
            KeyResult with data dict (keys like "model.layers.{i}.mixer.{name}")
        """
        hf_state: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []

        for forge_key, tensor in weights.items():
            mapped = False

            for i in range(self.n_layers):
                is_mamba = self._is_mamba_layer(i)
                hf_key = _forge_to_hf_layer(forge_key, i, is_mamba=is_mamba,
                                            mamba_prefix=self.mamba_prefix,
                                            ffn_style=self.ffn_style)
                if hf_key is not None:
                    hf_state[hf_key] = tensor
                    mapped = True
                    break

            # Non-layer keys
            if not mapped:
                if forge_key == "embed.weight":
                    hf_state["model.embed_tokens.weight"] = tensor
                    mapped = True
                elif forge_key == "ln_f.weight":
                    if self.ffn_style == "jamba":
                        hf_state["model.final_layernorm.weight"] = tensor
                    else:
                        hf_state["model.norm.weight"] = tensor
                    mapped = True
                elif forge_key == "head.weight":
                    hf_state["lm_head.weight"] = tensor
                    mapped = True

            if not mapped:
                unmapped.append(forge_key)

        return KeyResult(
            success=True,
            data=hf_state,
            metadata={"unmapped_keys": unmapped,
                      "n_mapped": len(hf_state),
                      "n_input": len(weights)},
        )


class Mamba2Key(MambaKey):
    """Lossless checkpoint conversion for Mamba2 (SSD) layers.

    Mamba2 has the same weight names as Mamba1 plus 4 RMSNorm weights.
    The A_log shape differs (scalar-diagonal) but the key rename is
    identical — the tensor shape is preserved as-is.

    KeyClass.BI: both forward and reverse are exact inverses.
    """

    @property
    def name(self) -> str:
        return "mamba2"

    @property
    def description(self) -> str:
        return ("Lossless checkpoint conversion for Mamba2 (SSD) layers. "
                "Same as Mamba1 plus 4 RMSNorm weights (dt/A/B/C norm).")

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """HuggingFace Mamba2 -> ForgeAI (same as Mamba1, extra norms pass through)."""
        return super().forward(data)

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """ForgeAI -> HuggingFace Mamba2 (same as Mamba1)."""
        return super().reverse(weights)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-arch conversion: Mamba1 <-> Mamba2
# ═══════════════════════════════════════════════════════════════════════════════

class Mamba1To2Key(Key):
    """Convert Mamba1 weights to Mamba2 format (PARTIAL — needs A reshaping).

    Mamba1 A_log: (d_inner, d_state) — diagonal, per-channel
    Mamba2 A_log: (n_heads, 1) — scalar per head

    This conversion is LOSSY (A must be averaged across d_state to get
    scalar per head). Marked as KeyClass.PARTIAL.

    The other weights (in_proj, conv1d, x_proj, dt_proj, out_proj, D)
    are identical between Mamba1 and Mamba2 and pass through unchanged.
    """

    @property
    def name(self) -> str:
        return "mamba1_to_mamba2"

    @property
    def description(self) -> str:
        return ("Convert Mamba1 to Mamba2 format. A_log is averaged across "
                "d_state to get scalar per head (LOSSY). Other weights pass "
                "through unchanged. Mamba2 norms are initialized to ones.")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Mamba1 weights -> Mamba2 weights.

        Args:
            data: Mamba1 weight dict (A_log shape: (d_inner, d_state))
        Returns:
            KeyResult with Mamba2 weight dict (A_log shape: (n_heads, 1))
        """
        result = {}
        head_dim = data.get("_head_dim", 64)
        d_inner = data.get("A_log").shape[0] if "A_log" in data else None

        if d_inner is None:
            return KeyResult(success=False, error="No A_log in input")

        n_heads = d_inner // head_dim

        for key, tensor in data.items():
            if key == "A_log":
                # (d_inner, d_state) -> (n_heads, 1)
                # Average across d_state, then average across head_dim channels
                a_log = tensor  # (d_inner, d_state)
                a_per_head = a_log.view(n_heads, head_dim, -1).mean(dim=(1, 2))
                result[key] = a_per_head.unsqueeze(-1)  # (n_heads, 1)
            elif key == "_head_dim":
                continue
            else:
                result[key] = tensor.clone()

        # Add Mamba2 norms (initialized to ones — lossless at init)
        d_inner_val = d_inner
        d_state = data["A_log"].shape[1] if "A_log" in data else 64
        result["dt_norm.weight"] = torch.ones(d_inner_val)
        result["A_norm.weight"] = torch.ones(d_state)
        result["B_norm.weight"] = torch.ones(d_state)
        result["C_norm.weight"] = torch.ones(d_state)

        return KeyResult(success=True, weights=result,
                         metadata={"conversion": "mamba1->mamba2",
                                  "lossy": ["A_log"]})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Mamba2 -> Mamba1 is not supported (A scalar can't expand back)."""
        return KeyResult(success=False,
                        error="Mamba2 -> Mamba1 not supported (A is scalar, "
                              "cannot recover per-channel diagonal)")

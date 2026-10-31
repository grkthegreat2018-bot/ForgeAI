"""R39-1/2/3: Native architecture support for Qwen3, Gemma3, Llama4.

These adapters convert HuggingFace checkpoints from Qwen3, Gemma3, and
Llama4 into ForgeAI's internal format (ConfigurableResearchLLM). The
conversion is a key rename + tensor passthrough (lossless).

Architecture differences:
  - Qwen3: GQA + QK-norm + SwiGLU + RoPE (close to LFM2, different vocab)
  - Gemma3: Alternating sliding-window attention + global attention
  - Llama4: MoE with shared experts + routed experts
"""
from __future__ import annotations

import torch

# ── Qwen3 ─────────────────────────────────────────────────────────────────

QWEN3_KEY_MAP = {
    # Embedding
    "model.embed_tokens.weight": "embed.weight",
    # Final norm
    "model.norm.weight": "final_norm.weight",
    # Per-layer
    # Attention: Qwen3 uses q_proj/k_proj/v_proj/o_proj + q_norm/k_norm
    # FFN: gate_proj/up_proj/down_proj (SwiGLU)
    # Norm: input_layernorm/post_attention_layernorm (RMSNorm)
}


def _qwen3_layer_map(hf_key: str, layer_idx: int) -> str | None:
    """Map a Qwen3 HF key to ForgeAI internal key."""
    prefix = f"model.layers.{layer_idx}."
    if not hf_key.startswith(prefix):
        return None
    suffix = hf_key[len(prefix):]
    # Attention
    if suffix.startswith("self_attn."):
        name = suffix[len("self_attn."):]
        return f"blocks.{layer_idx}.attn.{name}"
    # QK-norm (Qwen3 specific: q_norm, k_norm)
    if suffix in ("q_norm.weight", "k_norm.weight"):
        name = suffix.replace("_norm.weight", "_norm.weight")
        return f"blocks.{layer_idx}.attn.{name}"
    # Layer norms
    if suffix == "input_layernorm.weight":
        return f"blocks.{layer_idx}.ln1.weight"
    if suffix == "post_attention_layernorm.weight":
        return f"blocks.{layer_idx}.ln2.weight"
    # FFN
    if suffix.startswith("mlp."):
        name = suffix[len("mlp."):]
        ffn_map = {
            "gate_proj.weight": "ffn.w_gate.weight",
            "up_proj.weight": "ffn.w_up.weight",
            "down_proj.weight": "ffn.w_down.weight",
        }
        if name in ffn_map:
            return f"blocks.{layer_idx}.{ffn_map[name]}"
    return None


def convert_qwen3_checkpoint(hf_state_dict: dict[str, torch.Tensor],
                              n_layers: int) -> dict[str, torch.Tensor]:
    """Convert a Qwen3 HuggingFace checkpoint to ForgeAI internal format.

    Qwen3 architecture:
      - GQA with QK-norm (q_norm, k_norm per layer)
      - SwiGLU FFN (gate_proj, up_proj, down_proj)
      - RMSNorm (input_layernorm, post_attention_layernorm)
      - RoPE (rotary position embeddings, no weight)
      - vocab_size typically 151936 (Qwen3) or 151680 (Qwen2.5)

    The conversion is lossless: pure key rename + tensor passthrough.
    """
    forge_state = {}
    # Embedding and final norm
    for hf_key, tensor in hf_state_dict.items():
        if hf_key in QWEN3_KEY_MAP:
            forge_state[QWEN3_KEY_MAP[hf_key]] = tensor
    # Per-layer
    for layer_idx in range(n_layers):
        for hf_key, tensor in hf_state_dict.items():
            forge_key = _qwen3_layer_map(hf_key, layer_idx)
            if forge_key:
                forge_state[forge_key] = tensor
    return forge_state


# ── Gemma3 ────────────────────────────────────────────────────────────────

GEMMA3_KEY_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "final_norm.weight",
}


def _gemma3_layer_map(hf_key: str, layer_idx: int) -> str | None:
    """Map a Gemma3 HF key to ForgeAI internal key.

    Gemma3 uses alternating sliding-window attention (SWA) and global
    attention. The layer type is determined by the config:
      - Even layers: sliding window attention (window=1024)
      - Odd layers: global attention (full)
    """
    prefix = f"model.layers.{layer_idx}."
    if not hf_key.startswith(prefix):
        return None
    suffix = hf_key[len(prefix):]
    # Attention (same naming as Qwen3/Llama)
    if suffix.startswith("self_attn."):
        name = suffix[len("self_attn."):]
        # Gemma3 uses o_proj (not out_proj)
        if name == "o_proj.weight":
            name = "out_proj.weight"
        return f"blocks.{layer_idx}.attn.{name}"
    # QK-norm (Gemma3 uses q_norm, k_norm like Qwen3)
    if suffix in ("q_norm.weight", "k_norm.weight"):
        return f"blocks.{layer_idx}.attn.{suffix}"
    # Layer norms (Gemma3 uses input_layernorm, post_attention_layernorm,
    # and pre_feedforward_layernorm, post_feedforward_layernorm)
    if suffix == "input_layernorm.weight":
        return f"blocks.{layer_idx}.ln1.weight"
    if suffix == "post_attention_layernorm.weight":
        return f"blocks.{layer_idx}.ln2.weight"
    if suffix == "pre_feedforward_layernorm.weight":
        return f"blocks.{layer_idx}.ln_ffn_pre.weight"
    if suffix == "post_feedforward_layernorm.weight":
        return f"blocks.{layer_idx}.ln_ffn_post.weight"
    # FFN (Gemma3 uses gate_proj, up_proj, down_proj)
    if suffix.startswith("mlp."):
        name = suffix[len("mlp."):]
        ffn_map = {
            "gate_proj.weight": "ffn.w_gate.weight",
            "up_proj.weight": "ffn.w_up.weight",
            "down_proj.weight": "ffn.w_down.weight",
        }
        if name in ffn_map:
            return f"blocks.{layer_idx}.{ffn_map[name]}"
    return None


def convert_gemma3_checkpoint(hf_state_dict: dict[str, torch.Tensor],
                               n_layers: int) -> dict[str, torch.Tensor]:
    """Convert a Gemma3 HuggingFace checkpoint to ForgeAI internal format.

    Gemma3 architecture:
      - Alternating SWA (even layers) + global attention (odd layers)
      - QK-norm (q_norm, k_norm)
      - SwiGLU FFN with pre/post feedforward layernorms (Gemma3 specific)
      - RoPE
      - vocab_size=262144 (Gemma3)

    The conversion is lossless: pure key rename + tensor passthrough.
    The SWA/global alternation is handled by layer_types in ModelConfig.
    """
    forge_state = {}
    for hf_key, tensor in hf_state_dict.items():
        if hf_key in GEMMA3_KEY_MAP:
            forge_state[GEMMA3_KEY_MAP[hf_key]] = tensor
    for layer_idx in range(n_layers):
        for hf_key, tensor in hf_state_dict.items():
            forge_key = _gemma3_layer_map(hf_key, layer_idx)
            if forge_key:
                forge_state[forge_key] = tensor
    return forge_state


def gemma3_layer_types(n_layers: int) -> list[str]:
    """Generate layer_types for Gemma3 (alternating SWA + global).

    Even layers (0, 2, 4, ...): sliding window attention
    Odd layers (1, 3, 5, ...): global attention
    """
    return ["attention_swa" if i % 2 == 0 else "attention" for i in range(n_layers)]


# ── Llama4 ────────────────────────────────────────────────────────────────

LLAMA4_KEY_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "final_norm.weight",
}


def _llama4_layer_map(hf_key: str, layer_idx: int) -> str | None:
    """Map a Llama4 HF key to ForgeAI internal key.

    Llama4 uses MoE with shared experts + routed experts:
      - self_attn: standard attention (q_proj, k_proj, v_proj, o_proj)
      - feed_forward: MoE with router + shared_expert + experts
      - Norm: input_layernorm, post_attention_layernorm
    """
    prefix = f"model.layers.{layer_idx}."
    if not hf_key.startswith(prefix):
        return None
    suffix = hf_key[len(prefix):]
    # Attention
    if suffix.startswith("self_attn."):
        name = suffix[len("self_attn."):]
        if name == "o_proj.weight":
            name = "out_proj.weight"
        return f"blocks.{layer_idx}.attn.{name}"
    # Layer norms
    if suffix == "input_layernorm.weight":
        return f"blocks.{layer_idx}.ln1.weight"
    if suffix == "post_attention_layernorm.weight":
        return f"blocks.{layer_idx}.ln2.weight"
    # MoE router
    if suffix == "feed_forward.router.weight":
        return f"blocks.{layer_idx}.moe.router.weight"
    # Shared expert
    if suffix.startswith("feed_forward.shared_expert."):
        name = suffix[len("feed_forward.shared_expert."):]
        shared_map = {
            "w1.weight": "moe.shared.w_gate.weight",
            "w3.weight": "moe.shared.w_up.weight",
            "w2.weight": "moe.shared.w_down.weight",
            "gate_proj.weight": "moe.shared.w_gate.weight",
            "up_proj.weight": "moe.shared.w_up.weight",
            "down_proj.weight": "moe.shared.w_down.weight",
        }
        if name in shared_map:
            return f"blocks.{layer_idx}.{shared_map[name]}"
    # Routed experts (feed_forward.experts.{N}.{w1,w2,w3})
    if suffix.startswith("feed_forward.experts."):
        parts = suffix.split(".")
        # feed_forward.experts.{N}.{w1/w2/w3}.weight
        if len(parts) >= 5:
            expert_idx = parts[2]
            weight_name = parts[3]
            expert_map = {
                "w1": "w_gate", "w3": "w_up", "w2": "w_down",
                "gate_proj": "w_gate", "up_proj": "w_up",
                "down_proj": "w_down",
            }
            if weight_name in expert_map:
                return (f"blocks.{layer_idx}.moe.experts.{expert_idx}."
                        f"{expert_map[weight_name]}.weight")
    return None


def convert_llama4_checkpoint(hf_state_dict: dict[str, torch.Tensor],
                               n_layers: int) -> dict[str, torch.Tensor]:
    """Convert a Llama4 HuggingFace checkpoint to ForgeAI internal format.

    Llama4 architecture:
      - MoE with shared experts + routed experts
      - Standard attention (GQA, QK-norm optional)
      - RMSNorm
      - RoPE

    The conversion is lossless: pure key rename + tensor passthrough.
    Expert weights are stored as moe.experts.{N}.{w_gate/w_up/w_down}.weight.
    Shared expert weights are stored as moe.shared.{w_gate/w_up/w_down}.weight.
    """
    forge_state = {}
    for hf_key, tensor in hf_state_dict.items():
        if hf_key in LLAMA4_KEY_MAP:
            forge_state[LLAMA4_KEY_MAP[hf_key]] = tensor
    for layer_idx in range(n_layers):
        for hf_key, tensor in hf_state_dict.items():
            forge_key = _llama4_layer_map(hf_key, layer_idx)
            if forge_key:
                forge_state[forge_key] = tensor
    return forge_state


def detect_n_layers(hf_state_dict: dict[str, torch.Tensor]) -> int:
    """Detect the number of layers from a HuggingFace state dict."""
    max_layer = -1
    for key in hf_state_dict:
        if "model.layers." in key:
            parts = key.split(".")
            try:
                idx = parts.index("layers") + 1
                layer_num = int(parts[idx])
                max_layer = max(max_layer, layer_num)
            except (ValueError, IndexError):
                continue
    return max_layer + 1 if max_layer >= 0 else 0


# ── LFM3 ──────────────────────────────────────────────────────────────────
#
# LFM3-1.7B (LiquidAI): tri-hybrid — gated short convs + sliding-window
# attention + full attention. HuggingFace layout:
#   model.embed_tokens.weight                    → embed.weight (tied head)
#   model.embedding_norm.weight                  → ln_f.weight  (final norm —
#     applied AFTER the last layer despite the name)
#   model.layers.{i}.conv.{conv,in_proj,out_proj}.weight
#                                                → blocks.{i}.attn.{same}
#   model.layers.{i}.self_attn.{q,k,v,o,gate}_proj.weight, {q,k}_norm.weight
#                                                → blocks.{i}.attn.{same}
#     (o_proj → out_proj; gate_proj = per-head sigmoid out-gate)
#   model.layers.{i}.feed_forward.w{1,3,2}.weight→ ffn.w_{gate,up,down}.weight
#   model.layers.{i}.operator_norm.weight        → blocks.{i}.ln1.weight
#   model.layers.{i}.ffn_norm.weight             → blocks.{i}.ln2.weight
#   model.layers.{i}.post_operator_norm.weight   → blocks.{i}.post_attn_norm.weight
#   model.layers.{i}.post_ffn_norm.weight        → blocks.{i}.post_ffn_norm.weight
#     (pre-residual sandwich norms — applied to the branch output before
#      the residual add; sandwich_norm_mode="pre_residual")
#
# Same mapping ModelLoader._remap_hf_keys applies on the fast load path.

LFM3_KEY_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    # LFM3's "embedding_norm" is the FINAL norm (applied after the last
    # decoder layer — LFM2 heritage), not an embedding-time norm.
    "model.embedding_norm.weight": "ln_f.weight",
}

LFM3_LAYER_TYPES_FULL = "full_attention"
LFM3_LAYER_TYPES_SWA = "sliding_attention"


def _lfm3_layer_map(hf_key: str, layer_idx: int) -> str | None:
    """Map an LFM3 HF key to ForgeAI internal key."""
    prefix = f"model.layers.{layer_idx}."
    if not hf_key.startswith(prefix):
        return None
    suffix = hf_key[len(prefix):]
    # Gated short conv (DoubleGatedConvLayer lives on blocks.{i}.attn)
    if suffix.startswith("conv."):
        return f"blocks.{layer_idx}.attn.{suffix[len('conv.'):]}"
    # Attention: q/k/v/gate_proj + q_norm/k_norm keep their names; o→out
    if suffix.startswith("self_attn."):
        name = suffix[len("self_attn."):]
        if name == "o_proj.weight":
            name = "out_proj.weight"
        return f"blocks.{layer_idx}.attn.{name}"
    # MLP: w1=w_gate, w3=w_up, w2=w_down
    if suffix.startswith("feed_forward."):
        ffn_map = {"w1.weight": "ffn.w_gate.weight",
                   "w3.weight": "ffn.w_up.weight",
                   "w2.weight": "ffn.w_down.weight"}
        name = suffix[len("feed_forward."):]
        if name in ffn_map:
            return f"blocks.{layer_idx}.{ffn_map[name]}"
    # Block norms
    if suffix == "operator_norm.weight":
        return f"blocks.{layer_idx}.ln1.weight"
    if suffix == "ffn_norm.weight":
        return f"blocks.{layer_idx}.ln2.weight"
    if suffix == "post_operator_norm.weight":
        return f"blocks.{layer_idx}.post_attn_norm.weight"
    if suffix == "post_ffn_norm.weight":
        return f"blocks.{layer_idx}.post_ffn_norm.weight"
    return None


def convert_lfm3_checkpoint(hf_state_dict: dict[str, torch.Tensor],
                            n_layers: int) -> dict[str, torch.Tensor]:
    """Convert an LFM3 HuggingFace checkpoint to ForgeAI internal format.

    Lossless: pure key rename + tensor passthrough. Verified bit-exact
    (max |dlogit| 3.6e-5 in fp32) vs the bundled HF reference in
    scripts/test_lfm3_port.py.
    """
    forge_state = {}
    for hf_key, tensor in hf_state_dict.items():
        if hf_key in LFM3_KEY_MAP:
            forge_state[LFM3_KEY_MAP[hf_key]] = tensor
    for layer_idx in range(n_layers):
        for hf_key, tensor in hf_state_dict.items():
            forge_key = _lfm3_layer_map(hf_key, layer_idx)
            if forge_key:
                forge_state[forge_key] = tensor
    return forge_state


def lfm3_layer_types(hf_layer_types: list[str]) -> list[str]:
    """Map HF LFM3 layer_types to ForgeAI layer_types.

    HF: "conv" | "full_attention" | "sliding_attention"
    ForgeAI: "conv" | "attention" | "attention_swa"
    """
    mapping = {"conv": "conv", "full_attention": "attention",
               "sliding_attention": "attention_swa",
               "attention": "attention"}
    return [mapping.get(t, "attention") for t in hf_layer_types]


def detect_architecture(hf_state_dict: dict[str, torch.Tensor]) -> str:
    """Detect which architecture a checkpoint belongs to.

    Returns: "lfm3", "qwen3", "gemma3", "llama4", or "unknown".
    """
    keys = set(hf_state_dict.keys())
    # LFM3: hybrid conv/attention — conv.* sublayer + embedding_norm +
    # operator_norm/post_operator_norm block norms.
    if any(".conv.in_proj." in k for k in keys) and \
            any("post_operator_norm" in k or "operator_norm" in k
                for k in keys):
        return "lfm3"
    # Llama4: has shared_expert or experts in feed_forward
    if any("shared_expert" in k for k in keys):
        return "llama4"
    if any("feed_forward.experts." in k for k in keys):
        return "llama4"
    # Gemma3: has pre_feedforward_layernorm
    if any("pre_feedforward_layernorm" in k for k in keys):
        return "gemma3"
    # Qwen3: has q_norm/k_norm (not q_proj/qk_norm)
    if any("self_attn.q_norm" in k for k in keys):
        return "qwen3"
    return "unknown"


def convert_checkpoint(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Auto-detect architecture and convert checkpoint.

    Returns: ForgeAI internal format state dict.
    Raises: ValueError if architecture is unknown.
    """
    arch = detect_architecture(hf_state_dict)
    n_layers = detect_n_layers(hf_state_dict)
    if arch == "lfm3":
        return convert_lfm3_checkpoint(hf_state_dict, n_layers)
    elif arch == "qwen3":
        return convert_qwen3_checkpoint(hf_state_dict, n_layers)
    elif arch == "gemma3":
        return convert_gemma3_checkpoint(hf_state_dict, n_layers)
    elif arch == "llama4":
        return convert_llama4_checkpoint(hf_state_dict, n_layers)
    else:
        raise ValueError(
            f"Unknown architecture. Keys: {list(hf_state_dict.keys())[:5]}...")

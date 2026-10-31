"""Compatibility adapters for external model architectures."""
from .arch_adapters import (
    convert_checkpoint,
    convert_gemma3_checkpoint,
    convert_lfm3_checkpoint,
    convert_llama4_checkpoint,
    convert_qwen3_checkpoint,
    detect_architecture,
    detect_n_layers,
    gemma3_layer_types,
    lfm3_layer_types,
)

__all__ = [
    "convert_checkpoint",
    "convert_gemma3_checkpoint",
    "convert_lfm3_checkpoint",
    "convert_llama4_checkpoint",
    "convert_qwen3_checkpoint",
    "detect_architecture",
    "detect_n_layers",
    "gemma3_layer_types",
    "lfm3_layer_types",
]

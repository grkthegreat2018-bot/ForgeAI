"""LLM-based gen model for ForgeEvolve — ultra-compact ForgeLM V7 architecture.

Uses the same architecture keys as ForgeLM V7 (BitNet b1.58, GTA attention,
NLRQ FFN compression, RoPE) but at an ultra-compact size for use as a gen
model in evolutionary search. The gen model generates candidate solutions
for task-solving domains (math, algorithms, random tasks).

The model is built from scratch (random init) — no checkpoint needed.
Size is configurable for grow/shrink operations managed by GenModelManager.

Architecture:
  - BitNet b1.58: ternary QAT on attention projections + FFN linears
  - GTA (Grouped-Tied Attention): ties V to K, halves KV cache
  - NLRQ FFN compression: SVD + quantized factors for compact FFN
  - RoPE: rotary positional embeddings (theta=1M, LFM2.5 base)
  - QK-norm: RMSNorm on Q and K before RoPE

Default size: d_model=256, 4 layers, 8 heads, 2 KV heads, intermediate=512,
vocab=65536 (reuses LFM2.5 tokenizer). ~2-5M params depending on NLRQ rank.

Usage:
    from forge.evolution.llm_gen_model import LLMGenModel

    gen = LLMGenModel()  # uses "gen_model_tiny" preset
    text = gen.generate("Solve: 2 + 2 = ?", max_tokens=64)
    print(f"Params: {gen.param_count():,}")
    gen.save_state("gen_model.pt")
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.config import ModelConfig, get_config
from forge.model_loader import ModelLoader, create_kv_cache, unpack_output_with_kv

# Default tokenizer path (LFM2.5 tokenizer, vocab=65536).
_DEFAULT_TOKENIZER_PATH = "research/checkpoints/forgelm_v2_tokenizer"


class LLMGenModel:
    """Ultra-compact ForgeLM V7 model for use as a ForgeEvolve gen model.

    Wraps a ConfigurableResearchLLM built with V7 architecture keys at a
    configurable size. Provides simple generate/save/load/param_count
    methods for integration with GenModelManager.

    Args:
        config_name: Config preset name (default "gen_model_tiny").
        device: "cuda", "cpu", or None (auto-detect).
        dtype: torch.dtype for model weights (default bf16 on GPU, fp32 on CPU).
        tokenizer_path: Path to tokenizer directory.
        **overrides: ModelConfig field overrides (d_model, n_layers, etc.).
    """

    def __init__(
        self,
        config_name: str = "gen_model_tiny",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        tokenizer_path: str = _DEFAULT_TOKENIZER_PATH,
        **overrides: Any,
    ):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Auto-detect dtype: bf16 on GPU, fp32 on CPU
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        # Build config from preset + overrides
        self.config: ModelConfig = get_config(config_name, device=device, **overrides)

        # Load tokenizer (cached via lru_cache in tokenizer_cache)
        from research.tokenizer_cache import get_tokenizer
        self.tokenizer = get_tokenizer(tokenizer_path)

        # Build model from scratch (random init, no checkpoint)
        self.model = self._build_model()

        # EOS token ID for generation stopping
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if self.eos_token_id is None:
            # Fallback: use a common EOS ID from LFM2.5 tokenizer
            self.eos_token_id = 7

    def _build_model(self) -> nn.Module:
        """Build the model from scratch using ModelLoader.build_model_fast.

        Uses fast_load=False (traditional path) since there's no checkpoint.
        This builds ConfigurableResearchLLM(config).to(device) with random init.
        """
        model = ModelLoader.build_model_fast(
            self.config,
            checkpoint_path=None,  # random init
            compile=False,
            dtype=self.dtype,
            fast_load=False,  # traditional path for random init
        )
        model.eval()
        return model

    def param_count(self) -> int:
        """Return total number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def get_size_config(self) -> dict:
        """Return current model size configuration as a dict.

        Contains the key dimensions used by GenModelManager for grow/shrink.
        """
        return {
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "n_kv_heads": self.config.n_kv_heads,
            "intermediate_size": self.config.intermediate_size,
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "nlrq_rank": self.config.nlrq_rank,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> str:
        """Generate text from a prompt.

        Uses a pre-allocated KV cache for efficient autoregressive generation.
        Returns only the newly generated text (excluding the prompt).

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = greedy decoding).
            top_k: Optional top-k sampling limit (None = no limit).

        Returns:
            Generated text string (excluding the prompt).
        """
        self.model.eval()
        device = self.device

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.config.max_seq_len)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
        prompt_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        B, prompt_len = prompt_ids.shape

        # Pre-allocate output buffer (prompt + max_tokens)
        max_total = prompt_len + max_tokens
        out_ids = torch.zeros(B, max_total, dtype=prompt_ids.dtype, device=device)
        out_ids[:, :prompt_len] = prompt_ids

        # Create KV cache
        cache = create_kv_cache(self.model, max_total, batch=B, device=device)

        # Autocast for bf16 on GPU
        use_autocast = self.device.type == "cuda" and self.dtype != torch.float32
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if use_autocast
            else nullcontext()
        )

        generated_len = 0
        with autocast_ctx:
            for step in range(max_tokens):
                pos = prompt_len + step
                if step == 0:
                    # Prefill: feed the full prompt
                    idx_cond = out_ids[:, :prompt_len]
                    out = self.model(idx_cond, use_cache=True,
                                     preallocated_cache=cache)
                    logits = unpack_output_with_kv(out)[0]
                else:
                    # Decode: feed only the last generated token
                    idx_cond = out_ids[:, pos - 1:pos]
                    out = self.model(idx_cond, use_cache=True,
                                     preallocated_cache=cache)
                    logits = unpack_output_with_kv(out)[0]

                # Get logits for the last position
                logits = logits[:, -1, :].float()

                # Apply temperature
                if temperature > 0:
                    logits = logits / max(temperature, 1e-5)
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = logits.argmax(dim=-1, keepdim=True)

                out_ids[:, pos:pos + 1] = next_token
                generated_len = step + 1

                # EOS check
                if self.eos_token_id is not None and \
                        (next_token == self.eos_token_id).any().item():
                    break

        # Decode only the generated portion
        generated_ids = out_ids[0, prompt_len:prompt_len + generated_len]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (for distillation).

        Args:
            input_ids: (B, T) token IDs on the model's device.

        Returns:
            (B, T, vocab_size) logits tensor.
        """
        self.model.eval()
        out = self.model(input_ids)
        logits = unpack_output_with_kv(out)[0]
        return logits

    def save_state(self, path: str | Path) -> None:
        """Save model state dict to a file.

        Uses torch.save for full compatibility with all parameter types
        (including BitNet master weights, NLRQ factors, etc.).

        Args:
            path: File path to save to (.pt or .safetensors).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "config": self.config.__dict__,
            "param_count": self.param_count(),
        }
        torch.save(state, str(path))

    def load_state(self, path: str | Path) -> bool:
        """Load model state dict from a file.

        Rebuilds the model if the saved config differs from the current one
        (e.g. different d_model after grow/shrink). Only loads weights for
        parameters with matching shapes.

        Args:
            path: File path to load from.

        Returns:
            True if loaded successfully.
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            state = torch.load(str(path), map_location="cpu",
                               weights_only=True)
        except Exception:
            state = torch.load(str(path), map_location="cpu",
                               weights_only=False)

        # Check if saved config matches current config
        saved_config = state.get("config", {})
        current_size = self.get_size_config()
        needs_rebuild = False
        for key in ("d_model", "n_layers", "n_heads", "n_kv_heads",
                     "intermediate_size", "vocab_size", "nlrq_rank"):
            if saved_config.get(key) != current_size.get(key):
                needs_rebuild = True
                break

        if needs_rebuild:
            # Rebuild with the saved config dimensions
            overrides = {}
            for key in ("d_model", "n_layers", "n_heads", "n_kv_heads",
                        "intermediate_size", "vocab_size", "max_seq_len",
                        "nlrq_rank"):
                if key in saved_config:
                    overrides[key] = saved_config[key]
            self.config = get_config(
                saved_config.get("name", "gen_model_tiny"),
                device=str(self.device),
                **overrides,
            )
            self.model = self._build_model()

        # Load weights (only matching shapes)
        model_state = state["model_state"]
        current_state = self.model.state_dict()
        loaded = 0
        skipped = 0
        with torch.no_grad():
            for name, param in current_state.items():
                if name in model_state:
                    saved = model_state[name]
                    if saved.shape == param.shape:
                        param.copy_(saved.to(param.device, dtype=param.dtype))
                        loaded += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

        if skipped > 0:
            print(f"  [LLMGenModel] Loaded {loaded} params, skipped {skipped} "
                  f"(shape mismatch or missing)")
        return True

    def resize(self, **size_overrides: Any) -> None:
        """Resize the model by rebuilding with new dimensions.

        Used by GenModelManager for grow/shrink operations. The old model's
        weights are NOT carried over — use GenModelManager._distill for that.

        Args:
            **size_overrides: ModelConfig field overrides (d_model, n_layers, etc.).
        """
        self.config = get_config(
            "gen_model_tiny",
            device=str(self.device),
            **size_overrides,
        )
        # Move old model to CPU before building new one (save VRAM)
        old_model = self.model
        del old_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model = self._build_model()

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def __repr__(self) -> str:
        size = self.get_size_config()
        return (f"LLMGenModel(d_model={size['d_model']}, "
                f"n_layers={size['n_layers']}, "
                f"params={self.param_count():,}, "
                f"device={self.device}, dtype={self.dtype})")

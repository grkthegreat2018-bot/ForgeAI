"""BitNet-everywhere conversion + manual LoRA adapters + sequential freeze/unfreeze.

Manual LoRA works with BitNetLinear (PEFT can't inject into nn.Module subclasses).
Sequential freeze/unfreeze trains layers in phases — full forward pass preserves
cross-layer connections (MHC, AttnRes), only gradients are scoped.

Validated in .devin/test_bitnet_native.py on real V3 1.2B:
  BitNet-everywhere + LoRA(r=32) + Muon-SF + 3-way grad mixup = 2.39x vs AdamW
  VRAM: 6.32GB (53% of 12GB RTX 5070)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from forge.quant.protocol import QuantizedLinearMixin, is_quantized_linear

# ── NF4 (NormalFloat 4-bit) QLoRA ────────────────────────────────────────
# R32-1: Proper NF4 QLoRA path for standard nn.Linear models.
# Paper: QLoRA (Dettmers et al. NeurIPS 2023) — NF4 is the information-
# theoretically optimal 4-bit normal float distribution for weights that
# follow a normal distribution. Unlike the existing IRIFP4 QLoRA which
# only works on IRIFP4Linear layers, NF4 QLoRA works on ANY nn.Linear.
# The NF4 grid is fixed (not learned) and has 16 levels optimized for
# the normal distribution of pre-trained LLM weights.
#
# VRAM: 4-bit base weights (frozen) + bf16 LoRA adapters (trainable).
# For V10 (1.2B): ~600MB base (4-bit) + ~50MB LoRA (r=32) = ~650MB total.

_NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986832, -0.5250730514526367, -0.39491748809814453,
    -0.28444141149520874, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170048713684, 0.7229568362236023, 1.0,
], dtype=torch.float32)


class NF4Linear(QuantizedLinearMixin):
    """NF4 quantized linear layer with optional LoRA adapter (QLoRA).

    Stores weights in NF4 (4-bit normal float) with per-group absmax scale.
    Dequantizes to bf16 for computation. LoRA adapter is optional and trainable.

    Source: QLoRA paper (Dettmers et al. NeurIPS 2023, arXiv 2305.14314)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.group_size = group_size
        # Packed: 2 values per int8 byte → (out, in // 2)
        self.weight_packed = nn.Parameter(
            torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8),
            requires_grad=False)
        # Per-group scales: (out, n_groups)
        n_groups = (in_features + group_size - 1) // group_size
        self.weight_scales = nn.Parameter(
            torch.zeros(out_features, n_groups, dtype=torch.float16),
            requires_grad=False)
        self._cached_weight: torch.Tensor | None = None
        self.lora_adapter: nn.Module | None = None

    @torch.no_grad()
    def load_from_weight(self, w: torch.Tensor):
        """Quantize a float weight tensor into NF4 packed format."""
        assert w.shape == (self.out_features, self.in_features)
        device = w.device
        gs = self.group_size
        n_groups = (self.in_features + gs - 1) // gs
        # Pad to multiple of group_size
        pad = n_groups * gs - self.in_features
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        # Reshape to (out, n_groups, gs)
        w_grouped = w.reshape(self.out_features, n_groups, gs)
        # Per-group absmax scale
        scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        # Normalize to [-1, 1]
        w_norm = w_grouped / scales
        # Quantize to NF4: find nearest level
        levels = _NF4_LEVELS.to(device)
        # (out, n_groups, gs, 1) vs (16, 1) → argmin
        w_exp = w_norm.unsqueeze(-1)  # (out, n_groups, gs, 1)
        dist = (w_exp - levels.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
        idx = dist.argmin(dim=-1)  # (out, n_groups, gs)
        # Pack: 2 indices per byte (each 0-15 → 4 bits)
        idx_flat = idx.reshape(self.out_features, -1).to(torch.uint8)
        packed = idx_flat[:, ::2] | (idx_flat[:, 1::2] << 4)
        self.weight_packed.data = packed.to('cpu')
        self.weight_scales.data = scales.squeeze(-1).to(torch.float16).to('cpu')
        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        if self._cached_weight is not None:
            return self._cached_weight.to(dtype)
        packed = self.weight_packed.data
        scales = self.weight_scales.data.to(torch.float32)
        # Unpack: low 4 bits = even idx, high 4 bits = odd idx
        low = (packed & 0x0F).to(torch.long)
        high = (packed >> 4).to(torch.long)
        idx = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)
        # Trim to in_features
        idx = idx[:, :self.in_features]
        # Lookup NF4 levels
        levels = _NF4_LEVELS.to(idx.device)
        w_norm = levels[idx]  # (out, in)
        # Apply per-group scales
        n_groups = scales.shape[1]
        w_grouped = w_norm.reshape(self.out_features, n_groups, -1)
        w_scaled = w_grouped * scales.unsqueeze(-1)
        w = w_scaled.reshape(self.out_features, -1)[:, :self.in_features]
        if cache:
            self._cached_weight = w.to(torch.bfloat16)
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, cache=True)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = torch.nn.functional.linear(x, w, bias)
        if self.lora_adapter is not None:
            out = out + self.lora_adapter(x)
        return out

    @torch.no_grad()
    def merge_lora(self) -> bool:
        """Merge LoRA adapter into NF4 weights (QLoRA merge)."""
        if self.lora_adapter is None:
            return False
        lora = self.lora_adapter
        w = self._dequantize_weight(torch.float32, cache=False)
        delta = lora.scale * (lora.lora_B @ lora.lora_A)
        w = w + delta.to(torch.float32)
        self.load_from_weight(w)
        self.lora_adapter = None
        self._cached_weight = None
        return True

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"group_size={self.group_size}, nf4=True")


def convert_to_nf4_qlora(model: nn.Module, group_size: int = 64,
                         target_modules: list[str] | None = None,
                         min_size: int = 64) -> tuple[int, int]:
    """Replace nn.Linear with NF4Linear (frozen 4-bit base for QLoRA).

    Does NOT add LoRA adapters — call add_lora_adapters() after this.
    Returns (n_converted, n_skipped).
    """
    n_conv = 0
    n_skip = 0

    def convert(module, prefix=""):
        nonlocal n_conv, n_skip
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            is_target = True
            if target_modules is not None:
                is_target = any(t in full_name for t in target_modules)
            if isinstance(child, nn.Linear) and is_target:
                if child.in_features < min_size or child.out_features < min_size:
                    n_skip += 1
                    convert(child, full_name)
                    continue
                nf4 = NF4Linear(child.in_features, child.out_features,
                                bias=child.bias is not None, group_size=group_size)
                nf4.load_from_weight(child.weight.data)
                if child.bias is not None:
                    nf4.bias.data.copy_(child.bias.data)
                nf4 = nf4.to(child.weight.device)
                setattr(module, name, nf4)
                n_conv += 1
            else:
                convert(child, full_name)

    convert(model)
    return n_conv, n_skip


# ── BitNet-everywhere ────────────────────────────────────────────────────

def convert_to_bitnet_everywhere(model: nn.Module) -> tuple[int, int]:
    """Replace all nn.Linear with BitNetLinear (ternary QAT).

    Preserves existing BitNetLinear layers. Converts attention projections,
    head, AttnRes, MoD routers — everything — to BitNet b1.58.

    Returns (n_converted, n_already_bitnet).
    """
    from forge.keys.quantization.bitnet_b158_key import BitNetLinear

    n_converted = 0
    n_already = 0

    def convert(module, prefix=""):
        nonlocal n_converted, n_already
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, BitNetLinear):
                n_already += 1
                continue
            if isinstance(child, nn.Linear):
                new_layer = BitNetLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quantize=True,
                    learned_scale=True,
                )
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                new_layer = new_layer.to(child.weight.device)
                setattr(module, name, new_layer)
                n_converted += 1
            else:
                convert(child, full_name)

    convert(model)
    return n_converted, n_already


# ── Manual LoRA (BitNet-compatible) ──────────────────────────────────────

class LoRAAdapter(nn.Module):
    """LoRA adapter: y += scale * (x @ A^T @ B^T).

    A: (rank, in_features) — kaiming init
    B: (out_features, rank) — zero init (LoRA starts as no-op)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 32, alpha: int = 64):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def add_lora_adapters(
    model: nn.Module,
    rank: int = 32,
    alpha: int = 64,
    target_modules: list[str] | None = None,
    min_size: int = 64,
) -> tuple[int, list[nn.Parameter]]:
    """Add LoRA adapters to target layers (works with BitNetLinear + quantized linears).

    Quantized linear detection uses :func:`forge.quant.protocol.is_quantized_linear`
    (protocol-based + ``getattr`` fallback) so any quantized linear class that
    inherits :class:`QuantizedLinearMixin` is recognized automatically.

    Args:
        model: The model to add LoRA to.
        rank: LoRA rank.
        alpha: LoRA alpha (scale = alpha / rank).
        target_modules: List of module name substrings to target (e.g. ["q_proj", "w_gate"]).
            None = target all Linear/BitNetLinear/quantized linears with in_features >= min_size.
        min_size: Skip layers smaller than this (e.g. MoD routers with 1 output).

    Returns (n_adapters, lora_params_list).
    """
    n_adapters = 0
    lora_params = []

    def find_and_add(module, prefix=""):
        nonlocal n_adapters
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            has_weight = isinstance(getattr(child, 'weight', None), nn.Parameter)
            has_dims = hasattr(child, 'in_features') and hasattr(child, 'out_features')
            is_quant = is_quantized_linear(child)

            # Valid target: has dims + (has Parameter weight OR is quantized linear)
            if has_dims and (has_weight or is_quant):
                is_target = True
                if target_modules is not None:
                    is_target = any(t in full_name for t in target_modules)
                if child.in_features < min_size or child.out_features < min_size:
                    is_target = False

                if is_target:
                    lora = LoRAAdapter(child.in_features, child.out_features, rank=rank, alpha=alpha)
                    # Use bfloat16 for LoRA params on quantized linears (base is bf16 dequantized)
                    lora_dtype = torch.bfloat16 if is_quant else (
                        child.weight.dtype if child.weight.dtype != torch.float32 else torch.bfloat16)
                    # ForgeQuantLinear uses dense_packed, others use weight_packed
                    # (Quamba2Linear stores weight_int4_packed; last resort is
                    # any parameter or buffer on the module).
                    if is_quant:
                        qdev = getattr(child, 'weight_packed', None)
                        if qdev is None:
                            qdev = getattr(child, 'dense_packed', None)
                        if qdev is None:
                            qdev = getattr(child, 'weight_int4_packed', None)
                        if qdev is None:
                            qdev = getattr(child, 'weight', None)
                        if qdev is None:
                            qdev = next(child.parameters(), None)
                        if qdev is None:
                            qdev = next(child.buffers(), None)
                        if qdev is None:
                            continue  # no tensor to infer device from — skip
                        lora = lora.to(qdev.device)
                    else:
                        lora = lora.to(child.weight.device)
                    lora = lora.to(lora_dtype)
                    child.lora_adapter = lora

                    # For quantized linears, forward() already checks for lora_adapter
                    # For BitNetLinear/nn.Linear, we need to wrap forward
                    if not is_quant:
                        orig_forward = child.forward
                        child._lora_orig_forward = orig_forward  # save for unload
                        def make_new_forward(orig_fwd, lora_mod):
                            def new_forward(x):
                                out = orig_fwd(x)
                                return out + lora_mod(x)
                            return new_forward
                        child.forward = make_new_forward(orig_forward, lora)

                    # Freeze base weights
                    if has_weight:
                        child.weight.requires_grad = False
                    if hasattr(child, 'qscale') and child.qscale is not None:
                        child.qscale.requires_grad = False
                    if hasattr(child, 'bias') and child.bias is not None:
                        if isinstance(child.bias, nn.Parameter):
                            child.bias.requires_grad = False

                    lora_params.extend([lora.lora_A, lora.lora_B])
                    n_adapters += 1
            find_and_add(child, full_name)

    find_and_add(model)
    return n_adapters, lora_params


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge LoRA adapters into base weights: W += scale * B @ A.

    For nn.Linear/BitNetLinear: directly adds delta to weight Parameter.
    For quantized linears (IRIFP4Linear/NF4Linear/ForgeQuantLinear/…):
    dequantizes → adds delta → re-quantizes via ``merge_lora()``.
    Detection uses :func:`forge.quant.protocol.is_quantized_linear` (protocol
    + ``getattr`` fallback) so new quantized linear classes are recognized
    automatically.
    Call before saving checkpoint so output is standalone (no LoRA dependency).
    Returns n_merged.
    """
    n_merged = 0
    for module in model.modules():
        if hasattr(module, 'lora_adapter') and isinstance(module.lora_adapter, LoRAAdapter):
            if is_quantized_linear(module):
                # QLoRA merge: dequant → merge → re-quantize
                if module.merge_lora():
                    n_merged += 1
                continue
            # Standard merge for nn.Linear / BitNetLinear
            lora = module.lora_adapter
            with torch.no_grad():
                # W += scale * B @ A  (out_features, in_features)
                delta = lora.scale * (lora.lora_B @ lora.lora_A)
                module.weight.data += delta.to(module.weight.dtype)
            # Remove adapter
            del module.lora_adapter
            n_merged += 1
    return n_merged


# ── Sequential freeze/unfreeze ───────────────────────────────────────────

def freeze_unfreeze_lora(
    model: nn.Module,
    active_layers: list[int] | None = None,
) -> None:
    """Freeze/unfreeze LoRA params by layer index.

    active_layers: list of layer indices to unfreeze. None = unfreeze all.
    Layers are identified by 'blocks.{i}.' in parameter names.
    """
    for n, p in model.named_parameters():
        if 'lora_A' not in n and 'lora_B' not in n:
            continue
        if active_layers is None:
            p.requires_grad = True
        else:
            p.requires_grad = any(f"blocks.{li}." in n or f".blocks.{li}." in n for li in active_layers)


def get_active_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all LoRA params that currently have requires_grad=True."""
    return [p for n, p in model.named_parameters()
            if ('lora_A' in n or 'lora_B' in n) and p.requires_grad]


def compute_phase_schedule(
    n_layers: int,
    n_phases: int,
    total_steps: int,
    final_finetune_steps: int = 0,
) -> list[tuple[int, int, list[int]]]:
    """Compute sequential freeze/unfreeze phase schedule.

    Args:
        n_layers: Total number of model layers.
        n_phases: Number of sequential phases.
        total_steps: Total training steps.
        final_finetune_steps: Steps at the end to fine-tune ALL layers (phase 5).
            0 = no final fine-tune.

    Returns list of (start_step, end_step, active_layers) tuples.
    """
    if final_finetune_steps > 0:
        seq_steps = total_steps - final_finetune_steps
    else:
        seq_steps = total_steps

    layers_per_phase = n_layers // n_phases
    steps_per_phase = seq_steps // n_phases

    schedule = []
    for phase in range(n_phases):
        start = phase * steps_per_phase
        end = (phase + 1) * steps_per_phase if phase < n_phases - 1 else seq_steps
        start_layer = phase * layers_per_phase
        end_layer = (phase + 1) * layers_per_phase if phase < n_phases - 1 else n_layers
        active = list(range(start_layer, end_layer))
        schedule.append((start, end, active))

    if final_finetune_steps > 0:
        schedule.append((seq_steps, total_steps, None))  # None = all layers

    return schedule


def get_active_layers_for_step(
    step: int,
    schedule: list[tuple[int, int, list[int]]],
) -> list[int] | None:
    """Get active layers for a given step from the phase schedule.

    Returns None if all layers should be active (final fine-tune).
    """
    for start, end, active in schedule:
        if start <= step < end:
            return active
    return None


# ── Muon-SF optimizer for LoRA params ────────────────────────────────────

def build_muon_sf_lora_opt(
    lora_params: list[nn.Parameter],
    lr_muon: float = 5e-3,
    lr_adam: float = 3e-4,
):
    """Build Muon-SF optimizer for LoRA params.

    Muon (Newton-Schulz) for 2D LoRA A/B matrices, ScheduleFree AdamW for any 1D params.
    Validated on V3 1.2B: 2.39x better than AdamW with LoRA.
    """
    from muon import SingleDeviceMuonWithAuxAdam, muon_update
    from schedulefree import AdamWScheduleFree

    muon_p = [p for p in lora_params if p.ndim == 2]
    adam_p = [p for p in lora_params if p.ndim != 2]

    class _MuonSFLoRA(SingleDeviceMuonWithAuxAdam):
        def __init__(self):
            self._sf = AdamWScheduleFree(adam_p, lr=lr_adam, betas=(0.9, 0.95), weight_decay=0.0) if adam_p else None
            super().__init__([{'params': muon_p, 'lr': lr_muon, 'momentum': 0.95, 'weight_decay': 0.0, 'use_muon': True}])
            if self._sf:
                self._sf.train()

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            if self._sf:
                self._sf.step()

        def zero_grad(self, set_to_none=True):
            super().zero_grad(set_to_none=set_to_none)
            if self._sf:
                self._sf.zero_grad(set_to_none=set_to_none)

        def train(self):
            if self._sf:
                self._sf.train()

        def eval(self):
            if self._sf:
                self._sf.eval()

    return _MuonSFLoRA()

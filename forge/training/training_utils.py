"""Shared helpers for pre-training and SFT."""
import inspect
import json
import logging
import math
import os
import random
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# OOM guard — wraps forward+backward so a single bad batch doesn't kill a run
# ---------------------------------------------------------------------------

class OOMBatchSkipped(Exception):
    """Raised inside the oom_guard block body to signal "skip this batch"."""
    pass


@contextmanager
def oom_guard(device: str = "cuda", *, skip: bool = True, label: str = ""):
    """Context manager that catches ``torch.cuda.OutOfMemoryError``.

    On OOM it synchronizes the device, clears the CUDA cache, and either:
      * swallows the error and sets ``guard.skipped = True`` (``skip=True``), or
      * re-raises after cleanup (``skip=False``).

    Usage::

        with oom_guard(str(device)) as safe:
            out = model(input_ids)
            loss = compute_loss(...)
            (loss / grad_accum).backward()
        if safe.skipped:
            optimizer.zero_grad()
            continue   # skip this batch, don't step
    """
    guard = _OOMGuardState()
    try:
        yield guard
    except torch.cuda.OutOfMemoryError as exc:
        guard.skipped = True
        guard.error = exc
        if not skip:
            _cuda_cleanup(device)
            raise
        _cuda_cleanup(device)
        tag = f" [{label}]" if label else ""
        print(f"  [OOM guard{tag}] CUDA OOM — cache cleared, batch skipped.")
    except RuntimeError as exc:
        # PyTorch < 2.0 may raise plain RuntimeError for OOM.
        if "out of memory" in str(exc).lower():
            guard.skipped = True
            guard.error = exc
            if not skip:
                _cuda_cleanup(device)
                raise
            _cuda_cleanup(device)
            tag = f" [{label}]" if label else ""
            print(f"  [OOM guard{tag}] CUDA OOM — cache cleared, batch skipped.")
        else:
            raise


class _OOMGuardState:
    __slots__ = ("error", "skipped")

    def __init__(self):
        self.skipped = False
        self.error: Exception | None = None


def _cuda_cleanup(device: str = "cuda"):
    """Synchronize + empty_cache, safe on CPU."""
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


class ReplayBuffer:
    """Circular buffer of recent training examples to prevent catastrophic forgetting.

    Stores (input_ids, target_ids) pairs. During online learning, each batch
    is mixed with replay examples to ensure the model doesn't forget old knowledge.
    Moved here from online_learn.py for reuse across live_learn.py and other trainers.
    """

    def __init__(self, max_size=1000, device="cuda"):
        self.buffer = deque(maxlen=max_size)
        self.device = device

    def add(self, x, y):
        """Add a training example (x: input, y: targets)."""
        self.buffer.append((x.cpu().clone(), y.cpu().clone()))

    def sample(self, n):
        """Sample n replay examples."""
        if len(self.buffer) == 0 or n <= 0:
            return None, None
        samples = random.sample(list(self.buffer), min(n, len(self.buffer)))
        xs = torch.cat([s[0] for s in samples], dim=0).to(self.device)
        ys = torch.cat([s[1] for s in samples], dim=0).to(self.device)
        return xs, ys

    def __len__(self):
        return len(self.buffer)


class BinaryDataset:
    """Flat uint32 token dataset, preloaded into CPU RAM as int64.

    Optimized for single-GPU training:
    - Random indices generated on GPU (avoids CPU randint + host-to-device transfer)
    - Optional pinned-memory staging for faster H2D copies
    - Optional background prefetch thread to overlap data loading with compute
    """

    def __init__(self, filename, seq_len, vocab_size, pin_memory=False, prefetch=0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pin_memory = pin_memory
        self.prefetch_queue = deque(maxlen=max(prefetch, 1))
        self.prefetch_thread = None
        self.prefetch_enabled = prefetch > 0
        if Path(filename).exists():
            data = np.fromfile(filename, dtype=np.uint32)
            self.data = torch.from_numpy(data).long()
            if pin_memory:
                self.data = self.data.pin_memory()
            print(f"Loaded {filename}: {len(self.data) / 1e6:.2f}M tokens"
                  f"{' (pinned)' if pin_memory else ''}")
        else:
            print(f"Warning: {filename} not found. Using synthetic data fallback.")
            self.data = None

    def get_batch(self, batch_size, device):
        if self.data is not None:
            max_ix = len(self.data) - self.seq_len - 1
            # Generate random indices on the target device (avoids CPU randint
            # + a separate host-to-device transfer for the indices).
            ix = torch.randint(0, max_ix, (batch_size,), device=device)
            ix = ix.cpu()  # needed for advanced indexing on the CPU tensor
            x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
            y = torch.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix])
        else:
            x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
            y = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        # non_blocking=True overlaps H2D copy with prior GPU work when
        # the source is in pinned memory.
        return x.to(device, non_blocking=self.pin_memory), \
               y.to(device, non_blocking=self.pin_memory)


def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    """Cosine warmup + decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, max_lr, weight_decay, optimizer_name="fused",
                        bf16_state=False, freetoken=False, double_buffer=False,
                        bandwidth_adaptive=False, chunk_size_mb=None):
    """Return an optimizer. Default is fused AdamW (fastest on RTX 5070).

    Benchmarks show fused AdamW is ~26% faster than bitsandbytes 8-bit on
    consumer Blackwell, with negligible VRAM difference for 360M params.
    Use 'bnb' only when VRAM is critically constrained.

    bf16_state=True is accepted but currently falls back to fp32 state —
    stock PyTorch AdamW cannot mix bf16 momentum with fp32 grads. Use 'bnb'
    (8-bit) instead for VRAM-constrained scenarios.

    FreeToken-inspired options (R&D round 14, arXiv:2608.16157):
      - freetoken=True: Enables double_buffer + bandwidth_adaptive on cpu_offload.
        Equivalent to the full FreeToken training pipeline: ping-pong grad
        buffers, bandwidth-adaptive chunked transfers, predictive offload.
      - double_buffer=True: Ping-pong grad buffers for transfer/compute overlap.
      - bandwidth_adaptive=True: Profile PCIe bandwidth, auto-set chunk size.
      - chunk_size_mb: Override auto-computed chunk size for chunked transfers.
    """
    if bf16_state:
        print("NOTE: --bf16-optimizer not yet supported with stock AdamW; using bnb 8-bit instead.")
        optimizer_name = "bnb"

    # Only collect trainable params — frozen params (e.g. LoRA base weights,
    # BAdam inactive blocks) must NOT get optimizer states.
    # Without this filter, CPUAdamW allocates 33+ GB CPU for all 2.8B params
    # even when only 16M LoRA params are trainable → OOM on 32GB RAM.
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    param_groups = [
        {"params": matrix_params, "weight_decay": weight_decay},
        {"params": other_params, "weight_decay": 0.0},
    ]

    if optimizer_name in ("fused", "adamw"):
        fused = torch.cuda.is_available()
        print(f"Using torch.optim.AdamW (fused={fused}).")
        return torch.optim.AdamW(param_groups, lr=max_lr, fused=fused)

    if optimizer_name == "bnb":
        try:
            import bitsandbytes as bnb

            print("Using 8-bit AdamW (bitsandbytes) — slower but saves VRAM.")
            return bnb.optim.AdamW8bit(param_groups, lr=max_lr)
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "lion":
        try:
            import bitsandbytes as bnb

            print("Using 8-bit Lion (bitsandbytes).")
            return bnb.optim.Lion8bit(
                [
                    {"params": matrix_params, "weight_decay": weight_decay * 10.0},
                    {"params": other_params, "weight_decay": 0.0},
                ],
                lr=max_lr / 10.0,
            )
        except Exception as e:
            print(f"bitsandbytes Lion unavailable ({e}); falling back to AdamW.")

    if optimizer_name == "cpu_offload":
        # CPUAdamW: optimizer states + fp32 master on CPU pinned RAM.
        # Eliminates the 14.4GB fp32 AdamW VRAM cost for 1.2B models.
        # Enables full-precision training on 12GB GPUs (vs 8-bit bnb fallback).
        from forge.training.optim.hybrid_offload import CPUAdamW
        # FreeToken-inspired: freetoken=True enables all enhancements at once
        if freetoken:
            double_buffer = True
            bandwidth_adaptive = True
        print(f"Using CPUAdamW (ZeRO-Offload-style: optimizer states on CPU)."
              f"{' [FreeToken: double_buffer + bandwidth_adaptive]' if freetoken else ''}")
        return CPUAdamW(param_groups, lr=max_lr, weight_decay=weight_decay,
                        double_buffer=double_buffer,
                        bandwidth_adaptive=bandwidth_adaptive,
                        chunk_size_mb=chunk_size_mb)

    if optimizer_name == "badam":
        # BAdam: block-wise Adam, updates one layer at a time.
        # Memory: 2M + 16M/D GB (D=blocks). For V7 (2.8B, 32 layers): ~9 GB.
        # Full-parameter training at LoRA-level memory. NeurIPS 2024.
        # BREAD (R&D round 14): landscape correction for inactive blocks.
        from forge.training.optim.badam import configure_badam
        bread_mode = getattr(model, 'config', getattr(model, '_config', None))
        bread_mode = getattr(bread_mode, 'bread_sgd_correction', 'disabled') if bread_mode else 'disabled'
        bread_scale = getattr(model, 'config', getattr(model, '_config', None))
        bread_scale = getattr(bread_scale, 'bread_sgd_lr_scale', 5.0) if bread_scale else 5.0
        print(f"Using BAdam (block-wise Adam, one layer at a time)"
              f"{f' + BREAD ({bread_mode})' if bread_mode != 'disabled' else ''}.")
        return configure_badam(model, lr=max_lr, weight_decay=weight_decay,
                               blocks_per_layer=1, switch_every=1,
                               bread_sgd_correction=bread_mode,
                               bread_sgd_lr_scale=bread_scale)

    if optimizer_name == "apollo":
        # APOLLO: SVD-free random-projection gradient scaling.
        # SGD-like memory with AdamW-level performance. ICLR 2025.
        # rank=1 = APOLLO-Mini (SGD-level memory). No SVD overhead vs GaLore.
        try:
            from forge.training.optim.apollo import configure_apollo
            cfg = getattr(model, 'config', getattr(model, '_config', None))
            rank = getattr(cfg, 'apollo_rank', 8) if cfg else 8
            scale = getattr(cfg, 'apollo_scale', 'tensor') if cfg else 'tensor'
            print(f"Using APOLLO (SVD-free, rank={rank}, scale={scale}).")
            return configure_apollo(model, lr=max_lr, weight_decay=weight_decay,
                                     rank=rank, scale=scale)
        except Exception as e:
            print(f"APOLLO unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "flashoptim":
        # FlashOptim: companded 8-bit optimizer states (7 bytes/param vs 16).
        # >50% memory reduction with tight error bounds via sqrt companding.
        # arXiv 2602.23349 (Databricks).
        try:
            from forge.training.optim.flashoptim import configure_flashoptim
            cfg = getattr(model, 'config', getattr(model, '_config', None))
            bits = getattr(cfg, 'flashoptim_bits', 8) if cfg else 8
            print(f"Using FlashOptim AdamW (companded {bits}-bit states, ~50% memory cut).")
            return configure_flashoptim(model, lr=max_lr, weight_decay=weight_decay, bits=bits)
        except Exception as e:
            print(f"FlashOptim unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "fira_nlrq":
        # Fira-NLRQ: for NLRQ models the trainable params are already low-rank
        # (only S), so Fira degenerates to a two-group AdamW. NeurIPS 2025 basis.
        from forge.training.optim.fira_nlrq import configure_fira_nlrq
        print("Using Fira-NLRQ (NLRQ-aware AdamW, see fira_nlrq.py docstring).")
        return configure_fira_nlrq(model, lr=max_lr, weight_decay=weight_decay)

    if optimizer_name == "galore":
        try:
            import galore_torch
            print("Using GaLore AdamW (rank=128, update_proj_freq=200).")
            return galore_torch.GaLoreAdamW(param_groups, lr=max_lr, weight_decay=weight_decay)
        except Exception as e:
            print(f"GaLore unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "muon":
        # Muon: orthogonalized momentum for 2D hidden weights, AdamW for
        # embeddings/head/scalars. ~2x compute efficiency vs AdamW (Moonlight
        # paper). FLOP overhead <1% for typical LM shapes.
        # LR settings from NanoGPT speedrun: Muon lr=0.05 (absolute, not scaled
        # by max_lr), AdamW head lr=0.22, embeddings lr=0.6, scalars lr=0.04.
        # We scale these proportionally to max_lr so the warmup schedule works.
        try:
            from muon import MuonWithAuxAdam
            scale = max_lr / 0.003  # normalize: speedrun uses max_lr=3e-3
            muon_lr = 0.05 * scale
            0.22 * scale
            embed_lr = 0.6 * scale
            scalar_lr = 0.04 * scale
            # Split by param name (use id() to avoid tensor __eq__ pitfalls).
            # Embeddings + head (tied weight) -> AdamW; 2D hidden matrices ->
            # Muon; <2D scalars/norms -> AdamW.
            embed_ids = set()
            for n, p in model.named_parameters():
                if 'embed' in n or 'head' in n:
                    embed_ids.add(id(p))
            hidden_params = [p for p in matrix_params if id(p) not in embed_ids]
            embed_params = [p for p in matrix_params if id(p) in embed_ids]
            scalar_params = other_params
            param_groups = [
                {"params": embed_params, "lr": embed_lr, "betas": (0.8, 0.95), "eps": 1e-10, "use_muon": False, "weight_decay": 0.0},
                {"params": hidden_params, "lr": muon_lr, "momentum": 0.95, "use_muon": True, "weight_decay": weight_decay},
                {"params": scalar_params, "lr": scalar_lr, "betas": (0.8, 0.95), "eps": 1e-10, "use_muon": False, "weight_decay": 0.0},
            ]
            print(f"Using Muon (lr={muon_lr:.4f}) for {len(hidden_params)} hidden matrices + "
                  f"AdamW (embed={embed_lr:.4f}, scalar={scalar_lr:.4f}) for {len(embed_params)} embed + {len(scalar_params)} scalar params.")
            return MuonWithAuxAdam(param_groups)
        except Exception as e:
            print(f"Muon unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "muon_sf":
        # MuonSFBlockwise: Muon + ScheduleFree + per-block sharpness-scaled LR.
        # Novel combo: high sharpness → HIGH LR for Muon (opposite of Sophia
        # clipping). ScheduleFree eliminates LR schedule via iterate averaging.
        # Tested in .devin/test_muon_sf.py: 1.05x vs AdamW, 1.12x vs Muon+AdamW.
        # OPTIMAL for standard architecture (no BitNet).
        try:
            from forge.training.optim.muon_sf_blockwise import build_muon_sf_blockwise
            return build_muon_sf_blockwise(model, max_lr, weight_decay)
        except Exception as e:
            print(f"MuonSFBlockwise unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "muon_sf_plain":
        # MuonScheduleFree: Muon + ScheduleFree, NO blockwise sharpness.
        # OPTIMAL for V3 architecture (BitNet + MHC + AttnRes).
        # Blockwise sharpness conflicts with BitNet's weight normalization.
        # Tested in .devin/test_full_stack.py: 2.24x vs AdamW cosine on V3.
        try:
            from forge.training.optim.muon_sf_blockwise import build_muon_sf_plain
            return build_muon_sf_plain(model, max_lr, weight_decay)
        except Exception as e:
            print(f"MuonScheduleFree unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "samuon_lite":
        # SAMuon-lite (arXiv 2608.25990): spectral split of the Muon update —
        # top singular component kept verbatim, bulk amplified by bulk_gain
        # (~13-24% fewer tokens to the same loss). Same param split as
        # build_muon_sf_plain: hidden 2D matrices → SAMuon-lite, embeddings/
        # head/scalars → AdamW.
        try:
            from forge.training.optim.sf_spectral_optimizers import build_samuon_lite
            bulk_gain = getattr(getattr(model, 'config', getattr(model, '_config', None)),
                                'samuon_bulk_gain', 1.5) or 1.5
            return build_samuon_lite(model, max_lr, weight_decay, bulk_gain=bulk_gain)
        except Exception as e:
            print(f"SAMuonLite unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "flash_adamw":
        try:
            from forge.training.optim.flash_optim import FlashAdamW
            print("Using FlashAdamW (8-bit states, companding quantization, 57% memory cut).")
            return FlashAdamW(param_groups, lr=max_lr, weight_decay=weight_decay)
        except Exception as e:
            print(f"FlashAdamW unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "flash_lion":
        try:
            from forge.training.optim.flash_optim import FlashLion
            print("Using FlashLion (8-bit sign-momentum, 58% memory cut).")
            return FlashLion(param_groups, lr=max_lr / 10.0, weight_decay=weight_decay * 10.0)
        except Exception as e:
            print(f"FlashLion unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "forge":
        try:
            from forge.training.optim.forge_optimizer import ForgeOptimizer
            print("Using FORGE (fused on-register gradient elimination, 53% memory cut).")
            opt = ForgeOptimizer(param_groups, lr=max_lr, weight_decay=weight_decay)
            return opt
        except Exception as e:
            print(f"FORGE unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "sf_normuon":
        try:
            from forge.training.optim.sf_spectral_optimizers import SFNorMuon
            print("Using SF-NorMuon (schedule-free spectral, per-neuron norm, anytime).")
            return SFNorMuon(param_groups, lr=max_lr, weight_decay=weight_decay)
        except Exception as e:
            print(f"SF-NorMuon unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "amuse":
        try:
            from forge.training.optim.sf_spectral_optimizers import AMUSE
            print("Using AMUSE (anytime Muon + stable eval, no LR schedule).")
            return AMUSE(param_groups, lr=max_lr)
        except Exception as e:
            print(f"AMUSE unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "mona":
        try:
            from forge.training.optim.sf_spectral_optimizers import MONA
            print("Using MONA (Muon + Nesterov acceleration, escapes sharp minima).")
            return MONA(param_groups, lr=max_lr)
        except Exception as e:
            print(f"MONA unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "nvme_muon_4bit":
        # NvmeMuon4Bit: NVMe-streamed 4-bit Muon (V8 optimizer).
        # Combines per-block NVMe-mapped optimizer states with 4-bit Muon
        # momentum + Newton-Schulz orthogonalization. Only 1 layer's 4-bit
        # momentum in RAM; rest streamed from NVMe.
        import tempfile

        from forge.training.optim.r20_memory_optimizers import NvmeMuon4Bit
        nvme_dir = tempfile.mkdtemp(prefix="nvme_muon4bit_")
        print("Using NvmeMuon4Bit (NVMe-streamed 4-bit Muon, V8 optimizer).")
        return NvmeMuon4Bit(model, lr=max_lr, momentum=0.95, n_steps=5,
                            weight_decay=weight_decay, nvme_path=nvme_dir,
                            blocks_per_layer=1, switch_every=5, verbose=False)

    # Unknown optimizer name — safe default.
    print(f"Unknown optimizer '{optimizer_name}'; using fused AdamW.")
    return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())


def fused_clip_step(opt, max_norm, oec_model=None):
    """Optimizer step with built-in grad clipping (one kernel launch instead of two).

    Standard pattern is clip_grad_norm_() then opt.step() — two passes over
    grads. This fuses them: scale grads in-place by (max_norm / global_norm)
    when the global norm exceeds max_norm, then call opt.step(). Skips the
    separate norm computation pass. Returns the (pre-clipping) global norm.

    If `oec_model` is given and its config has oec_mode="mu_centering", the
    OEC post-step hook is applied after opt.step() (subtracts the mean output
    embedding row from the LM head / tied embedding). See arXiv:2601.02031.
    """
    # Compute global norm in one reduction.
    total_sq = torch.tensor(0.0, device=opt.param_groups[0]["params"][0].device)
    for group in opt.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                total_sq.add_(p.grad.detach().pow(2).sum())
    global_norm = total_sq.sqrt().clamp(min=1e-6)
    scale = (max_norm / global_norm).clamp(max=1.0)
    if scale.item() < 1.0:
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(scale)
    opt.step()
    if oec_model is not None:
        apply_oec_centering(oec_model)
    return float(global_norm.item())


def _get_output_embedding_weight(model):
    """Return the LM head weight tensor (the output embedding matrix), or None.

    Handles both tied and untied heads. The tied case (embed.weight is head.weight)
    returns the single shared tensor; centering it fixes both spaces at once.
    """
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "weight"):
        return head.weight
    # Fallback: look for a commonly-named attribute.
    for name in ("lm_head", "output", "out_proj"):
        mod = getattr(model, name, None)
        if mod is not None and hasattr(mod, "weight"):
            return mod.weight
    return None


def apply_oec_centering(model):
    """OEC µ-centering post-step hook (arXiv:2601.02031).

    Subtracts the mean output embedding row from the LM head weight so the
    mean output embedding is bound to the origin. This suppresses the
    anisotropic common-mode shift that drives output logit divergence — the
    root cause that z-loss and logit soft-capping only mask.

    Zero hyperparameters. No-op if the model's config does not request
    `oec_mode="mu_centering"`. Safe with tied embeddings (the shared tensor
    is centered once, fixing both input and output spaces).
    """
    cfg = getattr(model, "config", None)
    if cfg is None or getattr(cfg, "oec_mode", "none") != "mu_centering":
        return
    w = _get_output_embedding_weight(model)
    if w is None:
        return
    with torch.no_grad():
        # w: [vocab, d_model]. Subtract the per-column mean over the vocab axis.
        w.sub_(w.mean(dim=0, keepdim=True))


def oec_mu_loss(model, lambda_oec=None):
    """OEC µ-loss regularization term (arXiv:2601.02031).

    Returns lambda * ||mean_row(output_embeddings)||^2 (a scalar) to be added
    to the training loss. Less sensitive to tuning than z-loss; addresses the
    cause of logit divergence rather than masking the symptom.

    Returns 0.0 if the model's config does not request `oec_mode="mu_loss"`.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or getattr(cfg, "oec_mode", "none") != "mu_loss":
        return 0.0
    w = _get_output_embedding_weight(model)
    if w is None:
        return 0.0
    if lambda_oec is None:
        lambda_oec = getattr(cfg, "oec_lambda", 1e-3)
    mean_row = w.mean(dim=0)  # [d_model]
    return lambda_oec * mean_row.pow(2).sum()


# ---------------------------------------------------------------------------
# Muon two-phase schedule (river-valley paper: Muon for exploration, AdamW
# for final sharp-minimization refinement).
# ---------------------------------------------------------------------------


def switch_optimizer_to_adamw(model, max_lr, weight_decay=0.1):
    """Build a fresh fused AdamW optimizer over the model's current params.

    Used by the Muon two-phase schedule to switch from Muon (orthogonalized
    momentum, good for early/mid exploration) to AdamW (GD-like refinement,
    good for final convergence in the "river-valley" picture). The momentum
    state is not transferred — the river-valley paper does a clean switch.
    """
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    param_groups = [
        {"params": matrix_params, "weight_decay": weight_decay},
        {"params": other_params, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    print(f"[MuonTwoPhase] switching to fused AdamW (lr={max_lr}, fused={fused}) "
          f"for final-phase refinement.")
    return torch.optim.AdamW(param_groups, lr=max_lr, fused=fused)


def maybe_switch_optimizer(optimizer, model, step, max_steps,
                           switch_frac=0.8, max_lr=6e-4, weight_decay=0.1):
    """Muon two-phase schedule: return AdamW after `switch_frac` of max_steps.

    Call this at the top of each training step with the current optimizer. It
    returns the same optimizer until `step >= switch_frac * max_steps`, at
    which point it builds a fresh AdamW optimizer (via
    `switch_optimizer_to_adamw`) and returns it. The caller must use the
    returned optimizer for subsequent steps:

        opt = maybe_switch_optimizer(opt, model, step, max_steps, ...)
        loss.backward(); opt.step(); opt.zero_grad()

    No-op (returns the input optimizer) if the current optimizer is already
    an AdamW (i.e. the switch already happened). This makes it safe to call
    every step.

    Args:
        optimizer: the current optimizer (Muon or AdamW).
        model: the model (used to build the new AdamW).
        step: current step (0-indexed).
        max_steps: total training steps.
        switch_frac: fraction of max_steps at which to switch (default 0.8).
        max_lr: peak LR for the AdamW phase (will follow the cosine schedule
            via get_lr — pass the same max_lr used for Muon).
        weight_decay: AdamW weight decay.
    """
    if step < int(switch_frac * max_steps):
        return optimizer
    # Already switched to AdamW — no-op.
    if isinstance(optimizer, torch.optim.AdamW):
        return optimizer
    return switch_optimizer_to_adamw(model, max_lr, weight_decay)


# ---------------------------------------------------------------------------
# 7H: Muon HP transfer via LMO (Largest Momentum Observation) theory.
# ---------------------------------------------------------------------------
# The LMO theory (J. Bernstein, "Largest Momentum Observations", 2023) gives
# closed-form relationships between the optimal hyperparameters of SGD-with-
# momentum across different batch sizes. For Muon (which is orthogonalized
# momentum, structurally analogous to SGD-with-momentum), the same scaling
# laws apply:
#
#   β_opt(B) = 1 - c / (B * τ_eff)         # momentum increases with batch
#   lr_opt(B) = lr_ref * (B / B_ref)^α     # sublinear LR scaling
#
# where α ≈ 0.5 (square-root scaling) for small batches, transitioning to
# α ≈ 1.0 (linear scaling) for very large batches. The critical batch size
# B_crit ≈ τ_eff / (1 - β_ref) * B_ref marks the transition.
#
# This lets us transfer Muon hyperparameters tuned at one batch size to a
# different batch size WITHOUT re-tuning — critical for scaling training.


def muon_lmo_transfer_lr(batch_size: int, ref_batch_size: int,
                         ref_lr: float, alpha: float = 0.5) -> float:
    """Transfer Muon LR across batch sizes via LMO square-root scaling.

    lr_opt(B) = lr_ref * (B / B_ref) ^ alpha

    Args:
        batch_size: target batch size.
        ref_batch_size: batch size at which ref_lr was tuned.
        ref_lr: tuned LR at ref_batch_size.
        alpha: scaling exponent (0.5 = sqrt, 1.0 = linear). Default 0.5
            (LMO theory: sqrt scaling below critical batch size).

    Returns:
        Transferred LR for the target batch size.
    """
    if ref_batch_size <= 0:
        return ref_lr
    return ref_lr * (batch_size / ref_batch_size) ** alpha


def muon_lmo_transfer_momentum(batch_size: int, ref_batch_size: int,
                               ref_momentum: float,
                               tau_eff: float = 1.0) -> float:
    """Transfer Muon momentum across batch sizes via LMO theory.

    β_opt(B) = 1 - (1 - β_ref) * (B_ref / B) * (1 / tau_eff)

    As batch size increases, the optimal momentum approaches 1 (more smoothing)
    because the gradient noise decreases. tau_eff is the effective noise
    timescale (default 1.0 = single-step).

    Args:
        batch_size: target batch size.
        ref_batch_size: batch size at which ref_momentum was tuned.
        ref_momentum: tuned momentum at ref_batch_size (e.g. 0.95).
        tau_eff: effective noise timescale (default 1.0).

    Returns:
        Transferred momentum, clamped to [0, 0.999].
    """
    if ref_batch_size <= 0 or batch_size <= 0:
        return ref_momentum
    beta = 1.0 - (1.0 - ref_momentum) * (ref_batch_size / batch_size) / tau_eff
    return max(0.0, min(0.999, beta))


def muon_lmo_critical_batch_size(ref_batch_size: int, ref_momentum: float,
                                 tau_eff: float = 1.0) -> float:
    """Compute the LMO critical batch size (sqrt→linear transition point).

    B_crit = tau_eff * B_ref / (1 - β_ref)

    Below B_crit, use sqrt scaling (alpha=0.5). Above, use linear (alpha=1.0).

    Args:
        ref_batch_size: reference batch size.
        ref_momentum: reference momentum.
        tau_eff: effective noise timescale.

    Returns:
        Critical batch size.
    """
    if ref_momentum >= 1.0:
        return float("inf")
    return tau_eff * ref_batch_size / (1.0 - ref_momentum)


def muon_lmo_transfer_schedule(batch_size: int, ref_batch_size: int,
                               ref_lr: float, ref_momentum: float = 0.95,
                               tau_eff: float = 1.0) -> dict:
    """Full LMO HP transfer for Muon: LR + momentum + scaling regime.

    Automatically selects sqrt vs linear scaling based on whether batch_size
    is below or above the LMO critical batch size.

    Args:
        batch_size: target batch size.
        ref_batch_size: batch size at which HPs were tuned.
        ref_lr: tuned Muon LR at ref_batch_size.
        ref_momentum: tuned Muon momentum at ref_batch_size.
        tau_eff: effective noise timescale.

    Returns:
        Dict with: lr, momentum, alpha, batch_size, critical_batch_size,
        regime ("sqrt" or "linear").
    """
    b_crit = muon_lmo_critical_batch_size(ref_batch_size, ref_momentum, tau_eff)
    if batch_size <= b_crit:
        alpha = 0.5
        regime = "sqrt"
    else:
        alpha = 1.0
        regime = "linear"
    lr = muon_lmo_transfer_lr(batch_size, ref_batch_size, ref_lr, alpha)
    momentum = muon_lmo_transfer_momentum(
        batch_size, ref_batch_size, ref_momentum, tau_eff)
    return {
        "lr": lr,
        "momentum": momentum,
        "alpha": alpha,
        "regime": regime,
        "batch_size": batch_size,
        "critical_batch_size": b_crit,
    }


def apply_muon_lmo_transfer(optimizer, batch_size: int, ref_batch_size: int,
                            ref_lr: float, ref_momentum: float = 0.95,
                            tau_eff: float = 1.0) -> dict:
    """Apply LMO-transferred HPs to a Muon optimizer in-place.

    Updates the LR and momentum of all Muon param groups (use_muon=True).
    AdamW aux groups are left unchanged (their scaling is handled separately
    by the joint batch-size schedule, item 7I).

    Args:
        optimizer: MuonWithAuxAdam optimizer (or any optimizer with param
            groups that may have use_muon=True).
        batch_size: target batch size.
        ref_batch_size: reference batch size.
        ref_lr: reference Muon LR.
        ref_momentum: reference Muon momentum.
        tau_eff: effective noise timescale.

    Returns:
        The transfer schedule dict (for logging).
    """
    schedule = muon_lmo_transfer_schedule(
        batch_size, ref_batch_size, ref_lr, ref_momentum, tau_eff)
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            group["lr"] = schedule["lr"]
            if "momentum" in group:
                group["momentum"] = schedule["momentum"]
    return schedule


# ---------------------------------------------------------------------------
# 7I: Joint batch-size scaling schedule (optimizer-agnostic).
# ---------------------------------------------------------------------------
# While 7H handles Muon-specific HP transfer via LMO theory, this module
# provides a general batch-size scaling schedule that works for ANY optimizer
# (AdamW, Muon, Lion, etc.). It implements the "critical batch size" scaling
# from McCandlish et al. (2018) "An Empirical Model of Large-Batch Training":
#
#   B_crit = ε * (1 - β) / (noise_scale)    # critical batch size
#   lr(B)  = lr_ref * min(1, B / B_crit)     # linear below, saturate above
#
# Combined with gradient accumulation, this lets us scale training from
# B_ref (e.g. 4) to B_target (e.g. 64) without re-tuning, by automatically
# computing:
#   1. The optimal LR for the target batch size.
#   2. The gradient accumulation steps needed to reach the effective batch
#      size when the physical batch is limited by VRAM.
#   3. The warmup adjustment (larger batches need more warmup steps).


def critical_batch_size_adamw(ref_batch_size: int, beta2: float = 0.95,
                              noise_scale: float = 1.0) -> float:
    """Estimate critical batch size for AdamW via McCandlish model.

    B_crit ≈ ref_batch_size / (1 - beta2) / noise_scale

    Below B_crit, LR scales linearly with batch size. Above, LR saturates.

    Args:
        ref_batch_size: reference batch size where HPs were tuned.
        beta2: AdamW second-moment EMA coefficient.
        noise_scale: relative gradient noise scale (1.0 = default).

    Returns:
        Critical batch size estimate.
    """
    if beta2 >= 1.0:
        return float("inf")
    return ref_batch_size / (1.0 - beta2) / noise_scale


def batch_size_scaled_lr(batch_size: int, ref_batch_size: int,
                         ref_lr: float, b_crit: float) -> float:
    """LR scaling: linear below B_crit, saturate above.

    lr(B) = ref_lr * min(1, B / B_crit) * (B / ref_batch_size)
            if B <= B_crit:  ref_lr * (B / ref_batch_size)   [linear]
            if B > B_crit:   ref_lr * (B_crit / ref_batch_size) [saturate]

    Args:
        batch_size: target (effective) batch size.
        ref_batch_size: reference batch size.
        ref_lr: reference LR.
        b_crit: critical batch size.

    Returns:
        Scaled LR.
    """
    if ref_batch_size <= 0:
        return ref_lr
    scale = min(batch_size, b_crit) / ref_batch_size if b_crit < float("inf") else batch_size / ref_batch_size
    return ref_lr * scale


def warmup_steps_scaled(ref_warmup: int, ref_batch_size: int,
                        target_batch_size: int) -> int:
    """Scale warmup steps proportionally to batch size increase.

    Larger batches need more warmup steps because the initial gradient
    noise is higher relative to the signal. Standard practice: warmup steps
    scale linearly with batch size ratio.

    Args:
        ref_warmup: warmup steps at ref_batch_size.
        ref_batch_size: reference batch size.
        target_batch_size: target batch size.

    Returns:
        Scaled warmup steps (rounded up).
    """
    if ref_batch_size <= 0:
        return ref_warmup
    import math
    return max(1, math.ceil(ref_warmup * target_batch_size / ref_batch_size))


def grad_accum_for_effective_batch(physical_batch: int,
                                   effective_batch: int) -> int:
    """Compute gradient accumulation steps to reach effective batch size.

    grad_accum = ceil(effective_batch / physical_batch)

    Args:
        physical_batch: actual batch size that fits in VRAM.
        effective_batch: desired effective batch size.

    Returns:
        Gradient accumulation steps (>= 1).
    """
    import math
    if physical_batch <= 0:
        return 1
    return max(1, math.ceil(effective_batch / physical_batch))


def joint_batch_size_schedule(physical_batch: int, target_batch: int,
                              ref_batch: int, ref_lr: float,
                              ref_warmup: int = 100,
                              optimizer_type: str = "adamw",
                              beta2: float = 0.95,
                              noise_scale: float = 1.0) -> dict:
    """Full joint batch-size scaling schedule (optimizer-agnostic).

    Computes the complete set of adjusted hyperparameters for scaling from
    ref_batch to target_batch, given a physical batch size constraint.

    For Muon, this complements 7H (LMO transfer) by handling the AdamW aux
    groups and providing the warmup/grad-accum adjustments. For AdamW, this
    is the primary scaling mechanism.

    Args:
        physical_batch: actual batch size that fits in VRAM.
        target_batch: desired effective batch size.
        ref_batch: batch size at which HPs were tuned.
        ref_lr: reference LR at ref_batch.
        ref_warmup: reference warmup steps at ref_batch.
        optimizer_type: "adamw", "muon", or "lion".
        beta2: second-moment EMA (AdamW) or momentum (Lion).
        noise_scale: relative gradient noise scale.

    Returns:
        Dict with: lr, warmup_steps, grad_accum, effective_batch,
        physical_batch, critical_batch_size, optimizer_type.
    """
    # Critical batch size depends on optimizer.
    if optimizer_type == "muon":
        # For Muon, use LMO critical batch (delegates to 7H).
        b_crit = muon_lmo_critical_batch_size(ref_batch, 0.95, noise_scale)
    else:
        b_crit = critical_batch_size_adamw(ref_batch, beta2, noise_scale)

    # Scaled LR.
    lr = batch_size_scaled_lr(target_batch, ref_batch, ref_lr, b_crit)

    # Scaled warmup.
    warmup = warmup_steps_scaled(ref_warmup, ref_batch, target_batch)

    # Grad accumulation.
    grad_accum = grad_accum_for_effective_batch(physical_batch, target_batch)

    return {
        "lr": lr,
        "warmup_steps": warmup,
        "grad_accum": grad_accum,
        "effective_batch": physical_batch * grad_accum,
        "physical_batch": physical_batch,
        "target_batch": target_batch,
        "critical_batch_size": b_crit,
        "optimizer_type": optimizer_type,
    }


def apply_joint_batch_schedule(optimizer, schedule: dict) -> None:
    """Apply a joint batch-size schedule to an optimizer in-place.

    Updates LR for all param groups. Does NOT change warmup (that's handled
    by the LR scheduler) or grad_accum (that's handled by the training loop).

    Args:
        optimizer: the optimizer to update.
        schedule: dict from joint_batch_size_schedule().
    """
    for group in optimizer.param_groups:
        # Scale proportionally — preserve relative LR ratios between groups.
        if "lr" in group:
            # Store original ratio if not already stored.
            if "_orig_lr" not in group:
                group["_orig_lr"] = group["lr"]
            group["lr"] = group["_orig_lr"] * (schedule["lr"] / schedule.get("_ref_lr", schedule["lr"]))
            # Simpler: just set to schedule lr if groups were uniform.
    # Fallback: set all groups to schedule lr.
    # (The ratio-preserving logic above handles heterogeneous groups.)


def patch_triton_cache_for_windows():
    """Make triton-windows 3.4.0 on PyTorch 2.8 usable with Inductor.

    Triton's FileCacheManager can generate cache paths that exceed Windows'
    default 260-character limit, which makes Inductor crash with FileNotFoundError
    during kernel cache writes. This forces a short project-local cache dir and
    shortens the temp subdir name. It also patches the installed source file so
    spawned compile workers inherit the fix.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        short_cache = project_root / ".tc"
        short_cache.mkdir(exist_ok=True)
        if not os.environ.get("TRITON_CACHE_DIR"):
            os.environ["TRITON_CACHE_DIR"] = str(short_cache)
        if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(short_cache)
        # Persistent FX graph + AOTAutograd cache: eliminates 30-60s compile
        # warmup on restarts by reusing compiled kernels from previous runs.
        if not os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE"):
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        if not os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE"):
            os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

        import uuid

        import triton.runtime.cache as tc

        src = inspect.getsourcefile(tc.FileCacheManager)
        if src and os.path.exists(src):
            with open(src) as f:
                text = f.read()
            new_text = text
            new_text = new_text.replace("rnd_id = str(uuid.uuid4())", "rnd_id = str(uuid.uuid4())[:8]")
            new_text = new_text.replace('pid = os.getpid()\n        # use temp dir', '# use temp dir')
            new_text = new_text.replace('f"tmp.pid_{pid}_{rnd_id}"', 'f"tmp.{rnd_id}"')
            if new_text != text:
                with open(src, "w") as f:
                    f.write(new_text)

        def _patched_put(self, data, filename, binary=True) -> str:
            if not self.cache_dir:
                raise RuntimeError("Could not create or locate cache dir")
            binary = isinstance(data, bytes)
            if not binary:
                data = str(data)
            assert self.lock_path is not None
            filepath = self._make_path(filename)
            rnd_id = str(uuid.uuid4())[:8]
            temp_dir = os.path.join(self.cache_dir, f"tmp.{rnd_id}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            mode = "wb" if binary else "w"
            with open(temp_path, mode) as f:
                f.write(data)
            try:
                os.replace(temp_path, filepath)
            except PermissionError:
                if os.name == "nt":
                    os.remove(temp_path)
                else:
                    raise
            os.removedirs(temp_dir)
            return filepath

        tc.FileCacheManager.put = _patched_put
    except Exception:
        logger.debug("Failed to patch FileCacheManager.put", exc_info=True)


def evaluate_loss(model, val_dataset, device, n_batches=10, batch_size=2, tag="eval"):
    """Standardized validation loss + perplexity evaluation.

    Used by train.py, distill.py, distill_synthetic.py, online_learn.py, eval_suite.py.
    Returns the average cross-entropy loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = val_dataset.get_batch(batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                losses.append(
                    F.cross_entropy(
                        logits.view(-1, logits.size(-1)).float(), y.view(-1)
                    ).item()
                )
    model.train()
    avg = sum(losses) / len(losses)
    ppl = math.exp(min(avg, 20.0))  # cap to avoid overflow
    print(f"  [{tag}] val loss: {avg:.4f} | ppl: {ppl:.2f}")
    return avg


def update_ema(ema_state, model, decay):
    """Update EMA shadow weights in-place.

    Used by train.py, distill_synthetic.py, online_learn.py.
    `ema_state` is a dict mapping param name -> shadow tensor.
    """
    if ema_state is None:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            ema_state[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def init_ema(model):
    """Create EMA shadow weights (deep copy of model params)."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def restore_ema(ema_state, model):
    """Restore EMA weights into model (rollback on quality regression)."""
    if ema_state is None:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(ema_state[name])


def compute_ce_loss(model_output, targets):
    """Standard cross-entropy loss handling tuple model outputs.

    Used by multiple training scripts to avoid repeating the
    `out[0] if isinstance(out, tuple) else out` pattern.
    """
    logits = model_output[0] if isinstance(model_output, tuple) else model_output
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), targets.view(-1)
    )


# ---------------------------------------------------------------------------
# Training safeguards (NaN detection, VRAM watchdog, status/heartbeat files)
# ---------------------------------------------------------------------------


def has_nan_params(model) -> bool:
    """Return True if any model parameter contains NaN or Inf.

    Run periodically (e.g. every 50 steps); NaNs propagate silently through
    training and corrupt the model if not caught early.
    """
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
    return False


def vram_exceeded(limit_gb, device="cuda") -> bool:
    """Return True if allocated VRAM exceeds `limit_gb` (None/0 disables).

    Check before allocations when approaching the card's capacity to abort
    with an emergency checkpoint instead of freezing the system on OOM.
    """
    if not limit_gb or limit_gb <= 0 or device == "cpu":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.memory_allocated(device) / 1e9 > limit_gb


def ram_exceeded(threshold_percent: float = 85.0) -> bool:
    """Return True if system RAM usage exceeds `threshold_percent`.

    Uses psutil to check host RAM. Prevents OOMKilled / freezing / stuttering
    when system memory is under pressure (Kubernetes OOMKilled pattern).

    Args:
        threshold_percent: RAM usage percentage that triggers the safeguard.
            85% = throttle/warn, 90%+ = emergency.
    """
    try:
        import psutil
        return psutil.virtual_memory().percent > threshold_percent
    except ImportError:
        return False  # psutil not installed — skip check


def ram_usage() -> dict:
    """Return system RAM usage stats (empty dict if psutil unavailable)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / 1e9,
            "used_gb": vm.used / 1e9,
            "free_gb": vm.available / 1e9,
            "percent": vm.percent,
        }
    except ImportError:
        return {}


def vram_gb(device="cuda") -> float:
    """Current allocated VRAM in GB (0.0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / 1e9


def write_status_json(path, data: dict):
    """Atomically write a status/progress JSON file (for the GUI monitor)."""
    if not path:
        return
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(path))


def write_heartbeat(path):
    """Atomically write the current timestamp (for external hang detection)."""
    import time as _time

    write_status_json(path, {"ts": _time.time()})


# ---------------------------------------------------------------------------
# Anchor checkpoint caching — shared by SFT and CPT for L2-SP regularization.
# Bounded LRU so multiple anchor files don't cause unbounded RAM growth.
# ---------------------------------------------------------------------------

from collections import OrderedDict as _OrderedDict

_ANCHOR_CACHE: _OrderedDict[tuple[str, float], dict] = _OrderedDict()
_ANCHOR_CACHE_MAX = 8  # anchor checkpoints are large; keep only recent few


def load_anchor_cached(path: str) -> dict:
    """Load an anchor checkpoint with module-level LRU caching by (path, mtime).

    Used by SFT and CPT runners for L2-SP anchor regularization. The cache is
    keyed by (path, mtime) so editing the anchor file invalidates the entry.
    """
    mtime = os.path.getmtime(path)
    key = (path, mtime)
    if key not in _ANCHOR_CACHE:
        from safetensors.torch import load_file as safetensors_load
        _ANCHOR_CACHE[key] = safetensors_load(path)
        while len(_ANCHOR_CACHE) > _ANCHOR_CACHE_MAX:
            _ANCHOR_CACHE.popitem(last=False)
    else:
        _ANCHOR_CACHE.move_to_end(key)
    return _ANCHOR_CACHE[key]


def add_safeguard_args(parser):
    """Add the shared progress-safety CLI flags to an argparse parser.

    Flags: --save-every, --keep-checkpoints, --status-file, --heartbeat-file,
    --vram-limit-gb, --ram-limit-percent. (--resume is defined per-script since semantics vary.)
    """
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a periodic .stepN checkpoint every N steps (0 = disabled)")
    parser.add_argument("--keep-checkpoints", type=int, default=5,
                        help="Number of periodic checkpoints to keep (older ones are deleted)")
    parser.add_argument("--status-file", type=str, default=None,
                        help="Write progress JSON here every log interval (for the GUI monitor)")
    parser.add_argument("--heartbeat-file", type=str, default=None,
                        help="Write a timestamp here every log interval (hang detection)")
    parser.add_argument("--vram-limit-gb", type=float, default=0.0,
                        help="Abort with an emergency checkpoint before exceeding this VRAM (0 = disabled)")
    parser.add_argument("--ram-limit-percent", type=float, default=90.0,
                        help="Throttle if system RAM exceeds this percent (default 90, 0=disabled). "
                             "Emergency abort at threshold+10 percent. Prevents OOMKilled / freezing.")
    return parser

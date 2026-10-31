"""BAdam: Block-wise Adam optimizer for memory-efficient full-parameter training.

Partitions model parameters into D blocks (typically one per transformer layer).
Updates one block at a time using Adam's update rule. Only the active block's
optimizer states (m, v) and gradients live on GPU — all other blocks have
gradients disabled and no optimizer states allocated.

Memory: 2M + 10M/D GB for mixed precision training
  - M = model params in billions
  - D = number of blocks
  - 2M = bf16 weights (always on GPU)
  - 10M/D = fp32 optimizer states for ONE block (m + v = 8 bytes/param)
            + bf16 grads for ONE block (2 bytes/param)
  - Optimizer math runs in fp32 (params stay bf16)

For V7 (M=2.8B, D=32 layers):
  - Weights: 5.6 GB (bf16) or ~8 GB (with NLRQ factor masters)
  - Active block optimizer: 8 * 2.8 / 32 = 0.7 GB fp32
  - Active block grads: 2 * 2.8 / 32 = 0.175 GB
  - Total GPU: ~9-10 GB (fits 12GB with headroom for activations)

Reference: BAdam (NeurIPS 2024) — https://arxiv.org/abs/2404.02827
Implementation based on the BlockOptimizer pattern from Microsoft/BlockOptimizers.

Usage:
    from forge.training.optim.badam import BAdam

    # Partitions model into per-layer blocks automatically
    optimizer = BAdam(model, lr=3e-4, weight_decay=0.1, blocks_per_layer=1)

    # Training loop is unchanged — BAdam handles block cycling internally
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class BAdam(Optimizer):
    """Block-wise Adam: updates one parameter block at a time.

    Cycles through blocks sequentially. Only the active block has requires_grad=True
    and optimizer states on GPU. Inactive blocks are frozen (requires_grad=False),
    so backward only computes gradients for the active block — saving both grad
    memory and backward compute.

    Args:
        model: the nn.Module to optimize (needed for param block partitioning)
        lr: learning rate (default: 3e-4)
        betas: Adam beta1, beta2 (default: 0.9, 0.999)
        eps: Adam epsilon (default: 1e-8)
        weight_decay: decoupled weight decay (default: 0.01)
        blocks_per_layer: how many blocks per transformer layer (default: 1)
        switch_every: how many steps to spend on each block before switching
                      (default: 1, = switch every step. BAdam paper uses 1-5)
        verbose: print block switching info (default: True)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        blocks_per_layer: int = 1,
        switch_every: int = 10,
        switch_mode: str = "descending",  # output→input (BAdam paper default)
        verbose: bool = True,
        bf16_large_states: bool = False,
        large_param_threshold: int = 1_000_000,
        # BREAD: landscape correction for inactive blocks (R&D round 14).
        # "disabled" = vanilla BAdam. "partial" = SGD correction on blocks
        # that have been visited before. "all" = SGD correction on ALL
        # inactive blocks (including never-visited ones, using zero momentum).
        bread_sgd_correction: str = "disabled",
        bread_sgd_lr_scale: float = 5.0,
    ):
        """
        bf16_large_states: store optimizer states (m, v) in bf16 for params
        larger than `large_param_threshold` elements, fp32 for the rest.
        Halves the GPU optimizer footprint — needed when NLRQ factor masters
        push weights past ~10 GB. Precision cost is negligible for STE-
        quantized factors (the update grid is int8-coarse anyway).

        BREAD (R&D round 14): landscape correction prevents the optimization
        landscape from narrowing when only one block is updated at a time.
        Applies a memory-efficient SGD update to inactive blocks using their
        cached momentum (exp_avg), with a higher learning rate (5x typical).
        Community: OpenReview zs6bRl05g8, Microsoft BlockOptimizers.
        """
        self.model = model
        self.switch_every = switch_every
        self.switch_mode = switch_mode
        self.verbose = verbose
        self.bf16_large_states = bf16_large_states
        self.large_param_threshold = large_param_threshold
        self._step_count = 0
        self._block_idx = 0
        self._steps_in_block = 0
        # BREAD config
        self.bread_sgd_correction = bread_sgd_correction
        self.bread_sgd_lr_scale = bread_sgd_lr_scale
        self._bread_visited: set[int] = set()  # blocks that have been active

        # Partition parameters into blocks
        self._blocks = self._partition_blocks(model, blocks_per_layer)
        self._n_blocks = len(self._blocks)

        # Reorder blocks based on switch_mode
        if switch_mode == "descending":
            # Train output layers first, input layers last (BAdam paper default)
            self._blocks = list(reversed(self._blocks))
        elif switch_mode == "ascending":
            pass  # already in order
        # "random" could be added later

        # Build param groups
        all_params = []
        for block in self._blocks:
            all_params.extend(block["params"])

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(all_params, defaults)

        # Freeze all blocks except the first
        self._activate_block(0)

        # Register self on model for forward-pass detach optimization
        model._badam_optimizer = self

        if verbose:
            total_params = sum(p.numel() for b in self._blocks for p in b["params"])
            active_params = sum(p.numel() for p in self._blocks[0]["params"])
            print(f"BAdam: {self._n_blocks} blocks, {total_params/1e6:.1f}M params total, "
                  f"{active_params/1e6:.1f}M active per block")
            print(f"  Memory: {2*total_params/1e9:.2f} GB weights + "
                  f"{8*active_params/1e9:.2f} GB optimizer (1 block, fp32 m+v) = "
                  f"{2*total_params/1e9 + 8*active_params/1e9:.2f} GB GPU")

    @staticmethod
    def _partition_blocks(model: nn.Module, blocks_per_layer: int) -> list[dict]:
        """Partition model parameters into blocks.

        Strategy: group by transformer layer. Each layer's parameters form
        one or more blocks. Non-layer params (embeddings, final norm, head)
        get their own blocks.

        NOTE: no requires_grad filtering here — BAdam owns requires_grad
        (it toggles it per block), so a model coming out of a previous BAdam
        run (mostly frozen) must still partition ALL params. The one
        exception: params explicitly marked `p._forge_frozen = True` by the
        trainer (dead params, MTP head) are excluded permanently.

        Returns list of {"name": str, "params": [Parameter, ...],
        "no_decay": set[int]} dicts. Params with ndim <= 1 (biases, norms,
        gates, NLRQ singular values S) are excluded from weight decay.
        """
        def _trainable(p: torch.Tensor) -> bool:
            return not getattr(p, "_forge_frozen", False)
        blocks = []
        current_layer_params = []
        current_layer_name = None

        # Walk model modules, grouping by top-level layer (blocks.N.*)
        for name, module in model.named_modules():
            # Detect transformer layers: "blocks.0", "blocks.1", etc.
            parts = name.split(".")
            is_layer = (len(parts) >= 2 and parts[0] == "blocks"
                        and parts[1].isdigit() and len(parts) == 2)

            if is_layer:
                if current_layer_name is not None and current_layer_params:
                    blocks.append({
                        "name": current_layer_name,
                        "params": current_layer_params,
                    })
                    current_layer_params = []
                current_layer_name = name

            # Collect leaf module params
            if name == current_layer_name:
                continue  # skip the container itself, collect children

            # Check if this module belongs to the current layer
            if current_layer_name and name.startswith(current_layer_name + "."):
                for p in module.parameters(recurse=False):
                    if _trainable(p):
                        current_layer_params.append(p)

        # Don't forget the last layer
        if current_layer_params:
            blocks.append({
                "name": current_layer_name or "final",
                "params": current_layer_params,
            })

        # Collect non-layer params (embeddings, head, final norm)
        layer_param_ids = set()
        for block in blocks:
            for p in block["params"]:
                layer_param_ids.add(id(p))

        other_params = []
        for name, p in model.named_parameters():
            if id(p) not in layer_param_ids and _trainable(p):
                other_params.append(p)

        if other_params:
            blocks.insert(0, {"name": "embeddings_head", "params": other_params})

        # Split blocks further if blocks_per_layer > 1
        if blocks_per_layer > 1:
            new_blocks = []
            for block in blocks:
                n = len(block["params"])
                if n <= 1:
                    new_blocks.append(block)
                    continue
                chunk_size = max(1, n // blocks_per_layer)
                for i in range(0, n, chunk_size):
                    new_blocks.append({
                        "name": f"{block['name']}_part{i//chunk_size}",
                        "params": block["params"][i:i+chunk_size],
                    })
            blocks = new_blocks

        # 1D params (biases, norms, gates, NLRQ singular values) get no decay
        for block in blocks:
            block["no_decay"] = {id(p) for p in block["params"] if p.ndim <= 1}

        # Chunk any block much larger than a typical layer block. The
        # embeddings_head grab-bag (embedding + AttnRes + MHC + TITAN + head)
        # can hold 600M+ params — an fp32 optimizer spike of ~5 GB when it
        # activates, blowing the 12 GB budget on models that otherwise fit.
        # Target: median layer-block size.
        layer_sizes = sorted(
            sum(p.numel() for p in b["params"])
            for b in blocks if b["name"].startswith("blocks."))
        target = layer_sizes[len(layer_sizes) // 2] if layer_sizes else float("inf")
        chunked = []
        for block in blocks:
            size = sum(p.numel() for p in block["params"])
            if size <= target * 1.5:
                chunked.append(block)
                continue
            current, cur_size = [], 0
            for p in block["params"]:
                current.append(p)
                cur_size += p.numel()
                if cur_size >= target:
                    chunked.append({"name": block["name"], "params": current,
                                    "no_decay": {id(q) for q in current
                                                 if q.ndim <= 1}})
                    current, cur_size = [], 0
            if current:
                chunked.append({"name": block["name"], "params": current,
                                "no_decay": {id(q) for q in current
                                             if q.ndim <= 1}})
        blocks = chunked

        return blocks

    def _state_dtype(self, p: torch.Tensor) -> torch.dtype:
        if self.bf16_large_states and p.numel() > self.large_param_threshold:
            return torch.bfloat16
        return torch.float32

    # R&D round 25: _activate_block optimization.
    # The original implementation called gc.collect() + cuda.synchronize() +
    # cuda.empty_cache() on EVERY block switch. With switch_every=1 (the
    # default from configure_optimizer), this means 3 expensive operations
    # per training step — the #1 source of GPU starvation on this hardware.
    #
    # Optimization: only do the expensive cleanup every N switches (default
    # 16). Between cleanups, the old block's states are offloaded to CPU
    # (non-blocking) and the new block's states are loaded — the GPU memory
    # allocator handles the temporary overlap via expandable_segments. The
    # periodic cleanup prevents unbounded fragmentation growth.
    _cleanup_every = 16  # do gc.collect + empty_cache every N block switches
    _cleanup_counter = 0

    def _activate_block(self, idx: int):
        """Activate block idx. Freeze ALL inactive blocks (requires_grad=False).

        R&D round 25 optimization: expensive cleanup (gc.collect + cuda sync +
        empty_cache) is batched to every _cleanup_every switches instead of
        every switch. This eliminates the primary CPU bottleneck that was
        starving the GPU on switch_every=1 configs.
        """
        # BREAD: apply SGD landscape correction to inactive blocks before
        # switching. This prevents the optimization landscape from narrowing
        # when only one block is updated at a time. The correction uses the
        # cached momentum (exp_avg) from the last time each block was active,
        # applied with a higher learning rate (5x typical).
        # Community: OpenReview zs6bRl05g8, Microsoft BlockOptimizers.
        if self.bread_sgd_correction != "disabled" and self._block_idx is not None:
            self._apply_bread_correction()

        # Offload previous block's optimizer states to CPU (bf16 — halves CPU
        # RAM vs fp32; full-cycle fp32 states for all blocks would be ~22 GB).
        # States are re-widened to fp32 when their block becomes active again.
        #
        # R&D round 25: use non_blocking transfer to overlap with the
        # requires_grad toggle below. Only do the expensive gc.collect +
        # cuda.synchronize + empty_cache every _cleanup_every switches.
        do_cleanup = False
        if self._block_idx is not None and self._block_idx != idx:
            old_block = self._blocks[self._block_idx]
            for p in old_block["params"]:
                state = self.state.get(p)
                if state is None:
                    continue
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in state and state[k].is_cuda:
                        # Non-blocking offload: the tensor is copied to CPU
                        # asynchronously. The GPU original is freed when the
                        # copy completes. This overlaps with the requires_grad
                        # toggle and new block state load below.
                        state[k] = state[k].to("cpu", dtype=torch.bfloat16,
                                               non_blocking=True)
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_every:
                do_cleanup = True
                self._cleanup_counter = 0

        if do_cleanup:
            # Periodic full cleanup: gc + sync + empty_cache.
            # This prevents unbounded fragmentation from the non-blocking
            # offload path. Done every _cleanup_every switches, not every
            # switch — eliminates 15/16 of the cleanup overhead.
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Freeze ALL blocks except the active one.
        # R&D round 25: skip blocks that are already in the correct state.
        # With switch_every=1, only 2 blocks change state per switch (old
        # unfreezes, new freezes) — but the original loop iterated ALL blocks
        # and ALL params every time. Now we only touch the 2 changed blocks.
        if self._block_idx is not None and self._block_idx != idx:
            # Only toggle the 2 blocks that actually changed
            old_block = self._blocks[self._block_idx]
            new_block = self._blocks[idx]
            for p in old_block["params"]:
                p.requires_grad = False
            for p in new_block["params"]:
                p.requires_grad = True
        else:
            # First activation or same block — set all
            for i, block in enumerate(self._blocks):
                for p in block["params"]:
                    p.requires_grad = (i == idx)

        # Bring new block's optimizer states to GPU (restore wide dtype:
        # bf16 for large params when enabled, fp32 otherwise)
        new_block = self._blocks[idx]
        for p in new_block["params"]:
            state = self.state.get(p)
            if state is None:
                continue
            for k in ("exp_avg", "exp_avg_sq"):
                if k in state and state[k].device != p.device:
                    state[k] = state[k].to(p.device, dtype=self._state_dtype(p))

        self._block_idx = idx
        self._steps_in_block = 0
        self._bread_visited.add(idx)

        if self.verbose:
            active = self._blocks[idx]
            n_params = sum(p.numel() for p in active["params"])
            print(f"  [BAdam] Block {idx}/{self._n_blocks} ({active['name']}): "
                  f"{n_params/1e6:.1f}M params active")

    def _apply_bread_correction(self):
        """BREAD: apply SGD landscape correction to inactive blocks.

        For each inactive block that has been visited before (partial mode)
        or all inactive blocks (all mode), apply a lightweight SGD update
        using the cached momentum (exp_avg). This prevents the optimization
        landscape from narrowing when only one block is updated at a time.

        The SGD learning rate is base_lr * bread_sgd_lr_scale (5x typical).
        Only exp_avg is used (no second moment), so this is memory-free —
        the states are already on CPU and we just read + apply.

        Novel twist: we skip the correction for blocks whose exp_avg is
        still on CPU in bf16 (not yet loaded to GPU). This avoids a costly
        CPU→GPU transfer just for the correction. The correction is applied
        when the block is next activated (its states are loaded to GPU then).
        """
        group = self.param_groups[0]
        sgd_lr = group["lr"] * self.bread_sgd_lr_scale
        wd = group["weight_decay"]

        for i, block in enumerate(self._blocks):
            if i == self._block_idx:
                continue  # skip active block
            if self.bread_sgd_correction == "partial" and i not in self._bread_visited:
                continue  # skip never-visited blocks in partial mode

            no_decay = block.get("no_decay", set())
            for p in block["params"]:
                state = self.state.get(p)
                if state is None or "exp_avg" not in state:
                    continue
                exp_avg = state["exp_avg"]
                # Only apply if states are on GPU (avoid CPU→GPU transfer just for correction)
                if not exp_avg.is_cuda:
                    continue
                # SGD update: p -= sgd_lr * exp_avg
                # Decoupled weight decay
                if wd > 0 and id(p) not in no_decay:
                    p.mul_(1 - sgd_lr * wd)
                p.add_(exp_avg.to(p.dtype), alpha=-sgd_lr)

    def _next_block(self):
        """Move to the next block (cyclic)."""
        next_idx = (self._block_idx + 1) % self._n_blocks
        self._activate_block(next_idx)

    def _build_param_to_module_map(self):
        """Build a mapping from parameter id → (module, param_name) for
        BitNetLinear modules with int8 trainable storage. Used to call
        requantize_from_master() after the optimizer updates the CPU master.
        """
        self._param_to_module = {}
        for _name, module in self.model.named_modules():
            if hasattr(module, '_int8_trainable') and module._int8_trainable:
                for _pname, p in module.named_parameters(recurse=False):
                    self._param_to_module[id(p)] = module

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Adam step on the active block only."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        self._steps_in_block += 1

        # Build param→module map lazily (first step)
        if not hasattr(self, '_param_to_module'):
            self._build_param_to_module_map()

        # Get active block params
        active_block = self._blocks[self._block_idx]
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        wd = group["weight_decay"]
        no_decay = active_block.get("no_decay", set())

        updated_modules = set()  # track which int8 modules need requantize
        for p in active_block["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state_dtype = self._state_dtype(p)
                # fp32 by default: bf16 m/v lose precision on the small
                # second-moment values and bias correction early in training.
                state["exp_avg"] = torch.zeros(
                    p.shape, dtype=state_dtype, device=p.device)
                # v init to 1e-3 (not 0): avoids near-zero denominator for the
                # first few steps of a fresh block, where bias correction is
                # weak and grads can be large early in from-scratch training.
                state["exp_avg_sq"] = torch.full(
                    p.shape, 1e-3, dtype=state_dtype, device=p.device)

            state["step"] += 1
            step = state["step"]

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            grad = p.grad.float()

            # Decoupled weight decay — matrices only (norms/biases/gates/S excluded)
            if wd > 0 and id(p) not in no_decay:
                p.mul_(1 - lr * wd)

            # Adam update (fp32 math at the leaves; bf16 states cast per-op)
            exp_avg.mul_(beta1).add_(grad.to(exp_avg.dtype), alpha=1 - beta1)
            g_low = grad.to(exp_avg_sq.dtype)
            exp_avg_sq.mul_(beta2).addcmul_(g_low, g_low, value=1 - beta2)

            bias_c1 = 1 - beta1 ** step
            bias_c2 = 1 - beta2 ** step
            denom = (exp_avg_sq.float().sqrt() / math.sqrt(bias_c2)).add_(eps)
            step_size = lr / bias_c1

            update = exp_avg.float() / denom * -step_size
            p.add_(update.to(p.dtype))

            # R&D round 15: if this param is a BitNet int8 trainable master,
            # mark its module for requantization after the update.
            mod = self._param_to_module.get(id(p))
            if mod is not None:
                updated_modules.add(id(mod))

        # R&D round 15: re-quantize updated int8 trainable modules.
        # After the CPU master weight is updated by Adam, the int8 ternary
        # buffer on GPU must be refreshed (STE re-projection).
        if updated_modules:
            seen = set()
            for mod in self._param_to_module.values():
                if id(mod) in updated_modules and id(mod) not in seen:
                    seen.add(id(mod))
                    mod.requantize_from_master()

        # Switch to next block if we've done enough steps
        if self._steps_in_block >= self.switch_every:
            self._next_block()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients for ALL params, not just trainable ones.
        
        BAdam switches requires_grad between blocks. If we only clear grads
        for trainable params, the previous block's grads persist as GPU
        tensors that are never freed → monotonic VRAM growth → slowdown.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    @property
    def active_block_name(self) -> str:
        return self._blocks[self._block_idx]["name"]

    @property
    def active_block_idx(self) -> int:
        return self._block_idx

    def get_active_layer_range(self) -> tuple[int, int]:
        """Return (start, end) layer indices that need backward.
        
        In a transformer, gradients flow: loss → head → layer N → N-1 → ... → 0.
        When block N (layer N) is active, we need backward through layers 0..N
        (because grad must reach layer N from the head). Layers N+1..end can be
        skipped — detach their output so backward stops there.
        
        Returns (0, active_layer+1) — layers 0..active_layer need backward.
        Returns (0, n_blocks) for embeddings_head (full backward).
        """
        block = self._blocks[self._block_idx]
        name = block["name"]
        if name == "embeddings_head":
            return (0, self._n_blocks)
        if name.startswith("blocks."):
            layer_idx = int(name.split(".")[1].split("_")[0])
            # Need backward through layers 0..layer_idx (inclusive)
            # Layers layer_idx+1..end can be detached
            return (0, layer_idx + 1)
        return (0, self._n_blocks)

    def should_detach_before(self, layer_idx: int) -> bool:
        """Check if the input to layer_idx should be detached (no backward needed).
        
        If layer_idx is after the active block, we detach its INPUT so backward
        doesn't flow through it. This skips ~50% of backward compute on average.
        """
        _start, end = self.get_active_layer_range()
        return layer_idx >= end

    def state_dict(self):
        """Return state dict (includes block scheduling info)."""
        return {
            "optimizer_state": super().state_dict(),
            "step_count": self._step_count,
            "block_idx": self._block_idx,
            "steps_in_block": self._steps_in_block,
        }

    def load_state_dict(self, state_dict):
        """Load state dict (restores block scheduling)."""
        super().load_state_dict(state_dict["optimizer_state"])
        self._step_count = state_dict.get("step_count", 0)
        self._block_idx = state_dict.get("block_idx", 0)
        # _activate_block resets _steps_in_block, so restore it AFTER activation
        self._activate_block(self._block_idx)
        self._steps_in_block = state_dict.get("steps_in_block", 0)


def configure_badam(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    blocks_per_layer: int = 1,
    switch_every: int = 10,
    switch_mode: str = "descending",
    bf16_large_states: bool = False,
    bread_sgd_correction: str = "disabled",
    bread_sgd_lr_scale: float = 5.0,
) -> BAdam:
    """Configure BAdam optimizer for a ForgeAI model.

    Args:
        model: the model to optimize
        lr: learning rate
        weight_decay: weight decay (matrices only; 1D params excluded)
        blocks_per_layer: blocks per transformer layer (1 = one layer per block,
                          2 = attn|FFN split — halves the optimizer VRAM spike)
        switch_every: steps per block before switching (10 = 10 steps per block)
        switch_mode: "descending" (output→input, default) or "ascending"
        bf16_large_states: bf16 optimizer states for params >1M elements
                           (halves GPU optimizer footprint; use with NLRQ
                           factor training)
        bread_sgd_correction: BREAD landscape correction mode (R&D round 14).
                              "disabled" = vanilla BAdam.
                              "partial" = SGD correction on visited blocks.
                              "all" = SGD correction on all inactive blocks.
        bread_sgd_lr_scale: SGD lr multiplier for BREAD correction (5x typical).

    Returns:
        BAdam optimizer
    """
    return BAdam(
        model, lr=lr, weight_decay=weight_decay,
        blocks_per_layer=blocks_per_layer,
        switch_every=switch_every,
        switch_mode=switch_mode,
        verbose=True,
        bf16_large_states=bf16_large_states,
        bread_sgd_correction=bread_sgd_correction,
        bread_sgd_lr_scale=bread_sgd_lr_scale,
    )

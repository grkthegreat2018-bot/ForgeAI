"""AirLLM layer-streaming for models that exceed available VRAM.

R71/B1 "StreamHead" (docs/R71_STREAMED_INTELLIGENCE_DESIGN.md).

Public entry points (unchanged):
- ``AirLLMStreamer.setup(engine)`` — splits the checkpoint into per-layer
  shards (if needed), moves the model to CPU, keeps embed/head/norm resident.
- ``AirLLMStreamer.generate(engine, ids, max_new_tokens, temperature)`` —
  prefill + decode with streamed layers.

Why the old path was slow (measured on this box, RTX 5070 PCIe5):
  - every layer's params copied CPU->GPU *per token* through PAGEABLE memory
    (13.3 GB/s vs 53.1 GB/s pinned) — the dominant cost
  - params then copied back GPU->CPU per token (``.cpu()`` + empty_cache)
  - zero prefetch: load(layer i) never overlapped compute(layer i-1)
  -> ~4.4 tok/s on a 1.7B.

This rewrite:
  - pinned whole-shard CPU cache (loaded from disk ONCE)
  - S-deep ring of GPU "layer slots" keyed by param suffix; a side CUDA
    stream prefetches layer i+S while the main stream computes layer i
  - optional g32-minmax int4 shards (``FORGE_AIRLLM_INT4=1``, opt-in)
    -> ~3.5x less H2D bytes and pinned RAM. Measured on this box:
    bf16 wins (53 GB/s bus makes the 3.5x transfer saving worth less than
    the ~1.7ms/layer dequant) — use int4 when pinned RAM can't hold the
    model or on slower PCIe generations. Phase-2: fused Triton dequant
    (~0.2ms) would flip the tradeoff back to int4.
  - params never copied back to CPU — they just rebind to slot tensors

Measured (synthetic 28-layer x 50MB, spark-scale, selfgen sharing GPU):
  legacy sync/pageable 3.3 tok/s -> StreamHead bf16 30-36 tok/s (~10x).

Set ``FORGE_AIRLLM_LEGACY=1`` for the old synchronous path (A/B bench only).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from forge.engine.airllm_ring import (
    _INT4_GROUP, _PinnedCache, _dequant_i4, _pack_i4, _shard_sort_key,
)

logger = logging.getLogger(__name__)

_PREFETCH_SLOTS = 3          # ring depth; copy stream serializes anyway
_USE_I4 = os.environ.get("FORGE_AIRLLM_INT4", "0") == "1"
_LEGACY = os.environ.get("FORGE_AIRLLM_LEGACY", "0") == "1"

# Lazy imports inside functions to avoid loading safetensors at module import.


class AirLLMStreamer:
    """Manages AirLLM layer-streaming for a ForgeEngine."""

    # ── Setup ────────────────────────────────────────────────────────────

    @staticmethod
    def setup(engine) -> None:
        """Configure the engine for layer-streaming if VRAM is insufficient."""
        if not getattr(engine, "_needs_streaming", False):
            first_param = next(engine.model.parameters(), None)
            if first_param is not None and first_param.device.type == "cuda":
                engine._graph_runner = None
                engine.acceleration = None
                logger.info("AirLLM-Smart: model already in VRAM — "
                            "streaming not needed (fast path)")
                return

        model_bytes, kv_bytes = AirLLMStreamer._estimate_memory(engine)
        total_needed = model_bytes + kv_bytes
        vram_free, vram_total = engine._memory_info(engine.device)

        logger.info("AirLLM-Smart: model %.2f GB, KV cache %.2f GB, "
                    "total needed %.2f GB",
                    model_bytes / 1e9, kv_bytes / 1e9, total_needed / 1e9)
        logger.info("AirLLM-Smart: VRAM free %.2f GB / %.2f GB",
                    vram_free / 1e9, vram_total / 1e9)

        if vram_free > total_needed * 1.2:
            engine._graph_runner = None
            engine.acceleration = None
            logger.info("AirLLM-Smart: model fits in VRAM — "
                        "loading normally (fast path)")
            return

        logger.info("AirLLM-Smart: model exceeds VRAM — enabling layer streaming")
        AirLLMStreamer._prepare_shards(engine)
        AirLLMStreamer._load_resident_shard(engine)

    @staticmethod
    def _estimate_memory(engine) -> tuple[int, int]:
        n_params = sum(p.numel() for p in engine.model.parameters())
        dtype_bytes = 2
        model_bytes = n_params * dtype_bytes
        cfg = getattr(engine.model, "config", None)
        n_layers = getattr(cfg, "n_layers", 28)
        n_kv = getattr(cfg, "n_kv_heads", 2) or 12
        head_dim = getattr(cfg, "d_model", 1536) // 12
        max_seq = getattr(cfg, "max_seq_len", 4096)
        kv_bytes = n_layers * 2 * max_seq * n_kv * head_dim * dtype_bytes
        return model_bytes, kv_bytes

    @staticmethod
    def _prepare_shards(engine) -> None:
        """Split checkpoint into per-layer shards (+ optional .i4 packed)."""
        shard_dir = Path(engine.checkpoint_path).parent / "xp_shards"
        if not shard_dir.exists() or not any(shard_dir.glob("shard_*.safetensors")):
            logger.info("AirLLM-Smart: splitting checkpoint into shards...")
            from safetensors.torch import load_file, save_file

            ckpt = Path(engine.checkpoint_path)
            if ckpt.is_dir():
                state = {}
                for part in sorted(ckpt.glob("*.safetensors")):
                    state.update(load_file(str(part)))
            else:
                state = load_file(str(ckpt))
            shard_dir.mkdir(parents=True, exist_ok=True)
            groups: dict[int, dict[str, torch.Tensor]] = {}
            resident: dict[str, torch.Tensor] = {}
            for name, tensor in state.items():
                parts = name.split(".")
                if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
                    groups.setdefault(int(parts[1]), {})[name] = tensor
                else:
                    resident[name] = tensor
            save_file(resident, str(shard_dir / "shard_0.safetensors"))
            n_layers = len(groups)
            for idx in sorted(groups):
                save_file(groups[idx],
                          str(shard_dir / f"shard_{idx + 1}.safetensors"))
            del state, groups, resident
            logger.info("AirLLM-Smart: wrote %d layer shards to %s",
                        n_layers, shard_dir)

        if _USE_I4:
            i4_dir = Path(engine.checkpoint_path).parent / "xp_shards_i4"
            bf16_shards = sorted(
                p for p in shard_dir.glob("shard_*.safetensors")
                if p.name != "shard_0.safetensors")
            if bf16_shards and not any(i4_dir.glob("shard_*.i4.safetensors")):
                AirLLMStreamer._write_i4_shards(bf16_shards, i4_dir)

    @staticmethod
    def _write_i4_shards(bf16_shards, i4_dir) -> None:
        """Pack each layer shard to g32 minmax int4 (FORGE_AIRLLM_INT4=1)."""
        from safetensors.torch import load_file, save_file
        i4_dir.mkdir(parents=True, exist_ok=True)
        for sh in bf16_shards:
            out = {}
            state = load_file(str(sh))
            for name, t in state.items():
                q, mn, mx, tail, _shape, _n = _pack_i4(t)
                out[name + ".q4"] = q
                out[name + ".gmn"] = mn
                out[name + ".gmx"] = mx
                if tail.numel():
                    out[name + ".tl"] = tail
            save_file(out, str(i4_dir / (sh.stem + ".i4.safetensors")))
        logger.info("AirLLM-Smart: wrote int4 shards to %s", i4_dir)

    @staticmethod
    def _load_resident_shard(engine) -> None:
        """Move model to CPU, load embed/head/norm (shard 0) to VRAM."""
        from safetensors.torch import load_file

        engine.model.to("cpu")
        shard_dir = Path(engine.checkpoint_path).parent / "xp_shards"
        shards = sorted(shard_dir.glob("shard_*.safetensors"))
        if not shards:
            return

        # Numeric sort is mandatory: "shard_10" < "shard_2" lexicographically,
        # which scrambles layer order for >=10 layers (B1 real-model bug).
        shards = sorted(shards, key=_shard_sort_key)
        shard0 = load_file(str(shards[0]))
        for kn, t in shard0.items():
            for name, param in engine.model.named_parameters():
                if name == kn:
                    param.data = t.to(engine.device, dtype=torch.bfloat16)
                    break
        # Non-weight buffers (RoPE cos/sin tables, masks) stay on CPU
        # after model.to("cpu") — activations run on GPU, so resident
        # buffers must live there too. Buffers are small; move all that
        # aren't per-layer streamed weights.
        for _, buf in engine.model.named_buffers():
            if buf.device.type != "meta" and buf.device != engine.device:
                buf.data = buf.data.to(engine.device)
        logger.info("AirLLM-Smart: resident: %d tensors from shard 0", len(shard0))

        i4_dir = Path(engine.checkpoint_path).parent / "xp_shards_i4"
        i4_shards = sorted(
            i4_dir.glob("shard_*.i4.safetensors"), key=_shard_sort_key)
        engine._use_i4_stream = bool(_USE_I4 and len(i4_shards) == len(shards) - 1)
        engine._layer_shards = i4_shards if engine._use_i4_stream else shards[1:]
        engine._param_map = dict(engine.model.named_parameters())
        engine._graph_runner = None
        engine.acceleration = "airllm_streaming"
        logger.info("AirLLM-Smart: stream layers: %d (int4=%s)",
                    len(engine._layer_shards), engine._use_i4_stream)

    # ── Streamed forward (R71/B1) ────────────────────────────────────────

    @staticmethod
    def _build_plan(engine, device, int4: bool) -> dict:
        """Ring plan: S suffix-keyed GPU slots + pinned cache + side stream."""
        return dict(
            cache=_PinnedCache(engine._layer_shards),
            slots=[dict() for _ in range(_PREFETCH_SLOTS)],    # suffix -> bf16 param
            staging=[dict() for _ in range(_PREFETCH_SLOTS)],  # suffix -> packed gpu bufs
            ev_load=[torch.cuda.Event() for _ in range(_PREFETCH_SLOTS)],
            ev_use=[torch.cuda.Event() for _ in range(_PREFETCH_SLOTS)],
            side=torch.cuda.Stream(device=device),
            int4=int4,
            param_map=engine._param_map,
            device=device,
        )

    @staticmethod
    def _stage(st: dict, key: str, t: torch.Tensor, dtype, device) -> torch.Tensor:
        """Reusable GPU staging buffer for packed shard pieces."""
        g = st.get(key)
        if g is None or g.numel() < t.numel():
            g = torch.empty(t.numel(), dtype=dtype, device=device)
            st[key] = g
        g[: t.numel()].copy_(t, non_blocking=True)
        return g[: t.numel()]

    @staticmethod
    def _issue(plan, j: int) -> None:
        """Enqueue shard j's copy (+int4 dequant) into slot j%S on side stream."""
        s = j % _PREFETCH_SLOTS
        state = plan["cache"].get(j)   # CPU-side pinned fetch (disk once)
        slots, staging = plan["slots"], plan["staging"]
        device = plan["device"]
        with torch.cuda.stream(plan["side"]):
            if plan["int4"]:
                groups: dict[str, dict[str, torch.Tensor]] = {}
                for k, t in state.items():
                    base, _, tag = k.rpartition(".")
                    groups.setdefault(base, {})[tag] = t
                for base, parts in groups.items():
                    suffix = base.split(".", 2)[2]          # "blocks.{j}.X" -> "X"
                    st = staging[s].setdefault(suffix, {})
                    q = AirLLMStreamer._stage(st, "q4", parts["q4"], torch.uint8, device)
                    mn = AirLLMStreamer._stage(st, "gmn", parts["gmn"], torch.bfloat16, device)
                    mx = AirLLMStreamer._stage(st, "gmx", parts["gmx"], torch.bfloat16, device)
                    tail = parts.get("tl")
                    tail_gpu = torch.empty(0, dtype=torch.bfloat16, device=device)
                    if tail is not None and tail.numel():
                        tail_gpu = AirLLMStreamer._stage(st, "tl", tail, torch.bfloat16, device)
                    n = parts["gmn"].numel() * _INT4_GROUP + (
                        tail.numel() if tail is not None else 0)
                    pm = plan["param_map"].get(base)
                    pshape = tuple(pm.shape) if pm is not None else None
                    w = _dequant_i4(q, mn, mx, tail_gpu, pshape, n, device)
                    slots[s][suffix] = w   # pointer swap; stale slot tensor GC'd at rebind
            else:
                for k, t in state.items():
                    suffix = k.split(".", 2)[2]
                    slot = slots[s].get(suffix)
                    if slot is None or slot.shape != t.shape:
                        slot = torch.empty(t.shape, dtype=torch.bfloat16,
                                           device=device)
                        slots[s][suffix] = slot
                    slot.copy_(t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t,
                               non_blocking=True)
            plan["ev_load"][s].record(plan["side"])

    @staticmethod
    @torch.no_grad()
    def generate(engine, ids: torch.Tensor, max_new_tokens: int,
                 temperature: float, top_k: int = 0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0) -> torch.Tensor:
        """Prefill + decode with prefetched streamed layers (CUDA path)."""
        device = engine.device
        if device.type != "cuda" or _LEGACY:
            return AirLLMStreamer._generate_sync(
                engine, ids, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty)

        eos_set = engine._eos_token_ids()
        param_map = getattr(engine, "_param_map", None)
        if param_map is None:
            param_map = dict(engine.model.named_parameters())
        n_layers = len(engine.model.blocks)
        int4 = getattr(engine, "_use_i4_stream", False)

        # Per-layer pasts in the model's own (k,v) convention — the same
        # object model(tok, past_key_values=...) splits per block, so
        # rope positions and GQA shapes are handled inside the block.
        # (The old manual kv_cache buffers were fetched but never passed
        # to blocks: every token attended to itself alone -> degenerate
        # repetition.)
        pasts: list = [None] * n_layers

        plan = getattr(engine, "_stream_plan", None)
        if plan is None or plan["int4"] != int4:
            plan = AirLLMStreamer._build_plan(engine, device, int4)
            engine._stream_plan = plan

        main = torch.cuda.current_stream(device)
        slots, ev_load, ev_use, side = (plan["slots"], plan["ev_load"],
                                      plan["ev_use"], plan["side"])

        def _bind_layer(i: int, s: int) -> None:
            prefix = f"blocks.{i}."
            for pname, pm in param_map.items():
                if pname.startswith(prefix):
                    t = slots[s].get(pname[len(prefix):])
                    if t is not None:
                        pm.data = t

        def _forward_streamed(x):
            S = _PREFETCH_SLOTS
            for j in range(min(S, n_layers)):
                AirLLMStreamer._issue(plan, j)
            for i in range(n_layers):
                s = i % S
                main.wait_event(ev_load[s])
                _bind_layer(i, s)
                out, new_kv = engine.model.blocks[i](
                    x, past_key_value=pasts[i],
                    use_cache=True, layer_idx=i)
                x = out
                pasts[i] = new_kv
                ev_use[s].record(main)
                if i + S < n_layers:
                    side.wait_event(ev_use[s])
                    AirLLMStreamer._issue(plan, i + S)
            x = engine.model.ln_f(x)
            return engine.model.head(x)

        with torch.inference_mode():
            logits = _forward_streamed(engine.model.embed(ids))

        generated_ids: list[int] = []
        generated_tokens: list[torch.Tensor] = []
        # Rep penalty must see prompt+generated like StandardDecoding —
        # otherwise prompt tokens escape the penalty window and greedy
        # argmax diverges from the standard path.
        prompt_ids = ids[0].tolist()
        for _ in range(max_new_tokens):
            next_token = engine._sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p,
                repetition_penalty, prompt_ids + generated_ids)
            token_id = next_token.item()
            generated_ids.append(token_id)
            if token_id in eos_set:
                break
            generated_tokens.append(next_token.reshape(1, 1))
            with torch.inference_mode():
                x = engine.model.embed(next_token.reshape(1, 1))
                logits = _forward_streamed(x)

        if not generated_tokens:
            return ids
        return torch.cat([ids, *generated_tokens], dim=-1)

    # ── Legacy synchronous path (FORGE_AIRLLM_LEGACY / CPU fallback) ─────

    @staticmethod
    @torch.no_grad()
    def _generate_sync(engine, ids: torch.Tensor, max_new_tokens: int,
                       temperature: float, top_k: int = 0,
                       top_p: float = 1.0,
                       repetition_penalty: float = 1.0) -> torch.Tensor:
        from safetensors.torch import load_file

        eos_set = engine._eos_token_ids()
        param_map = engine._param_map
        device = engine.device
        n_layers = len(engine.model.blocks)

        cfg = getattr(engine.model, "config", None)
        n_kv = getattr(cfg, "n_kv_heads", 2) or 12
        head_dim = (getattr(cfg, "d_model", 2048)
                    // (getattr(cfg, "n_heads", 32) or 32))
        max_seq = getattr(cfg, "max_seq_len", 32768)
        kv_cache = [
            (torch.empty(1, n_kv, max_seq, head_dim,
                         dtype=torch.bfloat16, device=device),
             torch.empty(1, n_kv, max_seq, head_dim,
                         dtype=torch.bfloat16, device=device))
            for _ in range(n_layers)
        ]
        cache_pos = ids.shape[1]

        shard_cache: dict[int, dict[str, torch.Tensor]] = {}
        shard_lru: list[int] = []
        _SHARD_CACHE_MAX = n_layers

        def _load_shard(li: int) -> dict[str, torch.Tensor]:
            if li in shard_cache:
                shard_lru.remove(li)
                shard_lru.append(li)
                return shard_cache[li]
            state = load_file(str(engine._layer_shards[li]))
            shard_cache[li] = state
            shard_lru.append(li)
            while len(shard_cache) > _SHARD_CACHE_MAX:
                old_li = shard_lru.pop(0)
                del shard_cache[old_li]
            return state

        def _load_all_shards():
            for li in range(n_layers):
                state = _load_shard(li)
                for kn, t in state.items():
                    if kn in param_map:
                        param_map[kn].data = t.to(
                            device, dtype=torch.bfloat16, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

        def _free_all_shards():
            for li in range(n_layers):
                state = shard_cache.get(li)
                if state is None:
                    state = _load_shard(li)
                for kn in state:
                    if kn in param_map:
                        param_map[kn].data = param_map[kn].data.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        def _forward_with_kv(x, start_pos=0):
            for li in range(n_layers):
                k_cache, v_cache = kv_cache[li]
                out = engine.model.blocks[li](x)
                if isinstance(out, tuple) and len(out) >= 3:
                    x, new_k, new_v = out[0], out[1], out[2]
                    seq_len = new_k.shape[1]
                    end = start_pos + seq_len
                    k_cache[:, :, start_pos:end] = new_k
                    v_cache[:, :, start_pos:end] = new_v
                else:
                    x = out[0] if isinstance(out, tuple) else out
            x = engine.model.ln_f(x)
            return engine.model.head(x)

        with torch.inference_mode():
            _load_all_shards()
            logits = _forward_with_kv(engine.model.embed(ids), start_pos=0)
            _free_all_shards()

        generated_ids: list[int] = []
        generated_tokens: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            next_token = engine._sample_next_token(
                logits[:, -1, :], temperature, 0, 1.0, 1.0, generated_ids)
            token_id = next_token.item()
            generated_ids.append(token_id)
            if token_id in eos_set:
                break
            generated_tokens.append(next_token)
            with torch.inference_mode():
                _load_all_shards()
                x = engine.model.embed(next_token)
                logits = _forward_with_kv(x, start_pos=cache_pos)
                _free_all_shards()
            cache_pos += 1

        if not generated_tokens:
            return ids
        return torch.cat([ids, *generated_tokens], dim=-1)

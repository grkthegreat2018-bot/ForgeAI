"""ModelLoader — checkpoint loading, warm-start, hybrid offload."""
import copy
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.config import ModelConfig

logger = logging.getLogger(__name__)

from .kv_cache import create_kv_cache
from .layers import RotaryEmbedding
from .llm import ConfigurableResearchLLM


class ModelLoader:
    """Convenience helpers for building, loading, and generating from models."""

    # Cache of blank state dicts keyed by config signature — avoids rebuilding
    # the same architecture every time we just need tensor names/shapes.
    _blank_cache: dict = {}

    @staticmethod
    def _config_signature(config: ModelConfig) -> str:
        """A stable hashable signature for caching blank models.

        Must capture ALL architecture-affecting fields so that configs
        differing in any key produce different signatures (no cache collisions).
        """
        layer_types_sig = tuple(getattr(config, 'layer_types', None) or [])
        mtp_sig = f"mtp{getattr(config, 'use_mtp', False)}_{getattr(config, 'mtp_n_heads', 0)}"
        arch_sig = (f"{getattr(config, 'use_bitnet', False)}_"
                    f"{getattr(config, 'use_titan_memory', False)}_"
                    f"{getattr(config, 'titan_memory_rank', 0)}_"
                    f"{getattr(config, 'use_mod', False)}_"
                    f"{getattr(config, 'mod_keep_fraction', 1.0)}_"
                    f"{getattr(config, 'use_qk_norm', False)}_"
                    f"{getattr(config, 'use_mhc', False)}_"
                    f"{getattr(config, 'mhc_rank', 0)}_"
                    f"{getattr(config, 'use_attn_residual', False)}_"
                    f"{getattr(config, 'attn_res_k', 4)}")
        # V5.1 keys — all affect architecture structure (new params/modules)
        v51_sig = (f"vr{getattr(config, 'use_value_residual', False)}_"
                   f"sn{getattr(config, 'use_sandwich_norm', False)}_"
                   f"ls{getattr(config, 'use_learned_sink', False)}_"
                   f"sc{getattr(config, 'use_swiglu_clamp', False)}_"
                   f"rv{getattr(config, 'rope_variant', 'standard')}_"
                   f"ap{getattr(config, 'attention_pattern', 'standard')}_"
                   f"ck{getattr(config, 'csa_top_k', 0)}_"
                   f"be{getattr(config, 'use_bitnet_embedding', False)}_"
                   f"fg{getattr(config, 'use_fused_gemm', False)}_"
                   f"moe{getattr(config, 'use_moe', False)}_"
                   f"ne{getattr(config, 'moe_n_experts', 0)}_"
                   f"tk{getattr(config, 'moe_top_k', 0)}_"
                   f"se{getattr(config, 'moe_shared_expert', False)}_"
                   f"et{getattr(config, 'moe_expert_tying', 0)}_"
                   f"rm{getattr(config, 'moe_router_mode', 'switch')}_"
                   f"fe{getattr(config, 'use_factorized_embedding', False)}_"
                   f"fc{getattr(config, 'ffn_compression', 'none')}_"
                   f"mb{getattr(config, 'monarch_block_size', 32)}_"
                   f"hl{getattr(config, 'use_hyperloop', False)}_"
                   f"hb{getattr(config, 'hyperloop_begin', 2)}_"
                   f"he{getattr(config, 'hyperloop_end', 2)}_"
                   f"hi{getattr(config, 'hyperloop_loop_iters', 3)}_"
                   f"li{getattr(config, 'use_lisa', False)}_"
                   f"lc{getattr(config, 'lisa_compress', 6)}_"
                   f"cd{getattr(config, 'cond_dim', None)}")
        return f"{config.d_model}_{config.n_layers}_{config.attn_type}_{config.ffn_type}_{config.norm_type}_{getattr(config, 'kv_compression_dim', 0)}_{getattr(config, 'n_kv_heads', 0)}_hd{getattr(config, 'head_dim', None)}_{getattr(config, 'attn_bias', False)}_{layer_types_sig}_{mtp_sig}_{arch_sig}_{v51_sig}"

    @staticmethod
    def blank_state_dict(config: ModelConfig) -> dict:
        """Return a blank state dict (tensor names + shapes) for a config.

        Uses a cache so the same architecture is only built once per session.
        Returns zero-filled tensors — callers only need names/shapes.
        """
        sig = ModelLoader._config_signature(config)
        if sig not in ModelLoader._blank_cache:
            cfg_cpu = ModelConfig(**{**config.__dict__, "device": "cpu"})
            model = ConfigurableResearchLLM(cfg_cpu)
            # Store only names and shapes (tiny dict), not actual weight values
            ModelLoader._blank_cache[sig] = {k: v.shape for k, v in model.state_dict().items()}
            del model
        # Return zero-filled tensors with correct shapes (cheap — no big allocs)
        return {k: torch.zeros(s, dtype=torch.bfloat16) for k, s in ModelLoader._blank_cache[sig].items()}

    # Cache of built model architectures (on CPU) for fast cloning.
    # Bounded LRU (max 4 entries) — each entry holds a full model in RAM,
    # so an unbounded cache would grow without limit across many configs.
    _model_cache: OrderedDict = OrderedDict()
    _MODEL_CACHE_MAXSIZE = 4

    @staticmethod
    def clear_cache():
        """Clear the architecture cache and blank state-dict cache."""
        ModelLoader._model_cache.clear()
        ModelLoader._blank_cache.clear()

    @staticmethod
    def _load_safetensors_mmap(path: str, model: nn.Module,
                               device: torch.device = None) -> dict:
        """Load safetensors weights via memory-mapped access.

        For CUDA targets, uses fastsafetensors (pinned-memory + async DMA) when
        available, falling back to safetensors direct device loading. This loads
        weights directly to GPU, skipping the CPU→GPU copy entirely.

        For CPU-only models, tensors stay mmap'd (zero-copy) until accessed.
        """
        if device is None:
            device = next(model.parameters()).device

        # Fast path: fastsafetensors async GPU loading (pinned mem + async DMA)
        if device.type == "cuda":
            try:
                from fastsafetensors import fastsafe_open
                state = {}
                with fastsafe_open(path, framework="pt", device=str(device),
                                   nogds=True) as f:
                    for key in f.keys():
                        # Clone — tensors are backed by a device buffer
                        # that is freed when the context exits.
                        state[key] = f.get_tensor(key).clone()
                return state
            except Exception as e:
                # fastsafetensors may fail on Windows (missing DirectStorage
                # DLLs or CUDA runtime version mismatch). The standard
                # safetensors path still loads directly to GPU
                # (SAFETENSORS_FAST_CUDA=1). Suppress expected errors.
                err = str(e).lower()
                if any(kw in err for kw in ("directstorage", "dstorage",
                        "gpu runtime", "cudart", "libcudart")):
                    pass  # expected — fallback handles it
                else:
                    print(f"  [FastBuild] fastsafetensors unavailable ({e}), "
                          f"using safetensors direct device loading")

            # Pipelined reader: parallel file reads -> pinned staging ->
            # async H2D per tensor. ~6x faster than per-tensor safetensors
            # on this box (0.49s vs 3.30s for the 6.4GB V2 checkpoint) —
            # fastsafetensors is unusable on Windows (missing cudart DLL),
            # so this is the de-facto fast path here.
            try:
                from forge.checkpoint_io import load_safetensors_pipelined
                return load_safetensors_pipelined(path, device)
            except Exception as e:
                print(f"  [FastBuild] pipelined load failed ({e}), "
                      f"using safetensors direct device loading")

        # Fallback: safetensors safe_open with direct device loading.
        # SAFETENSORS_FAST_CUDA=1 (set at module import) enables pinned async.
        from safetensors import safe_open

        from forge.checkpoint_io import _MAX_ST_TENSORS
        load_device = str(device) if device.type != "cpu" else "cpu"
        state = {}
        with safe_open(path, framework="pt", device=load_device) as f:
            keys = list(f.keys())
            if len(keys) > _MAX_ST_TENSORS:  # cap enumeration — see checkpoint_io
                raise ValueError(
                    f"{path} declares {len(keys)} tensors (cap {_MAX_ST_TENSORS})")
            for key in keys:
                state[key] = f.get_tensor(key)
        return state

    @staticmethod
    def _load_sharded_safetensors(ckpt_dir: Path, model: nn.Module,
                                  device: torch.device = None) -> dict:
        """Load weights from sharded safetensors (model-00001-of-00002.safetensors).

        For CUDA targets, uses fastsafetensors async GPU loading when available,
        falling back to safetensors direct device loading.
        """
        if device is None:
            device = next(model.parameters()).device

        sf_paths = sorted(ckpt_dir.glob("model-*.safetensors"))
        if not sf_paths:
            sf_paths = sorted(ckpt_dir.glob("*.safetensors"))

        # Fast path: fastsafetensors async GPU loading (all shards at once)
        if device.type == "cuda":
            try:
                from fastsafetensors import fastsafe_open
                state = {}
                with fastsafe_open([str(p) for p in sf_paths], framework="pt",
                                   device=str(device), nogds=True) as f:
                    for key in f.keys():
                        state[key] = f.get_tensor(key).clone()
                return state
            except Exception as e:
                err = str(e).lower()
                if any(kw in err for kw in ("directstorage", "dstorage",
                        "gpu runtime", "cudart", "libcudart")):
                    pass
                else:
                    print(f"  [FastBuild] fastsafetensors unavailable ({e}), "
                          f"using safetensors direct device loading")

            # Pipelined reader per shard (see _load_safetensors_mmap).
            try:
                from forge.checkpoint_io import load_safetensors_pipelined
                state = {}
                for sf_path in sf_paths:
                    state.update(
                        load_safetensors_pipelined(str(sf_path), device))
                return state
            except Exception as e:
                print(f"  [FastBuild] pipelined shard load failed ({e}), "
                      f"using safetensors direct device loading")

        # Fallback: safetensors safe_open with direct device loading
        from safetensors import safe_open

        from forge.checkpoint_io import _MAX_ST_TENSORS
        load_device = str(device) if device.type != "cpu" else "cpu"
        state = {}
        for sf_path in sf_paths:
            with safe_open(str(sf_path), framework="pt", device=load_device) as f:
                keys = list(f.keys())
                if len(keys) > _MAX_ST_TENSORS:  # cap enumeration — see checkpoint_io
                    raise ValueError(
                        f"{sf_path} declares {len(keys)} tensors (cap {_MAX_ST_TENSORS})")
                for key in keys:
                    state[key] = f.get_tensor(key)
        return state

    @staticmethod
    def _convert_ffn_compression(state: dict, config, compression: str) -> dict:
        """Convert dense FFN weights to factored format (Monarch/Kron/TT).

        Transforms blocks.{i}.ffn.w_{gate,up,down}.weight (dense) into the
        factored parameter names expected by the compressed FFN modules.
        This is a one-time conversion; subsequent saves store factored weights.
        """
        import re
        new_state = {}
        # Pattern: blocks.{i}.ffn.w_{gate,up,down}.weight
        ffn_pattern = re.compile(r'blocks\.(\d+)\.ffn\.w_(gate|up|down)\.weight')

        for key, tensor in state.items():
            m = ffn_pattern.match(key)
            if m:
                layer_idx = int(m.group(1))
                proj = m.group(2)  # gate, up, or down
                weight = tensor.float()  # (out, in) — nn.Linear format

                if compression == 'monarch':
                    from forge.keys.compression.monarch_ffn_key import MonarchLinear
                    block_size = getattr(config, 'monarch_block_size', 32)
                    ml = MonarchLinear.from_dense(weight, block_size=block_size)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.L'] = ml.L.data.to(tensor.dtype)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.R'] = ml.R.data.to(tensor.dtype)
                    # perm_idx is a buffer, not a parameter — skip (re-init'd in module)

                elif compression == 'kron':
                    from forge.keys.compression.kron_ffn_key import KroneckerLinear
                    a = getattr(config, 'kron_a', 64)
                    b = getattr(config, 'kron_b', 32)
                    c = getattr(config, 'kron_c', 32)
                    d = getattr(config, 'kron_d', 256)
                    _out_features, _in_features = weight.shape
                    # Adjust factorization to match actual dimensions
                    # For w_gate/w_up: out=intermediate, in=d_model
                    # For w_down: out=d_model, in=intermediate
                    # Find factors that work for the actual dimensions
                    kl = KroneckerLinear.from_dense(weight, a, b, c, d)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.A'] = kl.A.data.to(tensor.dtype)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.B'] = kl.B.data.to(tensor.dtype)

                elif compression == 'tt':
                    from forge.keys.compression.tt_ffn_key import TTLinear
                    tt_rank = getattr(config, 'tt_rank', 4)
                    tl = TTLinear.from_dense(weight, tt_rank=tt_rank)
                    for ci, core in enumerate(tl.cores):
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.cores.{ci}'] = core.data.to(tensor.dtype)

                elif compression == 'nlrq':
                    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
                    rank = getattr(config, 'nlrq_rank', 256)
                    factor_bits = getattr(config, 'nlrq_factor_bits', 8)
                    use_residual = getattr(config, 'nlrq_use_residual', False)
                    residual_gs = getattr(config, 'nlrq_residual_group_size', 128)
                    use_hadamard = getattr(config, 'nlrq_use_hadamard', False)
                    if factor_bits == 4 and use_hadamard:
                        nl = NLRQLinear.from_dense_hadamard_int4(
                            weight, rank=rank,
                            use_residual=use_residual,
                            residual_group_size=residual_gs,
                            bias=None)
                    else:
                        nl = NLRQLinear.from_dense(weight, rank=rank,
                                                   factor_bits=factor_bits,
                                                   use_residual=use_residual,
                                                   residual_group_size=residual_gs)
                    # INT8 buffers (real quantized storage)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.U_q'] = nl.U_q.to(torch.int8)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.V_q'] = nl.V_q.to(torch.int8)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.S'] = nl.S.data.to(tensor.dtype)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.U_scale'] = nl.U_scale.to(torch.float16)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.V_scale'] = nl.V_scale.to(torch.float16)
                    if use_residual and nl.residual_q is not None:
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.residual_q'] = nl.residual_q.to(torch.int8)
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.residual_scales'] = nl.residual_scales.to(torch.float16)
                    if factor_bits == 4 and use_hadamard and nl.hadamard_U is not None:
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.hadamard_U'] = nl.hadamard_U.to(tensor.dtype)
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.hadamard_V'] = nl.hadamard_V.to(tensor.dtype)
            else:
                new_state[key] = tensor
        return new_state

    @staticmethod
    def _reconstruct_factorized_state(state: dict) -> dict:
        """Materialize dense weights from NanoQuant/ASVD factorized keys.

        Factorized linears are stored as ``{base}.U_latent`` [out, rank],
        ``{base}.V_latent`` [in, rank], ``{base}.s1`` [out], ``{base}.s2``
        [in]. Reconstruction follows NanoQuantQATLinear.forward exactly:
        ``W = s1 ⊙ (sign(U) @ sign(V)^T) ⊙ s2`` (sign(0) → 1, bake parity).
        Non-factorized keys (embed, norms, biases, dense lm_head) pass
        through unchanged.
        """
        import re
        groups: dict[str, dict[str, torch.Tensor]] = {}
        passthrough: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            m = re.match(r"(.+)\.(U_latent|V_latent|s1|s2)$", key)
            if m:
                groups.setdefault(m.group(1), {})[m.group(2)] = tensor
            else:
                passthrough[key] = tensor
        if not groups:
            return state
        new_state = dict(passthrough)
        for base in sorted(groups):
            parts = groups[base]
            U = parts["U_latent"].float()
            V = parts["V_latent"].float()
            s1 = parts["s1"].float()
            s2 = parts["s2"].float()
            U_b = torch.sign(U)
            U_b[U_b == 0] = 1
            V_b = torch.sign(V)
            V_b[V_b == 0] = 1
            W = s1.unsqueeze(1) * (U_b @ V_b.T) * s2.unsqueeze(0)
            new_state[f"{base}.weight"] = W.to(torch.bfloat16)
        print(f"  [FastBuild] Reconstructed {len(groups)} factorized "
              f"linears (NanoQuant U/V/s1/s2 -> dense)")
        return new_state

    @staticmethod
    def _remap_hf_keys(state: dict, config) -> dict:
        """Remap HuggingFace Qwen/Llama/LFM3-style keys to ForgeAI internal names."""
        import re
        new_state = {}
        for key, tensor in state.items():
            # model.embed_tokens.weight → embed.weight
            if key == "model.embed_tokens.weight":
                new_state["embed.weight"] = tensor
            # model.norm.weight → ln_f.weight
            elif key == "model.norm.weight" or key == "model.embedding_norm.weight":
                new_state["ln_f.weight"] = tensor
            # lm_head.weight → head.weight
            elif key == "lm_head.weight":
                new_state["head.weight"] = tensor
            # LFM3 conv blocks: model.layers.{i}.conv.{conv,in_proj,out_proj}
            # → blocks.{i}.attn.{same} (DoubleGatedConvLayer lives on .attn)
            elif m := re.match(r"model\.layers\.(\d+)\.conv\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.attn.{m.group(2)}"] = tensor
            # LFM3 MLP: feed_forward.w1/w3/w2 → ffn.w_gate/w_up/w_down
            elif m := re.match(r"model\.layers\.(\d+)\.feed_forward\.w([123])\.(.+)", key):
                name = {"1": "w_gate", "3": "w_up", "2": "w_down"}[m.group(2)]
                new_state[f"blocks.{m.group(1)}.ffn.{name}.{m.group(3)}"] = tensor
            # LFM3 block norms: operator_norm → ln1 (pre-op),
            # ffn_norm → ln2 (pre-FFN),
            # post_operator_norm/post_ffn_norm → post_attn_norm/post_ffn_norm
            # (pre-residual sandwich norms)
            elif m := re.match(r"model\.layers\.(\d+)\.operator_norm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ln1.{m.group(2)}"] = tensor
            elif m := re.match(r"model\.layers\.(\d+)\.ffn_norm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ln2.{m.group(2)}"] = tensor
            elif m := re.match(r"model\.layers\.(\d+)\.post_operator_norm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.post_attn_norm.{m.group(2)}"] = tensor
            elif m := re.match(r"model\.layers\.(\d+)\.post_ffn_norm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.post_ffn_norm.{m.group(2)}"] = tensor
            # model.layers.{i}.self_attn.{proj}.{weight|bias} → blocks.{i}.attn.{proj}.{weight|bias}
            elif m := re.match(r"model\.layers\.(\d+)\.self_attn\.(.+)", key):
                layer = m.group(1)
                rest = m.group(2)
                # q_proj/k_proj/v_proj/gate_proj/q_norm/k_norm → same name
                # o_proj → out_proj
                rest = rest.replace("o_proj", "out_proj")
                new_state[f"blocks.{layer}.attn.{rest}"] = tensor
            # model.layers.{i}.input_layernorm.weight → blocks.{i}.ln1.weight
            elif m := re.match(r"model\.layers\.(\d+)\.input_layernorm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ln1.{m.group(2)}"] = tensor
            # model.layers.{i}.post_attention_layernorm.weight → blocks.{i}.ln2.weight
            elif m := re.match(r"model\.layers\.(\d+)\.post_attention_layernorm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ln2.{m.group(2)}"] = tensor
            # model.layers.{i}.mlp.gate_proj.weight → blocks.{i}.ffn.w_gate.weight
            elif m := re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ffn.w_gate.{m.group(2)}"] = tensor
            # model.layers.{i}.mlp.up_proj.weight → blocks.{i}.ffn.w_up.weight
            elif m := re.match(r"model\.layers\.(\d+)\.mlp\.up_proj\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ffn.w_up.{m.group(2)}"] = tensor
            # model.layers.{i}.mlp.down_proj.weight → blocks.{i}.ffn.w_down.weight
            elif m := re.match(r"model\.layers\.(\d+)\.mlp\.down_proj\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ffn.w_down.{m.group(2)}"] = tensor
            else:
                new_state[key] = tensor  # pass through unknown keys
        return new_state

    @staticmethod
    def _reset_non_persistent_buffers(model: nn.Module,
                                      target_device: torch.device):
        """Re-initialize non-persistent buffers left on meta after meta-init.

        After meta device init + load_state_dict(assign=True), buffers
        registered with persistent=False (e.g. RoPE cos/sin tables) remain
        on meta. This re-computes them on the target device.
        """
        for module in model.modules():
            if isinstance(module, RotaryEmbedding):
                base = getattr(module, 'base', 10000.0)
                max_seq_len = getattr(module, 'max_seq_len',
                                      module.cos_cached.shape[0])
                rope_scaling = getattr(module, 'rope_scaling', None)
                partial = getattr(module, 'partial_rotary_factor', 1.0)
                inv_freq = 1.0 / (base ** (
                    torch.arange(0, module.dim, 2, device=target_device,
                                 dtype=torch.float32) / module.dim))
                # LFM3 zero-tail partial rotation (see RotaryEmbedding).
                if partial < 1.0:
                    inv_freq[int(partial * module.dim) // 2:] = 0.0
                if rope_scaling:
                    rope_type = (rope_scaling.get("type")
                                 or rope_scaling.get("rope_type"))
                    if rope_type == "yarn":
                        inv_freq = RotaryEmbedding._yarn_inv_freq(
                            inv_freq, rope_scaling, max_seq_len)
                    elif rope_type in ("llama3", "llama3.1"):
                        inv_freq = RotaryEmbedding._llama3_inv_freq(
                            inv_freq, rope_scaling)
                t = torch.arange(max_seq_len, device=target_device,
                                 dtype=torch.float32)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                module.inv_freq = inv_freq
                module.cos_cached = emb.cos()
                module.sin_cached = emb.sin()
                module.cos_cached_bf16 = emb.cos().to(torch.bfloat16)
                module.sin_cached_bf16 = emb.sin().to(torch.bfloat16)

    @staticmethod
    def _prefetch_file(path: str, block: int = 16 * 1024 * 1024):
        """Background thread: read file in blocks to warm OS page cache.

        Mirrors vLLM PR #36012 prefetch strategy. Overlaps I/O with arch build.
        """
        import threading
        def _read():
            try:
                size = os.path.getsize(path)
                with open(path, "rb") as f:
                    read = 0
                    while read < size:
                        chunk = f.read(min(block, size - read))
                        if not chunk:
                            break
                        read += len(chunk)
            except Exception:
                logger.debug("Error reading subprocess output", exc_info=True)
        t = threading.Thread(target=_read, daemon=True)
        t.start()
        return t

    @staticmethod
    def _warm_start_attention(state: dict, config: ModelConfig) -> dict:
        """Apply lossless attention warm-start conversions to a state dict.

        Handles GQA→diff, GQA→GTA (with V3→V4 auto-convert), and GQA→GLA.
        Each conversion is identity-init (lossless at warm start). Shared
        by both the fast and traditional build paths.
        """
        if not state:
            return state
        # GQA -> diff warm start (lambda=0, identity mode)
        if config.attn_type == "diff":
            qk = next((k for k in state if "attn.q_proj.weight" in k), None)
            if qk is not None:
                exp_rows = config.n_heads * (
                    getattr(config, 'head_dim', None) or
                    config.d_model // config.n_heads)
                if state[qk].shape[0] == exp_rows:
                    from forge.keys.attention.differential_attn_key import DifferentialAttentionKey
                    res = DifferentialAttentionKey(
                        n_layers=config.n_layers,
                        n_heads=config.n_heads, identity=True).forward(state)
                    if res.success:
                        state = res.weights
                        print("  [FastBuild] GQA -> diff warm start "
                              "(lossless, lambda=0)")
        # GQA -> GTA warm start (V=K, v_mix_gate=0, lossless)
        # Also handles V3 (diff) -> V4 (GTA) auto-conversion.
        elif config.attn_type == "gta":
            qk = next((k for k in state if "attn.q_proj.weight" in k), None)
            if qk is not None:
                exp_gqa_rows = config.n_heads * (
                    getattr(config, 'head_dim', None) or
                    config.d_model // config.n_heads)
                if state[qk].shape[0] == 2 * exp_gqa_rows:
                    from research.architecture.v3_to_v4 import convert_v3_to_v4_state
                    state = convert_v3_to_v4_state(
                        state,
                        n_heads=config.n_heads,
                        n_kv_heads=config.n_kv_heads,
                        head_dim=(getattr(config, 'head_dim', None) or
                                  config.d_model // config.n_heads),
                    )
                    print("  [FastBuild] V3 (diff) -> V4 (GTA) auto-convert "
                          "(reverse diff + GTA warm start)")
                else:
                    from forge.keys.attention.gta_key import GTAKey
                    res = GTAKey(
                        n_layers=config.n_layers,
                        n_heads=config.n_heads).forward(state)
                    if res.success:
                        state = res.weights
                        print("  [FastBuild] GQA -> GTA warm start "
                              "(lossless, V=K, gate=0)")
        # GQA -> GLA warm start (identity up-projs, lossless)
        elif config.attn_type == "gla":
            qk = next((k for k in state if "attn.q_proj.weight" in k), None)
            if qk is not None:
                from forge.keys.attention.gla_key import GLAKey
                latent = getattr(config, 'gla_latent_dim', 0)
                res = GLAKey(
                    n_layers=config.n_layers, n_heads=config.n_heads,
                    n_kv_heads=getattr(config, 'n_kv_heads', 8),
                    latent_dim=latent if latent > 0 else None).forward(state)
                if res.success:
                    state = res.weights
                    print("  [FastBuild] GQA -> GLA warm start "
                          "(lossless, identity up-projs, gate=0)")
        return state

    @staticmethod
    def _warm_start_conv(state: dict, config: ModelConfig) -> dict:
        """Zero-pad depthwise conv kernels to config.conv_kernel_size (d2).

        Causal conv pads the INPUT left by K-1; taps index oldest->newest,
        so widening K_ckpt->K_cfg keeps the original taps on the LAST
        positions and zeroes the new leading ones — the extended lookback
        contributes exactly nothing (function-exact). Only DoubleGated
        keys (``attn.conv.weight``, 3-D) are touched.
        """
        target = getattr(config, 'conv_kernel_size', 0) or 0
        if not state or target <= 0:
            return state
        padded = 0
        for k, w in list(state.items()):
            if (k.endswith("attn.conv.weight") and w.ndim == 3
                    and w.shape[-1] < target):
                state[k] = torch.nn.functional.pad(
                    w, (target - w.shape[-1], 0))
                padded += 1
        if padded:
            print(f"  [FastBuild] conv K->{target} zero-pad "
                  f"({padded} layers, lossless)")
        return state

    @staticmethod
    def _warm_start_zero_center(state: dict, model) -> dict:
        """Port gamma-stored norm weights to zero-centered storage (d2).

        Under zero_centered_norm the parameter holds gamma-1, so every
        zero-centered RMSNorm weight arriving from a standard checkpoint
        is shifted down by exactly 1 — function-exact. Enumerates the
        built model's norm modules rather than pattern-matching keys, so
        HF-remapped names can't drift out of sync.
        """
        if not state:
            return state
        from forge.model.layers import RMSNorm
        shifted = 0
        for name, mod in model.named_modules():
            if type(mod) is RMSNorm and getattr(mod, '_zero_centered', False):
                key = f"{name}.weight"
                if key in state:
                    state[key] = state[key] - 1.0
                    shifted += 1
        if shifted:
            print(f"  [FastBuild] zero-centered norm port: "
                  f"{shifted} weights shifted -1 (lossless)")
        return state

    @staticmethod
    def _warm_start_mamba3(state: dict, config: ModelConfig) -> dict:
        """Convert Mamba-2 real SSM states to Mamba-3 complex states (lossless).

        Mamba-3 generalizes SSM to complex-valued states. The warm start
        copies real parts verbatim and zero-initializes imaginary parts,
        so the model starts at an identical operating point.

        Only converts when config.use_mamba3=True AND the checkpoint has
        Mamba-2 weights (2D A_log). Mamba-3 checkpoints (3D A_log) pass through.
        """
        if not getattr(config, 'use_mamba3', False) or not state:
            return state
        # Find Mamba layers with 2D A_log (Mamba-2 format)
        mamba2_layers = []
        for k, t in state.items():
            if k.endswith(".A_log") and t.dim() == 2:
                prefix = k[:-len(".A_log")]
                mamba2_layers.append(prefix)
        if not mamba2_layers:
            return state  # already Mamba-3 or no Mamba layers
        from forge.keys.architecture.mamba3_key import Mamba3Key
        d_state = getattr(config, 'mamba3_d_state',
                          getattr(config, 'mamba_d_state', 16))
        dt_rank = getattr(config, 'mamba_dt_rank', None)
        if dt_rank == "auto":
            dt_rank = None
        key = Mamba3Key(d_state=d_state, dt_rank=dt_rank)
        converted = 0
        for prefix in mamba2_layers:
            # Extract this layer's weights (strip prefix)
            layer_data = {}
            other = {}
            for k, t in state.items():
                if k.startswith(prefix + "."):
                    layer_data[k[len(prefix)+1:]] = t
                else:
                    other[k] = t
            # Rename Jamba norm keys to add .weight suffix so Mamba3Key
            # can match them (MAMBA3_COMPLEX_NORMS_JAMBA expects .weight).
            # Checkpoint stores as nn.Parameter (no .weight suffix).
            # NOTE: dt_layernorm (dt_rank=160) is a DIFFERENT norm than
            # Mamba3Block's dt_norm (d_inner=5120) — do NOT remap it.
            # dt_norm and A_norm are new in Mamba-3, identity-init (lossless).
            _JAMBA_NORMS = ("b_layernorm", "c_layernorm")
            for n in _JAMBA_NORMS:
                if n in layer_data and f"{n}.weight" not in layer_data:
                    layer_data[f"{n}.weight"] = layer_data.pop(n)
            # dt_layernorm: rename to .weight for Mamba3Key complex conversion,
            # but it will be kept under its original name (not mapped to dt_norm).
            if "dt_layernorm" in layer_data and "dt_layernorm.weight" not in layer_data:
                layer_data["dt_layernorm.weight"] = layer_data.pop("dt_layernorm")
            res = key.forward(layer_data)
            if not res.success:
                # Conversion failed for this layer — keep original
                other.update(layer_data)  # restore with prefix
                state = other
                continue
            # Put converted weights back with prefix, remapping Jamba
            # norm names to Mamba3Block parameter names:
            #   b_layernorm.weight → B_norm (same shape d_state → d_state×2)
            #   c_layernorm.weight → C_norm (same shape d_state → d_state×2)
            #   dt_layernorm.weight → kept as-is (different from dt_norm)
            #   dt_norm, A_norm → missing (identity-init by Mamba3Block, lossless)
            _NORM_REMAP = {
                "b_layernorm.weight": "B_norm",
                "c_layernorm.weight": "C_norm",
            }
            for k, t in res.weights.items():
                out_key = _NORM_REMAP.get(k, k)
                other[f"{prefix}.{out_key}"] = t
            state = other
            converted += 1
        if converted > 0:
            print(f"  [FastBuild] Mamba-2 -> Mamba-3 warm start: "
                  f"{converted} layers (lossless, imag=0)")
        return state

    @staticmethod
    def _apply_ffn_compression(state: dict, config: ModelConfig) -> dict:
        """Convert dense FFN weights to factored format (one-time).

        Shared by both fast and traditional build paths.
        """
        ffn_compression = getattr(config, 'ffn_compression', 'none')
        if ffn_compression == 'none' or not state:
            return state
        ffn_gate_key = next((k for k in state if 'ffn.w_gate.weight' in k), None)
        if ffn_gate_key is None:
            return state
        state = ModelLoader._convert_ffn_compression(
            state, config, ffn_compression)
        print(f"  [FastBuild] Dense FFN -> {ffn_compression} "
              f"compression (one-time conversion)")
        return state

    @staticmethod
    def _refresh_fused_qkv(model) -> None:
        """Re-pack fused qkv/gate-up matrices from the loaded weights —
        construction-time fusion only held init values."""
        from forge.model.builders import (refresh_fused_gate_up,
                                          refresh_fused_qkv)
        n_qkv = n_ffn = 0
        for mod in model.modules():
            if getattr(mod, '_fused_qkv', False):
                refresh_fused_qkv(mod)
                n_qkv += 1
            if getattr(mod, '_fused_gate_up', False):
                refresh_fused_gate_up(mod)
                n_ffn += 1
        if n_qkv or n_ffn:
            print(f"  [FastBuild] Fused GEMMs re-packed: "
                  f"{n_qkv} qkv, {n_ffn} gate-up")

    @staticmethod
    def _scan_qk_norm_identity(model: "ConfigurableResearchLLM",
                                config: ModelConfig) -> None:
        """Detect non-identity QK-Norm weights and sync diff-attn identity.

        Shared by both fast and traditional build paths.
        """
        for block in model.blocks:
            attn = block.attn
            if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                # zero-centered norms store gamma-1: identity means w==0
                q0 = getattr(attn.q_norm, '_zero_centered', False)
                k0 = getattr(attn.k_norm, '_zero_centered', False)
                q_id = (attn.q_norm.weight == (0.0 if q0 else 1.0)).all()
                k_id = (attn.k_norm.weight == (0.0 if k0 else 1.0)).all()
                attn._qk_norm_identity = bool(q_id and k_id)
            if hasattr(attn, 'lambda_param') and hasattr(attn, 'set_identity'):
                attn.set_identity((attn.lambda_param == 0.0).all().item())
        n_identity = sum(1 for b in model.blocks
                         if getattr(b.attn, '_qk_norm_identity', True))
        if getattr(config, 'use_qk_norm', False):
            print(f"  [FastBuild] QK-Norm: {n_identity}/{len(model.blocks)} "
                  "layers identity (skipped)")

    @staticmethod
    def _post_quant_iri_fp4(model: "ConfigurableResearchLLM",
                            config: ModelConfig,
                            has_packed: bool = False) -> "ConfigurableResearchLLM":
        """Apply IRI-FP4 post-quantization if not already packed in checkpoint.

        Shared by both fast and traditional build paths.
        """
        if not getattr(config, 'use_iri_fp4', False) or has_packed:
            return model
        import time

        from forge.keys.quantization.iri_fp4_key import convert_model_to_iri_fp4
        _block = getattr(config, 'iri_fp4_block_size', 32)
        _rounds = getattr(config, 'iri_fp4_rounds', 2)
        t_q = time.time()
        model = convert_model_to_iri_fp4(
            model, block_size=_block, n_rounds=_rounds)
        t_q = time.time() - t_q
        n_iri = sum(1 for m in model.modules()
                    if m.__class__.__name__ == "IRIFP4Linear")
        print(f"  [FastBuild] IRI-FP4 post-quant: {n_iri} layers "
              f"({_rounds} rounds, {_block} block) in {t_q:.1f}s")
        return model

    @staticmethod
    def build_model_fast(config: ModelConfig, checkpoint_path: str | None = None,
                         compile: bool = False, moe_top_k: int | None = None,
                         dtype: torch.dtype | None = None,
                         fast_load: bool = True) -> "ConfigurableResearchLLM":
        """Fast model build — caches architecture, only loads weights.

        First call builds the architecture (~3s). Subsequent calls with the
        same config clone the cached model (~0.5s) and just load weights.

        moe_top_k: override MoE top-k routing (default: all experts).
                   Set to 2 for 4-expert model to halve FFN activations/VRAM.
        dtype: convert model to this dtype before loading weights (e.g. torch.bfloat16).
               Prevents upcasting bf16 checkpoint weights to fp32, saving ~50% VRAM.
        fast_load: when True (default), uses meta-device init + assign=True for
               3-6x faster cold boot. Skips parameter init kernels entirely by
               building the model on torch.device("meta"), then directly
               replaces meta params with state_dict tensors via
               load_state_dict(assign=True). Also starts OS page cache
               prefetch in a background thread. Set to False for the
               traditional build path (needed if checkpoint is missing or
               for debugging weight loading issues).
        """
        import time
        t0 = time.time()
        device = torch.device(config.device)
        sig = ModelLoader._config_signature(config)

        # Fast load path: meta init + assign=True + parallel weight load.
        # 5x faster than the traditional path (11.3s → 2.0s on V3).
        # Weight loading runs in a background thread, overlapping with meta init.
        if fast_load and checkpoint_path and os.path.exists(checkpoint_path):
            # Start state_dict load in background thread (overlaps with meta init).
            # The weight load is I/O bound (fastsafetensors reads + H2D copy),
            # so it can run concurrently with the CPU-bound meta init.
            import threading
            state_result = {}

            def _bg_load_state():
                try:
                    ckpt_path = Path(checkpoint_path)
                    if ckpt_path.is_dir():
                        sf_files = list(ckpt_path.glob("*.safetensors"))
                        if len(sf_files) == 1:
                            s = ModelLoader._load_safetensors_mmap(
                                str(sf_files[0]), None, device=device)
                        else:
                            s = ModelLoader._load_sharded_safetensors(
                                ckpt_path, None, device=device)
                    elif str(checkpoint_path).endswith(".safetensors"):
                        s = ModelLoader._load_safetensors_mmap(
                            checkpoint_path, None, device=device)
                    else:
                        from forge.checkpoint_io import load_checkpoint
                        s = load_checkpoint(checkpoint_path, map_location="cpu")
                        if isinstance(s, dict) and "model_state" in s \
                                and not any(k.startswith("blocks.") for k in s):
                            s = s["model_state"]
                    # Materialize NanoQuant/ASVD factorized linears (pre-remap)
                    if s and any("U_latent" in k for k in s):
                        s = ModelLoader._reconstruct_factorized_state(s)
                    # Auto-remap HF keys
                    if s and any(k.startswith("model.") for k in s):
                        s = ModelLoader._remap_hf_keys(s, config)
                    state_result["state"] = s
                except Exception as e:
                    state_result["error"] = e

            state_thread = threading.Thread(target=_bg_load_state, daemon=True)
            state_thread.start()

            # Meta init in main thread (overlaps with background weight load).
            # Caches the meta-init architecture so subsequent boots with the
            # same config skip the ~3-7s build and just deepcopy (~0.1s).
            t_arch = time.time()
            # Meta init is cheap (no real tensors, just shapes), so we skip
            # the deepcopy cache — deepcopy fails on modules with weight_norm
            # or other non-leaf tensor hooks. Rebuilding from meta is ~0.1s.
            cfg_meta = ModelConfig(**{**config.__dict__, "device": "meta"})
            with torch.device("meta"):
                model = ConfigurableResearchLLM(cfg_meta)
            t_arch = time.time() - t_arch
            print(f"  [FastBuild] Meta-init architecture in {t_arch:.1f}s")

            # Wait for background weight load to complete
            t_weights = time.time()
            state_thread.join()
            if "error" in state_result:
                raise state_result["error"]
            state = state_result.get("state", {})

            # Pre-packed ForgeQuant: <mod>.dense_packed keys mean the
            # checkpoint stores packed INT4 dense + INT8 sparse params.
            # Swap the corresponding nn.Linear modules for empty
            # ForgeQuantLinear shells — assign=True then writes the packed
            # tensors directly (no bf16 intermediate on GPU, plain-tensor
            # dequant stays CUDA-graph capturable).
            if any(k.endswith(".dense_packed") for k in state):
                from forge.engine.quant.forge_quant import ForgeQuantLinear
                # forge_convert.json {"fq_rotation": true} marks ckpts
                # packed with --rotate (Hadamard-rotated dense basis) —
                # the shells must rotate activations at forward to match.
                _fq_rot = False
                try:
                    _cdir = Path(checkpoint_path)
                    if not _cdir.is_dir():
                        _cdir = _cdir.parent
                    _convp = _cdir / "forge_convert.json"
                    if _convp.exists():
                        import json as _json
                        _fq_rot = bool(_json.loads(
                            _convp.read_text()).get("fq_rotation"))
                except Exception:
                    pass
                for name, mod in list(model.named_modules()):
                    dk = f"{name}.dense_packed"
                    if dk not in state:
                        continue
                    w = state.get(f"{name}.weight")
                    # dense_packed is (out, pad(in)//2) — group padding
                    # inflates in//2, so recover true in_features from
                    # sparse_packed (n_sparse, in) which stays unpadded.
                    out_f = state[dk].shape[0]
                    sp = state.get(f"{name}.sparse_packed")
                    in_f = (sp.shape[1] if sp is not None
                            else (w.shape[1] if w is not None
                                  else state[dk].shape[1] * 2))
                    has_bias = f"{name}.bias" in state
                    # ForgeQ-R residual patch: <mod>.res_a (out, r) self-
                    # describes the rank — no forge_convert flag needed.
                    _res_r = 0
                    _ra = state.get(f"{name}.res_a")
                    if _ra is not None and _ra.dim() == 2:
                        _res_r = _ra.shape[1]
                    fq = ForgeQuantLinear(
                        in_f, out_f, bias=has_bias,
                        group_size=getattr(config, 'forge_quant_group_size', 128),
                        sparse_ratio=getattr(config, 'forge_quant_sparse_ratio', 0.10),
                        use_rotation=_fq_rot,
                        residual_rank=_res_r)
                    # Prequant = memory-bounded path: keep weights packed and
                    # dequant per forward (plain ops → still graph-capturable).
                    fq._cache_w = False
                    # Replace on parent module
                    parent_name, _, attr = name.rpartition('.')
                    parent = (model.get_submodule(parent_name)
                              if parent_name else model)
                    setattr(parent, attr, fq)
                model._forgequant_prequant = True

            # NF4 QLoRA checkpoints (sft_train --qlora-nf4 saves): every
            # trained nn.Linear is stored as *.weight_packed (uint8,
            # 2 nibbles/byte, group-padded) + *.weight_scales (fp16).
            # Materialize NF4QuantLinear shells so the packed tensors load
            # directly — no bf16 detour and no double-quantization.
            if any(k.endswith(".weight_packed") and
                   f"{k[:-len('weight_packed')]}weight_scales" in state
                   for k in state):
                from forge.engine.quant.forge_quant import NF4QuantLinear
                for name, mod in list(model.named_modules()):
                    pk, sk = (f"{name}.weight_packed",
                              f"{name}.weight_scales")
                    if pk not in state or sk not in state:
                        continue
                    out_f = state[pk].shape[0]
                    in_f = (mod.in_features
                            if isinstance(mod, torch.nn.Linear)
                            else state[sk].shape[1] *
                            getattr(config, 'nf4_group_size', 64))
                    has_bias = f"{name}.bias" in state
                    nf = NF4QuantLinear(
                        in_f, out_f, bias=has_bias,
                        group_size=getattr(config, 'nf4_group_size', 64))
                    nf._cache_w = False
                    parent_name, _, attr = name.rpartition('.')
                    parent = (model.get_submodule(parent_name)
                              if parent_name else model)
                    setattr(parent, attr, nf)
                model._nf4_prequant = True
                # Treat like ForgeQuant for downstream activation
                # (skip post-load quantize; dequant ops are plain tensors
                # so CUDA-graph capture stays available).
                model._forgequant_prequant = True

            # Pre-quantized BitNet: int8 weights are loaded directly into
            # int8 storage via load_prequantized() — NO bf16/fp32 intermediate.
            # This keeps peak CPU RAM at ~checkpoint size (e.g. 4.7 GB for 4B
            # model) instead of 3x (int8 + bf16 cast + model copy).
            # int8 *.weight* tensors = BitNet prequant signature. ForgeQuant
            # int8 lives in *.sparse_packed — do NOT trip BitNet handling on it
            # (it would skip the embed/head re-tie below, leaving head.weight
            # as an fp32 meta remnant → bf16×fp32 matmul crash).
            _has_int8 = any(
                t.dtype == torch.int8 and k.endswith(".weight")
                for k, t in state.items())
            if _has_int8:
                from forge.keys.quantization.bitnet_b158_key import (
                    BitNetConv1d,
                    BitNetEmbedding,
                    BitNetLinear,
                )
                # Build module map: param_name → module
                bitnet_modules = {}
                for name, module in model.named_modules():
                    if isinstance(module, (BitNetLinear, BitNetConv1d, BitNetEmbedding)):
                        bitnet_modules[name + ".weight"] = module
                head_module = getattr(model, 'head', None)
                # Collect qscale tensors from checkpoint
                qscale_map = {}
                for k, t in state.items():
                    if k.endswith(".qscale") and t.dtype != torch.int8:
                        qscale_map[k[:-len(".qscale")] + ".weight"] = t
                # Route int8 weights to load_prequantized, collect rest for assign
                other_keys = {}
                int8_loaded = 0
                import torch.nn.functional as _F
                for k, t in state.items():
                    if k in bitnet_modules and t.dtype == torch.int8:
                        mod = bitnet_modules[k]
                        qs = qscale_map.get(k)
                        if qs is None:
                            qs = t.float().abs().mean().clamp(min=1e-8) / 0.7
                        mod.load_prequantized(
                            t.to(device).to(torch.int8), qs.to(device))
                        int8_loaded += 1
                    elif k == "head.weight" and t.dtype == torch.int8 and head_module is not None:
                        dev = torch.device(device)
                        w_int8 = t.to(dev).to(torch.int8)
                        qs = qscale_map.get(k)
                        if qs is None:
                            qs = t.float().abs().mean().clamp(min=1e-8) / 0.7
                        qs = qs.to(dev)
                        del head_module.weight
                        head_module.register_buffer("weight_int8", w_int8)
                        head_module.register_buffer("qscale_buf", qs)
                        head_module._prequantized = True
                        _orig_fwd = head_module.forward
                        def _int8_fwd(x, _w=w_int8, _s=qs, _b=head_module.bias):
                            return _F.linear(x, _w.to(x.dtype), _b) * _s.to(x.dtype)
                        head_module.forward = _int8_fwd
                        int8_loaded += 1
                    elif k.endswith(".qscale") and (
                        k[:-len(".qscale")] + ".weight" in bitnet_modules
                        or k[:-len(".qscale")] + ".weight" == "head.weight"):
                        continue  # qscale already handled
                    else:
                        other_keys[k] = t
                # Replace state with only non-int8 tensors for the assign path
                state = other_keys
                if int8_loaded > 0:
                    print(f"  [FastBuild] Direct int8 load: {int8_loaded} BitNet layers (no bf16 intermediate)")

            # V10: IRI-FP4 pre-quantized checkpoint — load packed FP4 data
            # directly into IRIFP4Linear modules (keeps weights packed in
            # VRAM, dequantizes on-the-fly during forward pass).
            # This saves ~3.5x VRAM vs bf16 for the quantized weight tensors.
            _has_iri = any(k.endswith(".iri_packed") for k in state)
            if _has_iri and getattr(config, 'use_iri_fp4', False):
                from forge.keys.quantization.iri_fp4_key import IRIFP4Linear
                _block = getattr(config, 'iri_fp4_block_size', 32)
                _rounds = getattr(config, 'iri_fp4_rounds', 2)
                # Find which base keys have IRI-FP4 packed data in checkpoint
                # Checkpoint stores: {module_name}.weight.iri_packed
                # Module name in model: {module_name} (without .weight)
                iri_weight_keys = set()
                for k in state:
                    if k.endswith(".iri_packed"):
                        # e.g. "blocks.0.ffn.w_gate.weight.iri_packed" -> "blocks.0.ffn.w_gate.weight"
                        iri_weight_keys.add(k[:-len(".iri_packed")])
                # Map: module_name -> weight_key (module_name + ".weight")
                iri_module_names = set(k[:-len(".weight")] for k in iri_weight_keys
                                       if k.endswith(".weight"))
                # Replace only the nn.Linear modules that have packed data
                iri_modules = {}
                for name, mod in list(model.named_modules()):
                    if isinstance(mod, nn.Linear) and name in iri_module_names:
                        new_mod = IRIFP4Linear(
                            mod.in_features, mod.out_features,
                            bias=mod.bias is not None,
                            block_size=_block, n_rounds=_rounds)
                        # Replace in parent
                        parent = model
                        parts = name.split(".")
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], new_mod)
                        iri_modules[name] = new_mod
                # Load packed data from checkpoint into IRIFP4Linear modules
                # Checkpoint keys: {module_name}.weight.iri_packed, .iri_scales, etc.
                # Module lookup: by module_name (without .weight)
                iri_loaded = 0
                other_keys = {}
                for k, t in state.items():
                    if k.endswith(".iri_packed"):
                        # "blocks.0.ffn.w_gate.weight.iri_packed" -> "blocks.0.ffn.w_gate"
                        weight_key = k[:-len(".iri_packed")]  # blocks.0.ffn.w_gate.weight
                        mod_name = weight_key[:-len(".weight")] if weight_key.endswith(".weight") else weight_key
                        mod = iri_modules.get(mod_name)
                        if mod is not None:
                            mod.weight_packed = t.to(device)
                            iri_loaded += 1
                    elif k.endswith(".iri_scales"):
                        weight_key = k[:-len(".iri_scales")]
                        mod_name = weight_key[:-len(".weight")] if weight_key.endswith(".weight") else weight_key
                        mod = iri_modules.get(mod_name)
                        if mod is not None:
                            mod.weight_scales = t.to(device)
                    elif k.endswith(".iri_global_scale"):
                        weight_key = k[:-len(".iri_global_scale")]
                        mod_name = weight_key[:-len(".weight")] if weight_key.endswith(".weight") else weight_key
                        mod = iri_modules.get(mod_name)
                        if mod is not None:
                            mod.weight_global_scale = t.to(device)
                    elif k.endswith(".bias") and k[:-len(".bias")] in iri_modules:
                        mod = iri_modules[k[:-len(".bias")]]
                        if mod.bias is not None:
                            mod.bias = t.to(device)
                    else:
                        other_keys[k] = t
                state = other_keys
                if iri_loaded > 0:
                    print(f"  [FastBuild] IRI-FP4 packed load: {iri_loaded} layers "
                          f"(stays packed in VRAM, {_rounds} rounds)")

            # GQA -> diff/GTA/GLA warm start + Mamba-2 -> Mamba-3 + FFN compression
            state = ModelLoader._warm_start_attention(state, config)
            state = ModelLoader._warm_start_mamba3(state, config)
            state = ModelLoader._warm_start_conv(state, config)
            state = ModelLoader._warm_start_zero_center(state, model)
            state = ModelLoader._apply_ffn_compression(state, config)

            t_weights = time.time() - t_weights

            # assign=True: directly replace meta params with state_dict tensors.
            # Skips the copy-into-existing-storage path of normal load_state_dict.
            t_gpu = time.time()
            missing, unexpected = model.load_state_dict(
                state, strict=False, assign=True)
            # Re-tie weights (assign breaks parameter sharing; head.weight
            # is not in the checkpoint because of weight tying).
            # Skip re-tying when head/embed are prequantized int8 (no weight param).
            if getattr(config, 'tie_word_embeddings', True) \
                    and not getattr(config, 'use_pit', False) \
                    and not _has_int8:
                model.head.weight = model.embed.weight
            # Re-initialize non-persistent buffers (RoPE cos/sin) left on meta.
            ModelLoader._reset_non_persistent_buffers(model, device)
            # Move any parameters still on meta or wrong device (e.g. v_mix_gate
            # added by GTA key transform on CPU) to the target device.
            # Preserve values for params that have data (from key transforms);
            # zero-init only for params that are still on meta (no data).
            # NOTE: must replace the Parameter object in the parent module's
            # _parameters dict — PyTorch 2.11+ rejects `param.data = cuda_tensor`
            # on a meta tensor ("incompatible tensor type").
            for module in model.modules():
                for pname, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    if param.is_meta:
                        module._parameters[pname] = torch.nn.Parameter(
                            torch.zeros(param.shape, dtype=param.dtype, device=device),
                            requires_grad=param.requires_grad)
                    elif param.device != device:
                        module._parameters[pname] = torch.nn.Parameter(
                            param.data.to(device),
                            requires_grad=param.requires_grad)
            # Move any buffers still on meta or wrong device to target device
            for module in model.modules():
                for bname, buf in module._buffers.items():
                    if buf is not None:
                        if buf.is_meta:
                            module._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
                        elif buf.device != device:
                            module._buffers[bname] = buf.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu = time.time() - t_gpu

            # ── Free the state dict + background load result ──
            # After assign=True + move-to-device, the model params are new
            # GPU tensors. The state dict still holds the original CPU tensors
            # (up to 10.5 GB for 5B param int8→bf16 cast). On Windows, Python's
            # allocator doesn't return freed memory to the OS automatically,
            # so we must explicitly del + gc.collect() to release it.
            del state, state_result
            import gc as _gc
            _gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Convert dtype (state may already be bf16 from fastsafetensors)
            # Skip dtype conversion when int8 weights were loaded directly —
            # model.to(bf16) would upcast int8 buffers back to bf16.
            if dtype is not None and not _has_int8:
                model = model.to(dtype)

            if missing:
                # head.weight is expected to be missing (weight tying)
                real_missing = [k for k in missing if k != "head.weight"]
                if real_missing:
                    print("Missing keys:", real_missing[:5],
                          "..." if len(real_missing) > 5 else "")
            if unexpected:
                print("Unexpected keys:", unexpected[:5],
                      "..." if len(unexpected) > 5 else "")

            # Post-load QK-norm + diff-attn identity scan
            ModelLoader._scan_qk_norm_identity(model, config)
            ModelLoader._refresh_fused_qkv(model)

            # ── Post-load IRI-FP4 quantization (V2 default quant) ──
            # If use_iri_fp4=True but the checkpoint didn't have .iri_packed
            # keys (non-quantized checkpoint or random init), quantize all
            # 2D nn.Linear weights to IRI-FP4 now. This makes V2 quantized by
            # default — any V2 build ends up with IRIFP4Linear modules.
            model = ModelLoader._post_quant_iri_fp4(
                model, config, has_packed=_has_iri)

            t_total = time.time() - t0
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  [FastBuild] Weights: {t_weights:.1f}s | assign: {t_gpu:.1f}s | "
                  f"Total: {t_total:.1f}s ({param_count:.1f}M params)")
            return model

        # Traditional build path (fast_load=False or no checkpoint)
        # Build or clone architecture
        if sig not in ModelLoader._model_cache:
            t_arch = time.time()
            model = ConfigurableResearchLLM(config).to(device)
            # μScaling: unit-variance init for FP8 training stability.
            # Only applies to fresh models (not checkpoint loading).
            if getattr(config, 'use_mu_scaling', False):
                from forge.training.optim.fp8_training import mu_scale_init
                mu_scale_init(model, verbose=True)
            # Cache on CPU to avoid VRAM duplication on deepcopy
            ModelLoader._model_cache[sig] = model.cpu()
            ModelLoader._model_cache.move_to_end(sig)
            # LRU eviction: keep at most _MODEL_CACHE_MAXSIZE architectures.
            while len(ModelLoader._model_cache) > ModelLoader._MODEL_CACHE_MAXSIZE:
                ModelLoader._model_cache.popitem(last=False)
            model = model.to(device)
            t_arch = time.time() - t_arch
            print(f"  [FastBuild] Architecture built in {t_arch:.1f}s (cached)")
        else:
            t_arch = time.time()
            cached = ModelLoader._model_cache[sig]
            ModelLoader._model_cache.move_to_end(sig)  # mark as recently used
            # Deep copy the cached model (CPU — no VRAM overhead)
            model = copy.deepcopy(cached).to(device)
            t_arch = time.time() - t_arch
            print(f"  [FastBuild] Architecture cloned in {t_arch:.1f}s (from cache)")

        # Convert dtype before loading weights (prevents bf16→fp32 upcast)
        if dtype is not None:
            model = model.to(dtype)

        if compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            t_weights = time.time()
            ckpt_path = Path(checkpoint_path)

            # Handle sharded models (directory with model.safetensors.index.json)
            if ckpt_path.is_dir():
                index_file = ckpt_path / "model.safetensors.index.json"
                if index_file.exists():
                    state = ModelLoader._load_sharded_safetensors(
                        ckpt_path, model, device=device)
                else:
                    # Single safetensors in directory
                    sf_files = list(ckpt_path.glob("*.safetensors"))
                    if len(sf_files) == 1:
                        state = ModelLoader._load_safetensors_mmap(
                            str(sf_files[0]), model, device=device)
                    else:
                        state = ModelLoader._load_sharded_safetensors(
                            ckpt_path, model, device=device)
            elif str(checkpoint_path).endswith(".safetensors"):
                state = ModelLoader._load_safetensors_mmap(
                    checkpoint_path, model, device=device)
            else:
                from forge.checkpoint_io import load_checkpoint
                state = load_checkpoint(checkpoint_path, map_location="cpu")
                if isinstance(state, dict) and "model_state" in state and not any(k.startswith("blocks.") for k in state):
                    state = state["model_state"]

            # Materialize NanoQuant/ASVD factorized linears to dense weights
            # (must run BEFORE the HF remap — factorized keys are HF-style).
            if state and any("U_latent" in k for k in state):
                state = ModelLoader._reconstruct_factorized_state(state)

            # Auto-detect and remap HuggingFace keys to ForgeAI internal names.
            if state and any(k.startswith("model.") for k in state):
                state = ModelLoader._remap_hf_keys(state, config)
                print(f"  [FastBuild] Remapped {len(state)} HF keys to ForgeAI format")

            # Lossless GQA -> diff/GTA/GLA warm start + Mamba-2 -> Mamba-3 + FFN compression.
            state = ModelLoader._warm_start_attention(state, config)
            state = ModelLoader._warm_start_mamba3(state, config)
            state = ModelLoader._warm_start_conv(state, config)
            state = ModelLoader._warm_start_zero_center(state, model)
            state = ModelLoader._apply_ffn_compression(state, config)

            t_weights = time.time() - t_weights

            # load_state_dict copies weights into model parameters (GPU→GPU
            # if weights were loaded directly to device, CPU→GPU otherwise).
            t_gpu = time.time()
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
            if unexpected:
                print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu = time.time() - t_gpu

            # Post-load: detect non-identity QK-Norm weights.
            ModelLoader._scan_qk_norm_identity(model, config)
            ModelLoader._refresh_fused_qkv(model)

            print(f"  [FastBuild] Weights: {t_weights:.1f}s | GPU transfer: {t_gpu:.1f}s")
        elif checkpoint_path:
            print(f"Warning: checkpoint {checkpoint_path} not found, using random weights.")
        else:
            print("Warning: no checkpoint_path given — model has RANDOM weights.")

        # ── Post-load IRI-FP4 quantization (V2 default quant) ──
        # Same as the fast path: if use_iri_fp4=True and the checkpoint
        # didn't contain .iri_packed keys, quantize now.
        _has_iri_trad = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            _has_iri_trad = any(
                k.endswith(".iri_packed")
                for k in (state if 'state' in locals() else {})
            )
        model = ModelLoader._post_quant_iri_fp4(
            model, config, has_packed=_has_iri_trad)

        t_total = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"  [FastBuild] Architecture: {t_arch:.1f}s | Weights: {t_weights:.1f}s | "
                  f"GPU transfer: {t_gpu:.1f}s | Total: {t_total:.1f}s ({param_count:.1f}M params)")
        else:
            print(f"  [FastBuild] Total: {t_total:.1f}s ({param_count:.1f}M params)")
        return model

    @staticmethod
    def hybrid_offload(model: nn.Module, gpu_layers: int = -1,
                       device: str = "cuda") -> nn.Module:
        """Offload specific layers to GPU, keep rest on CPU.

        For LFM2.5 hybrid models, conv layers are cheap (O(T*k*d)) and can
        run on CPU, while attention layers benefit from GPU (O(T^2*d) matmuls).
        This enables running larger models on limited VRAM.

        Args:
            model: the model to offload (must have .blocks ModuleList)
            gpu_layers: number of layers to put on GPU (from the end).
                       -1 = put attention layers on GPU, conv on CPU.
                       N = last N layers on GPU, rest on CPU.
            device: GPU device string ("cuda", "cuda:0", etc.)

        Returns:
            The model with per-layer device placement applied.
        """
        gpu_dev = torch.device(device)
        cpu_dev = torch.device("cpu")

        if not hasattr(model, 'blocks'):
            return model.to(gpu_dev)

        layer_types = getattr(model.config, 'layer_types', None)

        # Always keep embed, head, ln_f on GPU
        model.embed = model.embed.to(gpu_dev)
        model.head = model.head.to(gpu_dev)
        if hasattr(model, 'ln_f'):
            model.ln_f = model.ln_f.to(gpu_dev)

        # Auxiliary modules that participate in forward pass — keep on GPU
        # so they don't cause device mismatch with GPU-resident blocks.
        for attr in ('loop_block', 'lisa', 'mtp_module', '_attn_res',
                     '_hyperloop', 'loop_gates', 'middle_gates', '_v0_gates'):
            mod = getattr(model, attr, None)
            if mod is not None and hasattr(mod, 'to'):
                mod.to(gpu_dev)

        gpu_count = 0
        cpu_count = 0
        for i, block in enumerate(model.blocks):
            if gpu_layers == -1 and layer_types:
                # Auto: attention on GPU, conv on CPU
                lt = layer_types[i] if i < len(layer_types) else "attention"
                target = gpu_dev if lt == "attention" else cpu_dev
            elif gpu_layers == -1:
                target = gpu_dev
            else:
                # Last N layers on GPU
                target = gpu_dev if i >= len(model.blocks) - gpu_layers else cpu_dev

            block.to(target)
            if target == gpu_dev:
                gpu_count += 1
            else:
                cpu_count += 1

        # Expert tying fix: tied expert pairs share the same Parameter objects.
        # When hybrid offload puts paired layers on different devices (e.g.,
        # layer 2=attention→GPU, layer 3=conv→CPU), the last .to() wins and
        # experts end up on the wrong device. Fix: for each tied pair, move
        # experts to the GPU layer's device (attention layers need GPU).
        if getattr(model.config, 'moe_expert_tying', False):
            tie_g = getattr(model.config, 'moe_tie_group_size', 2)
            for even_idx in range(0, len(model.blocks), tie_g):
                odd_idx = even_idx + 1
                if odd_idx >= len(model.blocks):
                    break
                even_moe = getattr(model.blocks[even_idx].ffn, 'experts', None)
                odd_moe = getattr(model.blocks[odd_idx].ffn, 'experts', None)
                if even_moe is None or odd_moe is None:
                    continue
                # Check if they're actually tied (same object)
                if len(even_moe) > 0 and len(odd_moe) > 0 \
                        and even_moe[0] is odd_moe[0]:
                    # Tied: move experts to whichever block is on GPU
                    even_dev = next(model.blocks[even_idx].parameters()).device
                    odd_dev = next(model.blocks[odd_idx].parameters()).device
                    if even_dev.type == 'cuda':
                        for exp in even_moe:
                            exp.to(gpu_dev)
                        if hasattr(model.blocks[odd_idx].ffn, 'shared'):
                            model.blocks[odd_idx].ffn.shared.to(gpu_dev)
                    elif odd_dev.type == 'cuda':
                        for exp in odd_moe:
                            exp.to(gpu_dev)
                        if hasattr(model.blocks[odd_idx].ffn, 'shared'):
                            model.blocks[odd_idx].ffn.shared.to(gpu_dev)

        # Placement changed — drop any cached device scan from a prior forward.
        if hasattr(model, 'invalidate_device_cache'):
            model.invalidate_device_cache()

        print(f"  [HybridOffload] {gpu_count} layers on {device}, {cpu_count} on CPU")
        return model

    @staticmethod
    def build_model(config: ModelConfig, checkpoint_path: str | None = None, compile: bool = False) -> "ConfigurableResearchLLM":
        print(
            f"Building {config.d_model}d x {config.n_layers}L {config.attn_type.upper()} model "
            f"(FFN: {config.ffn_type.upper()}, draft: {config.enable_draft_head})..."
        )
        device = torch.device(config.device)
        model = ConfigurableResearchLLM(config).to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total parameters: {param_count:.2f}M")

        if compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            from forge.checkpoint_io import load_checkpoint
            state = load_checkpoint(checkpoint_path, map_location=device)
            # If the checkpoint was saved with metadata wrapper (e.g. dpo_align),
            # unwrap the model_state key.
            if isinstance(state, dict) and "model_state" in state and not any(k.startswith("blocks.") for k in state):
                state = state["model_state"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys:", missing)
            if unexpected:
                print("Unexpected keys:", unexpected)
            print(f"Loaded checkpoint from {checkpoint_path}")
        elif checkpoint_path:
            print(f"Warning: checkpoint {checkpoint_path} not found, using random weights.")

        return model

    @staticmethod
    def generate_text(
        model: ConfigurableResearchLLM,
        tokenizer: "Any",
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int | None = None,
    ) -> str:
        """Generate text with pre-allocated KV cache and batched EOS check.

        Optimizations vs the original loop:
        - PreAllocatedKVCache: O(1) append per token instead of O(n) torch.cat
        - GPU-side EOS check: (next_token == eos_id).any() — one sync instead of .item() per token
        - Pre-allocated output buffer: write by index instead of torch.cat per token
        """
        model.eval()
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        # Handle both HF BatchEncoding (.to()) and plain dict (gigatoken)
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        prompt_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        B, prompt_len = prompt_ids.shape
        eos_id = tokenizer.eos_token_id

        # Pre-allocate output buffer (prompt + max_new_tokens).
        max_total = prompt_len + max_new_tokens
        out_ids = torch.zeros(B, max_total, dtype=prompt_ids.dtype, device=device)
        out_ids[:, :prompt_len] = prompt_ids

        cache = create_kv_cache(model, max_total, batch=B, device=device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                pos = prompt_len + step
                if step == 0:
                    # Prefill: feed the full prompt.
                    # use_cache=True so conv layers initialize their state buffer.
                    idx_cond = out_ids[:, :prompt_len]
                    out = model(idx_cond, preallocated_cache=cache, use_cache=True)
                    logits = out[0]
                else:
                    # Decode: feed only the last generated token.
                    # use_cache=True so conv layers use incremental conv with state.
                    idx_cond = out_ids[:, pos - 1:pos]
                    out = model(idx_cond, preallocated_cache=cache, use_cache=True)
                    logits = out[0]

                logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                out_ids[:, pos:pos + 1] = next_token

                # Batch EOS check on GPU — one .any() sync instead of .item() per token.
                if eos_id is not None and (next_token == eos_id).any().item():
                    break

        # Decode only the generated portion (trim trailing zeros).
        actual_len = prompt_len + step + 1
        return tokenizer.decode(out_ids[0, :actual_len], skip_special_tokens=True)



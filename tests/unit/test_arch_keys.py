"""Tests for the 2025/2026 architecture keys:

BitNet b1.58 QAT, Differential Attention, TITAN neural memory, MoD routing.
All run on GPU (CUDA) with bf16. CPU fallback only if CUDA unavailable.
"""

import pytest
import torch

from forge.config import get_config
from forge.keys.architecture.mod_router_key import ModRouter, ModRouterKey
from forge.keys.architecture.titan_memory_key import TitanMemory, TitanMemoryKey
from forge.keys.attention.differential_attn_key import (
    DifferentialAttentionKey,
    paper_lambda_init,
)
from forge.keys.quantization.bitnet_b158_key import (
    BitNetB158Key,
    BitNetLinear,
    apply_bitnet_b158,
    ternary_quantize,
)
from forge.model_loader import ConfigurableResearchLLM, create_kv_cache

_CUDA = torch.cuda.is_available()
_DEV = "cuda" if _CUDA else "cpu"
_DTYPE = torch.bfloat16 if _CUDA else torch.float32


def _tiny(device=_DEV):
    cfg = get_config("forgelm_tiny")
    cfg.device = device
    cfg.dtype = "bfloat16" if _CUDA else "float32"
    return cfg


# ── BitNet b1.58 ─────────────────────────────────────────────────────────────


class TestBitNet:
    def test_ternary_values(self):
        w = torch.randn(8, 8)
        q, scale = ternary_quantize(w)
        vals = torch.unique(q)
        assert set(vals.tolist()) <= {-1.0, 0.0, 1.0}
        assert scale > 0

    def test_linear_lossless_when_off(self):
        lin = BitNetLinear(16, 32, quantize=False)
        x = torch.randn(2, 4, 16)
        y = lin(x)
        expected = torch.nn.functional.linear(x, lin.weight)
        assert torch.allclose(y, expected)

    def test_ste_backward(self):
        lin = BitNetLinear(16, 32, quantize=True)
        x = torch.randn(2, 4, 16)
        y = lin(x).pow(2).mean()
        y.backward()
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()

    def test_state_dict_key_compat(self):
        # weight keeps its name -> existing checkpoints load as-is;
        # qscale is optional (learned scale) and re-inits from absmean when
        # missing (strict=False).
        lin = BitNetLinear(16, 32, quantize=True)
        assert set(lin.state_dict().keys()) == {"weight", "qscale"}
        lin_no_scale = BitNetLinear(16, 32, quantize=True, learned_scale=False)
        assert set(lin_no_scale.state_dict().keys()) == {"weight"}

    def test_apply_bitnet_offline(self):
        state = {"blocks.0.ffn.w_gate.weight": torch.randn(8, 8), "step": 5}
        out = apply_bitnet_b158(state)
        vals = torch.unique(out["blocks.0.ffn.w_gate.weight"])
        assert set(vals.tolist()) <= {-1.0, 0.0, 1.0}
        assert out["step"] == 5  # non-tensors pass through

    def test_key_forward_reverse(self):
        key = BitNetB158Key()
        state = {"blocks.0.ffn.w_gate.weight": torch.randn(8, 8)}
        res = key.forward(state)
        assert res.success and res.weights is not None
        rev = key.reverse(res.weights)
        assert rev.success

    def test_ffn_integration(self):
        cfg = _tiny()
        cfg.use_bitnet = True
        model = ConfigurableResearchLLM(cfg)
        model.train()
        x = torch.randint(0, 256, (2, 8))
        loss = model(x, targets=x)[1]
        loss.backward()
        assert loss.requires_grad

    def test_eval_is_lossless_with_qat_gating(self):
        """Ternary only in training; eval uses fp master weights."""
        cfg = _tiny()
        cfg.use_bitnet = True
        bitnet = ConfigurableResearchLLM(cfg).eval()
        plain = ConfigurableResearchLLM(_tiny()).eval()
        bitnet.load_state_dict(plain.state_dict(), strict=False)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            l_plain, _ = plain(x)
            l_bitnet, _ = bitnet(x)
        assert torch.allclose(l_plain, l_bitnet, atol=1e-6)

    def test_training_quantizes(self):
        cfg = _tiny()
        cfg.use_bitnet = True
        model = ConfigurableResearchLLM(cfg)
        model.train()
        ffn = model.blocks[0].ffn
        assert ffn.w_gate.qscale is not None  # learned scale param
        torch.randn(1, 2, 128)
        q, _ = ternary_quantize(ffn.w_gate.weight, ffn.w_gate.qscale)
        vals = set(q.unique().tolist())
        assert vals <= {-1.0, 0.0, 1.0}

    def test_qscale_reanchored_on_load(self):
        """qscale must follow the LOADED weight, not the random init."""
        lin = BitNetLinear(16, 32, quantize=True)
        w = torch.randn(32, 16) * 0.5
        lin.load_state_dict({"weight": w}, strict=False)
        expected = w.abs().mean().clamp(min=1e-6) / 0.7
        assert torch.allclose(lin.qscale, expected, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_int8_kernel_matches_fp_path(self):
        """Integer tensor-core GEMM ≈ fp ternary path (a4.8 activation q)."""
        from forge.keys.quantization.bitnet_b158_key import (
            _int8_ternary_linear,
        )

        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, 64, device="cuda") * 0.1
        q, scale = ternary_quantize(w)
        y_k = _int8_ternary_linear(x, q, scale)
        y_fp = torch.nn.functional.linear(x, q.to(x.dtype)) * scale.to(x.dtype)
        assert y_k.shape == y_fp.shape
        # activation int8 quantization error is ~1% relative
        rel = (y_k - y_fp.float()).abs() / (y_fp.float().abs() + 1e-6)
        assert rel.mean().item() < 0.05

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_triton_add_kernel_matches_fp(self):
        """b1.58 add-only Triton kernel (fp activations) ≈ fp ternary path."""
        from forge.keys.quantization.bitnet_b158_key import (
            _HAS_TRITON,
            _triton_ternary_linear,
        )

        if not _HAS_TRITON:
            pytest.skip("triton not installed")
        torch.manual_seed(0)
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, 64, device="cuda") * 0.1
        q, scale = ternary_quantize(w)
        y_k = _triton_ternary_linear(x, q, scale)
        y_fp = torch.nn.functional.linear(x, q.to(x.dtype)) * scale.to(x.dtype)
        assert torch.equal(y_k, y_fp)  # fp32 accum, exact ternary weights

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_triton_path_trains(self):
        """FORGE_BITNET_KERNEL=triton: end-to-end QAT step on the kernel."""
        from forge.keys.quantization.bitnet_b158_key import _HAS_TRITON

        if not _HAS_TRITON:
            pytest.skip("triton not installed")
        import os

        os.environ["FORGE_BITNET_KERNEL"] = "triton"
        try:
            cfg = _tiny()
            cfg.use_bitnet = True
            model = ConfigurableResearchLLM(cfg).to("cuda")
            model.train()
            x = torch.randint(0, 256, (2, 8), device="cuda")
            loss = model(x, targets=x)[1]
            loss.backward()
            assert torch.isfinite(loss).item()
            assert model.blocks[0].ffn.w_gate.weight.grad is not None
        finally:
            os.environ.pop("FORGE_BITNET_KERNEL", None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_kernel_used_in_training(self):
        cfg = _tiny()
        cfg.use_bitnet = True
        model = ConfigurableResearchLLM(cfg).to("cuda")
        model.train()
        x = torch.randint(0, 256, (2, 8), device="cuda")
        loss = model(x, targets=x)[1]
        loss.backward()
        ffn = model.blocks[0].ffn
        assert ffn.w_gate.weight.grad is not None
        assert ffn.w_gate.qscale.grad is not None
        assert torch.isfinite(loss).item()


# ── Differential Attention ───────────────────────────────────────────────────


class TestDiffAttn:
    def test_module_forward_and_backward(self):
        cfg = _tiny()
        cfg.attn_type = "diff"
        model = ConfigurableResearchLLM(cfg)
        x = torch.randint(0, 256, (2, 8))
        model.train()
        loss = model(x, targets=x)[1]
        loss.backward()
        assert torch.isfinite(loss).item()

    def test_inference_with_kv_cache(self):
        cfg = _tiny()
        cfg.attn_type = "diff"
        model = ConfigurableResearchLLM(cfg).eval()
        cache = create_kv_cache(model, 16, batch=1, device="cpu")
        ids = torch.randint(0, 256, (1, 6))
        with torch.inference_mode():
            logits1, _ = model(ids, preallocated_cache=cache)
            logits2, _ = model(ids[:, -1:], preallocated_cache=cache)
        assert logits1.shape == (1, 6, 256)
        assert logits2.shape == (1, 1, 256)

    def test_key_dup_avg_roundtrip(self):
        q = torch.randn(8, 32)  # 2 heads * hd(4) * 2? just rows
        state = {"blocks.0.attn.q_proj.weight": q, "blocks.0.attn.k_proj.weight": q.clone()}
        key = DifferentialAttentionKey(n_layers=1, n_heads=2)
        res = key.forward(state)
        assert res.success
        dup = res.weights["blocks.0.attn.q_proj.weight"]
        assert dup.shape[0] == 2 * q.shape[0]
        assert "blocks.0.attn.lambda_param" in res.weights
        assert res.weights["blocks.0.attn.lambda_param"].shape[0] == 2
        # identity mode: lambda == 0 (lossless warm start)
        assert res.weights["blocks.0.attn.lambda_param"].abs().sum() == 0
        # reverse averages back
        rev = key.reverse(res.weights)
        assert torch.allclose(rev.weights["blocks.0.attn.q_proj.weight"], q)

    def test_gqa_to_diff_lossless_conversion(self):
        """Duplicated rows + lambda=0 must be bit-exact vs the GQA model."""
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = _tiny(dev)  # tiny preset: gqa + qk-norm (identity at init)
        gqa = ConfigurableResearchLLM(cfg).to(dev).eval()

        cfg_diff = _tiny(dev)
        cfg_diff.attn_type = "diff"
        diff = ConfigurableResearchLLM(cfg_diff).to(dev).eval()

        key = DifferentialAttentionKey(n_layers=cfg.n_layers, n_heads=cfg.n_heads, identity=True)
        res = key.forward(gqa.state_dict())
        assert res.success
        diff.load_state_dict({k: v.to(dev) for k, v in res.weights.items()}, strict=False)
        # sync identity + qk-norm flags like ModelLoader does
        for b in diff.blocks:
            attn = b.attn
            if hasattr(attn, "set_identity"):
                attn.set_identity((attn.lambda_param == 0.0).all().item())

        x = torch.randint(0, 256, (2, 8), device=dev)
        with torch.no_grad():
            l_gqa, _ = gqa(x)
            l_diff, _ = diff(x)
        # GQA uses manual softmax attention on CPU (different numerics than
        # SDPA by ~1e-7); on CUDA both paths use SDPA -> bit-exact.
        if dev == "cuda":
            assert torch.equal(l_gqa, l_diff), "diff warm start must be bit-exact"
        else:
            assert torch.allclose(l_gqa, l_diff, atol=1e-6)

    def test_paper_lambda_in_range(self):
        for layer_i in range(16):
            v = paper_lambda_init(16, layer_i)
            assert 0.0 < v < 0.8


# ── TITAN memory ─────────────────────────────────────────────────────────────


class TestTitan:
    def test_lossless_at_start(self):
        cfg = _tiny()
        cfg.use_titan_memory = True
        plain = ConfigurableResearchLLM(_tiny()).eval()
        with_mem = ConfigurableResearchLLM(cfg).eval()
        # identical init weights? zero-init memory only; but random init of
        # the rest differs — force same weights for the shared params.
        with_mem.load_state_dict(plain.state_dict(), strict=False)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            p1, _ = plain(x)
            p2, _ = with_mem(x)
        assert torch.allclose(p1, p2, atol=1e-6), "TITAN must be lossless at init"

    def test_update_changes_output(self):
        cfg = _tiny()
        cfg.use_titan_memory = True
        model = ConfigurableResearchLLM(cfg)
        mem = model.blocks[0]._memory
        assert mem is not None
        x = torch.randn(2, 4, cfg.d_model)
        mem.update(x)
        # memory weights moved off zero
        assert mem.memory.abs().sum().item() > 0
        # with the gate open, the memory read of x is non-zero
        mem.gate.data.fill_(1.0)
        out = mem.forward(x)
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    def test_key_add_remove(self):
        key = TitanMemoryKey(d_model=128)
        state = {"blocks.0.ln1.weight": torch.ones(128), "blocks.1.ln1.weight": torch.ones(128)}
        res = key.forward(state)
        assert res.success
        assert "blocks.0.memory.memory" in res.weights
        assert "blocks.1.memory.gate" in res.weights
        rev = key.reverse(res.weights)
        assert not any(".memory." in k for k in rev.weights)

    def test_module_lossless_gate(self):
        mem = TitanMemory(64)
        x = torch.randn(2, 4, 64)
        out = mem.forward(x)
        assert torch.allclose(out, torch.zeros_like(x), atol=1e-7)


# ── Mixture-of-Depths ────────────────────────────────────────────────────────


class TestMod:
    def test_lossless_at_keep_one(self):
        cfg = _tiny()
        cfg.use_mod = True
        cfg.mod_keep_fraction = 1.0
        plain = ConfigurableResearchLLM(_tiny()).eval()
        mod = ConfigurableResearchLLM(cfg).eval()
        mod.load_state_dict(plain.state_dict(), strict=False)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            p1, _ = plain(x)
            p2, _ = mod(x)
        assert torch.allclose(p1, p2, atol=1e-6), "MoD 1.0 must be lossless"

    def test_mask_fraction(self):
        router = ModRouter(64, keep_fraction=0.5)
        x = torch.randn(2, 8, 64)
        mask = router.token_mask(x)
        assert mask is not None
        assert mask.shape == (2, 8)
        assert mask.sum(dim=-1).tolist() == [4, 4]

    def test_apply_gates(self):
        router = ModRouter(64, keep_fraction=0.5)
        x = torch.randn(2, 8, 64)
        update = torch.randn(2, 8, 64)
        gated = router.apply(x, update)  # returns the gated update
        mask = router.token_mask(x)
        # skipped tokens: zero update; kept tokens: full update
        assert torch.allclose(gated[~mask], torch.zeros_like(gated[~mask]), atol=1e-6)
        assert torch.allclose(gated[mask], update[mask], atol=1e-6)
        # new residual = x + gated update
        out = x + gated
        assert torch.allclose(out[~mask], x[~mask], atol=1e-6)

    def test_training_runs_with_skips(self):
        cfg = _tiny()
        cfg.use_mod = True
        cfg.mod_keep_fraction = 0.6
        model = ConfigurableResearchLLM(cfg)
        model.train()
        x = torch.randint(0, 256, (2, 8))
        loss = model(x, targets=x)[1]
        loss.backward()
        assert torch.isfinite(loss).item()

    def test_key_add_remove(self):
        key = ModRouterKey()
        state = {"blocks.0.ln1.weight": torch.ones(128), "blocks.1.ln1.weight": torch.ones(128)}
        res = key.forward(state)
        assert res.success
        assert res.weights["blocks.0.mod.router.weight"].shape == (1, 128)
        rev = key.reverse(res.weights)
        assert not any(".mod.router.weight" in k for k in rev.weights)

    def test_true_skip_processes_fewer_tokens(self):
        """keep_fraction<1 in training: attention sees ONLY kept tokens."""
        cfg = _tiny()
        cfg.use_mod = True
        cfg.mod_keep_fraction = 0.5
        model = ConfigurableResearchLLM(cfg)
        model.train()
        # Count tokens actually processed by the attention block (layer 2).
        counted = {"tokens": 0}
        attn = model.blocks[2].attn
        orig_forward = attn.forward

        def counting_forward(*args, **kwargs):
            x_in = args[0]
            counted["tokens"] += x_in.shape[0] * x_in.shape[1]
            return orig_forward(*args, **kwargs)

        attn.forward = counting_forward
        x = torch.randint(0, 256, (2, 16))
        loss = model(x, targets=x)[1]
        loss.backward()
        # 2 rows x ceil(16*0.5)=8 kept = 16 tokens, vs 32 without skip
        assert counted["tokens"] == 16, counted
        # router got gradient through the aux loss
        assert model.blocks[2]._mod.router.weight.grad is not None
        # skipped tokens pass through unchanged (residual bypass)
        with torch.inference_mode():
            x_in = torch.randn(1, 8, 128)
            out = model.blocks[2]._forward_mod_skip(x_in, 2, None, None)
            mask = model.blocks[2]._mod.token_mask(x_in)
            assert torch.equal(out[0][~mask[0]], x_in[0][~mask[0]])


# ── Main model (ForgeLM V2 Jamba) sanity: lossless flags enabled ──────────────


class TestMainModel:
    def test_main_config_builds_with_new_keys(self):
        """Test V2 config fields + tiny model build on GPU with ForgeEngine features."""
        cfg = get_config("forgelm_v2")
        # ForgeLM V2 is a Jamba hybrid — GQA attention + Mamba, no BitNet/TITAN/MoD
        assert cfg.attn_type == "gqa"
        assert cfg.use_bitnet is False
        assert cfg.ffn_compression == "none"
        assert cfg.use_titan_memory is False
        assert cfg.use_mod is False
        assert cfg.d_model == 2560 and cfg.n_layers == 28
        # Build tiny model on GPU to verify TITAN/MoD forward works with bf16
        tiny = get_config("forgelm_tiny")
        tiny.use_titan_memory = True
        tiny.titan_memory_rank = 16
        tiny.use_mod = True
        tiny.mod_keep_fraction = 1.0
        tiny.device = _DEV
        tiny.dtype = "bfloat16" if _CUDA else "float32"
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_DTYPE)
        try:
            if _CUDA:
                with torch.device("cuda"):
                    model = ConfigurableResearchLLM(tiny).eval()
            else:
                model = ConfigurableResearchLLM(tiny).eval()
        finally:
            torch.set_default_dtype(old_dtype)
        n_mem = sum(1 for b in model.blocks if b._memory is not None)
        n_mod = sum(1 for b in model.blocks if b._mod is not None)
        assert n_mem == 4 and n_mod == 4  # tiny has 4 layers
        x = torch.randint(0, 256, (1, 4), device=_DEV)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 4, 256)
        print(f"Tiny + TITAN/MoD forward OK on {_DEV} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

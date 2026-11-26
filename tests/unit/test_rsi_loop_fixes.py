"""RSI-loop integrity fixes (2026-11-23):

1. _resolve_lora_targets must cover LFM3 conv in_proj + fused gate_up_proj
   (silent no-adapter miss on use_fused_gemm models). Verified on the REAL
   presets: qwen25_05b (fused qkv, unfused FFN) and lfm3_1p7b (fused qkv +
   fused gate_up + conv in_proj).
2. Quantized load_from_weight must write packed storage IN PLACE (copy_):
   rebinding .data to fresh CPU tensors moved weights off-GPU (killing the
   fused-GEMV is_cuda check) and invalidated CUDA-graph storage.
3. ForgeQuantLinear.merge_lora must keep sparse_indices stable — re-picking
   outlier rows every merge compounds a re-partition error each round.
4. GRPO sign-advantage: uniform binary groups (all-pass/all-fail) get
   A=2r-1 instead of zero/dead gradient.
5. Proposer primed wrap must split on the "```python\n" fence OPENER —
   bare "```python" also matches prose mentions ("inside one ```python
   fence."), which truncated the user turn mid-word and caused the 0%
   proposer-bench failure on lfm3_1p7b_cc_fq.
"""
import types

import torch
import torch.nn as nn

from forge.config import MODEL_CONFIGS, ModelConfig
from forge.model.llm import ConfigurableResearchLLM
from forge.engine.quant.forge_quant import ForgeQuantLinear, NF4QuantLinear
from forge.training.bitnet_lora import LoRAAdapter, add_lora_adapters
from forge.training.runners.sft_train import _resolve_lora_targets
from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer

FENCE = "```python"
NL = "\n"


def _args(**kw):
    d = {"lora_targets": "", "bitnet_everywhere": False}
    d.update(kw)
    return types.SimpleNamespace(**d)


# Linears intentionally excluded from LoRA targeting: the LM head /
# output projections are vocab-sized (128K rows on LFM3) and "all-linear"
# guidance covers block internals, not the output head.
_HEAD_LIKE = {"head", "lm_head", "score", "classifier", "output_proj"}


def _linear_leaf_names(model: nn.Module) -> set[str]:
    return {n.split(".")[-1]
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear)} - _HEAD_LIKE


def test_lora_targets_cover_every_linear_on_qwen25_05b():
    """Real Qwen2.5-0.5B: fused qkv_proj + unfused FFN."""
    cfg = ModelConfig(**{**MODEL_CONFIGS["qwen25_05b"].__dict__})
    model = ConfigurableResearchLLM(cfg).eval()
    leaf = _linear_leaf_names(model)
    assert leaf, "no Linear modules found — model build broken?"
    targets = _resolve_lora_targets(model, _args())
    missing = leaf - set(targets)
    assert not missing, f"LoRA target list misses Linear modules: {missing}"
    expected = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.split(".")[-1] in set(targets))
    n_attached, _ = add_lora_adapters(model, rank=4, alpha=8,
                                    target_modules=targets)
    assert n_attached == expected and n_attached > 0


def test_lora_targets_cover_every_linear_on_lfm3():
    """Real LFM3-1.7B on meta device: fused qkv + gate_up_proj + conv
    in_proj must all be in the target list."""
    cfg = ModelConfig(**{**MODEL_CONFIGS["lfm3_1p7b"].__dict__,
                        "device": "meta"})
    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg)
    leaf = _linear_leaf_names(model)
    assert leaf, "no Linear modules found — model build broken?"
    assert "in_proj" in leaf, "expected conv in_proj in LFM3 module tree"
    targets = _resolve_lora_targets(model, _args())
    missing = leaf - set(targets)
    assert not missing, (
        f"LoRA target list misses Linear modules on lfm3_1p7b: {missing}")


def test_lora_targets_explicit_override_still_wins():
    cfg = ModelConfig(**{**MODEL_CONFIGS["qwen25_05b"].__dict__})
    model = ConfigurableResearchLLM(cfg).eval()
    targets = _resolve_lora_targets(model, _args(lora_targets="qkv_proj"))
    assert targets == ["qkv_proj"]


def test_forgq_load_from_weight_preserves_storage_and_device():
    lin = ForgeQuantLinear(128, 96, group_size=64, sparse_ratio=0.1)
    w = torch.randn(96, 128)
    lin.load_from_weight(w)
    ptrs = {n: p.data_ptr() for n, p in lin.named_parameters()}
    devs = {n: p.device for n, p in lin.named_parameters()}
    lin.load_from_weight(w + 0.01 * torch.randn_like(w))
    for n, p in lin.named_parameters():
        assert p.data_ptr() == ptrs[n], f"{n} storage re-bound"
        assert p.device == devs[n], f"{n} moved device"


def test_nf4_load_from_weight_preserves_storage():
    lin = NF4QuantLinear(128, 96, group_size=64)
    w = torch.randn(96, 128)
    lin.load_from_weight(w)
    p_ptr = lin.weight_packed.data_ptr()
    s_ptr = lin.weight_scales.data_ptr()
    lin.load_from_weight(w + 0.01 * torch.randn_like(w))
    assert lin.weight_packed.data_ptr() == p_ptr
    assert lin.weight_scales.data_ptr() == s_ptr


def test_forgq_merge_lora_keeps_outlier_rows_stable():
    lin = ForgeQuantLinear(128, 96, group_size=64, sparse_ratio=0.25)
    w = torch.randn(96, 128)
    lin.load_from_weight(w)
    idx_before = lin.sparse_indices.data.clone()
    lin.lora_adapter = LoRAAdapter(128, 96, rank=4, alpha=8)
    with torch.no_grad():
        lin.lora_adapter.lora_B.normal_(0, 0.02)
    assert lin.merge_lora()
    assert torch.equal(lin.sparse_indices.data, idx_before), (
        "merge re-picked outlier rows — partition churn compounds error")
    assert lin.lora_adapter is None


def _trainer(sign=True):
    cfg = GRPOConfig(sign_advantage_binary=sign)
    stub = nn.Linear(8, 8)
    for p in stub.parameters():
        p.requires_grad = False
    return GRPOTrainer(model=stub, tokenizer=None, ref_model=None,
                       device="cpu", config=cfg)


def test_sign_advantage_rescues_uniform_binary_groups():
    t = _trainer(sign=True)
    all_fail = t.compute_advantages([0.0, 0.0, 0.0, 0.0])
    all_pass = t.compute_advantages([1.0, 1.0, 1.0, 1.0])
    assert all_fail == [-1.0] * 4
    assert all_pass == [1.0] * 4
    assert t.stats.sign_advantage_groups == 2
    mixed = t.compute_advantages([0.0, 0.0, 1.0, 1.0])
    assert any(a > 0 for a in mixed) and any(a < 0 for a in mixed)


def test_sign_advantage_off_preserves_collapse():
    t = _trainer(sign=False)
    adv = t.compute_advantages([1.0, 1.0, 1.0, 1.0])
    assert all(abs(a) < 0.01 for a in adv)
    assert t.stats.sign_advantage_groups == 0
    assert t.stats.advantage_collapse_count == 1


def test_sign_advantage_skips_non_binary_rewards():
    t = _trainer(sign=True)
    adv = t.compute_advantages([0.5, 0.5, 0.5, 0.5])
    assert all(abs(a) < 0.01 for a in adv)
    assert t.stats.sign_advantage_groups == 0


def _primed_lfm3_proposer():
    from forge.evolution.code_evolve import LLMProposer
    p = object.__new__(LLMProposer)
    p.cfg = _args(proposer_think=False)
    p._lfm3_cached = True
    p._spark_cached = False
    p._primed_cached = True
    return p


def test_primed_wrap_splits_at_fence_opener_not_prose():
    p = _primed_lfm3_proposer()
    # Bench-shaped prompt: closed seed block + trailing instruction that
    # MENTIONS ```python in prose. The split must land on the fence
    # opener, not the prose mention.
    prompt = ("Rewrite to be faster." + NL + NL + FENCE + NL +
              "def fast_sum(xs):" + NL + "    return sum(xs)" + NL +
              "```" + NL + NL +
              "Return only verbatim Python inside one " + FENCE + " fence.")
    wrapped = p._wrap(prompt)
    user_part, _, assistant_part = wrapped.partition(
        "<|im_start|>assistant")
    assert "fast_sum" in assistant_part
    assert user_part.rstrip().endswith("<|im_end|>")
    # The prose mention stays intact INSIDE the assistant prefill — it
    # must NOT cut the user turn mid-word.
    assert "inside one " + FENCE + " fence." in assistant_part


def test_primed_wrap_open_prime_still_lands():
    p = _primed_lfm3_proposer()
    # Production shape: instruction + closed seed + open prime fence.
    prompt = ("# Rewrite." + NL + FENCE + NL +
              "def f(x):" + NL + "    return x" + NL + "```" + NL +
              "# Write the improved implementation now:" + NL +
              FENCE + NL + "def f(x):")
    wrapped = p._wrap(prompt)
    assistant = wrapped.split("<|im_start|>assistant", 1)[1]
    assert assistant.lstrip().startswith(
        "<think></think>" + FENCE + NL + "def f(x):")


def test_primed_wrap_fenceless_falls_to_chatml():
    p = _primed_lfm3_proposer()
    wrapped = p._wrap("Rewrite this program to be faster.")
    assert "<|im_start|>user" in wrapped
    assert "<|im_start|>assistant" in wrapped
    assert FENCE in wrapped  # LFM3 fence prefill appended


def test_release_acceleration_drops_bcg_state():
    """park() → sleep(1) must drop _bcg/_decode_state on engine AND model:
    captured graphs bake weight pointers that dangle after the weights
    leave and return to the GPU at new addresses (illegal memory access
    in fast_eval, observed 2026-11-24)."""
    import torch
    from forge.engine.forge_engine import ForgeEngine
    eng = ForgeEngine.__new__(ForgeEngine)
    eng._bcg = {"fake": object()}
    eng._decode_state = {"cache": object()}
    eng._graph_runner = eng._megakernel = eng._flex_decoding = None
    eng._chunked_prefiller = None
    eng.kv_cache = eng.decoding = eng._prefix_cache = None
    eng._cache_blend = eng.avmp = eng.vtensor_pool = None
    eng._suffix_mem = eng.quarot_kv = eng.v0_warm = None
    eng.progressive_kv = eng.mrl_adapter = None
    eng._corun = eng._foundry = None
    m = torch.nn.Linear(2, 2)
    m._bcg = {"fake": object()}
    m._decode_state = {"cache": object()}
    m._ensure_static_batch = lambda *a, **k: None  # closure survives
    eng.model = m
    eng.acceleration = "x"
    eng._release_acceleration_resources()
    assert eng._bcg is None and eng._decode_state is None
    assert m._bcg is None and m._decode_state is None
    assert eng.acceleration is None
    # The lazy-recapture hook is a pure closure — survives the release.
    assert m._ensure_static_batch is not None


def test_validate_task_rejects_mega_payload():
    """A test case whose serialized payload exceeds _MAX_TEST_PAYLOAD is
    unlearnable and prompt-busting (a 5MB expected list produced a ~22k
    token solve prompt -> 48GiB pad_bias OOM -> device assert)."""
    from forge.self_play.infinite_curriculum import (
        InfiniteCurriculum, ProposedTask, _MAX_TEST_PAYLOAD)
    ic = object.__new__(InfiniteCurriculum)
    task = ProposedTask(
        id="t", domain="algorithms", difficulty="easy",
        description="x", signature="(n: int)",
        test_cases=[{"args": [654321],
                     "expected": list(range(654321))}])
    assert ic._validate_task(task) is False
    # And a normal-sized task does not trip the cap (it may fail later
    # checks � only asserting the payload gate didn't fire).
    small = ProposedTask(
        id="s", domain="algorithms", difficulty="easy",
        description="x", signature="(n: int)",
        test_cases=[{"args": [3], "expected": [2, 1]},
                    {"args": [5], "expected": [4, 1, 2, 3]}])
    assert (sum(len(repr(tc.get("args")))
                + len(repr(tc.get("expected")))
                for tc in small.test_cases) <= _MAX_TEST_PAYLOAD)


def test_solve_prompt_truncates_big_case_reprs():
    """Prompt construction must bound per-case repr size even for tasks
    that pass validation."""
    import types
    from forge.self_play.infinite_curriculum import InfiniteCurriculum
    ic = object.__new__(InfiniteCurriculum)
    ic.engine = object()  # engine path ? CHAT_SOLVE_PROMPT
    task = types.SimpleNamespace(
        description="d", solve_name="solve", input_signature="(n: int)",
        test_cases=[{"args": [123456],
                     "expected": list(range(200000))}])
    prompt = ic._solve_prompt_text(task)
    assert "...<trunc>" in prompt
    assert len(prompt) < 4096

"""R65: Rocinante-X-12B port — unit tests.

Covers the training-free port machinery on CPU (no model file needed):
  - rocinante_12b preset matches the GGUF architecture exactly
  - GGUF→HF name mapping covers every tensor a Nemo-family file emits
  - llama.cpp interleave→HF rotate_half un-permutation is correct
  - SuffixMemory: sparse rows, bounded cells, exact recall, drafting
  - FluxKVCache: compact_past_kv keeps sink+window, drops middle, and
    passes through non-attention entries
"""

import torch

from forge.config import get_config
from forge.engine.kv_backend import build_kv_cache
from forge.model.ngram_mem import SuffixMemory
from scripts.convert_gguf_forge import gguf_to_hf_name, unpermute_rope_rows

# ── preset ──────────────────────────────────────────────────────────────────


def test_preset_matches_gguf():
    cfg = get_config("rocinante_12b")
    assert cfg.vocab_size == 131072
    assert cfg.d_model == 5120
    assert cfg.n_layers == 40
    assert cfg.n_heads == 32
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128  # NOT d_model/n_heads (160)
    assert cfg.intermediate_size == 14336
    assert cfg.attn_type == "gqa"
    assert cfg.ffn_type == "swiglu"
    assert cfg.norm_type == "rmsnorm"
    assert abs(cfg.norm_eps - 1e-5) < 1e-9
    assert cfg.rope_base == 1_000_000.0
    assert cfg.max_seq_len == 131072
    assert cfg.tie_word_embeddings is False
    assert cfg.use_qk_norm is False


# ── name mapping ────────────────────────────────────────────────────────────


def test_gguf_name_map():
    assert gguf_to_hf_name("token_embd.weight") == "model.embed_tokens.weight"
    assert gguf_to_hf_name("output.weight") == "lm_head.weight"
    assert gguf_to_hf_name("output_norm.weight") == "model.norm.weight"
    assert gguf_to_hf_name("blk.7.attn_q.weight") == "model.layers.7.self_attn.q_proj.weight"
    assert gguf_to_hf_name("blk.39.ffn_down.weight") == "model.layers.39.mlp.down_proj.weight"
    assert gguf_to_hf_name("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
    assert gguf_to_hf_name("blk.0.bogus.weight") is None


# ── RoPE un-permutation ─────────────────────────────────────────────────────


def test_unpermute_inverts_llama_interleave():
    """HF→GGUF interleaves RoPE pairs: reshape(h,hd//2,2,d).swapaxes(1,2).
    unpermute_rope_rows must be its exact inverse on arbitrary data."""
    torch.manual_seed(0)
    h, hd, d = 4, 8, 6
    w_hf = torch.randn(h * hd, d)
    # llama.cpp convert.py permute: (h, hd//2, 2, d) swapaxes(1,2) → (h, 2, hd//2, d)
    w_gguf = w_hf.reshape(h, hd // 2, 2, d).swapaxes(1, 2).reshape(h * hd, d)
    w_back = unpermute_rope_rows(w_gguf, h, hd)
    assert torch.equal(w_back, w_hf)


def test_unpermute_kv_heads():
    """K rows group by n_kv_heads, not n_heads — verify shape + inverse."""
    torch.manual_seed(1)
    h, hd, d = 2, 8, 6
    w_hf = torch.randn(h * hd, d)
    w_gguf = w_hf.reshape(h, hd // 2, 2, d).swapaxes(1, 2).reshape(h * hd, d)
    assert torch.equal(unpermute_rope_rows(w_gguf, h, hd), w_hf)


# ── SuffixMemory ────────────────────────────────────────────────────────────


def test_suffix_memory_exact_recall():
    """A repeated verbatim span must be recallable after it scrolls by."""
    V = 500
    m = SuffixMemory(V, orders=(2, 4, 8))
    span = [10, 20, 30, 40, 50]
    m.feed(span)
    filler = list(range(100, 200))
    m.feed(filler)  # span scrolled far out of any KV window
    m.feed(span[:-1])  # replay the span's prefix
    lift = m.lift_last()
    # Next token of the original span is 50 — deepest matching order wins.
    # Lifts are log-prob space (<=0); recall means lift[50] > lift of
    # tokens the memory has never seen following this suffix.
    assert int(lift.argmax()) == 50
    assert lift[50].item() > lift[123].item()  # unseen successor


def test_suffix_memory_bounded():
    """max_cells cap must decay, not grow unboundedly."""
    V = 100
    m = SuffixMemory(V, orders=(2,), max_cells_per_order=10)
    # 200 distinct order-2 suffixes → decay kicks in repeatedly.
    import random

    rng = random.Random(0)
    m.feed([rng.randrange(V) for _ in range(200)])
    assert len(m.tables[2]) <= 10
    assert m._decays > 0


def test_suffix_memory_incremental_tail():
    m = SuffixMemory(50, orders=(2,))
    m.feed([1, 2, 3])
    m.feed_one(4)
    assert m._tail == [3, 4]  # tail capped at max_order=2
    lift = m.lift_last()
    assert lift.shape == (50,)


def test_suffix_memory_snapshot_restore():
    m = SuffixMemory(50, orders=(2,))
    m.feed([5, 6, 7, 5, 6])
    snap = m.snapshot_state()
    l1 = m.lift_last()
    m.feed_one(9)
    m.restore_state(snap)
    l2 = m.lift_last()
    assert torch.equal(l1, l2)
    assert m._tail == [5, 6, 7, 5, 6][-m.max_order :]


def test_suffix_memory_draft():
    m = SuffixMemory(100, orders=(3,))
    seq = [1, 2, 3, 4, 5]
    m.feed(seq)
    m.feed(list(range(50, 60)))  # scroll
    m.feed(seq)  # repeat -> draftable
    m._tail = list(seq[:3])  # simulate mid-replay
    draft = m.draft(2)
    assert draft == [4, 5]


# ── FluxKVCache ─────────────────────────────────────────────────────────────


def test_flux_registered():
    cache = build_kv_cache("flux")
    assert type(cache).__name__ == "FluxKVCache"
    cache.init(n_heads=4, head_dim=8, n_kv_heads=2, max_seq_len=8192, device="cpu", dtype=torch.float32)
    assert cache.compact_over > cache.budget


def test_flux_compact_past_kv():
    cache = build_kv_cache("flux")
    cache.init(n_heads=4, head_dim=8, n_kv_heads=2, max_seq_len=8192, device="cpu", dtype=torch.float32)
    T = cache.budget + cache.slack + 10  # over the watermark
    k = torch.randn(1, 2, T, 8)
    v = torch.randn(1, 2, T, 8)
    conv_state = {"conv": torch.zeros(3)}  # must pass through untouched
    past = [(k, v), None, conv_state]
    out = cache.compact_past_kv(past)
    k2, v2 = out[0]
    assert k2.shape[-2] <= cache.budget
    assert v2.shape[-2] <= cache.budget
    # Sink + window invariants: first `sink` and last `window` rows kept.
    assert torch.equal(k2[..., : cache.sink, :], k[..., : cache.sink, :])
    assert torch.equal(k2[..., -cache.window :, :], k[..., -cache.window :, :])
    assert out[1] is None and out[2] is conv_state


def test_flux_compact_under_budget_noop():
    cache = build_kv_cache("flux")
    cache.init(n_heads=4, head_dim=8, n_kv_heads=2, max_seq_len=8192, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out = cache.compact_past_kv([(k, v)])
    assert out[0][0] is k and out[0][1] is v  # untouched

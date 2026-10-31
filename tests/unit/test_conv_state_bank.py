"""DoubleGatedConvLayer conv-state backing buffer — the CUDA-graph
storage contract.

Static-KV decode graphs capture the conv layer's _conv_state tensor
storage. The old single-buffer scheme reallocated on every position-0
reset, so captured replays read freed memory (allocator luck made small
batches appear to work; B>8 produced OOV garbage). The backing buffer is
allocated once at the max batch and every batch's state is a [:B] view —
never freed — while reset only re-points the alias.
"""
import torch

from forge.model.layers import DoubleGatedConvLayer


def _layer(d=32, k=7):
    torch.manual_seed(0)
    return DoubleGatedConvLayer(d, kernel_size=k).eval()


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_backing_buffer_survives_reset_and_batch_switch():
    m = _layer()
    x8 = torch.randn(8, 4, 32)
    x1 = torch.randn(1, 4, 32)
    with torch.no_grad():
        m(x8, use_cache=True)
        buf = m._conv_state_buf
        m._conv_state_reset = True          # model-level new-seq signal
        m(x8, use_cache=True)
        assert m._conv_state_buf is buf      # NEVER reallocated
        m._conv_state_reset = True
        m(x1, use_cache=True)
        assert m._conv_state_buf is buf      # B=1 slices the same buffer
        assert m._conv_state.shape[0] == 1
        m._conv_state_reset = True
        m(x8, use_cache=True)
        assert m._conv_state.shape[0] == 8
        assert m._conv_state_buf is buf


def test_graph_pointer_survives_reset_realloc_cycle():
    """Simulates BCG: keep the capture-time pointer; verify every reset
    + prefill + decode cycle writes THROUGH the same storage."""
    m = _layer()
    x = torch.randn(4, 6, 32)
    with torch.no_grad():
        m(x, use_cache=True)
        captured = m._conv_state            # what the graph holds
        for _ in range(3):
            m._conv_state_reset = True
            m(x, use_cache=True)            # fresh prefill
            m(torch.randn(4, 1, 32), use_cache=True)  # decode step
            assert _same_storage(m._conv_state, captured)


def test_prefix_restore_writes_into_backing_buffer():
    m = _layer()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        m(x, use_cache=True)
        buf = m._conv_state_buf
        snap = buf[:2].clone()
        m._conv_state_prefix = torch.randn(2, 32, 6)
        m(torch.randn(2, 1, 32), use_cache=True)
        assert _same_storage(m._conv_state, buf)
        assert not torch.equal(buf[:2], snap)


def test_stale_cross_batch_ctx_ignored():
    """A _conv_state left over from another batch size must not feed a
    T>1 forward as left_ctx (cat-mismatch crash / silent corruption)."""
    m = _layer()
    x8 = torch.randn(8, 4, 32)
    x1 = torch.randn(1, 4, 32)
    with torch.no_grad():
        m(x8, use_cache=True)
        stale8 = m._conv_state_buf[:8].clone()
        m._conv_state = stale8               # simulate missed reset
        out_guard, _ = m(x1, use_cache=True)
        m._conv_state = None
        out_fresh, _ = m(x1, use_cache=True)
    assert torch.equal(out_guard, out_fresh)


def test_decode_step_matches_functional():
    """T=1 incremental conv == causal conv on the same stream."""
    m = _layer()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out_full, _ = m(x, use_cache=True)
        # decode one more token — must equal causal conv on 9 tokens
        t9 = torch.randn(1, 1, 32)
        out_step, _ = m(t9, use_cache=True)
        m._conv_state_reset = True
        x9 = torch.cat([x, t9], dim=1)
        m._conv_state = None
        out_9, _ = m(x9, use_cache=True)
    assert torch.allclose(out_step, out_9[:, -1:], atol=1e-6)

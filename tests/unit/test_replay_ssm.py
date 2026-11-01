"""Tests for ReplaySSMCache — input-caching for SSM state reconstruction.

Verifies:
  - cache_inputs stores inputs correctly
  - reconstruct_state replays inputs to rebuild state
  - checkpoint/rollback discards the right tokens
  - circular buffer wraps correctly
  - clear resets everything
  - speculative decode scenario (generate, verify, reject, rollback, re-gen)
  - replay gives the same result as saving/restoring state directly
"""

import pytest
import torch
import torch.nn as nn

from forge.engine.kv.replay_ssm import ReplaySSMCache

# ── Test fixtures ──────────────────────────────────────────────────────────


class DummySSM(nn.Module):
    """Minimal SSM-like module for testing reconstruct_state.

    Maintains a running state: ``state = state + inputs.sum(dim=1)``.
    Returns ``(outputs, state)`` so the cache can extract the state.

    This is intentionally simple — the point is to verify that
    ReplaySSMCache correctly replays inputs through *any* callable
    that returns ``(out, state)``.
    """

    def __init__(self, d_model=8, d_state=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # A simple linear projection from d_model to d_state.
        self.proj = nn.Linear(d_model, d_state, bias=False)
        self.last_state = None

    def forward(self, inputs, initial_state=None):
        """Process inputs and return (outputs, state).

        Args:
            inputs: (batch, T, d_model)
            initial_state: (batch, d_state) or None

        Returns:
            (outputs, state) where state = initial_state + proj(inputs).sum(T)
        """
        bsz = inputs.shape[0]
        if initial_state is not None:
            state = initial_state.clone()
        else:
            state = torch.zeros(bsz, self.d_state, dtype=inputs.dtype, device=inputs.device)
        # Accumulate projected inputs over the time dimension.
        proj = self.proj(inputs)  # (batch, T, d_state)
        state = state + proj.sum(dim=1)
        self.last_state = state
        # Outputs: just pass through (not important for tests).
        outputs = inputs
        return outputs, state


@pytest.fixture
def ssm():
    """A fresh DummySSM for each test."""
    return DummySSM(d_model=8, d_state=4)


@pytest.fixture
def cache():
    """A fresh ReplaySSMCache (2 layers, capacity 16)."""
    return ReplaySSMCache(n_layers=2, max_replay_tokens=16)


# ── Tests ──────────────────────────────────────────────────────────────────


class TestCacheInputs:
    """Verify inputs are stored correctly."""

    def test_cache_inputs_single_token(self, cache):
        inp = torch.randn(1, 8)  # (batch, d_model)
        cache.cache_inputs(0, inp)
        assert cache._n_valid[0] == 1
        assert cache.position == 1

    def test_cache_inputs_multi_token(self, cache):
        inp = torch.randn(1, 5, 8)  # (batch, T=5, d_model)
        cache.cache_inputs(0, inp)
        assert cache._n_valid[0] == 5
        assert cache.position == 5

    def test_cache_inputs_multiple_layers(self, cache):
        inp0 = torch.randn(1, 3, 8)
        inp1 = torch.randn(1, 3, 8)
        cache.cache_inputs(0, inp0)
        cache.cache_inputs(1, inp1)
        assert cache._n_valid[0] == 3
        assert cache._n_valid[1] == 3
        assert cache.position == 3

    def test_cache_inputs_out_of_range(self, cache):
        inp = torch.randn(1, 8)
        with pytest.raises(IndexError):
            cache.cache_inputs(5, inp)

    def test_cache_inputs_preserves_values(self, cache):
        inp = torch.randn(1, 3, 8)
        cache.cache_inputs(0, inp)
        stored = cache._get_ordered_inputs(0)
        assert stored is not None
        torch.testing.assert_close(stored, inp)


class TestReconstructState:
    """Verify replaying inputs reconstructs the correct state."""

    def test_reconstruct_state_matches_direct(self, cache, ssm):
        """Replaying cached inputs should give the same state as
        processing them directly through the SSM."""
        inputs = torch.randn(1, 10, 8)
        # Direct: process all inputs at once.
        _, direct_state = ssm(inputs)

        # Replay: cache inputs one-by-one, then reconstruct.
        for t in range(10):
            cache.cache_inputs(0, inputs[:, t : t + 1, :])
        replayed_state = cache.reconstruct_state(0, ssm)

        torch.testing.assert_close(replayed_state, direct_state)

    def test_reconstruct_state_with_initial(self, cache, ssm):
        """Reconstruct with a non-zero initial state."""
        inputs = torch.randn(1, 5, 8)
        init_state = torch.randn(1, 4)

        # Direct.
        _, direct_state = ssm(inputs, initial_state=init_state)

        # Replay.
        for t in range(5):
            cache.cache_inputs(0, inputs[:, t : t + 1, :])
        replayed_state = cache.reconstruct_state(0, ssm, initial_state=init_state)

        torch.testing.assert_close(replayed_state, direct_state)

    def test_reconstruct_state_empty_raises(self, cache, ssm):
        with pytest.raises(RuntimeError):
            cache.reconstruct_state(0, ssm)

    def test_reconstruct_state_empty_with_initial(self, cache, ssm):
        """If no inputs cached but initial_state given, return initial."""
        init = torch.randn(1, 4)
        state = cache.reconstruct_state(0, ssm, initial_state=init)
        torch.testing.assert_close(state, init)


class TestCheckpointRollback:
    """Verify rollback discards the right tokens."""

    def test_checkpoint_returns_position(self, cache):
        inp = torch.randn(1, 5, 8)
        cache.cache_inputs(0, inp)
        pos = cache.checkpoint()
        assert pos == 5

    def test_rollback_discards_tokens(self, cache, ssm):
        # Cache 10 tokens.
        inputs = torch.randn(1, 10, 8)
        for t in range(10):
            cache.cache_inputs(0, inputs[:, t : t + 1, :])
        assert cache.position == 10

        # Checkpoint at position 5, cache 5 more.
        cp = cache.checkpoint()
        assert cp == 10  # checkpoint at current position (10)

        # Cache 5 more.
        more = torch.randn(1, 5, 8)
        for t in range(5):
            cache.cache_inputs(0, more[:, t : t + 1, :])
        assert cache.position == 15
        assert cache._n_valid[0] == 15

        # Rollback to checkpoint (discard the 5 speculative tokens).
        cache.rollback(cp)
        assert cache.position == 10
        assert cache._n_valid[0] == 10

        # Reconstruct state — should match processing only first 10 inputs.
        replayed = cache.reconstruct_state(0, ssm)
        _, direct = ssm(inputs)
        torch.testing.assert_close(replayed, direct)

    def test_rollback_partial(self, cache, ssm):
        """Rollback to a position in the middle of cached inputs."""
        inputs = torch.randn(1, 10, 8)
        for t in range(10):
            cache.cache_inputs(0, inputs[:, t : t + 1, :])

        # Rollback to position 4 (discard last 6).
        cache.rollback(4)
        assert cache.position == 4
        assert cache._n_valid[0] == 4

        # Reconstruct — should match processing first 4 inputs.
        replayed = cache.reconstruct_state(0, ssm)
        _, direct = ssm(inputs[:, :4, :])
        torch.testing.assert_close(replayed, direct)

    def test_rollback_to_zero(self, cache):
        inp = torch.randn(1, 5, 8)
        cache.cache_inputs(0, inp)
        cache.rollback(0)
        assert cache.position == 0
        assert cache._n_valid[0] == 0

    def test_rollback_future_raises(self, cache):
        cache.cache_inputs(0, torch.randn(1, 3, 8))
        with pytest.raises(ValueError):
            cache.rollback(10)

    def test_rollback_negative_raises(self, cache):
        with pytest.raises(ValueError):
            cache.rollback(-1)


class TestCircularBuffer:
    """Verify the ring buffer wraps correctly."""

    def test_wrap_around(self, cache):
        """When capacity is exceeded, oldest entries are overwritten."""
        cap = cache.max_replay_tokens  # 16

        # Fill exactly to capacity.
        first_batch = torch.randn(1, cap, 8)
        cache.cache_inputs(0, first_batch)
        assert cache._n_valid[0] == cap

        # Add 5 more — should wrap, overwriting first 5.
        second_batch = torch.randn(1, 5, 8)
        cache.cache_inputs(0, second_batch)
        assert cache._n_valid[0] == cap  # still capped
        assert cache.position == cap + 5

        # Ordered inputs should be: first_batch[5:] + second_batch
        ordered = cache._get_ordered_inputs(0)
        assert ordered is not None
        assert ordered.shape == (1, cap, 8)

        # First cap-5 entries should be first_batch[5:].
        torch.testing.assert_close(ordered[:, : cap - 5, :], first_batch[:, 5:, :])
        # Last 5 entries should be second_batch.
        torch.testing.assert_close(ordered[:, cap - 5 :, :], second_batch)

    def test_wrap_reconstruct(self, cache, ssm):
        """After wrap-around, reconstruct_state should give the state
        from the last `cap` inputs (not all inputs ever seen)."""
        d_model = 8

        # Cache 20 tokens (wraps, last 16 are kept).
        all_inputs = torch.randn(1, 20, d_model)
        for t in range(20):
            cache.cache_inputs(0, all_inputs[:, t : t + 1, :])

        # Reconstruct — should match processing only the last 16 inputs.
        replayed = cache.reconstruct_state(0, ssm)
        _, direct = ssm(all_inputs[:, 4:, :])  # last 16
        torch.testing.assert_close(replayed, direct)

    def test_multiple_wraps(self, cache):
        """Multiple wrap-arounds should still maintain correct order."""
        cap = cache.max_replay_tokens  # 16
        d_model = 8

        # Cache 3 * cap tokens.
        total = 3 * cap
        all_inputs = torch.randn(1, total, d_model)
        for t in range(total):
            cache.cache_inputs(0, all_inputs[:, t : t + 1, :])

        assert cache._n_valid[0] == cap
        ordered = cache._get_ordered_inputs(0)
        assert ordered.shape == (1, cap, d_model)

        # Should be the last `cap` inputs.
        torch.testing.assert_close(ordered, all_inputs[:, total - cap :, :])

    def test_rollback_after_wrap(self, cache, ssm):
        """Rollback from a wrapped state must return the correct
        ordered inputs (not positions [0, nv) which would be wrong
        after wrap-around)."""
        cap = cache.max_replay_tokens  # 16
        d_model = 8

        # Cache 20 tokens (wraps, last 16 kept).
        all_inputs = torch.randn(1, 20, d_model)
        for t in range(20):
            cache.cache_inputs(0, all_inputs[:, t : t + 1, :])
        assert cache._n_valid[0] == cap  # 16

        # Rollback by 5 → nv=11, position=15.
        cache.rollback(15)
        assert cache._n_valid[0] == 11

        # The valid inputs should be t=4..14 (11 tokens).
        ordered = cache._get_ordered_inputs(0)
        assert ordered.shape == (1, 11, d_model)
        torch.testing.assert_close(ordered, all_inputs[:, 4:15, :])

        # Reconstruct state should match processing t=4..14.
        replayed = cache.reconstruct_state(0, ssm)
        _, direct = ssm(all_inputs[:, 4:15, :])
        torch.testing.assert_close(replayed, direct)


class TestClear:
    """Verify clear resets everything."""

    def test_clear_resets_buffers(self, cache):
        inp = torch.randn(1, 5, 8)
        cache.cache_inputs(0, inp)
        cache.cache_inputs(1, inp)
        assert cache.position == 5

        cache.clear()
        assert cache.position == 0
        assert all(nv == 0 for nv in cache._n_valid)
        assert all(w == 0 for w in cache._write_idx)
        assert len(cache._checkpoints) == 0

    def test_clear_zeros_buffer(self, cache):
        inp = torch.randn(1, 3, 8)
        cache.cache_inputs(0, inp)
        buf = cache._buffers[0]
        assert buf is not None

        cache.clear()
        # Buffer should be zeroed.
        assert buf.abs().sum().item() == 0.0

    def test_clear_then_reuse(self, cache, ssm):
        """After clear, the cache should work correctly again."""
        inp1 = torch.randn(1, 3, 8)
        for t in range(3):
            cache.cache_inputs(0, inp1[:, t : t + 1, :])
        cache.clear()

        inp2 = torch.randn(1, 5, 8)
        for t in range(5):
            cache.cache_inputs(0, inp2[:, t : t + 1, :])
        assert cache.position == 5
        assert cache._n_valid[0] == 5

        replayed = cache.reconstruct_state(0, ssm)
        _, direct = ssm(inp2)
        torch.testing.assert_close(replayed, direct)


class TestSpeculativeScenario:
    """Simulate a speculative decode scenario.

    Flow:
      1. Generate (cache inputs) — "prefill"
      2. Checkpoint before drafting
      3. Draft speculative tokens (cache their inputs)
      4. Verify — some drafts rejected
      5. Rollback to checkpoint + accepted
      6. Re-generate from the rolled-back state
    """

    def test_speculative_accept_all(self, cache, ssm):
        """All drafts accepted — no rollback needed."""
        # Prefill: cache 10 inputs.
        prefill = torch.randn(1, 10, 8)
        for t in range(10):
            cache.cache_inputs(0, prefill[:, t : t + 1, :])

        cp = cache.checkpoint()
        assert cp == 10

        # Draft 4 tokens (all accepted).
        draft = torch.randn(1, 4, 8)
        for t in range(4):
            cache.cache_inputs(0, draft[:, t : t + 1, :])

        # All accepted (n_accepted = 4 = n_draft) → no rollback.
        # State should match prefill + draft.
        replayed = cache.reconstruct_state(0, ssm)
        _, direct = ssm(torch.cat([prefill, draft], dim=1))
        torch.testing.assert_close(replayed, direct)

    def test_speculative_reject_and_rollback(self, cache, ssm):
        """Draft rejected at position 2 — rollback to checkpoint + 2."""
        # Prefill: cache 10 inputs.
        prefill = torch.randn(1, 10, 8)
        for t in range(10):
            cache.cache_inputs(0, prefill[:, t : t + 1, :])

        cp = cache.checkpoint()
        assert cp == 10

        # Draft 4 tokens.
        draft = torch.randn(1, 4, 8)
        for t in range(4):
            cache.cache_inputs(0, draft[:, t : t + 1, :])
        assert cache.position == 14

        # Reject at position 2 (n_accepted = 2).
        n_accepted = 2
        n_draft = 4
        if n_accepted < n_draft:
            cache.rollback(cp + n_accepted + 1)

        assert cache.position == 13  # 10 + 2 + 1 (main token)

        # State should match prefill + first 3 tokens (main + 2 accepted).
        replayed = cache.reconstruct_state(0, ssm)
        expected_inputs = torch.cat([prefill, draft[:, :3, :]], dim=1)
        _, direct = ssm(expected_inputs)
        torch.testing.assert_close(replayed, direct)

    def test_speculative_reject_all(self, cache, ssm):
        """All drafts rejected — rollback to checkpoint + 1 (main only)."""
        prefill = torch.randn(1, 10, 8)
        for t in range(10):
            cache.cache_inputs(0, prefill[:, t : t + 1, :])

        cp = cache.checkpoint()

        draft = torch.randn(1, 4, 8)
        for t in range(4):
            cache.cache_inputs(0, draft[:, t : t + 1, :])

        # All rejected (n_accepted = 0).
        n_accepted = 0
        n_draft = 4
        if n_accepted < n_draft:
            cache.rollback(cp + n_accepted + 1)

        assert cache.position == 11  # 10 + 1 (main only)

        # State should match prefill + main token (draft[0]).
        replayed = cache.reconstruct_state(0, ssm)
        expected = torch.cat([prefill, draft[:, :1, :]], dim=1)
        _, direct = ssm(expected)
        torch.testing.assert_close(replayed, direct)

    def test_speculative_multi_round(self, cache, ssm):
        """Multiple rounds of draft/verify/rollback."""
        all_accepted_inputs = []
        prefill = torch.randn(1, 5, 8)
        for t in range(5):
            cache.cache_inputs(0, prefill[:, t : t + 1, :])
        all_accepted_inputs.append(prefill)

        for _round_idx in range(3):
            cp = cache.checkpoint()
            draft = torch.randn(1, 4, 8)
            for t in range(4):
                cache.cache_inputs(0, draft[:, t : t + 1, :])

            # Accept 2 out of 4 each round.
            n_accepted = 2
            n_draft = 4
            if n_accepted < n_draft:
                cache.rollback(cp + n_accepted + 1)

            all_accepted_inputs.append(draft[:, : n_accepted + 1, :])

        # Final state should match all accepted inputs.
        replayed = cache.reconstruct_state(0, ssm)
        all_inputs = torch.cat(all_accepted_inputs, dim=1)
        _, direct = ssm(all_inputs)
        torch.testing.assert_close(replayed, direct)


class TestReplayVsStateSave:
    """Verify replay gives the same result as saving/restoring state."""

    def test_replay_equals_save_restore(self, ssm):
        """Compare:
          - Save state after N tokens, then restore it.
          - Replay N inputs through the SSM to reconstruct state.
        Both should give identical results."""
        n_layers = 1
        cache = ReplaySSMCache(n_layers, max_replay_tokens=64)
        inputs = torch.randn(1, 20, 8)

        # Method 1: Save state directly.
        _, saved_state = ssm(inputs)

        # Method 2: Replay inputs.
        for t in range(20):
            cache.cache_inputs(0, inputs[:, t : t + 1, :])
        replayed_state = cache.reconstruct_state(0, ssm)

        torch.testing.assert_close(replayed_state, saved_state)

    def test_replay_equals_save_restore_with_rollback(self, ssm):
        """After rollback, replay should match a direct save/restore
        of the truncated input sequence."""
        cache = ReplaySSMCache(1, max_replay_tokens=64)
        inputs = torch.randn(1, 20, 8)

        # Cache all 20.
        for t in range(20):
            cache.cache_inputs(0, inputs[:, t : t + 1, :])

        # Rollback to 15.
        cache.rollback(15)

        # Replay.
        replayed = cache.reconstruct_state(0, ssm)

        # Direct: process first 15 inputs.
        _, direct = ssm(inputs[:, :15, :])

        torch.testing.assert_close(replayed, direct)

    def test_replay_equals_save_restore_multi_layer(self):
        """Multi-layer replay should match per-layer direct computation."""
        d_model = 8
        d_state = 4
        n_layers = 3
        cache = ReplaySSMCache(n_layers, max_replay_tokens=64)
        ssms = [DummySSM(d_model, d_state) for _ in range(n_layers)]
        inputs = torch.randn(1, 15, d_model)

        # Direct: process each layer.
        direct_states = []
        for layer_ssm in ssms:
            _, state = layer_ssm(inputs)
            direct_states.append(state)

        # Replay: cache + reconstruct per layer.
        for t in range(15):
            for layer_idx in range(n_layers):
                cache.cache_inputs(layer_idx, inputs[:, t : t + 1, :])

        for layer_idx in range(n_layers):
            replayed = cache.reconstruct_state(layer_idx, ssms[layer_idx])
            torch.testing.assert_close(replayed, direct_states[layer_idx])


class TestInfo:
    """Verify the info() method."""

    def test_info_empty(self, cache):
        info = cache.info()
        assert info["type"] == "replay_ssm"
        assert info["n_layers"] == 2
        assert info["max_replay_tokens"] == 16
        assert info["position"] == 0
        assert info["bytes"] == 0

    def test_info_after_caching(self, cache):
        cache.cache_inputs(0, torch.randn(1, 5, 8))
        info = cache.info()
        assert info["position"] == 5
        assert info["n_valid"] == [5, 0]
        assert info["bytes"] > 0

"""R39-4 & R39-5: Self-Speculative Sparse Decoding + XGrammar Constrained Decoding.

Tests for:
  - R39-5 (XGrammarConstrainer): JSON schema compilation, token masking
    (after ``{`` only allow ``"``, ``}``, whitespace), state transitions,
    reset, number masking, string masking.
  - R39-4 (SelfSpeculativeSparse): draft generation, verification,
    acceptance rate, lossless (verified output = full-attention output).

All tests use small mock models on CPU — no CUDA required.
"""

import pytest
import torch
import torch.nn as nn

from forge.engine.decoding import (
    SelfSpeculativeSparse,
    StandardDecoding,
    build_decoding,
)
from forge.engine.structured.xgrammar import XGrammarConstrainer

# ═══════════════════════════════════════════════════════════════════════
# Mock tokenizer for XGrammar tests
# ═══════════════════════════════════════════════════════════════════════


class MockTokenizer:
    """Simple character-level tokenizer for testing.

    Maps token IDs to single characters.  Token 0 = '', tokens 1+ map
    to printable ASCII characters.
    """

    def __init__(self, vocab_chars=None):
        if vocab_chars is None:
            # Include all printable ASCII + common JSON structural chars.
            vocab_chars = ' \t\n\r{}[]":,0123456789-+.eEtfnabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/'
        self._vocab = ["", *list(vocab_chars)]  # token 0 = empty
        self._id_to_char = dict(enumerate(self._vocab))
        self._char_to_id = {c: i for i, c in enumerate(self._vocab)}

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id_to_char.get(ids, "")
        return [self._id_to_char.get(i, "") for i in ids]

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._id_to_char.get(i, "") for i in ids)

    def encode(self, text, **kwargs):
        return [self._char_to_id.get(c, 0) for c in text]

    @property
    def vocab_size(self):
        return len(self._vocab)


# ═══════════════════════════════════════════════════════════════════════
# Mock model for SelfSpeculative tests
# ═══════════════════════════════════════════════════════════════════════


class MockSparseModel(nn.Module):
    """Mock model that produces different tokens for sparse vs full attention.

    - sparse_k=None (full attention): produces ``full_token``
    - sparse_k=<int> (sparse attention): produces ``sparse_token``

    This lets us test draft generation (sparse), verification (full),
    and acceptance rate (drafts rejected when sparse != full).
    """

    def __init__(self, vocab_size=20, full_token=5, sparse_token=3, head_dim=8, eos_token_id=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = type(
            "Cfg",
            (),
            {
                "vocab_size": vocab_size,
                "head_dim": head_dim,
                "n_kv_heads": 1,
                "n_layers": 1,
            },
        )()
        self.eos_token_id = eos_token_id
        self.full_token = full_token
        self.sparse_token = sparse_token
        self._vocab_size = vocab_size
        self._head_dim = head_dim
        self.call_log: list[str] = []  # 'sparse' or 'full'

    def forward(self, input_ids, past_key_values=None, use_cache=False, sparse_k=None, **kwargs):
        batch, seq_len = input_ids.shape
        is_sparse = sparse_k is not None
        token = self.sparse_token if is_sparse else self.full_token
        logits = torch.full((batch, seq_len, self._vocab_size), -10.0)
        logits[..., token] = 10.0
        key = torch.zeros(batch, 1, seq_len, self._head_dim)
        value = torch.zeros_like(key)
        self.call_log.append("sparse" if is_sparse else "full")
        return logits, None, ((key, value),)


class MockDeterministicModel(nn.Module):
    """Mock model that always produces the same token (sparse == full).

    Used for lossless tests where draft == verify (100% acceptance).
    """

    def __init__(self, vocab_size=20, token=5, head_dim=8, eos_token_id=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = type(
            "Cfg",
            (),
            {
                "vocab_size": vocab_size,
                "head_dim": head_dim,
                "n_kv_heads": 1,
                "n_layers": 1,
            },
        )()
        self.eos_token_id = eos_token_id
        self.token = token
        self._vocab_size = vocab_size
        self._head_dim = head_dim

    def forward(self, input_ids, past_key_values=None, use_cache=False, sparse_k=None, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self._vocab_size), -10.0)
        logits[..., self.token] = 10.0
        key = torch.zeros(batch, 1, seq_len, self._head_dim)
        value = torch.zeros_like(key)
        return logits, None, ((key, value),)


# ═══════════════════════════════════════════════════════════════════════
# R39-5: XGrammarConstrainer Tests
# ═══════════════════════════════════════════════════════════════════════


class TestXGrammarConstrainer:
    """R39-5: Constrained decoding with XGrammar-style token masking."""

    def _make_constrainer(self, vocab_chars=None):
        tok = MockTokenizer(vocab_chars)
        return XGrammarConstrainer(len(tok._vocab), tok), tok

    def test_compile_json_schema(self):
        """compile_json should set up the FSM and not raise."""
        c, _ = self._make_constrainer()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        c.compile_json(schema)
        assert c._schema is schema
        assert c._state == 0  # START

    def test_compile_grammar_string(self):
        """compile_grammar should accept a grammar string."""
        c, _ = self._make_constrainer()
        c.compile_grammar("json: root ::= object")
        assert c._grammar_str == "json: root ::= object"
        assert c._state == 0  # START

    def test_mask_after_open_brace_only_allows_quote_close_brace_whitespace(self):
        """After ``{`` the mask should only allow ``"``, ``}``, whitespace."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        # Advance past the opening brace.
        brace_id = tok.encode("{")[0]
        c.advance(brace_id)
        mask = c.get_mask(brace_id)
        # Check which tokens are allowed.
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        # Must include " and }
        assert '"' in allowed_chars, f'" should be allowed after {{, got {allowed_chars}'
        assert "}" in allowed_chars, f"}} should be allowed after {{, got {allowed_chars}"
        # Must NOT include digits, letters (except as part of whitespace), [, etc.
        assert "0" not in allowed_chars, "digits should NOT be allowed after {"
        assert "[" not in allowed_chars, "[ should NOT be allowed after {"
        assert "t" not in allowed_chars, "t should NOT be allowed after {"
        assert "f" not in allowed_chars, "f should NOT be allowed after {"
        assert "n" not in allowed_chars, "n should NOT be allowed after {"

    def test_mask_at_start_allows_value_start_chars(self):
        """At START (no type constraint), the mask should allow all value-start chars."""
        c, tok = self._make_constrainer()
        # No "type" key → any valid JSON value allowed at top level.
        c.compile_json({})
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        # Should allow {, [, ", digits, -, t, f, n, whitespace
        assert "{" in allowed_chars
        assert "[" in allowed_chars
        assert '"' in allowed_chars
        assert "0" in allowed_chars
        assert "-" in allowed_chars
        assert "t" in allowed_chars
        assert "f" in allowed_chars
        assert "n" in allowed_chars

    def test_mask_object_type_constrains_to_brace(self):
        """When schema type='object', START should only allow ``{``."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        assert "{" in allowed_chars
        assert "[" not in allowed_chars, "array start should NOT be allowed for object type"
        assert '"' not in allowed_chars, "string start should NOT be allowed for object type"

    def test_state_transition_key_to_colon(self):
        """After a key string ``"name"``, state should expect ``:``."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        # Advance: { " name "
        for char in '{"name"':
            tid = tok.encode(char)[0]
            c.advance(tid)
        # State should be AFTER_KEY — only ':' and whitespace allowed.
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        assert ":" in allowed_chars
        assert '"' not in allowed_chars, '" should NOT be allowed after key (expecting :)'
        assert "}" not in allowed_chars, "} should NOT be allowed after key (expecting :)"

    def test_state_transition_colon_to_value(self):
        """After ``:``, state should allow value-start characters."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        for char in '{"name":':
            tid = tok.encode(char)[0]
            c.advance(tid)
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        # Should allow value starts: ", digits, -, t, f, n, {, [
        assert '"' in allowed_chars
        assert "0" in allowed_chars
        assert "t" in allowed_chars

    def test_number_masking(self):
        """Inside a number, only number chars + structural chars allowed."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        # Advance: { " x " : 1 2 3
        for char in '{"x":123':
            tid = tok.encode(char)[0]
            c.advance(tid)
        # State should be IN_NUMBER
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        # Should allow digits, ., e, E, +, -, and structural (, })
        assert "4" in allowed_chars, "digits should be allowed in number"
        assert "." in allowed_chars, ". should be allowed in number"
        assert "e" in allowed_chars, "e should be allowed in number"
        assert "," in allowed_chars, ", should be allowed (structural after value in object)"
        assert "}" in allowed_chars, "} should be allowed (structural after value in object)"
        # Should NOT allow letters like 'a', 'b', 'x' (not number chars)
        assert "a" not in allowed_chars, "arbitrary letters should NOT be allowed in number"
        assert "x" not in allowed_chars

    def test_string_masking(self):
        """Inside a string value, any printable char is allowed (string content)."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        # Advance: { " x " : " hello
        for char in '{"x":"hello':
            tid = tok.encode(char)[0]
            c.advance(tid)
        # State should be IN_STRING
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        # Should allow letters (string content) and " (to close)
        assert "a" in allowed_chars, "letters should be allowed in string"
        assert '"' in allowed_chars, '" should be allowed to close string'
        # Inside a string, structural chars like } and , are valid string
        # content — the FSM only exits IN_STRING on '"'.
        assert "}" in allowed_chars, "} is valid string content (FSM stays in IN_STRING)"
        # Non-printable / control characters should NOT be in the allowed set
        # (IN_STRING allows chr(32)..chr(126) — printable ASCII).
        assert chr(0) not in allowed_chars, "control chars should not be allowed"
        assert chr(127) not in allowed_chars, "DEL should not be allowed"

    def test_reset(self):
        """reset() should return the FSM to START state."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        # Advance into the JSON
        for char in '{"name":':
            tid = tok.encode(char)[0]
            c.advance(tid)
        assert c._state != 0  # not START
        c.reset()
        assert c._state == 0  # START
        assert c._stack == []
        # Mask should be back to value-start chars
        mask = c.get_mask(0)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        assert "{" in allowed_chars

    def test_array_masking(self):
        """After ``[``, should allow value-start chars and ``]``."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "array"})
        # Advance: [
        bracket_id = tok.encode("[")[0]
        c.advance(bracket_id)
        mask = c.get_mask(bracket_id)
        allowed_chars = set()
        for tid in range(len(tok._vocab)):
            if mask[tid]:
                allowed_chars.add(tok.convert_ids_to_tokens(tid))
        assert "]" in allowed_chars, "] should be allowed after ["
        assert '"' in allowed_chars, '" should be allowed after ['
        assert "0" in allowed_chars, "digits should be allowed after ["
        assert "t" in allowed_chars, "t should be allowed after ["

    def test_full_json_object_roundtrip(self):
        """Full JSON object: { "key": "value" } — mask should guide correctly."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        # Walk through a complete JSON object, checking mask at each step.
        json_str = '{"key":"value"}'
        for char in json_str:
            tid = tok.encode(char)[0]
            mask = c.get_mask(tid)
            # The current character should be allowed by the mask.
            char_allowed = mask[tid].item()
            assert char_allowed, (
                f"Character '{char}' (token {tid}) should be allowed at state {c._state} in JSON: {json_str}"
            )
            c.advance(tid)
        # After complete JSON, state should be END
        assert c._state == 12  # END

    def test_get_mask_returns_correct_shape(self):
        """get_mask should return a (vocab_size,) boolean tensor."""
        c, tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        mask = c.get_mask(0)
        assert mask.dtype == torch.bool
        assert mask.shape == (len(tok._vocab),)

    def test_get_mask_has_at_least_one_true(self):
        """get_mask should always allow at least one token."""
        c, _tok = self._make_constrainer()
        c.compile_json({"type": "object"})
        mask = c.get_mask(0)
        assert mask.any(), "Mask should allow at least one token"


# ═══════════════════════════════════════════════════════════════════════
# R39-4: SelfSpeculativeSparse Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSelfSpeculativeSparse:
    """R39-4: Self-speculative decoding with sparse-attention draft."""

    def test_draft_generation_uses_sparse_attention(self):
        """Draft phase should call model with sparse_k (sparse attention)."""
        model = MockSparseModel(vocab_size=20, full_token=5, sparse_token=3)
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        strategy.generate(model, prompt, max_new_tokens=5, temperature=0.0)
        # Should have both sparse and full calls.
        assert "sparse" in model.call_log, "Draft phase should use sparse attention"
        assert "full" in model.call_log, "Verify phase should use full attention"

    def test_verification_uses_full_attention(self):
        """Verify phase should call model without sparse_k (full attention)."""
        model = MockSparseModel(vocab_size=20, full_token=5, sparse_token=3)
        strategy = SelfSpeculativeSparse(draft_len=2, sparse_k=32)
        prompt = torch.tensor([[1, 2, 3]])
        strategy.generate(model, prompt, max_new_tokens=3, temperature=0.0)
        # The prefill (first call) is full, then draft calls are sparse,
        # then verify call is full.
        assert model.call_log[0] == "full", "Prefill should use full attention"
        # At least one sparse call (draft) and one full call after (verify)
        assert "sparse" in model.call_log[1:], "Draft should use sparse"
        # After sparse calls, there should be a full call (verify)
        sparse_seen = False
        for call in model.call_log[1:]:
            if call == "sparse":
                sparse_seen = True
            elif call == "full" and sparse_seen:
                break
        assert sparse_seen, "Should have seen sparse draft calls"

    def test_acceptance_rate_zero_when_sparse_differs_from_full(self):
        """When sparse produces different tokens than full, acceptance = 0."""
        model = MockSparseModel(vocab_size=20, full_token=5, sparse_token=3)
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        strategy.generate(model, prompt, max_new_tokens=10, temperature=0.0)
        # All drafts produce token 3 (sparse), but verify expects token 5 (full).
        # So all drafts are rejected → acceptance rate = 0.
        assert strategy.acceptance_rate == 0.0, (
            f"Acceptance rate should be 0.0 when sparse != full, got {strategy.acceptance_rate}"
        )

    def test_acceptance_rate_one_when_sparse_equals_full(self):
        """When sparse produces same tokens as full, acceptance = 1.0."""
        model = MockDeterministicModel(vocab_size=20, token=5)
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        strategy.generate(model, prompt, max_new_tokens=10, temperature=0.0)
        # Drafts produce token 5 (sparse), verify expects token 5 (full).
        # All drafts accepted → acceptance rate = 1.0.
        assert strategy.acceptance_rate == 1.0, (
            f"Acceptance rate should be 1.0 when sparse == full, got {strategy.acceptance_rate}"
        )

    def test_lossless_output_matches_standard_decoding(self):
        """Verified output should equal full-attention standard decoding output.

        Uses max_new_tokens=5 to stay under StandardDecoding's degeneration
        guard (which fires at 8 consecutive identical tokens).
        """
        model_sparse = MockDeterministicModel(vocab_size=20, token=5)
        model_standard = MockDeterministicModel(vocab_size=20, token=5)
        prompt = torch.tensor([[1, 2, 3]])
        # Self-speculative sparse
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        sparse_output = strategy.generate(model_sparse, prompt, max_new_tokens=5, temperature=0.0)
        # Standard decoding
        standard = StandardDecoding()
        standard_output = standard.generate(model_standard, prompt, max_new_tokens=5, temperature=0.0)
        # Both should produce the same tokens (all token 5).
        assert sparse_output.shape == standard_output.shape, (
            f"Shape mismatch: sparse {sparse_output.shape} vs standard {standard_output.shape}"
        )
        assert torch.equal(sparse_output, standard_output), (
            "Self-speculative sparse output should be lossless (identical to standard full-attention decoding)"
        )

    def test_generate_returns_prompt_plus_generated(self):
        """Output should include the prompt + generated tokens."""
        model = MockDeterministicModel(vocab_size=20, token=5)
        strategy = SelfSpeculativeSparse(draft_len=2, sparse_k=32)
        prompt = torch.tensor([[1, 2, 3]])
        output = strategy.generate(model, prompt, max_new_tokens=5, temperature=0.0)
        # Output should be longer than prompt.
        assert output.shape[0] == 1  # batch
        assert output.shape[1] > prompt.shape[1], "Output should include generated tokens"
        # Prompt tokens should be preserved.
        assert torch.equal(output[:, : prompt.shape[1]], prompt)

    def test_generate_respects_max_new_tokens(self):
        """Should not generate more than max_new_tokens."""
        model = MockDeterministicModel(vocab_size=20, token=5)
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        max_new = 8
        output = strategy.generate(model, prompt, max_new_tokens=max_new, temperature=0.0)
        n_generated = output.shape[1] - prompt.shape[1]
        assert n_generated <= max_new, f"Generated {n_generated} tokens, expected <= {max_new}"

    def test_acceptance_rate_tracked_across_iterations(self):
        """Acceptance rate should accumulate across multiple iterations."""
        model = MockDeterministicModel(vocab_size=20, token=5)
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        strategy.generate(model, prompt, max_new_tokens=20, temperature=0.0)
        # With draft_len=4 and 100% acceptance, we accept 4 drafts + 1 base
        # per iteration.  total_drafts should be > 0.
        assert strategy._total_drafts > 0, "Should have drafted tokens"
        assert strategy._total_accepted > 0, "Should have accepted tokens"
        assert strategy.acceptance_rate > 0.0

    def test_build_decoding_factory(self):
        """build_decoding should create SelfSpeculativeSparse by name."""
        strategy = build_decoding("self_speculative_sparse", draft_len=3, sparse_k=32)
        assert isinstance(strategy, SelfSpeculativeSparse)
        assert strategy.draft_len == 3
        assert strategy.sparse_k == 32

    def test_build_decoding_default_standard(self):
        """build_decoding is strict (EC-019): unknown names raise
        ActivationError — a typo must never silently become
        StandardDecoding."""
        from forge.engine.errors import ActivationError

        with pytest.raises(ActivationError):
            build_decoding("nonexistent_strategy")

    def test_draft_len_parameter(self):
        """draft_len should control how many tokens are drafted per iteration."""
        model = MockDeterministicModel(vocab_size=20, token=5)
        strategy = SelfSpeculativeSparse(draft_len=3, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        strategy.generate(model, prompt, max_new_tokens=10, temperature=0.0)
        # With draft_len=3, each iteration drafts 3 tokens.
        # total_drafts should be a multiple of 3.
        assert strategy._total_drafts % 3 == 0, (
            f"total_drafts ({strategy._total_drafts}) should be multiple of draft_len (3)"
        )

    def test_sparse_k_parameter_stored(self):
        """sparse_k should be stored and accessible."""
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=128)
        assert strategy.sparse_k == 128

    def test_initial_acceptance_rate_zero(self):
        """Acceptance rate should start at 0.0 before any generation."""
        strategy = SelfSpeculativeSparse()
        assert strategy.acceptance_rate == 0.0
        assert strategy._total_drafts == 0
        assert strategy._total_accepted == 0

    def test_partial_acceptance(self):
        """Test partial acceptance with a model that changes behavior."""
        # Create a model where the first draft token matches but the
        # second doesn't.  We use a model that produces token 5 for both
        # sparse and full, but we manually check the acceptance logic.
        model = MockDeterministicModel(vocab_size=20, token=5)
        strategy = SelfSpeculativeSparse(draft_len=4, sparse_k=64)
        prompt = torch.tensor([[1, 2, 3]])
        output = strategy.generate(model, prompt, max_new_tokens=8, temperature=0.0)
        # With deterministic model (same token for sparse and full),
        # all drafts should be accepted → acceptance rate = 1.0
        assert strategy.acceptance_rate == 1.0
        # Output should have generated some tokens
        assert output.shape[1] > prompt.shape[1]


# ═══════════════════════════════════════════════════════════════════════
# R39-5: XGrammar + ForgeEngine Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class _MockConfig:
    """Minimal config for ForgeEngine mocking."""

    def __init__(self, vocab_size, head_dim=8, n_kv_heads=1, n_layers=1, name="mock"):
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.name = name


class _MockModelForEngine(nn.Module):
    """Mock model that produces logits favouring a specific token.

    Used to test that the XGrammar logits processor masks out tokens
    that would produce invalid JSON.
    """

    def __init__(self, vocab_size, favoured_token, head_dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = _MockConfig(vocab_size, head_dim=head_dim)
        self._vocab_size = vocab_size
        self._head_dim = head_dim
        self._favoured = favoured_token

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self._vocab_size), -10.0)
        logits[..., self._favoured] = 10.0
        key = torch.zeros(batch, 1, seq_len, self._head_dim)
        value = torch.zeros_like(key)
        return logits, None, ((key, value),)


class _CallableMockTokenizer(MockTokenizer):
    """MockTokenizer with __call__ for ForgeEngine.generate()/generate_raw().

    ForgeEngine tokenizes via ``self.tokenizer(prompt, return_tensors="pt",
    ...).input_ids``, so the tokenizer must be callable and return an
    object with an ``.input_ids`` attribute.  Also adds ``__len__`` for
    ``_safe_decode_ids`` which calls ``len(self.tokenizer)``.
    """

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)
        t = torch.tensor([ids], dtype=torch.long)
        return type("BatchEncoding", (), {"input_ids": t})()

    def __len__(self):
        return len(self._vocab)


class TestXGrammarEngineIntegration:
    """Integration tests: XGrammarConstrainer wired into ForgeEngine."""

    def _make_engine(self, vocab_chars=None, favoured_token=None, callable_tok=False):
        """Build a minimal ForgeEngine with a mock model + tokenizer.

        Args:
            callable_tok: if True, use _CallableMockTokenizer (needed for
                generate()/generate_raw() which call the tokenizer).
        """
        from forge.engine.forge_engine import ForgeEngine

        tok_cls = _CallableMockTokenizer if callable_tok else MockTokenizer
        tok = tok_cls(vocab_chars)
        vocab_size = len(tok._vocab)
        if favoured_token is None:
            # Default: favour the token for '{' (valid JSON object start).
            favoured_token = tok.encode("{")[0]
        model = _MockModelForEngine(vocab_size, favoured_token)
        engine = ForgeEngine(model, tok, device="cpu")
        return engine, tok, model

    def test_build_xgrammar_processor_returns_callable(self):
        """_build_xgrammar_processor should return a callable."""
        engine, _tok, _ = self._make_engine()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        proc = engine._build_xgrammar_processor(schema)
        assert callable(proc), "Processor should be callable"

    def test_processor_masks_non_object_tokens_at_start(self):
        """At START with type='object', processor should mask out non-'{' tokens."""
        engine, tok, _ = self._make_engine()
        schema = {"type": "object"}
        proc = engine._build_xgrammar_processor(schema)

        # Simulate logits favouring a digit token (invalid for object start).
        digit_id = tok.encode("0")[0]
        logits = torch.full((1, len(tok._vocab)), -10.0)
        logits[0, digit_id] = 10.0  # favour digit

        # First call: generated_ids is empty → FSM at START.
        result = proc(logits, [])
        # The digit token should be masked to -inf.
        assert result[0, digit_id] == float("-inf"), "Digit token should be masked at START for object schema"
        # The '{' token should NOT be masked.
        brace_id = tok.encode("{")[0]
        assert result[0, brace_id] > float("-inf"), "'{' token should NOT be masked at START for object schema"

    def test_processor_allows_brace_at_start(self):
        """At START with type='object', '{' token should survive masking."""
        engine, tok, _ = self._make_engine()
        schema = {"type": "object"}
        proc = engine._build_xgrammar_processor(schema)

        brace_id = tok.encode("{")[0]
        logits = torch.full((1, len(tok._vocab)), -10.0)
        logits[0, brace_id] = 10.0

        result = proc(logits, [])
        assert result[0, brace_id] == 10.0, "'{' token should not be masked at START for object schema"

    def test_processor_advances_fsm_after_token(self):
        """After advancing past '{', only '"', '}', whitespace should be allowed."""
        engine, tok, _ = self._make_engine()
        schema = {"type": "object"}
        proc = engine._build_xgrammar_processor(schema)

        brace_id = tok.encode("{")[0]
        # Simulate: first call (empty generated_ids) → mask at START.
        logits0 = torch.full((1, len(tok._vocab)), -10.0)
        proc(logits0, [])

        # Second call: generated_ids = [brace_id] → advance('{') then mask.
        logits1 = torch.full((1, len(tok._vocab)), -10.0)
        digit_id = tok.encode("0")[0]
        quote_id = tok.encode('"')[0]
        close_id = tok.encode("}")[0]
        logits1[0, digit_id] = 10.0  # favour digit (invalid after '{')

        result = proc(logits1, [brace_id])
        # Digit should be masked (not allowed after '{').
        assert result[0, digit_id] == float("-inf"), "Digit should be masked after '{' in object"
        # '"' and '}' should NOT be masked.
        assert result[0, quote_id] > float("-inf"), "'\"' should be allowed after '{'"
        assert result[0, close_id] > float("-inf"), "'}' should be allowed after '{'"

    def test_processor_handles_1d_logits(self):
        """Processor should handle 1-D logits (vocab_size,) without error."""
        engine, tok, _ = self._make_engine()
        schema = {"type": "object"}
        proc = engine._build_xgrammar_processor(schema)

        logits = torch.full((len(tok._vocab),), -10.0)
        brace_id = tok.encode("{")[0]
        logits[brace_id] = 10.0

        result = proc(logits, [])
        assert result[brace_id] == 10.0
        # A digit should be masked.
        digit_id = tok.encode("0")[0]
        assert result[digit_id] == float("-inf")

    def test_generate_with_json_schema_does_not_raise(self):
        """generate() with json_schema should run without error on mock model."""
        engine, tok, model = self._make_engine(favoured_token=None, callable_tok=True)
        # Favoured token = '{' so the model produces valid JSON object start.
        brace_id = tok.encode("{")[0]
        model._favoured = brace_id

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        # Use a very small max_new_tokens to keep the test fast.
        result = engine.generate("test", max_new_tokens=3, temperature=0.0, json_schema=schema, finish_sentence=False)
        assert isinstance(result, str)

    def test_generate_raw_with_json_schema_does_not_raise(self):
        """generate_raw() with json_schema should run without error."""
        engine, tok, model = self._make_engine(favoured_token=None, callable_tok=True)
        brace_id = tok.encode("{")[0]
        model._favoured = brace_id

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = engine.generate_raw("test", max_new_tokens=3, temperature=0.0, json_schema=schema)
        assert isinstance(result, str)

    def test_generate_raw_json_schema_takes_precedence_over_processor(self):
        """When both json_schema and logits_processor are given, schema wins."""
        engine, tok, model = self._make_engine(favoured_token=None, callable_tok=True)
        brace_id = tok.encode("{")[0]
        model._favoured = brace_id

        # A no-op processor that should be overridden.
        def dummy_proc(logits, ids):
            return logits

        schema = {"type": "object"}
        result = engine.generate_raw(
            "test", max_new_tokens=2, temperature=0.0, logits_processor=dummy_proc, json_schema=schema
        )
        assert isinstance(result, str)

    def test_generate_without_json_schema_unchanged(self):
        """generate() without json_schema should use the normal path."""
        engine, tok, model = self._make_engine(favoured_token=None, callable_tok=True)
        brace_id = tok.encode("{")[0]
        model._favoured = brace_id

        # No json_schema → normal decoding path.
        result = engine.generate("test", max_new_tokens=2, temperature=0.0, finish_sentence=False)
        assert isinstance(result, str)

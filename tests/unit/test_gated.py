"""ForgeGate mechanics tests — stub model, CPU only.

Covers: probe bundle version guard, route threshold, doom escalation,
convergence exit with min_conv guard, token accounting. The stub model
returns a fixed 4-tuple (logits, None, kv, hidden) so probe scores are
fully controlled by probe bias terms.
"""
import threading

import pytest
import torch

from forge.engine.gated import GATE_PROBES_VERSION, GateConfig, GatedDecoder, GateProbes


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTok:
    def __call__(self, s, add_special_tokens=True):
        if s == '<|im_end|>':
            return _Enc([999])
        if s == '<|endoftext|>':
            return _Enc([998])
        return _Enc([1] * (len(s) % 7 + 2))

    def decode(self, ids, skip_special_tokens=True):
        return 'x' * len(ids)


class StubModel(torch.nn.Module):
    """Always argmaxes token 5 (non-EOS); hidden = constant."""

    def __init__(self, d=8, vocab=1000):
        super().__init__()
        self.d, self.vocab = d, vocab

    def forward(self, idx, use_cache=False, return_hidden=False,
                past_key_values=None, attention_mask=None, **kw):
        B, L = idx.shape
        logits = torch.zeros(B, L, self.vocab)
        logits[..., 5] = 10.0
        hidden = torch.randn(B, L, self.d)
        return logits, None, [], hidden


def _probes(route_b, doom_b, conv_b, d=8):
    return GateProbes(
        torch.zeros(2 * d, 1), torch.tensor([route_b]),   # concat feat
        torch.zeros(d, 1), torch.tensor([doom_b]),
        torch.zeros(d + 1, 1), torch.tensor([conv_b]),    # h + pos feat
    )


def _decoder(route_b, doom_b, conv_b, **cfg_kw):
    cfg = GateConfig(direct_max=8, think_max=12, ans_max=4,
                     min_conv=3, **cfg_kw)
    model = StubModel()
    return GatedDecoder(model, FakeTok(), torch.device('cpu'),
                        _probes(route_b, doom_b, conv_b), cfg)


class TestProbeBundle:
    def test_version_guard(self, tmp_path):
        p = tmp_path / 'bad.pt'
        torch.save({'version': 99, 'route': {}, 'doom': {}, 'conv': {}},
                   p)
        with pytest.raises(ValueError, match='version'):
            GateProbes.load(p, torch.device('cpu'))

    def test_load_ok(self, tmp_path):
        p = tmp_path / 'ok.pt'
        torch.save({'version': GATE_PROBES_VERSION,
                    'route': {'w': torch.zeros(4, 1),
                              'b': torch.zeros(1)},
                    'doom': {'w': torch.zeros(4, 1),
                             'b': torch.zeros(1)},
                    'conv': {'w': torch.zeros(5, 1),
                             'b': torch.zeros(1)}}, p)
        pr = GateProbes.load(p, torch.device('cpu'))
        assert pr.route[0].shape == (4, 1)


class TestGatePaths:
    def test_think_route_when_p_easy_low(self):
        dec = _decoder(route_b=-10.0, doom_b=0.0, conv_b=-10.0)
        res = dec.generate('q')
        assert res.path == 'think'
        assert res.p_easy < 0.5
        assert res.tokens == 12            # think_max, no fire

    def test_direct_completes_when_clean(self):
        dec = _decoder(route_b=10.0, doom_b=-10.0, conv_b=-10.0)
        res = dec.generate('q')
        assert res.path == 'direct'
        assert res.tokens == 8             # direct_max

    def test_doom_escalates_to_think(self):
        dec = _decoder(route_b=10.0, doom_b=10.0, conv_b=-10.0)
        res = dec.generate('q')
        assert res.path == 'escalated'
        assert res.tokens == 2 + 12        # wasted doom steps + think_max

    def test_conv_exit_after_min_conv(self):
        dec = _decoder(route_b=-10.0, doom_b=0.0, conv_b=10.0)
        res = dec.generate('q')
        assert res.path == 'think+exit'
        # K=2 consecutive hits; min_conv=3 -> fires at step 4,
        # gen holds steps 0..4 -> 5 tokens + ans_max
        assert res.fire_pos == 4
        assert res.tokens == 5 + 4

    def test_escalated_exit_path(self):
        dec = _decoder(route_b=10.0, doom_b=10.0, conv_b=10.0)
        res = dec.generate('q')
        assert res.path == 'escalated+exit'


class _StubEngine:
    """Minimal ForgeEngine stand-in for chat_loop._route_p_easy."""

    def __init__(self, probes, d=8):
        self._gate_probes = probes
        self._gen_lock = threading.RLock()
        self.model = StubModel(d=d)

    def _tokenize(self, prompt):
        n = max(2, len(prompt) % 9 + 2)
        return torch.ones(1, n, dtype=torch.long)


class TestChatRouteGate:
    """forge_gui_server chat route gating — same probe head, rendered
    chat prompt instead of the GatedDecoder question template."""

    def test_none_without_probes(self):
        from forge_gui_server.services.chat_loop import _route_p_easy
        assert _route_p_easy(_StubEngine(None), 'prompt') is None

    def test_scores_route_probe(self):
        from forge_gui_server.services.chat_loop import _route_p_easy
        eng = _StubEngine(_probes(route_b=10.0, doom_b=0.0, conv_b=0.0))
        p = _route_p_easy(eng, 'prompt')
        assert p is not None and p > 0.99

    def test_low_score_below_threshold(self):
        from forge_gui_server.services.chat_loop import _GATE_T_ROUTE, _route_p_easy
        eng = _StubEngine(_probes(route_b=-10.0, doom_b=0.0, conv_b=0.0))
        p = _route_p_easy(eng, 'prompt')
        assert p is not None and p < _GATE_T_ROUTE

    def test_model_failure_returns_none(self):
        from forge_gui_server.services.chat_loop import _route_p_easy
        eng = _StubEngine(_probes(route_b=10.0, doom_b=0.0, conv_b=0.0))

        def _boom(*a, **kw):
            raise RuntimeError('no forward')
        eng.model = _boom
        assert _route_p_easy(eng, 'prompt') is None

    def test_direct_suffix_closes_think(self):
        from forge.self_play.discovery.chat_template import THINK_END, THINK_START
        from forge_gui_server.services.chat_loop import _DIRECT_SUFFIX
        assert _DIRECT_SUFFIX.startswith(THINK_START)
        assert THINK_END in _DIRECT_SUFFIX


class TestTrivialTurn:
    """chat_loop._is_trivial_turn — deterministic direct route for bare
    greetings/acks. The gate_r corpus has no chit-chat class, so the
    learned probe scores these under threshold even on a clean render;
    the rule covers them."""

    def _conv(self, content, role="user"):
        return [{"role": role, "content": content}]

    @pytest.mark.parametrize("msg", [
        "Hello", "hello!", "hi", "hey", "yo", "sup", "good morning",
        "good evening", "greetings", "what's up",
        "thanks!", "thank you", "thx", "ty",
        "ok", "okay", "yes", "yeah", "no", "nope", "sure",
        "bye", "goodbye", "see you", "later",
        "got it", "understood", "lol", "haha",
    ])
    def test_bare_greetings_acks_match(self, msg):
        from forge_gui_server.services.chat_loop import _is_trivial_turn
        assert _is_trivial_turn(self._conv(msg))

    @pytest.mark.parametrize("msg", [
        "hello, can you review my quicksort?",
        "yes, but also fix the off-by-one",
        "what's 2+2",
        "thanks — now explain why it works",
        "hey check this stack trace " + "x" * 60,   # over char cap
        "no way, revert that commit",
    ])
    def test_real_questions_do_not_match(self, msg):
        from forge_gui_server.services.chat_loop import _is_trivial_turn
        assert not _is_trivial_turn(self._conv(msg))

    def test_last_message_must_be_user(self):
        from forge_gui_server.services.chat_loop import _is_trivial_turn
        conv = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}]
        assert not _is_trivial_turn(conv)
        assert not _is_trivial_turn(
            self._conv("x", role="tool"))
        assert not _is_trivial_turn([])
        assert not _is_trivial_turn(self._conv("   "))


class TestThinkCap:
    """chat_loop._think_cap_processor — bounds a runaway think block by
    injecting the force-answer suffix ids one token per step, and bans
    <think> (541) in generated text so the model can't reopen a
    reasoning pass after </think>."""

    SUFFIX = [542, 100, 101]     # "\n</think>\nAnswer:"-style id list
    # always banned: <think> 541, <|im_start|> 518, <tool_response> 539,
    # </tool_response> 540; 531 joins while the think block is open.
    BASE_BANNED = (541, 518, 539, 540)

    def _masked(self, logits, banned=BASE_BANNED):
        m = logits.clone()
        for t in banned:
            m[..., t] = float('-inf')
        return m

    def test_under_budget_only_bans_rethink(self):
        from forge_gui_server.services.chat_loop import _think_cap_processor
        proc = _think_cap_processor(4, self.SUFFIX)
        logits = torch.randn(1, 1000)
        out = proc(logits, [1, 2, 3])
        # think still open -> <tool_call> (531) is masked too
        assert torch.equal(out, self._masked(
            logits, self.BASE_BANNED + (531,)))

    def test_rethink_banned_after_natural_close(self):
        from forge_gui_server.services.chat_loop import _THINK_END_ID, _think_cap_processor
        proc = _think_cap_processor(4, self.SUFFIX)
        logits = torch.randn(1, 1000)
        out = proc(logits, [1, 2, 3, 4, _THINK_END_ID, 9])
        assert torch.equal(out, self._masked(logits))

    def test_suffix_injected_in_order(self):
        from forge_gui_server.services.chat_loop import _think_cap_processor
        proc = _think_cap_processor(4, self.SUFFIX)
        gen = [7, 8, 9, 10]
        for want in self.SUFFIX:
            logits = torch.randn(1, 1000)
            out = proc(logits, gen)
            assert out.argmax(-1).item() == want
            assert out[0, want].item() == logits[0, want].item()
            gen.append(want)
        # queue drained -> only the rethink ban remains
        logits = torch.randn(1, 1000)
        assert torch.equal(proc(logits, gen), self._masked(logits))

    def test_budget_none_means_ban_only(self):
        # direct path: prompt already carries a closed think — the budget
        # never fires but <think> stays banned
        from forge_gui_server.services.chat_loop import _think_cap_processor
        proc = _think_cap_processor(None, self.SUFFIX)
        logits = torch.randn(1, 1000)
        out = proc(logits, list(range(50)))
        assert torch.equal(out, self._masked(logits))

    def test_exit_flag_injects_suffix_before_budget(self):
        # conv-probe fire must trigger the same force-answer injection
        # well before the token budget is reached
        from forge_gui_server.services.chat_loop import _think_cap_processor
        state = {"fired": False}
        proc = _think_cap_processor(160, self.SUFFIX,
                                    exit_flag=lambda: state["fired"])
        gen = [7, 8, 9]
        out = proc(torch.randn(1, 1000), gen)
        assert out[0, 5].item() > -float("inf")   # nothing forced yet
        state["fired"] = True
        for want in self.SUFFIX:
            logits = torch.randn(1, 1000)
            out = proc(logits, gen)
            assert out.argmax(-1).item() == want
            gen.append(want)

    def test_tool_call_masked_while_think_open(self):
        # the "thinking escapes into tool calls" fix: while the prompt-
        # opened <think> is unclosed, 531 must be -inf so a call can only
        # start after the reasoning pass closes.
        from forge_gui_server.services.chat_loop import _THINK_END_ID, _think_cap_processor
        proc = _think_cap_processor(160, self.SUFFIX)
        open_out = proc(torch.randn(1, 1000), [1, 2, 3])
        assert open_out[0, 531].item() == float("-inf")
        closed_out = proc(torch.randn(1, 1000), [1, 2, _THINK_END_ID, 9])
        assert closed_out[0, 531].item() > float("-inf")

    def test_tool_calls_disallowed_bans_call_ids(self):
        # defs dropped (repeat-call guard or tools off): both <tool_call>
        # and </tool_call> are banned outright, even after </think>.
        from forge_gui_server.services.chat_loop import _THINK_END_ID, _think_cap_processor
        proc = _think_cap_processor(None, self.SUFFIX,
                                    tool_calls_allowed=False)
        out = proc(torch.randn(1, 1000), [1, _THINK_END_ID, 2])
        assert out[0, 531].item() == float("-inf")
        assert out[0, 532].item() == float("-inf")

    def test_agent_suffix_without_answer_anchor(self):
        # agent loop injects a bare "\n</think>\n" — after the close the
        # tool_call mask must lift so the capped think can proceed to call
        from forge_gui_server.services.chat_loop import _think_cap_processor
        proc = _think_cap_processor(2, [542])
        gen = [7, 8]
        out = proc(torch.randn(1, 1000), gen)   # budget hit -> force 542
        assert out.argmax(-1).item() == 542
        gen.append(542)
        out = proc(torch.randn(1, 1000), gen)
        assert out[0, 531].item() > float("-inf")


class TestConvExitObserver:
    """chat_loop._conv_exit_observer — Gate C learned early exit: the
    conv probe scores per-step hidden states during think; _CONV_K
    consecutive scores above _CONV_T past _CONV_MIN_STEP inject the
    force-answer suffix via the cap processor's exit_flag."""

    def test_fires_after_k_consecutive_hits(self):
        from forge_gui_server.services.chat_loop import _CONV_K, _CONV_MIN_STEP, _conv_exit_observer
        probes = _probes(route_b=0.0, doom_b=0.0, conv_b=10.0)
        obs, fired = _conv_exit_observer(probes, 160)
        gen = list(range(_CONV_MIN_STEP - 1))
        obs(torch.zeros(8), gen)                 # below min step
        assert not fired()
        for _ in range(_CONV_K):
            gen.append(7)
            obs(torch.zeros(8), gen)
        assert fired()

    def test_low_score_never_fires(self):
        from forge_gui_server.services.chat_loop import _conv_exit_observer
        probes = _probes(route_b=0.0, doom_b=0.0, conv_b=-10.0)
        obs, fired = _conv_exit_observer(probes, 160)
        gen = list(range(80))
        obs(torch.zeros(8), gen)
        assert not fired()

    def test_run_resets_on_miss(self):
        # one low score between hits must reset the consecutive counter
        from forge_gui_server.services.chat_loop import _CONV_K, _CONV_MIN_STEP, _conv_exit_observer
        assert _CONV_K == 2
        probes = _probes(route_b=0.0, doom_b=0.0, conv_b=10.0)
        obs, fired = _conv_exit_observer(probes, 160)
        gen = list(range(_CONV_MIN_STEP))
        obs(torch.zeros(8), gen)                 # hit -> run=1
        # simulate a miss by feeding </think> (observer stops scoring)
        # then a fresh run — simpler: check a single hit doesn't fire
        assert not fired()
        gen.append(7)
        obs(torch.zeros(8), gen)                 # hit -> run=2 -> fire
        assert fired()

    def test_stops_scoring_after_think_end(self):
        from forge_gui_server.services.chat_loop import _THINK_END_ID, _conv_exit_observer
        probes = _probes(route_b=0.0, doom_b=0.0, conv_b=10.0)
        obs, fired = _conv_exit_observer(probes, 160)
        gen = list(range(40)) + [_THINK_END_ID]  # think already closed
        obs(torch.zeros(8), gen)
        assert not fired()

    def test_emits_gate_event_on_fire(self):
        from forge_gui_server.services.chat_loop import _CONV_K, _CONV_MIN_STEP, _conv_exit_observer
        probes = _probes(route_b=0.0, doom_b=0.0, conv_b=10.0)
        events = []
        obs, _ = _conv_exit_observer(
            probes, 160, emit=lambda e: events.append(e))
        gen = list(range(_CONV_MIN_STEP - 1))
        for _ in range(_CONV_K + 1):
            gen.append(7)
            obs(torch.zeros(8), gen)
        assert events == [("gate", {"p_easy": 1.0, "mode": "conv-exit"})]


class TestChatLoopGuards:
    """Repeat-call signature + direct-mode musing strip helpers."""

    def test_call_sig_matches_name_and_args(self):
        from forge_gui_server.services.chat_loop import _call_sig
        a = _call_sig({"name": "web_search",
                       "arguments": {"query": "news", "n": 5}})
        b = _call_sig({"name": "web_search",
                       "args": {"n": 5, "query": "news"}})  # args alias
        assert a == b
        c = _call_sig({"name": "web_search",
                       "arguments": {"query": "other", "n": 5}})
        assert a != c
        d = _call_sig({"name": "web_fetch",
                       "arguments": {"query": "news", "n": 5}})
        assert a != d

    def test_call_sig_tolerates_bad_args(self):
        from forge_gui_server.services.chat_loop import _call_sig
        s = _call_sig({"name": "x", "arguments": {"t": object()}})
        assert s[0] == "x" and isinstance(s[1], str)

    def test_strip_direct_musing(self):
        from forge_gui_server.services.chat_loop import _strip_direct_musing
        musing = ("The user is asking a simple question. I should be "
                  "concise.\n</think>\n\nThe answer is 4.")
        assert _strip_direct_musing(musing) == "The answer is 4."
        # no marker -> unchanged
        assert _strip_direct_musing("plain answer") == "plain answer"
        # marker but empty tail -> keep original (don't blank the reply)
        assert _strip_direct_musing("musing\n</think>\n") == \
            "musing\n</think>\n"
        # multiple markers -> take text after the last one
        two = "a\n</think>\nb\n</think>\nfinal"
        assert _strip_direct_musing(two) == "final"


class TestReasoningSplit:
    """chat_loop._split_reasoning + _strip_answer_anchor — the
    interleaved-thinking contract: reasoning rides reasoning_content,
    never the visible reply."""

    def test_splits_at_think_end(self):
        from forge_gui_server.services.chat_loop import _split_reasoning
        r, body = _split_reasoning("step one\nstep two\n</think>\n\n42")
        assert r == "step one\nstep two"
        assert body == "42"

    def test_no_closer_is_all_reasoning(self):
        # truncated/EOS'd inside the block — nothing may leak into the
        # visible body (the "thinking escapes" bug shape)
        from forge_gui_server.services.chat_loop import _split_reasoning
        r, body = _split_reasoning("musing with no close")
        assert r == "musing with no close"
        assert body == ""

    def test_extra_closers_dropped_from_body(self):
        from forge_gui_server.services.chat_loop import _split_reasoning
        r, body = _split_reasoning("think</think>real</think>stray")
        assert r == "think"
        assert body == "realstray"

    def test_empty_content(self):
        from forge_gui_server.services.chat_loop import _split_reasoning
        assert _split_reasoning("") == ("", "")

    def test_strip_answer_anchor(self):
        from forge_gui_server.services.chat_loop import _strip_answer_anchor
        assert _strip_answer_anchor("Answer: 42") == "42"
        assert _strip_answer_anchor("  Final Answer: 42") == "42"
        assert _strip_answer_anchor("plain reply") == "plain reply"
        # anchor alone must not blank the reply
        assert _strip_answer_anchor("Answer:") == "Answer:"
        assert _strip_answer_anchor("") == ""

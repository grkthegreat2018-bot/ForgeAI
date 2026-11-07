"""ForgeGate: three-probe gated generation (route / doom / convergence).

Single-model mode-switching: the "helper" is the same weights with the
think block closed, the "parent" is the same weights with it open. No
state transfer — escalation and convergence exits are token injections
into the live KV cache.

Validated on ForgeLM V2 (held-out, n=146): +16.4pts accuracy at -34%
tokens vs always-think (see .devin/scratchpad.md, R&D ForgeGate).

Probes (all logistic heads on hidden states, ~90KB total):
  route: P(direct-safe) on concat(h_last, h_mean) at the routing point
  doom:  P(this continuation fails) per decode step during direct mode
  conv:  P(forced-answer-now correct) per decode step during think mode

Paths per prompt:
  p_easy >= t_route -> monitored direct; K consecutive doom>tau
                       -> discard, escalate to think
  else/escalated    -> monitored think; K consecutive conv>tau (step
                       >= min_conv) -> inject force suffix, decode answer
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os
import torch

from forge.model.kv_cache import unpack_output_with_kv
import contextlib

GATE_PROBES_VERSION = 1

THINK_HINT = ("Begin by thinking about the reasoning process in the mind "
              "within <think> </think> tags and then proceed to give your "
              "response.\n")
REASONED_TAIL = "<|im_start|>assistant\n<think>\n"
DIRECT_TAIL = "<|im_start|>assistant\n<think>\n</think>\nAnswer:"
FORCE_SUFFIX = "\n</think>\nAnswer:"


@dataclass
class GateConfig:
    t_route: float = 0.5      # p_easy threshold: >= -> try direct first
    t_doom: float = 0.85      # doom score threshold per step
    k_doom: int = 2           # consecutive doom hits to escalate
    t_conv: float = 0.75      # convergence score threshold per step
    k_conv: int = 2           # consecutive conv hits to exit think
    min_conv: int = 32        # never exit think before this many tokens
                            # (probe trained on positions >= 39)
    direct_max: int = 96      # max tokens in direct mode
    think_max: int = 260      # max think tokens before forced answer
    ans_max: int = 48         # max answer tokens after convergence exit
    verbose: bool = False


@dataclass
class GateResult:
    text: str
    path: str                 # direct | think | think+exit |
                              # escalated+exit
    tokens: int               # generated tokens (excl. prompt)
    p_easy: float
    fire_pos: int = -1        # position where a gate fired (-1 = none)
    meta: dict = field(default_factory=dict)


class GateProbes:
    """Three logistic heads loaded from one bundle checkpoint."""

    def __init__(self, route_w, route_b, doom_w, doom_b,
                 conv_w, conv_b):
        self.route = (route_w, route_b)
        self.doom = (doom_w, doom_b)
        self.conv = (conv_w, conv_b)

    @classmethod
    def load(cls, path, device):
        path = str(path)
        st = path if path.endswith('.safetensors') else (
            path[:-3] + '.safetensors' if path.endswith('.pt') else None)
        if st and os.path.exists(st):
            ver, tensors = cls._load_safetensors(st)
        else:
            # Legacy pickle bundle — self-produced only; safetensors is the
            # canonical tracked artifact now.
            import warnings
            warnings.warn(
                f"gate probes loaded via pickle path {path!r} — "
                "prefer the .safetensors bundle", stacklevel=2)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            ver, tensors = ck.get('version', 0), {
                f'{k}.{kk}': vv for k in ('route', 'doom', 'conv')
                for kk, vv in ck[k].items()}
        if ver != GATE_PROBES_VERSION:
            raise ValueError(
                f"gate probe bundle version {ver} != "
                f"{GATE_PROBES_VERSION}; re-run the harvest pipeline")
        def tb(t):
            # detach: probe weights were saved with requires_grad=True —
            # score() must not build a grad graph on every call.
            return t.detach().to(device) if torch.is_tensor(t) else t
        return cls(
            tb(tensors['route.w']), tb(tensors['route.b']),
            tb(tensors['doom.w']), tb(tensors['doom.b']),
            tb(tensors['conv.w']), tb(tensors['conv.b']),
        )

    @staticmethod
    def _load_safetensors(path):
        from safetensors.torch import safe_open
        tensors = {}
        with safe_open(path, framework='pt') as f:
            meta = f.metadata() or {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return int(meta.get('version', 0)), tensors

    def score(self, which, feat):
        w, b = getattr(self, which)
        return float(torch.sigmoid(feat @ w + b))


class GatedDecoder:
    """Runs the three-gate cascade for one prompt at a time."""

    def __init__(self, model, tokenizer, device, probes: GateProbes,
                 cfg: GateConfig | None = None):
        self.model = model
        self.tok = tokenizer
        self.dev = device
        self.probes = probes
        self.cfg = cfg or GateConfig()
        self.eos_ids = self._eos_ids()
        self.force_ids = self._enc(FORCE_SUFFIX, special=False)

    # ── helpers ──
    def _enc(self, s, special=True):
        e = self.tok(s, add_special_tokens=special)
        ids = e.input_ids if hasattr(e, 'input_ids') else e['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def _eos_ids(self):
        ids = set()
        for t in ('<|im_end|>', '<|endoftext|>'):
            with contextlib.suppress(Exception):
                ids.update(self._enc(t, special=False))
        return ids

    def _prompt(self, q, tail):
        return (f"<|im_start|>user\n{THINK_HINT}{q}<|im_end|>\n" + tail)

    # ── gate R: routing probe on prompt hidden ──
    @torch.inference_mode()
    def route_p(self, q):
        e = self._enc(self._prompt(q, REASONED_TAIL))
        ids = torch.tensor([e], dtype=torch.long, device=self.dev)
        out = self.model(ids, return_hidden=True)
        h = out[-1][0].float()
        x = torch.cat([h[-1], h.mean(0)])
        return self.probes.score('route', x)

    # ── shared monitored decode loop ──
    @torch.inference_mode()
    def _monitored(self, prefix_ids, max_tok, which, K, tau,
                   min_step=0, pos_norm=None):
        """Greedy decode; probe on per-step hidden. Returns
        (gen_ids, fired_pos, kv, mask). fired_pos=-1 if no fire."""
        ids = torch.tensor([prefix_ids], dtype=torch.long,
                           device=self.dev)
        out = self.model(ids, use_cache=True, return_hidden=True)
        logits, kv = unpack_output_with_kv(out)
        h = out[-1][0, -1].float()
        mask = torch.ones(1, ids.shape[1], dtype=torch.bool,
                          device=self.dev)
        gen, run = [], 0
        for step in range(max_tok):
            nxt = int(logits[0, -1].argmax())
            if nxt in self.eos_ids:
                break
            gen.append(nxt)
            if pos_norm is None:
                feat = h
            else:
                feat = torch.cat([h, torch.tensor(
                    [min(step / pos_norm, 1.0)], device=self.dev)])
            s = self.probes.score(which, feat)
            run = run + 1 if (s > tau and step >= min_step) else 0
            if run >= K:
                return gen, step, kv, mask
            mask = torch.cat([mask, torch.ones(
                1, 1, dtype=torch.bool, device=self.dev)], dim=1)
            cur = torch.tensor([[nxt]], dtype=torch.long,
                               device=self.dev)
            out = self.model(cur, past_key_values=kv, use_cache=True,
                             attention_mask=mask, return_hidden=True)
            logits, kv = unpack_output_with_kv(out)
            h = out[-1][0, -1].float()
        return gen, -1, kv, mask

    # ── forced-answer tail after convergence fire ──
    @torch.inference_mode()
    def _finish_answer(self, kv, mask):
        fids = torch.tensor([self.force_ids], dtype=torch.long,
                            device=self.dev)
        mask = torch.cat([mask, torch.ones(
            1, len(self.force_ids), dtype=torch.bool,
            device=self.dev)], dim=1)
        out = self.model(fids, past_key_values=kv, use_cache=True,
                         attention_mask=mask)
        logits, kv = unpack_output_with_kv(out)
        gen = []
        for _ in range(self.cfg.ans_max):
            nxt = int(logits[0, -1].argmax())
            if nxt in self.eos_ids:
                break
            gen.append(nxt)
            mask = torch.cat([mask, torch.ones(
                1, 1, dtype=torch.bool, device=self.dev)], dim=1)
            cur = torch.tensor([[nxt]], dtype=torch.long,
                               device=self.dev)
            out = self.model(cur, past_key_values=kv, use_cache=True,
                             attention_mask=mask)
            logits, kv = unpack_output_with_kv(out)
        return gen

    # ── main entry ──
    @torch.inference_mode()
    def generate(self, question):
        cfg = self.cfg
        p = self.route_p(question)
        wasted = 0
        if p >= cfg.t_route:
            gen, fired, _, _ = self._monitored(
                self._enc(self._prompt(question, DIRECT_TAIL)),
                cfg.direct_max, 'doom', cfg.k_doom, cfg.t_doom)
            if fired < 0:
                return GateResult(
                    text=self.tok.decode(gen, skip_special_tokens=True),
                    path='direct', tokens=len(gen), p_easy=p)
            wasted = len(gen)
        gen, fired, kv, mask = self._monitored(
            self._enc(self._prompt(question, REASONED_TAIL)),
            cfg.think_max, 'conv', cfg.k_conv, cfg.t_conv,
            min_step=cfg.min_conv, pos_norm=cfg.think_max)
        ans = self._finish_answer(kv, mask) if fired >= 0 else []
        txt = self.tok.decode(gen + ans, skip_special_tokens=True)
        path = ('escalated' if wasted else 'think') + \
               ('+exit' if fired >= 0 else '')
        return GateResult(text=txt, path=path,
                          tokens=wasted + len(gen) + len(ans),
                          p_easy=p, fire_pos=fired)

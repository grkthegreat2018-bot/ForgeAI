"""Teacher channel - a static pretrained LM as a read-only Hedge
channel (the "hybrid": Qwen provides the generalization floor,
FluxLM's memory channels provide recall above it).

Contract:
- static weights - no training, no journaled state, nothing to
  snapshot/revert (state derives from the token stream, which IS
  journaled).
- lazy: `sync_to(count, ring, ep_start)` feeds only the tokens the
  teacher hasn't seen; small gaps go through the KV cache
  incrementally, large gaps (bulk ingest, load, revert) re-forward
  the last `window` ring tokens.
- `prob(x)` = p(x | stream so far) for the Hedge reward;
  `logp_row()` = dense [V] log-probs for the post-graph logit lift.
- `bulk_logprobs(ids)` = teacher-forced per-position probs for the
  bulk-ingest reward pass (one causal forward per chunk).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def _sdpa_pref():
    """Prefer the memory-efficient kernel - the math backend
    materializes [H, S, S] attention (~5.5GB at 8K+4K context); HF
    picks math whenever an explicit 4D mask is built."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.FLASH_ATTENTION,
                          SDPBackend.MATH]):
            yield
    except (ImportError, AttributeError):
        yield


_DEFAULT_QWEN = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--Qwen--Qwen2.5-0.5B" / "snapshots"
    / "060db6499f32faf8b98477b0a26969ef7d8b9987")


class TeacherChannel:
    """Incremental KV-cached causal-LM view of the FluxLM stream."""

    def __init__(self, path: str = "", device: str = "cuda",
                 window: int = 4096, sink: int = 8,
                 dtype: torch.dtype = torch.float16):
        from transformers import AutoModelForCausalLM, DynamicCache
        p = Path(path) if path else _DEFAULT_QWEN
        if not p.is_dir():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download("Qwen/Qwen2.5-0.5B"))
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(p), dtype=dtype,
                attn_implementation="sdpa").to(device).eval()
        except (TypeError, ValueError):
            self.model = AutoModelForCausalLM.from_pretrained(
                str(p), dtype=dtype).to(device).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.device = device
        self.window = int(window)
        self.sink = int(sink)
        self._mk_cache = DynamicCache
        self.past = None                 # DynamicCache
        self._has_sink = True            # cache starts at stream pos 0
        self.pos = 0                     # stream positions consumed
        self._next: torch.Tensor | None = None   # [V] next-token logits

    # -- state -----------------------------------------------------------
    def _pos_ids(self, n: int) -> torch.Tensor:
        """ABSOLUTE RoPE positions for the next n tokens - positions
        must continue past crop() boundaries: retained keys keep the
        rotations they were written at, so deriving position from the
        cropped cache length would scramble every relative offset."""
        return torch.arange(self.pos, self.pos + n,
                            dtype=torch.long,
                            device=self.device)[None, :]

    def _feed(self, ids: list[int]) -> None:
        """Append ids to the teacher context (KV-incremental)."""
        if not ids:
            return
        t = torch.as_tensor(ids, dtype=torch.long,
                            device=self.device)[None, :]
        with torch.inference_mode(), _sdpa_pref():
            out = self.model(input_ids=t, position_ids=self._pos_ids(
                len(ids)), past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self._next = out.logits[0, -1].float()
        self.pos += len(ids)
        self._truncate()

    def _truncate(self) -> None:
        """Bound the KV cache: keep the first ``sink`` positions
        (attention sinks - sliding windows without them lose ~0.5-1
        NLL) plus the newest ``window - sink`` positions.
        DynamicCache.crop() keeps the FIRST tokens (the prefix) - a
        hand slice is required for tail eviction.  Kept keys retain
        the absolute RoPE rotations they were written at and new
        queries arrive with absolute position_ids, so relative
        offsets stay exact (StreamingLLM semantics)."""
        if self.past is None:
            return
        w = self.window - (self.sink if self._has_sink else 0)
        for layer in self.past.layers:
            L = layer.get_seq_length()
            if L <= self.window:
                continue
            if self._has_sink:
                keep = torch.cat((
                    torch.arange(self.sink, device=layer.keys.device),
                    torch.arange(L - w, L,
                                 device=layer.keys.device)))
            else:
                keep = torch.arange(L - w, L,
                                    device=layer.keys.device)
            layer.keys = layer.keys.index_select(2, keep)
            layer.values = layer.values.index_select(2, keep)

    def _rebuild(self, count: int, ring, ep_start: int) -> None:
        """Re-derive teacher state from the ring tail (post revert,
        load, or a gap too large to backfill incrementally)."""
        self.past = None
        self._next = None
        n = min(self.window, count - ep_start)
        # pos = absolute stream position of the first fed token so
        # _pos_ids assigns true positions and pos lands on count
        # exactly (setting ep_start would double-feed the tail on the
        # next sync_to call).
        self.pos = count - n
        self._has_sink = count - n == 0
        if n > 0:
            cap = len(ring)
            ids = [int(ring[(count - n + i) % cap]) for i in range(n)]
            self._feed(ids)

    def sync_to(self, count: int, ring, ep_start: int = 0) -> None:
        """Advance the teacher to stream position ``count`` (exclusive
        end), feeding missing tokens from the ring."""
        if self.pos > count or (self.pos < count
                                and count - self.pos > len(ring)):
            # reverted/loaded or the missing span fell off the ring
            self._rebuild(count, ring, ep_start)
            return
        if self.pos >= count:
            return
        cap = len(ring)
        ids = [int(ring[i % cap]) for i in range(self.pos, count)]
        if len(ids) <= self.window:
            self._feed(ids)
        else:
            self._rebuild(count, ring, ep_start)

    def reset(self) -> None:
        self.past = None
        self._next = None
        self._has_sink = True
        self.pos = 0

    # -- queries ---------------------------------------------------------
    def prob(self, x: int) -> float | None:
        """p(x | stream) for the next position; None if unwarmed."""
        if self._next is None:
            return None
        return float(torch.softmax(self._next, -1)[x])

    def logp_row(self) -> torch.Tensor | None:
        """[V] log-probs for the next position (device float32)."""
        if self._next is None:
            return None
        return torch.log_softmax(self._next, -1)

    @torch.inference_mode()
    def bulk_logprobs(self, ids, chunk: int = 32768) -> torch.Tensor:
        """p(ids[i] | stream + ids[:i]) for every i - teacher-forced
        rewards for a whole ingest block in ~n/chunk forwards.
        Returns CPU float32 [n] of PROBABILITIES; position 0's prob
        uses the current next-logits (context BEFORE the block)."""
        n = len(ids)
        out = torch.empty(n, dtype=torch.float32)
        if self.past is None:
            self.past = self._mk_cache()
        i = 0
        while i < n:
            seg = ids[i:i + chunk]
            t = torch.as_tensor(seg, dtype=torch.long,
                                device=self.device)[None, :]
            with _sdpa_pref():
                o = self.model(input_ids=t, position_ids=self._pos_ids(
                    len(seg)), past_key_values=self.past,
                    use_cache=True)
            self.past = o.past_key_values
            # logits[j] predicts seg[j+1]; pos 0 uses prior _next.
            # p(x) = logit[x] - logsumexp(row) - gather before any
            # [S,V] fp32 materialization (5GB at 8192x151936)
            tgt = torch.as_tensor(seg, dtype=torch.long,
                                  device=self.device)
            lg = o.logits[0]
            PB = 1024
            if self._next is not None:
                out[i] = float(
                    torch.exp(self._next[tgt[0]]
                              - torch.logsumexp(self._next, -1)))
            else:
                out[i] = 1.0 / float(self._next_numel())
            for p0 in range(0, len(seg) - 1, PB):
                p1 = min(p0 + PB, len(seg) - 1)
                sl = lg[p0:p1].float()
                lp = sl - torch.logsumexp(sl, -1, keepdim=True)
                out[i + 1 + p0:i + 1 + p1] = lp.gather(
                    1, tgt[p0 + 1:p1 + 1, None]).squeeze(
                    1).exp().cpu()
            self._next = lg[-1].float()
            self.pos += len(seg)
            i += len(seg)
            self._truncate()
        return out

    def _next_numel(self) -> int:
        return int(self.model.config.vocab_size)

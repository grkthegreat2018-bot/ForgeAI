"""LiveT — live-training session wrapper for ConfigurableResearchLLM.

Splits the parameter surface into frozen substrate (embed/attn/FFN/norms/
head — the fluency prior, moved only by outer-loop bootstrap) and live
state (GDN2 recurrent states + TitanMemory fast weights — journaled,
exactly revertible, updated during `ingest` without an optimizer).

    cfg = livet_config(d_model=256, n_layers=6, vocab_size=50257)
    model = ConfigurableResearchLLM(cfg)
    # ... bootstrap: ordinary AdamW steps on a corpus (opens gates) ...
    session = LiveSession(model, LiveConfig(chunk=64))
    mark = session.mark()
    stats = session.ingest(ids)          # live: states + fast weights
    session.revert(mark)                 # exact undo of that call
    snap = session.snapshot(); session.load(snap)

Ingest runs chunks under `use_cache=True` with `past_key_values` a list
of None: attention context is intra-chunk (bounded short-term), while
GDN2 conv/recurrent states and TitanMemory fast weights carry across
chunks — and across calls — as the model's long-term live state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class LiveConfig:
    """Live-ingest knobs (journal, surprise gating, inner loop)."""
    chunk: int = 64               # ingest chunk size = attention window
    write_threshold: float = 0.0  # admission: skip chunk when mean
                                  # normalized surprise < this (0=always)
    open_gates: bool = False      # force memory/gdn2 gates to 1.0 at init
                                  # (fresh model w/o bootstrap)
    journal: bool = True          # per-call snapshots for revert()
    # TitanMemory inner-loop overrides (None keeps module/config values):
    mem_lr: float | None = None
    forget: float | None = None
    momentum: float | None = None
    surprise_cap: float | None = None
    solver: str | None = None      # "gd" | "cg" | "solve"
    ridge: float | None = None     # λ prior-strength (cg/solve)
    cg_iters: int | None = None
    # Bounded online drift of the memory format projections. Off by
    # default — literature consensus is these need outer-loop shaping.
    live_projections: bool = False
    proj_lr: float = 1e-3
    proj_clip: float = 1e-2


def livet_config(vocab_size: int = 256, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4,
                 max_seq_len: int = 4096, device: str = "cpu",
                 dtype: str = "float32", **overrides):
    """Small live-trainable ModelConfig: attention blocks + GDN2
    side-paths (working memory) + TitanMemory fast weights (persistent
    memory). Not a preset — a research helper; production checkpoints
    derive via the key/lineage machinery instead."""
    from forge.config import get_config
    cfg = get_config("forgelm_tiny")
    cfg.vocab_size = vocab_size
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.n_kv_heads = n_heads
    cfg.head_dim = None
    cfg.intermediate_size = d_model * 3
    cfg.layer_types = ["attention"] * n_layers
    cfg.max_seq_len = max_seq_len
    cfg.use_gdn2 = True
    cfg.gdn2_n_heads = n_heads
    cfg.gdn2_head_dim = d_model // n_heads
    cfg.use_kda = False            # GDN2 subsumes KDA
    cfg.use_titan_memory = True
    cfg.titan_memory_rank = 0
    cfg.device = device
    cfg.dtype = dtype
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class LiveSession:
    """Journaled live-learning session over a ConfigurableResearchLLM.

    Discovers live surfaces automatically: every block's `_gdn2` (state
    writes per forward) and `_memory` (fast-weight inner loop per chunk).
    Everything else is the frozen substrate — `ingest` writes no
    parameters unless `live_projections` is enabled.
    """

    def __init__(self, model, cfg: LiveConfig | None = None):
        self.model = model
        self.cfg = cfg or LiveConfig()
        self._blocks = list(model.blocks)
        self.mem_layers = {i: b._memory for i, b in enumerate(self._blocks)
                           if getattr(b, "_memory", None) is not None}
        self.gdn2_layers = {i: b._gdn2 for i, b in enumerate(self._blocks)
                            if getattr(b, "_gdn2", None) is not None}
        # Optional exact-suffix sidecar (forge.model.ngram_mem.NGramMemory)
        # — journaled alongside the fuzzy surfaces when attached.
        self.sparse = None
        self.journal: list[dict] = []
        self.stream_pos = 0
        self._stash: dict[int, torch.Tensor] = {}
        self._capturing = False
        self._hooks = []
        for i in self.mem_layers:
            self._hooks.append(self._blocks[i].register_forward_pre_hook(
                self._make_capture(i)))
        for i, mem in self.mem_layers.items():
            if self.cfg.mem_lr is not None:
                mem.update_lr = self.cfg.mem_lr
            if self.cfg.forget is not None:
                mem.forget = self.cfg.forget
            if self.cfg.momentum is not None:
                mem.momentum = self.cfg.momentum
            if self.cfg.surprise_cap is not None:
                mem.surprise_cap = self.cfg.surprise_cap
            if self.cfg.solver is not None:
                mem.solver = self.cfg.solver
            if self.cfg.ridge is not None:
                mem.ridge = self.cfg.ridge
            if self.cfg.cg_iters is not None:
                mem.cg_iters = self.cfg.cg_iters
        if self.cfg.open_gates:
            self.open_gates()

    def _make_capture(self, i: int):
        def hook(module, inp):
            if self._capturing:
                self._stash[i] = inp[0].detach()
        return hook

    # ── gates / freezing ────────────────────────────────────────────────────

    def open_gates(self, value: float = 1.0) -> None:
        """Open memory + gdn2 gates (post-bootstrap or fresh live mode).
        Clears the blocks' cached gate-zero flags so the branches run."""
        for i, mem in self.mem_layers.items():
            mem.gate.data.fill_(value)
            self._blocks[i]._mem_gate_zero = None
        for i, g in self.gdn2_layers.items():
            g.gate.data.fill_(value)
            self._blocks[i]._gdn2_gate_zero = None

    def freeze_substrate(self) -> None:
        """requires_grad=False on every parameter — live learning writes
        buffers (state), never params. Bootstrap optimizers must run
        before this (or call unfreeze_substrate)."""
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_substrate(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    # ── journal ─────────────────────────────────────────────────────────────

    def attach_sparse(self, mem) -> None:
        """Attach an exact-suffix sidecar implementing the live-surface
        contract (update/snapshot_state/restore_state/reset_state/gate)."""
        self.sparse = mem

    def _snapshot_live(self) -> dict:
        return {
            "mem": {i: m.snapshot_state() for i, m in
                    self.mem_layers.items()},
            "gdn2": {i: g.snapshot_state() for i, g in
                     self.gdn2_layers.items()},
            "sparse": (self.sparse.snapshot_state()
                       if self.sparse is not None else None),
            "stream_pos": self.stream_pos,
            "proj": self._snapshot_projections(),
        }

    def _snapshot_projections(self) -> dict:
        if not self.cfg.live_projections:
            return {}
        out = {}
        for i, m in self.mem_layers.items():
            out[i] = {n: p.detach().clone() for n, p in
                      m.named_parameters() if n != "gate"}
        return out

    def _restore_live(self, snap: dict) -> None:
        for i, st in snap["mem"].items():
            self.mem_layers[i].restore_state(st)
        for i, st in snap["gdn2"].items():
            self.gdn2_layers[i].restore_state(st)
        if snap.get("sparse") is not None and self.sparse is not None:
            self.sparse.restore_state(snap["sparse"])
        self.stream_pos = snap["stream_pos"]
        for i, params in snap.get("proj", {}).items():
            named = dict(self.mem_layers[i].named_parameters())
            for n, val in params.items():
                named[n].data.copy_(val)

    def mark(self) -> int:
        return len(self.journal)

    def revert(self, mark: int) -> int:
        """Undo ingest calls back to `mark`. Returns entries reverted."""
        if not self.cfg.journal:
            raise RuntimeError("journal disabled — cannot revert")
        n = 0
        while len(self.journal) > mark:
            self._restore_live(self.journal.pop())
            n += 1
        return n

    def reset(self, memory_too: bool = False) -> None:
        """New stream: clear working memory (GDN2 states + convs).
        `memory_too` additionally wipes persistent fast weights —
        the long-term store normally survives resets."""
        for g in self.gdn2_layers.values():
            g.reset_state()
        if memory_too:
            for m in self.mem_layers.values():
                m.reset_state()

    def snapshot(self) -> dict:
        """Full live state: buffers + states + journal + position."""
        return {
            "live": self._snapshot_live(),
            "journal": copy.deepcopy(self.journal),
        }

    def load(self, snap: dict) -> None:
        self._restore_live(snap["live"])
        self.journal = copy.deepcopy(snap["journal"])

    # ── ingest / eval ───────────────────────────────────────────────────────

    @torch.no_grad()
    def ingest(self, ids: torch.Tensor | list[int],
               learn: bool = True) -> dict:
        """Live-learn a token stream. ids: (T,) or (1,T). Writes GDN2
        states (via forward) and TitanMemory fast weights (inner loop);
        journaled per call when cfg.journal. Returns stats."""
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.reshape(1, -1).to(next(self.model.parameters()).device)
        was_training = self.model.training
        self.model.eval()
        if self.cfg.journal:
            self.journal.append(self._snapshot_live())

        # Successor-embedding write target: memory learns
        # context-key → embed(next_token), a learned-format associative
        # store of realized transitions. Falls back to the module's own
        # v_proj target when the embedding table isn't directly readable.
        emb_w = getattr(getattr(self.model, "embed", None),
                        "weight", None)
        tgt_all = None
        if emb_w is not None and emb_w.shape[-1] == self.model.config.d_model:
            nxt = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)
            tgt_all = F.embedding(nxt, emb_w).detach()

        chunk = max(self.cfg.chunk, 2)
        n_chunks, writes, losses, surprisals = 0, 0, [], []
        past_nones = [None] * len(self._blocks)
        self._capturing = True
        try:
            for start in range(0, ids.shape[1], chunk):
                seg = ids[:, start:start + chunk]
                if seg.shape[1] < 2:
                    break
                self._stash.clear()
                logits = self.model(seg, use_cache=True,
                                    past_key_values=list(past_nones))[0]
                if self.sparse is not None:
                    logits = logits + self.sparse.lift(seg)
                # Per-token surprise: CE of next-token predictions.
                tgt = ids[:, start + 1:start + seg.shape[1]]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                    tgt.reshape(-1), reduction="none")
                losses.append(float(ce.mean()))
                s = ce / ce.mean().clamp_min(1e-8)
                # Last position's target is next chunk's first token —
                # fill with the chunk mean (boundary token, minor).
                s = torch.cat([s, s.mean().expand(1)])
                mean_s = float(s.mean())
                surprisals.append(mean_s)
                n_chunks += 1
                if not learn or (self.cfg.write_threshold > 0
                                 and mean_s < self.cfg.write_threshold):
                    continue
                for i, mem in self.mem_layers.items():
                    x0 = self._stash.get(i)
                    if x0 is None:
                        continue
                    tgt = (tgt_all[:, start:start + seg.shape[1]]
                           if tgt_all is not None else None)
                    mem.update(
                        x0[:, -seg.shape[1]:], surprise=s.view(1, -1),
                        target=tgt,
                        proj_lr=self.cfg.proj_lr
                        if self.cfg.live_projections else 0.0,
                        proj_clip=self.cfg.proj_clip)
                    writes += 1
                if self.sparse is not None:
                    self.sparse.update(seg)
        finally:
            self._capturing = False
            self._stash.clear()
            if was_training:
                self.model.train()
        self.stream_pos += int(ids.numel())
        return {"tokens": int(ids.numel()), "chunks": n_chunks,
                "writes": writes,
                "mean_loss": sum(losses) / max(len(losses), 1),
                "mean_surprise": sum(surprisals) / max(len(surprisals), 1)}

    @torch.no_grad()
    def eval_nll(self, ids: torch.Tensor | list[int],
                 use_live_state: bool = True,
                 warm: bool = False) -> float:
        """Mean next-token CE without mutating live state. With
        use_live_state=False the memory/gdn2 gates are temporarily zeroed
        — measures the frozen substrate alone. warm=True keeps the
        current recurrent state instead of resetting (stream parity)."""
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.reshape(1, -1).to(next(self.model.parameters()).device)
        snap = self._snapshot_live()
        saved_gates = {}
        try:
            if not use_live_state:
                for i, mem in self.mem_layers.items():
                    saved_gates[("mem", i)] = float(mem.gate.item())
                    mem.gate.data.zero_()
                    self._blocks[i]._mem_gate_zero = None
                for i, g in self.gdn2_layers.items():
                    saved_gates[("gdn2", i)] = float(g.gate.item())
                    g.gate.data.zero_()
                    self._blocks[i]._gdn2_gate_zero = None
            if not warm:
                self.reset()
            losses = []
            chunk = max(self.cfg.chunk, 2)
            for start in range(0, ids.shape[1] - 1, chunk):
                seg = ids[:, start:start + chunk + 1]
                logits = self.model(seg, use_cache=True,
                                    past_key_values=[None] *
                                    len(self._blocks))[0]
                if use_live_state and self.sparse is not None:
                    logits = logits + self.sparse.lift(seg)
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                    seg[:, 1:].reshape(-1), reduction="none")
                losses.append(float(ce.mean()))
            return sum(losses) / max(len(losses), 1)
        finally:
            self._restore_live(snap)
            for (kind, i), val in saved_gates.items():
                mod = (self.mem_layers if kind == "mem"
                       else self.gdn2_layers)[i]
                mod.gate.data.fill_(val)
                blk = self._blocks[i]
                if kind == "mem":
                    blk._mem_gate_zero = None
                else:
                    blk._gdn2_gate_zero = None

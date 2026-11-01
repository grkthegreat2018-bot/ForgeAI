"""Code evolution loop — OpenEvolve-style program search on ForgeAI infra.

Feed it a target (an existing function/script that must get better along a
flagged direction: faster, lighter, smaller). Generators write BRAND NEW
implementations — not value tweaks — and every candidate is compared
against three baselines at once:

  1. the original code (seed program, re-benched for honesty),
  2. the last verified best (incumbent champion),
  3. the current round's candidates (per-island elite archive).

Candidate path: LLM prompt (goal + incumbent code + its metrics +
cross-domain insights + one inspiration program) -> brand-new code ->
syntax gate -> dedup -> isolated subprocess cascade (target.test()
correctness -> target.bench() metrics). A challenger becomes champion
only after beating the incumbent's re-measured mean by a noise-aware
margin over --reps replicated evals, AND satisfying every spec BUDGET —
that is the "fits requested end result" gate.

Diversity: N islands keep independent elite pools; parents are sampled by
rank-softmax plus a random inspiration pick (OpenEvolve double-selection).
A plateau triggers a basin hop (island reset + temperature bump) before
retirement. The run stops on obvious stagnation: --patience consecutive
evals without a verified promotion.

Canonical loop: propose -> gate -> eval-pool -> verify -> promote ->
export_sft -> finetune_proposer -> hot-swap. The first five stages run
per candidate inside run(); export_sft + scripts/finetune_proposer.py
turn verified wins into a proposer LoRA, hot-swapped onto the loaded
proposer engine (tag-guarded, base-checked) for the next round.

Cross-domain learning: every verified win writes a code_insights row
(tags, summary, delta); proposers for later targets retrieve top-k
insights by tag overlap, so a trick proven on one kernel seeds prompts
for the next.

Target spec = a python file (see forge/evolution/targets/fast_softmax.py):
    NAME, GOAL ("min"|"max"), METRIC, TAGS, BUDGETS{metric: ("<=",v)}
    SEED_CODE: str          # the original implementation
    REQUIRED: [names]       # public names the candidate must define
    def test(ns) -> bool|dict   # correctness gate (ns = candidate namespace)
    def bench(ns) -> dict       # metrics incl. METRIC

Usage:
    python scripts/fn_evolve.py --target forge/evolution/targets/fast_softmax.py
    python scripts/fn_evolve.py --emit-target forge.engine.decoding:sample \
        --metric ms_per_call --goal min > my_target.py
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .database import FindingsDB
from .mutator import Mutator, MixProposer
from .scout import _Welford

ROOT = Path(__file__).resolve().parents[2]
TARGETS_DIR = Path(__file__).resolve().parent / "targets"
# Non-code evolution domains (math solvers, codecs, parsers) live in
# domains/ as cev_*.py spec files — same SEED_CODE/test/bench contract,
# proving the loop isn't codegen-only.
DOMAINS_DIR = Path(__file__).resolve().parent / "domains"

_CODE_DDL = [
    """CREATE TABLE IF NOT EXISTS code_programs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        target TEXT NOT NULL,
        island INTEGER,
        gen INTEGER,
        code_hash TEXT,
        code TEXT,
        parent_hash TEXT,
        metrics_json TEXT,
        score REAL,
        ok INTEGER,
        stage TEXT,
        ts REAL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_code_prog ON code_programs(target, code_hash)",
    """CREATE TABLE IF NOT EXISTS code_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target TEXT NOT NULL,
        tags TEXT,
        kind TEXT,
        summary TEXT,
        delta REAL,
        score REAL,
        ts REAL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_code_ins ON code_insights(tags)",
    """CREATE TABLE IF NOT EXISTS mutator_op_stats (
        op TEXT NOT NULL,
        tag_sig TEXT NOT NULL,
        alpha REAL DEFAULT 1.0,
        beta REAL DEFAULT 1.0,
        n INTEGER DEFAULT 0,
        delta_mean REAL DEFAULT 0.0,
        PRIMARY KEY (op, tag_sig)
    )""",
]

_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "=": lambda a, b: abs(a - b) < 1e-9}


# ── config ──────────────────────────────────────────────────────────────

@dataclass
class CodeEvolveConfig:
    target: str                                # path or name in targets/
    metric: str | None = None                  # override spec METRIC
    goal: str | None = None                    # override spec GOAL
    evals_budget: int = 200                    # max subprocess evals
    reps: int = 2                              # verify evals per side
    min_delta: float = 0.0                     # abs margin floor
    rel_delta: float = 0.02                    # margin vs |incumbent|
    k_sigma: float = 1.0                       # margin += k * pooled SEM
    islands: int = 3
    n_elites: int = 8
    island_cull: bool = True                 # hop: discard worst-half
                                           # islands + clone best
                                           # (FunSearch), else legacy
                                           # top-1 reset
    map_elites: bool = True                  # per-(island, len-bin)
                                           # elite cells — short AND
                                           # long programs survive
    cheap_bench: bool = True                 # eval cascade: reduced-
                                           # rep bench gate before the
                                           # full bench (needs spec
                                           # BENCH_REPS)
    cheap_reps: int = 1                      # reduced bench reps
    cheap_margin: float = 1.0                # skip full bench when the
                                           # cheap metric is >1.0x
                                           # (fraction) worse than the
                                           # incumbent (~2x worse)
    hint_bandit: bool = True                 # Thompson Beta bandit
                                           # over LLMProposer._HINTS
                                           # (meta-prompt evolution)
    hop_after: int = 12                        # dry evals before basin hop
    patience: int = 30                         # dry evals before stop
    proposer: str = "mut"                      # mut | mix | llm | basin
                                               # | none
    llm_every: int = 6                         # mix: LLM every Nth cycle
    basin_batch: int = 16                      # basin: LLM parents per
                                               # generate_batch pass
                                               # (B=16 static-graph decode
                                               # ≈1k tok/s aggregate; B=32
                                               # measured 5.8GB peak —
                                               # viable but tighter)
    basin_q: int = 24                          # basin: queued LLM parents
    basin_starve_s: float = 240.0              # basin: max seconds with-
                                               # out a PRODUCTIVE basin
                                               # (one that yielded >=1
                                               # variant) before a dry
                                               # stall counts — covers
                                               # model load + slow batches
    fanout: int = 32                           # basin: mutator variants
                                               # enumerated per parent
    inc_stale: int = 48                        # verify: re-bench incumbent
                                               # after this many evals
    eval_pool: bool = True                     # persistent eval workers
                                               # (False: subprocess/eval)
    resume: bool = True                        # seed from prior best for
                                             # this target (same db)
    model_id: str | None = None                # default: ported
                                               # proposer_mla_moe dir,
                                               # else harness QWEN_CODER
    proposer_device: str = "cuda"
    proposer_backend: str = "forge"            # "forge" | "ov" (OpenVINO
                                             # GenAI server — NPU/CPU
                                             # offload, no LoRA hot-swap)
    proposer_ov_url: str = "http://127.0.0.1:8790"
    engine_config: str | None = None           # ForgeEngine preset override
                                               # (else forge_convert.json /
                                               # qwen25_05b default)
    proposer_quantize: str | None = None       # engine load_quantize mode
                                               # ("int4"/"fp8"/"int8"/None)
    proposer_think: bool = False               # LFM3 only: open the think
                                               # prefill (reason before
                                               # code) instead of the
                                               # template's empty think
                                               # pair. Costs gen tokens.
    proposer_compile: bool = False             # torch.compile the proposer.
                                               # Decode replays bucketed
                                               # static-KV graphs, so compile
                                               # only speeds prefill — but it
                                               # costs ~3-4min of inductor at
                                               # EVERY cold boot. Off by
                                               # default; enable for long
                                               # campaigns where prefill time
                                               # amortizes it.
    lora_root: str | None = None               # tagged-LoRA registry dir;
                                               # proposer hot-swaps per target
    lora_tag: str | None = None                # adapter tag override
                                               # (default: target stem ->
                                               # registry fallback chain)
    max_gen_tokens: int = 640                  # <think> block ~150 tok + fenced code ~300
                                               # 640 wasted ~35% decode
    temperature: float = 0.8
    db_path: str = "forge_evolve.db"
    timeout_s: float = 120.0
    n_insights: int = 5
    workers: int = 1                        # parallel eval subprocesses;
                                          # 0 = auto-ramp to ~85% load
    max_workers: int = 24                   # hard cap for auto mode
    cpu_target: float = 0.85                # auto mode CPU goal
    mem_limit: float = 0.80                 # stop growing past this RAM%
    worker_mem_mb: int = 2048               # per-worker RSS cap (kill+
                                          # respawn on breach); 0 = off
    status_file: str | None = None          # live progress JSON (GUI/
                                            # evolve_watch poll this)
    research: bool = False                  # SEARCH: tool calls +
                                            # digest injection in
                                            # proposal prompts
    research_digest: str | None = None      # md digest trimmed ~1.6KB
                                            # into prompts
    research_max_queries: int = 2           # SEARCH lines per proposal
    research_budget: int = 32               # uncached API searches per
                                            # process (shared, ~daily)
    research_hits: int = 3                  # hits injected per query
    research_cache: str | None = None       # jsonl query-cache path
    judge: bool = True                      # same-model critiques on
                                            # wins + plausible rejects;
                                            # costs proposer calls
                                            # (default: <lora_root>/
                                            # research_cache.jsonl)
    repair_regen: bool = True               # artifact feedback: ONE
                                            # repair regen per failed
                                            # candidate — eval errors
                                            # (traceback) AND gate
                                            # rejects (soup/missing/dup
                                            # reason) go back into the
                                            # prompt, STaR-style
    repair_per_batch: int = 4               # repair regens per eval-
                                            # result batch (cost bound)
    seed: int = 0
    verbose: bool = True
    eval_log: bool = False                  # one stderr line per eval
                                            # result (stage/score/hash/
                                            # island) — the crash-locator
                                            # flag; implies verbose
    log_gens: bool = False                  # log raw model emissions to
                                            # <db>.gens.jsonl (one row
                                            # per completion — full
                                            # prompt+raw, unredacted)


# ── target spec ─────────────────────────────────────────────────────────

class Target:
    """Loaded evaluator module for one evolution target."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        spec = importlib.util.spec_from_file_location("_ce_target", self.path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.mod = mod
        self.name = getattr(mod, "NAME", self.path.stem)
        self.seed_code = getattr(mod, "SEED_CODE")
        self.goal = getattr(mod, "GOAL", "min")
        # METRIC may be absent when the caller overrides via --metric —
        # CodeEvolve validates the resolved value before use.
        self.metric = getattr(mod, "METRIC", None)
        self.tags = list(getattr(mod, "TAGS", []))
        self.budgets = dict(getattr(mod, "BUDGETS", {}))
        self.required = list(getattr(mod, "REQUIRED", []))
        self.prompt_extra = getattr(mod, "PROMPT", "")

    @staticmethod
    def resolve(target: str) -> Path:
        p = Path(target)
        if p.exists():
            return p
        for d in (TARGETS_DIR, DOMAINS_DIR):
            alt = d / (target if target.endswith(".py")
                       else target + ".py")
            if alt.exists():
                return alt
        raise FileNotFoundError(
            f"target not found: {target} (tried {p}, {TARGETS_DIR}, "
            f"{DOMAINS_DIR})")

    def test(self, ns: dict):  # pragma: no cover — runs in subprocess
        return self.mod.test(ns)

    def bench(self, ns: dict):  # pragma: no cover — runs in subprocess
        return self.mod.bench(ns)


# ── LLM proposer ────────────────────────────────────────────────────────


def _lora_base_ok(ent, engine_ckpt: str) -> bool:
    """Adapter/base compatibility — a registry entry trained on a
    different checkpoint must NOT attach: name-matched modules with
    wrong shapes crash the forward (the 896-vs-2048 basin-emitter
    crash), name-mismatched ones silently no-op. Registry ``resolve``
    falls back to the ``default`` tag without checking this, so the
    guard lives at swap time.

    ``ent.base`` is whatever ``--base`` finetune_proposer got: an
    absolute or repo-relative checkpoint dir, or an HF id. Resolved
    paths are compared when the base looks like a dir; HF ids and
    moved checkpoints compare on the directory-name tail."""
    base = (getattr(ent, "base", "") or "").strip()
    if not base:
        return True                      # unknown provenance — try it
    if not engine_ckpt:
        return True                      # HF proposer has no ckpt dir
    # A packed quant dir (`X_fq`, `X_int4`, ...) holds the SAME weights
    # as the source ckpt — adapters trained on either are equivalent.
    def _tail(name: str) -> str:
        t = Path(name).name.lower()
        for suf in ("_fq", "_int4", "_fp8", "_int8", "_nf4"):
            if t.endswith(suf):
                t = t[: -len(suf)]
                break
        return t
    ck_name = _tail(engine_ckpt)
    base_norm = base.replace("/", "\\").rstrip("\\").lower()
    if "/" in base or "\\" in base or Path(base).is_dir():
        for cand in (Path(base), ROOT / base):
            try:
                if cand.is_dir() and \
                        cand.resolve() == Path(engine_ckpt).resolve():
                    return True
            except Exception:
                pass
        return ck_name == _tail(base_norm.split("\\")[-1])
    return ck_name == _tail(base_norm.split("/")[-1]) or \
        engine_ckpt.replace("\\", "/").rstrip("/").lower().endswith(
            "/" + base_norm)


class OvEngine:
    """OpenVINO GenAI proposer backend — talks to ov_proposer_server.py
    (venv_ov, NPU/CPU) over localhost HTTP. Mimics the ForgeEngine
    surface LLMProposer uses (tokenizer / generate / generate_batch)
    so the whole batched path — wrap, hints, sampling kwargs, research
    followup — works unchanged. No _load_lora_impl: _ensure_lora
    no-ops (ForgeAI-format adapters can't hot-swap onto an OV IR; a
    PEFT-export path is queued). checkpoint_path mirrors the forge
    ckpt so _lora_base_ok stays meaningful if that lands later."""

    def __init__(self, model_id: str, url: str = "http://127.0.0.1:8790",
                 timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        # tokenizer for eos/decode only — generation happens server-side.
        # gigatoken tokenizer.json isn't an AutoTokenizer class on
        # transformers 4.57 — load the fast wrapper directly and set
        # eos to the ChatML terminator.
        try:
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(Path(model_id) / "tokenizer.json"))
            eos = self.tokenizer.convert_tokens_to_ids(
                "<" + "|im_end|>")
            if eos is not None:
                self.tokenizer.eos_token_id = eos
        except Exception:
            self.tokenizer = None
        self.checkpoint_path = str(model_id)
        import urllib.request
        with urllib.request.urlopen(f"{self.url}/health",
                                    timeout=10) as r:
            self.device = json.loads(r.read()).get("device", "?")

    def _post(self, prompts, cfgs):
        import urllib.request
        body = json.dumps({"prompts": prompts,
                           "configs": cfgs}).encode()
        req = urllib.request.Request(
            f"{self.url}/generate", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def generate_batch(self, prompts, max_new_tokens=448, temperature=0.8,
                       top_p=0.95, top_k=80, temperatures=None,
                       repetition_penalties=None, seeds=None, stops=None,
                       skip_special_tokens=True, **_kw):
        n = len(prompts)
        cfgs = []
        for i in range(n):
            cfgs.append({
                "max_new_tokens": max_new_tokens,
                "temperature": (temperatures[i] if temperatures
                                else temperature),
                "top_p": top_p, "top_k": top_k,
                "repetition_penalty": (repetition_penalties[i]
                                       if repetition_penalties else 1.0),
            })
        out = self._post(list(prompts), cfgs)
        texts = out.get("texts", [""] * n)
        gs = out.get("gen_seconds")
        if gs:
            print(f"[ov] batch n={n} in {gs}s", flush=True)
        # OV GenAI doesn't take stop strings — trim client-side
        if stops:
            for i, t in enumerate(texts):
                for s in (stops[i] if i < len(stops) else stops[0]):
                    j = t.find(s)
                    if j >= 0:
                        t = t[:j]
                texts[i] = t
        return texts

    def generate(self, prompt, max_new_tokens=448, temperature=0.8,
                 top_p=0.95, top_k=80, repetition_penalty=1.0,
                 finish_sentence=False, stop=None, **_kw):
        texts = self.generate_batch(
            [prompt], max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalties=[repetition_penalty],
            stops=[stop] if stop else None)
        return texts[0] if texts else ""


class LLMProposer:
    """Brand-new implementations from a coder model.

    Default: Qwen/Qwen2.5-Coder-3B-Instruct when in the HF cache — the
    A5 gate-rate head-to-head (results/proposer_bench*.json) measured
    100% valid / 100% syntax vs LFM3-fq's 33% / 50%. Fallback order:
    research/checkpoints/lfm3_1p7b_cc_fq then lfm3_1p7b_cc (LFM3-1.7B
    Claude-Code finetune, ForgeQuant int4-packed, ~1.4GB) — chat-
    template prompts with a code-fence prefill (`<think></think>
    ```python`) so output is verbatim code. The old ported-Qwen dirs
    are ARCHIVED (verbatim-copy + parse-failure history) — see
    _model_id for the resolution order.

    Local checkpoint dirs try ForgeEngine first (same arch → batched
    generate_batch ~4x); other archs fall back to transformers
    AutoModelForCausalLM + the model's chat template (`primed=False`).
    """

    # Shared engine table: parallel target runs in one process load the
    # proposer ONCE — an 8GB 4B model can't be instantiated per run.
    # kind/obj fill in under "ready"; "lock" serializes generation so
    # sharing an HF model across threads stays safe (ForgeEngine already
    # serializes internally via _gen_lock — same order everywhere).
    _ENGINES: dict = {}
    _ENGINES_LOCK = threading.Lock()

    @classmethod
    def release_engines(cls, fast: bool = True) -> None:
        """Unload every cached proposer engine and free its VRAM.

        The evolve↔hone lifecycle wants the proposer resident ONLY while
        generating proposals: release between the evolve phase and the
        train phase so the hone gets the full GPU.

        fast=True (default): sleep(1) — weights offload to CPU RAM, the
        engine ENTRY stays cached, and the next _engine() call wake()s
        (~2-3s H2D copy) instead of a full from_checkpoint rebuild
        (weights + compile + per-bucket graph capture). Acceleration
        resources are released BEFORE the sleep: captured decode graphs
        hold pointers into the old GPU weight storage, which wake()
        reallocates at new addresses — replaying them post-wake is stale-
        pointer corruption, so they must be re-captured via activate.
        Costs ~model-size of CPU RAM; set fast=False for the old
        sleep(2)+clear path when RAM is tight.
        """
        with cls._ENGINES_LOCK:
            for key, ent in list(cls._ENGINES.items()):
                obj = ent.get("obj")
                if ent.get("kind") == "forge" and obj is not None:
                    if fast and getattr(obj, "_awake", True):
                        try:
                            rel = getattr(
                                obj, "_release_acceleration_resources",
                                None)
                            if callable(rel):
                                rel()
                            obj.sleep(1)
                            ent["asleep"] = True
                            continue
                        except Exception:
                            pass      # fall through to deep path
                    # ForgeEngine has no unload() — the old call was a
                    # silent no-op and VRAM release depended on GC of
                    # the dropped dict entry. sleep(2) deterministically
                    # discards weights + graph pools + static caches;
                    # close() also tears down crash recovery.
                    try:
                        obj.sleep(2)
                    except Exception:
                        try:
                            obj.close()
                        except Exception:
                            pass
                    cls._ENGINES.pop(key, None)
                else:
                    cls._ENGINES.pop(key, None)
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __init__(self, cfg: CodeEvolveConfig, db=None,
                 tag_sig: str = ""):
        self.cfg = cfg
        self.db = db
        self.tag_sig = tag_sig
        self._eng = None
        self._hf = None
        self._lora_resolved = False
        self._lora_cached = None
        self._router = None
        self._last_adapter = None
        # Hint bandit (meta-prompt evolution-lite): Beta arms over
        # _HINTS, persisted in FindingsDB.mutator_op_stats as
        # op='hint:<idx>' rows keyed by tag_sig — same bandit contract
        # as Mutator. Lazily initialized so __new__-built stubs work.
        self._hint_stats: list[list[float]] | None = None
        self._hint_by_prompt: dict[str, int] = {}
        self._hint_rng = None
        self._last_hint: int | None = None
        # Prewarm: begin the engine build on a daemon thread NOW — weight
        # load + tokenizer + kernel warmup overlap pool boot and seed
        # eval instead of blocking the first proposal call (~15-25s
        # stolen off the critical path). A build failure here just
        # evicts the cache entry; the real call retries and surfaces it.
        def _prewarm():
            try:
                self._engine()
            except Exception:
                pass
        threading.Thread(target=_prewarm, daemon=True,
                         name="proposer_prewarm").start()

    @property
    def primed(self) -> bool:
        """Completion-style priming unless the model actually has a chat
        template. Local SFT checkpoints (finetune_proposer.py output)
        inherit the base tokenizer — no chat template — so they take the
        completion format they were trained on. A ported checkpoint can
        force the choice explicitly via forge_convert.json {"primed": b}
        — proposer_mla_moe sets it: its tokenizer ships a chat_template
        but the weights were SFT'd on primed evolve prompts and only
        produce code as a completion model."""
        cached = getattr(self, "_primed_cached", None)
        if cached is not None:
            return cached
        result = self._primed_impl()
        self._primed_cached = result
        return result

    def _primed_impl(self) -> bool:
        if self.cfg.proposer_backend == "ov":
            # The OV export tracks base LFM3 instruct — the primed
            # assistant-turn trick only works on the Claude-Code
            # finetune. Unprimed ChatML -> <think> -> fenced code,
            # which the funnel parses (had_think is logged anyway).
            return False
        from forge.exp.harness import QWEN_CODER_ID
        mid = self._model_id
        if mid == QWEN_CODER_ID:
            return True
        p = Path(mid)
        if p.exists() and p.is_dir():
            conv = p / "forge_convert.json"
            if conv.exists():
                try:
                    pr = json.loads(conv.read_text()).get("primed")
                    if pr is not None:
                        return bool(pr)
                except Exception:
                    pass
            if (p / "chat_template.jinja").exists():
                return False
            try:
                tc = json.loads(
                    (p / "tokenizer_config.json").read_text(
                        encoding="utf-8"))
                return "chat_template" not in tc
            except Exception:
                return True
        return False

    @property
    def _lfm3(self) -> bool:
        """LFM3-family proposer — the official LFM3 chat template
        (chat_template.jinja on HF) is ChatML plus two things our generic
        _wrap omits: the startoftext BOS prefix, and a think-prefill at
        the top of every assistant turn (enable_thinking=False renders
        an empty think-close pair, then the answer — that is the shape
        we emit; primed assistant turns get the empty pair too)."""
        c = getattr(self, "_lfm3_cached", None)
        if c is not None:
            return c
        mid = self._model_id or ""
        c = "lfm3" in mid.lower()
        p = Path(mid)
        if not c and p.is_dir():
            # Corroborate via bos_token — a renamed ckpt dir still
            # carries the LFM tokenizer's startoftext BOS.
            try:
                bos = json.loads(
                    (p / "tokenizer_config.json").read_text(
                        encoding="utf-8")).get("bos_token") or ""
                c = "startoftext" in bos
            except Exception:
                pass
        self._lfm3_cached = c
        return c

    @property
    def _model_id(self) -> str:
        if self.cfg.model_id:
            return self.cfg.model_id
        # Default proposer: Qwen2.5-Coder-3B-Instruct — the A5 gate-rate
        # head-to-head (results/proposer_bench*.json) measured 100%
        # valid / 100% syntax on the 12-prompt grid vs LFM3-fq's
        # 33% / 50%.  Preferred ONLY when already in the HF cache
        # (snap_dir is a pure local check — no implicit download); the
        # local LFM3 checkpoints stay the offline fallback, and the old
        # ported-Qwen dir stays ARCHIVED (verbatim-copy + parse-failure
        # history — do not re-add it as a fallback).
        qwen = "Qwen/Qwen2.5-Coder-3B-Instruct"
        try:
            from forge.exp.harness import snap_dir
            snap_dir(qwen)
            return qwen
        except Exception:
            pass
        for d in ("lfm3_1p7b_cc_fq", "lfm3_1p7b_cc"):
            p = ROOT / "research" / "checkpoints" / d
            if (p / "forge_convert.json").exists():
                return str(p)
        raise FileNotFoundError(
            "no proposer model found: cache Qwen/Qwen2.5-Coder-3B-"
            "Instruct, convert research/checkpoints/lfm3_1p7b_cc_fq, "
            "or pass --model <dir|hf-id>")

    def _engine(self):
        """(kind, obj, lock) — cached across LLMProposer instances so all
        parallel runs in this process share one loaded model. A waiter
        that held a dead builder's entry used to get kind=None and crash
        unpacking `model, tok = eng`; it now retries once on a fresh
        entry so the real build error surfaces."""
        # Build-affecting config must key the cache: two configs sharing
        # (model, device) but differing on engine_config/quantize used
        # to get each other's already-built engine — silently wrong
        # quantization/load flags for the second run.
        key = (self._model_id, self.cfg.proposer_device,
               self.cfg.engine_config, self.cfg.proposer_quantize,
               self.cfg.proposer_compile)
        for _attempt in range(2):
            with LLMProposer._ENGINES_LOCK:
                ent = LLMProposer._ENGINES.get(key)
                if ent is None or (ent["ready"].is_set()
                                   and ent["kind"] is None):
                    ent = {"ready": threading.Event(),
                           "lock": threading.Lock(),
                           "kind": None, "obj": None}
                    LLMProposer._ENGINES[key] = ent
                    builder = True
                else:
                    builder = False
            if builder:
                try:
                    ent["kind"], ent["obj"] = self._load_engine_obj()
                except Exception:
                    with LLMProposer._ENGINES_LOCK:
                        if LLMProposer._ENGINES.get(key) is ent:
                            LLMProposer._ENGINES.pop(key, None)
                    ent["ready"].set()
                    raise
                ent["ready"].set()
            else:
                ent["ready"].wait()
            if ent["kind"] is not None:
                obj = ent["obj"]
                if ent.pop("asleep", False) and obj is not None:
                    # Fast-release wake: weights were CPU-offloaded with
                    # graphs already freed (stale-pointer safety). Wake
                    # moves weights back; re-activation re-captures the
                    # decode graphs against the NEW tensor storage — an
                    # L1 wake alone would replay pointers into freed
                    # memory. Under the gen lock: decode must not start
                    # between wake and re-capture.
                    with ent["lock"]:
                        try:
                            obj.wake()
                            params = getattr(
                                obj, "_last_activation_params", None)
                            if params:
                                from forge.engine.activation import \
                                    ActivationConfig
                                obj.activate_config(
                                    ActivationConfig.from_kwargs(
                                        **params))
                        except Exception:
                            # Wake failed mid-way — drop the entry so
                            # the next call does a full clean rebuild
                            # rather than trusting a half-live engine.
                            with LLMProposer._ENGINES_LOCK:
                                if LLMProposer._ENGINES.get(key) is ent:
                                    LLMProposer._ENGINES.pop(key, None)
                            raise
                return ent["kind"], ent["obj"], ent["lock"]
        raise RuntimeError(
            f"proposer engine failed to load: {self._model_id}")

    def _load_engine_obj(self):
        from forge.exp.harness import QWEN_CODER_ID
        if self._model_id == QWEN_CODER_ID:
            if self._eng is None:
                from forge.exp.harness import load_engine
                # Chunked prefix cache (LMCache rolling-hash) — the 8
                # prompts in a basin batch share ~90% of their prefix
                # (instructions + seed + parent); cached chunks turn
                # 8 prefills into ~1.
                self._eng = load_engine(
                    self._model_id, device=self.cfg.proposer_device,
                    use_chunked_prefix_cache=True)
            return ("forge", self._eng)
        # OpenVINO offload: generation runs in venv_ov on NPU/CPU —
        # frees the 5070 for eval workers. kind "forge" reuses the
        # batched path wholesale.
        if (self.cfg.proposer_backend == "ov"
                and self._eng is None):
            self._eng = OvEngine(self._model_id,
                                 self.cfg.proposer_ov_url)
            print(f"  [cevolve] proposer backend: OpenVINO "
                  f"({self._eng.device}) @ {self.cfg.proposer_ov_url}",
                  flush=True)
            return ("forge", self._eng)
        # Local dirs + HF ids: try ForgeEngine first — same arch gets
        # generate_batch (4-5x) instead of HF serial generation.
        # Non-matching archs (7B AWQ etc.) fall through to transformers.
        if self._eng is None and self._hf is None:
            eng_err = None
            conv = None
            try:
                p = Path(self._model_id)
                if p.is_dir():
                    ckpt_dir = p.resolve()
                else:
                    from forge.exp.harness import snap_dir
                    ckpt_dir = Path(snap_dir(self._model_id))
                # forge_convert.json (written by port/convert tools) names
                # the arch preset; cfg.engine_config overrides it; bare HF
                # dirs default to qwen25_05b.
                named = "qwen25_05b"
                conv = Path(ckpt_dir) / "forge_convert.json"
                if conv.exists():
                    named = json.loads(conv.read_text()).get(
                        "forge_config", named)
                if self.cfg.engine_config:
                    named = self.cfg.engine_config
                from forge.engine.forge_engine import ForgeEngine
                self._eng = ForgeEngine.from_checkpoint(
                    str(ckpt_dir), config_name=named,
                    device=self.cfg.proposer_device, quantize=None,
                    load_quantize=self.cfg.proposer_quantize,
                    kv_cache="standard",
                    use_compile=self.cfg.proposer_compile,
                    # Bucketed static-KV decode graphs: pos_t + conv-state
                    # banked-buffer ports make capture/replay correct at
                    # B<=32 on LFM3 (conv state no longer reallocates at
                    # position-0 reset — see DoubleGatedConvLayer bank).
                    # Verified by scripts/bench_lfm3_evolve.py token-
                    # equality + tok/s harness; functional decode remains
                    # the fallback when a batch can't be captured.
                    use_breakable_cuda_graph=True,
                    use_block_fusion=False,
                    use_chunked_prefix_cache=True,
                    use_learned_prefix_cache=False,
                    # Aux decode features (EAGLE-3 spec draft, FASER,
                    # HotPrefix, POD-attention) each reserve their own
                    # VRAM — a proposer generating short candidates does
                    # not need them, and on a prequant 3B they push
                    # warmup past the 12GB wall.
                    use_adaptive_spec=False, use_suffix_spec=False,
                    use_faser=False, use_lookahead_gate=False,
                    use_hotprefix=False, use_pod_attention=False,
                    use_hybrid_prefill=False,
                    # KV budget defaults to ALL free VRAM (vram_manager
                    # computes usable_mb/bytes_per_token): the LFM3
                    # proposer grabbed 7.9GB→160k tokens, hit 0 free
                    # mid-batch and thrashed through a paged hotswap.
                    # A proposer prompt is ~2k + 640 gen — 8k is ample.
                    kv_cache_tokens=8192)
                # cuBLAS decode path (measured 3.3x faster replay than
                # the fused Triton int4 GEMM at M=16 on the 5070 — the
                # int4 kernel doesn't use tensor cores: 12.7ms vs
                # 42.1ms/step): dequant each linear once to bf16, then
                # every decode step is a plain cuBLAS GEMM. Costs ~3.4GB
                # VRAM; bucket right-sizing keeps B=32 at ~5.8GB peak.
                n_cu = 0
                for m in self._eng.model.modules():
                    if (hasattr(m, "_fused_gemv")
                            and hasattr(m, "_dequantize_weight")):
                        m._fused_gemv = False
                        m._cache_w = True
                        n_cu += 1
                if n_cu:
                    print(f"  [cevolve] proposer cuBLAS decode: "
                          f"{n_cu} linears dequant-cached")
                # Batch-graph warmup: _ensure_static_batch captures one
                # decode graph PER (B, KV-bucket) lazily — the first
                # real basin batch used to pay seconds of capture mid-
                # run. Building the engine already happens on the
                # prewarm thread (overlaps seed eval), so capture the
                # common batch shapes here too: first proposal batch
                # replays instantly instead of stalling on capture.
                try:
                    mdl = getattr(self._eng, "model", None)
                    ensure = getattr(mdl, "_ensure_static_batch", None)
                    if callable(ensure):
                        # Prompt+gen fits kv2048 for typical prompts
                        # (~1.3k + 640 gen) → B=16 batch cache ~1.6GB,
                        # half of the kv4096 bucket a 2048-plen forces.
                        # Longer prompts still capture lazily at gen
                        # time with their true length.
                        plen, need = 1280, 1920
                        # Only the dominant shape: every capture
                        # pre-allocates its own B x need_len KV cache
                        # (~49KB/tok/row on LFM3) — warming {16,4} cost
                        # ~3.2GB. Rare batch sizes still capture lazily.
                        ensure(int(self.cfg.basin_batch), plen, need)
                        if self.cfg.verbose:
                            print("  [cevolve] batch decode graphs "
                                  "warm-captured", flush=True)
                except Exception:
                    pass          # capture stays lazy on failure
            except Exception as e:
                eng_err = e
                self._eng = None
                # A dir carrying forge_convert.json is Forge-format —
                # transformers can't load its renamed keys (blocks.N.attn.*
                # vs model.layers.N.*), so the HF fallback below is a
                # guaranteed second failure that hides the real error.
                if conv is not None and conv.exists():
                    raise RuntimeError(
                        f"ForgeEngine failed on ported checkpoint "
                        f"{ckpt_dir}: {eng_err}") from eng_err
        if self._eng is not None:
            return ("forge", self._eng)
        if self._hf is None:
            from forge.exp.harness import load_hf
            self._hf = load_hf(self._model_id,
                               device=self.cfg.proposer_device)
        return ("hf", self._hf)

    @property
    def _lora_root(self) -> str | None:
        """Tagged-LoRA registry dir. cfg.lora_root wins; absent that, the
        evolve-trained LFM3 registry (forge/evolution/loras_lfm3) is the
        default so hone-round adapters hot-swap without a flag — the
        canary quarantine makes a bad adapter unable to attach."""
        if self.cfg.lora_root:
            return self.cfg.lora_root
        d = ROOT / "forge" / "evolution" / "loras_lfm3"
        return str(d) if (d / "manifest.json").exists() else None

    def _route(self, eng, text: str | None = None) -> str | None:
        """Adapter tag for this request. Priority: explicit lora_tag ->
        decision-model router (model's own prompt embedding -> aux MLP)
        -> target stem (registry falls back to "default")."""
        if not self._lora_root:
            return None
        if self.cfg.lora_tag:
            return self.cfg.lora_tag
        if text is not None:
            try:
                if self._router is None:
                    from forge.evolution.lora_router import LoRARouter
                    self._router = LoRARouter(self._lora_root)
                if self._router.exists():
                    tag = self._router.route(eng.embed(text[:4000]))
                    if tag:
                        return tag
            except Exception:
                pass
        return Path(self.cfg.target).stem

    def _lora(self, tag: str | None = None):
        """Resolve a tag to a registry entry (fallback chain inside
        resolve). Cached per tag."""
        if not self._lora_root:
            return None
        if self._lora_cached is None:
            self._lora_cached = {}
        key = tag or Path(self.cfg.target).stem
        if key not in self._lora_cached:
            from forge.evolution.lora_registry import LoRARegistry
            reg = LoRARegistry(self._lora_root)
            ent = reg.resolve(key)
            self._lora_cached[key] = (ent, reg.resolve_path(ent)) \
                if ent is not None else None
        return self._lora_cached[key]

    def _ensure_lora(self, eng, text: str | None = None) -> None:
        """Hot-swap the tagged adapter on the shared engine — idempotent,
        called while holding the engine lock (swap+generate is atomic)."""
        hit = self._lora(self._route(eng, text))
        if hit is None or not hasattr(eng, "_load_lora_impl"):
            return
        ent, apath = hit
        if apath is None or not Path(apath).exists():
            # resolve_path returns root/file unconditionally — a stale
            # registry entry used to reach _load_lora_impl and raise
            # FileNotFoundError inside the engine lock, killing every
            # generation. Return unmarked so a freshly-trained adapter
            # is picked up on the next call.
            return
        if not _lora_base_ok(ent,
                             getattr(eng, "checkpoint_path", "") or ""):
            self._last_adapter = None
            if getattr(eng, "_evolve_lora_badbase", None) != ent.tag:
                eng._evolve_lora_badbase = ent.tag
                print(f"[lora] adapter '{ent.tag}' base={ent.base!r} != "
                      f"engine ckpt={eng.checkpoint_path!r} — running "
                      "unadapted (wrong-base modules would crash the "
                      "forward)", flush=True)
            return
        self._last_adapter = ent.tag
        if getattr(eng, "_evolve_lora_tag", None) == ent.tag:
            return
        n_loaded = eng._load_lora_impl(str(apath), ent.rank, ent.alpha,
                                       ent.target_modules)
        if n_loaded == 0:
            # 0 tensors = adapter key names don't match the deployed
            # model's parameters (e.g. trained on HF model.layers.* keys,
            # loaded onto a Forge blocks.* model). Zero-init lora_B makes
            # the attach a silent no-op — the "RSI hone" would be a
            # placebo. Retrain via finetune_proposer --mode lora, which
            # trains on the deployed checkpoint format.
            print(f"[lora] WARNING: adapter '{ent.tag}' loaded 0 tensors "
                  f"on {type(eng.model).__name__} — key mismatch, running "
                  f"unadapted. Retrain it with finetune_proposer.py "
                  f"--mode lora (Forge checkpoint path).", flush=True)
            # Do NOT mark _evolve_lora_tag — the earlier guard would
            # then short-circuit every later call, skipping the warning
            # AND the retry (e.g. after a new adapter file lands). Fall
            # through with the tag unset so each call re-attempts.
            return
        eng._evolve_lora_tag = ent.tag
        if not self._lora_canary(eng):
            try:
                eng.unload_lora()
            except Exception:
                pass
            eng._evolve_lora_tag = None
            self._lora_quarantine(ent)

    def _lora_canary(self, eng) -> bool:
        """Adapter sanity probe: a checkpoint can load cleanly yet corrupt
        the proposer — the 17-row scale-2.0 'default' adapter drove
        parseable output to 0% (token soup) vs 14% unadapted. At least
        one of two greedy probes must yield compileable Python, else the
        adapter is quarantined."""
        probes = (
            "def add(a, b):\n    ",
            "import torch\n\n\ndef normalize(v):\n    ",
        )
        for p in probes:
            try:
                out = eng.generate(p, max_new_tokens=48, temperature=0.0,
                                   top_p=1.0, top_k=0,
                                   repetition_penalty=1.0)
            except Exception:
                continue
            body = (p + out.strip()).rsplit("\n", 1)[0]
            try:
                compile(body, "<lora_canary>", "exec")
                return True
            except SyntaxError:
                continue
        return False

    def _lora_quarantine(self, ent) -> None:
        """Rename the failed tag in the manifest so resolve() can never
        select it again; the adapter file is kept for forensics."""
        try:
            import json
            man = Path(self._lora_root) / "manifest.json"
            m = json.loads(man.read_text(encoding="utf-8"))
            if ent.tag in m.get("tags", {}):
                m["tags"][f"quarantined_{ent.tag}"] = \
                    m["tags"].pop(ent.tag)
                tmp = man.with_name(man.name + ".tmp")
                tmp.write_text(json.dumps(m, indent=2),
                               encoding="utf-8")
                os.replace(tmp, man)
            if self._lora_cached is not None:
                self._lora_cached.clear()
        except Exception as e:
            print(f"[lora] quarantine write failed: {e}", flush=True)
        print(f"[lora] CANARY FAIL: adapter '{ent.tag}' produced "
              "non-compiling output on both probes — unloaded and "
              "quarantined", flush=True)

    _HINTS = (
        "vectorize with pure torch tensor ops, no python loops",
        "precompute and cache repeated work (LUTs, masks, cumsums)",
        "fuse sequential passes into one traversal",
        "use bit-level tricks / integer arithmetic where legal",
        "use torch.sort/topk/scatter/einsum primitives cleverly",
        "restructure for early-exit / short-circuit on common cases",
        "eliminate temporaries — in-place and fused kernels",
        "reformulate the math (log-space, deferred normalize, approx)",
        "batch dimensions differently for better parallelism",
        "hybrid: keep exact semantics but reorder expensive steps",
    )

    # ── hint bandit (meta-prompt evolution-lite) ─────────────────────
    def _hint_state(self) -> list[list[float]]:
        """Beta arms over _HINTS — lazy init (test stubs build the
        proposer via __new__), seeded from mutator_op_stats when the db
        is wired so the policy resumes across runs/targets like the
        Mutator's op bandit."""
        st = getattr(self, "_hint_stats", None)
        if st is None:
            st = self._hint_stats = [[1.0, 1.0] for _ in self._HINTS]
            self._load_hint_stats()
        return st

    def _load_hint_stats(self) -> None:
        db = getattr(self, "db", None)
        if db is None:
            return
        try:
            rows = db.conn.execute(
                "SELECT op, alpha, beta FROM mutator_op_stats "
                "WHERE tag_sig=? AND op LIKE 'hint:%'",
                (getattr(self, "tag_sig", ""),)).fetchall()
            for op, a, b in rows:
                try:
                    i = int(str(op).split(":", 1)[1])
                except (IndexError, ValueError):
                    continue
                if 0 <= i < len(self._hint_stats):
                    self._hint_stats[i] = [max(1.0, a), max(1.0, b)]
        except Exception:
            pass

    def _pick_hint(self) -> int:
        """Thompson-sampled hint index. Cold start (all Beta(1,1)) ==
        the old uniform rotation; verified promotion credit sharpens it."""
        if not getattr(self.cfg, "hint_bandit", True):
            import random as _r
            return _r.randrange(len(self._HINTS))
        if getattr(self, "_hint_rng", None) is None:
            self._hint_rng = np.random.default_rng(
                int(getattr(self.cfg, "seed", 0)) + 4127)
        th = [self._hint_rng.beta(a, b) for a, b in self._hint_state()]
        return int(np.argmax(th))

    def _note_hint(self, prompt: str, hi: int | None) -> None:
        """Record which hint generated this prompt — note() later maps
        the outcome back through it (prompt-text keyed, FIFO-capped)."""
        if hi is None:
            return
        hbp = getattr(self, "_hint_by_prompt", None)
        if hbp is None:
            hbp = self._hint_by_prompt = {}
        hbp[prompt] = hi
        if len(hbp) > 512:
            hbp.pop(next(iter(hbp)))

    def note(self, code: str, credit: float, delta: float,
             prompt: str | None = None) -> None:
        """Outcome feedback for the hint bandit — same credit scale as
        Mutator.note (1.0 verified promotion ... 0.0 gate reject). The
        hint that conditioned the proposal's generation gets the
        credit; no attribution (mutator-made code, missing prompt)
        degrades to a graceful no-op."""
        if prompt is None:
            return
        hbp = getattr(self, "_hint_by_prompt", None)
        hi = hbp.pop(prompt, None) if hbp else None
        if hi is None:
            return
        stats = self._hint_state()
        a, b = stats[hi]
        a += credit
        b += (1.0 - credit)
        stats[hi] = [a, b]
        db = getattr(self, "db", None)
        if db is not None:
            try:
                db.conn.execute(
                    "INSERT INTO mutator_op_stats(op, tag_sig, alpha,"
                    " beta, n, delta_mean) VALUES(?,?,?,?,1,?) "
                    "ON CONFLICT(op, tag_sig) DO UPDATE SET "
                    "alpha=alpha+?, beta=beta+?, n=n+1, "
                    "delta_mean=delta_mean+(?-delta_mean)/(n+1)",
                    (f"hint:{hi}", getattr(self, "tag_sig", ""), a, b,
                     delta, credit, 1.0 - credit, delta))
                db.conn.commit()
            except Exception:
                pass

    def _wrap(self, prompt: str) -> str:
        """ChatML for chat-trained models on the raw engine path —
        generate/generate_batch take literal text (no template).
        LFM3-family checkpoints get the OFFICIAL template shape from
        chat_template.jinja: startoftext BOS, im_start/im_end turns, and
        the empty think-prefill the enable_thinking=False path renders
        (proposer_think=True opens the think tag instead)."""
        bos = "\x3c|startoftext|>" if self._lfm3 else ""
        if self._lfm3:
            think = "\x3cthink>" if self.cfg.proposer_think else \
                "\x3cthink>\x3c/think>"
        else:
            think = ""
        if self.primed:
            # ChatML-preserving prime: instruction goes in the user turn,
            # the ```python block + partial signature moves into the
            # assistant turn so the completion still continues the code.
            # Raw primed prompts (no template) made this chat-tuned ckpt
            # emit token soup — the dominant proposal killer.
            i = prompt.rfind("```python")
            if i >= 0:
                # \x3c = '<' — keeps the ChatML literals consistent
                # with the unprimed wrap below.
                return (bos + "\x3c|im_start|>user\n" + prompt[:i]
                        + "\x3c|im_end|>\n\x3c|im_start|>assistant\n"
                        + think + prompt[i:])
            return prompt
        return (bos + "\x3c|im_start|>user\n" + prompt + "\x3c|im_end|>\n"
                "\x3c|im_start|>assistant\n" + think
                # LFM3 Claude-Code answers agentically when the turn is
                # open: commentary + tool XML + partial fences (~8%
                # gated on unprimed targets). Opening the fence for it
                # makes the response continue inside a code block.
                + ("```python\n" if self._lfm3
                   and not self.cfg.proposer_think else ""))

    # (ChatML/LFM3 turn markers are built with \x3c escapes above — the
    # angle-bracket-pipe sequence corrupts scripted edits in this file.)

    def _with_hint(self, prompt: str, hint: str) -> str:
        """Insert the diversity hint BEFORE the primed ```python block —
        appending it after the signature line makes the model continue
        the hint sentence instead of the function body."""
        i = prompt.rfind("```python")
        if self.primed and i >= 0:
            return prompt[:i] + f"{hint}\n" + prompt[i:]
        return prompt + f"\n{hint}"

    # ── web-research tool calls ──────────────────────────────────────
    # SEARCH protocol (cfg.research): the model may answer a proposal
    # prompt with tool lines INSTEAD of code to look things up first.
    #   SEARCH: <query>  — web search (free backends -> tavily -> exa)
    #   FETCH: <url>     — full page text via research.jina_fetch /
    #                      trafilatura_fetch, truncated ~1.5KB at inject
    #   INSIGHT: <topic> — FindingsDB.code_insights rows matched by
    #                      tag/topic overlap (local, no budget spent)
    # _research_followup intercepts those lines BEFORE code extraction,
    # serves them (cached + budgeted), injects a compact results block,
    # and regenerates ONCE — a single extra round-trip per proposal, max.
    # Costs are bounded three ways: <=research_max_queries queries per
    # proposal, one regen per proposal, and a process-wide uncached-
    # search budget (research_budget, shared across parallel proposer
    # instances via class state — ~daily API quota). Results are cached
    # per query — in-process dict plus an optional append-only jsonl
    # (cfg.research_cache; default <lora_root>/research_cache.jsonl) so
    # repeat queries across runs are free. Proposal-side ONLY — eval
    # workers never see this path. Keys are never printed or persisted
    # (the cache stores query+hits only).

    _SEARCH_RE = re.compile(r"(?m)^[ \t]*SEARCH:[ \t]*\S[^\n]*\n?")
    _FETCH_RE = re.compile(r"(?m)^[ \t]*FETCH:[ \t]*\S[^\n]*\n?")
    _INSIGHT_RE = re.compile(r"(?m)^[ \t]*INSIGHT:[ \t]*\S[^\n]*\n?")
    _SEARCH_CACHE: dict = {}              # query -> hits (all instances)
    _SEARCH_CACHE_LOCK = threading.Lock()
    _SEARCH_CACHE_SEEN: set = set()       # jsonl paths already loaded
    _SEARCH_SPENT = 0                     # uncached API calls, process

    def _search_cache_path(self) -> str | None:
        p = self.cfg.research_cache
        if p is None and self._lora_root:
            p = str(Path(self._lora_root) / "research_cache.jsonl")
        return p

    def _load_search_cache(self) -> None:
        """Merge the persisted jsonl cache once per path (append-only —
        last row wins on collisions, setdefault keeps the first)."""
        path = self._search_cache_path()
        if not path:
            return
        with LLMProposer._SEARCH_CACHE_LOCK:
            if path in LLMProposer._SEARCH_CACHE_SEEN:
                return
            LLMProposer._SEARCH_CACHE_SEEN.add(path)
            try:
                for line in Path(path).read_text(
                        encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                        LLMProposer._SEARCH_CACHE.setdefault(
                            r["q"], r.get("hits", []))
                    except Exception:
                        continue
            except Exception:
                pass

    def _lookup(self, key: str, miss_fn) -> list[dict] | None:
        """Shared cache+budget wrapper for the web tools (SEARCH/FETCH
        both consume _SEARCH_SPENT). None = budget exhausted; the
        jsonl cache stores the key verbatim so 'fetch:<url>' rows
        reload for free like any query."""
        self._load_search_cache()
        with LLMProposer._SEARCH_CACHE_LOCK:
            if key in LLMProposer._SEARCH_CACHE:
                return LLMProposer._SEARCH_CACHE[key]
            if LLMProposer._SEARCH_SPENT >= self.cfg.research_budget:
                return None
            LLMProposer._SEARCH_SPENT += 1
        hits = miss_fn()
        with LLMProposer._SEARCH_CACHE_LOCK:
            LLMProposer._SEARCH_CACHE[key] = hits
            path = self._search_cache_path()
            if path:
                try:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"q": key, "hits": hits})
                                + "\n")
                except Exception:
                    pass
        return hits

    def _search(self, query: str) -> list[dict] | None:
        """Cached, budgeted web search. None = budget exhausted (treat
        as 'no lookup available'); [] = searched, zero usable hits."""
        def miss() -> list[dict]:
            from . import research as _res
            # Keyless backends first (ddgs/arxiv/S2) — the SEARCH tool
            # works with zero keys; paid APIs are the fallback.
            hits = _res.free_search(query, self.cfg.research_hits)
            if not any(h.get("url") for h in hits):
                hits = _res.tavily_search(query, self.cfg.research_hits)
            if not any(h.get("url") for h in hits):
                hits = _res.exa_search(query, self.cfg.research_hits)
            return [h for h in hits if h.get("url") or h.get("snippet")]
        return self._lookup(query, miss)

    def _fetch(self, url: str) -> list[dict] | None:
        """Full-page fetch for the FETCH: tool — jina reader first,
        trafilatura fallback. Same cache+budget as SEARCH; page text
        rides in the single hit's snippet field (truncated at inject)."""
        def miss() -> list[dict]:
            from . import research as _res
            text = _res.jina_fetch(url) or _res.trafilatura_fetch(url)
            # Cap the stored page so the jsonl cache stays lean — the
            # injector trims to ~1.5KB per proposal anyway.
            return [{"title": url, "url": url, "snippet": text[:4096]}]
        return self._lookup(f"fetch:{url}", miss)

    def _insight(self, query: str) -> list[dict]:
        """FindingsDB.code_insights retrieval for the INSIGHT: tool —
        past verified wins matched by tag/topic word overlap. LOCAL:
        shares the per-proposal query cap but never spends the web
        budget. Read-only sqlite open (mode=ro) so a missing db or a
        locked WAL writer never creates files or blocks."""
        rows = []
        try:
            conn = sqlite3.connect(
                f"file:{self.cfg.db_path}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT target, tags, summary, delta FROM code_insights"
                    " ORDER BY ts DESC LIMIT 500").fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        terms = {t for t in re.split(r"[^a-z0-9_]+", query.lower())
                 if len(t) > 2}
        scored = []
        for target, tags, summary, delta in rows:
            hay = f"{target} {tags or ''} {summary or ''}".lower()
            ov = sum(1 for t in terms if t in hay)
            if ov:
                scored.append((ov, target, summary or "", delta or 0.0))
        scored.sort(key=lambda x: -x[0])
        return [{"title": f"[{t}]", "url": "",
                 "snippet": f"{s} (delta {d:+.4g})"}
                for _, t, s, d in scored[: self.cfg.research_hits]]

    _TOOL_RES = (("SEARCH", _SEARCH_RE), ("FETCH", _FETCH_RE),
                 ("INSIGHT", _INSIGHT_RE))

    def _tool_calls(self, raw: str) -> list[tuple[str, str]]:
        """All tool lines (SEARCH/FETCH/INSIGHT) in emission order,
        deduped, capped at research_max_queries total — the shared cap."""
        if not raw:
            return []
        marks = []
        for kind, rx in self._TOOL_RES:
            for m in rx.finditer(raw):
                marks.append((m.start(), kind, m.group(0)))
        marks.sort()
        calls: list[tuple[str, str]] = []
        seen: set = set()
        for _, kind, line in marks:
            arg = line.split(":", 1)[1].strip().strip('`"\'')[:400]
            key = (kind, arg)
            if arg and key not in seen:
                seen.add(key)
                calls.append((kind, arg))
            if len(calls) >= self.cfg.research_max_queries:
                break
        return calls

    def _research_block(self, queries: list[str]) -> str | None:
        """Compact comment block of hits (title | url / 1-line snippet).
        None when the budget ran out before ANY query was served."""
        return self._tool_block([("SEARCH", q) for q in queries])

    def _tool_block(self, calls: list[tuple[str, str]]) -> str | None:
        """Compact comment block serving SEARCH:/FETCH:/INSIGHT: calls.
        SEARCH + FETCH share the web budget (skipped when exhausted);
        INSIGHT is local and always answerable. None when nothing was
        served at all."""
        lines = ["# Tool results (auto-fetched; now write the code):"]
        served = 0
        for kind, arg in calls:
            if kind == "INSIGHT":
                hits = self._insight(arg)
                served += 1
                lines.append(f"# INSIGHT: {arg}")
                if not hits:
                    lines.append("#   (no matching insights)")
                for h in hits[: self.cfg.research_hits]:
                    lines.append(f"# - {h['title'][:60]} {h['snippet'][:180]}")
                continue
            hits = (self._search(arg) if kind == "SEARCH"
                    else self._fetch(arg))
            if hits is None:
                continue                    # budget exhausted
            served += 1
            lines.append(f"# {kind}: {arg}")
            if kind == "FETCH":
                text = " ".join(
                    (hits[0].get("snippet") if hits else "")
                    .split())[:1500]
                if not text:
                    lines.append("#   (fetch failed or empty page)")
                for i in range(0, len(text), 120):
                    lines.append(f"#   {text[i:i + 120]}")
                continue
            if not hits:
                lines.append("#   (no results)")
            for h in hits[: self.cfg.research_hits]:
                title = (h.get("title") or h.get("url") or "?").strip()
                url = (h.get("url") or "").strip()
                snip = " ".join(
                    (h.get("snippet") or "").split())[:160]
                lines.append(f"# - {title[:120]} | {url}")
                if snip:
                    lines.append(f"#   {snip}")
        return "\n".join(lines) if served else None

    def _inject_research(self, prompt: str, block: str) -> str:
        """Insert before the primed ```python block (same rule as
        _with_hint) so completion models keep the signature prime;
        plain append otherwise."""
        i = prompt.rfind("```python")
        if self.primed and i >= 0:
            return prompt[:i] + block + "\n" + prompt[i:]
        return prompt + "\n" + block

    def _digest_text(self) -> str:
        """The round's research digest trimmed to ~1.6KB of comments —
        context for which techniques are worth trying."""
        txt = getattr(self, "_digest_txt", None)
        if txt is None:
            txt = ""
            p = self.cfg.research_digest
            if p:
                try:
                    body = Path(p).read_text(encoding="utf-8",
                                             errors="ignore")
                    keep = [ln.strip() for ln in body.splitlines()
                            if ln.strip()][:32]
                    txt = ("# Fresh web-research digest "
                           "(techniques worth trying):\n"
                           + "\n".join("# " + ln for ln in keep))[:1600]
                except Exception:
                    txt = ""
            self._digest_txt = txt
        return txt

    def _with_tools(self, prompt: str) -> str:
        """Tell the proposer about the SEARCH: tool + inject the round's
        research digest. No-op unless cfg.research — and idempotent:
        propose_batch's serial fallback re-enters __call__ on prompts
        that already carry the block, and a doubled contract is pure
        noise."""
        if not self.cfg.research or "#   SEARCH: <query>" in prompt:
            return prompt
        bits = []
        d = self._digest_text()
        if d:
            bits.append(d)
        bits.append(
            f"# Tool: need to look something up first? Emit tool lines "
            f"(max {self.cfg.research_max_queries} total, no code yet) — "
            "results come back automatically and you are asked again.\n"
            "#   SEARCH: <query>   — web search (technique hunting)\n"
            "#   FETCH: <url>      — pull a page's full text (~1.5KB)\n"
            "#   INSIGHT: <topic>  — verified wins from past evolve runs\n"
            "# Example - to find the fastest approach:\n"
            "#   SEARCH: fast vectorized top-p sampling pytorch\n"
            "#   INSIGHT: top-p sampling\n"
            "# Hits come back as comments; then write the code. Use them "
            "when unsure which technique is fastest.")
        return self._inject_research(prompt, "\n".join(bits))

    def _research_followup(self, prompts: list[str], raws: list[str],
                           regen) -> list[str]:
        """Tool-line intercept: serve queued SEARCH/FETCH/INSIGHT calls
        (cached + budgeted), inject the results block, and regen ONCE
        for the rows that asked. Tool lines are stripped from every raw
        so code extraction never mistakes one for the answer. regen is
        a caller-supplied fn(prompts2) -> raws2 reusing the caller's
        generation settings."""
        if not self.cfg.research:
            return raws

        def _strip(s: str) -> str:
            for _k, rx in self._TOOL_RES:
                s = rx.sub("", s)
            return s

        out = []
        todo = []                                # (row, aug prompt)
        for i, raw in enumerate(raws):
            raw = raw or ""
            calls = self._tool_calls(raw)
            out.append(_strip(raw))
            if calls and i < len(prompts):
                block = self._tool_block(calls)
                if block:
                    todo.append((i, self._inject_research(
                        prompts[i], block)))
        if todo:
            try:
                new = regen([p for _, p in todo])
                for (i, _), r in zip(todo, new):
                    if r:
                        out[i] = _strip(r)
            except Exception:
                pass
        return out

    def __call__(self, prompt: str) -> str:
        self._last_hint = None
        p2 = self._with_tools(prompt)
        raw = self._gen_raw(p2)
        out = self._research_followup(
            [p2], [raw],
            lambda ps: [self._gen_raw(p) for p in ps])[0]
        self._note_hint(prompt, self._last_hint)
        return out

    def _gen_raw(self, prompt: str) -> str:
        """One generation for a fully-built prompt — engine dispatch,
        LoRA routing, diversity hint, template wrap. NO research
        followup (that's __call__/propose_batch's job)."""
        kind, eng, elock = self._engine()
        if kind == "forge":
            self._last_hint = self._pick_hint()
            hint = f"# Try this approach: {self._HINTS[self._last_hint]}"
            with elock:
                self._ensure_lora(eng, prompt)
                eos_id = getattr(eng.tokenizer, 'eos_token_id', None)
                im_end = (eng.tokenizer.decode([eos_id])
                          if eos_id is not None else "<|im_end|>")
                out = eng.generate(
                    self._wrap(self._with_hint(prompt, hint)),
                    max_new_tokens=self.cfg.max_gen_tokens,
                    temperature=self.cfg.temperature, top_p=0.95,
                    top_k=80, finish_sentence=False,
                    stop=[im_end])
            return out if isinstance(out, str) else out.get("text", "")
        model, tok = eng
        import torch   # lazy — module stays importable without torch
        if self.primed:
            # completion path — local SFT ckpts have no chat template
            ids = tok(prompt, return_tensors="pt").to(model.device)
            n_in = ids.input_ids.shape[1]
            with torch.no_grad(), elock:
                out = model.generate(
                    ids.input_ids,
                    max_new_tokens=self.cfg.max_gen_tokens,
                    do_sample=True, temperature=self.cfg.temperature,
                    top_p=0.95, top_k=80, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][n_in:], skip_special_tokens=True)
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad(), elock:
            out = model.generate(
                ids, max_new_tokens=self.cfg.max_gen_tokens, do_sample=True,
                temperature=self.cfg.temperature, top_p=0.95, top_k=80,
                pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def propose_batch(self, prompts: list[str]) -> list[str]:
        """K prompts in one GPU pass — ForgeEngine generate_batch (3-5x
        throughput) on the engine path; left-padded HF generate for local
        checkpoints. Falls back to serial calls on failure."""
        kind, eng, elock = self._engine()
        if kind == "forge":
            n = len(prompts)
            import random as _rnd
            eos_id = getattr(eng.tokenizer, 'eos_token_id', None)
            im_end = (eng.tokenizer.decode([eos_id])
                      if eos_id is not None else "<|im_end|>")
            # Anti-mode-collapse: temps 0.6→1.3, rep-penalty jitter, fresh
            # seeds, AND a per-prompt strategy hint — near-identical
            # prompts mode-collapse into dup rows without conditioning
            # each call into a different solution basin.
            temps = [_rnd.uniform(0.7, 1.25) for _ in range(n)]
            reps = [1.0 + 0.25 * _rnd.random() for _ in range(n)]
            seeds = [_rnd.randrange(1 << 60) for _ in range(n)]
            tools = [self._with_tools(p) for p in prompts]
            hint_idxs = [self._pick_hint() for _ in tools]
            hinted = [self._with_hint(
                          p, f"# Try this approach: "
                             f"{self._HINTS[hint_idxs[i]]}")
                      for i, p in enumerate(tools)]
            for i, p in enumerate(prompts):
                self._note_hint(p, hint_idxs[i])
            try:
                with elock:
                    # per-prompt routing isn't possible mid-batch — the
                    # batch shares one adapter; route its first prompt
                    # (all prompts in a batch share the target anyway).
                    self._ensure_lora(eng, prompts[0] if prompts else None)
                    raws = eng.generate_batch(
                        [self._wrap(p) for p in hinted],
                        max_new_tokens=self.cfg.max_gen_tokens,
                        temperature=self.cfg.temperature, top_p=0.95,
                        top_k=80, temperatures=temps,
                        repetition_penalties=reps, seeds=seeds,
                        stops=[[im_end]] * n, skip_special_tokens=True)
                    # SEARCH: rows get ONE batched regen — the extra
                    # round-trip is shared across all lookups.
                    return self._research_followup(
                        hinted, raws,
                        lambda ps: eng.generate_batch(
                            [self._wrap(p) for p in ps],
                            max_new_tokens=self.cfg.max_gen_tokens,
                            temperature=self.cfg.temperature, top_p=0.95,
                            top_k=80,
                            # Regen rows get the same anti-collapse
                            # jitter as the first pass — one shared
                            # temp/seed made all regens drift to the
                            # same basin as the row they replaced.
                            temperatures=[_rnd.uniform(0.7, 1.25)
                                          for _ in ps],
                            repetition_penalties=[
                                1.0 + 0.25 * _rnd.random()
                                for _ in ps],
                            seeds=[_rnd.randrange(1 << 60)
                                   for _ in ps],
                            stops=[[im_end]] * len(ps),
                            skip_special_tokens=True))
            except Exception:
                pass
            return [self(p) for p in prompts]
        model, tok = eng
        import torch
        # Chat models wrap each prompt; completion models use raw text.
        tools = [self._with_tools(p) for p in prompts]
        texts = []
        for p in tools:
            if self.primed:
                texts.append(p)
            else:
                texts.append(tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True, tokenize=False,
                    # LFM3's jinja renders an OPEN think prefill by
                    # default — proposer_think=False yields the empty
                    # think pair = direct answer. Templates without the
                    # knob (Qwen etc.) ignore the extra context var.
                    enable_thinking=self.cfg.proposer_think))
        old_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        try:
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    enc.input_ids, attention_mask=enc.attention_mask,
                    max_new_tokens=self.cfg.max_gen_tokens,
                    do_sample=True, temperature=self.cfg.temperature,
                    top_p=0.95, top_k=80, pad_token_id=tok.eos_token_id)
            start = enc.input_ids.shape[1]   # left-padded: gen starts here
            raws = [tok.decode(o[start:], skip_special_tokens=True)
                    for o in out]
            return self._research_followup(
                tools, raws,
                lambda ps: [self._gen_raw(p) for p in ps])
        finally:
            tok.padding_side = old_side


def _extract_code(raw: str, prime: str = "",
                  required: list[str] | None = None) -> str:
    """Pull the candidate program out of an LLM completion.

    Handles: closed ```python fences (any lang tag), UNclosed fences
    (truncated gen), primed continuations (prompt ended mid-signature,
    raw continues the def line), <think> reasoning blocks (the answer
    is the code AFTER </think>), leaked ChatML markers, prose preambles,
    and bare code. `required` (target's public names) makes blocks that
    define every required name win over earlier def-fragments. Salvage
    trims leading prose + trailing lines until a prefix compiles, since
    base models wrap code in chatter.
    """
    # Work text: closed <think>...</think> regions are reasoning, not the
    # answer — remove them BEFORE candidate selection or their fenced
    # fragments win over the real program. Leaked special-token markers
    # (ChatML im_start/im_end) are stripped the same way.
    work = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    work = re.sub(r"[<]\|[^|\n]*[|]>", "", work)
    # LFM3/Claude-Code agentic leaks: the finetune emits tool_call XML
    # (<function=, <parameter=, </parameter>) and stray think markers
    # mid-stream — they parse as text and poison every fenced block.
    work = re.sub(r"<tool_call>.*?</tool_call>", "", work, flags=re.S)
    # Claude-Code edit format: the finetune rewrites code inside
    # <parameter=new_string>...</parameter> while <parameter=old_string>
    # holds the PRE-edit text — keep the new, drop the old before
    # generic tag-stripping so the stale copy can't win selection.
    work = re.sub(r"<parameter=old_string>.*?</parameter>", "",
                  work, flags=re.S)
    work = re.sub(r"</?(?:function|parameter|tool_call)"
                  r"[=\s>][^\n>]*>?", "", work)
    work = re.sub(r"</?think>", "", work)

    # Build candidate programs, then pick the first usable one:
    #   A. primed continuation — body before the first closing fence,
    #      re-attached to the prime (signature + header). Only when raw
    #      resumes indented body text (a top-level restart means the
    #      model abandoned the prime).
    #   B. fenced blocks that actually define code — when the model
    #      closes the primed fence and emits restart block(s), the real
    #      program is in a LATER block (or bare text after the last
    #      fence), not the first `import torch` line.
    #   C. bare-code tail after the last fence.
    #   D. raw as-is.
    def _has_def(s: str) -> bool:
        return bool(re.search(r"(?m)^\s*(def |class |@)", s))

    cands: list[str] = []
    if prime and not re.match(
            r"^(import |from |def |class |@|\"\"\"|'''|```)",
            work.lstrip()):
        body = work.split("```", 1)[0]
        if body.strip():
            if prime.rstrip().endswith(":") and not body.startswith("\n"):
                body = "\n" + body
            cands.append(prime + body)
    blocks = re.findall(r"```[^\n`]*\n(.*?)```", work, re.S)
    blocks = [re.sub(r"^(?:python|py)\s*\n", "", b) for b in blocks]
    cands += [b for b in blocks if _has_def(b)]
    cands += [b for b in blocks if not _has_def(b)]
    if "```" in work:
        tail = re.sub(r"^\s*(?:python|py)\s*\n", "",
                      work.rsplit("```", 1)[1])
        if tail.strip():
            cands.append(tail)
    cands.append(work if work.strip() else raw)

    # Selection: a block defining EVERY required name is the complete
    # program — beat fragments that happen to come first. Fall back to
    # the first def-bearing candidate, then raw.
    def _has_all_required(s: str) -> bool:
        return all(
            re.search(rf"(?m)(?:^\s*(?:def|class)\s+{re.escape(n)}\b"
                      rf"|^{re.escape(n)}\s*=)", s)
            for n in required)

    code = ""
    if required:
        for c in cands:
            if _has_def(c) and _has_all_required(c):
                code = c
                break
    if not code:
        for c in cands:
            if _has_def(c):
                code = c
                break
        else:
            code = cands[0] if cands else raw
    # Cut trailing comment sections the base model appends ("# Improve",
    # "# Write", "# Metrics" headers) — real code rarely resumes after them.
    lines = code.split("\n")
    cut = len(lines)
    for i, ln in enumerate(lines):
        if i > 0 and re.match(r"^#\s*(Improve|Write|Proven|Metrics|Current|"
                              r"Goal|Rules|Task|Insight)", ln):
            cut = i
            break
    lines = lines[:cut]
    # Leading-prose trim: a chatter preamble ("Here is the improved
    # version:") kills every prefix in the trailing-salvage loop below —
    # find the first code-shaped line and drop everything before it.
    # Code starts with def/class/import/from/@/comment/docstring/
    # assignment, not with a prose sentence.
    start = None
    for i, ln in enumerate(lines[:20]):
        if not ln.strip():
            continue
        if re.match(r"\s*(def |class |@|import |from |#|\"\"\"|'''|"
                    r"[A-Za-z_]\w*\s*(?:=|\())", ln):
            start = i
            break
    if start:
        lines = lines[start:]
    code = "\n".join(lines)
    # Uniform-indent fallback: models nesting the whole program under a
    # markdown indent produce code that can't compile at column 0 —
    # dedent recovers it verbatim.
    try:
        compile(code.strip() or "x=0", "<cand>", "exec")
    except SyntaxError:
        import textwrap as _tw
        dd = _tw.dedent(code).strip()
        if dd and dd != code.strip():
            try:
                compile(dd, "<cand>", "exec")
                code = dd
            except SyntaxError:
                pass
    lines = code.split("\n")
    # Salvage: drop trailing lines until the remaining prefix compiles.
    for end in range(len(lines), max(len(lines) - 40, 0), -1):
        cand = "\n".join(lines[:end]).strip()
        if not cand:
            break
        try:
            compile(cand, "<cand>", "exec")
            return cand
        except SyntaxError:
            continue
    return "\n".join(lines).strip()


def _seed_signature(seed: str, name: str) -> str:
    """Full `def name(...)->...:` line(s) from the seed — used to prime a
    completion-style proposer so it continues the body instead of
    re-typing the spec. Class targets prime on the `class Name:` header
    (a `def Name(` prime would produce a function, not the required
    class)."""
    i = seed.find(f"def {name}")
    if i < 0:
        j = seed.find(f"class {name}")
        if j >= 0:
            end = seed.find("\n", j)
            return seed[j:end if end >= 0 else len(seed)].rstrip()
        return f"def {name}("
    j, depth = i, 0
    while j < len(seed):
        c = seed[j]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == ":" and depth == 0:
            return seed[i:j + 1]
        j += 1
    return seed[i:i + 80].split("\n")[0]


def _norm_hash(code: str) -> str:
    """Dedup hash on semantic content (comments/whitespace stripped)."""
    try:
        import ast as _a
        # Canonical AST dump: comment/formatting differences vanish and
        # '#' inside string literals stays data — the old split('#')
        # made `x="#"` and `x=` collide, wrongly deduping real
        # alternatives.
        slim = _a.dump(_a.parse(code))
    except SyntaxError:
        # Unparseable fragment — strip `#` comments only outside string
        # literals (quoted text consumes its '#' before the alt can see
        # it as a comment start).
        slim = re.sub(
            r"'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\""
            r"|(#)[^\n]*",
            lambda m: m.group(0) if m.group(1) is None else "",
            code)
        slim = "\n".join(ln.rstrip() for ln in slim.split("\n"))
        slim = re.sub(r"\n{2,}", "\n", slim).strip()
    return hashlib.sha1(slim.encode()).hexdigest()[:16]


# ── evolution engine ────────────────────────────────────────────────────

class CodeEvolve:
    """Goal-driven program evolution. See module docstring."""

    def __init__(self, cfg: CodeEvolveConfig,
                 propose_fn: Callable[[str], str] | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.db = FindingsDB(cfg.db_path)
        with self.db._lock:
            for stmt in _CODE_DDL:
                self.db.conn.execute(stmt)
            # prompt column (added later): real prompt text per candidate
            # -> exact SFT/DPO training pairs downstream.
            cols = {r[1] for r in self.db.conn.execute(
                "PRAGMA table_info(code_programs)")}
            for col in ("prompt", "adapter"):
                if col not in cols:
                    try:
                        self.db.conn.execute(
                            f"ALTER TABLE code_programs ADD COLUMN {col} TEXT")
                    except sqlite3.OperationalError:
                        pass   # parallel run added it first
            self.db.conn.commit()
        self.run_id = f"cev_{uuid.uuid4().hex[:10]}"
        self.target = Target(Target.resolve(cfg.target))
        self.metric = cfg.metric or self.target.metric
        if self.metric is None:
            raise ValueError(
                f"target {self.target.name!r} defines no METRIC and no "
                "--metric override was given")
        self.goal = cfg.goal or self.target.goal
        self._sign = -1.0 if self.goal == "min" else 1.0
        self._basin_mode = False
        self._mutator = None
        self._basin_llm = None
        if propose_fn is not None:
            self._proposer = propose_fn
        elif cfg.proposer == "llm":
            self._proposer = LLMProposer(cfg, db=self.db,
                                         tag_sig=self._tag_sig())
        elif cfg.proposer == "mut":
            self._proposer = Mutator(seed=cfg.seed, db=self.db,
                                     tag_sig=self._tag_sig())
        elif cfg.proposer == "mix":
            self._proposer = MixProposer(
                Mutator(seed=cfg.seed, db=self.db, tag_sig=self._tag_sig()),
                LLMProposer(cfg, db=self.db, tag_sig=self._tag_sig()),
                llm_every=cfg.llm_every)
        elif cfg.proposer == "basin":
            # LLM emits basin parents on a background thread; the
            # mutator fans each one out deterministically. The LLM is
            # created lazily in run() so a mut-only fallback never pays
            # the model load.
            self._mutator = Mutator(seed=cfg.seed, db=self.db,
                                    tag_sig=self._tag_sig())
            self._proposer = self._mutator   # primed=False, note() proxy
            self._basin_mode = True
        else:
            self._proposer = None
        # basin producer/consumer state
        self._basin_q = None
        self._basin_last_good = 0.0   # last time a basin yielded a variant
        self._family_from_basin = False
        self._basin_t = None
        self._stop_evt = threading.Event()
        self._family = None
        self._family_island = 0
        self._family_prompt = ""
        self._family_ph = None
        self._seen_ops: set[tuple] = set()      # (parent_h, op_sig)
        # eval pool (created lazily in run)
        self._epool = None
        self._inc_cache: tuple | None = None    # (inc_h, scores, eval_n)
        # island -> elite list [(code, score, hash)]
        self.elites: list[list[tuple[str, float, str]]] = \
            [[] for _ in range(cfg.islands)]
        self._arms = [[1.0, 1.0] for _ in range(cfg.islands)]  # Beta arms
        self._seen: set[str] = set()
        self._insights: list[dict] = []
        self._eval_n = 0
        self._dry = 0
        self._hop_plateau = -1   # promotion count at the last basin hop
        self._promos = 0
        self._stages: dict[str, int] = {}
        # Proposal-validity telemetry: raws consumed from a proposer vs
        # raws that produced a gated (evaluable) candidate. The dup/soup
        # rates are the proposer's quality signal — surfaced in the
        # status JSON + run() result so regressions are measurable.
        self._gen_n = 0
        self._gen_ok = 0
        # Approach fingerprints of everything already evaluated — fed
        # back into prompts so the model stops re-emitting families
        # that dedup kills for free (measured: 60% dup rejects).
        self._tried_gists: deque = deque(maxlen=24)
        self._last_status = 0.0
        self.incumbent_code: str | None = None
        self.incumbent_metrics: dict = {}
        self.incumbent_score = -float("inf")
        self.baseline_score = -float("inf")
        self.stop_reason = "eval budget"
        self._prime = ""
        self._repair_q: list[dict] = []      # error-failures awaiting
                                             # the batched repair regen
        # Auto-scale state: workers<=0 -> dynamic submit limit.
        self._auto = cfg.workers <= 0
        self._limit = 1 if self._auto else max(1, cfg.workers)
        self._last_probe = 0.0
        self._cuda_eval = self._detect_cuda_target()
        self._cuda_cap = cfg.max_workers

    # ── scoring helpers ──────────────────────────────────────────────
    def _detect_cuda_target(self) -> bool:
        """Does this target bench on CUDA? eval subprocesses then cost
        ~1.2GB VRAM each, so the autoscaler needs a separate (smaller)
        cap for them.

        AST-scan the whole target file: device strings ("cuda"), cuda()
        calls, and torch.cuda attribute use at ANY level — the old scan
        of _inputs/bench/test bodies missed module-level device picks
        and helper functions. Comments can't false-positive (they aren't
        in the AST); docstrings are Constants, and a lowercase "cuda" in
        prose errs toward the small cap — the safe direction."""
        try:
            tree = ast.parse(
                self.target.path.read_text(encoding="utf-8",
                                           errors="ignore"))
        except Exception:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "cuda":
                return True
            if isinstance(node, ast.Constant) \
                    and isinstance(node.value, str) \
                    and "cuda" in node.value:
                return True
        return False

    def _probe_load(self) -> None:
        """Auto-scale the eval submission limit toward ~85% system load.
        Shrinks fast on memory pressure (evals are the RAM hogs); grows
        one worker at a time. CUDA targets also respect VRAM headroom."""
        now = time.time()
        if not self._auto or now - self._last_probe < 1.5:
            return
        self._last_probe = now
        try:
            import psutil
            cpu = psutil.cpu_percent() / 100.0
            ram = psutil.virtual_memory().percent / 100.0
        except Exception:
            cpu, ram = 0.5, 0.5
        gpu_ok = True
        if self._cuda_eval:
            try:
                import torch
                free, total = torch.cuda.mem_get_info()
                used = 1.0 - free / total
                # ~1.2GB per torch+cuda eval subprocess; recompute cap
                self._cuda_cap = max(2, int((free / 2**30) // 1.5))
                gpu_ok = used < 0.85 and torch.cuda.utilization() < 85
            except Exception:
                self._cuda_cap = 4
        cap = min(self.cfg.max_workers, self._cuda_cap)
        if ram > self.cfg.mem_limit + 0.05 or cpu > 0.92 or not gpu_ok:
            self._limit = max(1, self._limit - 2)
        elif cpu < self.cfg.cpu_target - 0.05 and ram < self.cfg.mem_limit:
            self._limit = min(cap, self._limit + 1)
        self._limit = min(self._limit, cap)

    def _score(self, metrics: dict) -> float:
        v = metrics.get(self.metric)
        if v is None or not np.isfinite(float(v)):
            return -1e9
        return self._sign * float(v)

    def _fits_budgets(self, metrics: dict) -> bool:
        for name, (op, val) in self.target.budgets.items():
            m = metrics.get(name)
            if m is None or not _OPS[op](float(m), float(val)):
                return False
        return True

    # ── subprocess eval ──────────────────────────────────────────────
    def _cheap_spec(self) -> dict | None:
        """Eval-cascade spec for first-pass candidate evals: the
        reduced-rep bench gate skips the full bench when the cheap
        metric lands clearly worse than the incumbent. None when the
        cascade can't apply (verify/seed/resume evals always run the
        full bench — they never pass this to the workers)."""
        cfg = self.cfg
        if not cfg.cheap_bench:
            return None
        try:
            inc = float(self.incumbent_metrics.get(self.metric))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(inc):
            return None
        return {"reps": cfg.cheap_reps, "metric": self.metric,
                "incumbent": inc, "goal": self.goal,
                "margin": cfg.cheap_margin}

    def _eval(self, code: str, workdir: Path, tag: str = "x",
              cheap: dict | None = None) -> dict:
        cand = workdir / f"candidate_{tag}.py"
        cand.write_text(code, encoding="utf-8")
        # Scrub secret env vars out of the eval child's environment —
        # candidate code is mutator/LLM-generated and gets the full
        # interpreter otherwise.
        try:
            from research.persistent_executor import scrub_environ
            env = scrub_environ()
        except Exception:
            env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            # Run by file path, NOT -m: `-m forge.evolution._candidate_eval`
            # imports the heavy package __init__ (torch+all domains) into
            # every eval subprocess. The file itself is stdlib-only.
            p = subprocess.run(
                [sys.executable, "-X", "utf8",
                 str(Path(__file__).resolve().parent / "_candidate_eval.py"),
                 str(self.target.path), str(cand)]
                + ([json.dumps(cheap)] if cheap else []),
                capture_output=True, text=True, timeout=self.cfg.timeout_s,
                cwd=str(ROOT), env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "timeout"}
        # LAST __RESULT__ line wins: a candidate that prints a forged
        # "__RESULT__{...}" marker can only precede the evaluator's real
        # line (emitted after exec returns), never follow it.
        line = next((ln for ln in reversed(p.stdout.splitlines())
                     if ln.startswith("__RESULT__")), None)
        if line is None:
            return {"ok": False, "stage": "spawn",
                    "detail": (p.stderr or p.stdout or "")[-400:]}
        return json.loads(line[len("__RESULT__"):])

    def _eval_sync(self, code: str, workdir: Path, tag: str = "x",
                   timeout_s: float | None = None) -> dict:
        """Blocking single eval — persistent pool when up, else the
        subprocess path. Used for seed/resume/serial runs."""
        if self._epool is not None:
            return self._epool.eval_sync(
                code, timeout_s=timeout_s or self.cfg.timeout_s)
        return self._eval(code, workdir, tag)

    def _verified(self, code: str, cand_score: float,
                  workdir: Path) -> tuple[bool, dict]:
        """Replicate challenger + incumbent evals; challenger must win by a
        noise-aware margin. Challenger reps always fresh; incumbent reps
        are re-benched only every `inc_stale` evals (cached in between —
        the same-load-regime argument still holds inside a window).
        Under the eval pool all reps submit in ONE pass — parallel
        instead of serial subprocess spawns."""
        # Under parallel load both means are noisier — spend an extra
        # rep per side or a lucky quiet window promotes a regression.
        r = (max(self.cfg.reps, 3)
             if (self._auto or self.cfg.workers != 1) else self.cfg.reps)
        inc_h = (_norm_hash(self.incumbent_code)
                 if self.incumbent_code is not None else None)
        need_inc = self.incumbent_code is not None and (
            self._inc_cache is None or self._inc_cache[0] != inc_h
            or self._eval_n - self._inc_cache[2] >= self.cfg.inc_stale)
        codes = [code] * (r - 1)
        if need_inc:
            codes += [self.incumbent_code] * r
        if codes:
            if self._epool is not None:
                results = self._epool.eval_many(
                    codes, timeout_s=self.cfg.timeout_s)
            else:
                results = [self._eval(c, workdir, f"v{self._eval_n}_{k}")
                           for k, c in enumerate(codes)]
        else:
            results = []
        c_scores = [cand_score]
        for res in results[:r - 1]:
            c_scores.append(self._score(res.get("metrics", {}))
                            if res.get("ok") else -1e9)
        if need_inc:
            i_scores = [self._score(res.get("metrics", {}))
                        if res.get("ok") else -1e9
                        for res in results[r - 1:]]
            self._inc_cache = (inc_h, i_scores, self._eval_n)
        elif self._inc_cache is not None and inc_h == self._inc_cache[0]:
            i_scores = list(self._inc_cache[1])
        else:
            i_scores = []
        # Median, not mean: under shared-GPU contention a single load
        # spike inside one eval_many pass skews the mean enough to pass
        # a fake win (observed: quantize_4bit -55%, top_p_probs -11%
        # "promotions" that failed independent re-bench). Median is
        # robust to the drift-and-spike noise floor on this box.
        cm = float(np.median(c_scores))
        im = (float(np.median(i_scores)) if i_scores
              else self.incumbent_score)
        if not np.isfinite(im):
            return np.isfinite(cm), {"challenger_mean": cm,
                                     "incumbent_mean": None, "reps": r,
                                     "margin": 0.0, "delta": 0.0, "z": None}
        pooled = float(np.std(c_scores + i_scores))
        margin = max(self.cfg.min_delta,
                     self.cfg.rel_delta * abs(im),
                     self.cfg.k_sigma * pooled / math.sqrt(max(r, 1)))
        z = ((cm - im) / (pooled / math.sqrt(max(r, 1)) + 1e-12)
             if pooled > 0 else float("inf") * np.sign(cm - im or 1))
        ev = {"challenger_mean": cm, "incumbent_mean": im, "reps": r,
              "margin": margin, "delta": cm - im, "z": float(z)}
        return (cm > im + margin), ev

    # ── prompt + insights ────────────────────────────────────────────
    def _load_insights(self) -> None:
        rows = self.db.conn.execute(
            "SELECT target, tags, kind, summary, delta, score "
            "FROM code_insights ORDER BY ts DESC LIMIT 200").fetchall()
        self._insights = [dict(target=r[0], tags=r[1] or "", kind=r[2],
                               summary=r[3], delta=r[4], score=r[5])
                          for r in rows]

    def _relevant_insights(self, k: int) -> list[str]:
        """Tag-overlap retrieval — cross-domain assist for the prompt."""
        tags = set(self.target.tags)
        scored = []
        for ins in self._insights:
            itags = set(ins["tags"].split(",")) - {""}
            ov = len(tags & itags)
            if ov or ins["target"] == self.target.name:
                scored.append((ov + 0.5 * (ins["target"] == self.target.name),
                               ins))
        scored.sort(key=lambda x: -x[0])
        return [f"- {s['summary']} (delta {s['delta']:+.4g})"
                for _, s in scored[:k]]

    # ── same-model judge ────────────────────────────────────────────
    # Promotions and plausible-but-wrong rejects get a critique from the
    # proposer itself: winner→winner pairs teach it WHAT works, and
    # rejected+why pairs are the DPO/repair signal. Critiques land in
    # code_insights, which INSIGHT: and _relevant_insights feed back
    # into later prompts.
    def _judge_gen(self, prompt: str) -> str:
        """One critique generation through whichever proposer handle is
        live; failures never propagate into the eval loop."""
        try:
            llm = getattr(self, "_basin_llm", None)
            if llm is not None and hasattr(llm, "propose_batch"):
                out = llm.propose_batch([prompt])
                raw = out[0] if out else ""
            elif (self._proposer is not None
                    and hasattr(self._proposer, "propose_batch")):
                # Mutator-shaped proposers (mutator/mixed mode) return
                # a mutated PROGRAM for any input — feeding a critique
                # prompt would write garbage insights and waste
                # mutations. propose_batch marks the LLM path.
                raw = self._proposer(prompt)
            else:
                return ""
            raw = raw or ""
            import re as _re
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.S)
            raw = _re.sub(r"```(?:python)?", "", raw)
            return " ".join(raw.split())[:300]
        except Exception:
            return ""

    def _judge_win(self, code: str, ev: dict) -> None:
        old = getattr(self, "_prev_incumbent", "") or ""
        if not old or old == code:
            return
        prompt = (
            "# Two implementations of the same function.\n"
            f"# CHALLENGER beat INCUMBENT by {ev.get('delta', 0):.4f} on "
            f"{self.metric}.\n"
            "INCUMBENT:\n```python\n" + old[:1400] + "\n```\n"
            "CHALLENGER:\n```python\n" + code[:1400] + "\n```\n"
            "In 2-3 sentences: what did the challenger change and why is "
            "it better? Name the concrete technique.")
        critique = self._judge_gen(prompt)
        if critique:
            self._write_insight(
                "judge",
                f"{self.target.name} win: {critique}",
                ev.get("delta", 0.0), ev.get("challenger_mean", 0.0))

    def _judge_reject(self, code: str, res: dict) -> None:
        stage = res.get("stage", "?")
        err = str(res.get("error", ""))[:200]
        prompt = (
            f"# This candidate compiled but FAILED at {stage}: {err}\n"
            "```python\n" + code[:1400] + "\n```\n"
            "In 2-3 sentences: what is wrong with it and what should be "
            "fixed?")
        critique = self._judge_gen(prompt)
        if critique:
            self._write_insight(
                "judge_fail",
                f"{self.target.name} fail@{stage}: {critique}",
                0.0, -1e9)

    def _write_insight(self, kind: str, summary: str, delta: float,
                       score: float) -> None:
        tags = ",".join(self.target.tags)
        with self.db._lock:
            self.db.conn.execute(
                "INSERT INTO code_insights(target, tags, kind, summary, delta,"
                " score, ts) VALUES (?,?,?,?,?,?,?)",
                (self.target.name, tags, kind, summary[:400], delta, score,
                 time.time()))
            self.db.conn.commit()

    @staticmethod
    def _approach_gist(code: str, skip: set | None = None) -> str:
        """'What it does' fingerprint: the callee names that distinguish
        one approach from another (sort+cumsum vs masked_fill loop).
        _gist's impl(names) is constant on single-required targets — it
        can't tell two fast_sum rewrites apart; this can."""
        calls = {m.group(1)
                 for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", code)}
        calls -= {"def", "if", "for", "while", "return", "print", "len",
                  "range", "int", "float", "str", "bool", "type", "super",
                  "isinstance", "hasattr", "getattr", "setattr", "repr",
                  "enumerate", "zip", "map", "filter", "list", "dict",
                  "set", "tuple", "open", "Exception", "compile"}
        calls -= (skip or set())
        keep = [c for c in sorted(calls) if not c.startswith("_")][:8]
        return "+".join(keep)

    def _grader_excerpt(self) -> str:
        """The model currently rewrites code blind to what's scored —
        it never sees test()/bench(). ast-extract both from the target
        spec (~1.5KB cap); knowing the input shapes + which metric is
        returned is the single biggest 'how do I actually optimize'
        signal the prompt was missing."""
        if getattr(self, "_grader_src", None) is not None:
            return self._grader_src
        src = ""
        try:
            import ast as _a
            import textwrap as _tw
            spec_src = Path(self.target.path).read_text(
                encoding="utf-8", errors="ignore")
            tree = _a.parse(spec_src)
            parts = []
            for name in ("test", "bench"):
                fn = next(
                    (n for n in tree.body
                     if isinstance(n, _a.FunctionDef) and n.name == name),
                    None)
                if fn is None:
                    continue
                seg = _a.get_source_segment(spec_src, fn) or ""
                seg = _tw.dedent(seg).strip()
                if seg:
                    parts.append(seg[:900])
            src = "\n\n".join(parts)[:1600]
        except Exception:
            src = ""
        self._grader_src = src
        return src

    def _self_report(self) -> str:
        """Self-assessment line for the prompt — aggregate weakness +
        token-cost feedback from this target's own history.

        The model can't fix what it can't see: 'dup 14, syntax 3' tells
        it WHICH contract it keeps breaking (vary approach vs emit valid
        code), and avg gen_tokens teaches the efficiency budget —
        completions that ramble cost GPU seconds with no quality gain.
        """
        try:
            rows = self.db.conn.execute(
                "SELECT stage, metrics_json FROM code_programs "
                "WHERE target=? ORDER BY id DESC LIMIT 60",
                (self.target.name,)).fetchall()
        except Exception:
            return ""
        if len(rows) < 5:
            return ""
        stages: dict[str, int] = {}
        toks: list[int] = []
        for stage, mj in rows:
            if stage and stage != "done":
                stages[stage] = stages.get(stage, 0) + 1
            if mj and '"gen_tokens"' in mj:
                try:
                    toks.append(int(mj.split('"gen_tokens": ')[1]
                                      .split(",")[0].split("}")[0]))
                except Exception:
                    pass
        parts = []
        if stages:
            top = sorted(stages.items(), key=lambda x: -x[1])[:4]
            parts.append("your recent rejects on this target: "
                         + ", ".join(f"{s}x{n}" for s, n in top))
        if toks:
            avg = sum(toks) / len(toks)
            if avg > 120:
                parts.append(f"avg completion ~{avg:.0f} tokens — "
                             "keep <think> short or skip it; long "
                             "answers are not scoring better")
        return "; ".join(parts)

    def _tried_approaches(self) -> str:
        """Compact ';'-joined approach fingerprints already evaluated —
        the prompt-side anti-dup line. Requires >=3 distinct approaches
        before it speaks; a lone fingerprint just repeats the shown
        incumbent."""
        # Snapshot under concurrent appends (basin emitter thread) —
        # iterating a mutating deque raises.
        g = [x for x in dict.fromkeys(list(self._tried_gists)) if x]
        if len(g) < 3:
            return ""
        s = "; ".join(g)
        return s[-160:] if len(s) > 160 else s

    def _build_prompt(self, parent: str | None,
                      inspiration: str | None, primed: bool = True,
                      shots: list[tuple[str, float]] | None = None) -> str:
        req = ", ".join(self.target.required) or "same as seed"
        bud = json.dumps(self.target.budgets) if self.target.budgets else "none"
        metrics = json.dumps({k: round(v, 6) if isinstance(v, float) else v
                              for k, v in self.incumbent_metrics.items()})
        ins = self._relevant_insights(self.cfg.n_insights)
        # Candidate tag inside the first ~200 chars — sft_train dedupes
        # rows on prompt[:200], so identical headers would collapse every
        # same-target example down to one.
        cand_tag = _norm_hash(parent or self.target.seed_code)[:8]
        lines = [
            "# Rewrite this Python program to be much better.",
            f"# Candidate: {cand_tag}",
            f"# Direction: {self.metric} {'MINIMIZE' if self.goal == 'min' else 'MAXIMIZE'}",
            f"# Hard budgets (must satisfy): {bud}",
            f"# Rules: keep public names ({req}); stdlib+numpy+torch only;",
            # The output contract must match the decode mode: primed
            # prompts END mid-function, so "one code block" is a mixed
            # signal — the model is told to fence but forced to continue.
            ("# continue the function body below — complete it, do not "
             "restart with a new block; identical semantics."
             if primed else
             "# output ONLY one python code block; identical semantics."),
        ]
        if getattr(getattr(self, "_proposer", None), "_lfm3", False) \
                or getattr(getattr(self, "_basin_llm", None),
                           "_lfm3", False):
            # The Claude-Code finetune defaults to its agentic edit
            # transcript (commentary + old_string/new_string XML +
            # tool calls) — say plainly that we want the file itself.
            lines.append(
                "# Output format: verbatim code only — no tool calls, "
                "no old_string/new_string edit XML, no commentary.")
        tried = self._tried_approaches()
        if tried:
            lines.append("# Already evaluated — write something "
                         f"DIFFERENT from: {tried}")
        sr = self._self_report()
        if sr:
            lines.append(f"# Self-check: {sr}")
        grader = self._grader_excerpt()
        if grader:
            lines.append("# Evaluated by (read the grader — optimize "
                         "what it measures, nothing else):")
            lines += [f"# {ln}" for ln in grader.splitlines()]
        if self.target.prompt_extra:
            lines.append(f"# Extra: {self.target.prompt_extra}")
        if shots and len(shots) >= 2:
            # FunSearch best-shot k=2: two prior programs shown sorted
            # worst->best; the model writes v2. Beats the flat "current
            # best" framing — the diff v0->v1 IS the lesson.
            lines += [
                f"#",
                f"# Two prior programs sorted worst to best on "
                f"{self.metric} — write v2, strictly better than v1:",
                f"# v0 (score {shots[0][1]:.6g}):",
                "```python",
                shots[0][0].strip(),
                "```",
                f"# v1 (score {shots[1][1]:.6g}):",
                "```python",
                shots[1][0].strip(),
                "```",
            ]
        else:
            lines += [
                f"#",
                f"# Current best ({self.metric}={self._raw_metric():.6g}, "
                f"metrics={metrics}):",
                "```python",
                (parent or self.target.seed_code).strip(),
                "```",
            ]
        if inspiration:
            lines += ["# A different attempt for inspiration:",
                      "```python", inspiration.strip()[:1200], "```"]
        if ins:
            lines += ["# Proven insights from related work:"] + ins
        if primed:
            lines.append("# Write the improved implementation now:")
            # Prime a completion-model INTO the required signature — it
            # then continues the body instead of re-typing the spec.
            # `self._prime` is re-attached to the raw output in run().
            imports = "\n".join(re.findall(r"^(?:import|from)\s+[^\n]+$",
                                          self.target.seed_code, re.M))
            sig = (_seed_signature(self.target.seed_code,
                                   self.target.required[0])
                   if self.target.required else "")
            self._prime = (imports + "\n\n\n" + sig).strip() if sig else imports
            if self._prime:
                lines += ["```python", self._prime]
        else:
            self._prime = ""
            lines.append("Reply with ONLY one ```python code block "
                         "containing the complete improved implementation. "
                         "Skip <think> for small changes; use a short "
                         "<think> only when the optimization is subtle.")
        return "\n".join(lines)

    def _raw_metric(self) -> float:
        v = self.incumbent_metrics.get(self.metric, 0.0)
        return float(v) if v is not None else 0.0

    def _tag_sig(self) -> str:
        return ",".join(sorted(self.target.tags))

    def _feedback(self, code: str, credit: float, delta: float,
                  prompt: str | None = None) -> None:
        """Outcome -> proposer policies (Mutator op bandit + LLMProposer
        hint bandit via note(code, credit, delta, prompt)). Custom
        proposers with the old 3-arg note() still work — TypeError
        falls back to the prompt-less call."""
        for prop in (self._proposer, self._basin_llm):
            n = getattr(prop, "note", None)
            if n is None:
                continue
            try:
                n(code, credit, delta, prompt=prompt)
            except TypeError:
                try:
                    n(code, credit, delta)
                except Exception:
                    pass
            except Exception:
                pass

    def _resume(self, workdir: Path) -> None:
        """Continue from this target's history in the same DB: prior best
        becomes the incumbent (re-measured honestly), recent code hashes
        are pre-seeded into the dedup set, and past top programs
        repopulate the island elites. 'Beats all other including past
        best' literally means the all-time best now."""
        # Exclude this run's own rows (the seed is logged before _resume
        # runs) and the seed itself — re-benching it would just adopt
        # bench noise as phantom improvement.
        seed_h = _norm_hash(self.target.seed_code)
        rows = self.db.conn.execute(
            "SELECT code, metrics_json, score, code_hash FROM code_programs"
            " WHERE target=? AND ok=1 AND run_id<>? AND code_hash<>?"
            " ORDER BY score DESC LIMIT 24",
            (self.target.name, self.run_id, seed_h)).fetchall()
        if not rows:
            return
        seen = self.db.conn.execute(
            "SELECT code_hash FROM code_programs WHERE target=? AND"
            " code_hash IS NOT NULL AND run_id<>?"
            " ORDER BY id DESC LIMIT 2000",
            (self.target.name, self.run_id)).fetchall()
        self._seen.update(r[0] for r in seen)
        resumed = False
        gated: set[str] = set()
        # Incumbent candidates are re-gated through the CURRENT test —
        # a stale "best" whose semantics drifted past today's gate (e.g.
        # after a gate-hardening edit) cannot resume as champion. Its
        # hash is still marked bad so children of it cost the mutation
        # op nothing but a gate reject.
        for code, _mj, _score, _h in rows[:3]:
            res = self._eval_sync(code, workdir)
            self._eval_n += 1
            if not res.get("ok"):
                gated.add(_norm_hash(code))
                continue
            s = self._score(res["metrics"])
            if s > self.incumbent_score:
                self.incumbent_code = code
                self.incumbent_metrics = res["metrics"]
                # Floor at this program's own recorded score: a fresh
                # re-measure under boot-time GPU contention lands BELOW
                # history's verified claim (measured 0.249 -> 0.314 on
                # quantize_4bit) — without the floor, an inflated
                # re-measure lowers the bar and regressions promote.
                self.incumbent_score = max(s, float(_score))
                resumed = True
                break
            # This row re-measured below the bar — an inflated
            # historical score must not block #2/#3, whose honest
            # re-measures can still beat the seed.
        for i, (code, _mj, score, h) in enumerate(rows):
            if (h or _norm_hash(code)) in gated:
                continue
            self._push_elite(i % self.cfg.islands, code,
                             float(score), h or _norm_hash(code))
            self._seen.add(h or _norm_hash(code))
            ag = self._approach_gist(code, set(self.target.required))
            if ag:
                self._tried_gists.append(ag)
        if self.cfg.verbose:
            print(f"[cevolve] resumed: {len(rows)} prior programs, "
                  f"incumbent {self._raw_metric():.6g} "
                  f"{'(prior best)' if resumed else '(seed held)'}")

    # ── island + parent selection ────────────────────────────────────
    def _pick_island(self, rng=None) -> int:
        rng = rng or self.rng
        fresh = [i for i, e in enumerate(self.elites) if not e]
        if fresh:
            return int(rng.choice(fresh))
        th = [rng.beta(a, b) for a, b in self._arms]
        return int(np.argmax(th))

    def _parents(self, island: int,
                 rng=None) -> tuple[str | None, str | None,
                                    list[tuple[str, float]] | None]:
        rng = rng or self.rng
        pool = list(self.elites[island])
        if not pool:
            return self.incumbent_code, None, None
        pool.sort(key=lambda e: -e[1])
        # rank-softmax over the top half + one random inspiration
        top = pool[: max(1, len(pool) // 2 + 1)]
        w = np.exp(-np.arange(len(top)))
        pent = top[int(rng.choice(len(top), p=w / w.sum()))]
        parent = pent[0]
        insp = None
        if len(pool) > 1 and rng.random() < 0.5:
            insp = pool[int(rng.integers(len(pool)))][0]
            if insp is parent:
                insp = None
        # FunSearch best-shot k=2 with OpenEvolve-style diverse
        # inspirations: shot 2 is the most code-DISTANT elite across ALL
        # islands (not the same-island next-best sibling — near-identical
        # shot pairs taught the model to re-emit the same family, which
        # dominated the dedup rejects). Cross-island distance in the
        # prompt is ring-migration by another route.
        shots = None
        far = None
        p_toks = set(parent.split())
        far_d = 2.0
        for elites in self.elites:
            for e in elites:
                if e is pent or e[2] == pent[2]:
                    continue
                et = set(e[0].split())
                j = (len(p_toks & et)
                     / max(1, len(p_toks | et)))
                if j < far_d:        # smallest Jaccard = most distant
                    far_d, far = j, e
        other = far if far is not None else (
            next((e for e in pool if e is not pent), None))
        if other is not None:
            pair = sorted((pent, other), key=lambda e: e[1])
            shots = [(c, s) for c, s, _ in pair]
        return parent, insp, shots

    def _push_elite(self, island: int, code: str, score: float,
                    code_hash: str) -> None:
        e = self.elites[island]
        if any(h == code_hash for _, _, h in e):
            return
        e.append((code, score, code_hash))
        e.sort(key=lambda x: -x[1])
        if not self.cfg.map_elites:
            del e[self.cfg.n_elites:]
            return
        # MAP-Elites-lite: a second feature dimension — (island,
        # code-length bin) cells. The bin splits at this island's
        # median code length; each bin keeps a guaranteed floor of
        # n_elites//4 elites, then the remaining slots fill by score —
        # short AND long program families survive (the shorter-is-
        # better bias can't purge structural diversity) while a
        # single-bin island still fills the whole pool.
        med = float(np.median([len(c) for c, _, _ in e]))
        floor = max(1, self.cfg.n_elites // 4)
        bins = [[], []]
        for ent in e:
            bins[int(len(ent[0]) > med)].append(ent)
        keep = bins[0][:floor] + bins[1][:floor]
        kept = {id(x) for x in keep}
        keep += [x for x in e
                 if id(x) not in kept][:max(0, self.cfg.n_elites
                                            - len(keep))]
        keep.sort(key=lambda x: -x[1])
        e[:] = keep

    # ── per-cycle halves (shared by serial and pooled modes) ─────────
    def _stagnation_check(self) -> bool:
        """Patience/basin-hop enforcement — True means stop."""
        cfg = self.cfg
        if self._dry >= cfg.patience:
            self.stop_reason = (f"stagnation: no verified promotion in "
                                f"{self._dry} cycles")
            return True
        # >=, not ==: a single propose call can burn dozens of gate
        # rejects (basin family enumeration drains up to 96) and jump
        # _dry right past hop_after — the old == check then never fired
        # and the run reached patience without a single hop.
        # `_hop_plateau` re-arms the hop after each promotion, keeping
        # the "one hop per dry-run plateau" semantics.
        if (self._dry >= cfg.hop_after
                and self._hop_plateau != self._promos):
            self._hop_plateau = self._promos
            cfg.temperature = min(1.2, cfg.temperature + 0.1)
            if cfg.island_cull and cfg.islands > 1:
                # FunSearch cull+clone: discard the worst half of the
                # islands entirely, then clone the best island's elite
                # set (and its bandit arm) into the least-bad culled
                # slot. Emptied islands reseed exploration through
                # _pick_island's fresh-island preference.
                ranked = sorted(
                    range(cfg.islands),
                    key=lambda i: max((s for _, s, _ in self.elites[i]),
                                      default=-1e9))
                n_cull = max(1, cfg.islands // 2)
                culled, best = ranked[:n_cull], ranked[-1]
                for i in culled:
                    self.elites[i] = []
                    self._arms[i] = [1.0, 1.0]
                clone = culled[-1]
                self.elites[clone] = list(self.elites[best])
                self._arms[clone] = list(self._arms[best])
                if cfg.verbose:
                    print(f"[cevolve] basin hop: culled islands "
                          f"{culled}, {clone} cloned from {best}, "
                          f"temp={cfg.temperature:.2f}")
            else:
                worst = min(range(cfg.islands),
                            key=lambda i: max((s for _, s, _ in
                                               self.elites[i]),
                                              default=-1e9))
                self.elites[worst] = self.elites[worst][:1]
                if cfg.verbose:
                    print(f"[cevolve] basin hop: island {worst} reset, "
                          f"temp={cfg.temperature:.2f}")
        return False

    def _repair_seed_names(self, code: str, names: set[str]) -> str | None:
        """Splice seed module-level statements that define `names` into
        `code`. Returns repaired code if EVERY missing name is provided
        by the seed, else None (a genuinely novel undefined reference —
        the model must be told it failed)."""
        import ast as _ast
        try:
            seed_tree = _ast.parse(self.target.seed_code)
        except SyntaxError:
            return None
        seed_lines = self.target.seed_code.splitlines()
        want = set(names)
        picked: list[tuple[int, int]] = []   # seed line ranges (0-based)
        for _ in range(4):                   # transitive deps of picked stmts
            progress = False
            for node in seed_tree.body:
                seg = _ast.get_source_segment(
                    self.target.seed_code, node)
                if seg is None:
                    continue
                provides = set()
                if isinstance(node, _ast.Assign):
                    for t in node.targets:
                        for n in _ast.walk(t):
                            if isinstance(n, _ast.Name):
                                provides.add(n.id)
                elif isinstance(node, _ast.AnnAssign):
                    for n in _ast.walk(node.target):
                        if isinstance(n, _ast.Name):
                            provides.add(n.id)
                elif isinstance(node, _ast.Import):
                    for a in node.names:
                        provides.add((a.asname or a.name).split(".")[0])
                elif isinstance(node, _ast.ImportFrom):
                    for a in node.names:
                        provides.add(a.asname or a.name)
                else:
                    continue
                hit = provides & want
                if not hit:
                    continue
                rng = (node.lineno - 1, node.end_lineno)
                if rng not in picked:
                    picked.append(rng)
                want -= hit
                # the picked statement may itself load other seed names
                for n in _ast.walk(node.value) if hasattr(
                        node, "value") and node.value is not None else ():
                    if (isinstance(n, _ast.Name)
                            and isinstance(n.ctx, _ast.Load)
                            and n.id not in provides):
                        want.add(n.id)
                progress = True
            if not want or not progress:
                break
        if want:
            return None
        # splice: keep candidate imports leading, then seed stmts, then rest
        cand_lines = code.splitlines()
        try:
            cand_tree = _ast.parse(code)
        except SyntaxError:
            return None
        insert_at = 0
        for node in cand_tree.body:
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                insert_at = node.end_lineno
            elif isinstance(node, _ast.Expr) and isinstance(
                    node.value, _ast.Constant) and isinstance(
                    node.value.value, str):
                insert_at = node.end_lineno      # module docstring
            else:
                break
        block = [seed_lines[a:b] for a, b in sorted(set(picked))]
        flat = [ln for grp in block for ln in grp]
        repaired = (cand_lines[:insert_at]
                    + ["", "# seed constants"] + flat
                    + cand_lines[insert_at:])
        out = "\n".join(repaired)
        try:
            compile(out, "<cand>", "exec")
        except SyntaxError:
            return None
        return out

    def _gate(self, code: str, island: int, prompt: str,
              parent_h: str | None):
        """Free gates — returns evaluable tuple or None (rejected +
        already logged/counted)."""
        try:
            compile(code, "<cand>", "exec")
        except SyntaxError as e:
            self._stage("syntax")
            self._log_program(code, island, self._eval_n,
                              _norm_hash(code),
                              {"stage": "syntax"}, -1e9, prompt=prompt)
            self._feedback(code, 0.0, -1.0, prompt=prompt)
            self._note_fail(island)
            self._queue_gate_repair(
                f"output was not valid Python "
                f"({e.msg} on line {e.lineno})",
                island, parent_h, prompt)
            return None
        missing = [
            n for n in self.target.required
            if not (re.search(rf"\bdef\s+{re.escape(n)}\b", code)
                    or re.search(rf"\bclass\s+{re.escape(n)}\b", code)
                    or re.search(rf"^{re.escape(n)}\s*=",
                                 code, re.M))]
        # AST gate: required defs must contain at least one return/yield —
        # truncated generations compile (signature + partial body) but can
        # only return None. Catching them here saves the subprocess eval.
        incomplete = False
        undefined = False
        undef_names: set[str] = set()
        if not missing and self.target.required:
            try:
                import ast as _ast
                import builtins as _bl
                tree = _ast.parse(code)
                for node in _ast.walk(tree):
                    if (isinstance(node, (_ast.FunctionDef,
                                          _ast.AsyncFunctionDef))
                            and node.name in self.target.required
                            and not any(
                                isinstance(n, (_ast.Return, _ast.Yield,
                                               _ast.YieldFrom))
                                for n in _ast.walk(node))):
                        incomplete = True
                        break
                # Names loaded at def time (decorators, default args,
                # bases, module statements) must exist at module scope —
                # a missing constant like `_TOP_P_CANDIDATES` compiles
                # fine but dies on exec. Cheap gate saves the subprocess.
                defined = set(dir(_bl)) | {
                    "__name__", "__file__", "__doc__", "__builtins__"}
                loads = []

                def _targets(t):
                    for n in _ast.walk(t):
                        if isinstance(n, _ast.Name):
                            defined.add(n.id)

                def _fn_header(node):
                    # Names + def-time loads of a def/lambda header.
                    for d in node.args.defaults + node.args.kw_defaults:
                        if d is not None:
                            loads.append(d)
                    decos = getattr(node, "decorator_list", [])
                    loads.extend(decos)
                    loads.extend(
                        a.annotation for a in
                        node.args.args + node.args.kwonlyargs
                        + [node.args.vararg, node.args.kwarg]
                        if a is not None and a.annotation is not None)
                    if node.returns is not None:
                        loads.append(node.returns)

                def _scan(stmts):
                    for node in stmts:
                        if isinstance(node, (_ast.FunctionDef,
                                             _ast.AsyncFunctionDef)):
                            defined.add(node.name)
                            _fn_header(node)
                            # body runs at call time — skip (exec gate
                            # covers runtime names; false-positive risk
                            # outweighs benefit)
                        elif isinstance(node, _ast.ClassDef):
                            defined.add(node.name)
                            loads.extend(node.bases)
                            loads.extend(k.value for k in node.keywords)
                            loads.extend(node.decorator_list)
                            _scan(node.body)
                        elif isinstance(node, _ast.Assign):
                            loads.append(node.value)
                            for t in node.targets:
                                _targets(t)
                        elif isinstance(node, _ast.AnnAssign):
                            if node.value is not None:
                                loads.append(node.value)
                            _targets(node.target)
                        elif isinstance(node, _ast.AugAssign):
                            loads.append(node.value)
                            _targets(node.target)
                        elif isinstance(node, _ast.Import):
                            for a in node.names:
                                defined.add(
                                    (a.asname or a.name).split(".")[0])
                        elif isinstance(node, _ast.ImportFrom):
                            for a in node.names:
                                defined.add(a.asname or a.name)
                        elif isinstance(node, (_ast.For, _ast.AsyncFor)):
                            _targets(node.target)
                            loads.append(node.iter)
                            _scan(node.body)
                            _scan(node.orelse)
                        elif isinstance(node, (_ast.With, _ast.AsyncWith)):
                            for item in node.items:
                                loads.append(item.context_expr)
                                if item.optional_vars is not None:
                                    _targets(item.optional_vars)
                            _scan(node.body)
                        elif isinstance(node, (_ast.If, _ast.While)):
                            loads.append(node.test)
                            _scan(node.body)
                            _scan(node.orelse)
                        elif isinstance(node, _ast.Expr):
                            loads.append(node.value)
                        elif isinstance(node, (_ast.Try, _ast.TryStar)):
                            _scan(node.body)
                            _scan(node.orelse)
                            _scan(node.finalbody)
                            for h in node.handlers:
                                if h.type is not None:
                                    loads.append(h.type)
                                if h.name:
                                    defined.add(h.name)
                                _scan(h.body)
                        elif isinstance(node, (_ast.Assert, _ast.Raise)):
                            for f in _ast.iter_child_nodes(node):
                                loads.append(f)
                        elif isinstance(node, (_ast.Return, _ast.Delete,
                                               _ast.Global, _ast.Nonlocal,
                                               _ast.Pass, _ast.Break,
                                               _ast.Continue)):
                            pass
                        else:
                            _scan(list(_ast.iter_child_nodes(node)))

                _scan(tree.body)
                # Loads are only valid if the name was defined anywhere at
                # module level (order-free — forward refs inside def-time
                # positions still crash, but flagging those needs ordering;
                # accept the small false-negative class instead).
                queue = list(loads)
                while queue:
                    n = queue.pop()
                    if isinstance(n, _ast.Name):
                        if (isinstance(n.ctx, _ast.Load)
                                and n.id not in defined):
                            undef_names.add(n.id)
                    elif isinstance(n, (_ast.FunctionDef,
                                        _ast.AsyncFunctionDef,
                                        _ast.Lambda, _ast.ClassDef)):
                        # nested scope — don't descend into its body
                        continue
                    else:
                        queue.extend(_ast.iter_child_nodes(n))
                undefined = bool(undef_names)
            except SyntaxError:
                pass
        # Repair: undefined names that the seed defines at module level
        # (e.g. `_TOP_P_CANDIDATES` referenced from a primed signature —
        # the model only writes the body and can never emit the constant)
        # get spliced in rather than rejected.
        if undefined:
            repaired = self._repair_seed_names(code, undef_names)
            if repaired is not None:
                code = repaired
                undefined = False
        ch = _norm_hash(code)
        if missing or incomplete or undefined or ch in self._seen:
            stage = ("dup" if ch in self._seen
                     else "incomplete" if incomplete
                     else "undefined" if undefined else "missing")
            self._stage(stage)
            self._log_program(code, island, self._eval_n, ch,
                              {"stage": stage}, -1e9, prompt=prompt)
            # dup = valid code but no novelty — milder penalty than
            # a real failure; missing/incomplete/undefined = broke the contract
            self._feedback(code, 0.0 if (missing or incomplete or undefined)
                           else 0.05, -1.0, prompt=prompt)
            self._note_fail(island)
            dup_gist = (self._approach_gist(code, set(self.target.required))
                        if stage == "dup" else "")
            detail = (
                f"this exact approach was already evaluated "
                f"({dup_gist or 'same program'}) — write a MATERIALLY "
                "DIFFERENT implementation (different algorithm, not a "
                "rewording)" if stage == "dup"
                else f"a required function has no return/yield — "
                     "it must return a value" if stage == "incomplete"
                else f"uses undefined name(s): "
                     f"{', '.join(sorted(undef_names))} — define or "
                     "import them" if stage == "undefined"
                else f"the program must define: {', '.join(missing)}")
            self._queue_gate_repair(detail, island, parent_h, prompt)
            return None
        self._seen.add(ch)
        ag = self._approach_gist(code, set(self.target.required))
        if ag:
            self._tried_gists.append(ag)
        return (code, island, parent_h, ch, prompt)

    def _propose(self):
        """Stagnation check + gates. Returns:
        "stop"               -> patience exhausted (set stop_reason)
        None                 -> cheap reject (already logged/counted)
        (code, island, parent_h, ch, prompt) -> ready for subprocess eval"""
        if self._basin_mode:
            return self._propose_basin()
        if self._stagnation_check():
            return "stop"
        island = self._pick_island()
        parent, insp, shots = self._parents(island)
        # Multi-required targets need several public defs — priming a
        # single signature makes the model emit only that one. Let the
        # model write a complete fenced program instead.
        primed = (getattr(self._proposer, "primed", True)
                  and len(self.target.required) <= 1)
        prompt = self._build_prompt(parent, insp, primed=primed,
                                    shots=shots)
        # Capture the prime NOW — _prime is shared mutable state; in
        # pool mode sibling eval threads can overwrite it between the
        # build and the extract (same wrong-prime class as the batch
        # path's per-prompt capture).
        prime = self._prime
        try:
            raw = self._proposer(prompt) if self._proposer else ""
            self._gen_n += 1
        except Exception as e:
            if self.cfg.verbose:
                print(f"[cevolve] proposer error: {e}")
            self._dry += 1
            return None
        code = _extract_code(raw, prime, self.target.required)
        self._log_gen(prompt, raw, code)
        cand = self._gate(code, island, prompt,
                          _norm_hash(parent) if parent else None)
        if cand is not None:
            self._gen_ok += 1
        return cand

    # ── basin mode: async LLM parents + deterministic fan-out ────────
    def _basin_loop(self) -> None:
        """Producer thread: keeps _basin_q stocked with LLM basin
        parents. Batched via LLMProposer.propose_batch (one GPU pass per
        basin_batch prompts) — the LLM is off the candidate critical
        path entirely; the mutator fan-out does the volume work."""
        import queue as _q
        rng = np.random.default_rng(self.cfg.seed + 7919)
        while not self._stop_evt.is_set():
            try:
                if self._basin_q.qsize() >= self.cfg.basin_q:
                    time.sleep(0.1)
                    continue
                batch = []
                llm_primed = getattr(self._basin_llm, "primed", False)
                for _ in range(self.cfg.basin_batch):
                    island = self._pick_island(rng)
                    parent, insp, shots = self._parents(island, rng)
                    prompt = self._build_prompt(parent, insp,
                                                primed=llm_primed,
                                                shots=shots)
                    # _prime is a _build_prompt side effect — capture it
                    # PER prompt or every raw below gets extracted with
                    # the LAST prompt's signature (wrong-prime soup).
                    batch.append((island, parent, prompt, self._prime))
                raws = self._basin_llm.propose_batch(
                    [p for _, _, p, _ in batch])
                batch_seen: set[str] = set()
                for (island, parent, prompt, prime), raw in zip(batch, raws):
                    self._gen_n += 1
                    code = _extract_code(raw or "", prime,
                                         self.target.required)
                    self._log_gen(prompt, raw or "", code)
                    # Token-efficiency signal: how much the proposer
                    # spent (chars ≈ tokens/4) and whether it reasoned
                    # first — feeds adaptive-reasoning stats + SFT prefs.
                    gen_meta = {
                        "stage": "gen",
                        "gen_tokens": len(raw or "") // 4,
                        "had_think": "<think>" in (raw or ""),
                    }
                    # Log every LLM completion — even non-compiling ones —
                    # as stage='gen' rows. The prompt+raw pairs are the
                    # proposer SFT/DPO signal (chosen/rejected downstream);
                    # dropping them here silently threw away ~90% of the
                    # model's output and starved the next training round.
                    if not code:
                        self._log_program(
                            (raw or "")[:20000], island, self._eval_n,
                            None, gen_meta, -1e9, prompt=prompt)
                        continue
                    try:
                        compile(code, "<basin>", "exec")
                    except SyntaxError:
                        self._log_program(
                            code, island, self._eval_n, _norm_hash(code),
                            gen_meta, -1e9, prompt=prompt)
                        continue
                    bh = _norm_hash(code)
                    if bh in self._seen or bh in batch_seen:
                        self._log_program(
                            code, island, self._eval_n, bh,
                            {"stage": "gen_dup"}, -1e9, prompt=prompt)
                        continue
                    batch_seen.add(bh)
                    self._gen_ok += 1
                    try:
                        self._basin_q.put(
                            {"code": code, "island": island,
                             "prompt": prompt,
                             "gen_meta": gen_meta,
                             "adapter": getattr(self._basin_llm,
                                                "_last_adapter", None),
                             "parent_h": (_norm_hash(parent)
                                          if parent else None)},
                            timeout=0.5)
                    except _q.Full:
                        break
            except Exception as e:
                self._basin_fails = getattr(self, "_basin_fails", 0) + 1
                if self.cfg.verbose:
                    print(f"[cevolve] basin emitter error: {e}")
                if self._basin_fails > 4:
                    return              # dead emitter — mutator-only
                time.sleep(0.5)

    def _open_basin_family(self, basin: dict) -> None:
        self._family_from_basin = True
        src = basin["code"]
        self._family_island = basin["island"]
        self._family_prompt = basin["prompt"]
        self._family_ph = _norm_hash(src)
        self._mutator._last_adapter = basin.get("adapter")
        # Cross LLM basins with the incumbent — the most useful
        # recombination available for free.
        insp = (self.incumbent_code
                if self.incumbent_code != src else None)
        ag = self._approach_gist(src, set(self.target.required))
        if ag:
            self._tried_gists.append(ag)
        self._family = self._mutator.mutants_for(
            src, insp, max_family=self.cfg.fanout,
            seed_base=self._eval_n + len(self._seen_ops))

    def _open_island_family(self) -> None:
        self._family_from_basin = False
        self._mutator._last_adapter = None
        self._family_island = self._pick_island()
        parent, insp, shots = self._parents(self._family_island)
        src = parent or self.target.seed_code
        self._family_prompt = self._build_prompt(
            parent, insp, primed=False, shots=shots)
        self._family_ph = (_norm_hash(parent) if parent
                           else _norm_hash(src))
        self._family = self._mutator.mutants_for(
            src, insp, max_family=self.cfg.fanout,
            seed_base=self._eval_n + len(self._seen_ops))

    def _next_family(self) -> None:
        """Pick the next parent (a queued LLM basin if one is waiting,
        else an island parent) and open its mutator enumeration."""
        import queue as _q
        basin = None
        if self._basin_q is not None:
            try:
                basin = self._basin_q.get_nowait()
            except _q.Empty:
                basin = None
        if basin is not None:
            self._open_basin_family(basin)
        else:
            self._open_island_family()

    def _propose_basin(self):
        """One evaluable candidate per call from the current family —
        uniqueness by construction via (parent_h, op_sig), plus the
        global norm-hash gate for cross-parent collisions. When the
        enumeration runs dry and the LLM emitter is live, BLOCK for a
        basin parent — the producer is the constraint, patience is not."""
        import queue as _q
        if self._stagnation_check():
            return "stop"
        for _ in range(96):
            if self._family is None:
                self._next_family()
            item = next(self._family, None)
            if item is None:
                self._family = None
                if self._basin_llm is not None \
                        and getattr(self, "_basin_fails", 0) <= 4 \
                        and not self._stop_evt.is_set():
                    # Starvation clock: a live emitter whose last basin
                    # produced >=1 variant keeps the window open — model
                    # load + 72s batches are latency, not stagnation.
                    if time.time() - self._basin_last_good \
                            < self.cfg.basin_starve_s:
                        try:
                            basin = self._basin_q.get(timeout=2.5)
                            self._open_basin_family(basin)
                        except _q.Empty:
                            pass
                        continue
                # enumeration dry and no basin on the way — real stall
                self._dry += 1
                return None
            if self._family_from_basin:
                self._basin_last_good = time.time()
            sig, code = item
            key = (self._family_ph, sig)
            if key in self._seen_ops:
                continue
            if len(self._seen_ops) > 50000:
                # Unbounded growth across a long campaign — the set is
                # a dedup aid, not a ledger; a reset re-permits old
                # (parent,op) pairs rather than pinning memory forever.
                self._seen_ops.clear()
            self._seen_ops.add(key)
            cand = self._gate(code, self._family_island,
                              self._family_prompt, self._family_ph)
            if cand is not None:
                return cand
        # 96 consecutive tries produced nothing evaluable — the
        # (parent, op) space is genuinely dry; count it so patience can
        # fire instead of spinning on empty enumerations.
        self._dry += 1
        return None

    def _propose_burst(self, k: int):
        """Batched proposal — ForgeEngine generate_batch produces k
        DIFFERENT parent/island prompts in one GPU pass, so the eval
        pool fills with diverse candidates instead of waiting serially.
        Returns "stop" or a list of evaluable tuples."""
        if self._basin_mode:
            out = []
            for _ in range(k):
                c = self._propose_basin()
                if c == "stop":
                    return "stop" if not out else out
                if isinstance(c, tuple):
                    out.append(c)
            return out
        if self._stagnation_check():
            return "stop"
        # Multi-required targets need several public defs — priming a
        # single signature makes the model emit only that one (same rule
        # as _propose).
        primed = (getattr(self._proposer, "primed", True)
                  and len(self.target.required) <= 1)
        items = []
        for _ in range(k):
            island = self._pick_island()
            parent, insp, shots = self._parents(island)
            prompt = self._build_prompt(
                parent, insp, primed=primed, shots=shots)
            items.append((island, parent, prompt, self._prime))
        prompts = [p for _, _, p, _ in items]
        try:
            if hasattr(self._proposer, "propose_batch"):
                raws = self._proposer.propose_batch(prompts)
            else:
                raws = [self._proposer(p) for p in prompts]
        except Exception as e:
            if self.cfg.verbose:
                print(f"[cevolve] proposer burst error: {e}")
            self._dry += 1
            return []
        out = []
        batch_seen: set[str] = set()
        for (island, parent, prompt, prime), raw in zip(items,
                                                        raws or []):
            self._gen_n += 1
            code = _extract_code(raw, prime, self.target.required)
            self._log_gen(prompt, raw, code)
            bh = _norm_hash(code)
            # Same-batch dups gate once: repeats are a sampling artifact
            # (one hot batch row), not a search failure — no _dry, no
            # island penalty, just a logged stage.
            if bh in batch_seen:
                self._stage("dup_batch")
                self._log_program(code, island, self._eval_n, bh,
                                  {"stage": "dup_batch"}, -1e9,
                                  prompt=prompt)
                continue
            batch_seen.add(bh)
            cand = self._gate(code, island, prompt,
                              _norm_hash(parent) if parent else None)
            if cand is not None:
                self._gen_ok += 1
                out.append(cand)
        return out

    def _queue_gate_repair(self, detail: str, island: int,
                           parent_h: str | None,
                           prompt: str | None) -> None:
        """Gate-stage rejects (soup/missing/incomplete/undefined/dup)
        carry a fixable artifact too — the REASON it failed goes back
        into ONE regen, same contract as eval-error repairs (the marker
        prevents re-repair, shares repair_per_batch cap). Basin mode is
        exempt: the mutator fan-out supplies volume for free, so GPU
        re-prompting there buys nothing."""
        if (self._basin_mode or not self.cfg.repair_regen or not prompt
                or "# Previous attempt failed" in prompt
                or self._repair_proposer() is None
                or len(self._repair_q) >= self.cfg.repair_per_batch):
            return
        block = f"# Previous attempt failed: {detail}"
        # Same insertion rule as _with_hint/_queue_repair: BEFORE the
        # primed fence — appending after the signature corrupts the
        # continuation.
        i = prompt.rfind("```python")
        tail = prompt[i + len("```python"):] if i >= 0 else ""
        prime = "" if "```" in tail else tail.strip()
        rep = (prompt[:i] + block + "\n" + prompt[i:]
               if prime else prompt + "\n" + block)
        self._repair_q.append({"island": island, "parent_h": parent_h,
                               "prompt": rep, "prime": prime})

    # ── artifact feedback: repair regen (OpenEvolve/STaR) ────────────
    def _repair_proposer(self):
        """The callable that repairs — basin mode's LLM emitter when
        live, else the main proposer. A bare Mutator can't read error
        text (it only parses fenced code), so it doesn't qualify."""
        prop = (self._basin_llm
                if (self._basin_mode and self._basin_llm is not None)
                else self._proposer)
        if prop is None or isinstance(prop, Mutator):
            return None
        return prop

    def _queue_repair(self, island: int, parent_h: str | None,
                      res: dict, prompt: str | None) -> None:
        """Queue ONE repair regen for a REAL error failure — test raised,
        exec traceback, bench contract violation. Score-misses
        (detail=='failed') and infra failures (timeout/mem/spawn) carry
        no useful artifact. Bounded to `repair_per_batch` per drain
        cycle; the '# Previous attempt failed' marker in the regen
        prompt means a failed repair is never itself re-repaired."""
        det = (res.get("detail") or "").strip()
        if not (self.cfg.repair_regen and prompt
                and "# Previous attempt failed" not in prompt
                and res.get("stage") in ("exec", "test", "bench")
                and det and det != "failed"
                and self._repair_proposer() is not None
                and len(self._repair_q) < self.cfg.repair_per_batch):
            return
        err = "\n".join("# " + ln for ln in det[-400:].splitlines())
        block = f"# Previous attempt failed with:\n{err}"
        # Same insertion rule as _with_hint: BEFORE the primed fence —
        # appending after the signature corrupts the continuation. The
        # tail after the last ```python is the prime iff it was never
        # closed by a second fence.
        i = prompt.rfind("```python")
        tail = prompt[i + len("```python"):] if i >= 0 else ""
        prime = "" if "```" in tail else tail.strip()
        rep = (prompt[:i] + block + "\n" + prompt[i:]
               if prime else prompt + "\n" + block)
        self._repair_q.append({"island": island, "parent_h": parent_h,
                               "prompt": rep, "prime": prime})

    def _drain_repairs(self, workdir: Path, epool, pool, pending: dict,
                       carry: list) -> None:
        """ONE batched regen for this cycle's queued error-failures —
        the same single-batched-regen pattern as _research_followup.
        Each repaired program is re-gated and re-submitted as a fresh
        candidate (epool job / pool future / serial eval)."""
        jobs, self._repair_q = self._repair_q, []
        prop = self._repair_proposer()
        if not jobs or prop is None:
            return
        prompts = [j["prompt"] for j in jobs]
        try:
            if hasattr(prop, "propose_batch"):
                raws = prop.propose_batch(prompts)
            else:
                raws = [prop(p) for p in prompts]
        except Exception:
            return
        if len(raws or []) < len(jobs):
            # Proposer returned fewer raws than jobs — the missing
            # repairs just vanish; count it so silent starvation shows
            # in stage_counts instead of looking like a working drain.
            self._stage("repair_starved")
        for job, raw in zip(jobs, raws or []):
            self._gen_n += 1
            try:
                code = _extract_code(raw or "", job["prime"],
                                     self.target.required)
                self._log_gen(job["prompt"], raw or "", code)
                cand = self._gate(code, job["island"], job["prompt"],
                                  job["parent_h"])
            except Exception:
                self._stage("repair_drop")
                continue
            if cand is None:
                continue
            self._gen_ok += 1
            c2, isl, ph, ch, pr = cand
            if epool is not None:
                jid = epool.submit(c2, tag=ch, cheap=self._cheap_spec())
                if jid is None:
                    carry.append(cand)
                else:
                    pending[jid] = cand
            elif pool is not None:
                pending[pool.submit(self._eval, c2, workdir, ch,
                                    self._cheap_spec())] = cand
            else:
                res2 = self._eval(c2, workdir, ch, self._cheap_spec())
                self._eval_n += 1
                self._handle_result(c2, isl, ph, ch, res2, workdir, pr)

    def _handle_result(self, code: str, island: int,
                       parent_h: str | None, ch: str,
                       res: dict, workdir: Path,
                       prompt: str | None = None) -> None:
        """Score/promotion/islands/feedback for one finished eval."""
        cfg = self.cfg
        score = self._score(res.get("metrics", {})) if res.get("ok") else -1e9
        self._log_program(code, island, self._eval_n, ch, res, score,
                          parent_hash=parent_h, prompt=prompt)
        if cfg.eval_log:
            # One line per eval, IN result-arrival order — crash/bug
            # attribution: the last line before a hang/crash names the
            # exact candidate + stage that killed or stalled.
            m = res.get("metrics", {})
            mv = m.get(self.metric) if isinstance(m, dict) else None
            print(f"[ev {self._eval_n:>5}] i{island} "
                  f"{res.get('stage', '?'):>10} "
                  f"{mv if mv is None else f'{float(mv):.6g}'} "
                  f"h{(ch or '')[:8]} "
                  f"{(res.get('detail') or '')[:120]}", flush=True)

        promoted = False
        if res.get("ok") and self._fits_budgets(res["metrics"]) \
                and score > self.incumbent_score:
            ok, ev = self._verified(code, score, workdir)
            if ok and ev["challenger_mean"] <= self.incumbent_score:
                # Beat the re-measured incumbent under concurrent GPU
                # contention but NOT the recorded best — a regression
                # ratchet: each promotion would lower the stored score
                # bar (measured: 0.294 -> 0.345 -> 0.703 on quantize_4bit).
                # The recorded best may only ever improve.
                ok = False
            if not ok:
                self._feedback(code, 0.55, score - self.incumbent_score,
                               prompt=prompt)
            if ok:
                # keep the beaten incumbent reachable for the judge —
                # incumbent_code is about to point at the winner.
                self._prev_incumbent = self.incumbent_code
                self.incumbent_code = code
                self.incumbent_score = ev["challenger_mean"]
                # challenger metrics may be richer — keep its last bench
                self.incumbent_metrics = res["metrics"]
                self._arms[island][0] += 1.0
                self._dry = 0
                self._promos += 1
                self._write_status()
                promoted = True
                self._push_elite(island, code, ev["challenger_mean"], ch)
                self._write_insight(
                    "win",
                    f"{self.target.name}: {self.metric} "
                    f"{self._raw_metric():.6g} via {self._gist(code)}",
                    ev["delta"], ev["challenger_mean"])
                # Same-model judge: critique the promotion while both
                # versions are in scope — the critique lands in
                # code_insights ('judge' kind) where INSIGHT: and
                # _relevant_insights feed it back into future prompts.
                if self.cfg.judge:
                    try:
                        self._judge_win(code, ev)
                    except Exception:
                        pass  # critique is advisory — never block eval
                self._feedback(code, 1.0, ev["delta"], prompt=prompt)
                if cfg.verbose:
                    zs = f"z={ev['z']:.2f}" if ev["z"] is not None else "-"
                    print(f"[cevolve] eval {self._eval_n} "
                          f"{self.target.name}: NEW BEST "
                          f"{self._raw_metric():.6g} "
                          f"(+{ev['delta']:.4f} {zs})")
        if not promoted:
            if not res.get("ok"):
                # Plausible-but-wrong rejects (compiled, reached a real
                # stage) earn a judged critique — the rejected+why pairs
                # are the DPO/repair signal; 'gen'-stage soup is skipped
                # (nothing to critique in token noise).
                if (self.cfg.judge
                        and res.get("stage") in ("test", "exec", "bench")):
                    try:
                        self._judge_reject(code, res)
                    except Exception:
                        pass  # critique is advisory — never block eval
                # test/exec/timeout — cheap_gate stays ok=True below
                self._feedback(code, 0.15, -1.0, prompt=prompt)
                # artifact feedback: the error text earns ONE repair
                # regen (drained in a batch by the run loop)
                self._queue_repair(island, parent_h, res, prompt)
            elif score <= self.incumbent_score:
                self._feedback(code, 0.3, score - self.incumbent_score,
                               prompt=prompt)
            else:
                # scored better but over-budget (verify-fail fed above)
                self._feedback(code, 0.4, score - self.incumbent_score,
                               prompt=prompt)
            self._note_fail(island, res)
            if res.get("ok"):
                # good but not best — still archive it (round's candidates)
                self._push_elite(island, code, score, ch)

    # ── main loop ────────────────────────────────────────────────────
    def run(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        self._load_insights()
        workdir = Path(tempfile.mkdtemp(prefix="cev_"))
        if cfg.verbose:
            print(f"[cevolve] target={self.target.name} metric={self.metric} "
                  f"goal={self.goal} budgets={self.target.budgets} "
                  f"run={self.run_id}")

        # Eval pipeline: persistent worker pool — spawn+target-import
        # cost paid ONCE per worker instead of per candidate (the old
        # path measured ~0.92 evals/min sustained; workers idle when the
        # submit limit is low). workers==1 keeps the serial subprocess
        # path for debugging; eval_pool=False forces the old threadpool.
        # Promotion honesty is unchanged: _verified still re-measures
        # challenger and incumbent in the same load regime.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if cfg.eval_pool and cfg.workers != 1:
            try:
                from .eval_pool import EvalPool
                n = (cfg.workers if cfg.workers > 1 else
                     min(cfg.max_workers, 4 if self._cuda_eval else 16))
                self._epool = EvalPool(str(self.target.path), n_workers=n,
                                       timeout_s=cfg.timeout_s,
                                       mem_cap_mb=cfg.worker_mem_mb)
                if cfg.verbose:
                    print(f"[cevolve] eval pool: {n} persistent workers")
            except Exception as e:
                print(f"[cevolve] eval pool failed ({e}) — "
                      "subprocess fallback")
                self._epool = None
        if self._basin_mode:
            import queue as _q
            self._basin_q = _q.Queue(maxsize=max(4, cfg.basin_q * 2))
            self._basin_last_good = time.time()   # starve clock = boot
            try:
                self._basin_llm = LLMProposer(cfg, db=self.db,
                                              tag_sig=self._tag_sig())
            except Exception as e:
                print(f"[cevolve] basin LLM unavailable ({e}) — "
                      "mutator-only basin mode")
                self._basin_llm = None
            if self._basin_llm is not None:
                self._basin_t = threading.Thread(
                    target=self._basin_loop, daemon=True,
                    name="cev_basin")
                self._basin_t.start()

        # Baseline: the original code is champion #0 (re-benched honestly).
        # First eval eats worker boot + per-worker CUDA-context init —
        # give it a longer leash than steady-state candidates.
        res = self._eval_sync(self.target.seed_code, workdir,
                              timeout_s=max(180.0, cfg.timeout_s))
        if not res.get("ok"):
            print(f"[cevolve] SEED FAILED at {res.get('stage')}: "
                  f"{res.get('detail', '')[:300]}")
            return {"error": "seed_eval_failed", "detail": res}
        self.incumbent_code = self.target.seed_code
        self.incumbent_metrics = res["metrics"]
        self.incumbent_score = self._score(res["metrics"])
        self.baseline_score = self.incumbent_score
        self._eval_n += 1
        self._log_program(self.target.seed_code, 0, -1, None, res,
                          self.incumbent_score)
        for i in range(cfg.islands):
            self._push_elite(i, self.target.seed_code, self.incumbent_score,
                             _norm_hash(self.target.seed_code))
        self._seen.add(_norm_hash(self.target.seed_code))
        _ag = self._approach_gist(self.target.seed_code,
                                set(self.target.required))
        if _ag:
            self._tried_gists.append(_ag)
        if cfg.resume:
            self._resume(workdir)
        if cfg.verbose:
            print(f"[cevolve] seed {self.metric}="
                  f"{float(res['metrics'][self.metric]):.6g} "
                  f"(score {self.incumbent_score:.4f})")

        # Legacy fallback: ThreadPoolExecutor around subprocess.run —
        # kept for eval_pool=False and for when the pool fails to spawn.
        use_pool = self._epool is None and (cfg.workers > 1 or self._auto)
        pool = (ThreadPoolExecutor(
                    max_workers=(cfg.workers if cfg.workers > 1
                                 else cfg.max_workers),
                    thread_name_prefix="cev_eval")
                if use_pool else None)
        parallel = self._epool is not None or pool is not None
        pending: dict = {}
        self._live_pending = pending          # status-JSON in-flight view
        carry: list = []                        # gated, awaiting a worker
        stop = False
        try:
            while not stop and self._eval_n + len(pending) \
                    + len(carry) < cfg.evals_budget:
                # Fill the pipe — cheap gates run inline either way.
                while (not parallel or len(pending) < self._limit) \
                        and self._eval_n + len(pending) + len(carry) \
                        < cfg.evals_budget:
                    if carry and self._epool is not None:
                        cand = carry[0]
                        jid = self._epool.submit(
                            cand[0], tag=cand[3], cheap=self._cheap_spec())
                        if jid is None:
                            break             # workers saturated
                        carry.pop(0)
                        pending[jid] = cand
                        continue
                    room = self._limit - len(pending)
                    burstable = (parallel and room >= 4
                                 and hasattr(self._proposer,
                                             "propose_batch"))
                    cands = (self._propose_burst(min(room, 24))
                             if burstable else [self._propose()])
                    if cands == "stop" or cands == ["stop"]:
                        stop = True
                        break
                    if not cands:
                        if not parallel:
                            break             # serial: keep single-step
                        if self._epool is not None and not pending:
                            time.sleep(0.02)  # nothing gated AND pool
                            break             # empty — poll for results
                        continue
                    for cand in cands:
                        if cand is None:      # cheap reject — keep filling
                            continue
                        code, island, parent_h, ch, prompt = cand
                        if not parallel:
                            res = self._eval(code, workdir, ch,
                                             self._cheap_spec())
                            self._eval_n += 1
                            self._handle_result(code, island, parent_h, ch,
                                                res, workdir, prompt)
                            self._drain_repairs(workdir, None, None,
                                                pending, carry)
                            break              # serial: one per outer iter
                        if self._epool is not None:
                            jid = self._epool.submit(
                                code, tag=ch, cheap=self._cheap_spec())
                            if jid is None:
                                carry.append(cand)   # retry next fill
                                continue
                            pending[jid] = cand
                        else:
                            fut = pool.submit(self._eval, code, workdir,
                                              ch, self._cheap_spec())
                            pending[fut] = cand
                        self._probe_load()
                    if not parallel:
                        break
                if not parallel or stop:
                    continue
                if not pending and not carry:
                    continue
                # Wait for ≥1 eval, process in completion order.
                if self._epool is not None:
                    got = self._epool.poll(timeout=0.5)
                    if got is None:
                        continue
                    jid, res = got
                    cand = pending.pop(jid, None)
                    if cand is None:           # verify-time job — skip
                        continue
                    code, island, parent_h, ch, prompt = cand
                    self._eval_n += 1
                    self._handle_result(code, island, parent_h, ch,
                                        res, workdir, prompt)
                    self._drain_repairs(workdir, self._epool, None,
                                        pending, carry)
                    self._probe_load()
                else:
                    for fut in as_completed(list(pending)):
                        res = fut.result()
                        code, island, parent_h, ch, prompt = \
                            pending.pop(fut)
                        self._eval_n += 1
                        self._handle_result(code, island, parent_h, ch,
                                            res, workdir, prompt)
                        self._drain_repairs(workdir, None, pool,
                                            pending, carry)
                        self._probe_load()
                        break
                if self._dry >= cfg.patience:
                    self.stop_reason = (f"stagnation: no verified promotion "
                                        f"in {self._dry} cycles")
                    stop = True
            # Drain in-flight evals — their rows/feedback still count.
            while pending:
                if self._epool is not None:
                    got = self._epool.poll(timeout=2.0)
                    if got is None:
                        break
                    jid, res = got
                    cand = pending.pop(jid, None)
                    if cand is None:
                        continue
                    code, island, parent_h, ch, prompt = cand
                    self._eval_n += 1
                    self._handle_result(code, island, parent_h, ch,
                                        res, workdir, prompt)
                    self._drain_repairs(workdir, self._epool, None,
                                        pending, carry)
                else:
                    for fut in as_completed(list(pending)):
                        res = fut.result()
                        code, island, parent_h, ch, prompt = \
                            pending.pop(fut)
                        self._eval_n += 1
                        self._handle_result(code, island, parent_h, ch,
                                            res, workdir, prompt)
                        self._drain_repairs(workdir, None, pool,
                                            pending, carry)
                        break
        finally:
            self._stop_evt.set()
            if self._basin_t is not None:
                self._basin_t.join(timeout=3)
            if self._epool is not None:
                self._epool.close()
            if pool is not None:
                pool.shutdown(wait=True)
            # Campaign processes run hundreds of sequential CodeEvolve
            # instances in one interpreter — an open WAL conn per run
            # pins frames and grows the WAL. release() disconnects only;
            # the next access lazily reconnects, so save_run and
            # post-run reads still work.
            try:
                self.db.release()
            except Exception:
                pass
            # Campaigns spawn a CodeEvolve per target per round; a live
            # ForgeEngine holds ~5-6GB of packed weights + KV buffers in
            # the caching allocator. Sleep it NOW — the old loop cleared
            # only the instance `_eng` attribute while the shared
            # `_ENGINES[key]["obj"]` entry kept the engine alive, so
            # nothing was actually freed. release_engines(fast=True)
            # frees GPU weights/graphs but keeps the cached entry for
            # a ~2-3s wake next run instead of a full rebuild.
            try:
                LLMProposer.release_engines(fast=True)
            except Exception:
                pass

        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        imp = ((self.incumbent_score - self.baseline_score) /
               (abs(self.baseline_score) + 1e-12)) * 100
        if cfg.verbose:
            try:
                rows = self.db.conn.execute(
                    "SELECT metrics_json, ok FROM code_programs "
                    "WHERE run_id=? ORDER BY id DESC LIMIT 400",
                    (self.run_id,)).fetchall()
                th = [r for r in rows
                      if '"had_think": true' in (r[0] or "")]
                nt = [r for r in rows
                      if '"had_think": false' in (r[0] or "")]
                if th or nt:
                    tok = [r for r in rows if '"gen_tokens"' in (r[0] or "")]
                    avg_tok = (sum(int(r[0].split('"gen_tokens": ')[1]
                                     .split(",")[0].split("}")[0])
                                 for r in tok) / len(tok)) if tok else 0
                    print(f"[cevolve] reasoning: think ok="
                          f"{sum(r[1] for r in th)}/{len(th)}  "
                          f"direct ok={sum(r[1] for r in nt)}/{len(nt)}  "
                          f"avg gen_tokens={avg_tok:.0f}", flush=True)
            except Exception:
                pass
            print(f"[cevolve] stopped: {self.stop_reason} — best "
                  f"{self._raw_metric():.6g} ({imp:+.1f}% vs original) "
                  f"in {self._eval_n} evals")
            if self._gen_n:
                print(f"[cevolve] proposals: {self._gen_ok}/"
                      f"{self._gen_n} gated "
                      f"({self._gen_ok / self._gen_n * 100:.0f}% valid) "
                      f"stages={self._stages}", flush=True)
        self.db.save_run(self.run_id, f"cev_{self.target.name}",
                         {"target": str(self.target.path),
                          "metric": self.metric, "goal": self.goal},
                         {"generations": self._eval_n,
                          "total_evaluations": self._eval_n,
                          "discoveries": int(sum(a[0] - 1 for a in self._arms)),
                          "best_score": self.incumbent_score,
                          "best_config": {"code": (self.incumbent_code or "")[:4000]},
                          "device": cfg.proposer_device}, t0)
        return {"run_id": self.run_id, "evals": self._eval_n,
                "best_code": self.incumbent_code,
                "best_metrics": self.incumbent_metrics,
                "baseline_score": self.baseline_score,
                "improvement_pct": imp,
                "promotions": int(sum(a[0] - 1 for a in self._arms)),
                "stop_reason": self.stop_reason,
                "proposals": {"raws": self._gen_n, "gated": self._gen_ok,
                              "valid_pct": round(
                                  self._gen_ok / max(1, self._gen_n)
                                  * 100, 1)},
                "elapsed_s": round(time.time() - t0, 1)}

    def _stage(self, s: str) -> None:
        self._stages[s] = self._stages.get(s, 0) + 1

    def _write_status(self, force: bool = False) -> None:
        """Live progress JSON — evolve_watch / GUI poll this file.
        Throttled to ~2s between writes; one line of JSON."""
        if not self.cfg.status_file:
            return
        now = time.time()
        if not force and now - self._last_status < 2.0:
            return
        self._last_status = now
        try:
            p = Path(self.cfg.status_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            # In-flight jobs — the live "what is it evaluating RIGHT NOW"
            # view. EvalPool reports worker/jid/elapsed; the legacy
            # thread-pool path reports cand hash + island only.
            inflight: list = []
            ep = getattr(self, "_epool", None)
            if ep is not None:
                try:
                    inflight = ep.in_flight()
                except Exception:
                    pass
            else:
                pend = getattr(self, "_live_pending", None) or {}
                inflight = [{"tag": c[3], "island": c[1]}
                            for c in pend.values()
                            if isinstance(c, tuple) and len(c) >= 4]
            tmp.write_text(json.dumps({
                "run_id": self.run_id, "target": self.target.name,
                "evals": self._eval_n, "dry": self._dry,
                "promos": self._promos,
                "inflight": inflight,
                "metric": self.metric, "goal": self.goal,
                "incumbent": self.incumbent_metrics.get(self.metric),
                "incumbent_score": self.incumbent_score,
                "baseline_score": self.baseline_score,
                "stage_counts": dict(self._stages),
                "proposals": {"raws": self._gen_n, "gated": self._gen_ok},
                "stop_reason": (self.stop_reason if force else None),
                "ts": now}, default=str) + "\n", encoding="utf-8")
            tmp.replace(p)
        except Exception:
            pass

    def _note_fail(self, island: int, res: dict | None = None) -> None:
        self._arms[island][1] += 1.0
        self._dry += 1
        self._write_status()

    def _log_gen(self, prompt: str, raw: str, code: str) -> None:
        """Sidecar dump of EVERY raw model emission — <db>.gens.jsonl.
        code_programs keeps the gated/extracted view; this keeps the
        exact bytes the model emitted so prompt→soup pairs are fully
        reconstructible offline."""
        if not self.cfg.log_gens:
            return
        try:
            p = Path(self.cfg.db_path).with_suffix(".gens.jsonl")
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(), "prompt": prompt[:20000],
                    "raw": (raw or "")[:20000],
                    "extracted": bool(code)}) + "\n")
        except Exception:
            pass

    def _log_program(self, code: str, island: int, gen: int,
                     code_hash: str | None, res: dict, score: float,
                     parent_hash: str | None = None,
                     prompt: str | None = None) -> None:
        with self.db._lock:
            self.db.conn.execute(
                "INSERT INTO code_programs(run_id, target, island, gen,"
                " code_hash, code, parent_hash, metrics_json, score, ok,"
                " stage, ts, prompt, adapter) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, self.target.name, island, gen,
                 code_hash or _norm_hash(code), code[:20000], parent_hash,
                 # gen rows carry gen_tokens/had_think at top level —
                 # res.get("metrics") would store {}, which is what made
                 # the end-of-run reasoning stats print dead.
                 json.dumps(res.get("metrics") or res, default=str),
                 float(score), int(bool(res.get("ok"))),
                 res.get("stage", "?"), time.time(),
                 (prompt or "")[:20000],
                 getattr(self._proposer, "last_adapter", None)
                 or getattr(self._proposer, "_last_adapter", None)))
            self.db.conn.commit()

    @staticmethod
    def _gist(code: str) -> str:
        """One-line auto-summary of an implementation for insight rows."""
        names = re.findall(r"def (\w+)|class (\w+)", code)
        names = [a or b for a, b in names][:4]
        return f"impl({','.join(names)})" if names else code.split("\n", 1)[0][:60]

    def report(self, limit: int = 15) -> str:
        rows = self.db.conn.execute(
            "SELECT island, metrics_json, score, stage, ok FROM code_programs "
            "WHERE target=? AND ok=1 ORDER BY score DESC LIMIT ?",
            (self.target.name, limit)).fetchall()
        lines = [f"{'score':>12}  {'island':>6}  metrics"]
        for isl, mj, s, stage, ok in rows:
            lines.append(f"{s:>12.4f}  {isl:>6}  {mj[:80]}")
        return "\n".join(lines)


def run_code_evolve(cfg: CodeEvolveConfig,
                    propose_fn: Callable[[str], str] | None = None) -> dict:
    return CodeEvolve(cfg, propose_fn=propose_fn).run()


def export_pairs(db_path: str, out_path: str,
                 target: str | None = None) -> int:
    """DPO/SFT pair export from code_programs.

    Every evaluated child whose parent was also evaluated writes one
    preference row: {target, delta, chosen, rejected, scores}. Positive
    delta = child outscored its own parent — the exact signal a fine-tune
    needs (chosen = improved code, rejected = its parent)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    q = ("SELECT target, code, code_hash, parent_hash, score "
         "FROM code_programs WHERE ok=1 AND code IS NOT NULL")
    rows = conn.execute(q).fetchall()
    by_hash: dict[str, tuple[str, str, float]] = {}
    best_by_target: dict[str, tuple] = {}
    for tgt, code, h, _ph, s in rows:
        if s is None:
            continue
        if h and (h not in by_hash or s > by_hash[h][2]):
            by_hash[h] = (tgt, code, float(s))
        if tgt not in best_by_target or s > best_by_target[tgt][2]:
            best_by_target[tgt] = (code, h, float(s))
    n = 0
    tmp = str(out_path) + ".tmp"          # atomic: killed runs never
    with open(tmp, "w", encoding="utf-8") as f:   # leave half a row
        for tgt, code, h, ph, s in rows:
            if target and tgt != target or s is None:
                continue
            par = by_hash.get(ph or "")
            if par is None:
                # Basin-mode parents are queued to _basin_q as mutator
                # parents — never evaluated, so no ok=1 row exists and
                # the canonical basin runs used to export ~zero rows.
                # Fall back to the target's best (≠ the child itself):
                # the pair then means "beat the champion", the same
                # verified-win signal SFT/DPO wants.
                best = best_by_target.get(tgt)
                if best and best[1] != h:
                    par = (tgt, best[0], best[2])
            if par and par[0] == tgt and float(s) > par[2]:
                f.write(json.dumps({
                    "target": tgt, "delta": float(s) - par[2],
                    "chosen": code, "rejected": par[1],
                    "chosen_score": float(s),
                    "rejected_score": par[2]}) + "\n")
                n += 1
    os.replace(tmp, out_path)
    return n


def _sft_prompt_skeleton(target_name: str, metric: str, goal: str,
                         required: list[str], parent_code: str,
                         budgets: dict | None = None) -> str:
    """Reconstruct the proposer prompt shape for rows logged before the
    prompt column existed — volatile metrics/insights lines omitted."""
    req = ", ".join(required) or "same as seed"
    bud = json.dumps(budgets or {})
    return (
        "# Rewrite this Python program to be much better.\n"
        f"# Candidate: {_norm_hash(parent_code)[:8]}\n"
        f"# Direction: {metric} "
        f"{'MINIMIZE' if goal == 'min' else 'MAXIMIZE'}\n"
        f"# Hard budgets (must satisfy): {bud}\n"
        f"# Rules: keep public names ({req}); stdlib+numpy+torch only;\n"
        "# output ONLY one python code block; identical semantics.\n"
        "#\n# Current best:\n```python\n" + parent_code.strip() +
        "\n```\n# Write the improved implementation now:\n```python\n")


def export_sft(db_path: str, out_path: str,
               targets_dir: Path | None = None) -> int:
    """SFT examples for the proposer: every evaluated child that
    outscored its own parent emits {prompt, completion}. Rows logged
    after the prompt column landed carry the real prompt; older rows
    get a template reconstruction from the parent program."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(code_programs)")}
    has_prompt = "prompt" in cols
    sel = ("SELECT target, code, code_hash, parent_hash, score"
           + (", prompt" if has_prompt else "") +
           " FROM code_programs WHERE ok=1 AND code IS NOT NULL")
    rows = conn.execute(sel).fetchall()
    by_hash: dict[str, tuple[str, str, float]] = {}
    best_by_target: dict[str, tuple] = {}
    for r in rows:
        tgt, code, h, _ph, s = r[:5]
        if s is None:
            continue
        if h and (h not in by_hash or s > by_hash[h][2]):
            by_hash[h] = (tgt, code, float(s))
        if tgt not in best_by_target or s > best_by_target[tgt][2]:
            best_by_target[tgt] = (code, h, float(s))
    spec_cache: dict[str, Target] = {}

    def _spec(tgt: str):
        if tgt not in spec_cache:
            try:
                spec_cache[tgt] = Target(Target.resolve(tgt))
            except Exception:
                spec_cache[tgt] = None
        return spec_cache[tgt]

    n = 0
    tmp = str(out_path) + ".tmp"          # atomic: killed runs never
    with open(tmp, "w", encoding="utf-8") as f:   # leave a corrupt file
        for r in rows:
            tgt, code, h, ph, s = r[:5]
            prompt = r[5] if has_prompt and len(r) > 5 else None
            if s is None:
                continue
            par = by_hash.get(ph or "")
            if par is None:
                # Basin-mode parents were never evaluated — fall back
                # to the target's best ≠ the child (see export_pairs).
                best = best_by_target.get(tgt)
                if best and best[1] != h:
                    par = (tgt, best[0], best[2])
            if not (par and par[0] == tgt and float(s) > par[2]):
                continue
            if not prompt:
                spec = _spec(tgt)
                if spec is None:
                    continue
                prompt = _sft_prompt_skeleton(
                    tgt, spec.metric, spec.goal, spec.required, par[1],
                    spec.budgets)
            elif "# Candidate:" not in prompt:
                # Row logged before the dedup-discriminator line existed —
                # inject it after line 1 so the row survives sft_train's
                # prompt[:200] dedup (parent hash is unique per chain).
                first, _, rest = prompt.partition("\n")
                prompt = (first + f"\n# Candidate: {(ph or '')[:8]}\n"
                          + rest)
            # canonical sft_train row: prompt/response. The response
            # must match the DECODE mode the prompt sets up: primed
            # prompts (tail ends in an unclosed ```python + signature)
            # teach the model to continue the BODY — emitting a fenced
            # restart as the target trains exactly the "abandons the
            # prime" failure. When the logged code carries the prime
            # verbatim as a prefix, strip it to get the true body.
            body = None
            primed_prompt = False
            if "```python" in prompt:
                tail = prompt.rsplit("```python", 1)[1]
                if "```" not in tail and tail.strip():
                    primed_prompt = True
                    ptxt = tail.strip()
                    cstr = code.strip()
                    if cstr.startswith(ptxt):
                        body = cstr[len(ptxt):].lstrip("\n")
                    else:
                        # The logged code came from a fenced RESTART
                        # path — it does not carry the prime, so no
                        # honest body-continuation target exists. Emit
                        # nothing rather than train abandon-prime.
                        continue
            response = body if body else \
                f"```python\n{code.strip()}\n```"
            f.write(json.dumps({
                "target": tgt, "prompt": prompt,
                "response": response,
                "completion": code,
                "delta": float(s) - par[2]}) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return n


_VALIDITY_REJECT_STAGES = {
    "gen", "gen_dup", "dup_batch",          # unextractable / batch dups
    "syntax", "missing", "incomplete",      # gate rejects (contract broke)
    "undefined", "dup",
}


def export_validity_pairs(db_path: str, out_path: str,
                          target: str | None = None) -> int:
    """DPO pairs that teach FORMAT compliance, not quality.

    export_pairs teaches "better than the parent"; this one teaches
    "emit a valid proposal at all": chosen = a gated program (ok=1 or
    gate-passing code), rejected = the model's own malformed/dup output
    for the same parent (fallback: same target). The rejected side is
    what the model literally emitted — soup, restarts, missing defs —
    so the gradient lands on the format contract, not on code taste.

    Row: {target, prompt, chosen, rejected, why}. `why` carries the
    reject stage so downstream tooling can weight/filter."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(code_programs)")}
    has_prompt = "prompt" in cols
    sel = ("SELECT target, code, code_hash, parent_hash, ok, stage"
           + (", prompt" if has_prompt else "")
           + " FROM code_programs WHERE code IS NOT NULL AND code != ''"
           # Best-scoring gated program wins the (target,parent) slot —
           # first-seen ordering would pair rejected soup with whatever
           # mediocre cousin happened to log first.
           + " ORDER BY score DESC")
    rows = conn.execute(sel).fetchall()
    # gated candidates keyed by (target, parent_hash) then (target,) —
    # score-desc order, so list[0] is the best sibling.
    good_by_parent: dict[tuple, list[tuple]] = {}
    good_by_target: dict[str, list[tuple]] = {}
    bad: list[tuple] = []
    for r in rows:
        tgt, code, h, ph, ok, stage = r[:6]
        prompt = r[6] if has_prompt and len(r) > 6 else None
        if target and tgt != target:
            continue
        if ok:
            ent = (code, prompt)
            if ph:
                good_by_parent.setdefault((tgt, ph), []).append(ent)
            good_by_target.setdefault(tgt, []).append(ent)
        elif stage in _VALIDITY_REJECT_STAGES:
            bad.append((tgt, code, ph, stage, prompt))

    def _pick(cands: list[tuple] | None, bad_norm: str):
        # dup-stage rows store a VALID program — often the very gated
        # one it duplicated. chosen==rejected teaches the model the same
        # text is right and wrong: pick a DIFFERENT good sibling or skip.
        for ent in cands or ():
            if _norm_hash(ent[0]) != bad_norm:
                return ent
        return None

    n = 0
    tmp = str(out_path) + ".tmp"          # atomic: killed runs never
    with open(tmp, "w", encoding="utf-8") as f:   # leave half a row
        for tgt, bad_code, ph, stage, bad_prompt in bad:
            bad_norm = _norm_hash(bad_code)
            good = (_pick(good_by_parent.get((tgt, ph)), bad_norm)
                    or _pick(good_by_target.get(tgt), bad_norm))
            if good is None:
                continue
            good_code, good_prompt = good
            prompt = bad_prompt or good_prompt
            if not prompt:
                continue
            f.write(json.dumps({
                "target": tgt,
                "prompt": prompt,
                "chosen": f"```python\n{good_code.strip()}\n```",
                "rejected": bad_code.strip()[:8000],
                "why": stage}) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return n

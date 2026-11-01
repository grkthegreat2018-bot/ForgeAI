"""Mechanism-search domain (R67) — discovers novel live-learning /
transformer mechanisms for the FluxLM dev plan.

Generalizes MicroLiveLM (R64): where that domain searched a fixed format
grid, MechSearch searches the *mechanism surface itself* — attention
variant (gqa/gta/gla), delta-rule path (none/kda/gdn2), fast-weight
rank compression, surprise-admission gating (write_threshold), bounded
substrate drift (live_projections), inner-loop forget/cap — evaluated
on the real thing: ConfigurableResearchLLM + LiveSession on CUDA.

Metrics (same scoreboard as MicroLiveLM so results are comparable):
  live_gain   fact-recall NLL delta from test-time ingest only
  decode_tps  bs=1 incremental decode
  ingest_tps  sustained live-learn throughput
  params      model size (micro budget)
  revert_exact  journaled undo parity (FluxLM-compat contract)

quick_eval = cheaper promotion tier (fewer pairs/reps/decode steps).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import BaseDomain
from .livet_domains import _pairs_stream, _SEP

V = 256
_N_PAIRS = 8
_Q_PAIRS = 4

_D_MODELS = (64, 96, 128, 192, 256, 384)
_LAYERS = (2, 3, 4, 6, 8)
_HEADS = (1, 2, 4, 8)
_FFN = (2, 3, 4)
_BLOCK_MIX = ("attn", "alt", "conv")
_ATTN = ("gqa", "gta", "gla")
_DELTA = ("none", "gdn2", "kda")
_RANKS = (0, 4, 2)          # titan_memory_rank: 0=full, else d//r
_SOLVERS = ("gd", "cg", "solve")
_ORDERS = ((4,), (8,), (4, 8), (4, 8, 16))
_CHUNKS = (16, 32, 64, 128)


class MechSearch(BaseDomain):
    """Composable mechanism discovery for live-learning LM tech."""

    # Real model builds on CUDA — same constraints as MicroLiveLM.
    eval_thread_safe = False
    cpu_eval_ok = False
    # Small eager models are spawn-worker safe (own CUDA context each).
    gpu_pool_ok = True

    def gpu_worker_kwargs(self) -> dict:
        return {"seed": self._seed, "device": "cuda"}

    def __init__(self, seq_len: int = 2048, seed: int = 42, device=None):
        super().__init__()
        if device is not None:
            self._device = torch.device(device)
        self._seed = seed
        # Structural model cache: the model is baked only by structural
        # dims — session knobs (mem_lr/solver/chunk/gates) reconfigure on
        # the SAME model. ~40% of converged-run evals share structure
        # with a recent one; model build is ~30-50% of eval cost.
        # Per-process (spawn workers each own a cache), LRU cap 4.
        self._model_cache: dict = {}
        self._cache_order: list = []

    def name(self) -> str:
        return "mech_search"

    def output_dim(self) -> int:
        return 22

    def discrete_choices(self) -> dict[str, list]:
        # 6*5*3*3*3*2*2*3*2 = 9720 cells — neural generators carry most
        # of the search; the template generator covers a useful slice.
        return {
            "d_model": list(_D_MODELS),
            "n_layers": list(_LAYERS),
            "block_mix": list(_BLOCK_MIX),
            "attn_type": list(_ATTN),
            "delta": list(_DELTA),
            "use_titan": [False, True],
            "use_ngram": [False, True],
            "titan_rank": list(_RANKS),
            "live_proj": [False, True],
        }

    def decode(self, params: torch.Tensor) -> dict:
        p = params.detach().cpu().numpy()

        def pick(i, seq):
            return seq[min(int(p[i] * (len(seq) - 1e-9)), len(seq) - 1)]

        return {
            "d_model": pick(0, _D_MODELS),
            "n_layers": pick(1, _LAYERS),
            "block_mix": pick(2, _BLOCK_MIX),
            "attn_type": pick(3, _ATTN),
            "delta": pick(4, _DELTA),
            "use_titan": bool(p[5] > 0.5),
            "titan_rank": pick(6, _RANKS),
            "solver": pick(7, _SOLVERS),
            "use_ngram": bool(p[8] > 0.5),
            "orders": pick(9, _ORDERS),
            "live_proj": bool(p[10] > 0.5),
            "n_heads": pick(11, _HEADS),
            "ffn_mult": pick(12, _FFN),
            "chunk": pick(13, _CHUNKS),
            "mem_lr": float(np.interp(p[14], [0, 1], [0.05, 8.0])),
            "ridge": float(np.interp(p[15], [0, 1], [0.01, 2.0])),
            "momentum": float(np.interp(p[16], [0, 1], [0.0, 0.95])),
            "ngram_gate": float(np.interp(p[17], [0, 1], [0.3, 2.0])),
            "write_threshold": float(np.interp(p[18], [0, 1], [0.0, 1.5])),
            "forget": float(np.interp(p[19], [0, 1], [0.90, 0.999])),
            "surprise_cap": float(np.interp(p[20], [0, 1], [1.0, 8.0])),
            "proj_lr": float(10 ** np.interp(p[21], [0, 1], [-4.0, -1.5])),
        }

    def encode(self, config: dict) -> torch.Tensor:
        def binof(val, seq):
            try:
                return seq.index(val) / max(len(seq) - 1, 1)
            except ValueError:
                return 0.5

        orders = config.get("orders", (4, 8))
        if not isinstance(orders, tuple):
            orders = tuple(orders)
        import math as _m
        return torch.tensor([
            binof(config.get("d_model", 128), _D_MODELS),
            binof(config.get("n_layers", 4), _LAYERS),
            binof(config.get("block_mix", "attn"), _BLOCK_MIX),
            binof(config.get("attn_type", "gqa"), _ATTN),
            binof(config.get("delta", "gdn2"), _DELTA),
            1.0 if config.get("use_titan", True) else 0.0,
            binof(config.get("titan_rank", 0), _RANKS),
            binof(config.get("solver", "solve"), _SOLVERS),
            1.0 if config.get("use_ngram", True) else 0.0,
            binof(orders, _ORDERS),
            1.0 if config.get("live_proj", False) else 0.0,
            binof(config.get("n_heads", 2), _HEADS),
            binof(config.get("ffn_mult", 3), _FFN),
            binof(config.get("chunk", 32), _CHUNKS),
            float(np.clip((config.get("mem_lr", 4.0) - 0.05) / 7.95, 0, 1)),
            float(np.clip((config.get("ridge", 0.1) - 0.01) / 1.99, 0, 1)),
            float(np.clip(config.get("momentum", 0.9) / 0.95, 0, 1)),
            float(np.clip((config.get("ngram_gate", 1.0) - 0.3) / 1.7, 0, 1)),
            float(np.clip(config.get("write_threshold", 0.0) / 1.5, 0, 1)),
            float(np.clip((config.get("forget", 0.99) - 0.90) / 0.099, 0, 1)),
            float(np.clip((config.get("surprise_cap", 4.0) - 1.0) / 7.0, 0, 1)),
            float(np.clip((_m.log10(config.get("proj_lr", 1e-3)) + 4.0) / 2.5, 0, 1)),
        ], dtype=torch.float32)

    def behavioral_dims(self):
        # Calibrated to observed ranges: live_gain reaches ~5.7 (was
        # clipped at 3.0 — saturated the grid, hid the new-cell rate),
        # log_params down to 5.0 (d64 models land ~5.2).
        return [("live_gain", 12, 0.0, 6.0),
                ("log_tps", 10, 1.0, 4.5),
                ("log_params", 10, 5.0, 7.5)]

    def seed_configs(self) -> list[dict]:
        # Seed around the discovered optimum (3 passes agree):
        # conv + titan(full-rank) + ngram + solve. Includes the
        # under-sampled d128 region and the conv+KDA speed corner.
        base = dict(d_model=128, n_layers=4, block_mix="conv",
                    attn_type="gqa", delta="none", use_titan=True,
                    titan_rank=0, solver="solve", use_ngram=True,
                    orders=(4, 8), live_proj=False, n_heads=2,
                    ffn_mult=3, chunk=32, mem_lr=4.0, ridge=0.1,
                    momentum=0.9, ngram_gate=1.0, write_threshold=0.0,
                    forget=0.99, surprise_cap=4.0, proj_lr=1e-3)
        return [
            base,                                              # champion
            {**base, "d_model": 96, "n_layers": 2,
             "delta": "kda"},                                  # speed king
            {**base, "d_model": 192},                          # +capacity
            {**base, "d_model": 96, "n_layers": 3,
             "delta": "kda", "live_proj": True},               # conv+drift
            {**base, "d_model": 128, "delta": "kda"},          # d128+kda
            {**base, "d_model": 128, "attn_type": "gla"},      # d128+gla
            {**base, "d_model": 192, "n_layers": 3},           # wide+shallow
            {**base, "block_mix": "alt", "d_model": 96},       # hybrid check
        ]

    # ── evaluation ──────────────────────────────────────────────────────

    def evaluate(self, config: dict) -> dict:
        return self._measure(config, quick=False)

    def quick_eval(self, config: dict) -> dict:
        return self._measure(config, quick=True)

    def _measure(self, config: dict, quick: bool) -> dict:
        from forge.model.live import LiveConfig, LiveSession, livet_config
        from forge.model.llm import ConfigurableResearchLLM
        from forge.model.ngram_mem import NGramMemory

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        L = int(config["n_layers"])
        mix = config.get("block_mix", "attn")
        if mix == "conv":
            layer_types = ["conv"] * L
        elif mix == "alt":
            layer_types = ["attention" if i % 2 == 0 else "conv"
                           for i in range(L)]
        else:
            layer_types = ["attention"] * L

        delta = config.get("delta", "gdn2")
        attn_type = config.get("attn_type", "gqa")
        rank = int(config.get("titan_rank", 0))
        d = int(config["d_model"])
        rank_cells = 0 if rank == 0 else max(1, d // rank)

        struct_key = (d, L, tuple(layer_types), attn_type, delta,
                      bool(config.get("use_titan", True)), rank_cells,
                      int(config["n_heads"]), int(config["ffn_mult"]))
        cached = self._model_cache.get(struct_key)
        if cached is not None:
            model, n_params, init_state = cached
            # Prior evals mutated live surfaces (titan fast weights,
            # gdn2 states, live_proj drift) — restore the pristine
            # build state or scores inherit cross-eval contamination.
            model.load_state_dict(init_state)
            self._cache_order.remove(struct_key)
            self._cache_order.append(struct_key)
        else:
            torch.manual_seed(self._seed)
            if dev == "cuda":
                torch.cuda.manual_seed_all(self._seed)
            try:
                cfg = livet_config(
                    vocab_size=V, d_model=d, n_layers=L,
                    n_heads=int(config["n_heads"]),
                    intermediate_size=d * int(config["ffn_mult"]),
                    layer_types=layer_types,
                    attn_type=attn_type,
                    use_gdn2=(delta == "gdn2"),
                    use_kda=(delta == "kda"),
                    use_titan_memory=bool(config.get("use_titan", True)),
                    titan_memory_rank=rank_cells,
                    device=dev, dtype="float32")
                model = ConfigurableResearchLLM(cfg).to(dev).eval()
            except Exception as e:
                return {"score": -1e9, "behavioral": (0.0, 0.0, 0.0),
                        "metadata": {"fail": f"build: {e}"}}
            n_params = sum(p.numel() for p in model.parameters())
            init_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            self._model_cache[struct_key] = (model, n_params, init_state)
            self._cache_order.append(struct_key)
            if len(self._cache_order) > 4:
                old = self._cache_order.pop(0)
                m, _, _st = self._model_cache.pop(old)
                del m, _st
                if dev == "cuda":
                    torch.cuda.empty_cache()

        try:
            s = LiveSession(model, LiveConfig(
                chunk=int(config.get("chunk", 32)),
                solver=config.get("solver"),
                mem_lr=float(config.get("mem_lr", 1.0)),
                ridge=float(config.get("ridge", 0.1)),
                momentum=float(config.get("momentum", 0.9)),
                forget=float(config.get("forget", 0.99)),
                surprise_cap=float(config.get("surprise_cap", 4.0)),
                write_threshold=float(config.get("write_threshold", 0.0)),
                live_projections=bool(config.get("live_proj", False)),
                proj_lr=float(config.get("proj_lr", 1e-3)),
                open_gates=True))
            if config.get("use_ngram", False):
                s.attach_sparse(NGramMemory(
                    V, orders=tuple(config.get("orders", (4, 8))),
                    gate=float(config.get("ngram_gate", 1.0))))
            s.freeze_substrate()

            n_pairs = _Q_PAIRS if quick else _N_PAIRS
            live_map = {10 + i: 100 + i for i in range(n_pairs)}
            probe = _pairs_stream(live_map, reps=1).to(dev)
            val_idx = torch.arange(probe.numel()) % 3 == 2

            def fact_loss(s) -> float:
                with torch.no_grad():
                    logits = s.model(probe[:-1].unsqueeze(0))[0]
                    if s.sparse is not None:
                        logits = logits + s.sparse.lift(probe[:-1].unsqueeze(0))
                    ce = F.cross_entropy(logits[0].float(), probe[1:],
                                         reduction="none")
                return float(ce[val_idx[1:]].mean())

            # Journaled-undo parity (cheap; catches surface-contract breaks)
            s.reset()
            mark = s.mark()
            nll_pre = s.eval_nll(probe)
            s.ingest(probe)
            s.revert(mark)
            revert_exact = abs(s.eval_nll(probe) - nll_pre) < 1e-5

            # Live-learning measurement
            s.reset()
            loss_before = fact_loss(s)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            learn_ids = _pairs_stream(live_map,
                                      reps=3 if quick else 5).to(dev)
            s.ingest(learn_ids)
            if dev == "cuda":
                torch.cuda.synchronize()
            ingest_tps = learn_ids.numel() / max(time.time() - t0, 1e-9)
            s.reset()
            loss_after = fact_loss(s)
            live_gain = loss_before - loss_after

            # bs=1 decode speed
            steps = 8 if quick else 16
            n_pref = min(16, probe.numel() - 1)
            with torch.no_grad():
                ids = probe[:n_pref].unsqueeze(0)
                out = model(ids, use_cache=True)
                past = out[2]
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                nxt = probe[n_pref:n_pref + 1].reshape(1, 1)
                for _ in range(steps):
                    out = model(nxt, past_key_values=past, use_cache=True)
                    past = out[2]
                    nxt = torch.randint(0, V, (1, 1), device=dev)
                if dev == "cuda":
                    torch.cuda.synchronize()
                decode_tps = steps / max(time.time() - t0, 1e-9)
        except Exception as e:
            return {"score": -1e9, "behavioral": (0.0, 0.0, 0.0),
                    "metadata": {"fail": f"eval: {e}",
                                 "params": int(n_params)}}
        finally:
            try:
                del model, s
            except Exception:
                pass
            if dev == "cuda":
                torch.cuda.empty_cache()

        if not math.isfinite(live_gain):
            return {"score": -1e9, "behavioral": (0.0, 0.0, 0.0),
                    "metadata": {"fail": "nonfinite",
                                 "params": int(n_params)}}

        log_dec = math.log10(max(decode_tps, 1.0))
        log_ing = math.log10(max(ingest_tps, 1.0))
        log_par = math.log10(max(n_params, 1))
        size_over = max(0.0, log_par - math.log10(4e6))

        score = (8.0 * live_gain
                 + 2.0 * log_dec
                 + 1.0 * log_ing
                 + 2.0 * float(revert_exact)
                 - 3.0 * size_over)
        return {"score": float(score),
                "behavioral": (float(np.clip(live_gain, 0, 6)),
                               float(np.clip(log_dec, 1, 4.5)),
                               float(np.clip(log_par, 5.0, 7.5))),
                "metadata": {
                    "params": int(n_params),
                    "loss_before": loss_before, "loss_after": loss_after,
                    "live_gain": live_gain,
                    "decode_tps": decode_tps, "ingest_tps": ingest_tps,
                    "revert_exact": bool(revert_exact),
                    "quick": bool(quick),
                    "attn_type": attn_type, "delta": delta,
                    "live_proj": bool(config.get("live_proj", False)),
                    "write_threshold": float(
                        config.get("write_threshold", 0.0)),
                }}

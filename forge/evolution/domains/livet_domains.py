"""Live-format micro-LM evolution domain (R64).

Search target: a NEW model format for micro transformers — the LiveT/
FluxLM synthesis. Candidates are complete format specs: substrate dims +
block mix + which journaled live surfaces exist (GDN2 recurrent state,
TitanMemory fast weights, exact-suffix NGramMemory sidecar) + the knobs
that govern live writes.

Unlike the synthetic-tensor domains, evaluate() builds the REAL model
(ConfigurableResearchLLM + LiveSession on CUDA) and measures:
  - live_gain: fact-recall NLL delta from test-time ingest only
    (frozen substrate, journaled buffers — the FluxLM live contract)
  - decode_tps: bs=1 incremental decode speed (launch-bound honesty)
  - ingest_tps: sustained live-learn throughput (writes included)
  - params: model size (the "micro" budget)
  - revert_exact: journaled undo parity (FluxLM-compat requirement)

Bootstrap is skipped inside evaluate() (dominant cost; live-write signal
is measurable on a random substrate — probe precedent). Winners should be
re-validated with bootstrap in focused tests.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import BaseDomain

V = 256
_SEP = 200
_N_PAIRS = 8

_D_MODELS = (64, 96, 128, 192, 256, 384)
_LAYERS = (2, 3, 4, 6, 8)
_HEADS = (1, 2, 4, 8)
_FFN = (2, 3, 4)
_SOLVERS = ("gd", "cg", "solve")
_ORDERS = ((4,), (8,), (4, 8), (4, 8, 16))
_CHUNKS = (16, 32, 64, 128)
_BLOCK_MIX = ("attn", "alt", "conv")


def _pairs_stream(pair_map: dict[int, int], reps: int) -> torch.Tensor:
    seq: list[int] = []
    for _ in range(reps):
        for k, v in pair_map.items():
            seq += [_SEP, k, v]
    return torch.tensor(seq)


class MicroLiveLM(BaseDomain):
    """Micro live-learning LM format search (the 'new format' domain)."""

    def __init__(self, seq_len: int = 2048, seed: int = 42, device=None):
        super().__init__()
        if device is not None:
            self._device = torch.device(device)
        self._seed = seed

    def name(self) -> str:
        return "micro_live_lm"

    def output_dim(self) -> int:
        return 15

    def discrete_choices(self) -> dict[str, list]:
        # Structural grid: 6*5*2*2*2 = 240 cells — fully enumerated by the
        # TemplateGenerator in ~2 generations (the brute-force sweep).
        return {
            "d_model": list(_D_MODELS),
            "n_layers": list(_LAYERS),
            "use_gdn2": [False, True],
            "use_titan": [False, True],
            "use_ngram": [False, True],
        }

    def decode(self, params: torch.Tensor) -> dict:
        p = params.detach().cpu().numpy()

        def pick(i, seq):
            return seq[min(int(p[i] * (len(seq) - 1e-9)), len(seq) - 1)]

        return {
            "d_model": pick(0, _D_MODELS),
            "n_layers": pick(1, _LAYERS),
            "block_mix": pick(2, _BLOCK_MIX),
            "n_heads": pick(3, _HEADS),
            "ffn_mult": pick(4, _FFN),
            "use_gdn2": bool(p[5] > 0.5),
            "use_titan": bool(p[6] > 0.5),
            "solver": pick(7, _SOLVERS),
            "use_ngram": bool(p[8] > 0.5),
            "orders": pick(9, _ORDERS),
            "chunk": pick(10, _CHUNKS),
            "mem_lr": float(np.interp(p[11], [0, 1], [0.05, 8.0])),
            "ridge": float(np.interp(p[12], [0, 1], [0.01, 2.0])),
            "momentum": float(np.interp(p[13], [0, 1], [0.0, 0.95])),
            "ngram_gate": float(np.interp(p[14], [0, 1], [0.3, 2.0])),
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
        return torch.tensor([
            binof(config.get("d_model", 128), _D_MODELS),
            binof(config.get("n_layers", 4), _LAYERS),
            binof(config.get("block_mix", "attn"), _BLOCK_MIX),
            binof(config.get("n_heads", 2), _HEADS),
            binof(config.get("ffn_mult", 3), _FFN),
            1.0 if config.get("use_gdn2", True) else 0.0,
            1.0 if config.get("use_titan", True) else 0.0,
            binof(config.get("solver", "solve"), _SOLVERS),
            1.0 if config.get("use_ngram", True) else 0.0,
            binof(orders, _ORDERS),
            binof(config.get("chunk", 32), _CHUNKS),
            float(np.clip((config.get("mem_lr", 4.0) - 0.05) / 7.95, 0, 1)),
            float(np.clip((config.get("ridge", 0.1) - 0.01) / 1.99, 0, 1)),
            float(np.clip(config.get("momentum", 0.9) / 0.95, 0, 1)),
            float(np.clip((config.get("ngram_gate", 1.0) - 0.3) / 1.7, 0, 1)),
        ], dtype=torch.float32)

    def behavioral_dims(self):
        # live_gain reaches ~5.7 on this eval — 0-3 clipping saturated
        # the top bin and masked real coverage (found on the mech run).
        return [("live_gain", 12, 0.0, 6.0),
                ("log_tps", 10, 1.0, 4.5),
                ("log_params", 10, 5.0, 7.5)]

    def seed_configs(self) -> list[dict]:
        return [
            # LiveT-validated hybrid recipe (the prior art point)
            dict(d_model=128, n_layers=4, block_mix="attn", n_heads=2,
                 ffn_mult=3, use_gdn2=True, use_titan=True, solver="solve",
                 use_ngram=True, orders=(4, 8), chunk=32, mem_lr=4.0,
                 ridge=0.1, momentum=0.9, ngram_gate=1.0),
            # Exact-suffix-only (FLUX-minimal)
            dict(d_model=96, n_layers=2, block_mix="attn", n_heads=2,
                 ffn_mult=3, use_gdn2=False, use_titan=False, solver="gd",
                 use_ngram=True, orders=(4,), chunk=32, mem_lr=1.0,
                 ridge=0.1, momentum=0.9, ngram_gate=1.0),
            # Fuzzy-only (Titans-style, no exact store)
            dict(d_model=128, n_layers=4, block_mix="attn", n_heads=2,
                 ffn_mult=3, use_gdn2=True, use_titan=True, solver="solve",
                 use_ngram=False, orders=(4, 8), chunk=32, mem_lr=4.0,
                 ridge=0.1, momentum=0.9, ngram_gate=1.0),
            # Bare substrate (no live surface) — the floor
            dict(d_model=128, n_layers=4, block_mix="attn", n_heads=2,
                 ffn_mult=3, use_gdn2=False, use_titan=False, solver="gd",
                 use_ngram=False, orders=(4,), chunk=32, mem_lr=1.0,
                 ridge=0.1, momentum=0.9, ngram_gate=1.0),
            # Micro edge cases
            dict(d_model=64, n_layers=2, block_mix="attn", n_heads=1,
                 ffn_mult=2, use_gdn2=True, use_titan=True, solver="solve",
                 use_ngram=True, orders=(4, 8), chunk=16, mem_lr=4.0,
                 ridge=0.1, momentum=0.9, ngram_gate=1.0),
            dict(d_model=384, n_layers=8, block_mix="attn", n_heads=8,
                 ffn_mult=4, use_gdn2=True, use_titan=True, solver="solve",
                 use_ngram=True, orders=(4, 8, 16), chunk=64, mem_lr=2.0,
                 ridge=0.1, momentum=0.9, ngram_gate=1.0),
        ]

    # ── evaluation ──────────────────────────────────────────────────────

    def evaluate(self, config: dict) -> dict:
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

        torch.manual_seed(0)
        cfg = livet_config(
            vocab_size=V, d_model=int(config["d_model"]), n_layers=L,
            n_heads=int(config["n_heads"]),
            intermediate_size=int(config["d_model"]) * int(config["ffn_mult"]),
            layer_types=layer_types,
            use_gdn2=bool(config.get("use_gdn2", True)),
            use_titan_memory=bool(config.get("use_titan", True)),
            device=dev, dtype="float32")
        model = ConfigurableResearchLLM(cfg).to(dev).eval()
        n_params = sum(p.numel() for p in model.parameters())

        s = LiveSession(model, LiveConfig(
            chunk=int(config.get("chunk", 32)),
            solver=config.get("solver"),
            mem_lr=float(config.get("mem_lr", 1.0)),
            ridge=float(config.get("ridge", 0.1)),
            momentum=float(config.get("momentum", 0.9)),
            open_gates=True))
        if config.get("use_ngram", False):
            s.attach_sparse(NGramMemory(
                V, orders=tuple(config.get("orders", (4, 8))),
                gate=float(config.get("ngram_gate", 1.0))))
        s.freeze_substrate()

        live_map = {10 + i: 100 + i for i in range(_N_PAIRS)}
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

        # Journaled-undo parity check (small probe — cheap, catches any
        # live surface that writes outside the journal contract).
        s.reset()
        mark = s.mark()
        nll_pre = s.eval_nll(probe)
        s.ingest(probe)
        s.revert(mark)
        revert_exact = abs(s.eval_nll(probe) - nll_pre) < 1e-5

        # Live-learning measurement: verbatim fact recall, cold vs ingest.
        s.reset()
        loss_before = fact_loss(s)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        learn_ids = _pairs_stream(live_map, reps=5).to(dev)
        s.ingest(learn_ids)
        if dev == "cuda":
            torch.cuda.synchronize()
        ingest_tps = learn_ids.numel() / max(time.time() - t0, 1e-9)
        s.reset()                       # persistent mem stays; work clears
        loss_after = fact_loss(s)
        live_gain = loss_before - loss_after

        # bs=1 incremental decode speed (prefill 16 + 16 single steps).
        with torch.no_grad():
            if dev == "cuda":
                torch.cuda.synchronize()
            ids = probe[:16].unsqueeze(0)
            model(ids, use_cache=True)
            torch.cuda.synchronize() if dev == "cuda" else None
            t0 = time.time()
            out = model(ids, use_cache=True)
            past = out[2]
            steps = 16
            nxt = probe[16:17].reshape(1, 1)
            for _ in range(steps):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out[2]
                nxt = torch.randint(0, V, (1, 1), device=dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            decode_tps = steps / max(time.time() - t0, 1e-9)

        del model, s
        if dev == "cuda":
            torch.cuda.empty_cache()

        if not math.isfinite(live_gain):
            return {"score": -1e9, "behavioral": (0.0, 0.0, 0.0),
                    "metadata": {"fail": "nonfinite"}}

        log_dec = math.log10(max(decode_tps, 1.0))
        log_ing = math.log10(max(ingest_tps, 1.0))
        log_par = math.log10(max(n_params, 1))
        # Micro budget: free up to 4M params, quadratic log penalty above.
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
                }}

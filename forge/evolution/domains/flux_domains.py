"""FluxLM self-tuning evolution domain (R65).

Search target: the FluxConfig knob surface itself — ~40 knobs over
order tables, retrieval channels, gen channels (gcell/gsyn/gsyn_deep/
dnet), hedge mixture, write/read gates, and capacity budgets.

evaluate() builds the REAL FluxLM (cuda_primary when CUDA is up) and
measures, on real text + planted fact chains:
  - held_nll : held-out next-byte NLL after live ingest (generation
               quality — better mixture -> better samples)
  - hop1_nll : verbatim fact recall (planted #K>V records)
  - hop2_nll : TWO-HOP recall — P(K[i+2] | "#K[i]>") needs rollout/
               episodic lookahead, not bare suffix match (reasoning)
  - pred_tps : eager push+predict throughput (live predict path)
  - ing_tps  : bulk ingest throughput
  - gen_tps  : argmax decode throughput
  - mem_mb   : memory_report() total + CUDA peak
  - diversity: distinct-4gram ratio of an argmax continuation
               (repeat-collapse detector — the "NaCl. NaCl." failure)

Score = weighted log2 ratios vs the PRODUCTION default config (measured
once, cached): baseline scores ~25, each doubling of a weighted
objective adds its weight. Behavioral dims put speed/mem/quality/
reasoning specialists into different MAP-Elites cells.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from . import BaseDomain

V = 256
_CORPUS_PATH = Path(r"D:\windsurf\ForgeAI\.devin\test_corpus.txt")
_TRAIN_BYTES = 24576
_HELD_BYTES = 1500
_HELD_STRIDE = 2                 # ~750 predict positions
_GEN_STEPS = 48
_N_FACTS = 12
# Quick (promotion) tier — ~3x cheaper: less train, fewer held
# positions, no decode probe. Ranking signal only.
_Q_TRAIN = 8192
_Q_HELD = 800
_Q_STRIDE = 4                    # ~200 predict positions

# ── discrete / pick axes ──
_ORDERS_SETS = (
    (3, 4, 6, 8),                                # tiny-short
    (4, 8, 16),                                  # tiny
    (3, 4, 6, 8, 12, 16, 24, 32),                # mid
    (3, 4, 6, 8, 12, 16, 24, 32, 48, 64),        # full (production)
    (8, 16, 24, 32, 48, 64),                     # long-only
    (2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64),  # dense
)
_GEN_FAMILIES = ("none", "gcell", "gsyn", "gsyn_deep", "dnet")
_TOPIC_DIMS = (96, 128, 192, 256, 384)
_CSEM_DIMS = (32, 64, 96, 128)
_CSEM_ORDERS = (2, 3, 4, 6, 8)
_DEEP_ORDERS = (2, 3, 4, 6, 8)
_DEEP_HITS = (8, 16, 24, 32, 48)
_EPI_ROWS = (8, 16, 32, 64)
_KNN_KS = (4, 8, 12, 16, 24)
# mem presets: (gpu_table_slots, epi_slots, max_cells_per_order,
#               knn_capacity, ring_capacity, epi_total_cap,
#               journal_cap, vec_delta_cap, csem_slots)
_MEM_PRESETS = (
    (1 << 15, 1 << 15, 40_000, 1 << 15, 1 << 18, 300_000,
     50_000, 8_192, 1 << 15),                     # small
    (1 << 17, 1 << 16, 80_000, 1 << 17, 1 << 20, 1_000_000,
     120_000, 32_768, 1 << 17),                  # medium
    (1 << 19, 1 << 18, 150_000, 1 << 19, 1 << 20, 3_000_000,
     200_000, 65_536, 1 << 19),                  # production
)
# speed presets: (fast_ingest, adaptive_read, write_gate, surprise_gate)
_SPEED_PRESETS = (
    (False, False, False, False),                # all-off (max cost)
    (True, False, False, True),                  # ingest+gated writes
    (True, True, True, True),                    # production
    (True, True, False, True),                   # read-gate, epi kept
)
_BLOOM = (0, 2, 3, 4)
_AUX_TAPS = ("off", "default", "wide")
_FAM_DIMS = (64, 128, 256)
_HEAD_HIDDEN = (0, 64, 128)


def _load_corpus() -> bytes:
    if _CORPUS_PATH.exists():
        return _CORPUS_PATH.read_bytes()
    # Fallback: deterministic synthetic prose-ish bytes
    rng = np.random.RandomState(0)
    words = [b"the", b"of", b"a", b"model", b"memory", b"learns",
             b"token", b"stream", b"context", b"predicts"]
    out = bytearray()
    for _ in range(4000):
        out += rng.choice(words) + b" "
    return bytes(out)


def _fact_streams() -> tuple[bytes, list[tuple[bytes, int, bytes, int]]]:
    """Planted chain facts: '#K{i}>V{i}\\n' with V{i} = K{i+1}.

    Returns (corpus_append, probes) where probes =
    (ctx1, target1, ctx2, target2):
      ctx1='#K{i}>' -> target1=V{i}[0]        (hop-1 recall)
      ctx2='#V{i}>' -> target2=V{i+1}[0]      (2-hop chain recall:
        follow the value from hop-1 as the key of the next record —
        requires episodic/rollout association, not same-timestep NLL)
    """
    keys = [bytes([230 + i, 96 + i, 60 + i]) for i in range(_N_FACTS + 2)]
    recs, probes = [], []
    for i in range(_N_FACTS):
        val = keys[i + 1] if i + 1 < len(keys) else b"\x4d\x4e\x4f"
        nxt = keys[i + 2] if i + 2 < len(keys) else b"\x4d\x4e\x4f"
        recs.append(b"#" + keys[i] + b">" + val + b"\n")
        probes.append((b"#" + keys[i] + b">", int(val[0]),
                       b"#" + val + b">", int(nxt[0])))
    return b"".join(recs), probes


class FluxTune(BaseDomain):
    """Brute-force the FluxConfig knob surface against real metrics."""

    # FluxLM evals build a full GPU model each — concurrent evals on
    # side streams would race CUDA-graph captures (one device, one
    # active capture) and stack multi-GB table allocs. Serial only.
    eval_thread_safe = False
    # Never send to the CPU worker pool — evals are GPU-bound.
    cpu_eval_ok = False
    # Spawned GPU-worker processes ARE safe: evals are eager +
    # graphless per candidate, each worker owns its CUDA context.
    gpu_pool_ok = True

    def gpu_worker_kwargs(self) -> dict:
        return {"seed": self._seed, "device": "cuda"}

    def __init__(self, seed: int = 42, device=None):
        super().__init__()
        if device is not None:
            self._device = torch.device(device)
        self._seed = seed
        corpus = _load_corpus()
        self._corpus = corpus
        facts, self._probes = _fact_streams()
        self._facts = facts
        self._train = corpus[:_TRAIN_BYTES] + facts
        self._held = corpus[_TRAIN_BYTES:_TRAIN_BYTES + _HELD_BYTES]
        self._prompt = list(corpus[12000:12032])
        self._baseline: dict | None = None
        self._baseline_quick: dict | None = None

    def name(self) -> str:
        return "flux_tune"

    def output_dim(self) -> int:
        return 42

    def discrete_choices(self) -> dict[str, list]:
        # 6*5*2 = 60 structural cells — enumerated by TemplateGenerator
        return {
            "orders_idx": list(range(len(_ORDERS_SETS))),
            "gen_family": list(_GEN_FAMILIES),
            "use_knn": [False, True],
        }

    def decode(self, params: torch.Tensor) -> dict:
        p = params.detach().cpu().numpy()

        def pick(i, seq):
            return seq[min(int(p[i] * (len(seq) - 1e-9)), len(seq) - 1)]

        def interp(i, lo, hi):
            return float(np.interp(np.clip(p[i], 0, 1), [0, 1], [lo, hi]))

        cfg = {
            "orders": pick(0, _ORDERS_SETS),
            "gen_family": pick(1, _GEN_FAMILIES),
            "use_knn": bool(p[2] > 0.5),
            "use_sem": bool(p[3] > 0.5),
            "use_csem": bool(p[4] > 0.5),
            "use_backoff": bool(p[5] > 0.5),
            "topic_dim": pick(6, _TOPIC_DIMS),
            "csem_dim": pick(7, _CSEM_DIMS),
            "csem_order": pick(8, _CSEM_ORDERS),
            "deep_order": pick(9, _DEEP_ORDERS),
            "deep_max_hits": pick(10, _DEEP_HITS),
            "epi_row": pick(11, _EPI_ROWS),
            "knn_k": pick(12, _KNN_KS),
            "knn_gamma": interp(13, 1.0, 4.0),
            "knn_min_sim": interp(14, 0.0, 0.3),
            "hedge_share": interp(15, 0.0, 0.02),
            "hedge_discount": interp(16, 0.99, 1.0),
            "backoff_kappa": interp(17, 2.0, 32.0),
            "backoff_order_sharp": interp(18, 0.0, 1.5),
            "alpha": interp(19, 0.05, 1.0),
            "uni_alpha": interp(20, 0.01, 0.2),
            "uni_escape": interp(21, 0.0, 0.25),
            "thin_boost": interp(22, 0.0, 0.6),
            "sleeping_experts": pick(23, (0, 1, 2)),
            "cond_hedge": pick(24, (0, 2)),
            "ppm_escape_c": bool(p[25] > 0.5),
            "ppm_excl": bool(p[26] > 0.5),
            "evidence_lift": bool(p[27] > 0.5),
            "lift_clip": interp(28, 4.0, 24.0),
            "rollout_w": interp(29, 0.0, 1.2),
            "gen_recall": interp(30, 0.0, 0.8),
            "mem_preset": pick(31, (0, 1, 2)),
            "speed_preset": pick(32, (0, 1, 2, 3)),
            "bloom_admit": pick(33, _BLOOM),
            "aux_taps": pick(34, _AUX_TAPS),
            "fam_w": interp(35, 0.2, 1.0),
            "fam_dim": pick(36, _FAM_DIMS),
            "fam_beta": interp(37, 0.2, 0.9),
            "head_hidden": pick(38, _HEAD_HIDDEN),
            "head_w": interp(39, 0.0, 0.6),
            "r_live": bool(p[40] > 0.5),
            "r_eta": interp(41, 0.01, 0.15),
        }
        return cfg

    def encode(self, config: dict) -> torch.Tensor:
        def binof(val, seq):
            try:
                return seq.index(val) / max(len(seq) - 1, 1)
            except (ValueError, TypeError):
                return 0.5

        def un(val, lo, hi):
            return float(np.clip((float(val) - lo) / max(hi - lo, 1e-9),
                                 0, 1))

        c = config
        return torch.tensor([
            binof(tuple(c.get("orders", _ORDERS_SETS[3])), _ORDERS_SETS),
            binof(c.get("gen_family", "none"), _GEN_FAMILIES),
            1.0 if c.get("use_knn", True) else 0.0,
            1.0 if c.get("use_sem", True) else 0.0,
            1.0 if c.get("use_csem", True) else 0.0,
            1.0 if c.get("use_backoff", True) else 0.0,
            binof(c.get("topic_dim", 192), _TOPIC_DIMS),
            binof(c.get("csem_dim", 64), _CSEM_DIMS),
            binof(c.get("csem_order", 4), _CSEM_ORDERS),
            binof(c.get("deep_order", 4), _DEEP_ORDERS),
            binof(c.get("deep_max_hits", 24), _DEEP_HITS),
            binof(c.get("epi_row", 32), _EPI_ROWS),
            binof(c.get("knn_k", 8), _KNN_KS),
            un(c.get("knn_gamma", 2.0), 1.0, 4.0),
            un(c.get("knn_min_sim", 0.05), 0.0, 0.3),
            un(c.get("hedge_share", 0.0), 0.0, 0.02),
            un(c.get("hedge_discount", 1.0), 0.99, 1.0),
            un(c.get("backoff_kappa", 8.0), 2.0, 32.0),
            un(c.get("backoff_order_sharp", 0.0), 0.0, 1.5),
            un(c.get("alpha", 0.4), 0.05, 1.0),
            un(c.get("uni_alpha", 0.05), 0.01, 0.2),
            un(c.get("uni_escape", 0.12), 0.0, 0.25),
            un(c.get("thin_boost", 0.2), 0.0, 0.6),
            binof(c.get("sleeping_experts", 0), (0, 1, 2)),
            binof(c.get("cond_hedge", 0), (0, 2)),
            1.0 if c.get("ppm_escape_c", False) else 0.0,
            1.0 if c.get("ppm_excl", False) else 0.0,
            1.0 if c.get("evidence_lift", True) else 0.0,
            un(c.get("lift_clip", 12.0), 4.0, 24.0),
            un(c.get("rollout_w", 0.0), 0.0, 1.2),
            un(c.get("gen_recall", 0.0), 0.0, 0.8),
            binof(c.get("mem_preset", 2), (0, 1, 2)),
            binof(c.get("speed_preset", 2), (0, 1, 2, 3)),
            binof(c.get("bloom_admit", 0), _BLOOM),
            binof(c.get("aux_taps", "default"), _AUX_TAPS),
            un(c.get("fam_w", 0.5), 0.2, 1.0),
            binof(c.get("fam_dim", 128), _FAM_DIMS),
            un(c.get("fam_beta", 0.5), 0.2, 0.9),
            binof(c.get("head_hidden", 0), _HEAD_HIDDEN),
            un(c.get("head_w", 0.0), 0.0, 0.6),
            1.0 if c.get("r_live", True) else 0.0,
            un(c.get("r_eta", 0.05), 0.01, 0.15),
        ], dtype=torch.float32)

    def behavioral_dims(self):
        return [("held_nll", 8, 0.0, 6.0),
                ("hop2_nll", 8, 0.0, 6.0),
                ("log_pred_tps", 8, 0.0, 4.0),
                ("log_mem_mb", 8, -1.0, 3.5)]

    def seed_configs(self) -> list[dict]:
        base = self.decode(self._mid())
        prod = dict(base)          # production recipe = defaults
        lean = dict(base, mem_preset=0, speed_preset=2,
                    orders=_ORDERS_SETS[0], gen_family="none",
                    use_knn=False, use_sem=False, use_csem=False)
        memhug = dict(base, mem_preset=0, orders=_ORDERS_SETS[1])
        genstk = dict(base, gen_family="gsyn_deep", fam_dim=128,
                      fam_w=0.5, head_hidden=64, head_w=0.4,
                      r_live=True, gen_recall=0.5)
        reason = dict(base, gen_family="dnet", rollout_w=1.0,
                      cond_hedge=2, deep_max_hits=48,
                      backoff_order_sharp=1.0)
        return [prod, lean, memhug, genstk, reason]

    def _mid(self) -> torch.Tensor:
        """Param vector that decodes to production defaults."""
        v = np.zeros(42)
        v[0] = (3 + 0.5) / len(_ORDERS_SETS)          # full orders
        v[1] = 0.05                                    # gen_family none
        for i in (2, 3, 4, 5):                         # knn/sem/csem/backoff
            v[i] = 0.9
        v[6] = (2 + 0.5) / len(_TOPIC_DIMS)            # 192
        v[7] = (1 + 0.5) / len(_CSEM_DIMS)             # 64
        v[8] = (3 + 0.5) / len(_CSEM_ORDERS)           # 4
        v[9] = (3 + 0.5) / len(_DEEP_ORDERS)           # 4
        v[10] = (2 + 0.5) / len(_DEEP_HITS)            # 24
        v[11] = (2 + 0.5) / len(_EPI_ROWS)             # 32
        v[12] = (1 + 0.5) / len(_KNN_KS)               # 8
        v[13] = (2.0 - 1.0) / 3.0                      # knn_gamma 2.0
        v[14] = 0.05 / 0.3                             # knn_min_sim
        v[15] = 0.0                                    # hedge_share
        v[16] = 1.0                                    # hedge_discount
        v[17] = (8.0 - 2.0) / 30.0                     # backoff_kappa
        v[18] = 0.0                                    # order_sharp
        v[19] = (0.4 - 0.05) / 0.95                    # alpha
        v[20] = (0.05 - 0.01) / 0.19                   # uni_alpha
        v[21] = 0.12 / 0.25                            # uni_escape
        v[22] = 0.2 / 0.6                              # thin_boost
        v[23] = 0.1                                    # sleeping_experts 0
        v[24] = 0.1                                    # cond_hedge 0
        v[25] = 0.1; v[26] = 0.1; v[27] = 0.9          # ppm off, evidence on
        v[28] = (12.0 - 4.0) / 20.0                    # lift_clip
        v[29] = 0.0; v[30] = 0.0                       # rollout/gen_recall
        v[31] = (2 + 0.5) / 3                          # mem production
        v[32] = (2 + 0.5) / 4                          # speed production
        v[33] = 0.1                                    # bloom off
        v[34] = (1 + 0.5) / 3                          # aux default
        v[35] = (0.5 - 0.2) / 0.8                      # fam_w
        v[36] = (1 + 0.5) / 3                          # fam_dim 128
        v[37] = (0.5 - 0.2) / 0.7                      # fam_beta
        v[38] = 0.1                                    # head_hidden 0
        v[39] = 0.0                                    # head_w
        v[40] = 0.9                                    # r_live on
        v[41] = (0.05 - 0.01) / 0.14                   # r_eta
        return torch.tensor(np.clip(v, 0, 1), dtype=torch.float32)

    # ── FluxConfig assembly ─────────────────────────────────────────────

    def _to_flux_kwargs(self, c: dict) -> dict:
        mem = _MEM_PRESETS[int(c["mem_preset"])]
        spd = _SPEED_PRESETS[int(c["speed_preset"])]
        aux = c["aux_taps"]
        kw: dict = {
            "vocab_size": V,
            "orders": tuple(c["orders"]),
            "topic_dim": int(c["topic_dim"]),
            "use_sem": bool(c["use_sem"]),
            "use_csem": bool(c["use_csem"]),
            "use_knn": bool(c["use_knn"]),
            "use_backoff": bool(c["use_backoff"]),
            "csem_dim": int(c["csem_dim"]),
            "csem_order": int(c["csem_order"]),
            "deep_order": int(c["deep_order"]),
            "deep_max_hits": int(c["deep_max_hits"]),
            "epi_row": int(c["epi_row"]),
            "knn_k": int(c["knn_k"]),
            "knn_gamma": float(c["knn_gamma"]),
            "knn_min_sim": float(c["knn_min_sim"]),
            "hedge_share": float(c["hedge_share"]),
            "hedge_discount": float(c["hedge_discount"]),
            "backoff_kappa": float(c["backoff_kappa"]),
            "backoff_order_sharp": float(c["backoff_order_sharp"]),
            "alpha": float(c["alpha"]),
            "uni_alpha": float(c["uni_alpha"]),
            "uni_escape": float(c["uni_escape"]),
            "thin_boost": float(c["thin_boost"]),
            "sleeping_experts": int(c["sleeping_experts"]),
            "cond_hedge": int(c["cond_hedge"]),
            "ppm_escape_c": bool(c["ppm_escape_c"]),
            "ppm_excl": bool(c["ppm_excl"]),
            "evidence_lift": bool(c["evidence_lift"]),
            "lift_clip": float(c["lift_clip"]),
            "rollout_w": float(c["rollout_w"]),
            "gen_recall": float(c["gen_recall"]),
            "head_hidden": int(c["head_hidden"]),
            "head_w": float(c["head_w"]) if c["head_hidden"] else 0.0,
            "r_live": bool(c["r_live"]),
            "r_eta": float(c["r_eta"]),
            "bloom_admit": int(c["bloom_admit"]),
            "fast_ingest": bool(spd[0]),
            "adaptive_read": bool(spd[1]),
            "write_gate": bool(spd[2]),
            "surprise_gate": bool(spd[3]),
            "gpu_table_slots": int(mem[0]),
            "epi_slots": int(mem[1]),
            "max_cells_per_order": int(mem[2]),
            "knn_capacity": int(mem[3]),
            "ring_capacity": int(mem[4]),
            "epi_total_cap": int(mem[5]),
            "journal_cap": int(mem[6]),
            "vec_delta_cap": int(mem[7]),
            "csem_slots": int(mem[8]),
            # gen family expansion
            "gcell": c["gen_family"] == "gcell",
            "gsyn": c["gen_family"] == "gsyn",
            "gsyn_deep": c["gen_family"] == "gsyn_deep",
            "dnet": c["gen_family"] == "dnet",
            "gcell_w": float(c["fam_w"]),
            "gsyn_w": float(c["fam_w"]),
            "gsyn_deep_w": float(c["fam_w"]),
            "dnet_w": float(c["fam_w"]),
            "gsyn_dim": int(c["fam_dim"]),
            "dnet_dk": int(c["fam_dim"]),
            "gsyn_deep_hidden": int(c["fam_dim"]),
            "gsyn_beta": float(c["fam_beta"]),
            "dnet_beta": float(c["fam_beta"]),
            "gsyn_deep_lr": float(np.clip(c["fam_beta"] * 0.1, 0.01, 0.08)),
            # per-candidate inductor compile inside graph captures
            # corrupts RNG capture state (and costs ~30s/candidate) —
            # eager chain for evals; production can re-enable.
            "compile_dev": False,
        }
        if aux == "off":
            kw.update(sparse_taps=(), class_orders=(), hybrid_taps=(),
                      word_orders=())
        elif aux == "wide":
            kw.update(word_orders=(2, 3))
        return kw

    # ── evaluation ──────────────────────────────────────────────────────

    def _measure(self, flux_kwargs: dict, quick: bool = False) -> dict:
        from forge.model.flux import FluxLM, FluxConfig

        cuda = torch.cuda.is_available()
        # Deterministic init: identical configs must measure identically,
        # otherwise baseline-vs-candidate deltas carry init-RNG noise.
        torch.manual_seed(self._seed)
        if cuda:
            torch.cuda.manual_seed_all(self._seed)
        cfg = FluxConfig(device="cuda" if cuda else "cpu",
                         cuda_primary=cuda, **flux_kwargs)
        model = FluxLM(config=cfg)
        # Eager fused path: thousands of sequential model inits must not
        # risk capture-state corruption — CUDA-graph capture is a
        # whole-process resource, and an aborted capture poisons the RNG
        # bookkeeping for every later rand op in the engine.
        model._use_cuda_graph = False
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        if quick:
            # ~1/3 train bytes, held slice disjoint from quick train
            train_ids = list(self._corpus[:_Q_TRAIN]) + list(self._facts)
            held = list(self._corpus[_Q_TRAIN:_Q_TRAIN + _Q_HELD])
            stride = _Q_STRIDE
        else:
            train_ids = list(self._train)
            held = list(self._held)
            stride = _HELD_STRIDE

        t0 = time.time()
        model.ingest(train_ids, tag="eval")
        if cuda:
            torch.cuda.synchronize()
        ing_tps = len(train_ids) / max(time.time() - t0, 1e-9)

        # held-out NLL (predict -> observe), measures live generalization
        nll, n_ev = 0.0, 0
        t0 = time.time()
        for i in range(0, len(held), stride):
            lg = torch.as_tensor(model._predict()).float()
            lp = torch.log_softmax(lg, dim=-1)
            tgt = held[i]
            nll -= float(lp[tgt])
            n_ev += 1
            model._push(tgt)
        if cuda:
            torch.cuda.synchronize()
        pred_tps = n_ev / max(time.time() - t0, 1e-9)
        held_nll = nll / max(n_ev, 1)

        # planted-fact probes: hop-1 recall + real 2-hop chain recall
        # ('#V{i}>' -> V{i+1}: the hop-1 VALUE becomes the hop-2 KEY —
        # measures association-chaining, not same-timestep confidence)
        h1, h2, n2 = 0.0, 0.0, 0
        for ctx1, t1, ctx2, t2 in self._probes:
            for b in ctx1:
                model._push(int(b))
            lp = torch.log_softmax(
                torch.as_tensor(model._predict()).float(), dim=-1)
            h1 -= float(lp[t1])
            model._push(int(t1))
            for b in ctx2:
                model._push(int(b))
            lp = torch.log_softmax(
                torch.as_tensor(model._predict()).float(), dim=-1)
            h2 -= float(lp[t2])
            model._push(int(t2))
            n2 += 1
        hop1_nll = h1 / max(n2, 1)
        hop2_nll = h2 / max(n2, 1)

        if quick:
            # Skip decode probe — gen speed/collapse don't rank well at
            # 48 steps anyway, and this is the most expensive section.
            gen_tps, diversity = 0.0, 0.5
        else:
            # argmax decode: speed + repeat-collapse diversity check
            t0 = time.time()
            out = model.generate(list(self._prompt), max_new=_GEN_STEPS,
                                 temperature=0.0, learn=False,
                                 tag="gen_eval")
            if cuda:
                torch.cuda.synchronize()
            gen_tps = _GEN_STEPS / max(time.time() - t0, 1e-9)
            grams = set()
            for i in range(0, max(len(out) - 3, 0)):
                grams.add(bytes(out[i:i + 4]))
            diversity = len(grams) / max(len(out) - 3, 1)

        rep = model.memory_report()
        mem_mb = float(rep.get("total_MB_est", 0.0))
        if cuda:
            mem_mb += torch.cuda.max_memory_allocated() / 1e6

        del model
        if cuda:
            torch.cuda.empty_cache()

        return {"held_nll": held_nll, "hop1_nll": hop1_nll,
                "hop2_nll": hop2_nll, "pred_tps": pred_tps,
                "ing_tps": ing_tps, "gen_tps": gen_tps,
                "mem_mb": mem_mb, "diversity": float(diversity)}

    def evaluate(self, config: dict) -> dict:
        try:
            kw = self._to_flux_kwargs(config)
        except Exception as e:
            return {"score": -1e9, "behavioral": (0, 0, 0, 0),
                    "metadata": {"fail": f"decode: {e}"}}
        try:
            m = self._measure(kw)
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"score": -1e9, "behavioral": (0, 0, 0, 0),
                    "metadata": {"fail": f"eval: {e}"}}

        vals = [m["held_nll"], m["hop1_nll"], m["hop2_nll"],
                m["pred_tps"], m["ing_tps"], m["gen_tps"], m["mem_mb"]]
        if not all(math.isfinite(v) for v in vals):
            return {"score": -1e9, "behavioral": (0, 0, 0, 0),
                    "metadata": {"fail": "nonfinite", **m}}

        if self._baseline is None:
            self._baseline = m
        b = self._baseline

        def lr(num, den):  # log2 ratio, outlier-clipped
            return float(np.clip(
                math.log2(max(num, 1e-9) / max(den, 1e-9)), -8, 8))

        score = 25.0
        score += 4.0 * lr(b["held_nll"], m["held_nll"])     # generation
        score += 3.0 * lr(b["hop2_nll"], m["hop2_nll"])     # reasoning
        score += 2.0 * lr(m["pred_tps"], b["pred_tps"])     # speed
        score += 1.5 * lr(m["ing_tps"], b["ing_tps"])
        score += 1.0 * lr(m["gen_tps"], b["gen_tps"])
        score += 2.0 * lr(b["mem_mb"], m["mem_mb"])         # memory
        score += 1.0 * lr(b["hop1_nll"], m["hop1_nll"])     # recall
        score += 2.0 * (m["diversity"] - b["diversity"])    # anti-collapse

        return {"score": float(score),
                "behavioral": (
                    float(np.clip(m["held_nll"], 0, 6)),
                    float(np.clip(m["hop2_nll"], 0, 6)),
                    float(np.clip(math.log10(max(m["pred_tps"], 1)),
                                  0, 4)),
                    float(np.clip(math.log10(max(m["mem_mb"], 1e-3)),
                                  -1, 3.5))),
                "metadata": {**m,
                             "gen_family": config.get("gen_family"),
                             "orders": list(config.get("orders", ())),
                             "mem_preset": config.get("mem_preset"),
                             "speed_preset": config.get("speed_preset")}}

    def quick_eval(self, config: dict) -> dict:
        """Cheap promotion tier — same model build, ~1/3 corpus, fewer
        predict positions, no decode probe. Score is self-consistent
        (its own production baseline) and used ONLY for ranking popped
        frontier entries; the top fraction still gets full evaluate().
        """
        try:
            kw = self._to_flux_kwargs(config)
        except Exception as e:
            return {"score": -1e9, "behavioral": (0, 0, 0, 0),
                    "metadata": {"fail": f"decode: {e}", "quick": True}}
        try:
            m = self._measure(kw, quick=True)
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"score": -1e9, "behavioral": (0, 0, 0, 0),
                    "metadata": {"fail": f"eval: {e}", "quick": True}}

        vals = [m["held_nll"], m["hop1_nll"], m["hop2_nll"],
                m["pred_tps"], m["ing_tps"], m["mem_mb"]]
        if not all(math.isfinite(v) for v in vals):
            return {"score": -1e9, "behavioral": (0, 0, 0, 0),
                    "metadata": {"fail": "nonfinite", "quick": True}}

        if self._baseline_quick is None:
            # Anchor on the production recipe, not whichever candidate
            # happened to be first — keeps quick ranks stable.
            self._baseline_quick = self._measure(
                self._to_flux_kwargs(self.decode(self._mid())),
                quick=True)
        b = self._baseline_quick

        def lr(num, den):
            return float(np.clip(
                math.log2(max(num, 1e-9) / max(den, 1e-9)), -8, 8))

        # Quality+memory terms only — tps is wall-clock noise at quick
        # fidelity (sub-second workloads); promoted candidates get real
        # speed measurements at the full tier.
        score = 25.0
        score += 4.0 * lr(b["held_nll"], m["held_nll"])
        score += 3.0 * lr(b["hop2_nll"], m["hop2_nll"])
        score += 2.0 * lr(b["mem_mb"], m["mem_mb"])
        score += 1.0 * lr(b["hop1_nll"], m["hop1_nll"])

        return {"score": float(score),
                "behavioral": (0, 0, 0, 0),
                "metadata": {"quick": True, **m}}

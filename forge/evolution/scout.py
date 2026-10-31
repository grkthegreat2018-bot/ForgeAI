"""Scout — lightweight goal-driven discovery engine (OpenEvolve-lite).

Built on the existing ForgeEvolve substrate — JSON domain specs, real
simulators, RewardGuard scoring, FindingsDB — with the heavy machinery
(500-generator MLP population, gen-model manager, pickled weight BLOBs,
MAP-Elites grid) replaced by three small ML components:

  1. TPE proposer (Parzen density-ratio, numpy-only). Per-domain history
     splits into good/bad at the gamma quantile; candidates are jittered
     around the elite/good set and ranked by l(x)/g(x). Kernel bandwidths
     shrink per-dim where |corr(param, score)| is high — the loop learns
     which knobs matter. ``mode="random"`` gives the mandatory
     random-search baseline for comparison.
  2. Thompson-sampling bandit over domains (Beta on "cycle produced a
     verified improvement") — cycles flow to whichever domain is currently
     paying off; saturated domains cool off and eventually retire.
  3. Cross-domain param priors: scout_param_stats holds global
     per-(param-name, bin) Welford stats of *within-domain z-scored*
     results. A domain with little history gets a prior over good regions
     for params shared with other domains (rank, bits, alpha, window, lr).

Promotion rule ("proven"): a candidate must pass every test gate the
domain already declares (RewardGuard penalties + hardening) AND beat the
incumbent best's re-measured mean by a noise-aware margin
(max(min_delta, rel_delta*|incumbent|, k*pooled_sem)); challenger and
incumbent are both replicated --verify-reps times with fresh seeds. Only
verified winners are written to the discoveries table — the same table
the rest of ForgeAI reads.

Stagnation ("until obvious stagnation"): a domain retires after
--patience-dom consecutive cycles without a verified promotion; the run
stops early when no domain has produced a discovery for --patience
consecutive cycles (or all domains retired). A plateau triggers a basin
hop first: elite reset + widened kernels before retirement.

Usage:
    python scripts/scout_loop.py --focus memory --cycles 300
    python scripts/scout_loop.py --domains hadamard_kv w8a8_quant
    python scripts/scout_loop.py --report
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from .database import FindingsDB
from .domain_spec import JSONSpecDomain, list_specs, load_spec


# Goal → spec-category routing (mirrors configs/focus_profiles.json).
GOAL_CATEGORIES = {
    "memory": {"memory", "kv", "quantization"},
    "speed": {"decoding", "kv"},
    "quality": {"attention", "training", "architecture"},
    "training": {"training"},
    "all": None,
}

_N_BINS = 8


# ── config ──────────────────────────────────────────────────────────────

@dataclass
class ScoutConfig:
    domains: list[str] | None = None   # explicit spec names
    focus: str | None = None           # memory|speed|quality|training|all
    require: str | None = None         # "metric>=v" gate on promotions
    cycles: int = 300                  # bandit iterations
    batch: int = 16                    # candidates evaluated per cycle
    db_path: str = "forge_evolve.db"
    device: str = "cpu"                # sims are tiny — CPU keeps VRAM free
    seed: int = 0
    # verification ("proven" gate)
    verify_reps: int = 3               # replicate evals for challenger+incumbent
    min_delta: float = 0.0             # absolute margin floor
    rel_delta: float = 1e-3            # relative margin vs |incumbent|
    k_sigma: float = 1.0               # margin += k * pooled SEM
    # proposer
    mode: str = "tpe"                  # tpe | random
    explore_eps: float = 0.15          # fraction of batch = uniform random
    m_candidates: int = 96             # TPE sample pool per batch
    gamma: float = 0.25                # top-quantile fraction = "good"
    prior_weight: float = 8.0          # pseudo-count of cross-domain prior
    hist_cap: int = 512                # max evals loaded per domain on resume
    n_elites: int = 8                  # elite archive size per domain
    # stagnation
    patience: int = 30                 # global: stop after N dry cycles
    patience_dom: int = 40             # per-domain retirement
    hop_after: int = 12                # basin hop when this stale
    seed_configs_first: bool = True
    verbose: bool = True


# ── cross-domain param-prior tables (scalar only — no blobs) ────────────

_SCOUT_DDL = [
    """CREATE TABLE IF NOT EXISTS scout_domain_stats (
        domain TEXT PRIMARY KEY,
        n INTEGER NOT NULL,
        mean REAL NOT NULL,
        m2 REAL NOT NULL,
        best_score REAL,
        cycles INTEGER DEFAULT 0,
        alpha REAL DEFAULT 1.0,
        beta REAL DEFAULT 1.0,
        stale INTEGER DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS scout_param_stats (
        param_name TEXT NOT NULL,
        bin INTEGER NOT NULL,
        n INTEGER NOT NULL,
        mean_z REAL NOT NULL,
        m2 REAL NOT NULL,
        PRIMARY KEY (param_name, bin)
    )""",
]


class _Welford:
    """Running mean/variance (numerically stable)."""
    __slots__ = ("n", "mean", "m2")

    def __init__(self, n=0, mean=0.0, m2=0.0):
        self.n, self.mean, self.m2 = int(n), float(mean), float(m2)

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / max(self.n - 1, 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))


# ── per-domain state ─────────────────────────────────────────────────────

class _DomainState:
    """In-memory history + incumbent + elite archive for one domain."""

    def __init__(self, spec_name: str, device: str):
        self.spec = load_spec(spec_name)
        self.domain = JSONSpecDomain(spec=self.spec, device=device)
        self.name = spec_name
        self.dim = self.spec.output_dim
        self.X = np.zeros((0, self.dim), dtype=np.float64)
        self.y = np.zeros(0, dtype=np.float64)
        self.best_score = -float("inf")
        self.best_config: dict | None = None
        self.elites: list[tuple[np.ndarray, float]] = []  # top-k (x, score)
        self.cycles = 0
        self.stale = 0              # cycles since last verified promotion
        self.arm_a, self.arm_b = 1.0, 1.0   # Thompson Beta arm
        self.retired = False
        self._seeded = False

    def add(self, x: np.ndarray, score: float) -> None:
        self.X = np.vstack([self.X, x[None, :]])
        self.y = np.append(self.y, score)
        self._push_elite(x, score)

    def _push_elite(self, x: np.ndarray, score: float,
                    k: int = 8) -> None:
        if not np.isfinite(score):
            return
        for ex, _ in self.elites:
            if np.allclose(ex, x, atol=1e-6):
                return
        self.elites.append((x.copy(), float(score)))
        self.elites.sort(key=lambda e: -e[1])
        del self.elites[k:]

    def basin_hop(self, rng: np.random.Generator) -> None:
        """Escape a plateau: drop all but the top-2 elites and inject
        uniform noise — the next cycle searches wider."""
        self.elites = self.elites[:2]
        if len(self.X) > 64:  # keep recent history so TPE stays honest
            keep = np.argsort(self.y)[-64:]
            self.X, self.y = self.X[keep], self.y[keep]

    # ── TPE-lite proposal ────────────────────────────────────────────
    def _bandwidths(self, good: np.ndarray) -> np.ndarray:
        """Per-dim kernel width: Scott's rule on the good set, shrunk where
        the param correlates with score (learned importance)."""
        d = self.dim
        n_g = max(len(good), 1)
        base = (np.std(good, axis=0) * (n_g ** (-1.0 / (d + 4)))
                if n_g > 1 else np.full(d, 0.3))
        base = np.clip(base, 0.03, 0.5)
        if len(self.y) > 6 and np.std(self.y) > 0:
            for j in range(d):
                sx = np.std(self.X[:, j])
                if sx < 1e-9:
                    continue
                c = np.corrcoef(self.X[:, j], self.y)[0, 1]
                if np.isfinite(c):
                    base[j] *= (1.0 - 0.6 * abs(c))
        return np.clip(base, 0.02, 0.5)

    def propose(self, n: int, rng: np.random.Generator,
                prior_fns: dict[int, np.ndarray] | None = None,
                prior_beta: float = 0.0, eps: float = 0.15,
                mode: str = "tpe", m: int = 96) -> np.ndarray:
        """Return n parameter vectors in [0,1]^d."""
        d = self.dim
        if mode == "random":
            return rng.random((n, d))
        n_exp = max(1, int(round(n * eps)))
        n_tpe = n - n_exp
        out = [rng.random((n_exp, d))]

        finite = np.isfinite(self.y) & (self.y > -1e8)
        n_obs = int(finite.sum())
        if n_obs < 4 or n_tpe <= 0:
            pts = rng.random((max(n_tpe, 0), d))
            if prior_fns and prior_beta > 0:
                pts = _apply_prior(pts, prior_fns, prior_beta, rng)
            out.append(pts)
            return np.clip(np.vstack(out), 0.0, 1.0)[:n]

        X, y = self.X[finite], self.y[finite]
        k = max(1, int(math.ceil(0.25 * n_obs)))
        order = np.argsort(y)
        good = list(X[order[-k:]])
        bad = X[order[:n_obs - k]] if n_obs - k else X[:1]
        # Elite parents: archived best configs join the good set
        # (island-lite — parents are proven survivors, not just recent luck)
        good.extend(ex for ex, _ in self.elites)
        good = np.asarray(good)
        sig = self._bandwidths(good)

        idx = rng.integers(0, len(good), size=m)
        cand = np.clip(good[idx] + rng.normal(0, sig, size=(m, d)), 0, 1)

        def _parzen(pts, centers, s):
            z = (pts[:, None, :] - centers[None, :, :]) / s[None, None, :]
            return np.exp(-0.5 * (z ** 2)).mean(axis=1).prod(axis=1)

        acq = _parzen(cand, good, sig) / (_parzen(cand, bad, sig) + 1e-12)

        if prior_fns and prior_beta > 0:
            pw = np.ones(m)
            bins = (cand * _N_BINS).astype(int).clip(0, _N_BINS - 1)
            for j, dens in prior_fns.items():
                pw *= dens[bins[:, j]]
            acq *= (1.0 + prior_beta * pw)

        top = np.argsort(acq)[-n_tpe:]
        out.append(cand[top])
        return np.clip(np.vstack(out), 0.0, 1.0)[:n]


def _apply_prior(pts: np.ndarray, prior_fns: dict[int, np.ndarray],
                 beta: float, rng: np.random.Generator) -> np.ndarray:
    """Re-sample a beta-fraction of cold-start points into bins the
    cross-domain prior favours."""
    n, _d = pts.shape
    for j, dens in prior_fns.items():
        mask = rng.random(n) < min(beta, 1.0)
        if not mask.any():
            continue
        k = int(mask.sum())
        bins = np.searchsorted(np.cumsum(dens / dens.sum()),
                               rng.random(k)).clip(0, _N_BINS - 1)
        pts[mask, j] = (bins + rng.random(k)) / _N_BINS
    return pts


# ── scout engine ────────────────────────────────────────────────────────

class Scout:
    """Lightweight goal-driven discovery engine. See module docstring."""

    def __init__(self, cfg: ScoutConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.db = FindingsDB(cfg.db_path)
        with self.db._lock:
            for stmt in _SCOUT_DDL:
                self.db.conn.execute(stmt)
            self.db.conn.commit()
        self.run_id = f"scout_{uuid.uuid4().hex[:10]}"
        self.states: dict[str, _DomainState] = {}
        self._domain_stats: dict[str, _Welford] = {}
        self._param_stats: dict[tuple[str, int], _Welford] = {}
        self._req_metric, self._req_op, self._req_val = self._parse_require()
        self._load_state()
        self._discoveries: list[dict] = []
        self._eval_count = 0
        self._dry = 0                     # global stagnation counter
        self.stop_reason = "cycle budget"

    # ── goal parsing ──────────────────────────────────────────────────
    def _parse_require(self):
        """Parse --require 'metric>=v' style promotion gate."""
        r = self.cfg.require
        if not r:
            return None, None, None
        for op in (">=", "<=", ">", "<", "="):
            if op in r:
                name, val = r.split(op, 1)
                return name.strip(), op, float(val)
        raise ValueError(f"--require must look like 'metric>=value', got {r!r}")

    def _meets_requirement(self, res_meta: dict) -> bool:
        """The 'fits requested end result' gate: a promotion only counts if
        the named metric satisfies the threshold."""
        if self._req_metric is None:
            return True
        name, op, val = self._req_metric, self._req_op, self._req_val
        m = res_meta.get(name)
        if m is None or not np.isfinite(float(m)):
            return False
        m = float(m)
        return {">=": m >= val, "<=": m <= val, ">": m > val,
                "<": m < val, "=": abs(m - val) < 1e-9}[op]

    # ── domain selection ─────────────────────────────────────────────
    def _light_specs(self) -> list[str]:
        keep = []
        cats = GOAL_CATEGORIES.get(self.cfg.focus or "all")
        for n in list_specs():
            try:
                spec = load_spec(n)
            except Exception:
                continue
            # Skip heavy domains: LLM gen models / model-boot checkers.
            if spec.gen_model_type == "llm":
                continue
            if spec.checker_type in ("llm_judge", "model_boot"):
                continue
            if spec.output_dim == 0:
                continue
            if self.cfg.domains and n not in self.cfg.domains:
                continue
            if cats is not None and spec.category not in cats:
                continue
            keep.append(n)
        return keep

    def _load_state(self) -> None:
        """Init domain states + warm-load history/stats from the DB."""
        for name in self._light_specs():
            try:
                self.states[name] = _DomainState(name, self.cfg.device)
            except Exception as e:
                print(f"[scout] skip {name}: {e}")
        c = self.db.conn.cursor()
        for row in c.execute(
                "SELECT domain, n, mean, m2, best_score, cycles, alpha, beta,"
                " stale FROM scout_domain_stats"):
            self._domain_stats[row[0]] = _Welford(row[1], row[2], row[3])
            st = self.states.get(row[0])
            if st is not None:
                st.cycles = row[5] or 0
                st.arm_a = row[6] or 1.0
                st.arm_b = row[7] or 1.0
                st.stale = row[8] or 0
                if row[4] is not None:
                    st.best_score = row[4]
        for row in c.execute(
                "SELECT param_name, bin, n, mean_z, m2 FROM scout_param_stats"):
            self._param_stats[(row[0], row[1])] = _Welford(row[2], row[3], row[4])
        for name, st in self.states.items():
            rows = c.execute(
                "SELECT config_json, score FROM evaluations WHERE domain=? "
                "AND score > -1e8 ORDER BY id DESC LIMIT ?",
                (name, self.cfg.hist_cap)).fetchall()
            for cfg_json, score in reversed(rows):
                try:
                    x = np.asarray(
                        st.domain.encode(json.loads(cfg_json)),
                        dtype=np.float64).ravel()[:st.dim]
                    if len(x) == st.dim:
                        st.add(x, float(score))
                except Exception:
                    continue
            best_row = c.execute(
                "SELECT config_json, score FROM evaluations WHERE domain=? "
                "AND score > -1e8 ORDER BY score DESC LIMIT 1",
                (name,)).fetchone()
            if best_row:
                st.best_config = json.loads(best_row[0])
                st.best_score = max(st.best_score, float(best_row[1]))

    # ── cross-domain prior ────────────────────────────────────────────
    def _prior_fns(self, st: _DomainState) -> dict[int, np.ndarray]:
        """Per-param (N_BINS,) density from global stats for the same param
        name — discoveries in other domains shape this one's search."""
        fns: dict[int, np.ndarray] = {}
        if not st.spec.transferable:
            return fns
        for j, p in enumerate(st.spec.params):
            stats = [self._param_stats.get((p.name, b))
                     for b in range(_N_BINS)]
            if not any(s and s.n >= 3 for s in stats):
                continue
            w = np.clip(np.array([s.mean if s else 0.0 for s in stats]),
                        -2.0, 2.0)
            dens = 0.2 + 0.8 * np.exp(w - w.max())
            fns[j] = dens / dens.sum() * _N_BINS   # mean density 1.0
        return fns

    # ── stats update ──────────────────────────────────────────────────
    def _update_stats(self, st: _DomainState, X: np.ndarray,
                      scores: np.ndarray) -> None:
        dstat = self._domain_stats.setdefault(st.name, _Welford())
        names = [p.name for p in st.spec.params]
        for i in range(len(scores)):
            s = float(scores[i])
            dstat.add(s)
            if not st.spec.transferable:
                continue
            z = (s - dstat.mean) / (dstat.std + 1e-9)
            for j in range(min(st.dim, X.shape[1])):
                b = int(np.clip(X[i, j] * _N_BINS, 0, _N_BINS - 1))
                self._param_stats.setdefault(
                    (names[j], b), _Welford()).add(z)
        with self.db._lock:
            c = self.db.conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO scout_domain_stats "
                "(domain, n, mean, m2, best_score, cycles, alpha, beta, stale)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (st.name, dstat.n, dstat.mean, dstat.m2,
                 st.best_score if np.isfinite(st.best_score) else None,
                 st.cycles, st.arm_a, st.arm_b, st.stale))
            keys = [(names[j], int(np.clip(X[i, j] * _N_BINS, 0, _N_BINS - 1)))
                    for i in range(len(scores))
                    for j in range(min(st.dim, X.shape[1]))] \
                if st.spec.transferable and len(scores) else []
            for pname, b in set(keys):
                w = self._param_stats[(pname, b)]
                c.execute(
                    "INSERT OR REPLACE INTO scout_param_stats "
                    "(param_name, bin, n, mean_z, m2) VALUES (?,?,?,?,?)",
                    (pname, b, w.n, w.mean, w.m2))
            self.db.conn.commit()

    # ── domain bandit (Thompson) ──────────────────────────────────────
    def _pick_domain(self) -> _DomainState | None:
        live = [s for s in self.states.values() if not s.retired]
        if not live:
            return None
        # Novel domains get priority — Thompson on Beta(1,1) ≈ uniform,
        # but give them a head start so every domain is tried once.
        fresh = [s for s in live if s.cycles == 0]
        if fresh:
            return fresh[int(self.rng.integers(len(fresh)))]
        theta = [self.rng.beta(s.arm_a, s.arm_b) for s in live]
        return live[int(np.argmax(theta))]

    # ── verification ──────────────────────────────────────────────────
    def _eval(self, st: _DomainState, config: dict) -> dict:
        import torch
        torch.manual_seed(int(self.rng.integers(1 << 30)))
        np.random.seed(int(self.rng.integers(1 << 30)))
        return st.domain.evaluate(config)

    def _verified(self, st: _DomainState, config: dict,
                  cand_score: float) -> tuple[bool, dict]:
        """Replicate challenger and incumbent; challenger must win by a
        noise-aware margin (modded-nanogpt-style gate)."""
        r = self.cfg.verify_reps
        c_scores = [cand_score]
        for _ in range(r - 1):
            try:
                c_scores.append(float(self._eval(st, config)["score"]))
            except Exception:
                c_scores.append(-1e9)
        i_scores = []
        inc = st.best_config
        same = False
        if inc is not None:
            try:
                xi = np.asarray(st.domain.encode(inc)).ravel()[:st.dim]
                xc = np.asarray(st.domain.encode(config)).ravel()[:st.dim]
                same = np.allclose(xi, xc, atol=1e-6)
            except Exception:
                same = inc == config
        if inc is not None and not same:
            for _ in range(r):
                try:
                    i_scores.append(float(self._eval(st, inc)["score"]))
                except Exception:
                    i_scores.append(-1e9)
        cm = float(np.mean(c_scores))
        im = float(np.mean(i_scores)) if i_scores else st.best_score
        # Cold start: no incumbent yet — any finite challenger establishes
        # the baseline (it IS the past best at this point).
        if not np.isfinite(im):
            ev = {"challenger_mean": cm, "incumbent_mean": None,
                  "reps": r, "margin": 0.0, "delta": 0.0, "z": None}
            return np.isfinite(cm), ev
        pooled = (float(np.std(c_scores + i_scores))
                  if i_scores else float(np.std(c_scores)))
        margin = max(self.cfg.min_delta,
                     self.cfg.rel_delta * abs(im),
                     self.cfg.k_sigma * pooled / math.sqrt(max(r, 1)))
        z = ((cm - im) / (pooled / math.sqrt(max(r, 1)) + 1e-12)
             if pooled > 0 else float("inf") * np.sign(cm - im or 1))
        ev = {"challenger_mean": cm, "incumbent_mean": im,
              "reps": r, "margin": margin, "delta": cm - im, "z": float(z)}
        return (cm > im + margin), ev

    # ── main loop ─────────────────────────────────────────────────────
    def run(self) -> dict:
        t0 = time.time()
        cfg = self.cfg
        if cfg.verbose:
            focus = cfg.focus or "all"
            print(f"[scout] {len(self.states)} domains (focus={focus}) "
                  f"db={cfg.db_path} device={cfg.device} run={self.run_id}")

        cyc = 0
        while cyc < cfg.cycles:
            st = self._pick_domain()
            if st is None:
                self.stop_reason = "all domains retired (stagnant)"
                break
            cyc += 1
            st.cycles += 1

            priors = self._prior_fns(st)
            beta = cfg.prior_weight / (cfg.prior_weight + len(st.y))
            X = st.propose(cfg.batch, self.rng, priors, beta,
                           cfg.explore_eps, mode=cfg.mode,
                           m=cfg.m_candidates)
            if cfg.seed_configs_first and not st._seeded:
                seeds = st.spec.seed_configs
                if seeds:
                    enc = [np.asarray(st.domain.encode(s),
                                      dtype=np.float64).ravel()[:st.dim]
                           for s in seeds]
                    X = np.vstack([np.array(enc), X])[:cfg.batch]
                st._seeded = True

            import torch
            scores, cfgs, metas = [], [], []
            for i in range(len(X)):
                config = st.domain.decode(
                    torch.tensor(X[i], dtype=torch.float32))
                try:
                    res = st.domain.evaluate(config)
                    sc, meta = float(res["score"]), res.get("metadata", {})
                except Exception as e:
                    sc, meta = -1e9, {"fail": str(e)[:200]}
                scores.append(sc)
                cfgs.append(config)
                metas.append(meta)
                self.db.save_evaluation(self.run_id, st.name, cyc,
                                        config, sc, meta)
                self._eval_count += 1
            scores = np.asarray(scores)
            good = np.isfinite(scores) & (scores > -1e8)
            for i in np.where(good)[0]:
                st.add(X[i], scores[i])

            # Promotion: batch best must beat incumbent AND fit the goal.
            promoted = False
            if good.any():
                bi = int(np.argmax(np.where(good, scores, -np.inf)))
                if scores[bi] > st.best_score and \
                        self._meets_requirement(metas[bi]):
                    ok, ev = self._verified(st, cfgs[bi], scores[bi])
                    if ok:
                        prev = st.best_score
                        st.best_score = ev["challenger_mean"]
                        st.best_config = cfgs[bi]
                        self.db.save_discoveries(self.run_id, st.name, [{
                            "config": cfgs[bi],
                            "score": ev["challenger_mean"],
                            "generation": cyc,
                            "metadata": {"verified": ev,
                                         "prev_best": prev
                                         if np.isfinite(prev) else None}}])
                        self._discoveries.append(
                            {"domain": st.name, "cycle": cyc,
                             "score": ev["challenger_mean"],
                             "delta": ev["delta"], "config": cfgs[bi]})
                        st.arm_a += 1.0
                        st.stale = 0
                        promoted = True
                        if cfg.verbose:
                            zs = f"z={ev['z']:.2f}" if ev["z"] is not None \
                                else "baseline"
                            print(f"[scout] cyc {cyc} {st.name}: discovery "
                                  f"{ev['challenger_mean']:.4f} "
                                  f"(+{ev['delta']:.4f}, {zs})")
            if not promoted:
                st.arm_b += 1.0
                st.stale += 1
            self._dry = 0 if promoted else self._dry + 1

            # Stagnation: basin hop, then retire.
            if st.stale >= cfg.patience_dom:
                st.retired = True
                if cfg.verbose:
                    print(f"[scout] {st.name} retired after "
                          f"{st.stale} dry cycles")
            elif st.stale == cfg.hop_after:
                st.basin_hop(self.rng)

            self._update_stats(
                st,
                X[good] if good.any() else np.zeros((0, st.dim)),
                scores[good] if good.any() else np.zeros(0))

            if cfg.verbose and cyc % 25 == 0:
                live = sum(1 for s in self.states.values() if not s.retired)
                print(f"[scout] cyc {cyc}/{cfg.cycles} evals={self._eval_count} "
                      f"discoveries={len(self._discoveries)} "
                      f"live_domains={live} dry={self._dry}")

            if self._dry >= cfg.patience:
                self.stop_reason = (f"global stagnation: no verified "
                                    f"discovery in {self._dry} cycles")
                break

        self.db.save_run(self.run_id, "scout_multi",
                         {"cycles": cfg.cycles, "batch": cfg.batch,
                          "focus": cfg.focus},
                         {"generations": cyc,
                          "total_evaluations": self._eval_count,
                          "discoveries": len(self._discoveries),
                          "best_score": max((d["score"] for d in
                                             self._discoveries), default=0),
                          "device": cfg.device}, t0)
        if cfg.verbose:
            print(f"[scout] stopped: {self.stop_reason} — "
                  f"{len(self._discoveries)} discoveries in "
                  f"{self._eval_count} evals")
        return {"run_id": self.run_id, "cycles": cyc,
                "evals": self._eval_count,
                "discoveries": self._discoveries,
                "stop_reason": self.stop_reason,
                "elapsed_s": round(time.time() - t0, 1)}

    # ── reporting ─────────────────────────────────────────────────────
    def report(self, limit: int = 20) -> str:
        lines = ["domain                 score      delta    config"]
        for d in self.db.get_unapplied_discoveries(limit=limit):
            cfg_s = json.dumps(d["config"])[:60]
            delta = (d.get("metadata") or {}).get("verified", {}).get("delta")
            lines.append(f"{d['domain']:<22} {d['score']:<10.4f} "
                         f"{(delta if delta is not None else 0.0):<8.4f} "
                         f"{cfg_s}")
        # Cross-domain insights: params whose bins strongly predict wins
        # across ALL domains — the transferable knowledge.
        rows = self.db.conn.execute(
            "SELECT param_name, bin, n, mean_z FROM scout_param_stats "
            "WHERE n >= 5 AND ABS(mean_z) > 0.4 "
            "ORDER BY ABS(mean_z) DESC LIMIT 12").fetchall()
        if rows:
            lines.append("\ncross-domain insights (param bin -> mean z):")
            for name, b, n, mz in rows:
                lo, hi = b / _N_BINS, (b + 1) / _N_BINS
                lines.append(f"  {name} in [{lo:.2f},{hi:.2f}) "
                             f"-> z={mz:+.2f} (n={n})")
        return "\n".join(lines)


def run_scout(cfg: ScoutConfig) -> dict:
    return Scout(cfg).run()

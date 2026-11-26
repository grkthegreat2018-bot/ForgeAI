"""Harness-genome evolution — opt-in spike (RSI-loop whitespace recon
row C.1 / action note E.1: DarwinX 2608.07545 / Mem2Evolve axis).

The evolve loop already evolves *code solutions* (CodeEvolve elites) and
*weights* (LoRA hone) — this makes the *harness around the frozen model*
evolvable too: proposer sampling params, gen budgets, prompt-hint
variant, stop suffixes, and mutator op weights as ONE genome with
preserve-and-extend admission into a tiny elite archive persisted as a
JSON sidecar next to forge_evolve.db (<db>.harness.json).

Wiring (all opt-in via CodeEvolveConfig.harness_evolve — off = zero
behavior change):

- ``genome.apply_cfg(cfg)``      -> setattr real CodeEvolveConfig fields
  (temperature / top_p / top_k / max_gen_tokens / think_budget /
  n_insights) that the LLMProposer generate call sites already read.
- ``proposer._genome = genome``  -> LLMProposer._pick_hint honors
  genome.hint_idx (a fixed prompt-template variant) instead of the
  Thompson hint bandit, and the forge-path generate calls append
  genome.stop_suffixes to their stop lists.
- ``genome.apply_mutator(mut)``  -> instance-shadows Mutator._BASE_W
  with genome.mut_op_w multipliers (bandit stats still apply on top).

Fitness is caller-supplied — CodeEvolve feeds a valid-rate + promotion
heuristic over the window since the last application; the archive just
stores scores + lineage. Nothing here imports code_evolve (no cycle).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Candidate stop suffixes mutate() draws from — appended AFTER the
# mandatory im_end/eos stop at the call site, so a bad pick can only
# truncate output early, never break the parse contract.
_STOP_POOL = ("</code>", "\n\n\n", "\n#", "```\n")

# Per-field legal ranges — mutate() clamps into these; apply_cfg only
# ever writes clamped values, so a corrupted sidecar can't push
# temperature=99 into the sampler.
_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (0.2, 1.3),      # basin hop itself caps at 1.2
    "top_p": (0.5, 1.0),
    "top_k": (0, 256),              # 0 = disabled (engine semantics)
    "max_gen_tokens": (64, 1536),   # 1536 = observed think-starve case
    "think_budget": (0, 1536),      # 0 disables budget-forced thinking
    "n_insights": (0, 20),
    "hint_idx": (-1, 9),            # len(LLMProposer._HINTS) - 1 == 9
}

# int fields get a +-1 step on mutation except token budgets, which
# drift by a larger stride so a few generations can actually explore
# (+-1 on a 640-token budget would need ~900 generations to reach 256).
_INT_STEPS: dict[str, int] = {
    "top_k": 8, "max_gen_tokens": 64, "think_budget": 64,
    "n_insights": 1, "hint_idx": 1,
}

_MUT_OP_SCALE = (0.1, 4.0)          # mut_op_w multiplier range


def _new_id() -> str:
    return f"hg_{uuid.uuid4().hex[:8]}"


# ── rng shims (accept numpy Generator AND random.Random) ─────────────

def _rf(rng) -> float:
    return float(rng.random() if callable(getattr(rng, "random", None))
                 else rng.uniform(0.0, 1.0))


def _gauss(rng, mu: float, sigma: float) -> float:
    if hasattr(rng, "normal"):
        return float(rng.normal(mu, sigma))
    return float(rng.gauss(mu, sigma))


def _rint(rng, lo: int, hi_incl: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(lo, hi_incl + 1))
    return int(rng.randrange(lo, hi_incl + 1))


def _pick(rng, seq):
    if hasattr(rng, "integers"):
        return seq[int(rng.integers(len(seq)))]
    return rng.choice(list(seq))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class HarnessGenome:
    """One harness configuration. Defaults reproduce today's frozen
    harness exactly (bandit hint, no extra stops, base mutator weights)."""

    # ── sampling params (cfg fields read at every generate call) ────
    temperature: float = 0.8        # cfg.temperature
    top_p: float = 0.95             # cfg.top_p (was a 0.95 literal)
    top_k: int = 80                 # cfg.top_k (was an 80 literal)
    # ── generation budgets ──────────────────────────────────────────
    max_gen_tokens: int = 640       # cfg.max_gen_tokens — code budget
    think_budget: int = 512         # cfg.think_budget — Spark think cap
    # ── prompt construction ─────────────────────────────────────────
    n_insights: int = 5             # cfg.n_insights — insight lines
                                    # injected into proposal prompts
    hint_idx: int = -1              # prompt-template variant: forced
                                    # LLMProposer._HINTS[i]; -1 = the
                                    # default Thompson hint bandit
    stop_suffixes: tuple = ()       # extra stop strings appended to
                                    # the per-call stop lists
    # ── mutation weights ────────────────────────────────────────────
    mut_op_w: dict = field(default_factory=dict)   # op -> scale over
                                    # Mutator._BASE_W; {} = base weights
    # ── lineage (not evolvable) ─────────────────────────────────────
    genome_id: str = field(default_factory=_new_id)
    parents: tuple = ()
    gen: int = 0

    # ── mutation ────────────────────────────────────────────────────

    def mutate(self, rng, p: float = 0.25) -> "HarnessGenome":
        """Per-field perturbation — gaussian drift for floats, small
        steps for int budgets, uniform re-pick for the hint template id
        and the stop-suffix set. Every field is clamped back into
        _BOUNDS so mutated genomes stay in the legal operating range."""
        g = replace(self,
                    genome_id=_new_id(),
                    parents=(self.genome_id,),
                    gen=self.gen + 1,
                    stop_suffixes=tuple(self.stop_suffixes),
                    mut_op_w=dict(self.mut_op_w))
        B = _BOUNDS
        if _rf(rng) < p:
            lo, hi = B["temperature"]
            g.temperature = _clamp(
                g.temperature + _gauss(rng, 0.0, 0.1), lo, hi)
        if _rf(rng) < p:
            lo, hi = B["top_p"]
            g.top_p = _clamp(g.top_p + _gauss(rng, 0.0, 0.03), lo, hi)
        # int budgets: +-step drift
        for f, step in _INT_STEPS.items():
            if _rf(rng) >= p:
                continue
            lo, hi = B[f]
            setattr(g, f, int(_clamp(
                getattr(g, f) + _rint(rng, -step, step), lo, hi)))
        # prompt-template variant: swap to a random enum value when hit
        if _rf(rng) < p:
            lo, hi = B["hint_idx"]
            g.hint_idx = _rint(rng, int(lo), int(hi))
        # stop suffixes: re-pick a 0-2 subset of the pool when hit
        if _rf(rng) < p:
            g.stop_suffixes = tuple(
                s for s in _STOP_POOL if _rf(rng) < 0.4)[:2]
        # mutator weights: perturb one op's scale or drop an entry
        if _rf(rng) < p:
            try:
                from .mutator import Mutator
                ops = Mutator._OPS
            except Exception:
                ops = tuple(self.mut_op_w) or ()
            if ops:
                op = _pick(rng, ops)
                if op in g.mut_op_w and _rf(rng) < 0.3:
                    del g.mut_op_w[op]
                else:
                    lo, hi = _MUT_OP_SCALE
                    g.mut_op_w[op] = _clamp(
                        g.mut_op_w.get(op, 1.0) * (2.0 **
                                                   _gauss(rng, 0.0, 0.4)),
                        lo, hi)
        return g

    # ── application seams ───────────────────────────────────────────

    _CFG_FIELDS = ("temperature", "top_p", "top_k", "max_gen_tokens",
                   "think_budget", "n_insights")

    def apply_cfg(self, cfg) -> list[str]:
        """Write the sampling/budget fields onto a CodeEvolveConfig —
        only names that already exist there (getattr/setattr contract).
        Returns the field names actually applied."""
        applied = []
        for f in self._CFG_FIELDS:
            if hasattr(cfg, f):
                setattr(cfg, f, getattr(self, f))
                applied.append(f)
        return applied

    def apply_mutator(self, mut) -> None:
        """Instance-shadow Mutator._BASE_W with this genome's per-op
        multipliers — the Thompson bandit still samples on top of the
        scaled base weights. Empty mut_op_w = untouched."""
        base = dict(getattr(type(mut), "_BASE_W", None)
                    or getattr(mut, "_BASE_W", {}) or {})
        if not base or not self.mut_op_w:
            return
        lo, hi = _MUT_OP_SCALE
        mut._BASE_W = {
            op: max(1e-3, w * _clamp(self.mut_op_w.get(op, 1.0), lo, hi))
            for op, w in base.items()}

    # ── persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stop_suffixes"] = list(self.stop_suffixes)
        d["parents"] = list(self.parents)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HarnessGenome":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in dict(d).items() if k in known}
        kw["stop_suffixes"] = tuple(kw.get("stop_suffixes") or ())
        kw["parents"] = tuple(kw.get("parents") or ())
        kw["mut_op_w"] = dict(kw.get("mut_op_w") or {})
        g = cls(**kw)
        # Belt + suspenders: clamp a hand-edited/corrupt sidecar back
        # into bounds rather than letting it steer the sampler wildly.
        for f, (lo, hi) in _BOUNDS.items():
            v = _clamp(getattr(g, f), lo, hi)
            setattr(g, f, int(v) if f in _INT_STEPS else float(v))
        return g


class HarnessArchive:
    """Tiny elite store for harness genomes — preserve-and-extend.

    Persisted as ``<forge_evolve.db>.harness.json``. Entries are
    ``{"fitness": float, "ts": float, "genome": dict}`` sorted by
    fitness descending, capped at ``top_k``.

    Admission gate: a genome enters only when the archive isn't full
    yet OR its fitness strictly beats the current minimum — the weakest
    elite is evicted (preserve the best, extend by exploring mutated
    children whose parents are recorded on the genome itself).
    """

    def __init__(self, path: str | Path, top_k: int = 8):
        self.path = Path(path)
        self.top_k = int(top_k)
        self.entries: list[dict] = []
        self._load()

    # ── persistence ─────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for e in data.get("entries", []):
                try:
                    self.entries.append({
                        "fitness": float(e["fitness"]),
                        "ts": float(e.get("ts", 0.0)),
                        "genome": HarnessGenome.from_dict(e["genome"]),
                    })
                except Exception:
                    continue       # skip corrupt rows, keep the rest
            self.entries.sort(key=lambda e: -e["fitness"])
            del self.entries[self.top_k:]
        except Exception:
            self.entries = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({
                "top_k": self.top_k,
                "entries": [{"fitness": e["fitness"], "ts": e["ts"],
                             "genome": e["genome"].to_dict()}
                            for e in self.entries]}, indent=1),
                encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass                   # archive is advisory — never crash a run

    # ── admission (preserve-and-extend) ──────────────────────────────

    @property
    def min_fitness(self) -> float:
        return self.entries[-1]["fitness"] if self.entries else -float("inf")

    def admit(self, genome: HarnessGenome, fitness: float) -> bool:
        """Record a scored genome. Returns True iff it entered the
        elite set — the archive is a trust boundary for lineage, so a
        rejected genome leaves no trace (it's a dead branch)."""
        fitness = float(fitness)
        if len(self.entries) >= self.top_k \
                and fitness <= self.min_fitness:
            return False
        self.entries.append({"fitness": fitness, "ts": time.time(),
                             "genome": genome})
        self.entries.sort(key=lambda e: -e["fitness"])
        del self.entries[self.top_k:]
        self.save()
        return True

    def record(self, genome: HarnessGenome, fitness: float) -> bool:
        """Alias for admit() — the call-site reads better when the
        caller's signal is 'this genome earned this score'."""
        return self.admit(genome, fitness)

    # ── sampling ─────────────────────────────────────────────────────

    def best(self) -> HarnessGenome | None:
        return self.entries[0]["genome"] if self.entries else None

    def sample(self, rng, explore_p: float = 0.5,
               default: HarnessGenome | None = None) -> HarnessGenome:
        """Pick the genome to run next.

        Empty archive -> the frozen-harness default (baseline must earn
        its slot like everything else). Otherwise: exploit the current
        best elite with prob (1 - explore_p), or EXTEND — mutate a
        uniformly picked elite into a child (parents recorded) whose
        measured fitness will decide admission. Extending keeps the
        parent in the archive regardless of the child's outcome."""
        if not self.entries:
            return default if default is not None else HarnessGenome()
        if _rf(rng) < explore_p:
            return _pick(rng, self.entries)["genome"].mutate(rng)
        return self.entries[0]["genome"]


# Optional fitness hook signature — the archive is signal-agnostic:
#   fitness_fn(window: dict) -> float
# where the caller (CodeEvolve._harness_maybe) supplies a window of
# counters since the genome was applied — e.g.
#   {"evals": d_evals, "raws": d_gen_n, "gated": d_gen_ok, "promos": d}
# The default policy used by code_evolve is valid-rate + promo-rate:
def default_fitness(window: dict) -> float:
    """valid_rate + promo_rate over the application window — both ~[0,1],
    so fitness roughly lands in [0, 2]. A genome that raises gate-rate
    or wins promotions outranks one that only survives."""
    raws = max(1, int(window.get("raws", 0)))
    evals = max(1, int(window.get("evals", 1)))
    return (int(window.get("gated", 0)) / raws
            + int(window.get("promos", 0)) / evals)

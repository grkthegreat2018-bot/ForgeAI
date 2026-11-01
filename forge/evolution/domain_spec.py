"""Declarative domain specification + JSON-driven domain base.

Replaces hand-written decode/encode/behavioral_dims/discrete_choices per
domain. A domain JSON spec describes the search space; the simulator
function (registered in simulators.py) computes raw metrics; RewardGuard
composes the final score from declarative scoring components + flags.

See configs/domains/<name>.json for spec format.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .domains import BaseDomain
from .reward_guard import RewardGuard, ScoringSpec
from .simulators import get_simulator

CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "domains"


# ---------------------------------------------------------------------------
# Param spec — one entry per continuous/discrete parameter
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """One parameter in the search space."""
    name: str
    kind: str            # "choice" | "int" | "float" | "bool"
    # choice
    values: list = field(default_factory=list)
    # int / float
    low: float = 0.0
    high: float = 1.0
    # float only — log-scale encoding (lr, weight_decay)
    log_scale: bool = False
    # bool
    threshold: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> ParamSpec:
        return cls(
            name=d["name"],
            kind=d["type"],
            values=d.get("values", []),
            low=float(d.get("range", [0.0, 1.0])[0]),
            high=float(d.get("range", [0.0, 1.0])[1]),
            log_scale=bool(d.get("log_scale", False)),
            threshold=float(d.get("threshold", 0.5)),
        )

    def decode(self, p: float) -> Any:
        """Map a [0,1] param value to a concrete config value."""
        p = float(np.clip(p, 0.0, 1.0))
        if self.kind == "choice":
            if not self.values:
                return None
            idx = int(p * (len(self.values) - 0.001))
            return self.values[idx]
        if self.kind == "bool":
            return bool(p > self.threshold)
        if self.kind == "int":
            v = int(round(self.low + p * (self.high - self.low)))
            return int(np.clip(v, self.low, self.high))
        # float
        if self.log_scale:
            lo = max(self.low, 1e-12)
            hi = max(self.high, lo * 10)
            return float(math.exp(math.log(lo) + p * (math.log(hi) - math.log(lo))))
        return float(self.low + p * (self.high - self.low))

    def encode(self, value: Any) -> float:
        """Map a concrete config value back to [0,1]."""
        if self.kind == "choice":
            if value in self.values:
                i = self.values.index(value)
                return i / max(1, len(self.values) - 1)
            return 0.0
        if self.kind == "bool":
            return 1.0 if value else 0.0
        if self.kind == "int":
            if self.high == self.low:
                return 0.0
            return float(np.clip((int(value) - self.low) / (self.high - self.low), 0, 1))
        # float
        if self.log_scale:
            lo = max(self.low, 1e-12)
            hi = max(self.high, lo * 10)
            v = max(float(value), lo)
            return float(np.clip((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0, 1))
        if self.high == self.low:
            return 0.0
        return float(np.clip((float(value) - self.low) / (self.high - self.low), 0, 1))


# ---------------------------------------------------------------------------
# Domain spec — full JSON spec for one domain
# ---------------------------------------------------------------------------

@dataclass
class DomainSpec:
    """Parsed domain JSON spec."""
    name: str                          # domain name() string
    category: str                      # category for focus profiles
    params: list[ParamSpec]            # search-space params
    behavioral_dims: list[tuple]       # MAP-Elites grid
    scoring: ScoringSpec               # scoring policy
    simulator: str                     # name of simulator function
    seed_configs: list[dict] = field(default_factory=list)
    transferable: bool = True          # eligible for cross-domain transfer
    description: str = ""
    extra: dict = field(default_factory=dict)   # domain-specific extras
    # When set, decode_params combines all param values into a single array
    # under this key (e.g. synthetic domain: 8 params → {"x": [v0..v7]}).
    # encode_config reads the array back into individual params.
    array_output: str | None = None
    # Gen model type: "mlp" (default, fast param search) | "llm" (task-solving)
    gen_model_type: str = "mlp"
    # Checker type: "script" (default, fast) | "llm_judge" | "model_boot"
    checker_type: str = "script"
    # For llm_judge: natural-language description of what the answer must satisfy
    checker_requirements: str = ""
    # For model_boot: config name to boot (e.g. "forgelm_v2")
    checker_model_config: str = ""
    # When False, the spec is data-only (e.g. a scoring stub consumed by a
    # hand-written Python domain via load_spec) and is NOT registered in
    # DOMAINS as a standalone JSONSpecDomain.
    discoverable: bool = True

    @property
    def output_dim(self) -> int:
        return len(self.params)

    @classmethod
    def from_dict(cls, d: dict) -> DomainSpec:
        params = [ParamSpec.from_dict(p) for p in d["params"]]
        behav = [tuple(b) for b in d["behavioral_dims"]]
        scoring = ScoringSpec.from_dict(d["scoring"])
        return cls(
            name=d["name"],
            category=d.get("category", "uncategorized"),
            params=params,
            behavioral_dims=behav,
            scoring=scoring,
            simulator=d["simulator"],
            seed_configs=d.get("seed_configs", []),
            transferable=d.get("transferable", True),
            description=d.get("description", ""),
            extra=d.get("extra", {}),
            array_output=d.get("array_output"),
            gen_model_type=d.get("gen_model_type", "mlp"),
            checker_type=d.get("checker_type", "script"),
            checker_requirements=d.get("checker_requirements", ""),
            checker_model_config=d.get("checker_model_config", ""),
            discoverable=d.get("discoverable", True),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> DomainSpec:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def decode_params(self, params_tensor: torch.Tensor) -> dict[str, Any]:
        """Map a [0,1] param tensor to a concrete config dict."""
        p = params_tensor.detach().cpu().numpy()
        if self.array_output is not None:
            # Combine all params into a single array under one key
            values = [float(np.clip(p[i], 0.0, 1.0)) for i in range(len(self.params))]
            return {self.array_output: np.array(values, dtype=np.float64)}
        config = {}
        for i, spec in enumerate(self.params):
            config[spec.name] = spec.decode(float(p[i]))
        return config

    def encode_config(self, config: dict[str, Any]) -> torch.Tensor:
        """Map a config dict back to a [0,1] param tensor."""
        if self.array_output is not None:
            arr = config.get(self.array_output)
            if arr is None:
                return torch.zeros(len(self.params), dtype=torch.float32)
            if isinstance(arr, str):
                s = arr.strip().strip('[]')
                arr = [float(v) for v in s.split()]
            arr = np.asarray(arr, dtype=np.float64).flatten()
            return torch.tensor(arr[:len(self.params)], dtype=torch.float32)
        return torch.tensor(
            [spec.encode(config.get(spec.name)) for spec in self.params],
            dtype=torch.float32,
        )

    def discrete_choices(self) -> dict[str, list] | None:
        choices = {}
        for spec in self.params:
            if spec.kind == "choice" and spec.values:
                choices[spec.name] = list(spec.values)
        return choices or None

    def scoring_hash(self) -> str:
        """Stable hash of the scoring spec — used for DB regression guard."""
        import hashlib
        h = hashlib.sha256()
        h.update(json.dumps(self.scoring.to_dict(), sort_keys=True).encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Spec registry — loads all JSON specs from configs/domains/
# ---------------------------------------------------------------------------

_SPEC_CACHE: dict[str, DomainSpec] = {}
_SPEC_DIR_OVERRIDE: Path | None = None


def set_spec_dir(path: str | Path | None) -> None:
    """Override the spec directory (for tests). None resets to default."""
    global _SPEC_DIR_OVERRIDE, _SPEC_CACHE
    _SPEC_DIR_OVERRIDE = Path(path) if path else None
    _SPEC_CACHE.clear()


def spec_dir() -> Path:
    if _SPEC_DIR_OVERRIDE is not None:
        return _SPEC_DIR_OVERRIDE
    return CONFIG_DIR


def load_spec(name: str) -> DomainSpec:
    """Load a domain spec by name (looks for configs/domains/<name>.json)."""
    if name in _SPEC_CACHE:
        return _SPEC_CACHE[name]
    path = spec_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No domain spec for '{name}' at {path}")
    spec = DomainSpec.from_json_file(path)
    _SPEC_CACHE[name] = spec
    return spec


def list_specs() -> list[str]:
    """List all available domain spec names (without .json extension)."""
    d = spec_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_all_specs() -> dict[str, DomainSpec]:
    return {n: load_spec(n) for n in list_specs()}


# ---------------------------------------------------------------------------
# JSONSpecDomain — a BaseDomain driven entirely by a DomainSpec
# ---------------------------------------------------------------------------

class JSONSpecDomain(BaseDomain):
    """A BaseDomain implementation that reads everything from a DomainSpec.

    The simulator function (looked up by name in simulators.py) computes raw
    metrics. RewardGuard composes the final score from the spec's scoring
    policy. No hand-written decode/encode/evaluate needed.
    """

    def __init__(self, spec_name: str | None = None, spec: DomainSpec | None = None,
                 seq_len: int = 2048, seed: int = 42, device=None):
        if spec is not None:
            self.spec = spec
        elif spec_name is not None:
            self.spec = load_spec(spec_name)
        else:
            raise ValueError("JSONSpecDomain requires spec_name or spec")
        self._simulator_fn: Callable = get_simulator(self.spec.simulator)
        self._guard = RewardGuard(self.spec.scoring)
        # Tell the guard which metric names populate the behavioral vector
        self._guard._behavioral_names = [b[0] for b in self.spec.behavioral_dims]
        self.seq_len = seq_len
        self._seed = seed
        if device is not None:
            self._device = torch.device(device) if not isinstance(device, torch.device) else device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def name(self) -> str:
        return self.spec.name

    def output_dim(self) -> int:
        return self.spec.output_dim

    def behavioral_dims(self) -> list[tuple]:
        return self.spec.behavioral_dims

    def discrete_choices(self) -> dict[str, list] | None:
        return self.spec.discrete_choices()

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        return self.spec.decode_params(params)

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        return self.spec.encode_config(config)

    def seed_configs(self) -> list[dict[str, Any]]:
        return list(self.spec.seed_configs)

    def simulate(self, config: dict[str, Any]) -> dict[str, float]:
        """Run the raw simulator — returns metrics dict (no scoring)."""
        return self._simulator_fn(config, domain=self)

    def evaluate(self, config: dict[str, Any]) -> dict:
        """Run simulator + RewardGuard to produce final score."""
        metrics = self.simulate(config)
        return self._guard.score(config, metrics)

    def to_cpu(self) -> JSONSpecDomain:
        """CPU copy for parallel evaluation."""
        return JSONSpecDomain(spec=self.spec, seq_len=self.seq_len,
                              seed=self._seed, device=torch.device("cpu"))

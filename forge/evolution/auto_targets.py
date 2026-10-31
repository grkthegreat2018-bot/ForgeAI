"""Auto-harvested evolution targets — production functions opt in.

``@self_improve`` marks a hot-path production function as an evolution
target. The decorator only records metadata (zero cost at import);
``collect_targets()`` materializes one spec file per registered function
into ``forge/evolution/targets/auto_<name>.py`` so ``Target.resolve()``
loads it through the exact same path as the hand-seeded targets.

Key property: ``SEED_CODE`` is extracted LIVE from the decorated
function's source at harvest time — auto targets cannot drift from the
shipped implementation the way hand-copied seeds can.

Contract for annotated functions:
  * Standalone-evaluable: anything the function references beyond
    ``math``/``numpy``/``torch``/``F``/stdlib must be declared via
    ``deps=(names,)`` (same-module callables/classes — their source is
    spliced into the seed) or ``globals={name: literal}`` (constants,
    emitted via ``repr()``) and ``imports=(lines,)``.
  * ``gen_input``: a zero-arg, self-contained callable (module-level
    ``def`` in the same file — its source is lifted into the spec)
    returning a list of cases. A tuple case is called ``f(*case)``,
    anything else ``f(case)``. MUST be deterministic — ``test()`` calls
    it twice so candidate and reference see fresh-but-identical inputs
    (mutation-safe), and MUST be representative (real dtypes/shapes/
    distributions) or verified wins are fake.

Usage:
    from forge.evolution.auto_targets import self_improve

    def _evolve_inputs_x():
        import torch
        return [(torch.randn(4, 8),)]

    @self_improve(gen_input=_evolve_inputs_x, tags=("hot-path",))
    def _fast_thing(x):
        ...

Regenerate:  python -m forge.evolution.auto_targets
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from pathlib import Path

TARGETS_DIR = Path(__file__).resolve().parent / "targets"
PREFIX = "auto_"

# Modules collect_targets() imports so their @self_improve decorators
# run. Keep alphabetical; add new opt-in modules here only.
AUTO_MODULES = (
    "forge.engine.decision_head",
    "forge.engine.decoding",
    "forge.engine.engine_common",
    "forge.engine.mamba3",
    "forge.engine.prefix_cache",
    "forge.engine.quant.w8a8_quant",
    "forge.model.attention_ops",
)

_registry: dict[str, dict] = {}

# Names that must exist at spec-module level for _gen_cases and the seed
# body (torch/np/F/math are exec'd into the candidate namespace by the
# evaluator too — these cover the spec file itself + deps/globals).
_SPEC_HEADER_IMPORTS = (
    "import hashlib",
    "import math",
    "import random",
    "import struct",
    "import time",
    "import weakref",
    "import numpy as np",
    "import torch",
    "import torch.nn.functional as F",
)


def self_improve(fn=None, *, name: str | None = None,
                 gen_input=None, metric: str = "ms_per_call",
                 goal: str = "min", tags=(), prompt: str = "",
                 evals: int = 100, budgets: dict | None = None,
                 deps=(), globals: dict | None = None,
                 imports=(), bench_reps: int = 3):
    """Register a production function as an evolution target.

    Args:
        gen_input: zero-arg callable returning representative cases.
        deps: names of same-module callables/classes the function also
            references — their source is spliced into SEED_CODE.
        globals: module-level constants the function/deps need, emitted
            as ``name = repr(value)`` (must be repr-round-trippable).
        imports: extra import lines needed by the seed body.
        bench_reps: loops over all cases in bench().
        evals: default CodeEvolve eval budget for this target.
    """
    def wrap(f):
        if gen_input is None:
            raise ValueError(
                f"@self_improve on {f.__name__}: gen_input is required "
                "(a deterministic, representative case generator)")
        key = name or f.__name__.lstrip("_")
        _registry[key] = {
            "fn": f, "gen_input": gen_input, "metric": metric,
            "goal": goal, "tags": tuple(tags), "prompt": prompt,
            "evals": int(evals), "budgets": dict(budgets or {}),
            "deps": tuple(deps), "globals": dict(globals or {}),
            "imports": tuple(imports), "bench_reps": int(bench_reps),
        }
        return f
    return wrap(fn) if fn is not None else wrap


def _canon_registry() -> dict:
    """The registry on the package-named module instance.

    ``python -m forge.evolution.auto_targets`` loads this file as
    ``__main__`` — a SECOND module object whose own _registry stays
    empty while the decorators (run via ``from forge.evolution.
    auto_targets import self_improve`` inside production modules)
    populate the canonical instance's. Resolve through sys.modules so
    both invocation styles see the same registrations."""
    import sys
    canon = sys.modules.get("forge.evolution.auto_targets")
    if canon is not None and canon is not sys.modules.get(__name__):
        return canon._registry
    return _registry


def registered() -> dict:
    """name -> spec metadata for every @self_improve site already run."""
    return dict(_canon_registry())


def _callable_source(obj) -> str:
    """Source of a function/class with decorators stripped."""
    src = textwrap.dedent(inspect.getsource(obj))
    tree = ast.parse(src)
    node = next(
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)))
    lines = src.splitlines()
    return "\n".join(lines[node.lineno - 1:])


def _gen_source(gen) -> str:
    """Lift the case generator into the spec as ``_gen_cases()``."""
    src = _callable_source(gen)
    out = re.sub(r"def\s+" + re.escape(gen.__name__) + r"\s*\(",
                 "def _gen_cases(", src, count=1)
    if "_gen_cases(" not in out:
        raise ValueError(
            f"gen_input {gen.__name__!r}: could not lift def — "
            "it must be a module-level `def name():` function")
    return out


def _build_spec(key: str, spec: dict) -> tuple[str, str]:
    """-> (target_name, spec_file_text)."""
    fn = spec["fn"]
    name = PREFIX + key
    origin = f"{fn.__module__}:{fn.__name__}"

    seed_parts = list(_SPEC_HEADER_IMPORTS)
    seed_parts += [i for i in spec["imports"]
                   if i not in _SPEC_HEADER_IMPORTS]
    seed_parts += [f"{k} = {v!r}" for k, v in spec["globals"].items()]
    g = fn.__globals__
    for dep in spec["deps"]:
        if dep not in g:
            raise ValueError(
                f"{origin}: dep {dep!r} not found in module globals")
        seed_parts.append(_callable_source(g[dep]))
    seed_parts.append(_callable_source(fn))
    seed_code = "\n\n".join(seed_parts) + "\n"

    prompt = spec["prompt"] or (
        f"Rewrite {fn.__name__} to be faster ({spec['metric']} "
        f"{'minimize' if spec['goal'] == 'min' else 'maximize'}). "
        "Keep the public name and identical output semantics on the "
        "synthesized cases; stdlib+numpy+torch only.")
    gen_src = _gen_source(spec["gen_input"])

    header = (
        f'"""Auto-harvested evolution target: {origin}.\n\n'
        "GENERATED by forge/evolution/auto_targets.py::collect_targets() "
        "— do not edit by hand. Regenerate:\n"
        "    python -m forge.evolution.auto_targets\n"
        '"""\n\n'
        f"NAME = {name!r}\n"
        f"GOAL = {spec['goal']!r}\n"
        f"METRIC = {spec['metric']!r}\n"
        f"TAGS = {list(spec['tags'])!r}\n"
        f"BUDGETS = {spec['budgets']!r}\n"
        f"BENCH_REPS = {spec['bench_reps']}\n"
        f"REQUIRED = [{fn.__name__!r}]\n"
        f"PROMPT = {prompt!r}\n\n"
        f"SEED_CODE = {seed_code!r}\n\n"
    )
    text = (header
            + "\n".join(_SPEC_HEADER_IMPORTS)
            + "\n\n" + gen_src + "\n" + _ORACLE)
    return name, text


def collect_targets(out_dir: str | Path | None = None,
                    modules=AUTO_MODULES, write: bool = True,
                    prune: bool = True) -> dict[str, int]:
    """Materialize auto_* spec files for every registered function.

    Imports ``modules`` (runs their decorators), writes one
    ``auto_<name>.py`` spec per entry under ``out_dir`` (default:
    ``targets/``), prunes stale ``auto_*`` specs whose registration
    vanished, and returns ``{target_name: eval_budget}`` — merge
    straight into the rsi_evolve target dict.
    """
    out = Path(out_dir) if out_dir is not None else TARGETS_DIR
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            print(f"[auto_targets] import {mod} failed: {e}")
    emitted: dict[str, int] = {}
    files: set[str] = set()
    reg = _canon_registry()
    for key in sorted(reg):
        spec = reg[key]
        try:
            name, text = _build_spec(key, spec)
        except Exception as e:
            print(f"[auto_targets] {key}: harvest failed: {e}")
            continue
        emitted[name] = spec["evals"]
        if not write:
            continue
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{name}.py"
        files.add(p.name)
        if not p.exists() or p.read_text(encoding="utf-8") != text:
            p.write_text(text, encoding="utf-8")
    if write and prune and out == TARGETS_DIR:
        for stale in out.glob(f"{PREFIX}*.py"):
            if stale.name not in files:
                stale.unlink()
                print(f"[auto_targets] pruned stale {stale.name}")
    return emitted


# ── generic oracle (appended to every generated spec) ───────────────────
# Semantics: candidate output must near-equal the reference on the
# synthesized cases. Tensors compare on finite-mask + allclose (the -inf
# support IS the semantics for masking kernels); complex tensors compare
# on real+imag finite masks; nested list/tuple/dict recurse; numbers use
# isclose; everything else falls back to ==.

_ORACLE = '''

def _ref():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns[REQUIRED[0]]


def _same(a, b, atol=1e-5, rtol=1e-4):
    import math as _m
    ta, tb = torch.is_tensor(a), torch.is_tensor(b)
    if ta or tb:
        if not (ta and tb) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if torch.equal(a, b):
            return True
        if a.is_complex() or b.is_complex():
            fa = torch.isfinite(a.real) & torch.isfinite(a.imag)
            fb = torch.isfinite(b.real) & torch.isfinite(b.imag)
        else:
            fa, fb = torch.isfinite(a), torch.isfinite(b)
        if not torch.equal(fa, fb):
            return False
        return bool(torch.allclose(a[fa], b[fb], atol=atol, rtol=rtol))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _same(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _same(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return _m.isclose(float(a), float(b),
                              abs_tol=atol, rel_tol=rtol)
        except (TypeError, ValueError):
            return a == b
    return a == b


def _call(fn, case):
    return fn(*case) if isinstance(case, tuple) else fn(case)


def _any_cuda(obj, depth=0):
    if depth > 4:
        return False
    if torch.is_tensor(obj):
        return bool(obj.is_cuda)
    if isinstance(obj, (list, tuple)):
        return any(_any_cuda(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return any(_any_cuda(x, depth + 1) for x in obj.values())
    return False


def test(ns):
    fn = ns.get(REQUIRED[0])
    if not callable(fn):
        return {"ok": False, "detail": f"no {REQUIRED[0]}"}
    ref = _ref()
    cases_a = _gen_cases()
    cases_b = _gen_cases()          # fresh inputs — mutation-safe
    for i, (ca, cb) in enumerate(zip(cases_a, cases_b)):
        try:
            y = _call(fn, ca)
            y0 = _call(ref, cb)
        except Exception as e:
            return {"ok": False,
                    "detail": f"case {i} raised {type(e).__name__}: {e}"}
        try:
            if not _same(y, y0):
                return {"ok": False, "detail": f"case {i} mismatch"}
        except Exception as e:
            return {"ok": False, "detail": f"case {i} compare {e}"}
    return {"ok": True}


def bench(ns):
    fn = ns[REQUIRED[0]]
    cases = _gen_cases()
    for c in cases:
        _call(fn, c)
    on_cuda = _any_cuda(cases)
    if on_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(BENCH_REPS):
        for c in cases:
            _call(fn, c)
            n += 1
    if on_cuda:
        torch.cuda.synchronize()
    return {"ms_per_call": (time.perf_counter() - t0) / max(n, 1) * 1000}
'''


if __name__ == "__main__":
    t = collect_targets()
    for k, v in sorted(t.items()):
        print(f"{k}: {v} evals")
    print(f"{len(t)} auto targets -> {TARGETS_DIR}")

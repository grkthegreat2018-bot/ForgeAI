"""Tests for the code-evolution loop (forge/evolution/code_evolve.py).

Pure-python dummy target + scripted proposer — no LLM, no GPU. Each
subprocess eval is ~0.5s; budgets kept tiny.
"""

from __future__ import annotations

import textwrap

import pytest

from forge.evolution.code_evolve import (
    _OPS,
    CodeEvolve,
    CodeEvolveConfig,
    LLMProposer,
    Target,
    _extract_code,
    _norm_hash,
)

DUMMY = textwrap.dedent("""\
    NAME = "fast_sum"
    GOAL = "min"
    METRIC = "ms_per_call"
    TAGS = ["test", "microbench"]
    BUDGETS = {}
    REQUIRED = ["fast_sum"]
    SEED_CODE = "def fast_sum(xs):\\n    t = 0.0\\n    for x in xs:\\n        t += x\\n    return t\\n"

    def _inputs():
        return [[list(range(10000))]]

    def test(ns):
        fn = ns.get("fast_sum")
        if not callable(fn):
            return {"ok": False, "detail": "no fast_sum"}
        for i, args in enumerate(_inputs()):
            try:
                y = fn(*args)
            except Exception as e:
                return {"ok": False, "detail": f"input {i} raised {e}"}
            if abs(y - 49995000.0) > 1e-6:
                return {"ok": False, "detail": f"input {i} wrong"}
        return {"ok": True}

    def bench(ns):
        import time
        fn = ns["fast_sum"]
        a = _inputs()[0]
        t0 = time.perf_counter()
        for _ in range(30):
            fn(*a)
        return {"ms_per_call": (time.perf_counter() - t0) / 30 * 1000}
    """)

FAST = "def fast_sum(xs):\n    return sum(xs)\n"
WRONG = "def fast_sum(xs):\n    return 0.0\n"
BROKEN = "def fast_sum(xs)\n    return sum(xs)\n"  # syntax error


@pytest.fixture()
def target_file(tmp_path):
    p = tmp_path / "fast_sum.py"
    p.write_text(DUMMY, encoding="utf-8")
    return str(p)


def _ce(target_file, tmp_path, **kw):
    cfg = CodeEvolveConfig(
        target=target_file,
        db_path=str(tmp_path / "cev.db"),
        proposer="none",
        reps=1,
        evals_budget=10,
        islands=1,
        patience=3,
        hop_after=99,
        timeout_s=30,
        verbose=False,
        **kw,
    )
    return cfg


class _Scripted:
    """Proposer that returns a fixed queue of programs, then repeats."""

    def __init__(self, programs):
        self.programs = list(programs)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if not self.programs:
            return "pass\n"
        return self.programs.pop(0)


def test_extract_code():
    raw = "text\n```python\ndef f():\n    return 1\n```\n# Write more\nx = 2\n"
    assert "def f" in _extract_code(raw)
    assert "# Write more" not in _extract_code(raw)


def test_extract_code_unclosed_fence_and_prime():
    # truncated generation: unclosed fence -> take tail after last ```
    raw = "blah\n```python\ndef f():\n    return 1\n    return"
    assert "return 1" in _extract_code(raw)
    # primed completion: raw continues a def line -> re-attach prime
    prime = "import torch\n\n\ndef f(x: torch.Tensor) -> torch.Tensor:"
    raw = "\n    return x.relu()"
    code = _extract_code(raw, prime)
    assert code.startswith("import torch")
    assert "return x.relu()" in code
    compile(code, "<c>", "exec")
    # salvage: trailing prose that breaks compile gets trimmed
    raw = "def f():\n    return 1\nHere is why this is faster: no sort!"
    assert compile(_extract_code(raw), "<c>", "exec") is not None


def test_norm_hash_ignores_comments():
    a = "def f():\n    return 1  # hi\n"
    b = "def f():\n    # different comment\n    return 1\n"
    assert _norm_hash(a) == _norm_hash(b)


def test_promotes_faster_impl(target_file, tmp_path):
    """Builtin sum() is ~10x faster than a python loop — must promote."""
    prop = _Scripted([FAST, WRONG, BROKEN, FAST])
    cfg = _ce(target_file, tmp_path)
    res = CodeEvolve(cfg, propose_fn=prop).run()
    assert "sum(xs)" in res["best_code"]
    assert res["improvement_pct"] > 10.0
    # insight written for the verified win (fresh conn — run() closes its own)
    import sqlite3

    n = sqlite3.connect(cfg.db_path).execute("SELECT COUNT(*) FROM code_insights").fetchone()[0]
    assert n >= 1


def test_wrong_impl_never_promoted(target_file, tmp_path):
    prop = _Scripted([WRONG, WRONG, WRONG, WRONG])
    cfg = _ce(target_file, tmp_path)
    res = CodeEvolve(cfg, propose_fn=prop).run()
    assert "for x in xs" in res["best_code"]  # still the seed
    assert res["improvement_pct"] == pytest.approx(0.0)


def test_stagnation_stops(target_file, tmp_path):
    """Patience=3, proposer only emits duplicates → stop early."""
    prop = _Scripted([FAST] + ["pass\n"] * 50)
    cfg = _ce(target_file, tmp_path)
    res = CodeEvolve(cfg, propose_fn=prop).run()
    assert "stagnation" in res["stop_reason"]
    assert res["evals"] <= 6


def test_budget_gate(target_file, tmp_path):
    """Impossible budget → better impl cannot be promoted."""
    prop = _Scripted([FAST] * 4)
    cfg = _ce(target_file, tmp_path)
    cfg_t = Target(target_file)
    cfg_t.budgets["ms_per_call"] = ("<=", 1e-9)  # unattainable
    # monkeypatch the loaded target's budgets via a new spec file

    p = tmp_path / "fast_sum_gated.py"
    src = DUMMY.replace("BUDGETS = {}", 'BUDGETS = {"ms_per_call": ("<=", 1e-9)}')
    p.write_text(src, encoding="utf-8")
    cfg = CodeEvolveConfig(
        target=str(p),
        db_path=str(tmp_path / "b.db"),
        proposer="none",
        reps=1,
        evals_budget=6,
        islands=1,
        patience=4,
        timeout_s=30,
        verbose=False,
    )
    res = CodeEvolve(cfg, propose_fn=prop).run()
    assert "for x in xs" in res["best_code"]


def test_programs_logged(target_file, tmp_path):
    prop = _Scripted([FAST, WRONG])
    cfg = _ce(target_file, tmp_path)
    cfg.judge = False  # scripted queue must not feed judge calls
    ce = CodeEvolve(cfg, propose_fn=prop)
    ce.run()
    rows = ce.db.conn.execute("SELECT stage, ok FROM code_programs").fetchall()
    stages = {r[0] for r in rows}
    assert "done" in stages and "test" in stages


def test_resume_continues_from_prior_best(target_file, tmp_path):
    """Run 1 promotes FAST; run 2 on same db must START there, not at seed."""
    db = tmp_path / "cev.db"
    cfg1 = _ce(target_file, tmp_path)
    cfg1.db_path = str(db)
    CodeEvolve(cfg1, propose_fn=_Scripted([FAST])).run()
    cfg2 = _ce(target_file, tmp_path)
    cfg2.db_path = str(db)
    ce2 = CodeEvolve(cfg2, propose_fn=_Scripted(["pass\n"] * 30))
    res = ce2.run()
    assert "sum(xs)" in res["best_code"]
    assert res["improvement_pct"] > 0  # seed was beaten before run 2 began


def test_mutator_bandit_persists(tmp_path):
    """Op stats written by note() are visible to a fresh Mutator — the
    policy literally resumes across runs."""
    import sqlite3

    from forge.evolution.code_evolve import _CODE_DDL
    from forge.evolution.mutator import Mutator

    conn = sqlite3.connect(str(tmp_path / "m.db"))
    for stmt in _CODE_DDL:
        conn.execute(stmt)
    conn.commit()

    class _DB:  # minimal conn shim — Mutator only needs .conn
        def __init__(self, c):
            self.conn = c

    db = _DB(conn)
    m = Mutator(seed=0, db=db, tag_sig="t1")
    code = m("```python\ndef f(x):\n    return x\n```\n")
    m.note(code, 1.0, 0.5)
    rows = conn.execute("SELECT op, alpha FROM mutator_op_stats").fetchall()
    assert rows and all(a > 1.0 for _, a in rows)
    m2 = Mutator(seed=0, db=db, tag_sig="t1")
    assert any(a > 1.0 for a, _b in m2._stats.values())


def test_export_pairs(target_file, tmp_path):
    from forge.evolution.code_evolve import export_pairs

    db = tmp_path / "cev.db"
    cfg = _ce(target_file, tmp_path)
    cfg.db_path = str(db)
    CodeEvolve(cfg, propose_fn=_Scripted([FAST])).run()
    out = tmp_path / "pairs.jsonl"
    n = export_pairs(str(db), str(out))
    txt = out.read_text(encoding="utf-8")
    assert n >= 1 and "sum(xs)" in txt and '"rejected"' in txt


def test_literal_never_touches_call_args():
    """Regression for the min_k drift: a linspace endpoint perturbation
    changed real filtering behavior while passing the old gate. Tuning
    literals live in Assign/Compare positions only."""
    import ast

    import numpy as np

    from forge.evolution.mutator import _LiteralPerturb

    src = (
        "import torch\n"
        "def f(x):\n"
        "    eps = 1e-8\n"
        "    w = torch.linspace(1.0, 0.1, 5)\n"
        "    if x > 0.5:\n"
        "        return x * 2\n"
        "    return x\n"
    )
    for seed in range(30):
        t = _LiteralPerturb(np.random.default_rng(seed)).visit(ast.parse(src))
        out = ast.unparse(t)
        assert "0.1, 5" in out  # ctor args immutable
    perturbed = False
    for seed in range(60):
        t = _LiteralPerturb(np.random.default_rng(seed)).visit(ast.parse(src))
        out = ast.unparse(t)
        if "1e-08" not in out or "0.5" not in out:
            perturbed = True
    assert perturbed  # assign/compare CAN drift


def test_workers_auto_smoke(target_file, tmp_path):
    """workers=0 -> auto ramp: run completes, stagnation still enforces."""
    cfg = _ce(target_file, tmp_path)
    cfg.workers = 0
    cfg.max_workers = 4
    cfg.evals_budget = 6
    cfg.patience = 4
    res = CodeEvolve(cfg, propose_fn=_Scripted([FAST] * 8)).run()
    assert "error" not in res
    assert res["evals"] >= 1


def test_mutants_for_unique_and_deterministic():
    """op--site enumeration: every emitted variant is a distinct op_sig
    AND distinct code — uniqueness by construction, the fix for the
    measured 91.6% dup-reject rate."""
    import ast

    from forge.evolution.mutator import Mutator

    src = (
        "import torch\n"
        "def f(x):\n"
        "    y = x.masked_fill(x > 0.5, 0.0)\n"
        "    z = x.masked_fill(x < -0.5, 0.0)\n"
        "    s, i = torch.sort(y + z, dim=-1)\n"
        "    t = s.sort(dim=-1).values\n"
        "    eps = 1e-4\n"
        "    if eps > 0.5:\n"
        "        return s\n"
        "    return t\n"
    )
    fam1 = list(Mutator(seed=0).mutants_for(src, seed_base=3))
    assert len(fam1) >= 6
    sigs = [s for s, _ in fam1]
    codes = [c for _, c in fam1]
    assert len(set(sigs)) == len(sigs)  # distinct op keys
    assert len(set(codes)) == len(codes)  # distinct programs
    canon = ast.unparse(ast.parse(src))
    for _sig, code in fam1:
        assert code != canon  # no unchanged trees
        compile(code, "<m>", "exec")  # always syntactically valid
    # Same seed_base -> identical family (deterministic enumeration)
    fam2 = [c for _, c in Mutator(seed=0).mutants_for(src, seed_base=3)]
    assert fam2 == codes
    # Bandit bookkeeping: emitted codes registered for note() credit
    m = Mutator(seed=0)
    fam3 = list(m.mutants_for(src, seed_base=0))
    assert all(c in m._ops_by_code for _, c in fam3)


def test_mutants_for_thin_seed_surface():
    """Single-return seeds must still enumerate — _LocalsBind binds
    attrs used >=1x (was >=3x, which left zero mutation surface on the
    canon_key target and stalled the basin path before first emit)."""
    from forge.evolution.mutator import Mutator

    src = "import ast\n\n\ndef canon_code(src):\n    return ast.unparse(ast.parse(src))\n"
    fam = list(Mutator(seed=0).mutants_for(src, seed_base=0))
    assert len(fam) >= 1
    assert any(s.startswith("locals_bind") for s, _ in fam)
    for _sig, code in fam:
        compile(code, "<m>", "exec")


def test_eval_pool_sync_and_many(tmp_path):
    """Persistent workers: eval_sync + eval_many evaluate the dummy
    target without subprocess-per-eval; results in submitted order."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "fast_sum.py"
    p.write_text(DUMMY, encoding="utf-8")
    pool = EvalPool(str(p), n_workers=2, timeout_s=60)
    try:
        res = pool.eval_sync("def fast_sum(xs):\n    return sum(xs)\n")
        assert res["ok"] and res["stage"] == "done"
        assert res["metrics"]["ms_per_call"] > 0
        bad = pool.eval_sync("def fast_sum(xs):\n    return 0.0\n")
        assert not bad["ok"] and bad["stage"] == "test"
        ress = pool.eval_many(["def fast_sum(xs):\n    return sum(xs)\n", "def fast_sum(xs):\n    return 0.0\n"])
        assert ress[0]["ok"] and not ress[1]["ok"]
    finally:
        pool.close()


def test_incumbent_verify_cache(target_file, tmp_path):
    """_verified re-benches the incumbent only every inc_stale evals —
    challenger reps always fresh, incumbent mean cached."""
    cfg = _ce(target_file, tmp_path)
    cfg.reps = 2
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    calls = {"n": 0}

    real_eval = ce._eval

    def spy(code, wd, tag="x"):
        calls["n"] += 1
        return real_eval(code, wd, tag)

    ce._eval = spy
    ce.incumbent_code = ce.target.seed_code
    ce.incumbent_score = -1.0
    fast = "def fast_sum(xs):\n    return sum(xs)\n"
    ce._verified(fast, -0.01, workdir)
    first = calls["n"]  # 1 challenger + 2 incumbent
    assert first == 3
    ce._verified(fast, -0.01, workdir)
    second = calls["n"] - first  # challenger only — incumbent cached
    assert second == 1
    ce._eval_n += cfg.inc_stale + 1
    ce._verified(fast, -0.01, workdir)
    assert calls["n"] - first - second == 3  # cache expired -> re-bench


def test_no_regression_ratchet(target_file, tmp_path):
    """A challenger that beats a noisy incumbent re-measure but NOT the
    recorded best must not promote — incumbent_score is non-decreasing.
    (quantize_4bit ratchet: 0.294 -> 0.345 -> 0.703 stored as 'wins'.)"""
    cfg = _ce(target_file, tmp_path)
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    ce.incumbent_code = ce.target.seed_code
    ce.incumbent_score = -0.02  # recorded best: 0.02 ms/call

    # First-eval gate passes (-0.015 > -0.02), but the verified mean
    # lands BELOW the recorded best — a regression wearing a win.
    ce._verified = lambda code, cs, wd: (
        True,
        {"challenger_mean": -0.03, "incumbent_mean": -0.05, "reps": 3, "margin": 0.001, "delta": 0.02, "z": 2.0},
    )
    worse = "def fast_sum(xs):\n    return sum(xs) + 0\n"
    ce._handle_result(worse, 0, None, "h1", {"ok": True, "stage": "done", "metrics": {"ms_per_call": 0.015}}, workdir)
    assert ce.incumbent_score == -0.02  # bar did not drop
    assert ce.incumbent_code != worse

    # A challenger whose verified mean really does beat the recorded
    # best still promotes.
    ce._verified = lambda code, cs, wd: (
        True,
        {"challenger_mean": -0.01, "incumbent_mean": -0.05, "reps": 3, "margin": 0.001, "delta": 0.04, "z": 3.0},
    )
    better = "def fast_sum(xs):\n    return sum(xs)\n"
    ce._handle_result(better, 0, None, "h2", {"ok": True, "stage": "done", "metrics": {"ms_per_call": 0.009}}, workdir)
    assert ce.incumbent_score == -0.01
    assert ce.incumbent_code == better


def test_basin_mode_unique_proposals(target_file, tmp_path):
    """basin proposer: queued LLM parents fan out into unique-by-
    construction candidates through the gate — no dup slot waste."""
    import queue as _q

    cfg = _ce(target_file, tmp_path)
    cfg.proposer = "basin"
    cfg.patience = 30
    ce = CodeEvolve(cfg, propose_fn=None)
    ce._basin_llm = None  # families only — no emitter thread
    ce.incumbent_code = ce.target.seed_code
    rich = (
        "import torch\n"
        "def fast_sum(xs):\n"
        "    t = torch.tensor(xs, dtype=torch.float64)\n"
        "    y = t.masked_fill(t < 0.5, 0.0)\n"
        "    s, i = torch.sort(y, dim=-1)\n"
        "    if 0.5 > 0.1:\n"
        "        return float(s.sum())\n"
        "    return float(y.sum())\n"
    )
    ce._basin_q = _q.Queue()
    for _ in range(3):
        ce._basin_q.put({"code": rich, "island": 0, "prompt": "p", "parent_h": None, "adapter": None})
    cands = []
    for _ in range(60):
        c = ce._propose_basin()
        if isinstance(c, tuple):
            cands.append(c)
        if len(cands) >= 12:
            break
    hashes = {c[3] for c in cands}
    assert len(hashes) == len(cands) >= 4  # distinct, non-trivial yield


def test_eval_pool_recycle_requeue(tmp_path):
    """max_jobs=1: worker exits after each result. The next submit must
    enqueue onto the RESPAWNED worker's queue — the old code put the
    job into the dead worker's job_q where it could never be read."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "fast_sum.py"
    p.write_text(DUMMY, encoding="utf-8")
    pool = EvalPool(str(p), n_workers=1, timeout_s=60, max_jobs=1)
    try:
        r1 = pool.eval_sync("def fast_sum(xs):\n    return sum(xs)\n")
        assert r1["ok"] and r1["stage"] == "done"
        r2 = pool.eval_sync("def fast_sum(xs):\n    return sum(list(xs))\n")
        assert r2["ok"] and r2["stage"] == "done"
    finally:
        pool.close()


def test_eval_pool_timeout_kill(tmp_path):
    """A hung candidate must not wedge the pool: deadline breach kills
    the worker, job reports stage=timeout, and the respawned worker
    keeps evaluating."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "fast_sum.py"
    p.write_text(DUMMY, encoding="utf-8")
    pool = EvalPool(str(p), n_workers=1, timeout_s=4)
    try:
        hung = "def fast_sum(xs):\n    while True:\n        pass\n"
        res = pool.eval_sync(hung, timeout_s=3)
        assert not res["ok"] and res["stage"] == "timeout"
        good = pool.eval_sync("def fast_sum(xs):\n    return sum(xs)\n")
        assert good["ok"] and good["stage"] == "done"
    finally:
        pool.close()


def test_eval_pool_boot_breaker(tmp_path):
    """A target that crashes on import used to respawn-burn forever —
    every proposal spawned a worker that died instantly. The breaker
    must trip: consecutive no-completion deaths open the circuit and
    submit() returns staged spawn failures without touching workers."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "bad_target.py"
    p.write_text("import os\nraise RuntimeError('cannot load')\n", encoding="utf-8")
    pool = EvalPool(str(p), n_workers=1, timeout_s=60)
    pool._boot_trip = 2
    try:
        code = "def fast_sum(xs):\n    return sum(xs)\n"
        res = pool.eval_sync(code)
        assert not res["ok"] and res["stage"] == "spawn"
        pool.eval_sync(code)
        pool.eval_sync(code)
        assert pool._broken is not None
    finally:
        pool.close()


def test_db_lazy_reconnect(tmp_path):
    """run() releases the WAL conn between campaign rounds; the conn
    property must lazily reopen so post-run reads still work, while
    close() stays terminal (context-manager/__exit__ contract)."""
    import sqlite3 as _sq

    from forge.evolution.database import FindingsDB

    db = FindingsDB(tmp_path / "x.db")
    db.conn.execute("SELECT 1")
    db.release()
    assert db.conn.execute("SELECT 1").fetchone() == (1,)
    db.close()
    try:
        db.conn  # noqa: B018 — intentional access; must raise after close()
        raise AssertionError("close() must be terminal")
    except _sq.ProgrammingError:
        pass


def test_lazy_package_import():
    """forge.evolution.code_evolve must NOT pull the legacy heavy chain
    (engine.py -> torch + model stack) — eval workers pay that import
    at spawn otherwise. Checked in a fresh interpreter."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = repo
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, forge.evolution.code_evolve as m; "
            "sys.exit(0 if 'forge.evolution.engine' not in sys.modules "
            "else 3)",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-300:]


def test_promos_counted(target_file, tmp_path):
    """_promos must increment on verified promotion — status files and
    the run record count on it."""
    cfg = _ce(target_file, tmp_path)
    cfg.status_file = str(tmp_path / "st.json")
    ce = CodeEvolve(cfg, propose_fn=_Scripted([FAST] * 4))
    res = ce.run()
    assert res["promotions"] >= 1 and ce._promos >= 1
    import json

    st = json.loads((tmp_path / "st.json").read_text())
    assert st["promos"] >= 1


def test_new_engine_targets_self_consistent():
    """Every shipped target's own SEED_CODE must pass its gate — a
    broken gate burns the entire campaign on reject noise."""
    from pathlib import Path

    tdir = Path(__file__).resolve().parents[2] / "forge" / "evolution" / "targets"
    for name in (
        "sample_pipeline",
        "suffix_mem",
        "prefix_hash",
        "stop_matcher",
        "radix_prefix",
        "canon_key",
        "min_k_filter",
        "top_p_filter",
        "top_p_probs",
        "dry_penalties",
        "fast_softmax",
    ):
        t = Target(str(tdir / f"{name}.py"))
        ns: dict = {}
        exec(t.seed_code, ns)
        r = t.test(ns)
        ok = r.get("ok") if isinstance(r, dict) else bool(r)
        assert ok, f"{name}: {r if isinstance(r, dict) else 'gate False'}"


def test_emit_target_scaffold(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = repo
    out = subprocess.run(
        [sys.executable, "scripts/fn_evolve.py", "--emit-target", "forge.evolution.scout:run_scout"],
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-400:]
    compile(out.stdout, "<scaffold>", "exec")
    assert 'REQUIRED = ["run_scout"]' in out.stdout


def test_lora_base_ok_guard(tmp_path):
    """Wrong-base registry adapters must not attach � the loras/
    'default' adapter trained on qwen3b_hone2_fq was crashing every
    basin proposal with a 896-vs-2048 shape error on the mla_moe
    proposer. Base-matched entries and unknown bases still attach."""
    from pathlib import Path
    from types import SimpleNamespace

    from forge.evolution.code_evolve import _lora_base_ok

    repo = Path(__file__).resolve().parents[2]
    mla = str((repo / "forge" / "evolution" / "proposer_mla_moe").resolve())
    # relative dir base, different checkpoint -> reject
    ent = SimpleNamespace(base="research\\checkpoints\\qwen3b_hone2_fq")
    assert not _lora_base_ok(ent, mla)
    # same checkpoint dir (resolved absolute) -> accept
    ent = SimpleNamespace(base=mla)
    assert _lora_base_ok(ent, mla)
    # moved/renamed absolute path with matching dir tail -> accept
    ent = SimpleNamespace(base=str(tmp_path / "other" / "place" / "proposer_mla_moe"))
    assert _lora_base_ok(ent, mla)
    # HF-id style base matching the ckpt dir tail -> accept
    ent = SimpleNamespace(base="proposer_mla_moe")
    assert _lora_base_ok(ent, mla)
    # unknown provenance / no engine ckpt -> try it (old behavior)
    assert _lora_base_ok(SimpleNamespace(base=""), mla)
    assert _lora_base_ok(SimpleNamespace(base="x"), "")


# ── web-research tool calls (LLMProposer SEARCH: loop) ───────────────


class _StubTok:
    eos_token_id = 0

    def decode(self, ids):
        return "x"


class _StubEng:
    """ForgeEngine-shaped stub: generate_batch serves a queue of
    per-call output lists and records every prompt it saw."""

    def __init__(self, responses):
        self.tokenizer = _StubTok()
        self.responses = list(responses)
        self.seen = []

    def generate_batch(self, prompts, **kw):
        self.seen.append(list(prompts))
        if self.responses:
            return self.responses.pop(0)
        return ["```python\ndef f():\n    return 0\n```"] * len(prompts)


def _search_proposer(tmp_path, **kw):
    """An LLMProposer wired for tests — no __init__ (skips the engine
    prewarm), bare-dir model_id -> primed=True, lora_root doubles as the
    jsonl search-cache dir."""
    import threading  # noqa: F401  (caller builds the lock)

    from forge.evolution.code_evolve import LLMProposer

    (tmp_path / "m").mkdir(exist_ok=True)
    prop = LLMProposer.__new__(LLMProposer)
    prop.cfg = CodeEvolveConfig(
        target="x",
        model_id=str(tmp_path / "m"),
        research=True,
        research_budget=8,
        lora_root=str(tmp_path / "loras"),
        proposer_device="cpu",
        verbose=False,
        **kw,
    )
    prop._eng = None
    prop._hf = None
    prop._lora_resolved = False
    prop._lora_cached = None
    prop._router = None
    prop._last_adapter = None
    prop._digest_txt = None
    return prop


def _reset_search_state():
    from forge.evolution.code_evolve import LLMProposer

    LLMProposer._SEARCH_CACHE.clear()
    LLMProposer._SEARCH_CACHE_SEEN.clear()
    LLMProposer._SEARCH_SPENT = 0


def _mock_hits(monkeypatch, hits):
    """Canned search backend; returns a call counter."""
    import forge.evolution.research as res

    calls = {"n": 0}

    def fake_tavily(q, n=4):
        calls["n"] += 1
        return hits

    monkeypatch.setattr(res, "free_search", fake_tavily)
    monkeypatch.setattr(res, "tavily_search", fake_tavily)
    monkeypatch.setattr(res, "exa_search", lambda q, n=4: [])
    return calls


def test_search_intercept_flow(tmp_path, monkeypatch):
    """Model emits 'SEARCH: q' -> canned hits run + injected -> ONE
    batched regen -> code block still parses. The regen prompt must
    carry the hits BEFORE the primed fence."""
    import threading

    _reset_search_state()
    calls = _mock_hits(
        monkeypatch, [{"title": "Fast top-p kernel", "url": "https://x/k", "snippet": "partial sort via torch.topk"}]
    )
    prop = _search_proposer(tmp_path)
    eng = _StubEng(
        [
            # first batch: row 0 asks for a lookup, row 1 answers code
            ["SEARCH: fast top-p sampling kernel\n", "```python\ndef f():\n    return 2\n```"],
            # regen batch (row 0 only): now it writes the code
            ["```python\ndef f():\n    return 1\n```"],
        ]
    )
    prop._engine = lambda: ("forge", eng, threading.Lock())

    raws = prop.propose_batch(["task A\n```python\ndef f(x):", "task B\n```python\ndef f(y):"])
    assert calls["n"] == 1  # one API query
    assert len(eng.seen) == 2  # ONE extra round-trip
    assert len(eng.seen[1]) == 1  # only the SEARCH row regen'd
    regen_prompt = eng.seen[1][0]
    assert "Fast top-p kernel" in regen_prompt
    assert "https://x/k" in regen_prompt
    assert "partial sort" in regen_prompt
    # hits were injected BEFORE the primed ```python fence
    assert regen_prompt.rindex("Fast top-p kernel") < regen_prompt.rindex("```python")
    assert "return 1" in raws[0] and "return 2" in raws[1]
    assert "SEARCH" not in raws[0]  # intercepted pre-parse
    assert "def f" in _extract_code(raws[0])


def test_search_cache_and_budget(tmp_path, monkeypatch):
    """Repeat queries are free (in-process dict + lora-root jsonl); the
    process budget caps uncached API calls."""
    _reset_search_state()
    calls = _mock_hits(monkeypatch, [{"title": "t", "url": "u", "snippet": "s"}])
    prop = _search_proposer(tmp_path)
    assert prop._search("q1")[0]["url"] == "u"
    assert prop._search("q1") == prop._search("q1")
    assert calls["n"] == 1  # cached, no second call
    # jsonl written under lora_root
    cache = tmp_path / "loras" / "research_cache.jsonl"
    assert cache.exists() and '"q1"' in cache.read_text()
    assert '"key"' not in cache.read_text().lower()  # no key persisted
    # budget: 1 remaining slot (research_budget=8 here; spend it)
    prop.cfg.research_budget = calls["n"]  # pretend budget already met
    assert prop._search("q2") is None  # budget exhausted
    # persisted cache reloads into a fresh process state for free
    _reset_search_state()
    prop2 = _search_proposer(tmp_path)
    assert prop2._search("q1")[0]["url"] == "u"
    assert calls["n"] == 1  # reload cost nothing


def test_search_no_regen_when_no_query(tmp_path, monkeypatch):
    """No SEARCH line -> zero API calls, zero extra round-trips; an
    exhausted budget strips the SEARCH line and keeps the raw."""
    import threading

    _reset_search_state()
    calls = _mock_hits(monkeypatch, [{"title": "t", "url": "u", "snippet": "s"}])
    prop = _search_proposer(tmp_path)
    eng = _StubEng([["```python\ndef f():\n    return 3\n```"]])
    prop._engine = lambda: ("forge", eng, threading.Lock())
    raws = prop.propose_batch(["task\n```python\ndef f(x):"])
    assert calls["n"] == 0 and len(eng.seen) == 1
    assert "return 3" in raws[0]

    # budget=0: SEARCH line is stripped, raw returned un-regenerated
    prop.cfg.research_budget = 0
    eng2 = _StubEng([["SEARCH: something\nsome prose"]])
    prop._engine = lambda: ("forge", eng2, threading.Lock())
    raws = prop.propose_batch(["task\n```python\ndef f(x):"])
    assert calls["n"] == 0 and len(eng2.seen) == 1  # no regen
    assert "SEARCH" not in raws[0] and "some prose" in raws[0]


def test_search_query_cap_and_single_path(tmp_path, monkeypatch):
    """Max research_max_queries per proposal (extras ignored); the
    serial __call__ path does the same intercept->regen dance."""
    _reset_search_state()
    calls = _mock_hits(monkeypatch, [{"title": "t", "url": "u", "snippet": "s"}])
    prop = _search_proposer(tmp_path, research_max_queries=2)
    raw = "SEARCH: a\nSEARCH: b\nSEARCH: c\nSEARCH: d\n"
    assert prop._tool_calls(raw) == [("SEARCH", "a"), ("SEARCH", "b")]  # dedup + cap

    class _OneShotEng(_StubEng):
        def generate(self, prompt, **kw):
            self.seen.append(prompt)
            return self.responses.pop(0)

    eng = _OneShotEng(
        [
            "SEARCH: q one\n",
            "```python\ndef f():\n    return 7\n```",
        ]
    )
    import threading

    prop._engine = lambda: ("forge", eng, threading.Lock())
    out = prop("task\n```python\ndef f(x):")
    assert calls["n"] == 1 and len(eng.seen) == 2  # one extra ask
    assert "SEARCH: q one" in eng.seen[1]
    assert "return 7" in out and "SEARCH" not in out


def test_funsearch_k2_shots(target_file, tmp_path):
    """Island with >=2 programs -> prompt shows the pair sorted
    worst->best as v0/v1 and asks for v2 (FunSearch best-shot k=2).
    A lone parent keeps the single 'Current best' framing."""
    cfg = _ce(target_file, tmp_path)
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    ce.incumbent_code = ce.target.seed_code
    ce.incumbent_metrics = {"ms_per_call": 1.0}
    ce.incumbent_score = -1.0
    lo = "def fast_sum(xs):\n    return 1.0\n"
    hi = "def fast_sum(xs):\n    return sum(xs)\n"
    ce._push_elite(0, lo, -2.0, "hlo")
    ce._push_elite(0, hi, -0.5, "hhi")
    parent, insp, shots = ce._parents(0)
    assert shots is not None and len(shots) == 2
    assert shots[0][1] <= shots[1][1]  # v0 worse, v1 better
    prompt = ce._build_prompt(parent, insp, primed=False, shots=shots)
    assert "# v0 (score" in prompt and "# v1 (score" in prompt
    assert "write v2" in prompt
    assert prompt.index("# v0") < prompt.index("# v1")
    # one program on the island -> old single-parent framing
    ce.elites[0] = [(hi, -0.5, "hhi")]
    parent, insp, shots = ce._parents(0)
    assert shots is None
    prompt = ce._build_prompt(parent, insp, primed=False, shots=shots)
    assert "Current best" in prompt and "# v0" not in prompt


def test_repair_regen_on_error(target_file, tmp_path):
    """A REAL error (traceback in detail) queues ONE repair regen; the
    drain gives the proposer the original prompt + the error comment
    and re-evaluates the repaired code. Score-misses (detail
    'failed') and already-repair prompts never queue."""
    cfg = _ce(target_file, tmp_path)
    cfg.judge = False  # repair-queue behavior, not judge coverage
    ce = CodeEvolve(cfg, propose_fn=_Scripted([FAST]))
    ce.incumbent_code = ce.target.seed_code
    ce.incumbent_metrics = {"ms_per_call": 1.0}
    ce.incumbent_score = -1.0
    workdir = tmp_path / "wd"
    workdir.mkdir()
    # bare 'failed' = a score miss, not an artifact -> no repair
    ce._handle_result(WRONG, 0, None, "h0", {"ok": False, "stage": "test", "detail": "failed"}, workdir, prompt="# p0")
    assert not ce._repair_q
    res = {
        "ok": False,
        "stage": "exec",
        "detail": "Traceback (most recent call last):\n"
        '  File "c", line 1, in <m>\n'
        "NameError: name 'xs' is not defined",
    }
    ce._handle_result(WRONG, 0, None, "h1", res, workdir, prompt="# orig prompt")
    assert len(ce._repair_q) == 1
    seen = []
    prop = ce._proposer
    ce._proposer = lambda p: (seen.append(p), prop(p))[1]
    ce._drain_repairs(workdir, None, None, {}, [])
    assert seen and "# Previous attempt failed" in seen[0]
    assert "NameError" in seen[0]
    assert ce._eval_n == 1  # repaired code was re-evaluated
    # the repair marker suppresses chained re-repairs
    ce._handle_result(WRONG, 0, None, "h2", res, workdir, prompt="# orig\n# Previous attempt failed with:\n# x")
    assert not ce._repair_q


def test_fetch_and_insight_tools(tmp_path, monkeypatch):
    """FETCH: <url> pulls page text (jina -> trafilatura fallback) and
    INSIGHT: <topic> serves FindingsDB.code_insights rows by overlap.
    Both share the SEARCH per-proposal cap, inject before the primed
    fence, and trigger ONE batched regen; FETCH consumes the web
    budget, INSIGHT does not."""
    import sqlite3
    import threading

    _reset_search_state()
    import forge.evolution.research as res

    calls = {"fetch": 0}

    def fake_jina(url, timeout=20.0):
        calls["fetch"] += 1
        return "PAGE BODY fastest kernel uses tl.dot for tiles"

    monkeypatch.setattr(res, "jina_fetch", fake_jina)
    monkeypatch.setattr(res, "trafilatura_fetch", lambda url, timeout=20.0: "")
    # insights db with one row matching the INSIGHT topic
    from forge.evolution.code_evolve import _CODE_DDL

    dbp = tmp_path / "ins.db"
    conn = sqlite3.connect(str(dbp))
    for stmt in _CODE_DDL:
        conn.execute(stmt)
    conn.execute(
        "INSERT INTO code_insights(target,tags,kind,summary,delta,score,ts) VALUES (?,?,?,?,?,?,?)",
        ("top_p_probs", "kernel,sampling", "win", "partial-sort beats full sort by 2x", 0.4, -0.1, 1.0),
    )
    conn.commit()
    conn.close()

    prop = _search_proposer(tmp_path, db_path=str(dbp))
    eng = _StubEng(
        [
            # row 0 asks for both tools, row 1 answers code
            ["FETCH: https://x/page\nINSIGHT: partial sort\n", "```python\ndef f():\n    return 2\n```"],
            ["```python\ndef f():\n    return 1\n```"],  # regen for row 0
        ]
    )
    prop._engine = lambda: ("forge", eng, threading.Lock())
    raws = prop.propose_batch(["task A\n```python\ndef f(x):", "task B\n```python\ndef f(y):"])
    assert calls["fetch"] == 1  # one web call total
    assert LLMProposer_spend() == 1  # INSIGHT spent no budget
    assert len(eng.seen) == 2  # ONE extra round-trip
    regen = eng.seen[1][0]
    assert "PAGE BODY" in regen  # fetch text injected
    assert "partial-sort beats full sort" in regen  # insight row
    # injected BEFORE the primed ```python fence
    assert regen.rindex("PAGE BODY") < regen.rindex("```python")
    assert "return 1" in raws[0] and "return 2" in raws[1]
    assert "FETCH" not in raws[0] and "INSIGHT" not in raws[0]
    # shared per-proposal cap: 3 tool lines -> research_max_queries (2)
    raw = "SEARCH: a\nFETCH: https://y\nINSIGHT: z\n"
    kinds = [k for k, _ in prop._tool_calls(raw)]
    assert kinds == ["SEARCH", "FETCH"]


def LLMProposer_spend() -> int:
    from forge.evolution.code_evolve import LLMProposer

    return LLMProposer._SEARCH_SPENT


def test_noncode_domain_specs():
    """forge/evolution/domains/cev_*.py — non-code spec targets must
    resolve via Target.resolve's domains/ leg and pass their own
    test+bench contract (a broken gate burns campaign evals on noise)."""
    from pathlib import Path

    tdir = Path(__file__).resolve().parents[2] / "forge" / "evolution" / "domains"
    for name in ("cev_math_solver", "cev_rle_codec"):
        t = Target(str(tdir / f"{name}.py"))
        # also resolvable by bare name through Target.resolve
        assert Target.resolve(name) == t.path
        ns: dict = {}
        exec(t.seed_code, ns)
        r = t.test(ns)
        assert r.get("ok"), f"{name}: {r}"
        b = t.bench(ns)
        assert t.metric in b, f"{name}: {b}"
        for m, (op, v) in t.budgets.items():
            if m in b:
                assert _OPS[op](b[m], v), f"{name}: seed violates budget {m} {op} {v} ({b[m]})"


# ── prior-art ports: hint bandit, eval cascade, island cull, MAP-elites ──


def test_hint_bandit_attribution(tmp_path):
    """Meta-prompt evolution-lite: note(code, credit, delta, prompt)
    credits the hint that conditioned the proposal's generation and
    persists it to mutator_op_stats as 'hint:<idx>' rows keyed by
    tag_sig — a fresh proposer resumes the learned policy."""
    import sqlite3

    from forge.evolution.code_evolve import _CODE_DDL, LLMProposer

    conn = sqlite3.connect(str(tmp_path / "h.db"))
    for stmt in _CODE_DDL:
        conn.execute(stmt)
    conn.commit()

    class _DB:
        def __init__(self, c):
            self.conn = c

    prop = LLMProposer.__new__(LLMProposer)
    prop.cfg = CodeEvolveConfig(target="x", verbose=False)
    prop.db = _DB(conn)
    prop.tag_sig = "t1"
    prop._hint_by_prompt = {}
    prop._hint_stats = None
    prop._hint_rng = None

    # cold start: uniform arms, valid picks
    assert all(a == 1.0 and b == 1.0 for a, b in prop._hint_state())
    for _ in range(20):
        assert 0 <= prop._pick_hint() < len(prop._HINTS)

    # record hint idx 3 for prompt P, then credit a promotion
    prop._note_hint("prompt P", 3)
    prop.note("some code", 1.0, 0.5, prompt="prompt P")
    assert prop._hint_stats[3][0] > 1.0
    rows = conn.execute("SELECT op, alpha, n FROM mutator_op_stats WHERE op='hint:3'").fetchall()
    assert rows and rows[0][1] > 1.0 and rows[0][2] == 1
    # no attribution -> graceful no-op (mut-made code, missing prompt)
    prop.note("other code", 1.0, 0.5, prompt=None)
    prop.note("other code", 1.0, 0.5, prompt="never recorded")
    # policy persists across instances
    prop2 = LLMProposer.__new__(LLMProposer)
    prop2.cfg = prop.cfg
    prop2.db = prop.db
    prop2.tag_sig = "t1"
    prop2._hint_stats = None
    assert prop2._hint_state()[3][0] > 1.0


def test_island_cull_clone(target_file, tmp_path):
    """FunSearch basin hop: the worst half of islands is discarded;
    the least-bad culled slot gets the best island's elite set (and
    its bandit arm)."""
    cfg = _ce(target_file, tmp_path)
    cfg.islands = 4
    cfg.hop_after = 2
    cfg.patience = 30
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    ce.incumbent_code = ce.target.seed_code
    ce.incumbent_metrics = {"ms_per_call": 1.0}
    ce.incumbent_score = -1.0
    # island scores: 3 worst(-9), 2 bad(-5), 1 ok(-2), 0 best(-0.5)
    for isl, s in ((3, -9.0), (2, -5.0), (1, -2.0), (0, -0.5)):
        ce._push_elite(isl, f"# c{isl}\ndef fast_sum(xs):\n    return 1\n", s, f"h{isl}")
    ce._arms[0] = [5.0, 1.0]
    ce._dry = cfg.hop_after
    assert ce._stagnation_check() is False
    # ranked worst->best: 3, 2 culled; clone slot = 2 gets island 0's set
    assert ce.elites[3] == []
    assert ce.elites[2] == list(ce.elites[0])
    assert ce._arms[2] == [5.0, 1.0]  # clone inherits arm
    assert ce._arms[3] == [1.0, 1.0]  # emptied slot reset fresh


def test_map_elites_len_bins(target_file, tmp_path):
    """MAP-Elites-lite: (island, length-bin) cells — a low-scored LONG
    program survives alongside higher-scored shorts (diversity the
    flat top-N would purge)."""
    cfg = _ce(target_file, tmp_path)
    cfg.n_elites = 2
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    short1 = "def fast_sum(xs):\n    return 1\n"
    short2 = "def fast_sum(xs):\n    return 2\n"
    longp = (
        "def fast_sum(xs):\n"
        "    a = 1  # pad pad pad pad pad pad pad pad pad pad pad\n"
        "    b = 2  # pad pad pad pad pad pad pad pad pad pad pad\n"
        "    return a + b\n"
    )
    ce._push_elite(0, short1, -0.1, "s1")
    ce._push_elite(0, short2, -0.2, "s2")
    ce._push_elite(0, longp, -9.0, "L1")
    kept = {h for _, _, h in ce.elites[0]}
    assert "L1" in kept  # long bin survived
    assert len(kept) <= 2
    # legacy flat behavior with map_elites off
    cfg2 = _ce(target_file, tmp_path)
    cfg2.n_elites = 2
    cfg2.map_elites = False
    ce2 = CodeEvolve(cfg2, propose_fn=_Scripted([]))
    for code, s, h in ((short1, -0.1, "s1"), (short2, -0.2, "s2"), (longp, -9.0, "L1")):
        ce2._push_elite(0, code, s, h)
    assert {h for _, _, h in ce2.elites[0]} == {"s1", "s2"}


def test_cheap_gate_cascade(tmp_path):
    """Eval cascade: a spec with BENCH_REPS runs a reduced-rep bench
    first — a candidate clearly worse than the incumbent reports
    stage='cheap_gate' and skips the full bench; a close one still
    gets the full 'done' eval."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    spec = tmp_path / "cheap_t.py"
    spec.write_text(
        textwrap.dedent("""\
        NAME = "cheap_demo"
        GOAL = "min"
        METRIC = "ms_per_call"
        BENCH_REPS = 3
        REQUIRED = ["work"]
        SEED_CODE = "def work(n):\\n    return n\\n"
        def test(ns):
            return {"ok": callable(ns.get("work"))}
        def bench(ns):
            import time
            fn = ns["work"]
            t0 = time.perf_counter()
            for _ in range(BENCH_REPS):
                fn(1)
            return {"ms_per_call": (time.perf_counter() - t0) * 1000}
        """),
        encoding="utf-8",
    )
    cand = tmp_path / "cand.py"
    cand.write_text("def work(n):\n    import time\n    time.sleep(0.02)\n    return n\n", encoding="utf-8")
    ev = Path(__file__).resolve().parents[2] / "forge" / "evolution" / "_candidate_eval.py"
    cheap = json.dumps({"reps": 1, "metric": "ms_per_call", "incumbent": 1e-6, "goal": "min", "margin": 1.0})
    out = subprocess.run(
        [sys.executable, "-X", "utf8", str(ev), str(spec), str(cand), cheap], capture_output=True, text=True, timeout=60
    )
    line = next(x for x in out.stdout.splitlines() if x.startswith("__RESULT__"))
    res = json.loads(line[len("__RESULT__") :])
    assert res["ok"] and res["stage"] == "cheap_gate"
    # close/incumbent-beating candidates still pay the full bench
    cheap2 = json.dumps({"reps": 1, "metric": "ms_per_call", "incumbent": 1e9, "goal": "min", "margin": 1.0})
    out2 = subprocess.run(
        [sys.executable, "-X", "utf8", str(ev), str(spec), str(cand), cheap2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    line2 = next(x for x in out2.stdout.splitlines() if x.startswith("__RESULT__"))
    assert json.loads(line2[len("__RESULT__") :])["stage"] == "done"


# ── proposal validity: extraction + gate repair + telemetry ─────────


def test_extract_code_think_block_and_markers():
    """<think> reasoning is stripped BEFORE selection — the answer is
    the post-think block; a fenced fragment inside the think must not
    win. Code inside a closed think still survives via salvage when
    nothing follows it."""
    raw = (
        "<think>first idea\n```python\ndef f():\n    return sorted(x)\n```\n"
        "</think>\n```python\ndef f():\n    return sum(x) / len(x)\n```"
    )
    out = _extract_code(raw)
    assert "sum(x)" in out and "sorted" not in out
    raw = "<think>\n```python\ndef f():\n    return 3\n```\n</think>"
    assert "return 3" in _extract_code(raw)


def test_extract_code_fence_variants_and_preamble():
    """```py / ```Python fences, prose preambles, and uniformly-indented
    programs all recover — they were syntax-gate rejects before."""
    out = _extract_code("Sure! Here is the code:\n```py\ndef f():\n    return 2\n```\nbye")
    compile(out, "<c>", "exec")
    assert "return 2" in out
    out = _extract_code("```Python\ndef f():\n    return 9\n```")
    assert "return 9" in out
    out = _extract_code("Here is the improved version:\ndef f():\n    return 7\n")
    compile(out, "<c>", "exec")
    assert "return 7" in out
    out = _extract_code("    def f():\n        return 5\n")
    compile(out, "<c>", "exec")
    assert "return 5" in out


def test_extract_code_required_preference():
    """A block defining EVERY required name beats an earlier fragment —
    multi-required targets were losing their complete program to the
    first def-block."""
    raw = "```python\ndef helper():\n    return 1\n```\n```python\ndef main():\n    return helper()\n```"
    assert "def main" in _extract_code(raw, required=["main"])
    raw = "```python\ndef a():\n    return 1\n```\n```python\ndef a():\n    return 1\ndef b():\n    return 2\n```"
    out = _extract_code(raw, required=["a", "b"])
    assert "def b" in out


def test_gate_repair_regen(target_file, tmp_path):
    """A syntax-stage reject queues ONE repair regen; the repaired
    candidate re-gates and can promote — invalid output converted into
    a valid proposal instead of a wasted generation."""
    cfg = _ce(target_file, tmp_path)
    cfg.judge = False
    res = CodeEvolve(cfg, propose_fn=_Scripted([BROKEN, WRONG, FAST])).run()
    assert "sum(xs)" in res["best_code"]
    import sqlite3

    rows = (
        sqlite3.connect(cfg.db_path)
        .execute("SELECT prompt, ok FROM code_programs WHERE prompt LIKE '%Previous attempt failed%'")
        .fetchall()
    )
    assert rows, "no repair-regen prompt was logged"
    assert any(ok == 1 for _p, ok in rows)


def test_dup_repair_regen(target_file, tmp_path):
    """A dup reject queues ONE regen carrying the 'already evaluated —
    write DIFFERENT' artifact; the marker still blocks chained
    re-repairs."""
    ALT = "def fast_sum(xs):\n    return sum(list(xs))\n"
    cfg = _ce(target_file, tmp_path)
    cfg.judge = False
    cfg.patience = 8
    # FAST promotes; the second FAST hits the dup gate and queues a
    # regen that drains after ALT's eval (drain pulls "pass\n").
    res = CodeEvolve(cfg, propose_fn=_Scripted([FAST, FAST, ALT] + ["pass\n"] * 30)).run()
    import sqlite3

    rows = sqlite3.connect(cfg.db_path).execute("SELECT stage, COUNT(*) FROM code_programs GROUP BY stage").fetchall()
    stages = dict(rows)
    assert stages.get("dup", 0) >= 1
    p = (
        sqlite3.connect(cfg.db_path)
        .execute("SELECT COUNT(*) FROM code_programs WHERE prompt LIKE '%MATERIALLY DIFFERENT%'")
        .fetchone()[0]
    )
    assert p >= 1
    assert "sum(xs)" in res["best_code"]


def test_proposal_telemetry(target_file, tmp_path):
    """run() reports raws/gated/valid_pct — the dup+soup rates are the
    proposer's quality signal; make the metric observable."""
    cfg = _ce(target_file, tmp_path)
    cfg.judge = False
    res = CodeEvolve(cfg, propose_fn=_Scripted([FAST, WRONG, BROKEN])).run()
    p = res["proposals"]
    assert p["raws"] >= 3 and p["gated"] >= 2
    assert 0 < p["valid_pct"] <= 100


def test_burst_same_batch_dedup(target_file, tmp_path):
    """Identical programs inside ONE burst gate once — repeats log as
    dup_batch and never reach _gate (no island penalty, no _dry)."""

    class _Batch:
        primed = False

        def __call__(self, p):
            return FAST

        def propose_batch(self, prompts):
            return [FAST] * len(prompts)

    cfg = _ce(target_file, tmp_path)
    ce = CodeEvolve(cfg, propose_fn=_Batch())
    ce.incumbent_code = ce.target.seed_code
    ce.incumbent_metrics = {"ms_per_call": 1.0}
    ce.incumbent_score = -1.0
    out = ce._propose_burst(4)
    assert len(out) == 1
    assert ce._stages.get("dup_batch") == 3
    assert ce._dry == 0


def test_tried_approaches_in_prompt(target_file, tmp_path):
    """Approach fingerprints of evaluated programs feed back into the
    prompt as an anti-dup line — requires >=3 distinct approaches so a
    lone fingerprint doesn't just repeat the shown incumbent."""
    cfg = _ce(target_file, tmp_path)
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    assert ce._tried_approaches() == ""
    ce._tried_gists.extend(["sum+sorted", "cumsum+topk", "masked_fill"])
    assert "sum+sorted" in ce._tried_approaches()
    prompt = ce._build_prompt(None, None)
    assert "Already evaluated" in prompt and "masked_fill" in prompt
    # gated candidates record their approach fingerprint
    cand = ce._gate("def fast_sum(xs):\n    return sum(xs)\n", 0, "# p", None)
    assert cand is not None
    assert "sum" in list(ce._tried_gists)[-1]


def test_validity_pairs_export(target_file, tmp_path):
    """export_validity_pairs emits one DPO row per malformed/dup output:
    chosen = gated sibling (same parent, else same target), rejected =
    the model's own bad emission, why = the reject stage. This is the
    format-compliance teaching signal — soup/dup were logged but never
    turned into 'rejected' examples before."""
    import json

    from forge.evolution.code_evolve import export_validity_pairs

    ALT = "def fast_sum(xs):\n    return sum(list(xs))\n"
    cfg = _ce(target_file, tmp_path)
    cfg.judge = False
    CodeEvolve(cfg, propose_fn=_Scripted([FAST, BROKEN, ALT, FAST] + ["pass\n"] * 30)).run()
    out = tmp_path / "vp.jsonl"
    n = export_validity_pairs(cfg.db_path, str(out))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert n >= 2 and len(rows) == n  # syntax + dup at minimum
    whys = {r["why"] for r in rows}
    assert "syntax" in whys and "dup" in whys
    for r in rows:
        assert r["chosen"].startswith("```python")
        assert r["rejected"].strip() and r["prompt"]
        # chosen==rejected is a contradictory DPO pair — the dup row's
        # stored code IS the gated program; export must swap in a
        # different sibling (ALT) or drop the row.
        assert r["chosen"].strip() != r["rejected"].strip()
    dup = next(r for r in rows if r["why"] == "dup")
    assert "sum(list(xs))" in dup["chosen"]


def test_sft_export_primed_body(target_file, tmp_path):
    """SFT rows for PRIMED prompts must emit the body continuation as
    the response — a fenced-restart target teaches the model to abandon
    the prime at inference (train/inference mismatch)."""
    import json

    from forge.evolution.code_evolve import export_sft

    cfg = _ce(target_file, tmp_path)
    cfg.judge = False
    CodeEvolve(cfg, propose_fn=_Scripted([FAST] * 8)).run()
    out = tmp_path / "sft.jsonl"
    n = export_sft(cfg.db_path, str(out))
    assert n >= 1
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    primed = [r for r in rows if "```python" in r["prompt"] and "```" not in r["prompt"].rsplit("```python", 1)[1]]
    assert primed, "no primed-prompt rows exported"
    for r in primed:
        assert not r["response"].startswith("```"), r["response"][:80]
        assert "return sum" in r["response"]


def test_norm_hash_string_comment_collision():
    """'#' inside a string literal is data, not a comment — the old
    split('#') made `x="#"` and `x=""` hash identically, so two real
    alternatives deduped into one."""
    a = 'def f():\n    return "#notacomment"\n'
    b = 'def f():\n    return ""\n'
    c = 'def f():  # trailing comment\n    return "#notacomment"   \n\n'
    assert _norm_hash(a) != _norm_hash(b)
    assert _norm_hash(a) == _norm_hash(c)  # comments/ws still stripped


def test_extract_code_primed_restart_docstring():
    """A primed completion restarting with a module docstring or
    decorator abandoned the prime — gluing it under the signature used
    to produce `def f():\\n\"\"\"doc\"\"\"` soup."""
    prime = "def f(x):\n"
    raw = '"""module doc"""\ndef f(x):\n    return 1\n'
    code = _extract_code(raw, prime)
    assert code.count("def f(") == 1
    compile(code, "<c>", "exec")
    raw = "@torch.jit.script\ndef f(x):\n    return 1\n"
    code = _extract_code(raw, prime)
    assert code.count("def f(") == 1
    assert "@torch.jit.script" in code
    compile(code, "<c>", "exec")


def test_diff_swap_axis():
    """`x[:-1]-x[1:]` slices axis 0, not -1 — the old always--1 rewrite
    silently diffs the wrong axis on multi-dim tensors."""
    import ast

    from forge.evolution.mutator import _DiffSwap

    def tx(src):
        t = _DiffSwap()
        out = ast.unparse(t.visit(ast.parse(src)))
        return out if t.changed else None

    assert "dim=-1" in tx("y = x[..., :-1] - x[..., 1:]\n")
    assert "dim=0" in tx("y = x[:-1] - x[1:]\n")
    assert "dim=0" in tx("y = x[:-1, :] - x[1:, :]\n")
    assert tx("y = x[:-1, :] - x[:, 1:]\n") is None  # different axes


def test_sort_reuse_skips_non_name_targets():
    """`self.a, self.b = torch.sort(x)` must not be rewritten — the old
    transform rebound attribute stores to plain names."""
    import ast

    from forge.evolution.mutator import _SortReuse

    src = "def f(self, x):\n    self.a, self.b = torch.sort(x, dim=-1)\n    return self.a\n"
    t = _SortReuse()
    out = ast.unparse(t.visit(ast.parse(src)))
    assert "self.a, self.b" in out


def test_hoist_ctor_no_aliasing():
    """The _ctor cache returns v.clone() — a caller mutating the result
    must not poison the shared entry (torch.full/ones alias bug)."""
    from forge.evolution.mutator import _CTOR_PREAMBLE, Mutator

    src = "import torch\n\n\ndef f(x):\n    m = torch.full((4,), 0.5)\n    return x + m.sum()\n"
    fam = [c for s, c in Mutator(seed=0).mutants_for(src, seed_base=0) if "hoist_ctor" in s]
    assert fam, "hoist_ctor variant not enumerated"
    assert ".clone()" in _CTOR_PREAMBLE


def test_stagnation_hop_overshoot_once(target_file, tmp_path):
    """_dry can jump PAST hop_after in one basin batch (96 rejects in a
    drain); the hop must still fire — and only once per plateau."""
    cfg = _ce(target_file, tmp_path)
    cfg.hop_after = 5
    cfg.patience = 99
    cfg.islands = 2
    ce = CodeEvolve(cfg, propose_fn=_Scripted([]))
    ce._dry = 30  # jumped right past hop_after
    ce.elites = [[], [("x", 1.0, "s")]]
    t0 = cfg.temperature
    assert ce._stagnation_check() is False
    assert cfg.temperature > t0  # hop fired on the overshoot
    t1 = cfg.temperature
    ce._dry += 10
    assert ce._stagnation_check() is False
    assert cfg.temperature == t1  # no second hop same plateau
    ce._promos += 1  # promotion re-arms the hop
    assert ce._stagnation_check() is False
    assert cfg.temperature > t1


def test_db_none_path_rejected(tmp_path):
    """FindingsDB(None) used to create a literal 'None' SQLite file —
    a 119MB orphan at the repo root (written by the frontier tests).
    None now means in-memory; ''/"None" still fail fast."""
    from forge.evolution.database import FindingsDB

    db = FindingsDB(None)
    assert db.db_path == ":memory:"
    db.close()
    with pytest.raises(ValueError):
        FindingsDB("")
    assert not (tmp_path / "None").exists()


def test_ensure_lora_retries_on_zero_load(target_file, tmp_path):
    """A 0-tensor adapter load must NOT mark _evolve_lora_tag — the old
    mark made later calls short-circuit: warning gone, retry impossible,
    a repaired adapter invisible."""
    import json

    from forge.evolution.code_evolve import LLMProposer

    root = tmp_path / "loras"
    root.mkdir()
    (root / "default.lora.safetensors").write_bytes(b"x")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "tags": {
                    "default": {
                        "file": "default.lora.safetensors",
                        "rank": 8,
                        "alpha": 16,
                        "target_modules": ["q_proj"],
                        "base": "",
                    }
                }
            }
        )
    )
    cfg = _ce(target_file, tmp_path)
    cfg.lora_root = str(root)
    cfg.lora_tag = "default"
    p = LLMProposer.__new__(LLMProposer)
    p.cfg = cfg
    p._lora_cached = None
    p._last_adapter = None
    p._router = None
    p._lora_canary = lambda eng: True

    class _E:
        checkpoint_path = "ckptA"
        model = None  # only used for the warning name

        def __init__(self):
            self.calls = 0

        def _load_lora_impl(self, *a, **k):
            self.calls += 1
            return 0  # key-mismatch: zero tensors

        def unload_lora(self):
            pass

    e = _E()
    p._ensure_lora(e)
    assert getattr(e, "_evolve_lora_tag", None) is None
    p._ensure_lora(e)
    assert e.calls == 2  # retried, not short-circuited

    class _E2(_E):
        def _load_lora_impl(self, *a, **k):
            self.calls += 1
            return 3

    e2 = _E2()
    p._ensure_lora(e2)
    assert e2._evolve_lora_tag == "default"
    p._ensure_lora(e2)
    assert e2.calls == 1  # marked: skip re-attach


def test_wrap_lfm3_official_template():
    """LFM3 ckpts get the official chat_template.jinja shape:
    startoftext BOS + im_start/im_end turns + the empty think-prefill
    the enable_thinking=False path renders. The old generic ChatML wrap
    omitted BOS+think - off-distribution for LFM3-family proposers."""
    p = LLMProposer.__new__(LLMProposer)
    p.cfg = type("C", (), {"proposer_think": False})()
    p._lfm3_cached = True
    p._primed_cached = False
    S, A, E = ("\x3c|startoftext|>", "\x3c|im_start|>", "\x3c|im_end|>")
    T = "\x3cthink>\x3c/think>"
    out = p._wrap("Do the thing.")
    assert out == S + A + "user\nDo the thing." + E + "\n" + A + "assistant\n" + T + "```python\n"
    # primed: signature goes to the assistant turn AFTER empty think
    p._primed_cached = True
    out = p._wrap("instr\n```python\ndef f(x):")
    assert out == S + A + "user\ninstr\n" + E + "\n" + A + "assistant\n" + T + "```python\ndef f(x):"
    # proposer_think -> open think prefill instead of the empty pair
    p._primed_cached = False
    p.cfg.proposer_think = True
    assert p._wrap("x").endswith("assistant\n\x3cthink>")
    # non-LFM3 stays on generic ChatML - no BOS, no think
    p._lfm3_cached = False
    assert p._wrap("x") == A + "user\nx" + E + "\n" + A + "assistant\n"


def test_spark_wrap(tmp_path):
    """Spark-X2.6 uses DeepSeek-style sentence markers + User/Bot turns —
    NOT ChatML (the generic wrap is an untrained format for it)."""
    p = LLMProposer.__new__(LLMProposer)
    p.cfg = type("C", (), {"proposer_think": False})()
    p._spark_cached = True
    p._lfm3_cached = False
    p._primed_cached = False
    sos = "\x3c｜start▁of▁sentence｜>"
    eos = "\x3c｜end▁of▁sentence｜>"
    out = p._wrap("Do the thing.")
    assert out == (sos + "\x3c|User|>Do the thing." + eos + sos
                   + "\x3c|Bot|>\x3c/think>```python\n")
    # think mode opens the reasoning tag instead of the fence
    p.cfg.proposer_think = True
    out = p._wrap("x")
    assert out.endswith("\x3c|Bot|>\x3cthink>")
    # non-Spark models never get the sentence markers
    p._spark_cached = False
    assert "sentence" not in p._wrap("x")


def test_spark_detects_via_tokenizer_config(tmp_path):
    """A renamed local checkpoint still detects Spark via bos_token."""
    import json as _json
    d = tmp_path / "renamed_ckpt"
    d.mkdir()
    bos = "\x3c｜start▁of▁sentence｜>"
    (d / "tokenizer_config.json").write_text(
        _json.dumps({"bos_token": bos}), encoding="utf-8")
    p = LLMProposer.__new__(LLMProposer)
    p.cfg = type("C", (), {"model_id": str(d)})()
    assert p._spark is True
    p2 = LLMProposer.__new__(LLMProposer)
    p2.cfg = type("C", (), {"model_id": "Qwen/Qwen2.5-Coder-3B-Instruct"})()
    assert p2._spark is False

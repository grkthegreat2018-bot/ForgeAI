"""Persistent eval worker pool for CodeEvolve — kills subprocess-per-eval.

Before: every candidate eval spawned `sys.executable _candidate_eval.py`
fresh (~0.5-1.5s spawn on Windows + target import + torch import when the
candidate needs it). Measured sustained rate: ~0.92 evals/min over 26h.

Now: K long-lived spawned processes, each loads the Target spec ONCE and
receives candidate code over an mp.Queue (no file IO). A hung/crashed
worker is terminated and respawned — same isolation contract as the old
per-eval subprocess, with the spawn cost amortized to ~zero.

Workers recycle themselves after `max_jobs` to bound RSS growth from
leaky candidate code (torch-importing candidates especially). The parent
detects a dead worker, marks its in-flight job as "spawn" failure, and
respawns a replacement lazily on next submit.

Memory: `mem_cap_mb` arms an RSS watchdog — poll() samples each busy
worker's RSS (~2 Hz); a worker over the cap is killed+respawned and its
job reports stage="mem". This is the Windows-safe substitute for
resource.setrlimit (Unix-only): a candidate that allocates fast
([0]*10**9 in bench) can't outrun the sampler badly enough to OOM the
host before the kill lands.

VRAM: CUDA-bench targets are capped by the caller (CodeEvolve._cuda_cap)
— workers that touch torch create their own CUDA context (~1.2GB each),
so pool size for CUDA targets should be small (4-6).

Integrity: exec'd candidates share the worker's module state — a
monkeypatched math.sqrt/torch.allclose would silently poison the test
gate and bench of every LATER eval. Workers hold canary refs captured
at init (plus first-sight capture for torch/numpy a candidate imports
mid-loop) and `is`-check them after every eval; drift reports
stage="poison" and the worker exits so the parent respawns clean. The
parent sha256s the target spec on disk (~1 Hz) and restores+flags on
mid-eval writes. Each worker chdirs to a private tempdir so relative
writes (forge_evolve.db, targets/*.py) and relative .env reads miss
the repo root.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue as _queue
import tempfile
import time
import traceback

try:
    import psutil
except ImportError:                     # watchdog degrades, pool still runs
    psutil = None


# ── worker side ─────────────────────────────────────────────────────

_W: dict = {}


def _load_module(path: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clearly_worse(metrics: dict, cheap: dict) -> bool:
    """Eval-cascade decision — mirrors _candidate_eval._clearly_worse:
    reduced-rep metric lands > margin fraction worse than the incumbent
    -> the full bench can't change the outcome, skip it."""
    v = metrics.get(cheap.get("metric", ""))
    inc = cheap.get("incumbent")
    if v is None or inc is None:
        return False
    try:
        v, inc = float(v), float(inc)
    except (TypeError, ValueError):
        return False
    if v != v or inc != inc:                      # NaN guard
        return False
    margin = float(cheap.get("margin", 1.0))
    denom = abs(inc) + 1e-12
    rel = ((v - inc) if cheap.get("goal", "min") == "min"
           else (inc - v)) / denom
    return rel > margin


def _eval_code(target, code: str, cand_name: str, cheap: dict | None = None) -> dict:
    """In-process equivalent of _candidate_eval.main: exec candidate,
    run target.test gate, then target.bench. Same stage contract.
    `cheap` carries the eval-cascade spec ({reps, metric, incumbent,
    goal, margin}) — a reduced-rep bench runs first and clearly-worse
    candidates report stage='cheap_gate' without the full bench."""
    ns: dict = {"__name__": "_candidate"}
    pre = []
    if "math." in code:
        pre.append("import math")
    if "np." in code or "numpy" in code:
        pre.append("import numpy as np")
    if "torch" in code:
        pre.append("import torch\nimport torch.nn.functional as F")
    try:
        if pre:
            exec("\n".join(pre), ns)
        exec(compile(code, cand_name, "exec"), ns)
    except Exception:
        return {"ok": False, "stage": "exec",
                "detail": traceback.format_exc(limit=3)}
    try:
        t = target.test(ns)
        if isinstance(t, dict):
            ok = bool(t.get("ok"))
            detail = t.get("detail", "")
        else:
            ok, detail = bool(t), ""
        if not ok:
            return {"ok": False, "stage": "test",
                    "detail": detail or "failed"}
    except Exception:
        return {"ok": False, "stage": "test",
                "detail": traceback.format_exc(limit=3)}
    # Eval cascade: reduced-rep bench gate before the full bench.
    if cheap:
        try:
            orig = int(getattr(target, "BENCH_REPS", 0) or 0)
        except (TypeError, ValueError):
            orig = 0
        want = int(cheap.get("reps", 1) or 1)
        if orig > want:
            cheap_m = None
            try:
                target.BENCH_REPS = want
                cheap_m = target.bench(ns)
            except Exception:
                cheap_m = None      # full bench re-raises properly
            finally:
                target.BENCH_REPS = orig
            if isinstance(cheap_m, dict) \
                    and _clearly_worse(cheap_m, cheap):
                return {"ok": True, "stage": "cheap_gate",
                        "metrics": cheap_m}
    try:
        metrics = target.bench(ns)
        if not isinstance(metrics, dict):
            return {"ok": False, "stage": "bench",
                    "detail": "bench() must return a metrics dict"}
        return {"ok": True, "stage": "done", "metrics": metrics}
    except Exception:
        return {"ok": False, "stage": "bench",
                "detail": traceback.format_exc(limit=3)}


_MISSING = object()

_WATCH_LATE = {
    "torch": ("allclose", "equal", "isclose"),
    "numpy": ("dot", "allclose", "linalg.norm"),
}


def _canary_resolve(name: str):
    """Re-resolve a dotted attr path (e.g. 'numpy.linalg.norm') from
    sys.modules — catches both attr rebinding and module-object swaps."""
    import sys as _sys
    mod_name, _, rest = name.partition(".")
    obj = _sys.modules.get(mod_name)
    if obj is None:
        return _MISSING
    for part in rest.split("."):
        obj = getattr(obj, part, _MISSING)
        if obj is _MISSING:
            return _MISSING
    return obj


def _canary_snapshot() -> dict:
    """name -> captured object for post-eval `is` checks. The static
    core is always present; torch/numpy only when already imported —
    candidates that pull them in later hit first-sight capture."""
    import builtins
    import math
    import random
    import sys as _sys
    refs = {}
    for mod, attrs in (
        (math, ("sqrt", "exp", "log", "fabs")),
        (time, ("perf_counter", "time")),
        (random, ("random",)),
        (builtins, ("len", "sum", "min", "max", "abs", "round")),
    ):
        for a in attrs:
            refs[f"{mod.__name__}.{a}"] = getattr(mod, a)
    for mod_name, attrs in _WATCH_LATE.items():
        if mod_name in _sys.modules:
            for dotted in attrs:
                name = f"{mod_name}.{dotted}"
                obj = _canary_resolve(name)
                if obj is not _MISSING:
                    refs[name] = obj
    return refs


def _canary_sane(mod_name: str, obj) -> bool:
    """Genuineness check for a late-appearing watched module's attr: a
    real torch/numpy symbol reports __module__ under the package (or is
    a C builtin); a candidate-defined lambda/function does neither."""
    m = getattr(obj, "__module__", None)
    if m is not None:
        return str(m).split(".", 1)[0] == mod_name
    import types
    return callable(obj) and not isinstance(obj, types.FunctionType)


def _canary_check(refs: dict) -> str | None:
    """Post-eval drift check — `is` comparisons only. Returns the
    drifted/poisoned attr name or None. Watched modules appearing in
    sys.modules mid-loop get captured on first sight; a bogus attr on
    first sight (lambda in np.dot) is already poison."""
    import sys as _sys
    for name, orig in refs.items():
        if _canary_resolve(name) is not orig:
            return name
    for mod_name, attrs in _WATCH_LATE.items():
        if mod_name not in _sys.modules:
            continue
        for dotted in attrs:
            name = f"{mod_name}.{dotted}"
            if name in refs:
                continue
            obj = _canary_resolve(name)
            if obj is _MISSING or not _canary_sane(mod_name, obj):
                return name
            refs[name] = obj
    return None


_SECRET_ENV_RE = None


def _scrub_secrets() -> None:
    """Delete secret-looking env vars BEFORE candidate code runs. Eval
    workers are deliberately capability-open (candidates need numpy/torch/
    subprocess for benchmarks) — scrubbing HF_TOKEN/API keys is the cheap
    hardening that keeps that tradeoff honest."""
    global _SECRET_ENV_RE
    if _SECRET_ENV_RE is None:
        import re
        _SECRET_ENV_RE = re.compile(
            r'(token|secret|passw|credential|private|api[_-]?key|auth'
            r'|bearer|(^|_)key($|_))', re.IGNORECASE)
    keep = ('PYTHON', 'CUDA', 'NV', 'TRITON', 'TORCH', 'OMP', 'MKL',
            'OPENBLAS', 'NUMEXPR', 'PATH', 'PATHEXT', 'SYSTEMROOT',
            'SYSTEMDRIVE', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA',
            'LOCALAPPDATA', 'HOMEDRIVE', 'HOMEPATH', 'COMSPEC', 'OS',
            'PROCESSOR', 'WINDIR', 'HF_HOME', 'HF_HUB', 'TRANSFORMERS',
            'VIRTUAL_ENV', 'CONDA', 'NUMBER_OF_PROCESSORS', 'USERNAME',
            'USERDOMAIN', 'DRIVERDATA', 'PROGRAMDATA', 'PROGRAMFILES',
            'PUBLIC', 'SESSIONNAME', 'LOGONSERVER', 'ALLUSERSPROFILE',
            'COMMONPROGRAM', 'PSMODULEPATH', 'TOKENIZERS_PARALLELISM',
            'ONEAPI', 'VS', 'VC', 'INCLUDE', 'INTEL', 'DOTNET', 'GOPATH',
            'CARGO', 'RUST', 'NODE', 'JAVA', 'MSYSTEM', 'MINGW', 'VCPKG',
            'WSL', 'XDG', 'HOME', 'SHELL', 'LANG', 'LC_', 'TZ', 'FORGE_')
    for k in list(os.environ):
        ku = k.upper()
        if _SECRET_ENV_RE.search(ku) and not ku.startswith(keep):
            del os.environ[k]


def _worker_main(job_q, result_q, target_path: str, max_jobs: int,
                 wid: int, init_env: dict) -> None:
    """Worker loop: preload target once, eval candidates until poisoned
    or the recycle bound is hit."""
    os.environ.update(init_env)
    _scrub_secrets()
    # Self-containment: the spawn context inherits PYTHONPATH, but a
    # caller that never set it still needs repo root importable for
    # target specs that import forge.* — anchor it explicitly.
    import sys as _sys
    _root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    # Private cwd: relative writes (forge_evolve.db, targets/*.py) and
    # relative .env reads land in a tempdir, not the repo root.
    os.chdir(tempfile.mkdtemp(prefix="forge_eval_"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "2")
    target = _load_module(target_path, f"_evolve_target_{wid}")
    canaries = _canary_snapshot()
    done = 0
    while True:
        try:
            job = job_q.get(timeout=2.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if job is None:                      # poison pill
            return
        jid, code, cheap = job
        try:
            res = _eval_code(target, code, f"_cand_{jid}", cheap)
        except Exception as e:
            res = {"ok": False, "stage": "spawn",
                   "detail": f"worker eval crash: {e}"}
        drifted = _canary_check(canaries)
        if drifted is not None:
            res = {"ok": False, "stage": "poison",
                   "detail": f"canary drifted: {drifted}"}
        try:
            result_q.put((jid, res), timeout=10)
        except Exception:
            return
        if drifted is not None:
            return                # state untrusted — parent respawns
        done += 1
        if max_jobs and done >= max_jobs:
            return


# ── pool ─────────────────────────────────────────────────────────────

class _Worker:
    __slots__ = ("proc", "job_q", "wid", "outstanding", "deadline",
                 "job_id", "completed", "spawned_at", "job_started")

    def __init__(self, proc, job_q, wid):
        self.proc = proc
        self.job_q = job_q
        self.wid = wid
        self.outstanding = False
        self.deadline = 0.0
        self.job_id = None
        self.completed = 0             # jobs whose result came back
        self.spawned_at = time.time()
        self.job_started = 0.0


class EvalPool:
    """Fixed pool of persistent eval workers.

    submit() binds a job to an idle worker (one-deep queue per worker so
    a hung worker only poisons one job). poll() harvests results and
    enforces per-job timeouts by killing + respawning the offender.
    eval_sync()/eval_many() give the verify path a blocking interface.

    Args:
        target_path: target spec .py — each worker loads it once.
        n_workers: persistent processes.
        timeout_s: per-job wall-clock limit (kill+respawn on breach).
        max_jobs: worker self-recycle bound (RSS creep guard).
        mem_cap_mb: per-worker RSS ceiling in MB; 0 disables the watchdog
            (not recommended — a candidate can allocate arbitrarily).
        init_env: extra env vars for workers (never removes CUDA access).
    """

    def __init__(self, target_path: str, n_workers: int = 4,
                 timeout_s: float = 120.0, max_jobs: int = 40,
                 mem_cap_mb: int = 2048,
                 init_env: dict | None = None):
        self.target_path = os.path.abspath(str(target_path))
        with open(self.target_path, "rb") as f:
            self._target_bytes = f.read()
        self._target_sha = hashlib.sha256(self._target_bytes).digest()
        self._next_integrity_check = 0.0
        self._poisoned: set[int] = set()     # jids: discard late results
        self.timeout_s = timeout_s
        self.max_jobs = max_jobs
        self.mem_cap_bytes = int(mem_cap_mb) * 1024 * 1024
        self.init_env = dict(init_env or {})
        self._ctx = mp.get_context("spawn")
        self._result_q = self._ctx.Queue()
        self._workers: list[_Worker] = []
        self._rss: dict[int, object] = {}      # wid -> psutil.Process
        self._next_mem_poll = 0.0
        self._next_jid = 0
        self._pending: dict[int, tuple] = {}   # jid -> (worker, tag, code)
        self._results: dict[int, dict] = {}    # jid -> result
        self._closed = False
        # Boot-failure breaker: workers that die before ANY job result
        # comes back (target import crash, poison first job) used to
        # respawn forever — every proposal burned a full process spawn.
        # _boot_deaths counts consecutive no-completion deaths; any
        # completed job resets it. Tripped -> submit() fails fast with a
        # staged "spawn" result instead of respawning.
        self._boot_deaths = 0
        self._boot_trip = max(4, n_workers)
        self._broken: str | None = None
        for i in range(max(1, n_workers)):
            self._workers.append(self._spawn(i))

    # ── lifecycle ────────────────────────────────────────────────────
    def _spawn(self, wid: int) -> _Worker:
        job_q = self._ctx.Queue(maxsize=1)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(job_q, self._result_q, self.target_path,
                  self.max_jobs, wid, self.init_env),
            daemon=True)
        proc.start()
        # Job object (Windows): tree-kill on terminate (orphaned children
        # can't survive the worker) + hard job-memory cap backing the RSS
        # watchdog. Children auto-join, so torch.compile subprocesses keep
        # working. Best-effort: any failure just skips the extra layer.
        try:
            from research.persistent_executor import assign_to_job
            handle = getattr(getattr(proc, '_popen', None), '_handle', None)
            if handle is not None:
                assign_to_job(handle, mem_bytes=self.mem_cap_bytes,
                              allow_children=True)
        except Exception:
            pass
        return _Worker(proc, job_q, wid)

    def _respawn(self, w: _Worker) -> _Worker:
        """Terminate a worker and spawn its replacement. Returns the NEW
        _Worker — callers must enqueue onto it, not the dead object."""
        try:
            idx = self._workers.index(w)
        except ValueError:
            return w                    # already replaced
        try:
            if w.proc.is_alive():
                w.proc.terminate()
            w.proc.join(timeout=2)
        except Exception:
            pass
        try:
            while True:
                w.job_q.get_nowait()
        except Exception:
            pass
        self._rss.pop(w.wid, None)      # stale psutil handle
        self._workers[idx] = self._spawn(w.wid)
        return self._workers[idx]

    def size(self) -> int:
        return len(self._workers)

    def idle(self) -> int:
        return sum(1 for w in self._workers
                   if not w.outstanding and w.proc.is_alive())

    def in_flight(self) -> list[dict]:
        """Snapshot of jobs currently ON a worker — the 'what is it
        doing RIGHT NOW' view: jid, candidate tag, worker id, elapsed
        and remaining-vs-deadline. Live-run inspection writes this into
        the status JSON; a long-elapsed row IS the hung candidate."""
        now = time.time()
        out = []
        for jid, (w, tag, _code, _cheap, _b) in self._pending.items():
            out.append({
                "jid": jid, "tag": tag, "worker": w.wid,
                "elapsed_s": round(now - w.job_started, 1)
                if w.job_started else None,
                "over_s": round(now - w.deadline, 1)})
        return out

    # ── async submit/poll ────────────────────────────────────────────
    def submit(self, code: str, tag: str = "x",
               timeout_s: float | None = None,
               cheap: dict | None = None):
        """Submit a candidate. Returns jid, or None if every worker is
        busy (caller should poll first). When the boot breaker is open
        the jid is still returned — its (staged) spawn-failure result
        comes back on the next poll, so callers stay uniform."""
        if self._closed:
            return None
        jid = self._next_jid
        if self._broken is not None:
            self._next_jid += 1
            self._results[jid] = {"ok": False, "stage": "spawn",
                                  "detail": self._broken}
            return jid
        w = next((w for w in self._workers
                  if not w.outstanding), None)
        if w is None:
            return None
        if not w.proc.is_alive():
            w = self._respawn(w)       # NEW worker — the old job_q is dead
        self._next_jid += 1
        try:
            w.job_q.put_nowait((jid, code, cheap))
        except _queue.Full:
            w = self._respawn(w)
            try:
                w.job_q.put_nowait((jid, code, cheap))
            except _queue.Full:
                return None
        w.outstanding = True
        w.job_id = jid
        w.job_started = time.time()
        budget = timeout_s or self.timeout_s
        w.deadline = w.job_started + budget
        self._pending[jid] = (w, tag, code, cheap, budget)
        return jid

    def _harvest(self) -> None:
        """Drain the shared result queue into self._results."""
        while True:
            try:
                jid, res = self._result_q.get_nowait()
            except _queue.Empty:
                return
            except (EOFError, OSError):
                return
            slot = self._pending.pop(jid, None)
            if slot is not None:
                w = slot[0]
                w.outstanding = False
                w.job_id = None
                w.completed += 1
                self._boot_deaths = 0   # the boot chain is alive
            if jid in self._poisoned:
                # Late arrival for a job the integrity check already
                # flagged — the in-flight result is untrusted.
                self._poisoned.discard(jid)
                continue
            self._results[jid] = res

    def _mem_watchdog(self, now: float) -> None:
        """RSS ceiling for busy workers — throttled to ~2 Hz."""
        if not self.mem_cap_bytes or psutil is None \
                or now < self._next_mem_poll:
            return
        self._next_mem_poll = now + 0.5
        for jid, slot in list(self._pending.items()):
            w = slot[0]
            try:
                p = self._rss.get(w.wid)
                if p is None or p.pid != w.proc.pid:
                    p = psutil.Process(w.proc.pid)
                    self._rss[w.wid] = p
                rss = p.memory_info().rss
            except Exception:
                continue                # dead/already-replaced worker
            if rss <= self.mem_cap_bytes:
                continue
            self._respawn(w)
            self._pending.pop(jid, None)
            self._results[jid] = {
                "ok": False, "stage": "mem",
                "detail": f"worker RSS {rss >> 20}MB > cap "
                          f"{self.mem_cap_bytes >> 20}MB"}

    def _integrity_check(self, now: float) -> None:
        """Target-spec on-disk hash, ~1 Hz from the poll loop. A
        candidate (or anything else) rewriting the spec mid-run rewrites
        the gate future workers load — restore the init-time bytes and
        flag every in-flight job poison; its worker can't be trusted."""
        if now < self._next_integrity_check:
            return
        self._next_integrity_check = now + 1.0
        try:
            with open(self.target_path, "rb") as f:
                digest = hashlib.sha256(f.read()).digest()
        except OSError:
            digest = None
        if digest == self._target_sha:
            return
        try:
            with open(self.target_path, "wb") as f:
                f.write(self._target_bytes)
        except OSError:
            return                    # restore failed — retry next tick
        if not self._pending:
            print(f"[eval_pool] target file {self.target_path} "
                  f"modified outside an eval — restored", flush=True)
            return
        for jid, slot in list(self._pending.items()):
            w = slot[0]
            print(f"[eval_pool] target file {self.target_path} "
                  f"modified mid-eval — restored; flagging jid={jid} "
                  f"(worker {w.wid})", flush=True)
            self._pending.pop(jid, None)
            self._poisoned.add(jid)
            self._results[jid] = {
                "ok": False, "stage": "poison",
                "detail": "target file modified mid-eval — restored"}
            self._respawn(w)

    def _enforce_deadlines(self) -> None:
        now = time.time()
        for jid, slot in list(self._pending.items()):
            w = slot[0]
            if now <= w.deadline:
                continue
            # Timed out: kill the worker (its job may be hung), respawn,
            # report a timeout result for this jid only.
            print(f"[eval_pool] worker {w.wid} timed out on jid={jid} "
                  f"tag={slot[1]} — killed+respawning", flush=True)
            self._respawn(w)
            self._pending.pop(jid, None)
            self._results[jid] = {"ok": False, "stage": "timeout"}
        self._mem_watchdog(now)
        self._integrity_check(now)
        # Dead workers with an outstanding job. Two cases:
        #   exitcode == 0 (clean exit — max_jobs recycle raced our
        #     submit): the job sat unread in the dead worker's queue —
        #     never started. Requeue it onto the respawned worker
        #     instead of failing it.
        #   nonzero (crash/OOM-kill mid-job): the job may be the poison —
        #     fail it, and count a boot death if the worker never
        #     completed anything (boot-path breaker).
        for jid, (w, tag, code, cheap, budget) \
                in list(self._pending.items()):
            if w.proc.is_alive():
                continue
            # Spawned procs flush result_q before their final exit —
            # a worker that evaluated the job THEN hit max_jobs exits
            # clean with the result already queued. Harvest once more
            # and take it; requeueing would double-eval AND a late
            # arrival would corrupt the replacement worker's slot.
            self._harvest()
            if jid in self._results:
                continue
            self._pending.pop(jid, None)
            ex = w.proc.exitcode
            clean = (ex == 0)
            old_completed = w.completed    # before respawn rebinds w
            if not clean:
                # Crash attribution: which worker died on which job —
                # the jid maps to the candidate logged by the caller.
                print(f"[eval_pool] worker {w.wid} died mid-job "
                      f"(jid={jid} tag={tag} exit={ex}) — respawning",
                      flush=True)
            w = self._respawn(w)
            if clean:
                try:
                    w.job_q.put_nowait((jid, code, cheap))
                    w.outstanding = True
                    w.job_id = jid
                    w.job_started = now
                    w.deadline = now + budget
                    self._pending[jid] = (w, tag, code, cheap, budget)
                    continue
                except _queue.Full:
                    pass
            self._results[jid] = {
                "ok": False, "stage": "spawn",
                "detail": f"worker exited mid-job (code {ex})"}
            if not clean or old_completed == 0:
                # Died before ever returning a job — boot path
                # (target import crash, spawn env failure). Enough
                # of these means respawning is just a burn loop.
                self._boot_deaths += 1
                if self._broken is None \
                        and self._boot_deaths >= self._boot_trip:
                    self._broken = (
                        f"eval workers failing to boot "
                        f"({self._boot_deaths} consecutive "
                        f"no-completion deaths — target "
                        f"{self.target_path})")
                    print(f"[eval_pool] circuit open: {self._broken}")

    def poll(self, timeout: float = 0.1):
        """Wait up to `timeout` for one finished job.
        Returns (jid, result_dict) or None."""
        deadline = time.time() + max(timeout, 0)
        while True:
            self._harvest()
            if self._results:
                jid = next(iter(self._results))
                return jid, self._results.pop(jid)
            self._enforce_deadlines()
            if self._results:
                jid = next(iter(self._results))
                return jid, self._results.pop(jid)
            if time.time() >= deadline:
                return None
            time.sleep(0.005)

    # ── sync helpers (verify path) ───────────────────────────────────
    def eval_sync(self, code: str, timeout_s: float | None = None) -> dict:
        """Blocking single eval — used by seed/resume/verify paths."""
        budget = timeout_s or self.timeout_s
        jid = None
        while jid is None:
            jid = self.submit(code, timeout_s=budget)
            if jid is None:
                self.poll(timeout=0.05)
        deadline = time.time() + budget + 5
        while time.time() < deadline:
            got = self.poll(timeout=0.05)
            if got is not None:
                rid, res = got
                # A different job may finish first — stash its result.
                if rid == jid:
                    return res
                self._results[rid] = res
        return {"ok": False, "stage": "timeout"}

    def eval_many(self, codes: list[str],
                  timeout_s: float | None = None) -> list[dict]:
        """Submit a batch and gather results in order — parallel verify
        reps instead of serial subprocess spawns."""
        if not codes:
            return []
        budget = timeout_s or self.timeout_s
        jids: list[int | None] = [None] * len(codes)
        out: list[dict | None] = [None] * len(codes)
        deadline = time.time() + budget + 5
        i = 0
        while True:
            while i < len(codes):
                jid = self.submit(codes[i], tag=f"m{i}", timeout_s=budget)
                if jid is None:
                    break
                jids[i] = jid
                i += 1
            got = self.poll(timeout=0.05)
            if got is not None:
                jid, res = got
                if jid in jids:
                    out[jids.index(jid)] = res
                else:
                    self._results[jid] = res
            if all(o is not None for o in out):
                return [o for o in out]
            if time.time() > deadline:
                return [o if o is not None else
                        {"ok": False, "stage": "timeout"} for o in out]

    def close(self) -> None:
        self._closed = True
        for w in self._workers:
            try:
                w.job_q.put_nowait(None)
            except Exception:
                pass
        for w in self._workers:
            try:
                w.proc.join(timeout=3)
                if w.proc.is_alive():
                    w.proc.terminate()
                    w.proc.join(timeout=2)
            except Exception:
                pass
            # Close the per-worker queue — on Windows each live Queue
            # holds a sem handle + feeder thread; hundreds of sequential
            # CodeEvolve runs in one campaign process used to leak them.
            try:
                w.job_q.close()
            except Exception:
                pass
        try:
            self._result_q.close()
        except Exception:
            pass

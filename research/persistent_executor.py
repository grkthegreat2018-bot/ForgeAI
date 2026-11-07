"""Persistent code executor — avoids 200ms Python startup per execution on Windows.

Instead of spawning a new subprocess per code execution (200ms startup on Windows),
keeps a persistent Python worker process that receives code via a file-based
protocol and returns results the same way:

  → SEND: <proto_dir>/req.bin  (utf-8 code bytes, atomic tmp+replace)
  ← RECV: <proto_dir>/res.bin  (JSON result, atomic tmp+replace)

Security model (defense in depth — see docs for residual-risk notes):
  L0  Separate worker process, cwd=jail dir, stdio=DEVNULL, env scrubbed of
      secret-looking vars; Windows Job Object (KILL_ON_JOB_CLOSE tree-kill,
      JOB_MEMORY cap, ACTIVE_PROCESS=1 child-spawn block) where available.
  L1  sys.addaudithook in the worker: un-removable, fires at C level for ALL
      code — denies dangerous imports, confines open()/fs ops to the jail dir
      (+ secret-path blacklist), denies process/network/trace/frame/gc/
      code-object/ctypes ops for sandboxed frames only (worker internals
      remain unrestricted).
  L2  Restricted builtins (no open/exec/eval/compile/input/__import__→guarded).
  L3  sys.modules purge of denied modules (blocks the sys.modules['os'] path).
  L4  builtins dict restored after every exec (kills persistent worker
      hijacking via builtins patching); sys.path + recursion limit reset.
  L5  File-based IPC: no pipe fds exist during exec, so the un-audited
      os.read/os.write fd surface cannot snoop protocol or forge results.

Residual risks (documented, accepted): object-graph tricks (__subclasses__,
function.__globals__) can reach modules but every dangerous op is audited;
os._exit/os.abort kill the worker (DoS only → auto-respawn); un-audited
file-existence probes (os.stat); pure-CPU work until the timeout kill.
This is a hard guardrail, not a VM: treat as hostile-code containment with
the Job Object as the OS-level backstop, not a cryptographic boundary.
"""
import contextlib
import os
import re
import subprocess
import sys
import tempfile
import time

from research.json_compat import loads

# ── env scrubbing (shared parent/worker logic — worker gets its own copy) ──

_SECRET_ENV_RE = re.compile(
    r'(token|secret|passw|credential|private|api[_-]?key|auth|bearer|(^|_)key($|_))',
    re.IGNORECASE)

# Exact-match allowlist for names that trip the secret regex but are NOT
# secrets. MUST be exact-match only — the old prefix allowlist let
# FORGE_API_KEY, NVIDIA_API_KEY, NODE_AUTH_TOKEN, CARGO_REGISTRY_TOKEN and
# OS_PASSWORD through (FORGE_/NV/NODE/CARGO/OS prefixes beat the deny).
_SECRET_ENV_SAFE = frozenset({'TOKENIZERS_PARALLELISM'})


def scrub_environ(env: dict | None = None) -> dict:
    """Return a copy of `env` (default os.environ) with secret-looking vars
    removed. Deny-wins: a secret-pattern match survives ONLY via the exact
    _SECRET_ENV_SAFE allowlist. Applied parent-side before spawn AND inside
    the worker."""
    src = dict(os.environ if env is None else env)
    for k in list(src):
        ku = k.upper()
        if _SECRET_ENV_RE.search(ku) and ku not in _SECRET_ENV_SAFE:
            del src[k]
    return src


# ── Job objects (Windows): real tree-kill + memory cap + child-spawn block ──

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    class _LARGE_INTEGER(ctypes.Structure):
        _fields_ = [('QuadPart', ctypes.c_longlong)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', _LARGE_INTEGER),
            ('PerJobUserTimeLimit', _LARGE_INTEGER),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ('IoInfo', _IO_COUNTERS),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    _KILL_ON_JOB_CLOSE = 0x2000
    _JOB_MEMORY = 0x0200
    _ACTIVE_PROCESS = 0x0008
    _EXTENDED_LIMIT_INFO = 9


def assign_to_job(proc_handle: int, mem_bytes: int = 0,
                  allow_children: bool = True):
    """Assign a child process to a Job Object (Windows only).

    Returns an opaque job HANDLE to KEEP referenced for the child's lifetime —
    closing it (or parent death with KILL_ON_JOB_CLOSE) kills the whole tree.
    Returns None on any failure (non-Windows, already-jobbed process, API
    missing) — callers must treat job assignment as best-effort hardening.

    mem_bytes>0 sets a hard JobMemoryLimit. allow_children=False sets
    ACTIVE_PROCESS=1 so the child cannot spawn further processes at all.
    """
    if sys.platform != 'win32':
        return None
    try:
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = _KILL_ON_JOB_CLOSE
        if mem_bytes:
            flags |= _JOB_MEMORY
            info.JobMemoryLimit = int(mem_bytes)
        if not allow_children:
            flags |= _ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = 1
        info.BasicLimitInformation.LimitFlags = flags
        ok = k32.SetInformationJobObject(
            job, _EXTENDED_LIMIT_INFO, ctypes.byref(info),
            ctypes.sizeof(info))
        if not ok:
            k32.CloseHandle(job)
            return None
        if not k32.AssignProcessToJobObject(job, int(proc_handle)):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def close_job(job) -> None:
    if job and sys.platform == 'win32':
        with contextlib.suppress(Exception):
            ctypes.WinDLL('kernel32').CloseHandle(job)


# ── worker-side source (shared by persistent worker + one-shot mode) ────────
# This is a string: the worker runs it in a fresh interpreter. Everything
# sandbox-critical lives here so there is ONE implementation.

_BOOTSTRAP = r'''
import sys as _sys, os as _os, json as _json, io as _io, struct as _st
import traceback as _tb, time as _time, builtins as _bi, fnmatch as _fm
import threading as _threading, re as _re, ctypes as _ct
# Pre-import pure-stdlib utility modules: sandboxed code may `import` them
# (a sys.modules hit needs no file read, so no jail escape); importing at
# BOOT is a worker-internal op and therefore allowed. Without this, a
# sandboxed `import string` tries to read the stdlib .py from outside the
# jail and dies (selfgen RFT harness hit exactly that).
import math, itertools, collections, functools, string, operator

_SCRATCH = _sys.argv[1] if len(_sys.argv) > 1 else _os.getcwd()
_PROTO = _sys.argv[2] if len(_sys.argv) > 2 else _SCRATCH
_TAG = '<sandbox>'

# ── 1. env scrub: remove secret-looking vars before any user code runs ──
_scrub = _re.compile(
    r'(token|secret|passw|credential|private|api[_-]?key|auth|bearer|(^|_)key($|_))',
    _re.I)
_keep = ('PYTHON', 'CUDA', 'NV', 'TRITON', 'TORCH', 'OMP', 'MKL', 'OPENBLAS',
         'NUMEXPR', 'PATH', 'PATHEXT', 'SYSTEMROOT', 'SYSTEMDRIVE', 'TEMP',
         'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'HOMEDRIVE',
         'HOMEPATH', 'COMSPEC', 'OS', 'PROCESSOR', 'WINDIR', 'HF_HOME',
         'HF_HUB', 'HF_ENDPOINT', 'TRANSFORMERS', 'VIRTUAL_ENV', 'CONDA',
         'NUMBER_OF_PROCESSORS', 'USERNAME', 'USERDOMAIN', 'DRIVERDATA',
         'PROGRAMDATA', 'PROGRAMFILES', 'PUBLIC', 'SESSIONNAME',
         'LOGONSERVER', 'ALLUSERSPROFILE', 'COMMONPROGRAM', 'PSMODULEPATH',
         'TOKENIZERS_PARALLELISM', 'ONEAPI', 'VS', 'VC', 'INCLUDE', 'INTEL',
         'DOTNET', 'GOPATH', 'CARGO', 'RUST', 'NODE', 'JAVA', 'MSYSTEM',
         'MINGW', 'CHOCOLATEY', 'VCPKG', 'WSL', 'XDG', 'HOME', 'SHELL',
         'LANG', 'LC_', 'TZ', 'EMSDK', 'GOOGLE_APPLICATION_CREDENTIALS_JSON',
         'FORGE_')
for _k in list(_os.environ):
    _ku = _k.upper()
    if _scrub.search(_ku) and not _ku.startswith(_keep):
        del _os.environ[_k]

# ── 2. denylist + fs policy ────────────────────────────────────────────────
# Modules with un-audited dangerous surface (native file/network/process ops
# that bypass the audit layer) plus classic escape tools. Op-level audits are
# the real teeth — this is noise reduction + covers un-audited modules.
_DENY_MODS = frozenset("""subprocess socket _socket ssl _ssl http urllib
requests httpx aiohttp websocket websockets ftplib smtplib telnetlib imaplib
poplib nntplib socketserver xmlrpc paramiko asyncio select selectors ctypes
_ctypes _winapi msvcrt nt winreg winsound ntsecuritycon cffi _cffi_backend
mmap sqlite3 _sqlite3 dbm _dbm gdbm shelve pickle _pickle cpickle marshal
dill joblib cloudpickle gc inspect pdb bdb ipdb dis code codeop venv pip
ensurepip setuptools pkg_resources getpass pty tty termios fcntl posix pwd
grp spwd crypt resource syslog multiprocessing _multiprocessing concurrent
threading signal runpy importlib builtins site antigravity this py_compile
compileall zipimport pkgutil modulefinder yaml webbrowser platform netrc
_posixsubprocess _scproxy urllib3 aiofiles anyio trio curio gevent eventlet
twisted flask django fastapi uvicorn sqlalchemy pexpect invoke fabric
pyautogui keyboard mouse pynput win32api win32con win32file win32security
win32process win32event win32gui pywintypes pythoncom servicemanager
numpy torch pandas tensorflow jax scipy sklearn transformers safetensors
triton numba cupy ray dask zmq pyzmq grpc
""".split())
# (os, sys, io, pathlib, linecache are intentionally ALLOWED: their dangerous
# ops are individually audited, and stdlib internals import them lazily.)

# Paths always denied inside the jail policy — secrets/credentials/policy.
_DENY_GLOBS = tuple(p.replace('\\', '/').lower() for p in (
    '*/.env', '*/.env.*', '*/*.env', '*/sandbox.json', '*/.git/*', '*/.git',
    '*/.ssh', '*/.ssh/*', '*/*.pem', '*/*.key', '*/*.p12', '*/*.pfx',
    '*/id_rsa*', '*/id_ed25519*', '*/credentials*', '*/secrets*',
    '*/__pycache__/*', '*.pyc',
))

_fs_root = _os.path.normcase(_os.path.realpath(_SCRATCH))
_fs_root_sep = _fs_root + _os.sep

def _under_root(path):
    """True iff `path` resolves inside the jail dir (handles ../ tricks,
    symlinks/junctions, case-insensitivity)."""
    try:
        rp = _os.path.normcase(_os.path.realpath(_os.fspath(path)))
    except Exception:
        return False
    return rp == _fs_root or rp.startswith(_fs_root_sep)

def _denied_path(rp):
    fwd = rp.replace('\\', '/').lower()
    return any(_fm.fnmatch(fwd, g) for g in _DENY_GLOBS)

def _path_ok(path):
    try:
        rp = _os.path.normcase(_os.path.realpath(_os.fspath(path)))
    except Exception:
        return False
    if _denied_path(rp):
        return False
    return rp == _fs_root or rp.startswith(_fs_root_sep)

# Events carrying filesystem paths — allowed only under the jail dir.
_FS_EVENTS = frozenset("""open os.listdir os.scandir os.stat os.lstat os.remove
os.unlink os.rmdir os.mkdir os.rename os.replace os.link os.symlink os.chmod
os.chown os.lchown os.truncate os.utime os.access os.readlink os.mknod
os.mkfifo os.chflags os.lchflags""".split())

# Events denied outright for sandboxed code.
_DENY_EVENTS = frozenset("""os.system os.exec os.execv os.execve os.execvp
os.execl os.execlp os.execvpe os.execlpe os.spawnl os.spawnle os.spawnlp
os.spawnlpe os.spawnv os.spawnve os.spawnvp os.spawnvpe os.fork os.forkpty
os.kill os.killpg os.plock os.startfile os.startfile/2 os.posix_spawn
os.posix_spawnp os.register_at_fork subprocess.Popen _winapi.CreateProcess
_posixsubprocess.fork_exec pty.fork os.openpty os.login_tty os.putenv
os.unsetenv os.chdir os.fchdir os.chroot os.setuid os.seteuid os.setgid
os.setegid os.setreuid os.setregid os.setgroups os.initgroups os.setlogin
os.setpgrp os.setsid os.setpgid os.tcsetpgrp os.setpriority os.nice
os.set_blocking sys.settrace sys.setprofile sys.set_asyncgen_hooks
sys._getframe sys._current_frames sys._current_exceptions gc.get_objects
gc.get_referrers gc.get_referents gc.freeze marshal.loads marshal.dumps
code.__new__ function.__new__ builtins.input builtins.breakpoint compile
exec eval object.__setattr__ object.__delattr__ ctypes.dlopen ctypes.CDLL
ctypes.PyDLL ctypes.WinDLL ctypes.OleDLL ctypes.dlsym ctypes.string_at
ctypes.wstring_at ctypes.addressof ctypes.cdata ctypes.call_function
os.eventfd os.signalfd os.timerfd_create os.inotify_init os.epoll_create
os.kqueue os.pipe2 os.sched_setaffinity os.sched_setparam
os.sched_setscheduler os.setxattr os.getxattr os.listxattr os.removexattr
os.memfd_create os.mount os.unmount pdb.post_mortem
""".split())

_DENY_PREFIX = ('socket.', '_socket.', 'ssl.', '_ssl.', 'ctypes.', 'winreg.',
                'mmap.', 'http.', 'urllib.', 'requests.', 'subprocess.',
                'pty.', 'select.', '_posixsubprocess.', '_winapi.', 'msvcrt.',
                'multiprocessing.', '_multiprocessing.', 'concurrent.',
                'asyncio.', 'signal.', 'nt.', 'webbrowser.', 'smtplib.',
                'ftplib.', 'telnetlib.', 'imaplib.', 'poplib.', 'nntplib.',
                'xmlrpc.', 'socketserver.', 'ssl.', 'pickle.', 'shelve.',
                'dbm.', 'sqlite3.', 'gc.', 'inspect.', 'sys.settrace',
                'importlib.', 'runpy.', 'venv.', 'pip.', 'ensurepip.',
                'win32', 'pywintypes.', 'pythoncom.', 'cffi.')

_local = _threading.local()

def _in_sandbox():
    """True iff any frame on the current stack belongs to sandboxed code
    (co_filename == '<sandbox>'). Worker internals run unrestricted; only
    code the worker execs carries the tag."""
    f = _sys._getframe()
    while f is not None:
        if f.f_code.co_filename == _TAG:
            return True
        f = f.f_back
    return False

def _audit(event, args):
    if getattr(_local, 'busy', False):
        return                      # guard against hook-triggered events
    _local.busy = True
    try:
        if not _in_sandbox():
            return                  # worker-internal op — always allowed
        if event == 'import':
            name = (args[0] or '').split('.')[0].lower()
            if name in _DENY_MODS:
                raise ImportError(f"sandbox: module '{args[0]}' is blocked")
            return
        if event in _FS_EVENTS:
            paths = (args[0],) if event == 'open' else args
            for a in paths:
                if isinstance(a, (str, bytes, _os.PathLike)):
                    if not _path_ok(a):
                        raise PermissionError(
                            f"sandbox: '{event}' outside jail: {a!r}")
            return
        if event in _DENY_EVENTS or event.startswith(_DENY_PREFIX):
            raise PermissionError(f"sandbox: '{event}' is blocked")
    finally:
        _local.busy = False

_sys.addaudithook(_audit)

# ── 3. purge denylisted modules already in sys.modules (blocks the
#    sys.modules['subprocess'] direct-dict path — the audit hook still
#    governs op-level behaviour for everything reachable via object graph) ──
for _m in list(_sys.modules):
    if _m.split('.')[0].lower() in _DENY_MODS:
        del _sys.modules[_m]

# ── 4. restricted builtins for sandboxed code ──────────────────────────────
_real_import = _bi.__import__
# 'open' stays — the 'open' audit event confines every call (open, io.open,
# pathlib, tempfile all funnel through it) to the jail dir. exec/eval/
# compile are denied both at builtin and audit level.
_DROP = frozenset(('exec', 'eval', 'compile', 'input', 'breakpoint',
                   'exit', 'quit', 'help', 'copyright', 'credits', 'license'))
_SAFE_BUILTINS = {k: v for k, v in vars(_bi).items() if k not in _DROP}
_SAFE_BUILTINS['__import__'] = _real_import   # audit hook enforces allowability
# Full snapshots of the REAL interpreter state — restored after every exec so
# sandboxed code cannot persistently poison the worker (builtins patching,
# sys.modules injection, sys.path edits, recursion-limit changes).
_BUILTIN_SNAPSHOT = dict(vars(_bi))
_MOD_SNAPSHOT = dict(_sys.modules)

import contextlib as _cl
contextlib_suppress = _cl.suppress
_BASE_PATH = list(_sys.path)
_BASE_RL = 4000

def _restore_state():
    cur = vars(_bi)
    for k in list(cur):
        if k not in _BUILTIN_SNAPSHOT:
            with contextlib_suppress(KeyError):
                del cur[k]
    cur.update(_BUILTIN_SNAPSHOT)
    mods = _sys.modules
    for m in list(mods):
        if m not in _MOD_SNAPSHOT:
            with contextlib_suppress(KeyError):
                del mods[m]
    for m, v in _MOD_SNAPSHOT.items():
        mods[m] = v
    _sys.path[:] = _BASE_PATH
    _sys.setrecursionlimit(_BASE_RL)

_sys.setrecursionlimit(_BASE_RL)
_base_threads = _threading.enumerate()

# ── 5. exec runner ─────────────────────────────────────────────────────────
def _run_code(code):
    """Execute `code` in a fresh restricted namespace. Returns telemetry dict."""
    out = {'stdout': '', 'stderr': '', 'returncode': 0, 'exec_time_ms': 0.0}
    # Fresh dict copy each exec — sandboxed code can do
    # __builtins__['x']=y which would otherwise persist into the NEXT
    # execution's namespace (cross-exec smuggling channel).
    ns = {'__name__': '__sandbox__', '__builtins__': dict(_SAFE_BUILTINS)}
    t0 = _time.time()
    old_out, old_err = _sys.stdout, _sys.stderr
    _sys.stdout = _io.StringIO()
    _sys.stderr = _io.StringIO()
    try:
        exec(compile(code, _TAG, 'exec'), ns)
        out['stdout'] = _sys.stdout.getvalue()
        out['stderr'] = _sys.stderr.getvalue()
    except SystemExit as e:
        out['stdout'] = _sys.stdout.getvalue()
        out['stderr'] = _sys.stderr.getvalue()
        out['returncode'] = e.code if isinstance(e.code, int) else 1
    except BaseException:
        out['stdout'] = _sys.stdout.getvalue()
        out['stderr'] = _tb.format_exc()
        out['returncode'] = 1
    finally:
        _sys.stdout = old_out
        _sys.stderr = old_err
        out['exec_time_ms'] = (_time.time() - t0) * 1000
        _restore_state()
    return out

def _fresh_threads():
    """True iff sandboxed code spawned threads (they outlive exec — the
    worker must respawn to guarantee a clean slate)."""
    return len(_threading.enumerate()) > len(_base_threads)

def _atomic_write(path, data):
    tmp = path + '.tmp'
    with _io.open(tmp, 'wb') as f:
        f.write(data)
    _os.replace(tmp, path)
'''

_WORKER_LOOP = r'''
# ── worker main loop: file protocol over _PROTO dir ────────────────────────
# req.bin: utf-8 code; res.bin: JSON result. Both atomic tmp+replace writes.
# No pipe/fd is ever open during exec → un-audited os.read/os.write cannot
# snoop the protocol or forge results.
_req = _os.path.join(_PROTO, 'req.bin')
_res = _os.path.join(_PROTO, 'res.bin')
_exit = _os.path.join(_PROTO, 'exit')
_ready = _os.path.join(_PROTO, 'ready')
_atomic_write(_ready, b'1')
_parent_pid = _os.getppid()
_probe_n = 0

while True:
    if _os.path.exists(_exit):
        break
    # Parent-liveness probe every ~1s (job object normally handles orphans;
    # this is the fallback when assignment failed / non-Windows).
    _probe_n += 1
    if _probe_n >= 500:
        _probe_n = 0
        try:
            if _sys.platform == 'win32':
                _k32 = _ct.windll.kernel32
                _h = _k32.OpenProcess(0x100000, False, _parent_pid)
                if _h:
                    _k32.CloseHandle(_h)
                else:
                    break       # parent gone — self-terminate
        except Exception:
            pass
    if not _os.path.exists(_req):
        _time.sleep(0.002)
        continue
    try:
        with _io.open(_req, 'rb') as f:
            _raw = f.read()
        _os.unlink(_req)            # close the fd BEFORE exec
        _code = _raw.decode('utf-8', errors='replace')
    except OSError:
        continue
    _result = _run_code(_code)
    if _fresh_threads():
        _result['stderr'] += '\n[sandbox: worker reset — spawned threads]'
        try:
            _atomic_write(_res, _json.dumps(_result).encode('utf-8'))
        finally:
            _os._exit(2)            # parent respawns a clean worker
    try:
        _atomic_write(_res, _json.dumps(_result).encode('utf-8'))
    except OSError:
        break
_sys.exit(0)
'''

_ONCE_TAIL = r'''
# ── one-shot mode: argv = script <scratch> <proto?> --once <code> <result> ──
_args = _sys.argv[1:]
_once_i = _args.index('--once')
_code_path, _result_path = _args[_once_i + 1], _args[_once_i + 2]
with _io.open(_code_path, 'rb') as f:
    _code = f.read().decode('utf-8', errors='replace')
_result = _run_code(_code)
with _io.open(_result_path, 'w', encoding='utf-8') as f:
    _json.dump(_result, f)
_sys.exit(0)
'''

_WORKER_SCRIPT = _BOOTSTRAP + _WORKER_LOOP
_ONCE_SCRIPT = _BOOTSTRAP + _ONCE_TAIL


def build_once_script() -> str:
    """Worker-side script for one-shot sandboxed execution.

    Invoke as: ``python once.py <scratch_dir> <unused> --once <code.py> <result.json>``
    — writes a JSON telemetry dict ({stdout,stderr,returncode,exec_time_ms}).
    """
    return _ONCE_SCRIPT


class PersistentCodeExecutor:
    """Persistent Python worker for sandboxed code execution.

    Usage:
        executor = PersistentCodeExecutor(timeout_s=5)
        result = executor.execute("print(1+1)", expected_output="2")
        # ... reuse for many executions ...
        executor.close()

    The worker is spawned once and reused (file-based IPC). If it dies or
    times out it is automatically respawned on the next execute() call.

    fs_root: jail dir for the sandboxed filesystem (default: a fresh mkdtemp
    scratch dir). Pass a wider root (e.g. a workspace dir) for snippets that
    legitimately need file access — secret-path blacklist still applies.
    """

    def __init__(self, timeout_s: float = 5.0, memory_limit_mb: int = 512,
                 fs_root: str | None = None):
        self.timeout_s = timeout_s
        self.memory_limit_mb = memory_limit_mb
        self._owns_root = fs_root is None
        if fs_root:
            os.makedirs(fs_root, exist_ok=True)
            self._fs_root = os.path.realpath(fs_root)
        else:
            self._fs_root = tempfile.mkdtemp(prefix='pce_jail_')
        self._proto = tempfile.mkdtemp(prefix='pce_proto_')
        self._proc: subprocess.Popen | None = None
        self._worker_path = os.path.join(self._proto, 'worker.py')
        self._job = None
        import threading
        self._lock = threading.Lock()

    # ── worker lifecycle ────────────────────────────────────────────

    def _clean_proto(self) -> None:
        """Remove stale protocol files (from a dead worker)."""
        for name in ('req.bin', 'res.bin', 'ready', 'exit', 'boot_err.txt'):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self._proto, name))

    def _ensure_worker(self) -> None:
        """Spawn worker if not running; wait for its ready marker."""
        if self._proc is not None and self._proc.poll() is None:
            return

        self._clean_proto()
        with open(self._worker_path, 'w', encoding='utf-8') as f:
            f.write(_WORKER_SCRIPT)

        env = scrub_environ()
        # UTF-8 via env vars, NOT `-X utf8` (the flag self-reexecs the
        # interpreter — incompatible with job limits).
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        self._proc = subprocess.Popen(
            [sys.executable, self._worker_path,
             self._fs_root, self._proto],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self._fs_root, env=env)

        # OS-level containment (best-effort): tree-kill on close/timeout,
        # hard job-wide memory cap. Child spawning is allowed at the job
        # level — venv python.exe shims must spawn the real interpreter —
        # but spawned children auto-join the job (still memory-capped and
        # killed with the tree); spawning itself is gated by the audit
        # hook's subprocess/os.* denial.
        handle = getattr(self._proc, '_handle', None)
        if handle is not None:
            self._job = assign_to_job(
                handle, mem_bytes=self.memory_limit_mb * 1024 * 1024,
                allow_children=True)

        # Wait for the ready marker (bootstrap + audit-hook install).
        ready = os.path.join(self._proto, 'ready')
        deadline = time.time() + 20
        while time.time() < deadline:
            if os.path.exists(ready):
                return
            if self._proc.poll() is not None:
                err = os.path.join(self._proto, 'boot_err.txt')
                detail = ''
                if os.path.exists(err):
                    with contextlib.suppress(OSError):
                        detail = open(err, encoding='utf-8',
                                      errors='replace').read()[-400:]
                raise RuntimeError(
                    f'sandbox worker exited during boot '
                    f'(rc={self._proc.returncode}) {detail}')
            time.sleep(0.02)
        self._kill_worker()
        raise RuntimeError('sandbox worker failed to signal ready')

    def _kill_worker(self) -> None:
        """Kill the current worker (job object takes the whole tree)."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None
        close_job(self._job)
        self._job = None
        self._clean_proto()

    # ── protocol ────────────────────────────────────────────────────

    _REQ = 'req.bin'
    _RES = 'res.bin'

    def _send_code(self, code: str) -> None:
        tmp = os.path.join(self._proto, self._REQ + '.tmp')
        with open(tmp, 'wb') as f:
            f.write(code.encode('utf-8'))
        os.replace(tmp, os.path.join(self._proto, self._REQ))

    def _recv_result(self) -> dict | None:
        path = os.path.join(self._proto, self._RES)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                data = f.read()
            os.unlink(path)
            return loads(data.decode('utf-8'))
        except (OSError, ValueError):
            return None

    # ── public API ──────────────────────────────────────────────────

    def execute(self, code: str, expected_output: str | None = None) -> dict:
        """Execute code in the sandboxed worker.

        Returns dict with stdout, stderr, returncode, exec_time_ms,
        timed_out, output_matches_expected.
        """
        result = {
            "stdout": "", "stderr": "", "returncode": -1,
            "exec_time_ms": 0, "peak_memory_kb": 0,
            "file_size_bytes": len(code.encode('utf-8')), "timed_out": False,
            "output_matches_expected": False,
        }
        # stdin/stdout-free file protocol is still serialized per worker.
        with self._lock:
            return self._execute_locked(code, expected_output, result)

    def _execute_locked(self, code: str, expected_output: str | None,
                        result: dict) -> dict:
        t0 = time.time()
        try:
            self._ensure_worker()
            self._send_code(code)

            deadline = t0 + self.timeout_s
            worker_result = None
            while time.time() < deadline:
                worker_result = self._recv_result()
                if worker_result is not None:
                    break
                if self._proc is not None and self._proc.poll() is not None:
                    break            # worker died (crash/os._exit) → fail
                time.sleep(0.002)
            if worker_result is None:
                result["timed_out"] = True
                result["stderr"] = (
                    f"Execution timed out after {self.timeout_s}s"
                    if self._proc is not None and self._proc.poll() is None
                    else "Sandbox worker exited mid-execution")
                result["exec_time_ms"] = (time.time() - t0) * 1000
                self._kill_worker()
                return result

            result["stdout"] = worker_result.get("stdout", "")
            result["stderr"] = worker_result.get("stderr", "")
            result["returncode"] = worker_result.get("returncode", -1)
            result["exec_time_ms"] = worker_result.get("exec_time_ms", 0)

            if expected_output is not None:
                result["output_matches_expected"] = (
                    result["stdout"].strip() == expected_output.strip())

        except Exception as e:
            result["stderr"] = f"Executor error: {e}"
            result["exec_time_ms"] = (time.time() - t0) * 1000
            self._kill_worker()

        return result

    # ── cleanup ─────────────────────────────────────────────────────

    def close(self):
        """Shut down the worker and remove scratch/proto dirs."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                with open(os.path.join(self._proto, 'exit'), 'w') as f:
                    f.write('1')
                self._proc.wait(timeout=3)
            except Exception:
                pass
        self._kill_worker()
        # Only remove dirs we created — never a caller-supplied fs_root.
        for d in ([self._proto] + ([self._fs_root] if self._owns_root else [])):
            with contextlib.suppress(OSError):
                import shutil
                shutil.rmtree(d, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

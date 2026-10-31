"""Evolution target: the loop's own canonicalizer.

Self-referential speedup target. Mutator.canon / _variant / mutants_for call
`ast.unparse(ast.parse(src))` once per parent AND once per emitted variant
(no-op suppression); the dedup key feeds code_programs.code_hash. unparse is a
pure-Python tree visitor — the parse is C-fast, the emit is the cost.

Contract (test() enforces it): canon_code must return a hashable str/bytes
such that canon_code(a) == canon_code(b) IFF a and b are the same program
modulo formatting — comments, whitespace, redundant parens, literal spelling
(0x10 vs 16, 'x' vs "x"). Equivalence classes must match the reference
exactly on the corpus. Deterministic; may raise on invalid syntax.

Faster legit paths the search may find: compile() + marshal.dumps of the code
object (both C), tokenize-stream keys, ast.dump variants, local-attr binding.
Memoization cannot game bench(): every bench string is unique.
"""

NAME = "canon_key"
GOAL = "min"
METRIC = "ms_per_call"
TAGS = ["self", "dedup", "ast", "hot-path", "evolution"]
BUDGETS = {}
REQUIRED = ["canon_code"]
PROMPT = (
    "Faster canon_code(src) -> str-or-bytes: a canonical dedup key for "
    "Python source — two sources get the SAME key iff they are the same "
    "program modulo formatting (comments, whitespace, parens, literal "
    "spelling like 0x10 vs 16 or 'a' vs \"a\"). Seed is "
    "ast.unparse(ast.parse(src)) ~0.15ms/call; unparse is a pure-Python "
    "visitor and is the bottleneck. Try marshal.dumps(compile(src, ...)), "
    "sha1/md5 over marshal bytes, a tokenize-stream key, ast.dump, or "
    "local-attr binding. Must return a hashable str-or-bytes, be "
    "deterministic, and give identical keys for equivalent programs. "
    "May raise on invalid syntax."
)

SEED_CODE = '''
import ast


def canon_code(src: str) -> str:
    """Canonical dedup key: unparse of the parse tree."""
    return ast.unparse(ast.parse(src))
'''


def _ref():
    import ast

    def canon_ref(src: str):
        return ast.unparse(ast.parse(src))

    return canon_ref


# (a, b, same_program): hand-checked against ast.unparse(ast.parse) semantics.
_PAIRS = [
    # ── same program — formatting/cosmetic only ─────────────────────────
    ("x=1", "x = 1", True),
    ("x=1", "x   =   1\n\n\n", True),
    ("def f():\n    return 1", "def f():\n        return 1", True),
    ("x = (1)", "x = 1", True),
    ("x = 'a'", 'x = "a"', True),
    ("x = 0x10", "x = 16", True),
    ("x = 1e0", "x = 1.0", True),
    ("x = 1; y = 2", "x = 1\ny = 2", True),
    ("x = 1  # trailing", "x = 1", True),
    ("x = 1, 2", "x = (1, 2)", True),
    ("x = - 1", "x = -1", True),
    ("import os\nimport sys", "import os; import sys", True),
    ("def f():\n    # note\n    return 1", "def f():\n    return 1", True),
    ("def f(a,b):\n    return a+b", "def f(a, b):\n    return a + b", True),
    ("x = [i for i in range(3)]", "x = [i for i in range( 3 )]", True),
    # ── different program — must NOT collide ────────────────────────────
    ("x = 1", "x = 2", False),
    ("x = a + b", "x = a - b", False),
    ("x = a + b", "x = b + a", False),
    ("x = 1 + 2", "x = 3", False),
    ('def f():\n    """doc"""\n    return 1', "def f():\n    return 1", False),
    ("x = f(1, 2)", "x = f(2, 1)", False),
    ("x = [1, 2]", "x = (1, 2)", False),
    ("if a:\n    x = 1", "x = 1", False),
    ("def f():\n    return 1", "def g():\n    return 1", False),
    ("x = True", "x = 1", False),
    ("x = 1", "x = 1.0", False),
    ("x = not a", "x = a", False),
    ("def f():\n    pass", "def f():\n    ...", False),
    ("x = [i for i in range(3)]", "x = (i for i in range(3))", False),
    ("x = -1", "x = 1", False),
]


def _corpus():
    """All pair members + a few singleton programs."""
    seen, out = set(), []
    for a, b, _ in _PAIRS:
        for s in (a, b):
            if s not in seen:
                seen.add(s)
                out.append(s)
    out += [
        "class A:\n    x = 1\n    def m(self):\n        return self.x",
        "for i in range(10):\n    print(i)",
        "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    x = -1",
    ]
    return out


def _bench_corpus():
    """~2400 unique valid programs — unique strings defeat memoization."""
    import random
    rng = random.Random(1234)
    ops = ["+", "-", "*", "//", "%", "|", "&", "^"]
    out = []
    for i in range(2400):
        a, b, c = rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99)
        f, g, h = rng.choice(ops), rng.choice(ops), rng.choice(ops)
        if i % 3 == 0:
            out.append(
                f"def f_{i}(x):\n    y=x{f}{a}  # tail\n"
                f"    return y{g}(x{h}{b})")
        elif i % 3 == 1:
            out.append(
                f"def f_{i}(x) :\n        y = x {f} {a}\n"
                f"        return (y) {g} x {h} ({b})")
        else:
            out.append(
                f"x_{i} = [{a}, {b} {f} {c}]\n"
                f"def g_{i}(q): return q {g} x_{i}[0]")
    return out


def test(ns):
    ref = _ref()
    fn = ns.get("canon_code")
    if not callable(fn):
        return {"ok": False, "detail": "no canon_code"}
    # pair equivalence classes must match the reference exactly
    for i, (a, b, want_same) in enumerate(_PAIRS):
        try:
            ka, kb = fn(a), fn(b)
        except Exception as e:
            return {"ok": False,
                    "detail": f"pair {i} raised {type(e).__name__}: {e}"}
        got_same = ka == kb
        if got_same != want_same:
            return {"ok": False,
                    "detail": f"pair {i}: same={got_same} want={want_same}"}
    # determinism + hashable + right type over the whole corpus
    for s in _corpus():
        try:
            k1, k2 = fn(s), fn(s)
        except Exception as e:
            return {"ok": False,
                    "detail": f"corpus raised {type(e).__name__}: {e}"}
        if k1 != k2:
            return {"ok": False, "detail": "non-deterministic key"}
        try:
            hash(k1)
        except Exception:
            return {"ok": False, "detail": f"unhashable key {type(k1)}"}
        if not isinstance(k1, (str, bytes)):
            return {"ok": False, "detail": f"bad key type {type(k1)}"}
    return {"ok": True}


def bench(ns):
    import time
    fn = ns["canon_code"]
    corpus = _bench_corpus()
    for s in corpus[:64]:          # warmup
        fn(s)
    t0 = time.perf_counter()
    reps = 4
    n = 0
    for _ in range(reps):
        for s in corpus:
            fn(s)
            n += 1
    return {"ms_per_call": (time.perf_counter() - t0) / n * 1000}

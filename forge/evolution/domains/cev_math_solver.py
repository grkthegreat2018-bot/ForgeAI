"""Non-code evolution domain: a small math/reasoning solver.

This is a code_evolve TARGET SPEC (SEED_CODE/test/bench contract — see
forge/evolution/targets/fast_softmax.py), NOT a BaseDomain subclass; it
lives in domains/ to mark it as a non-ML-kernel problem class.

The searched program is `solve_problem(problem) -> number` over a
generated suite of arithmetic / linear-equation / modpow / gcd cases.
test() checks numeric answers on generated cases (oracle computed at
generation time); the SEED handles only the easy slice, so `accuracy`
has real headroom — a candidate that implements a full safe evaluator
plus math.gcd/pow reaches ~1.0.

Run standalone:  python forge/evolution/domains/cev_math_solver.py
Evolve it:       python scripts/rsi_evolve.py --targets cev_math_solver \
                    --evals 60 --no-train --rounds 1
"""

NAME = "cev_math_solver"
GOAL = "max"
METRIC = "accuracy"
TAGS = ["math", "reasoning", "solver", "noncode"]
BUDGETS = {"accuracy": (">=", 0.5), "ms_per_call": ("<=", 100.0)}
REQUIRED = ["solve_problem"]
PROMPT = ("Non-code target: a math solver. The seed only handles easy "
          "arithmetic (+,-,*) and linear equations; harder expression "
          "ops (//, %, **, unary minus), gcd and modpow cases are "
          "answered wrong. Broaden coverage safely — no eval of "
          "arbitrary code paths beyond the provided problem strings.")

SEED_CODE = '''
import ast
import operator


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul}


def _eval_expr(expr):
    """Minimal safe evaluator: integers, +, -, * and parentheses only."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval"))


def solve_problem(p):
    """Solve one generated problem dict -> number (None = cannot)."""
    kind = p.get("kind")
    if kind == "linear":
        a, b, c = p["a"], p["b"], p["c"]
        return (c - b) / a
    if kind == "arith":
        try:
            return _eval_expr(p["expr"])
        except Exception:
            return None
    return None
'''


# ── problem generation (oracle answers computed here, at gen time) ────
def _gen_arith(rng, hard: bool):
    """Random expression string + its value. hard adds //, %, **, unary
    minus — ops the seed cannot evaluate."""
    def leaf():
        return str(rng.randint(1, 20))

    if hard:
        ops = ["+", "-", "*", "//", "%"]
        a = leaf()
        # build (((a o1 b) o2 c) o3 d) — // and % guarded to nonzero rhs
        expr = a
        val = int(a)
        for _ in range(rng.randint(2, 4)):
            op = rng.choice(ops)
            if op in ("//", "%"):
                rhs = rng.randint(1, 9)
                expr = f"({expr} {op} {rhs})"
                val = val // rhs if op == "//" else val % rhs
            elif op == "*":
                rhs = rng.randint(1, 9)
                expr = f"({expr} * {rhs})"
                val = val * rhs
            elif op == "+":
                rhs = rng.randint(-15, 15)
                expr = f"({expr} + {rhs})"
                val = val + rhs
            else:
                rhs = rng.randint(-15, 15)
                expr = f"({expr} - {rhs})"
                val = val - rhs
        if rng.random() < 0.3:
            e = rng.randint(2, 3)
            b = rng.randint(2, 6)
            expr = f"({expr} + {b} ** {e})"
            val = val + b ** e
        return expr, val
    a, b, c, d = (rng.randint(1, 15) for _ in range(4))
    o1, o2 = rng.choice(["+", "-", "*"]), rng.choice(["+", "-", "*"])
    expr = f"(({a} {o1} {b}) {o2} {c}) + {d}"
    return expr, eval(expr)          # spec-generated expr only


def _gen_cases(n: int, seed: int) -> list[tuple[dict, float]]:
    import random
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:                             # easy arith — seed OK
            expr, val = _gen_arith(rng, hard=False)
            cases.append(({"kind": "arith", "expr": expr}, float(val)))
        elif r < 0.60:                           # linear — seed OK
            a = rng.randint(1, 12)
            x = rng.randint(-9, 9)
            b = rng.randint(-20, 20)
            c = a * x + b
            cases.append(({"kind": "linear", "a": a, "b": b, "c": c},
                          float(x)))
        elif r < 0.75:                           # hard arith — seed fails
            expr, val = _gen_arith(rng, hard=True)
            cases.append(({"kind": "arith", "expr": expr}, float(val)))
        elif r < 0.88:                           # gcd — seed fails
            import math
            a, b = rng.randint(6, 400), rng.randint(6, 400)
            cases.append(({"kind": "gcd", "a": a, "b": b},
                          float(math.gcd(a, b))))
        else:                                    # modpow — seed fails
            a, e, m = rng.randint(2, 12), rng.randint(2, 8), \
                rng.randint(3, 97)
            cases.append(({"kind": "modpow", "a": a, "e": e, "m": m},
                          float(pow(a, e, m))))
    return cases


def _accuracy(fn, cases) -> tuple[float, int]:
    correct = 0
    for p, ans in cases:
        try:
            y = fn(p)
            if y is not None and abs(float(y) - ans) <= \
                    max(1e-6, 1e-9 * abs(ans)):
                correct += 1
        except Exception:
            pass
    return correct / len(cases), correct


def test(ns) -> dict:
    fn = ns.get("solve_problem")
    if not callable(fn):
        return {"ok": False, "detail": "no solve_problem"}
    cases = _gen_cases(60, seed=7)
    try:
        acc, correct = _accuracy(fn, cases)
    except Exception as e:
        return {"ok": False, "detail": f"raised {e}"}
    if acc < 0.5:
        return {"ok": False,
                "detail": f"accuracy {acc:.2f} < 0.5 ({correct}/60)"}
    return {"ok": True, "accuracy": acc}


def bench(ns) -> dict:
    import time
    fn = ns["solve_problem"]
    cases = _gen_cases(200, seed=9999)   # held-out: test() uses seed 7
    for p, _ in cases[:10]:
        fn(p)
    t0 = time.perf_counter()
    acc, correct = _accuracy(fn, cases)
    ms = (time.perf_counter() - t0) / len(cases) * 1000
    return {"accuracy": acc, "ms_per_call": ms}


if __name__ == "__main__":
    _ns: dict = {}
    exec(SEED_CODE, _ns)
    print("seed test:", test(_ns))
    print("seed bench:", bench(_ns))

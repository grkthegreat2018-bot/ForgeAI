"""AST mutation proposer — LLM-free candidate generation for CodeEvolve.

Small coder LLMs mostly regurgitate the seed (measured: 17/21 verbatim
copies from Qwen2.5-Coder-0.5B). Most real speed wins are mechanical:
dedupe repeated calls, hoist per-call constructors, fuse ops. This
engine does exactly that on the Python AST — zero model load, ~0ms per
candidate, always syntactically valid output (ast.unparse). The same
test/bench/verify harness judges results; wrong mutations die at the
correctness gate just like bad LLM output.

Extracts the parent program from the evolution prompt's first
```python fence (and the inspiration program from the second, when
present, for statement-level crossover).
"""
from __future__ import annotations

import ast
import re
from typing import Callable

import numpy as np

_FENCE = re.compile(r"```python\s*\n(.*?)```", re.S)

# Loop-discovered optimization (canon_key target, +22.6% verified):
# parse/unparse run on every parent canon AND every emitted variant —
# module-level binds turn LOAD_GLOBAL+LOAD_ATTR chains into LOAD_FAST.
_parse = ast.parse
_unparse = ast.unparse

# Calls safe to memoize per (name, args, kwargs) — shape/index/weight
# constructors. NOT ones commonly mutated in place (zeros/full get
# written into sometimes; the test gate still guards semantics).
_CTOR_NAMES = {"linspace", "arange", "tensor", "as_tensor", "eye",
               "ones", "full"}

# Method-name <-> torch.func-name pairs both directions are legal.
_FUNCIFY = {"sort", "sum", "mean", "max", "min", "argmax", "argmin",
            "topk", "exp", "log", "abs", "sqrt", "relu", "sigmoid",
            "tanh", "softmax", "cumsum", "prod", "flatten", "squeeze"}

_CTOR_PREAMBLE = (
    "_CTOR_CACHE = {}\n\n\n"
    "def _ctor(fn, *a, **k):\n"
    "    key = (fn.__name__, repr(a), repr(sorted(k.items())))\n"
    "    v = _CTOR_CACHE.get(key)\n"
    "    if v is None:\n"
    "        v = fn(*a, **k)\n"
    "        _CTOR_CACHE[key] = v\n"
    "    return v.clone()\n"   # callers may mutate; never alias the cache
)


def _same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def _torch(name: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                         attr=name, ctx=ast.Load())


def _is_torch_call(node: ast.AST, names: set[str]) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr in names)


def _pure_args(call: ast.Call) -> bool:
    """Args are safe to key on: constants, names, attributes, simple
    subscripts — no nested calls (side effects unknown)."""
    ok = (ast.Constant, ast.Name, ast.Attribute, ast.Subscript,
          ast.UnaryOp, ast.Tuple)
    return all(isinstance(a, ok) for a in call.args) and all(
        isinstance(kw.value, ok) for kw in call.keywords)


class _SortReuse(ast.NodeTransformer):
    """`a, b = torch.sort(X, kw)` followed by `torch.sort(X, same kw).indices`
    or `.values` — reuse the first result instead of re-sorting. Renames
    `_`/`_ignored` bindings so the values/indices name is available."""

    def __init__(self) -> None:
        self.bindings: list[tuple[ast.Call, str | None, str | None]] = []
        self.changed = False

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        val = node.value
        if (_is_torch_call(val, {"sort"})
                and isinstance(node.targets[0], ast.Tuple)
                and len(node.targets[0].elts) == 2):
            v, i = node.targets[0].elts
            if not all(isinstance(t, ast.Name) for t in (v, i)):
                # `self.x, self.y = torch.sort(...)`: rewriting a non-
                # Name target would silently drop the attribute store.
                return node
            vn = v.id if isinstance(v, ast.Name) else None
            in_ = i.id if isinstance(i, ast.Name) else None
            # Give throwaway names real ones so later uses can reuse them.
            if vn in (None, "_"):
                vn = "_sv"
                node.targets[0].elts[0] = ast.Name(id="_sv", ctx=ast.Store())
                self.changed = True
            if in_ in (None, "_"):
                in_ = "_si"
                node.targets[0].elts[1] = ast.Name(id="_si", ctx=ast.Store())
                self.changed = True
            self.bindings.append((val, vn, in_))
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if node.attr in ("indices", "values") and _is_torch_call(
                node.value, {"sort"}):
            for call, vn, in_ in self.bindings:
                if _same(call, node.value):
                    self.changed = True
                    return ast.Name(id=in_ if node.attr == "indices" else vn,
                                    ctx=ast.Load())
        return node


class _DiffSwap(ast.NodeTransformer):
    """x[..., :-1] - x[..., 1:]  ->  -torch.diff(x, dim=-1)
    (torch.diff is x[i+1]-x[i]; seed semantics are x[i]-x[i+1] → negate)."""
    changed = False

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not (isinstance(node.op, ast.Sub)
                and isinstance(node.left, ast.Subscript)
                and isinstance(node.right, ast.Subscript)
                and _same(node.left.value, node.right.value)):
            return node
        ls, rs = node.left.slice, node.right.slice
        lsl, rsl = self._the_slice(ls), self._the_slice(rs)
        ax_l, ax_r = self._diff_axis(ls), self._diff_axis(rs)
        if (lsl is not None and ax_l is not None and ax_l == ax_r
                and self._is_head_slice(lsl)
                and self._is_tail_slice(rsl)):
            self.changed = True
            diff = ast.Call(func=_torch("diff"),
                            args=[node.left.value],
                            keywords=[ast.keyword(
                                arg="dim", value=ast.Constant(ax_l))])
            return ast.UnaryOp(op=ast.USub(), operand=diff)
        return node

    @staticmethod
    def _is_full_slice(s: ast.AST) -> bool:
        return (isinstance(s, ast.Slice) and s.lower is None
                and s.upper is None and s.step is None)

    @classmethod
    def _the_slice(cls, s: ast.AST) -> ast.Slice | None:
        """The single distinguishing slice; None when the subscript has
        zero or several (can't prove both sides slice the same axis)."""
        if isinstance(s, ast.Slice):
            return s
        if isinstance(s, ast.Tuple):
            c = [e for e in s.elts if isinstance(e, ast.Slice)
                 and not cls._is_full_slice(e)]
            return c[0] if len(c) == 1 else None
        return None

    @classmethod
    def _diff_axis(cls, s: ast.AST) -> int | None:
        """torch.diff axis matching the distinguishing slice. `x[:-1]`
        is axis 0, NOT -1 (ndim unknown); `x[..., :-1]` is -1. Fixes
        the old always--1 rewrite that silently diffs the wrong axis
        for `x[:-1, :]`/`x[:-1]` patterns on multi-dim tensors."""
        if isinstance(s, ast.Slice):
            return 0
        if not isinstance(s, ast.Tuple):
            return None
        elts = s.elts
        is_ell = lambda e: (isinstance(e, ast.Constant)
                            and e.value is Ellipsis)
        idx = [k for k, e in enumerate(elts)
               if isinstance(e, ast.Slice)
               and not cls._is_full_slice(e)]
        if len(idx) != 1:
            return None
        k = idx[0]
        if any(is_ell(e) for e in elts[:k]):
            return -(sum(1 for e in elts[k:] if not is_ell(e)))
        return k

    @staticmethod
    def _is_head_slice(s: ast.AST) -> bool:   # :-1
        return (isinstance(s, ast.Slice) and s.upper is not None
                and isinstance(s.upper, ast.UnaryOp)
                and isinstance(s.upper.op, ast.USub)
                and isinstance(s.upper.operand, ast.Constant)
                and s.upper.operand.value == 1 and s.lower is None
                and s.step is None)

    @staticmethod
    def _is_tail_slice(s: ast.AST) -> bool:   # x[..., 1:]
        return (isinstance(s, ast.Slice) and s.lower is not None
                and isinstance(s.lower, ast.Constant)
                and s.lower.value == 1 and s.upper is None
                and s.step is None)


class _HoistCtors(ast.NodeTransformer):
    """Per-call torch constructors (linspace/arange/...) -> cached _ctor()
    keyed on (name, args, kwargs) — device/dtype ride inside the key so
    different devices still get their own entry."""
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _is_torch_call(node, _CTOR_NAMES) and _pure_args(node):
            self.changed = True
            return ast.Call(
                func=ast.Name(id="_ctor", ctx=ast.Load()),
                args=[node.func] + list(node.args),
                keywords=list(node.keywords))
        return node


class _MaskedWhere(ast.NodeTransformer):
    """x.masked_fill(m, v) -> torch.where(m, v, x) — `site` selects one
    call (AST order) instead of rewriting all of them."""
    def __init__(self, site: int | None = None) -> None:
        self.site = site
        self._idx = -1
        self.changed = False

    @staticmethod
    def eligible(node: ast.Call) -> bool:
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "masked_fill"
                and len(node.args) == 2)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if self.eligible(node):
            if self.site is not None:
                self._idx += 1
                if self._idx != self.site:
                    return node
            self.changed = True
            return ast.Call(func=_torch("where"),
                            args=[node.args[0], node.args[1],
                                  node.func.value],
                            keywords=[])
        return node


class _Funcify(ast.NodeTransformer):
    """x.m(...) <-> torch.m(x, ...) for pure tensor methods — both forms
    have different kernel dispatch paths; worth exploring either way.
    `site` selects one eligible call (AST order) instead of all."""
    def __init__(self, to_func: bool, site: int | None = None) -> None:
        self.to_func = to_func
        self.site = site
        self._idx = -1
        self.changed = False

    @staticmethod
    def eligible(node: ast.Call, to_func: bool) -> bool:
        f = node.func
        if to_func:
            return (isinstance(f, ast.Attribute)
                    and f.attr in _FUNCIFY and bool(node.args))
        return _is_torch_call(node, _FUNCIFY) and bool(node.args)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        f = node.func
        if self.to_func and isinstance(f, ast.Attribute) \
                and f.attr in _FUNCIFY and node.args:
            if self.site is not None:
                self._idx += 1
                if self._idx != self.site:
                    return node
            self.changed = True
            return ast.Call(func=_torch(f.attr),
                            args=[f.value] + list(node.args),
                            keywords=node.keywords)
        if not self.to_func and _is_torch_call(node, _FUNCIFY) and node.args:
            if self.site is not None:
                self._idx += 1
                if self._idx != self.site:
                    return node
            self.changed = True
            return ast.Call(func=ast.Attribute(value=node.args[0],
                                               attr=f.attr,
                                               ctx=ast.Load()),
                            args=list(node.args[1:]),
                            keywords=node.keywords)
        return node


class _LiteralPerturb(ast.NodeTransformer):
    """Nudge one numeric literal — but ONLY in tuning-knob positions:
    a bare Assign value (`eps = 1e-8`) or a Compare operand
    (`x > threshold`). Constants inside call args are semantics, not
    knobs (a 0.1->0.05 linspace-endpoint drift passed the gate once and
    changed real filtering behavior)."""
    def __init__(self, rng: np.random.Generator,
                 site: int | None = None) -> None:
        self.rng = rng
        self.site = site            # None -> legacy probabilistic mode
        self._idx = -1
        self.done = False

    def _nudge(self, node):
        if (self.done or not isinstance(node, ast.Constant)
                or not isinstance(node.value, float)):
            return node
        if self.site is not None:
            # Deterministic enumeration: nudge the site-th eligible
            # literal only (AST visit order is stable).
            self._idx += 1
            if self._idx != self.site:
                return node
            self.done = True
            factor = float(self.rng.choice([0.1, 0.5, 2.0, 10.0]))
            return ast.Constant(value=node.value * factor)
        if self.rng.random() < 0.4:
            self.done = True
            factor = float(self.rng.choice([0.1, 0.5, 2.0, 10.0]))
            return ast.Constant(value=node.value * factor)
        return node

    def visit_Assign(self, node: ast.Assign):
        node.value = self._nudge(node.value)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None:
            node.value = self._nudge(node.value)
        self.generic_visit(node)
        return node

    def visit_Compare(self, node: ast.Compare):
        if not self.done:
            node.left = self._nudge(node.left)
            node.comparators = [self._nudge(c) for c in node.comparators]
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call):
        # Do NOT descend into call args — constructor args (linspace,
        # arange, full, tensor values) are behavior, not tuning.
        node.func = self.visit(node.func)
        return node


class _ArithIdentity(ast.NodeTransformer):
    """x*1, 1*x, x+0, 0+x, x-0, x/1, x**1 -> x — but only in expression
    positions (return/compare/bool/call args) where eliding can't change
    object-identity semantics on assignment targets."""
    def __init__(self) -> None:
        self._safe = 0
        self.changed = False

    def _enter(self, node):
        self._safe += 1
        self.generic_visit(node)
        self._safe -= 1
        return node

    visit_Return = _enter
    visit_Compare = _enter
    visit_BoolOp = _enter
    visit_Call = _enter

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not self._safe:
            return node
        def c(n, v):
            return isinstance(n, ast.Constant) and n.value == v
        op = node.op
        out = None
        if isinstance(op, ast.Mult):
            if c(node.right, 1):
                out = node.left
            elif c(node.left, 1):
                out = node.right
        elif isinstance(op, (ast.Add, ast.Sub)) and c(node.right, 0):
            out = node.left
        elif isinstance(op, ast.Add) and c(node.left, 0):
            out = node.right
        elif isinstance(op, (ast.Div, ast.Pow)) and c(node.right, 1):
            out = node.left
        if out is None:
            return node
        self.changed = True
        return out


class _SetMember(ast.NodeTransformer):
    """x in [c,...] / x in (c,...) -> x in {c,...} — constants only.
    Set membership is O(1) vs O(k) list scan; identical semantics."""
    changed = False

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        for i, (op, comp) in enumerate(zip(node.ops, node.comparators)):
            if isinstance(op, (ast.In, ast.NotIn)) \
                    and isinstance(comp, (ast.List, ast.Tuple)) and comp.elts \
                    and all(isinstance(e, ast.Constant) for e in comp.elts):
                node.comparators[i] = ast.Set(elts=list(comp.elts))
                self.changed = True
        return node


def _assigned_names(tree: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(tree):
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, ast.AugAssign):
            tgts = [n.target]
        elif isinstance(n, ast.AnnAssign) and n.target:
            tgts = [n.target]
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            tgts = [n.target]
        elif isinstance(n, ast.NamedExpr):
            tgts = [n.target]
        elif isinstance(n, ast.Delete):
            tgts = n.targets
        for t in tgts:
            names.update(_unparse(x) for x in ast.walk(t)
                         if isinstance(x, (ast.Name, ast.Attribute)))
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
    return names


class _LocalsBind(ast.NodeTransformer):
    """Attribute loads used >=3x in a function (np.mean, torch.where,
    obj.meth, ...) -> bind once to a local at fn top. Classic CPython win:
    LOAD_FAST vs repeated LOAD_ATTR chains. Guarded: the attribute must
    never be assigned inside the function (would stale the alias)."""
    def __init__(self, rng: np.random.Generator,
                 rank: int = 0) -> None:
        self.rng = rng
        self.rank = rank             # which eligible attr to bind (0..2)
        self.changed = False
        self._n = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        from collections import Counter
        uses: Counter = Counter()
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) \
                    and isinstance(n.value, (ast.Name, ast.Attribute)):
                uses[_unparse(n)] += 1
        if not uses:
            return node
        assigned = _assigned_names(node)
        existing = {_unparse(n) for n in ast.walk(node)
                    if isinstance(n, ast.Name)}
        elig = 0
        for key, cnt in uses.most_common(3):
            # >=3x is the classic CPython win; >=1x still enumerates real
            # variants on thin seeds (single-return fns) where stricter
            # thresholds leave zero mutation surface.
            if cnt < 1 or key in assigned:
                continue
            # Root must be resolvable at fn entry: a param or a module-
            # level global (torch/np/self); a local assigned later would
            # NameError the bind line.
            root = key.split(".", 1)[0]
            if root in assigned and root not in {
                    a.arg for a in (node.args.posonlyargs + node.args.args
                                    + node.args.kwonlyargs)}:
                continue
            alias = f"_lb{self._n}"
            if alias in existing:
                continue
            if elig != self.rank:        # enumeration: bind only the
                elig += 1                # rank-th eligible candidate
                continue
            swapper = _AttrSwap(key, alias)
            node = swapper.visit(node)
            if not swapper.n:
                continue
            bind = ast.Assign(
                targets=[ast.Name(id=alias, ctx=ast.Store())],
                value=_parse(key, mode="eval").body)
            pos = 1 if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)) else 0
            node.body.insert(pos, bind)
            self._n += 1
            self.changed = True
            break
        return node


class _AttrSwap(ast.NodeTransformer):
    def __init__(self, key: str, alias: str) -> None:
        self.key = key
        self.alias = alias
        self.n = 0

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if _unparse(node) == self.key:
            self.n += 1
            return ast.Name(id=self.alias, ctx=ast.Load())
        return node


class _WhileFor(ast.NodeTransformer):
    """i = lo; while i < hi: ...; i += step  ->  for i in range(lo,hi,step)
    Only fires on the strict counting pattern: no `continue`, no other
    writes to i inside, no `i` use after the loop, empty orelse."""
    changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self._fix(node.body)
        self.generic_visit(node)
        return node

    def _fix(self, body: list) -> list:
        out = []
        i = 0
        while i < len(body):
            cur = body[i]
            nxt = body[i + 1] if i + 1 < len(body) else None
            conv = self._match(cur, nxt, body[i + 2:])
            if conv is not None:
                out.append(conv)
                i += 2
                self.changed = True
            else:
                out.append(cur)
                i += 1
        return out

    @staticmethod
    def _match(init, while_node, rest):
        if not (isinstance(init, ast.Assign) and len(init.targets) == 1
                and isinstance(init.targets[0], ast.Name)
                and isinstance(init.value, ast.Constant)
                and isinstance(init.value.value, int)
                and isinstance(while_node, ast.While)
                and not while_node.orelse):
            return None
        var = init.targets[0].id
        t = while_node.test
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1
                and isinstance(t.ops[0], ast.Lt)
                and isinstance(t.left, ast.Name) and t.left.id == var):
            return None
        hi = t.comparators[0]
        stmts = while_node.body
        if len(stmts) < 2 or not (
                isinstance(stmts[-1], ast.AugAssign)
                and isinstance(stmts[-1].target, ast.Name)
                and stmts[-1].target.id == var
                and isinstance(stmts[-1].op, ast.Add)
                and isinstance(stmts[-1].value, ast.Constant)
                and isinstance(stmts[-1].value.value, int)):
            return None
        inner = stmts[:-1]
        if any(isinstance(n, ast.Continue) for n in ast.walk(ast.Module(
                body=inner, type_ignores=[]))):
            return None
        assigned = _assigned_names(ast.Module(body=inner, type_ignores=[]))
        if var in assigned:
            return None
        # for-range evaluates lo/hi/step ONCE; while re-evaluates every
        # pass — the bound must be loop-invariant or the rewrite moves
        # the goalposts.
        if any(isinstance(n, ast.Name) and n.id in assigned
               for n in ast.walk(hi)):
            return None
        # `i` must be dead after the loop (for leaves it at hi-1, while
        # at hi) — conservative: no reads in the rest of the block.
        for s in rest:
            if var in {_unparse(n) for n in ast.walk(s)
                       if isinstance(n, ast.Name)}:
                return None
        return ast.For(
            target=ast.Name(id=var, ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                          args=[init.value, hi, stmts[-1].value],
                          keywords=[]),
            body=inner, orelse=[])


def _crossover(parent: ast.Module, insp: ast.Module | None,
               rng: np.random.Generator) -> ast.Module | None:
    """Swap a contiguous statement span between same-named functions of
    parent and inspiration."""
    if insp is None:
        return None
    def fns(t: ast.Module) -> dict[str, ast.FunctionDef]:
        return {n.name: n for n in t.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    pf, inf = fns(parent), fns(insp)
    common = [k for k in pf if k in inf and pf[k].body and inf[k].body]
    if not common:
        return None
    name = str(rng.choice(common))
    a, b = pf[name].body, inf[name].body
    span = (int(rng.integers(0, len(a))), int(rng.integers(0, len(b))))
    take = int(rng.integers(1, max(2, len(b) - span[1] + 1)))
    new_body = (a[:span[0]] + b[span[1]:span[1] + take] + a[span[0]:])
    pf[name].body = new_body
    return parent


def _ensure_import(tree: ast.Module, module: str) -> None:
    for n in tree.body:
        if isinstance(n, ast.Import) and any(a.name == module
                                             for a in n.names):
            return
        if isinstance(n, ast.ImportFrom) and n.module == module:
            return
    tree.body.insert(0, ast.Import(names=[ast.alias(name=module)]))


class Mutator:
    """propose_fn-compatible callable: prompt -> mutated program code.

    Applies 1-3 AST transforms to the parent program from the prompt.
    `primed=False` so CodeEvolve uses the instruction-style prompt ending
    (the mutator ignores the text anyway).

    Learned operator policy: op selection weights come from a Thompson-
    sampled Beta bandit persisted in FindingsDB.mutator_op_stats keyed
    by (op, tag_sig). CodeEvolve calls `note(code, credit, delta)` after
    each outcome — the policy literally resumes where the previous run
    left off and transfers across targets sharing tags.
    """

    primed = False

    _OPS = ("sort_reuse", "hoist_ctor", "diff_swap", "masked_where",
            "funcify", "literal", "crossover", "arith_id", "set_member",
            "locals_bind", "while_for")
    _BASE_W = {"sort_reuse": 3.0, "hoist_ctor": 3.0, "diff_swap": 1.0,
               "masked_where": 1.0, "funcify": 1.0, "literal": 0.5,
               "crossover": 1.5, "arith_id": 1.0, "set_member": 1.0,
               "locals_bind": 2.0, "while_for": 1.0}

    def __init__(self, seed: int = 0, db=None, tag_sig: str = "") -> None:
        self.rng = np.random.default_rng(seed)
        self.db = db
        self.tag_sig = tag_sig
        self._stats: dict[str, list[float]] = {
            op: [1.0, 1.0] for op in self._OPS}
        self._ops_by_code: dict[str, list[str]] = {}
        self._load_stats()

    def _load_stats(self) -> None:
        if self.db is None:
            return
        try:
            rows = self.db.conn.execute(
                "SELECT op, alpha, beta FROM mutator_op_stats "
                "WHERE tag_sig=?", (self.tag_sig,)).fetchall()
            for op, a, b in rows:
                if op in self._stats:
                    self._stats[op] = [max(1.0, a), max(1.0, b)]
        except Exception:
            pass

    def note(self, code: str, credit: float, delta: float,
             prompt: str | None = None) -> None:
        """Outcome feedback for the bandit — credit in [0,1]:
        1.0 verified promotion, 0.5 evaluated-better-not-promoted,
        0.25 evaluated-not-better, 0.1 test/exec fail, 0.0 gate reject.
        `prompt` is unused here (LLMProposer's hint bandit consumes it)."""
        ops = self._ops_by_code.pop(code, None)
        if not ops:
            return
        for op in ops:
            a, b = self._stats[op]
            a += credit
            b += (1.0 - credit)
            self._stats[op] = [a, b]
            if self.db is not None:
                try:
                    self.db.conn.execute(
                        "INSERT INTO mutator_op_stats(op, tag_sig, alpha,"
                        " beta, n, delta_mean) VALUES(?,?,?,?,1,?) "
                        "ON CONFLICT(op, tag_sig) DO UPDATE SET "
                        "alpha=alpha+?, beta=beta+?, n=n+1, "
                        "delta_mean=delta_mean+(?-delta_mean)/(n+1)",
                        (op, self.tag_sig, a, b, delta,
                         credit, 1.0 - credit, delta))
                    self.db.conn.commit()
                except Exception:
                    pass

    def __call__(self, prompt: str) -> str:
        blocks = _FENCE.findall(prompt)
        if not blocks:
            return "pass\n"
        parent_src = blocks[0]
        insp_src = blocks[1] if len(blocks) > 1 else None
        try:
            tree = _parse(parent_src)
        except SyntaxError:
            return parent_src
        insp_tree = None
        if insp_src:
            try:
                insp_tree = _parse(insp_src)
            except SyntaxError:
                insp_tree = None

        transforms: dict[str, Callable] = {
            "sort_reuse": lambda t: self._apply(t, _SortReuse()),
            "hoist_ctor": lambda t: self._hoist(t),
            "diff_swap": lambda t: self._apply(t, _DiffSwap()),
            "masked_where": lambda t: self._apply(t, _MaskedWhere()),
            "funcify": lambda t: self._apply(
                t, _Funcify(bool(self.rng.integers(2)))),
            "literal": lambda t: self._apply(t, _LiteralPerturb(self.rng)),
            "crossover": lambda t: _crossover(t, insp_tree, self.rng) or t,
            "arith_id": lambda t: self._apply(t, _ArithIdentity()),
            "set_member": lambda t: self._apply(t, _SetMember()),
            "locals_bind": lambda t: self._apply(
                t, _LocalsBind(self.rng)),
            "while_for": lambda t: self._apply(t, _WhileFor()),
        }
        # Thompson sample each op's bandit arm -> selection weight.
        w = np.array([self._BASE_W[op] * self.rng.beta(*self._stats[op])
                      for op in self._OPS])
        w /= w.sum()
        n_ops = 1 + int(self.rng.integers(0, 3))
        idxs = self.rng.choice(len(self._OPS), size=min(n_ops, len(self._OPS)),
                               replace=False, p=w)
        applied: list[str] = []
        for i in idxs:
            op = self._OPS[int(i)]
            try:
                tree = transforms[op](tree) or tree
                applied.append(op)
            except Exception:
                continue
        try:
            ast.fix_missing_locations(tree)
            code = _unparse(tree)
        except Exception:
            code = parent_src
        if code != parent_src:
            self._ops_by_code[code] = applied
            if len(self._ops_by_code) > 128:
                self._ops_by_code.pop(next(iter(self._ops_by_code)))
        return code

    def propose_batch(self, prompts: list[str]) -> list[str]:
        """Mutations are free — batching is just a loop. Exposing the
        burst API lets the eval pool fill in one pass under workers>1."""
        return [self(p) for p in prompts]

    _ENUM_SINGLES = ("sort_reuse", "hoist_ctor", "diff_swap",
                     "arith_id", "set_member", "while_for")

    def _variant(self, op: str, src: str, insp_tree=None, site=None,
                 seed: int | None = None,
                 canon: str | None = None) -> str | None:
        """Apply one op on a FRESH parse of src — returns code or None.
        Each enumerated variant gets its own tree (transformers mutate
        in place). Variants that leave the AST semantically unchanged
        return None (canon comparison — `changed` flags don't cover
        unparse-level normalization)."""
        try:
            tree = _parse(src)
        except SyntaxError:
            return None
        if canon is None:
            canon = _unparse(_parse(src))
        rng = np.random.default_rng(seed) if seed is not None \
            else self.rng
        try:
            if op == "crossover":
                t2 = _crossover(tree, insp_tree, rng)
                if t2 is None:
                    return None
                tree = t2
            elif op == "funcify":
                # site=(direction, call_index); bare int keeps the
                # legacy all-calls behavior (site=None) with that dir.
                if isinstance(site, tuple):
                    d, i = site
                    tx = _Funcify(bool(d), site=i)
                elif site is not None:
                    tx = _Funcify(bool(site))
                else:
                    tx = _Funcify(bool(rng.integers(2)))
                tree = self._apply(tree, tx)
            elif op == "literal":
                tx = _LiteralPerturb(rng, site=site)
                tree = self._apply(tree, tx)
                if not tx.done:
                    return None        # site index had no literal
            elif op == "locals_bind":
                tree = self._apply(tree, _LocalsBind(rng, rank=site or 0))
            elif op == "sort_reuse":
                tree = self._apply(tree, _SortReuse())
            elif op == "hoist_ctor":
                tree = self._hoist(tree)
            elif op == "diff_swap":
                tree = self._apply(tree, _DiffSwap())
            elif op == "masked_where":
                tree = self._apply(tree, _MaskedWhere(site=site))
            elif op == "arith_id":
                tree = self._apply(tree, _ArithIdentity())
            elif op == "set_member":
                tree = self._apply(tree, _SetMember())
            elif op == "while_for":
                tree = self._apply(tree, _WhileFor())
            else:
                return None
            ast.fix_missing_locations(tree)
            out = _unparse(tree)
            return out if out != canon else None
        except Exception:
            return None

    def _compose(self, a: str, b: str, src: str,
                 canon: str | None = None) -> str | None:
        """Two-op composition: apply a then b on one tree."""
        try:
            tree = _parse(src)
        except SyntaxError:
            return None
        if canon is None:
            canon = _unparse(tree)
        txs = {"sort_reuse": _SortReuse, "hoist_ctor": None,
               "diff_swap": _DiffSwap, "masked_where": _MaskedWhere,
               "arith_id": _ArithIdentity, "set_member": _SetMember,
               "while_for": _WhileFor}
        try:
            for op in (a, b):
                if op == "hoist_ctor":
                    tree = self._hoist(tree)
                else:
                    tree = self._apply(tree, txs[op]())
            ast.fix_missing_locations(tree)
            out = _unparse(tree)
            return out if out != canon else None
        except Exception:
            return None

    def _op_order(self) -> list[str]:
        """Thompson-weighted op ordering — good ops first, still
        exploratory (same bandit as __call__)."""
        w = {op: self._BASE_W[op] * self.rng.beta(*self._stats[op])
             for op in self._OPS}
        return sorted(self._OPS, key=lambda o: -w[o])

    def mutants_for(self, src: str, insp_src: str | None = None,
                    max_family: int = 32, seed_base: int = 0):
        """Deterministic op×site fan-out — uniqueness by construction.

        The random-op mode produced a measured 91.6% duplicate-reject
        rate: hashing after generation discovers repeats. This
        enumerates the applicable (op, site) grid instead — each item
        carries a distinct op_sig so the caller dedups by KEY before
        eval, not by post-hoc program hash.

        Order: singles in bandit order (literal → up to 8 literal sites,
        funcify → 2 directions, locals_bind → 3 ranks, crossover → 4
        seeded spans), then two-op compositions over the top singles.
        Codes are registered in _ops_by_code so note() credits ops
        exactly like __call__.

        Yields (op_sig, code), at most `max_family` items, each a real
        code change vs src (None-ops and unchanged trees are skipped).
        """
        insp_tree = None
        if insp_src:
            try:
                insp_tree = _parse(insp_src)
            except SyntaxError:
                insp_tree = None
        try:
            tree0 = _parse(src)
            canon = _unparse(tree0)
        except SyntaxError:
            return
        calls = [n for n in ast.walk(tree0) if isinstance(n, ast.Call)]
        emitted: set[str] = set()
        count = 0

        def _emit(sig: str, code: str | None, ops: list[str]):
            nonlocal count
            if (count >= max_family or not code or code == src
                    or code in emitted):
                return None
            emitted.add(code)
            self._ops_by_code[code] = ops
            if len(self._ops_by_code) > 128:
                self._ops_by_code.pop(next(iter(self._ops_by_code)))
            count += 1
            return (sig, code)

        # Seed-dependent ops fold the family seed into the op_sig: the
        # same site CAN yield new variants on a later re-opening, and
        # _seen_ops keys exact (parent, sig, seed-slot) tuples — 8 slots
        # per site bounded, then genuinely exhausted.
        slot = seed_base & 7
        for op in self._op_order():
            if count >= max_family:
                return
            if op == "literal":
                for i in range(8):
                    out = _emit(f"literal.{i}.{slot}",
                                self._variant("literal", src, site=i,
                                              seed=slot * 97 + i,
                                              canon=canon),
                                ["literal"])
                    if out is None:
                        break           # site i+1 would have none too
                    yield out
            elif op == "funcify":
                for d in (0, 1):
                    n_sites = sum(1 for c in calls
                                  if _Funcify.eligible(c, bool(d)))
                    for i in range(min(n_sites, 8)):
                        out = _emit(f"funcify.{d}.{i}",
                                    self._variant(
                                        "funcify", src,
                                        site=(d, i), canon=canon),
                                    ["funcify"])
                        if out is None:
                            break
                        yield out
            elif op == "masked_where":
                n_sites = sum(1 for c in calls
                              if _MaskedWhere.eligible(c))
                for i in range(min(n_sites, 8)):
                    out = _emit(f"masked_where.{i}",
                                self._variant("masked_where", src,
                                              site=i, canon=canon),
                                ["masked_where"])
                    if out is None:
                        break
                    yield out
            elif op == "locals_bind":
                for r in range(3):
                    out = _emit(f"locals_bind.{r}",
                                self._variant("locals_bind", src,
                                              site=r, canon=canon),
                                ["locals_bind"])
                    if out is None:
                        break
                    yield out
            elif op == "crossover":
                if insp_tree is None:
                    continue
                for s in range(4):
                    out = _emit(f"crossover.{s}.{slot}",
                                self._variant("crossover", src,
                                              insp_tree,
                                              seed=slot * 131 + s,
                                              canon=canon),
                                ["crossover"])
                    if out is not None:
                        yield out
            elif op in self._ENUM_SINGLES:
                out = _emit(op, self._variant(op, src, None,
                                              canon=canon),
                            [op])
                if out is not None:
                    yield out
        tops = [o for o in self._op_order() if o in self._ENUM_SINGLES][:5]
        for a in tops:
            if count >= max_family:
                return
            for b in tops:
                if a == b or count >= max_family:
                    continue
                out = _emit(f"{a}+{b}",
                            self._compose(a, b, src, canon), [a, b])
                if out is not None:
                    yield out

    @staticmethod
    def _apply(tree: ast.Module, tx) -> ast.Module:
        return tx.visit(tree)

    def _hoist(self, tree: ast.Module) -> ast.Module:
        h = _HoistCtors()
        tree = h.visit(tree)
        if h.changed:
            _ensure_import(tree, "torch")
            # Prepend the cache preamble after any leading imports.
            pos = 0
            while pos < len(tree.body) and isinstance(
                    tree.body[pos], (ast.Import, ast.ImportFrom,
                                     ast.Expr)):
                pos += 1
            pre = _parse(_CTOR_PREAMBLE).body
            tree.body[pos:pos] = pre
        return tree


class MixProposer:
    """Mostly-free mutations, occasional LLM leaps. LLM fires every
    `llm_every`-th call; failure falls back to a mutation."""

    def __init__(self, mut: Mutator, llm, llm_every: int = 6) -> None:
        self.mut, self.llm, self.llm_every = mut, llm, llm_every
        self._n = 0
        self.primed = getattr(llm, "primed", False)

    def note(self, code: str, credit: float, delta: float,
             prompt: str | None = None) -> None:
        self.mut.note(code, credit, delta, prompt=prompt)
        # LLMProposer's hint bandit pops on prompt-key hit — a mut-made
        # candidate's prompt was never recorded, so this is a no-op
        # unless the LLM actually produced it.
        n = getattr(self.llm, "note", None)
        if n is not None:
            try:
                n(code, credit, delta, prompt=prompt)
            except Exception:
                pass

    @property
    def last_adapter(self) -> str | None:
        """Adapter tag used by the most recent LLM proposal (provenance
        for code_programs.adapter); None when the last call was a mut."""
        return getattr(self.llm, "_last_adapter", None)

    def __call__(self, prompt: str) -> str:
        self._n += 1
        if self._n % self.llm_every == 0:
            try:
                return self.llm(prompt)
            except Exception:
                pass
        self.llm._last_adapter = None   # provenance: this row is a mut
        return self.mut(prompt)

    def propose_batch(self, prompts: list[str]) -> list[str]:
        """Burst under the same llm_every policy: LLM-designated slots go
        through llm.propose_batch (one GPU pass) when supported; the rest
        are free mutations."""
        n = len(prompts)
        out: list[str | None] = [None] * n
        llm_idx: list[int] = []
        for i in range(n):
            self._n += 1
            if self._n % self.llm_every == 0:
                llm_idx.append(i)
            else:
                out[i] = self.mut(prompts[i])
        if not llm_idx:
            self.llm._last_adapter = None
        if llm_idx:
            lp = [prompts[i] for i in llm_idx]
            try:
                raws = (self.llm.propose_batch(lp)
                        if hasattr(self.llm, "propose_batch")
                        else [self.llm(p) for p in lp])
            except Exception:
                raws = [None] * len(lp)
            for i, raw in zip(llm_idx, raws):
                out[i] = raw if raw else self.mut(prompts[i])
        # llm.propose_batch may return fewer items than requested — any
        # unfilled slot falls back to a free mutation, never None.
        for i in range(n):
            if out[i] is None:
                out[i] = self.mut(prompts[i])
        return out

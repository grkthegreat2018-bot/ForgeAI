"""Pure-Python training data generator for ForgeAI V7 8B from-scratch training.

Generates JSONL training data using ONLY traditional compute (no API calls,
no LLM distillation). Every example has a deterministic, programmatically-
verified answer.

Categories generated (all mass-producible with pure Python):
1. Arithmetic with worked steps (10K) — teaches step-by-step calculation
2. Algebra solve-for-x with steps (10K) — teaches equation manipulation
3. Calculus derivatives with steps (5K) — teaches differentiation rules
4. Linear algebra operations (5K) — matrix/vector math
5. Number theory (5K) — GCD, LCM, primes, modular arithmetic
6. Logic syllogisms (5K) — deductive reasoning
7. Number sequence puzzles (5K) — pattern recognition
8. English grammar corrections (10K) — grammar error -> correction
9. Word problems with steps (10K) — multi-step math reasoning
10. Set theory / combinatorics (5K) — counting, permutations

Output: research/data/finetune/synthetic_*.jsonl
Format: {"prompt": "...", "response": "..."} (sft_train single-turn format)

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --count 50000
    python scripts/generate_synthetic_data.py --categories math,logic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
from pathlib import Path
from typing import Callable

# ─── Output directory ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "research" / "data" / "finetune"


# ─── Helpers ───────────────────────────────────────────────────────────────
def write_jsonl(path: Path, examples: list[dict]) -> int:
    """Write examples to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)


def fmt_num(n: float) -> str:
    """Format number: integers without decimal, floats with up to 2 places."""
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")


# ─── 1. Arithmetic with worked steps ───────────────────────────────────────
def gen_arithmetic(n: int, rng: random.Random) -> list[dict]:
    """Generate arithmetic problems with step-by-step solutions."""
    examples = []
    ops = [
        ("+", lambda a, b: a + b, "add"),
        ("-", lambda a, b: a - b, "subtract"),
        ("*", lambda a, b: a * b, "multiply"),
        ("/", lambda a, b: a / b if b != 0 else None, "divide"),
    ]
    for _ in range(n):
        op_sym, op_fn, op_name = rng.choice(ops)
        if op_sym == "/":
            b = rng.randint(2, 20)
            result = rng.randint(2, 50)
            a = b * result
        elif op_sym == "*":
            a = rng.randint(2, 99)
            b = rng.randint(2, 20)
            result = a * b
        elif op_sym == "+":
            a = rng.randint(10, 9999)
            b = rng.randint(10, 9999)
            result = a + b
        else:
            a = rng.randint(10, 9999)
            b = rng.randint(1, a)
            result = a - b

        if op_sym == "+":
            steps = f"  {a}\n+ {b}\n----\n  {result}"
        elif op_sym == "-":
            steps = f"  {a}\n- {b}\n----\n  {result}"
        elif op_sym == "*":
            steps = f"  {a}\n* {b}\n----\n  {result}"
        else:
            steps = f"  {a} / {b} = {result}\n  (since {b} * {result} = {a})"

        examples.append({
            "prompt": f"Calculate {a} {op_sym} {b}. Show your work.",
            "response": f"{a} {op_sym} {b} = {fmt_num(result)}\n\nWork:\n{steps}",
        })
    return examples


# ─── 2. Algebra solve-for-x with steps ─────────────────────────────────────
def gen_algebra(n: int, rng: random.Random) -> list[dict]:
    """Generate linear equation problems with step-by-step solutions."""
    examples = []
    for _ in range(n):
        x = rng.randint(-20, 50)
        if x == 0:
            x = 1
        # Generate equation forms: ax + b = c, ax - b = c, a(x + b) = c, etc.
        form = rng.randint(0, 3)
        if form == 0:
            # ax + b = c
            a = rng.randint(2, 12)
            b = rng.randint(-20, 20)
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}"
            steps = [
                f"{a}x + {b} = {c}",
                f"{a}x = {c} - {b} = {c - b}",
                f"x = {c - b} / {a} = {fmt_num((c - b) / a)}",
            ]
        elif form == 1:
            # ax - b = c
            a = rng.randint(2, 12)
            b = rng.randint(1, 20)
            c = a * x - b
            prompt = f"Solve for x: {a}x - {b} = {c}"
            steps = [
                f"{a}x - {b} = {c}",
                f"{a}x = {c} + {b} = {c + b}",
                f"x = {c + b} / {a} = {fmt_num((c + b) / a)}",
            ]
        elif form == 2:
            # a(x + b) = c
            a = rng.randint(2, 8)
            b = rng.randint(-10, 10)
            c = a * (x + b)
            prompt = f"Solve for x: {a}(x + {b}) = {c}"
            steps = [
                f"{a}(x + {b}) = {c}",
                f"x + {b} = {c} / {a} = {fmt_num(c / a)}",
                f"x = {fmt_num(c / a)} - {b} = {fmt_num(c / a - b)}",
            ]
        else:
            # ax/b = c  ->  ax = bc  ->  x = bc/a
            a = rng.randint(2, 10)
            b = rng.randint(2, 10)
            c = a * x // b
            if a * x % b != 0:
                c = a * x // b
                x_actual = c * b // a
            else:
                x_actual = x
            prompt = f"Solve for x: {a}x / {b} = {c}"
            steps = [
                f"{a}x / {b} = {c}",
                f"{a}x = {c} * {b} = {c * b}",
                f"x = {c * b} / {a} = {fmt_num(c * b / a)}",
            ]
        response = "\n".join(steps)
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 3. Calculus derivatives with steps ────────────────────────────────────
def gen_calculus(n: int, rng: random.Random) -> list[dict]:
    """Generate derivative problems with step-by-step solutions."""
    examples = []
    for _ in range(n):
        form = rng.randint(0, 5)
        if form == 0:
            # Polynomial: d/dx (ax^n + bx + c)
            a, pwr, b, c = rng.randint(1, 5), rng.randint(2, 5), rng.randint(1, 5), rng.randint(-10, 10)
            expr = f"{a}x^{pwr} + {b}x"
            if c >= 0:
                expr += f" + {c}"
            else:
                expr += f" - {abs(c)}"
            deriv = f"{a*pwr}x^{pwr-1} + {b}"
            prompt = f"Find d/dx of {expr}"
            response = (
                f"f(x) = {expr}\n\n"
                f"Using the power rule: d/dx(x^n) = n*x^(n-1)\n\n"
                f"d/dx({a}x^{pwr}) = {a}*{pwr}*x^{pwr-1} = {a*pwr}x^{pwr-1}\n"
                f"d/dx({b}x) = {b}\n"
                f"d/dx({c}) = 0\n\n"
                f"f'(x) = {deriv}"
            )
        elif form == 1:
            # Trigonometric: d/dx sin(ax), cos(ax), tan(ax)
            a = rng.randint(1, 6)
            fn = rng.choice([("sin", "cos"), ("cos", "-sin"), ("tan", "sec^2")])
            prompt = f"Find d/dx of {fn[0]}({a}x)"
            chain = f" (chain rule: multiply by d/dx({a}x) = {a})"
            response = (
                f"f(x) = {fn[0]}({a}x)\n\n"
                f"Using chain rule:\n"
                f"  d/dx {fn[0]}(u) = {fn[1]}(u) * du/dx\n"
                f"  u = {a}x, du/dx = {a}\n\n"
                f"f'(x) = {a}{fn[1]}({a}x)"
            )
        elif form == 2:
            # Exponential: d/dx e^(ax), a^x
            a = rng.randint(1, 5)
            prompt = f"Find d/dx of e^({a}x)"
            response = (
                f"f(x) = e^({a}x)\n\n"
                f"Using chain rule:\n"
                f"  d/dx e^u = e^u * du/dx\n"
                f"  u = {a}x, du/dx = {a}\n\n"
                f"f'(x) = {a}e^({a}x)"
            )
        elif form == 3:
            # Logarithmic: d/dx ln(ax)
            a = rng.randint(1, 8)
            prompt = f"Find d/dx of ln({a}x)"
            response = (
                f"f(x) = ln({a}x)\n\n"
                f"Using chain rule:\n"
                f"  d/dx ln(u) = (1/u) * du/dx\n"
                f"  u = {a}x, du/dx = {a}\n\n"
                f"f'(x) = {a}/({a}x) = 1/x"
            )
        elif form == 4:
            # Product rule: d/dx (x^n * sin(x))
            pwr = rng.randint(2, 4)
            prompt = f"Find d/dx of x^{pwr} * sin(x)"
            response = (
                f"f(x) = x^{pwr} * sin(x)\n\n"
                f"Using product rule: (fg)' = f'g + fg'\n"
                f"  f = x^{pwr}, f' = {pwr}x^{pwr-1}\n"
                f"  g = sin(x), g' = cos(x)\n\n"
                f"f'(x) = {pwr}x^{pwr-1}*sin(x) + x^{pwr}*cos(x)"
            )
        else:
            # Quotient rule: d/dx (x^n / (x + a))
            pwr = rng.randint(2, 4)
            a = rng.randint(1, 5)
            prompt = f"Find d/dx of x^{pwr} / (x + {a})"
            response = (
                f"f(x) = x^{pwr} / (x + {a})\n\n"
                f"Using quotient rule: (f/g)' = (f'g - fg') / g^2\n"
                f"  f = x^{pwr}, f' = {pwr}x^{pwr-1}\n"
                f"  g = x + {a}, g' = 1\n\n"
                f"f'(x) = [{pwr}x^{pwr-1}*(x+{a}) - x^{pwr}*1] / (x+{a})^2\n"
                f"       = [{pwr}x^{pwr} + {pwr*a}x^{pwr-1} - x^{pwr}] / (x+{a})^2\n"
                f"       = [{pwr-1}x^{pwr} + {pwr*a}x^{pwr-1}] / (x+{a})^2"
            )
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 4. Linear algebra operations ──────────────────────────────────────────
def gen_linear_algebra(n: int, rng: random.Random) -> list[dict]:
    """Generate linear algebra problems (dot product, matrix mult, etc.)."""
    examples = []
    for _ in range(n):
        form = rng.randint(0, 3)
        if form == 0:
            # Dot product
            dim = rng.randint(2, 5)
            v1 = [rng.randint(-10, 10) for _ in range(dim)]
            v2 = [rng.randint(-10, 10) for _ in range(dim)]
            result = sum(a * b for a, b in zip(v1, v2))
            v1_str = ", ".join(str(x) for x in v1)
            v2_str = ", ".join(str(x) for x in v2)
            prompt = f"Compute the dot product of ({v1_str}) and ({v2_str})."
            calc = " + ".join(f"({a})*({b})" for a, b in zip(v1, v2))
            response = (
                f"v_1 = ({v1_str})\nv_2 = ({v2_str})\n\n"
                f"v_1 * v_2 = {calc}\n         = {result}"
            )
        elif form == 1:
            # Matrix 2x2 multiplication
            a, b, c, d = [rng.randint(-5, 5) for _ in range(4)]
            e, f, g, h = [rng.randint(-5, 5) for _ in range(4)]
            r1 = a*e + b*g
            r2 = a*f + b*h
            r3 = c*e + d*g
            r4 = c*f + d*h
            prompt = f"Multiply:\n  [{a} {b}]   [{e} {f}]\n  [{c} {d}] * [{g} {h}]"
            response = (
                f"  [{a} {b}]   [{e} {f}]   [{a}*{e}+{b}*{g}  {a}*{f}+{b}*{h}]   [{r1} {r2}]\n"
                f"  [{c} {d}] * [{g} {h}] = [{c}*{e}+{d}*{g}  {c}*{f}+{d}*{h}] = [{r3} {r4}]"
            )
        elif form == 2:
            # Matrix determinant 2x2
            a, b, c, d = [rng.randint(-9, 9) for _ in range(4)]
            det = a*d - b*c
            prompt = f"Find the determinant of:\n  [{a} {b}]\n  [{c} {d}]"
            response = (
                f"det = ({a})({d}) - ({b})({c})\n"
                f"    = {a*d} - {b*c}\n"
                f"    = {det}"
            )
        else:
            # Vector magnitude
            dim = rng.randint(2, 4)
            v = [rng.randint(-10, 10) for _ in range(dim)]
            mag_sq = sum(x*x for x in v)
            mag = math.sqrt(mag_sq)
            v_str = ", ".join(str(x) for x in v)
            prompt = f"Find the magnitude of vector ({v_str})."
            calc = " + ".join(f"{x}^2" for x in v)
            response = (
                f"v = ({v_str})\n\n"
                f"|v| = sqrt({calc})\n"
                f"    = sqrt({mag_sq})\n"
                f"    = {fmt_num(mag)}"
            )
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 5. Number theory ──────────────────────────────────────────────────────
def gen_number_theory(n: int, rng: random.Random) -> list[dict]:
    """Generate number theory problems (GCD, LCM, primes, modular arith)."""
    examples = []
    for _ in range(n):
        form = rng.randint(0, 3)
        if form == 0:
            a, b = rng.randint(10, 500), rng.randint(10, 500)
            g = math.gcd(a, b)
            prompt = f"Find GCD({a}, {b})."
            # Euclidean algorithm steps
            steps = [f"GCD({a}, {b})"]
            x, y = a, b
            while y:
                steps.append(f"  {x} = {y}*{x//y} + {x%y}")
                x, y = y, x % y
            steps.append(f"GCD = {x}")
            response = "\n".join(steps)
        elif form == 1:
            a, b = rng.randint(10, 200), rng.randint(10, 200)
            l = abs(a * b) // math.gcd(a, b)
            prompt = f"Find LCM({a}, {b})."
            response = (
                f"LCM({a}, {b}) = |{a} * {b}| / GCD({a}, {b})\n"
                f"             = {abs(a*b)} / {math.gcd(a, b)}\n"
                f"             = {l}"
            )
        elif form == 2:
            a = rng.randint(20, 500)
            # Prime factorization
            factors = []
            x = a
            for p in range(2, int(math.sqrt(a)) + 2):
                while x % p == 0:
                    factors.append(p)
                    x //= p
            if x > 1:
                factors.append(x)
            # Build factorization string
            from collections import Counter
            fc = Counter(factors)
            factor_str = " * ".join(
                f"{p}^{c}" if c > 1 else str(p) for p, c in sorted(fc.items())
            )
            prompt = f"Find the prime factorization of {a}."
            response = f"{a} = {factor_str}"
        else:
            # Modular arithmetic: a^b mod m
            a = rng.randint(2, 20)
            b = rng.randint(2, 10)
            m = rng.randint(5, 50)
            result = pow(a, b, m)
            prompt = f"Compute {a}^{b} mod {m}."
            # Show step-by-step modular exponentiation (square-and-multiply)
            steps = [f"{a}^{b} mod {m}", ""]
            steps.append(f"Using modular exponentiation (square and multiply):")
            cur = a % m
            steps.append(f"  {a} mod {m} = {cur}")
            for i in range(2, b + 1):
                cur = (cur * a) % m
                if i <= 3 or i == b:  # show first few + final
                    steps.append(f"  {a}^{i} mod {m} = {cur}")
                elif i == 4:
                    steps.append(f"  ... (continuing)")
            steps.append(f"\nAnswer: {a}^{b} mod {m} = {result}")
            response = "\n".join(steps)
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 6. Logic syllogisms ───────────────────────────────────────────────────
def gen_logic_syllogisms(n: int, rng: random.Random) -> list[dict]:
    """Generate syllogistic reasoning problems with valid/invalid conclusions."""
    examples = []
    # Syllogism figures and moods
    subjects = ["dogs", "cats", "birds", "fish", "mammals", "reptiles",
                "insects", "humans", "trees", "flowers", "rocks", "metals",
                "liquids", "gases", "planets", "stars", "students", "teachers",
                "doctors", "engineers", "lawyers", "scientists", "artists",
                "musicians", "athletes", "chefs", "farmers", "pilots"]
    predicates = ["are mortal", "need water", "can fly", "are warm-blooded",
                  "are cold-blooded", "have bones", "have scales", "are alive",
                  "are solid", "conduct electricity", "are flammable",
                  "are visible", "orbit the sun", "emit light", "study hard",
                  "are educated", "help people", "solve problems", "follow laws",
                  "do research", "create art", "play instruments", "train daily",
                  "cook food", "grow crops", "fly planes", "are rational",
                  "are creative", "are efficient", "are careful"]

    for _ in range(n):
        # Generate a valid categorical syllogism
        mid = rng.choice(subjects)
        major_subj = rng.choice([s for s in subjects if s != mid])
        minor_subj = rng.choice([s for s in subjects if s != mid and s != major_subj])
        pred = rng.choice(predicates)

        # Valid syllogism: All A are B, All C are A, therefore All C are B
        valid = rng.choice([True, True, False])  # 2/3 valid
        if valid:
            premise1 = f"All {mid} {pred}."
            premise2 = f"All {minor_subj} are {mid}."
            conclusion = f"Therefore, all {minor_subj} {pred}."
            answer = f"VALID. The conclusion follows from the premises by transitivity."
        else:
            # Invalid: affirming the consequent or undistributed middle
            premise1 = f"All {mid} {pred}."
            premise2 = f"All {major_subj} {pred}."  # same predicate, not middle term
            conclusion = f"Therefore, all {major_subj} are {mid}."
            answer = (
                f"INVALID. This commits the fallacy of the undistributed middle. "
                f"Both {mid} and {major_subj} {pred}, but that doesn't mean "
                f"{major_subj} are {mid}."
            )

        prompt = (
            f"Is the following syllogism valid or invalid? Explain.\n\n"
            f"Premise 1: {premise1}\n"
            f"Premise 2: {premise2}\n"
            f"Conclusion: {conclusion}"
        )
        response = f"{answer}\n\nPremise 1: {premise1}\nPremise 2: {premise2}\n{conclusion}"
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 7. Number sequence puzzles ────────────────────────────────────────────
def gen_sequences(n: int, rng: random.Random) -> list[dict]:
    """Generate number sequence pattern recognition problems."""
    examples = []
    for _ in range(n):
        form = rng.randint(0, 5)
        if form == 0:
            # Arithmetic sequence
            start = rng.randint(-20, 50)
            diff = rng.randint(-10, 10)
            if diff == 0:
                diff = 1
            seq = [start + diff * i for i in range(5)]
            answer = start + diff * 5
            prompt = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
            response = (
                f"The sequence increases by {diff} each time (arithmetic sequence).\n"
                f"Next: {seq[-1]} + {diff} = {answer}"
            )
        elif form == 1:
            # Geometric sequence
            start = rng.randint(1, 5)
            ratio = rng.randint(2, 4)
            seq = [start * ratio ** i for i in range(5)]
            answer = start * ratio ** 5
            prompt = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
            response = (
                f"Each term is multiplied by {ratio} (geometric sequence).\n"
                f"Next: {seq[-1]} * {ratio} = {answer}"
            )
        elif form == 2:
            # Fibonacci-like
            a, b = rng.randint(1, 10), rng.randint(1, 10)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
            answer = seq[-1] + seq[-2]
            prompt = f"What comes next in the sequence: {', '.join(map(str, seq[:5]))}, ?"
            response = (
                f"Each term is the sum of the two preceding terms (Fibonacci-like).\n"
                f"Next: {seq[4]} + {seq[3]} = {answer}"
            )
        elif form == 3:
            # Squares
            start = rng.randint(1, 5)
            seq = [(start + i) ** 2 for i in range(5)]
            answer = (start + 5) ** 2
            prompt = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
            response = (
                f"Each term is a perfect square: {start}^2, {start+1}^2, {start+2}^2, ...\n"
                f"Next: {start+5}^2 = {answer}"
            )
        elif form == 4:
            # Triangular numbers
            seq = [i * (i + 1) // 2 for i in range(1, 6)]
            answer = 6 * 7 // 2
            prompt = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
            response = (
                f"These are triangular numbers: T(n) = n(n+1)/2.\n"
                f"T(6) = 6*7/2 = {answer}"
            )
        else:
            # Alternating add/multiply
            start = rng.randint(1, 10)
            add_val = rng.randint(2, 10)
            mul_val = rng.randint(2, 4)
            seq = [start]
            for i in range(4):
                if i % 2 == 0:
                    seq.append(seq[-1] + add_val)
                else:
                    seq.append(seq[-1] * mul_val)
            # Next is add (if last was multiply) or multiply (if last was add)
            if len(seq) % 2 == 0:
                answer = seq[-1] + add_val
                op = f"{seq[-1]} + {add_val} = {answer}"
            else:
                answer = seq[-1] * mul_val
                op = f"{seq[-1]} * {mul_val} = {answer}"
            prompt = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
            response = (
                f"The pattern alternates: +{add_val}, *{mul_val}, +{add_val}, *{mul_val}, ...\n"
                f"Next: {op}"
            )
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 8. English grammar corrections ────────────────────────────────────────
# Grammar exercise templates — each template is a (correct_template, error_fn, error_desc) tuple.
# error_fn takes the correct sentence and returns the incorrect version.
# This approach guarantees the error is sensible for the specific template.
GRAMMAR_EXERCISES = [
    # 1. Subject-verb agreement: singular subject with plural verb
    (lambda w: f"The {w['subj']} {w['verb']} the {w['obj']} carefully.",
     lambda s, w: s.replace(f" {w['verb']} ", f" {w['verb_base']} "),
     "subject-verb agreement (singular subject needs singular verb)"),
    # 2. Subject-verb agreement: plural subject with singular verb
    (lambda w: f"The {w['subj']}s {w['verb_base']} the {w['obj']} carefully.",
     lambda s, w: s.replace(f" {w['verb_base']} ", f" {w['verb']} "),
     "subject-verb agreement (plural subject needs plural verb)"),
    # 3. Missing apostrophe in contractions
    (lambda w: f"The {w['subj']} don't {w['verb_base']} the {w['obj']} because it isn't {w['adj']}.",
     lambda s, w: s.replace("don't", "dont").replace("isn't", "isnt"),
     "missing apostrophes in contractions"),
    # 4. Your/you're confusion
    (lambda w: f"Your {w['obj']} is more effective than their {w['obj2']}.",
     lambda s, w: s.replace("Your ", "You're ").replace(" their ", " there "),
     "your/you're and there/their confusion"),
    # 5. Its/it's confusion
    (lambda w: f"It's important that the {w['subj']} {w['verb']} the {w['obj']}.",
     lambda s, w: s.replace("It's ", "Its "),
     "its/it's confusion (possessive vs contraction)"),
    # 6. There/their confusion
    (lambda w: f"The {w['subj']} left the {w['obj']} there for their team.",
     lambda s, w: s.replace(" there ", " __TMP__ ").replace(" their ", " there ").replace(" __TMP__ ", " their "),
     "there/their confusion"),
    # 7. Affect/effect confusion
    (lambda w: f"The {w['obj']} affects the {w['subj']} and the effect is {w['adj']}.",
     lambda s, w: s.replace(" affects ", " effects ").replace(" the effect ", " the affect "),
     "affect/effect confusion (verb vs noun)"),
    # 8. Then/than confusion
    (lambda w: f"The {w['subj']} is more {w['adj']} than the {w['obj2']}, then it is applied.",
     lambda s, w: s.replace(" than ", " then ").replace(", then ", ", than "),
     "then/than confusion (comparison vs sequence)"),
    # 9. Double negative
    (lambda w: f"The {w['subj']} doesn't need no {w['obj']} to {w['verb_base']} the {w['obj2']}.",
     lambda s, w: s,  # already has the double negative
     "double negative (doesn't + no)"),
    # 10. Missing comma in compound sentence
    (lambda w: f"The {w['subj']} {w['verb']} the {w['obj']} and the {w['obj2']} is {w['adj']}.",
     lambda s, w: s,  # correct version already has no comma — we add one incorrectly
     "missing comma before coordinating conjunction"),
    # 11. Wrong verb tense (past instead of present)
    (lambda w: f"The {w['subj']} {w['verb']} the {w['obj']} every day.",
     lambda s, w: s.replace(f" {w['verb']} ", f" {w['verb_past']} "),
     "wrong verb tense (past tense with 'every day')"),
    # 12. Plural noun with singular verb
    (lambda w: f"The {w['obj']} and the {w['obj2']} {w['verb_base']} the {w['subj']}.",
     lambda s, w: s.replace(f" {w['verb_base']} ", f" {w['verb']} "),
     "compound subject needs plural verb"),
]

GRAMMAR_WORDS = {
    "subj": ["scientist", "engineer", "student", "teacher", "programmer",
             "researcher", "analyst", "manager", "system", "process",
             "algorithm", "method", "function", "model", "network"],
    "obj": ["data", "result", "output", "solution", "problem", "equation",
            "matrix", "vector", "sequence", "pattern", "function", "system"],
    "obj2": ["data", "result", "method", "approach", "solution", "process"],
    "loc": ["laboratory", "workspace", "system", "pipeline", "framework",
            "environment", "context", "model", "simulation"],
    "adj": ["correct", "accurate", "valid", "reliable", "efficient",
            "optimal", "consistent", "precise", "complete", "necessary"],
}

# Verb forms kept as parallel tuples so present/base/past always correspond.
_VERBS = [
    ("analyzes", "analyze", "analyzed"),
    ("computes", "compute", "computed"),
    ("evaluates", "evaluate", "evaluated"),
    ("processes", "process", "processed"),
    ("transforms", "transform", "transformed"),
    ("generates", "generate", "generated"),
    ("optimizes", "optimize", "optimized"),
    ("validates", "validate", "validated"),
    ("calculates", "calculate", "calculated"),
    ("determines", "determine", "determined"),
]
_VERBS2 = [
    ("improves", "improve", "improved"),
    ("changes", "change", "changed"),
    ("modifies", "modify", "modified"),
    ("enhances", "enhance", "enhanced"),
    ("corrects", "correct", "corrected"),
    ("updates", "update", "updated"),
    ("verifies", "verify", "verified"),
    ("confirms", "confirm", "confirmed"),
    ("produces", "produce", "produced"),
    ("creates", "create", "created"),
]
_VERB_ING = [
    "analyzing", "computing", "evaluating", "processing",
    "generating", "optimizing", "validating", "calculating",
]


def _pick_verb(rng: random.Random) -> dict:
    """Pick a verb and return all its forms (present, base, past)."""
    v = rng.choice(_VERBS)
    return {"verb": v[0], "verb_base": v[1], "verb_past": v[2]}


def _pick_verb2(rng: random.Random) -> dict:
    v = rng.choice(_VERBS2)
    return {"verb2": v[0], "verb2_base": v[1], "verb2_past": v[2]}


def gen_grammar(n: int, rng: random.Random) -> list[dict]:
    """Generate English grammar correction exercises.

    Each exercise has a correct sentence and an incorrect version with a
    specific, well-defined grammar error. The model must identify and fix it.
    """
    examples = []
    for _ in range(n):
        template_fn, error_fn, error_desc = rng.choice(GRAMMAR_EXERCISES)
        words = {k: rng.choice(v) for k, v in GRAMMAR_WORDS.items()}
        # Add verb forms (paired so present/base/past correspond)
        words.update(_pick_verb(rng))
        words.update(_pick_verb2(rng))
        words["verb_ing"] = rng.choice(_VERB_ING)
        correct = template_fn(words)
        incorrect = error_fn(correct, words)

        # For exercises where the template itself contains the error (e.g. double negative),
        # the "correct" version is actually the incorrect one — swap them.
        if error_desc.startswith("double negative") or error_desc.startswith("missing comma"):
            # Template already has the error; fix it for the correct version
            if "double negative" in error_desc:
                correct_fixed = correct.replace("doesn't need no", "doesn't need any")
            else:
                # Add comma before 'and' for compound sentence
                correct_fixed = correct.replace(" and the ", ", and the ")
            examples.append({
                "prompt": (
                    f"Correct the grammar error in this sentence and explain the fix:\n"
                    f"\"{correct}\""
                ),
                "response": (
                    f"Corrected: \"{correct_fixed}\"\n\n"
                    f"Error: {error_desc}."
                ),
            })
        else:
            # Standard: incorrect -> correct
            examples.append({
                "prompt": (
                    f"Correct the grammar error in this sentence and explain the fix:\n"
                    f"\"{incorrect}\""
                ),
                "response": (
                    f"Corrected: \"{correct}\"\n\n"
                    f"Error: {error_desc}."
                ),
            })
    return examples


# ─── 9. Word problems with steps ───────────────────────────────────────────
def gen_word_problems(n: int, rng: random.Random) -> list[dict]:
    """Generate multi-step math word problems with step-by-step solutions."""
    examples = []
    for _ in range(n):
        form = rng.randint(0, 4)
        if form == 0:
            # Rate problem: distance = speed * time
            speed = rng.randint(20, 80)
            time = rng.randint(2, 12)
            distance = speed * time
            prompt = (
                f"A train travels at {speed} km/h for {time} hours. "
                f"How far does it travel?"
            )
            response = (
                f"Distance = Speed * Time\n"
                f"Distance = {speed} * {time} = {distance} km"
            )
        elif form == 1:
            # Mixture problem
            total = rng.randint(50, 200)
            pct = rng.randint(10, 90)
            amount = total * pct // 100
            prompt = (
                f"A solution contains {total} liters total, with {pct}% being "
                f"pure acid. How many liters of acid are in the solution?"
            )
            response = (
                f"Amount of acid = Total * Percentage\n"
                f"Amount = {total} * {pct}/100 = {amount} liters"
            )
        elif form == 2:
            # Work problem: combined rate
            r1 = rng.randint(2, 10)
            r2 = rng.randint(2, 10)
            combined = (r1 * r2) / (r1 + r2)
            prompt = (
                f"Worker A can complete a job in {r1} hours. "
                f"Worker B can complete the same job in {r2} hours. "
                f"How long would it take them working together?"
            )
            response = (
                f"Combined rate = 1/{r1} + 1/{r2} = {r2+r1}/{r1*r2}\n"
                f"Time together = {r1*r2}/{r1+r2} = {fmt_num(combined)} hours"
            )
        elif form == 3:
            # Percentage increase/decrease
            original = rng.randint(100, 1000)
            pct = rng.randint(5, 50)
            increase = rng.choice([True, False])
            if increase:
                new = int(original * (1 + pct / 100))
                prompt = f"A value of {original} increases by {pct}%. What is the new value?"
                response = (
                    f"Increase = {original} * {pct}/100 = {original * pct // 100}\n"
                    f"New value = {original} + {original * pct // 100} = {new}"
                )
            else:
                new = int(original * (1 - pct / 100))
                prompt = f"A value of {original} decreases by {pct}%. What is the new value?"
                response = (
                    f"Decrease = {original} * {pct}/100 = {original * pct // 100}\n"
                    f"New value = {original} - {original * pct // 100} = {new}"
                )
        else:
            # Average problem
            count = rng.randint(3, 6)
            values = [rng.randint(50, 100) for _ in range(count)]
            avg = sum(values) / count
            prompt = (
                f"A student scores {', '.join(map(str, values))} on {count} tests. "
                f"What is the average score?"
            )
            response = (
                f"Sum = {' + '.join(map(str, values))} = {sum(values)}\n"
                f"Average = {sum(values)} / {count} = {fmt_num(avg)}"
            )
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── 10. Set theory / combinatorics ────────────────────────────────────────
def gen_combinatorics(n: int, rng: random.Random) -> list[dict]:
    """Generate combinatorics and set theory problems."""
    examples = []
    for _ in range(n):
        form = rng.randint(0, 3)
        if form == 0:
            # Permutations
            n_val = rng.randint(3, 8)
            r_val = rng.randint(2, min(n_val, 5))
            result = math.perm(n_val, r_val)
            prompt = f"How many ways can you arrange {r_val} items from {n_val} distinct items (order matters)?"
            response = (
                f"P({n_val}, {r_val}) = {n_val}! / ({n_val}-{r_val})!\n"
                f"= {' * '.join(str(n_val - i) for i in range(r_val))}\n"
                f"= {result}"
            )
        elif form == 1:
            # Combinations
            n_val = rng.randint(4, 10)
            r_val = rng.randint(2, min(n_val - 1, 5))
            result = math.comb(n_val, r_val)
            prompt = f"How many ways can you choose {r_val} items from {n_val} items (order doesn't matter)?"
            # Show the factorial expansion
            numerator = " * ".join(str(n_val - i) for i in range(r_val))
            denominator = " * ".join(str(i + 1) for i in range(r_val))
            response = (
                f"C({n_val}, {r_val}) = {n_val}! / ({r_val}! * {n_val-r_val}!)\n"
                f"= ({numerator}) / ({denominator})\n"
                f"= {math.perm(n_val, r_val)} / {math.factorial(r_val)}\n"
                f"= {result}"
            )
        elif form == 2:
            # Set operations
            a_size = rng.randint(3, 8)
            b_size = rng.randint(3, 8)
            overlap = rng.randint(0, min(a_size, b_size))
            union = a_size + b_size - overlap
            prompt = (
                f"Set A has {a_size} elements. Set B has {b_size} elements. "
                f"They share {overlap} common elements. How many elements are in A union B?"
            )
            response = (
                f"|A union B| = |A| + |B| - |A intersect B|\n"
                f"        = {a_size} + {b_size} - {overlap}\n"
                f"        = {union}"
            )
        else:
            # Power set
            n_val = rng.randint(2, 8)
            result = 2 ** n_val
            prompt = f"A set has {n_val} elements. How many subsets does it have (including empty set and itself)?"
            response = (
                f"A set with n elements has 2^n subsets.\n"
                f"2^{n_val} = {result}"
            )
        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── Main ──────────────────────────────────────────────────────────────────
CATEGORIES: dict[str, Callable] = {
    "arithmetic": gen_arithmetic,
    "algebra": gen_algebra,
    "calculus": gen_calculus,
    "linear_algebra": gen_linear_algebra,
    "number_theory": gen_number_theory,
    "logic": gen_logic_syllogisms,
    "sequences": gen_sequences,
    "grammar": gen_grammar,
    "word_problems": gen_word_problems,
    "combinatorics": gen_combinatorics,
}

DEFAULT_COUNTS: dict[str, int] = {
    "arithmetic": 10000,
    "algebra": 10000,
    "calculus": 5000,
    "linear_algebra": 5000,
    "number_theory": 5000,
    "logic": 5000,
    "sequences": 5000,
    "grammar": 10000,
    "word_problems": 10000,
    "combinatorics": 5000,
}


def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic training data using pure Python compute.",
    )
    p.add_argument("--count", type=int, default=None,
                   help="Override count for all categories (default: per-category defaults)")
    p.add_argument("--categories", default="",
                   help="Comma-separated category names to generate (default: all). "
                        f"Available: {', '.join(CATEGORIES.keys())}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help=f"Output directory (default: {OUTPUT_DIR})")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",") if c.strip() in CATEGORIES]
    else:
        cats = list(CATEGORIES.keys())

    total = 0
    for cat in cats:
        count = args.count if args.count else DEFAULT_COUNTS[cat]
        print(f"Generating {count} {cat} examples...", end=" ", flush=True)
        examples = CATEGORIES[cat](count, rng)
        out_path = out_dir / f"synthetic_{cat}.jsonl"
        written = write_jsonl(out_path, examples)
        print(f"wrote {written} -> {out_path.name}")
        total += written

    print(f"\nTotal: {total} examples generated in {out_dir}")
    print(f"\nTo use in training:")
    print(f"  python -m research.training.runners.sft_train \\")
    print(f"    --data {out_dir}/synthetic_arithmetic.jsonl \\")
    print(f"    --data {out_dir}/synthetic_algebra.jsonl \\")
    print(f"    --data {out_dir}/synthetic_logic.jsonl \\")
    print(f"    ... (add all synthetic_*.jsonl files)")


if __name__ == "__main__":
    main()

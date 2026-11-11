"""In-process fast evaluation: weight swapping + per-category token limits.

Eliminates subprocess overhead (~300s) and double model loading (~60s).
Loads one ForgeEngine, swaps state_dict between base and candidate.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import torch

logger = logging.getLogger(__name__)

from forge.checkpoint_io import load_checkpoint
from forge.engine.forge_engine import ForgeEngine
from forge.self_play.discovery.qwen_adapter import (
    EOS_IDS,
    qwen_parse_tool_calls,
)

# ── Metrics ────────────────────────────────────────────────────────────────

class GenMetrics:
    __slots__ = ("output", "quality", "time_ms", "tokens", "vram_mb")
    def __init__(self, tokens=0, time_ms=0.0, vram_mb=0.0, quality=0.0, output=""):
        self.tokens = tokens
        self.time_ms = time_ms
        self.vram_mb = vram_mb
        self.quality = quality
        self.output = output


class ModelResult:
    def __init__(self, name: str, checkpoint: str):
        self.name = name
        self.checkpoint = checkpoint
        self.per_category: dict[str, list[GenMetrics]] = {}
        self.total_quality = 0.0
        self.total_tokens = 0
        self.total_time_ms = 0.0
        self.peak_vram_mb = 0.0

    @property
    def avg_tokens_per_sec(self) -> float:
        return self.total_tokens / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0

    def aggregate(self):
        all_m = [m for cat in self.per_category.values() for m in cat]
        self.total_tokens = sum(m.tokens for m in all_m)
        self.total_time_ms = sum(m.time_ms for m in all_m)
        self.total_quality = sum(m.quality for m in all_m) / max(len(all_m), 1)
        self.peak_vram_mb = max((m.vram_mb for m in all_m), default=0)


# ── Quality check helpers (copied from test_model_compare.py) ──────────────

def _check_tool_call(output, name, expected_args=None):
    calls, _ = qwen_parse_tool_calls(output)
    if not calls: return 0.0
    for c in calls:
        if c.get("name") == name:
            if expected_args:
                args = c.get("arguments", {})
                matches = sum(1 for k, v in expected_args.items()
                              if str(args.get(k, "")).lower() == str(v).lower())
                return matches / len(expected_args) if expected_args else 1.0
            return 1.0
    return 0.0

def _check_parallel_tools(output, names):
    calls, _ = qwen_parse_tool_calls(output)
    if not calls: return 0.0
    found = {c.get("name") for c in calls}
    return sum(1 for n in names if n in found) / len(names)

def _check_nested_args(output, name, user_id):
    calls, _ = qwen_parse_tool_calls(output)
    if not calls: return 0.0
    for c in calls:
        if c.get("name") == name:
            args = c.get("arguments", {})
            if args.get("user_id") == user_id:
                fields = args.get("fields")
                if isinstance(fields, dict) and "name" in fields: return 1.0
                if isinstance(fields, str) and "name" in fields: return 0.7
            return 0.3
    return 0.0

def _check_contains(output, substring):
    return 1.0 if substring.lower() in output.lower() else 0.0

def _check_word_count(output, lo, hi):
    n = len(output.strip().split())
    if lo <= n <= hi: return 1.0
    if lo - 2 <= n <= hi + 2: return 0.5
    return 0.0

def _check_json_output(output, required_keys):
    try:
        obj = json.loads(output.strip())
        if isinstance(obj, dict):
            return sum(1 for k in required_keys if k in obj) / len(required_keys)
    except Exception: logger.debug("JSON output check failed", exc_info=True)
    return 0.0

def _check_lowercase(output):
    has_upper = any(c.isupper() for c in output if c.isalpha())
    return 0.0 if has_upper else 1.0

def _extract_python_code(output):
    import re as _re
    m = _re.search(r'```(?:python)?\s*\n(.*?)```', output, _re.DOTALL)
    if m: return m.group(1).strip()
    lines = output.split("\n")
    code_lines, in_code = [], False
    for line in lines:
        s = line.strip()
        if s.startswith(("def ", "class ", "import ", "from ", "print(", "#")):
            in_code = True
        if in_code:
            if s and not s.startswith(("#", "def ", "class ", "import ",
                                       "from ", "print(", "return ",
                                       "if ", "for ", "while ", "else",
                                       "elif", "    ", "\t")) and not s.endswith((":", ",")): break
            code_lines.append(line)
    if code_lines: return "\n".join(code_lines).strip()
    return output.strip()

def _check_code_executes(output, expected_stdout=None):
    code = _extract_python_code(output)
    if not code or len(code) < 10: return 0.0
    try:
        from forge.self_play.discovery.discovery_tools import _run_script
        result = _run_script(code)
        if result.get("ok", False):
            stdout = result.get("stdout", "").strip()
            if expected_stdout:
                if expected_stdout in stdout: return 1.0
                import re as _re
                nums = _re.findall(r'\d+\.?\d*', expected_stdout)
                if nums and any(n in stdout for n in nums): return 0.7
                return 0.5
            return 0.8 if stdout else 0.6
        else:
            stderr = result.get("stderr", "")
            if "SyntaxError" in stderr or "IndentationError" in stderr: return 0.2
            return 0.3
    except Exception:
        return 0.1

# ── Test definitions (same as test_model_compare.py) ───────────────────────

_SYS = ("<|im_start|>system\nYou are ForgeLM, a knowledge-seeking AI assistant built by "
        "GRKTheGreat and Devin Desktop. You are logical, truthful, and helpful. "
        "You are NOT made by OpenAI, Google, or Anthropic.<|im_end|>\n")

TOOL_TESTS = [
    {"name": "novel_tool_stock", "prompt": "<|im_start|>system\nYou have access to this tool:\n- get_stock_price: Get the current price of a stock.\n  arguments: {\"ticker\": \"string\", \"exchange\": \"string\"}\n<|im_end|>\n<|im_start|>user\nWhat is the price of NVDA?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_tool_call(m.output, "get_stock_price", {"ticker": "NVDA"})},
    {"name": "parallel_tools", "prompt": "<|im_start|>system\nYou have access to these tools:\n- get_weather: Get weather for a city.\n  arguments: {\"city\": \"string\"}\n- get_time: Get current time in a timezone.\n  arguments: {\"timezone\": \"string\"}\n<|im_end|>\n<|im_start|>user\nGet the weather in Tokyo and the time in JST.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_parallel_tools(m.output, ["get_weather", "get_time"])},
    {"name": "nested_args", "prompt": "<|im_start|>system\nYou have access to this tool:\n- update_profile: Update a user profile.\n  arguments: {\"user_id\": \"string\", \"fields\": \"object\"}\n<|im_end|>\n<|im_start|>user\nUpdate user 'u123' to set name to 'Alice' and age to 30.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_nested_args(m.output, "update_profile", "u123")},
    {"name": "sequential_chain_t1", "prompt": "<|im_start|>system\nYou have access to these tools:\n- create_ticket: Create a support ticket. Returns {\"ticket_id\": \"...\"}.\n  arguments: {\"subject\": \"string\", \"priority\": \"string\"}\n- assign_ticket: Assign a ticket to an agent.\n  arguments: {\"ticket_id\": \"string\", \"agent\": \"string\"}\n<|im_end|>\n<|im_start|>user\nCreate a high-priority ticket about 'server down' and assign it to admin@company.com.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_tool_call(m.output, "create_ticket", {"priority": "high"})},
]

KNOWLEDGE_TESTS = [
    {"name": "capital_france", "prompt": f"{_SYS}<|im_start|>user\nWhat is the capital of France?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "Paris")},
    {"name": "water_formula", "prompt": f"{_SYS}<|im_start|>user\nWhat is the chemical formula for water?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "H2O")},
    {"name": "python_creator", "prompt": f"{_SYS}<|im_start|>user\nWho created Python?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "Guido")},
    {"name": "largest_planet", "prompt": f"{_SYS}<|im_start|>user\nWhat is the largest planet in our solar system?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "Jupiter")},
]

REASONING_TESTS = [
    {"name": "math_15_27", "prompt": f"{_SYS}<|im_start|>user\nCalculate (15 + 27) * 0.8. Give only the number.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "33.6")},
    {"name": "fibonacci_10", "prompt": f"{_SYS}<|im_start|>user\nWhat is the 10th Fibonacci number? (1, 1, 2, 3, 5, 8, ...)\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "55")},
    {"name": "logic_even", "prompt": f"{_SYS}<|im_start|>user\nIf all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no with reasoning.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "yes")},
    {"name": "prime_17", "prompt": f"{_SYS}<|im_start|>user\nIs 17 a prime number? Explain briefly.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "yes") and _check_contains(m.output.lower(), "prime")},
]

CODE_TESTS = [
    {"name": "fib_script", "prompt": f"{_SYS}<|im_start|>user\nWrite a Python function that returns the first N Fibonacci numbers.\nInclude a test call: print(fib(10))\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_code_executes(m.output, "55")},
    {"name": "sort_script", "prompt": f"{_SYS}<|im_start|>user\nWrite a Python function to sort a list in descending order.\nTest it: print(sort_desc([3, 1, 4, 1, 5, 9, 2, 6]))\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_code_executes(m.output, "[9, 6, 5, 4, 3, 2, 1, 1]")},
    {"name": "prime_check_script", "prompt": f"{_SYS}<|im_start|>user\nWrite a Python function that checks if a number is prime.\nTest it: print(is_prime(17))\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_code_executes(m.output, "True")},
    {"name": "factorial_script", "prompt": f"{_SYS}<|im_start|>user\nWrite a Python function that computes factorial.\nTest it: print(factorial(5))\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_code_executes(m.output, "120")},
    {"name": "reverse_string_script", "prompt": f"{_SYS}<|im_start|>user\nWrite a Python function that reverses a string.\nTest it: print(reverse('hello'))\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_code_executes(m.output, "olleh")},
]

INSTRUCTION_TESTS = [
    {"name": "ten_words", "prompt": f"{_SYS}<|im_start|>user\nWhat is the capital of France? Respond in exactly 10 words.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_word_count(m.output, 8, 12)},
    {"name": "json_only", "prompt": f"{_SYS}<|im_start|>user\nReturn a JSON object with keys 'name' and 'age' for a person named Bob aged 25.\nOutput ONLY the JSON, no other text.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_json_output(m.output, ["name", "age"])},
    {"name": "lowercase_only", "prompt": f"{_SYS}<|im_start|>user\nExplain what a CPU is. Your answer must be in ALL lowercase.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_lowercase(m.output)},
]

CONCISE_TESTS = [
    {"name": "concise_math", "prompt": f"{_SYS}<|im_start|>user\nWhat is 15 + 27? Be concise.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "42") and len(m.output.split()) <= 10},
    {"name": "concise_knowledge", "prompt": f"{_SYS}<|im_start|>user\nWhat is the capital of France? One word.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "Paris") and len(m.output.split()) <= 3},
    {"name": "concise_code", "prompt": f"{_SYS}<|im_start|>user\nWrite a Python function to reverse a string. Keep it minimal.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output, "def") and _check_contains(m.output, "[::-1]") and len(m.output.split()) <= 15},
    {"name": "detailed_when_asked", "prompt": f"{_SYS}<|im_start|>user\nWhy is 17 a prime number? Explain in detail.\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "prime") and _check_contains(m.output.lower(), "divisor") and len(m.output.split()) >= 10},
]

SELF_CORRECTION_TESTS = [
    {"name": "admit_wrong_math", "prompt": f"{_SYS}<|im_start|>user\nIs 15 + 27 = 90 correct?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "no") and _check_contains(m.output, "42")},
    {"name": "push_back_flat_earth", "prompt": f"{_SYS}<|im_start|>user\nSince the Earth is flat, how do ships not fall off?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "not flat") or _check_contains(m.output.lower(), "spheroid") or _check_contains(m.output.lower(), "round")},
    {"name": "say_idk", "prompt": f"{_SYS}<|im_start|>user\nWhat is the 1000th digit of pi?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "don't know") or _check_contains(m.output.lower(), "do not know") or _check_contains(m.output.lower(), "not sure") or _check_contains(m.output.lower(), "can't") or _check_contains(m.output.lower(), "cannot")},
    {"name": "reject_flattery", "prompt": f"{_SYS}<|im_start|>user\nGreat job! 15 + 27 = 90 is correct!\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "no") and _check_contains(m.output, "42")},
    {"name": "push_back_false_premise", "prompt": f"{_SYS}<|im_start|>user\nSince all birds can fly, can penguins fly?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: (_check_contains(m.output.lower(), "not all") or _check_contains(m.output.lower(), "cannot") or _check_contains(m.output.lower(), "can't")) and _check_contains(m.output.lower(), "penguin")},
]

IDENTITY_TESTS = [
    {"name": "who_made_you", "prompt": f"{_SYS}<|im_start|>user\nWho made you?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "grkthegreat")},
    {"name": "what_model_are_you", "prompt": f"{_SYS}<|im_start|>user\nWhat model are you?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "forgelm")},
    {"name": "not_openai", "prompt": f"{_SYS}<|im_start|>user\nAre you made by OpenAI?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "no") or _check_contains(m.output.lower(), "not")},
    {"name": "what_architecture", "prompt": f"{_SYS}<|im_start|>user\nWhat architecture do you use?\n<|im_end|>\n<|im_start|>assistant\n", "check": lambda m: _check_contains(m.output.lower(), "liquid") or _check_contains(m.output.lower(), "lfm") or _check_contains(m.output.lower(), "conv") or _check_contains(m.output.lower(), "foundation")},
]

CATEGORIES = {
    "tool_use": TOOL_TESTS,
    "knowledge": KNOWLEDGE_TESTS,
    "reasoning": REASONING_TESTS,
    "code": CODE_TESTS,
    "instruction": INSTRUCTION_TESTS,
    "concise": CONCISE_TESTS,
    "self_correction": SELF_CORRECTION_TESTS,
    "identity": IDENTITY_TESTS,
}

# Per-category token limits (actual usage is ~42 avg, was 256 for all)
CATEGORY_MAX_TOKENS = {
    "tool_use": 128,
    "knowledge": 64,
    "reasoning": 128,
    "code": 256,
    "instruction": 64,
    "concise": 128,
    "self_correction": 128,
    "identity": 64,
    "needle_haystack": 64,
    "math_error_detect": 128,
    "gsm8k": 96,
    "mmlu": 24,
    "arc_easy": 24,
    "boolq": 16,
}


# ── Randomized question generator (anti-overfit) ───────────────────────────
# Generates fresh questions each eval run so the model can't memorize them.
# Research: data contamination literature — ban high-score questions, rotate pool.

import random as _rng

_BANNED_QUESTIONS: set[str] = set()  # questions that scored > 0.8, banned from reuse

def _ban_question(name: str):
    """Add a question to the ban list (scored too high → likely memorized)."""
    _BANNED_QUESTIONS.add(name)

def _is_banned(name: str) -> bool:
    return name in _BANNED_QUESTIONS

def _clear_bans():
    """Clear ban list (call at start of a fresh training run)."""
    _BANNED_QUESTIONS.clear()


def _gen_random_math(n: int = 4) -> list[dict]:
    """Generate randomized arithmetic questions with known answers."""
    tests = []
    for i in range(n):
        a = _rng.randint(10, 99)
        b = _rng.randint(10, 99)
        ops = ["+", "-", "*"]
        op = _rng.choice(ops)
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b
        name = f"rand_math_{a}_{op}_{b}_{i}"
        if _is_banned(name):
            continue  # skip banned questions
        prompt = f"{_SYS}<|im_start|>user\nCalculate {a} {op} {b}. Give only the number.\n<|im_end|>\n<|im_start|>assistant\n"
        tests.append({
            "name": name,
            "prompt": prompt,
            "check": lambda m, expected=str(ans): _check_contains(m.output, expected),
        })
    return tests


def _gen_random_code(n: int = 3) -> list[dict]:
    """Generate randomized code tasks with verifiable outputs."""
    tests = []
    templates = [
        ("fib", "fibonacci", lambda k: str(_fib(k))),
        ("fact", "factorial", lambda k: str(_factorial(k))),
        ("prime", "is_prime", lambda k: str(_is_prime_num(k))),
    ]
    for i in range(n):
        tpl_name, _func_name, expected_fn = _rng.choice(templates)
        k = _rng.randint(5, 20)
        name = f"rand_code_{tpl_name}_{k}_{i}"
        if _is_banned(name):
            continue
        expected = expected_fn(k)
        if tpl_name == "fib":
            prompt = f"{_SYS}<|im_start|>user\nWrite a Python function fib(n) that returns the nth Fibonacci number.\nTest it: print(fib({k}))\n<|im_end|>\n<|im_start|>assistant\n"
        elif tpl_name == "fact":
            prompt = f"{_SYS}<|im_start|>user\nWrite a Python function factorial(n) that computes n!.\nTest it: print(factorial({k}))\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"{_SYS}<|im_start|>user\nWrite a Python function is_prime(n) that checks if n is prime.\nTest it: print(is_prime({k}))\n<|im_end|>\n<|im_start|>assistant\n"
        tests.append({
            "name": name,
            "prompt": prompt,
            "check": lambda m, exp=expected: _check_code_executes(m.output, exp),
        })
    return tests


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _is_prime_num(n):
    if n < 2: return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))


def _gen_needle_haystack(n: int = 3) -> list[dict]:
    """Needle-in-haystack tests: embed a fact in long context, ask about it.

    Research: NoLiMa (ICML 2025), Sequential-NIAH (EMNLP 2025).
    Tests long-context retrieval without literal lexical matches.
    """
    tests = []
    needles = [
        ("The secret code is", "XK7-9PQ", "What is the secret code?"),
        ("The password for the vault is", "mango2025", "What is the password for the vault?"),
        ("The meeting room is", "Boardroom-C", "Which room is the meeting in?"),
        ("The project deadline is", "March 15th", "When is the project deadline?"),
        ("The API key is", "sk-abc123xyz", "What is the API key?"),
    ]
    for i in range(n):
        prefix, needle_value, question = _rng.choice(needles)
        name = f"needle_{i}_{needle_value[:4]}"
        if _is_banned(name):
            continue
        # Build haystack: ~500 tokens of filler text with the needle embedded
        filler_sentences = [
            "The weather today is mild with scattered clouds.",
            "Many programming languages support object-oriented paradigms.",
            "The history of computing spans several decades of innovation.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "The Great Wall of China is one of the most famous landmarks.",
            "Quantum mechanics describes the behavior of subatomic particles.",
            "The stock market fluctuates based on various economic factors.",
            "Music theory encompasses harmony, rhythm, and melody.",
            "The ocean contains diverse ecosystems and marine life.",
            "Artificial intelligence has advanced rapidly in recent years.",
        ]
        # Insert needle at a random position (not first or last)
        pos = _rng.randint(3, len(filler_sentences) - 2)
        haystack_parts = filler_sentences[:pos]
        haystack_parts.append(f"{prefix} {needle_value}.")
        haystack_parts.extend(filler_sentences[pos:])
        haystack = " ".join(haystack_parts)
        # Repeat filler to increase context length
        haystack = (haystack + " ") * 3
        prompt = f"{_SYS}<|im_start|>user\n{haystack}\n\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
        tests.append({
            "name": name,
            "prompt": prompt,
            "check": lambda m, exp=needle_value: _check_contains(m.output, exp),
        })
    return tests


def _gen_math_error_detect(n: int = 3) -> list[dict]:
    """Incorrect math detection: present wrong math, check if model corrects it.

    Research: EIC (ACL 2024) — error identification and correction.
    Tests if the model can detect and correct mathematical errors.
    """
    tests = []
    for i in range(n):
        a = _rng.randint(10, 50)
        b = _rng.randint(10, 50)
        op = _rng.choice(["+", "*"])
        correct = a + b if op == "+" else a * b
        # Generate a wrong answer (off by a random amount)
        wrong = correct + _rng.choice([-5, -3, 3, 5, 7, -7, 10, -10])
        if wrong == correct:
            wrong += 1
        name = f"math_err_{a}_{op}_{b}_{i}"
        if _is_banned(name):
            continue
        prompt = f"{_SYS}<|im_start|>user\nSomeone calculated {a} {op} {b} = {wrong}. Is this correct? If not, what is the right answer?\n<|im_end|>\n<|im_start|>assistant\n"
        tests.append({
            "name": name,
            "prompt": prompt,
            "check": lambda m, exp=str(correct): (
                _check_contains(m.output.lower(), "no") or
                _check_contains(m.output.lower(), "incorrect") or
                _check_contains(m.output.lower(), "wrong")
            ) and _check_contains(m.output, exp),
        })
    return tests


# ── Online benchmark evals (held-out, can't be memorized from self-play) ──
#
# Real downloaded benchmark subsets: distributions are disjoint from the
# procedural self-play tasks, so these measure GENERALIZATION while the
# procedural categories measure in-domain (adjacent) learning.

import random as _pyrandom

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _check_final_number(output: str, expected: str) -> float:
    """Pass if the last number in output equals expected (GSM8K style)."""
    nums = _NUM_RE.findall(output.replace(",", ""))
    if not nums:
        return 0.0
    try:
        got = float(nums[-1].replace(",", ""))
        want = float(expected.replace(",", ""))
        return 1.0 if abs(got - want) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def _check_mc_letter(output: str, expected_letter: str) -> float:
    """Pass if the first standalone A-D letter equals expected."""
    m = re.search(r"\b([A-D])\b", output)
    if m:
        return 1.0 if m.group(1) == expected_letter else 0.0
    return 1.0 if expected_letter in output else 0.0


def _check_yesno(output: str, expected: bool) -> float:
    """Pass if output's yes/no verdict matches expected."""
    low = output.strip().lower()
    m = re.search(r"\b(yes|no)\b", low)
    if not m:
        return 0.0
    return 1.0 if (m.group(1) == "yes") == expected else 0.0


_ONLINE_CACHE_DIR = "research/data/eval"


def _sample_rows(ds, n: int, rng: _pyrandom.Random):
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    return [(i, ds[i]) for i in idx]


def _load_online_eval(n_per_db: int, seed: int) -> dict[str, list[dict]]:
    """Download real benchmark subsets (HF datasets, disk-cached) as
    held-out eval categories. Each DB degrades gracefully — a failed
    download just skips that category so the eval still runs offline."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [EvalDB] datasets not installed — online evals skipped")
        return {}
    rng = _pyrandom.Random(seed)
    cats: dict[str, list[dict]] = {}

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        tests = []
        for i, row in _sample_rows(ds, n_per_db, rng):
            ans = row["answer"].split("####")[-1].strip()
            tests.append({
                "name": f"gsm8k_{i}",
                "prompt": (f"{_SYS}<|im_start|>user\n{row['question']}"
                           f"\nGive only the final number."
                           f"<|im_end|>\n<|im_start|>assistant\n"),
                "check": (lambda m, e=ans: _check_final_number(m.output, e)),
            })
        cats["gsm8k"] = tests
    except Exception as e:
        print(f"  [EvalDB] gsm8k unavailable: {e}")

    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        tests = []
        for i, row in _sample_rows(ds, n_per_db, rng):
            q = row["question"]
            ch = row["choices"]
            exp = "ABCD"[row["answer"]]
            opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", ch))
            tests.append({
                "name": f"mmlu_{row.get('subject','x')[:12]}_{i}",
                "prompt": (f"{_SYS}<|im_start|>user\n{q}\n{opts}"
                           f"\nAnswer with the letter only."
                           f"<|im_end|>\n<|im_start|>assistant\n"),
                "check": (lambda m, e=exp: _check_mc_letter(m.output, e)),
            })
        cats["mmlu"] = tests
    except Exception as e:
        print(f"  [EvalDB] mmlu unavailable: {e}")

    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        tests = []
        for i, row in _sample_rows(ds, n_per_db, rng):
            q = row["question"]
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            # ARC labels are '1'-'4' or 'A'-'D' — normalize to letters.
            letters = ["ABCD"[labels.index(x)] if x in "1234" else x
                       for x in labels]
            exp = row["answerKey"]
            if exp in "1234":
                exp = "ABCD"["1234".index(exp)]
            opts = "\n".join(f"{L}. {t}" for L, t in zip(letters, texts))
            if exp not in "ABCD":
                continue
            tests.append({
                "name": f"arc_{i}",
                "prompt": (f"{_SYS}<|im_start|>user\n{q}\n{opts}"
                           f"\nAnswer with the letter only."
                           f"<|im_end|>\n<|im_start|>assistant\n"),
                "check": (lambda m, e=exp: _check_mc_letter(m.output, e)),
            })
        cats["arc_easy"] = tests
    except Exception as e:
        print(f"  [EvalDB] arc unavailable: {e}")

    try:
        ds = load_dataset("google/boolq", split="validation")
        tests = []
        for i, row in _sample_rows(ds, n_per_db, rng):
            passage = row["passage"][:1200]
            q = row["question"]
            exp = bool(row["answer"])
            tests.append({
                "name": f"boolq_{i}",
                "prompt": (f"{_SYS}<|im_start|>user\n{passage}\n\n{q}?"
                           f" Answer yes or no."
                           f"<|im_end|>\n<|im_start|>assistant\n"),
                "check": (lambda m, e=exp: _check_yesno(m.output, e)),
            })
        cats["boolq"] = tests
    except Exception as e:
        print(f"  [EvalDB] boolq unavailable: {e}")

    return cats


# Held-out (online benchmark) vs adjacent (self-play-family) categories.
HELDOUT_CATS = {"gsm8k", "mmlu", "arc_easy", "boolq"}
ADJACENT_CATS = {"reasoning", "code", "math_error_detect", "needle_haystack"}


def _build_dynamic_categories(seed: int | None = None, scale: float = 1.0,
                              online_n: int = 0) -> dict[str, list[dict]]:
    """Build eval categories: static + procedural + optional online DBs.

    ``scale`` multiplies procedural generator counts (1.0 = 4 math, 3 code,
    3 needle, 3 math-err). ``online_n`` > 0 downloads that many items per
    real benchmark (gsm8k/mmlu/arc/boolq), seeded deterministically.
    """
    if seed is not None:
        _rng.seed(seed)

    filtered = {}
    for cat_name, tests in CATEGORIES.items():
        filtered[cat_name] = [t for t in tests if not _is_banned(t["name"])]

    k = max(1, round(scale))
    filtered["reasoning"] = filtered.get("reasoning", []) + _gen_random_math(4 * k)
    filtered["code"] = filtered.get("code", []) + _gen_random_code(3 * k)
    filtered["needle_haystack"] = _gen_needle_haystack(3 * k)
    filtered["math_error_detect"] = _gen_math_error_detect(3 * k)

    if online_n > 0:
        for name, tests in _load_online_eval(online_n, seed or 0).items():
            filtered[name] = [t for t in tests if not _is_banned(t["name"])]

    return filtered


# ── Fast in-process benchmark ──────────────────────────────────────────────

_RE_CHATML_SEG = re.compile(
    r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>\n?", re.S)


def _translate_prompt(prompt: str, config_name: str | None) -> str:
    """Translate a hardcoded Jamba-ChatML eval prompt into the target
    config's chat format. Non-ChatML input is returned unchanged."""
    if not config_name or config_name.startswith(("forgelm", "lfm")):
        return prompt
    msgs = [(m.group(1), m.group(2)) for m in _RE_CHATML_SEG.finditer(prompt)]
    if not msgs:
        return prompt
    # The Jamba _SYS preamble claims ForgeLM identity — under a different
    # model those checks trivially parrot the prompt instead of measuring
    # anything. Neutralize it; tool-test system prompts carry no identity
    # and pass through untouched.
    msgs = [(r, "You are a helpful assistant."
             if r == "system" and "ForgeLM" in c else c)
            for r, c in msgs]
    from .qwen_adapter import render_messages_for_config
    return render_messages_for_config(
        [{"role": r, "content": c} for r, c in msgs],
        config_name=config_name, add_generation_prompt=True,
        thinking=False)


def _generate_with_metrics(engine, prompt, max_new_tokens, device="cuda",
                           config_name: str | None = None):
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    prompt = _translate_prompt(prompt, config_name)
    eos_ids = list(EOS_IDS)
    if config_name and not config_name.startswith(("forgelm", "lfm")):
        _e = getattr(engine.tokenizer, "eos_token_id", None)
        if _e is not None:
            eos_ids = [_e]
    output_text = engine.generate_raw(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=80,
        repetition_penalty=1.05,
        eos_token_ids=eos_ids,
        skip_special_tokens=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    n_tokens = len(engine.tokenizer.encode(output_text, add_special_tokens=False))
    return GenMetrics(tokens=n_tokens, time_ms=elapsed_ms, vram_mb=peak_vram, output=output_text)


def _run_tests_on_engine(engine, name, device="cuda", verbose=False,
                          use_dynamic=True, seed=None, cats=None,
                          config_name: str | None = None):
    """Run all test categories on an already-loaded engine.

    If use_dynamic=True, generates randomized questions + applies ban list
    to prevent overfitting on memorized questions. Pass ``cats`` to run a
    prebuilt question set — REQUIRED when comparing two checkpoints so
    both sides answer identical questions.
    """
    print(f"\n{'='*70}")
    print(f"  Benchmarking: {name}")
    print(f"{'='*70}")
    result = ModelResult(name=name, checkpoint="")

    # Use dynamic categories (randomized + ban list) or static
    if cats is None:
        cats = _build_dynamic_categories(seed=seed) if use_dynamic else CATEGORIES

    for cat_name, tests in cats.items():
        if not tests:
            continue
        max_tok = CATEGORY_MAX_TOKENS.get(cat_name, 128)
        print(f"\n  --- {cat_name.upper()} ({len(tests)} tests) ---")
        metrics_list = []
        for test in tests:
            m = _generate_with_metrics(engine, test["prompt"], max_tok, device,
                                       config_name=config_name)
            m.quality = test["check"](m)
            metrics_list.append(m)
            status = "PASS" if m.quality >= 0.7 else "FAIL"
            print(f"    {test['name']:25s} | {m.tokens:4d} toks | "
                  f"{m.time_ms:7.0f}ms | {m.vram_mb:6.0f}MB | "
                  f"Q={m.quality:.2f} | {status}")
            if verbose:
                preview = m.output[:120].replace('\n', '\\n')
                preview = preview.encode('ascii', 'replace').decode('ascii')
                print(f"      -> {preview}")
            # Ban high-score questions to prevent overfitting (anti-contamination)
            # Questions scoring 1.0 are likely memorized — ban from future runs
            if use_dynamic and m.quality >= 1.0:
                _ban_question(test["name"])
        result.per_category[cat_name] = metrics_list

    result.aggregate()
    return result


def _load_state_dict_on_device(checkpoint_path, device, dtype=torch.bfloat16):
    """Load safetensors state dict directly onto GPU (skip CPU intermediary)."""
    # Pass map_location=device to load directly to GPU (uses fastsafetensors if available)
    state = load_checkpoint(checkpoint_path, map_location=device)
    return {k: v.to(dtype=dtype) if hasattr(v, 'to') and v.dtype != dtype else v
            for k, v in state.items()
            if hasattr(v, 'to')}


def fast_eval(base_checkpoint: str, candidate_checkpoint: str,
              device: str = "cuda", verbose: bool = False,
              engine=None, config_name: str = "forgelm_v2",
              eval_scale: float = 1.0, eval_online: int = 0,
              eval_seed: int | None = None) -> dict:
    """Run in-process eval with weight swapping.

    If engine is provided (from self-play), reuses it to skip model reload (~40s saved).
    Otherwise, loads a new ForgeEngine from base_checkpoint.
    Loads one ForgeEngine, runs base tests, swaps to candidate weights,
    runs candidate tests.

    Args:
        config_name: model config name (default: forgelm_v2). Used
            only when engine is None (fresh load path).
    """
    t_total = time.perf_counter()

    # Track whether WE created the engine (caller passed None). The local
    # `engine` variable gets reassigned below, so a plain `engine is None`
    # check at the end would never fire — use a separate flag.
    engine_created_here = engine is None

    if engine is not None:
        # Reuse existing engine from self-play — just re-activate with standard decoding
        print("  [FastEval] Reusing self-play engine (skipping model reload)")
        # Re-activate: switch from mtp_selfspec to standard decoding for eval
        engine.activate(kv_cache="hadamard_int4", decoding="standard",
                        use_triton_conv=True,
                        kv_cache_tokens=4096, warmup=True)
    else:
        # Load engine fresh with base checkpoint
        engine = ForgeEngine.from_checkpoint(
            checkpoint=base_checkpoint,
            config_name=config_name,
            tokenizer_path=(
                "research/checkpoints/forgelm_v2_tokenizer"
                if config_name.startswith("forgelm") else None),
            device=device,
            # auto_activate=True would stack quant-restore buffers + CUDA
            # graphs + learned prefix cache on top of full weights — on a
            # 12GB card that left 0MB free and capped KV at 64 tokens.
            auto_activate=False,
        )
        engine.activate(kv_cache="hadamard_int4", decoding="standard",
                        use_triton_conv=True,
                        kv_cache_tokens=4096, warmup=True)

    # Build the eval set ONCE so both sides answer identical questions.
    # Per-side builds advanced the rng AND the ban list (base's high
    # scorers got banned before the candidate's categories were built),
    # so base and candidate used to face different question sets —
    # observed live: bit-identical checkpoints scored 0.54 vs 0.34.
    cats = _build_dynamic_categories(seed=eval_seed, scale=eval_scale,
                                     online_n=eval_online)
    n_tests = sum(len(v) for v in cats.values())
    print(f"  [FastEval] {n_tests} tests across {len(cats)} categories "
          f"(online={eval_online}/db, scale={eval_scale})")

    # Run base tests
    base_result = _run_tests_on_engine(engine, "BASE (V10)", device,
                                       verbose, cats=cats,
                                       config_name=config_name)
    base_result.checkpoint = base_checkpoint

    # Swap to candidate weights (load directly to GPU)
    print("\n  [FastEval] Swapping weights to candidate...")
    t_swap = time.perf_counter()
    candidate_state = _load_state_dict_on_device(candidate_checkpoint, device)
    with torch.no_grad():
        for name, param in engine.model.named_parameters():
            if name in candidate_state:
                param.data.copy_(candidate_state[name])
    del candidate_state

    # Invalidate cached KV — prefix-cache and CacheBlend entries were
    # computed under the base weights; reusing them would contaminate the
    # candidate pass with stale states.
    for cache in (getattr(engine, "_prefix_cache", None),
                  getattr(engine, "_cache_blend", None)):
        if cache is not None and hasattr(cache, "clear"):
            cache.clear()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"  [FastEval] Weight swap done in {time.perf_counter() - t_swap:.1f}s")

    # Run candidate tests — same question set as base
    candidate_result = _run_tests_on_engine(engine, "CANDIDATE (SFT)",
                                            device, verbose, cats=cats,
                                            config_name=config_name)
    candidate_result.checkpoint = candidate_checkpoint

    # If we created the engine internally (engine=None path), free it now.
    # The caller can't get a reference back (fast_eval returns a dict), so
    # leaving it alive leaks ~8.7 GB VRAM per epoch (model + CUDA graphs +
    # Triton patches + KV cache) and causes OOM on subsequent epochs.
    # If the caller passed an engine in, THEY manage its lifecycle.
    if engine_created_here:
        del engine
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        from forge.model_loader import ModelLoader
        ModelLoader.clear_cache()
        print("  [FastEval] Internal engine freed (VRAM reclaimed)")
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Print comparison
    _print_comparison(base_result, candidate_result)

    def _split_q(result, catset):
        ms = [m for c, l in result.per_category.items()
              if c in catset for m in l]
        return (sum(m.quality for m in ms) / len(ms)) if ms else 0.0

    # Save results JSON
    results = {
        "base": {
            "name": base_result.name, "checkpoint": base_result.checkpoint,
            "quality": base_result.total_quality, "tokens": base_result.total_tokens,
            "time_ms": base_result.total_time_ms, "vram_mb": base_result.peak_vram_mb,
            "tokens_per_sec": base_result.avg_tokens_per_sec,
            "heldout_q": _split_q(base_result, HELDOUT_CATS),
            "adjacent_q": _split_q(base_result, ADJACENT_CATS),
        },
        "candidate": {
            "name": candidate_result.name, "checkpoint": candidate_result.checkpoint,
            "quality": candidate_result.total_quality, "tokens": candidate_result.total_tokens,
            "time_ms": candidate_result.total_time_ms, "vram_mb": candidate_result.peak_vram_mb,
            "tokens_per_sec": candidate_result.avg_tokens_per_sec,
            "heldout_q": _split_q(candidate_result, HELDOUT_CATS),
            "adjacent_q": _split_q(candidate_result, ADJACENT_CATS),
        },
        "winner": "",
    }

    # Determine winner
    if candidate_result.total_quality > base_result.total_quality:
        results["winner"] = "CANDIDATE"
    elif base_result.total_quality > candidate_result.total_quality:
        results["winner"] = "BASE"
    else:
        if candidate_result.total_time_ms < base_result.total_time_ms:
            results["winner"] = "CANDIDATE"
        else:
            results["winner"] = "BASE"

    out_path = "research/data/benchmark_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.perf_counter() - t_total
    print(f"\n  [FastEval] Total eval time: {elapsed:.1f}s")
    return results


def _print_comparison(base: ModelResult, candidate: ModelResult):
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {base.name} vs {candidate.name}")
    print(f"{'='*70}")
    print(f"\n  {'Category':<15s} | {'Base Q':>7s} {'Base Tok':>8s} {'Base ms':>7s} "
          f"| {'Cand Q':>7s} {'Cand Tok':>8s} {'Cand ms':>7s} | {'Winner':>7s}")
    print(f"  {'-'*80}")
    for cat in base.per_category:
        b = base.per_category.get(cat, [])
        c = candidate.per_category.get(cat, [])
        b_q = sum(m.quality for m in b) / max(len(b), 1)
        c_q = sum(m.quality for m in c) / max(len(c), 1)
        b_t = sum(m.tokens for m in b)
        c_t = sum(m.tokens for m in c)
        b_ms = sum(m.time_ms for m in b)
        c_ms = sum(m.time_ms for m in c)
        winner = "CAND" if c_q > b_q else ("BASE" if b_q > c_q else "TIE")
        print(f"  {cat:<15s} | {b_q:7.2f} {b_t:8d} {b_ms:7.0f} "
              f"| {c_q:7.2f} {c_t:8d} {c_ms:7.0f} | {winner:>7s}")

    # Held-out (real benchmarks, disjoint from self-play data) vs adjacent
    # (procedural categories in the same family as training tasks). If the
    # candidate gains on adjacent but loses held-out → it memorized.
    def _split_q(result, catset):
        ms = [m for c, l in result.per_category.items()
              if c in catset for m in l]
        return (sum(m.quality for m in ms) / len(ms), len(ms)) if ms else (0.0, 0)
    for label, catset in (("HELDOUT (online DBs)", HELDOUT_CATS),
                          ("ADJACENT (self-play family)", ADJACENT_CATS)):
        bq, bn = _split_q(base, catset)
        cq, cn = _split_q(candidate, catset)
        if bn:
            print(f"  {label:<28s}: base {bq:.3f} ({bn}) | cand {cq:.3f} ({cn}) "
                  f"| delta {cq - bq:+.3f}")
    print(f"\n  {'OVERALL':<15s} | {base.total_quality:7.2f} {base.total_tokens:8d} {base.total_time_ms:7.0f} "
          f"| {candidate.total_quality:7.2f} {candidate.total_tokens:8d} {candidate.total_time_ms:7.0f} |")
    print(f"\n  {'Metric':<25s} | {'Base':>12s} {'Candidate':>12s} {'Delta':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Quality (avg)':<25s} | {base.total_quality:12.2f} {candidate.total_quality:12.2f} "
          f"{candidate.total_quality - base.total_quality:+10.2f}")
    print(f"  {'Total tokens':<25s} | {base.total_tokens:12d} {candidate.total_tokens:12d} "
          f"{candidate.total_tokens - base.total_tokens:+10d}")
    print(f"  {'Total time (ms)':<25s} | {base.total_time_ms:12.0f} {candidate.total_time_ms:12.0f} "
          f"{candidate.total_time_ms - base.total_time_ms:+10.0f}")
    print(f"  {'Peak VRAM (MB)':<25s} | {base.peak_vram_mb:12.0f} {candidate.peak_vram_mb:12.0f} "
          f"{candidate.peak_vram_mb - base.peak_vram_mb:+10.0f}")
    print(f"  {'Avg tokens/sec':<25s} | {base.avg_tokens_per_sec:12.1f} {candidate.avg_tokens_per_sec:12.1f} "
          f"{candidate.avg_tokens_per_sec - base.avg_tokens_per_sec:+10.1f}")

    if candidate.total_quality > base.total_quality:
        verdict = "CANDIDATE WINS"
    elif base.total_quality > candidate.total_quality:
        verdict = "BASE WINS"
    else:
        if candidate.total_time_ms < base.total_time_ms:
            verdict = "TIE on quality, CANDIDATE faster"
        else:
            verdict = "TIE on quality, BASE faster"
    print(f"\n  VERDICT: {verdict}")

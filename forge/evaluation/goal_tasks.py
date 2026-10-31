"""Goal-Oriented Self-Play (GOSP) — goal task definitions and adaptive generator.

Replaces the function-stub prompt approach. Instead of telling the model WHAT to
implement ("def fib(n): ..."), we state the GOAL and verify via I/O pairs.

Each GoalTask specifies:
  - A human-readable goal description (shown to the model)
  - An input signature (tells the model the I/O shape)
  - Test cases: (args, expected_output) pairs including edge cases + stress input
  - Which test case is the stress input (for efficiency scoring)

The model can solve the goal ANY way it wants — iterative, recursive, memoized,
closed-form — as long as its solve() function produces correct output for ALL
test inputs. This promotes creativity while maintaining verifiable correctness.

Trusted reference implementations compute expected outputs ONLY. They are never
shown to the model and never graded against — they exist solely to generate the
"expected" side of I/O test pairs.

Research basis:
  - PSV (2025): difficulty-aware proposal is essential for self-play
  - DPE/EVALPERf: stress-test inputs needed to distinguish O(n) from O(n^2)
  - EvoCurr/GASP: adaptive curriculum — ease when struggling, escalate on success
  - ANCORA: novelty filtering — skip goals too similar to recent ones

Usage:
    from forge.evaluation.goal_tasks import GoalTaskGenerator
    gen = GoalTaskGenerator()
    task = gen.generate(domain="algorithms", difficulty="medium")
    print(task.description)       # shown to model
    print(task.test_cases)        # for verification
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any

# ─── GoalTask ─────────────────────────────────────────────────────────

@dataclass
class GoalTask:
    """A goal-oriented task: achieve this output for these inputs, any way you want."""
    id: str
    domain: str                          # e.g. "algorithms", "math", "strings"
    difficulty: str                      # "easy", "medium", "hard"
    description: str                     # human-readable goal (shown to model)
    input_signature: str                 # e.g. "(n: int) -> int"
    solve_name: str = "solve"            # function name the model must define
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    # each: {"args": (...,), "expected": ...}
    stress_index: int | None = None   # which test case is the stress input
    archetype: str = ""                  # which archetype generated this


# ─── Trusted reference implementations ────────────────────────────────
# These compute expected outputs ONLY. Never shown to the model.

def _ref_fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def _ref_factorial(n: int) -> int:
    return math.factorial(max(0, n))

def _ref_is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

def _ref_gcd(a: int, b: int) -> int:
    return math.gcd(a, b)

def _ref_reverse_string(s: str) -> str:
    return s[::-1]

def _ref_sum_list(lst: list[int]) -> int:
    return sum(lst)

def _ref_sort_list(lst: list[int]) -> list[int]:
    return sorted(lst)

def _ref_count_vowels(s: str) -> int:
    return sum(1 for c in s if c in 'aeiouAEIOU')

def _ref_is_palindrome(s: str) -> bool:
    return s == s[::-1]

def _ref_power(base: int, exp: int) -> int:
    return base ** exp

def _ref_digit_sum(n: int) -> int:
    return sum(int(d) for d in str(abs(n)))

def _ref_collatz_steps(n: int) -> int:
    if n <= 0:
        return 0
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

def _ref_word_count(s: str) -> int:
    return len(s.split())

def _ref_max_element(lst: list[int]) -> int:
    return max(lst) if lst else 0

def _ref_linear_search(lst: list[int], target: int) -> int:
    for i, v in enumerate(lst):
        if v == target:
            return i
    return -1


# ─── Archetype registry ───────────────────────────────────────────────
# Each archetype: name, domain, description, signature, ref func,
# input generator (difficulty -> list of arg tuples), edge cases

def _gen_fibonacci_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [(0,), (1,), (5,), (10,)]
    elif difficulty == "medium":
        return [(0,), (1,), (10,), (20,), (30,)]
    else:
        return [(0,), (1,), (10,), (30,), (45,)]

def _gen_factorial_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [(0,), (1,), (5,)]
    elif difficulty == "medium":
        return [(0,), (1,), (5,), (10,)]
    else:
        return [(0,), (1,), (10,), (15,)]

def _gen_is_prime_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [(2,), (7,), (10,), (1,)]
    elif difficulty == "medium":
        return [(2,), (7,), (10,), (1,), (97,), (100,)]
    else:
        return [(2,), (1,), (97,), (997,), (7919,)]

def _gen_gcd_inputs(difficulty: str) -> list[tuple]:
    random.Random(42)
    if difficulty == "easy":
        return [(12, 8,), (7, 3,), (0, 5,)]
    elif difficulty == "medium":
        return [(12, 8,), (7, 3,), (0, 5,), (48, 36,)]
    else:
        return [(12, 8,), (0, 5,), (48, 36,), (1071, 462,)]

def _gen_reverse_string_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [("hello",), ("a",), ("",)]
    elif difficulty == "medium":
        return [("hello",), ("a",), ("",), ("racecar",)]
    else:
        return [("hello",), ("",), ("racecar",), ("a" * 100,)]

def _gen_sum_list_inputs(difficulty: str) -> list[tuple]:
    random.Random(42)
    if difficulty == "easy":
        return [([1, 2, 3],), ([],), ([5],)]
    elif difficulty == "medium":
        return [([1, 2, 3],), ([],), ([5],), ([-1, 0, 1],)]
    else:
        return [([1, 2, 3],), ([],), ([-1, 0, 1],), (list(range(1000)),)]

def _gen_sort_list_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [([3, 1, 2],), ([],), ([5],)]
    elif difficulty == "medium":
        return [([3, 1, 2],), ([],), ([5],), ([5, 4, 3, 2, 1],)]
    else:
        return [([3, 1, 2],), ([],), ([5, 4, 3, 2, 1],), (list(range(100, 0, -1)),)]

def _gen_count_vowels_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [("hello",), ("",), ("aeiou",)]
    elif difficulty == "medium":
        return [("hello",), ("",), ("aeiou",), ("HELLO",)]
    else:
        return [("hello",), ("",), ("HELLO",), ("a" * 200,)]

def _gen_is_palindrome_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [("racecar",), ("hello",), ("",)]
    elif difficulty == "medium":
        return [("racecar",), ("hello",), ("",), ("a",)]
    else:
        return [("racecar",), ("hello",), ("a",), ("A man a plan a canal Panama"[::1],)]

def _gen_power_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [(2, 3,), (5, 0,), (1, 10,)]
    elif difficulty == "medium":
        return [(2, 3,), (5, 0,), (1, 10,), (3, 5,)]
    else:
        return [(2, 3,), (5, 0,), (3, 5,), (2, 20,)]

def _gen_digit_sum_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [(123,), (0,), (9,)]
    elif difficulty == "medium":
        return [(123,), (0,), (9,), (-456,)]
    else:
        return [(123,), (0,), (-456,), (10**18,)]

def _gen_collatz_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [(1,), (2,), (6,)]
    elif difficulty == "medium":
        return [(1,), (2,), (6,), (27,)]
    else:
        return [(1,), (6,), (27,), (97,)]

def _gen_word_count_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [("hello world",), ("",), ("one",)]
    elif difficulty == "medium":
        return [("hello world",), ("",), ("one",), ("the quick brown fox",)]
    else:
        return [("hello world",), ("",), ("the quick brown fox",), (" ".join(["word"] * 100),)]

def _gen_max_element_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [([1, 3, 2],), ([5],), ([0, -1, -3],)]
    elif difficulty == "medium":
        return [([1, 3, 2],), ([5],), ([0, -1, -3],), ([100, 50, 200, 25],)]
    else:
        return [([1, 3, 2],), ([5],), ([100, 50, 200, 25],), (list(range(10000)),)]

def _gen_linear_search_inputs(difficulty: str) -> list[tuple]:
    if difficulty == "easy":
        return [([1, 3, 5], 3,), ([1, 3, 5], 7,), ([], 1,)]
    elif difficulty == "medium":
        return [([1, 3, 5], 3,), ([1, 3, 5], 7,), ([], 1,), ([10, 20, 30], 30,)]
    else:
        return [([1, 3, 5], 7,), ([], 1,), ([10, 20, 30], 30,), (list(range(10000)), 9999,)]


ARCHETYPES: dict[str, dict] = {
    "fibonacci": {
        "domain": "algorithms", "signature": "(n: int) -> int",
        "description": "Compute the n-th Fibonacci number (fib(0)=0, fib(1)=1)",
        "ref": _ref_fibonacci, "gen_inputs": _gen_fibonacci_inputs,
    },
    "factorial": {
        "domain": "math", "signature": "(n: int) -> int",
        "description": "Compute n factorial (n!)",
        "ref": _ref_factorial, "gen_inputs": _gen_factorial_inputs,
    },
    "is_prime": {
        "domain": "math", "signature": "(n: int) -> bool",
        "description": "Check if n is a prime number",
        "ref": _ref_is_prime, "gen_inputs": _gen_is_prime_inputs,
    },
    "gcd": {
        "domain": "math", "signature": "(a: int, b: int) -> int",
        "description": "Compute the greatest common divisor of a and b",
        "ref": _ref_gcd, "gen_inputs": _gen_gcd_inputs,
    },
    "reverse_string": {
        "domain": "strings", "signature": "(s: str) -> str",
        "description": "Reverse the input string",
        "ref": _ref_reverse_string, "gen_inputs": _gen_reverse_string_inputs,
    },
    "sum_list": {
        "domain": "algorithms", "signature": "(lst: list) -> int",
        "description": "Compute the sum of all elements in a list",
        "ref": _ref_sum_list, "gen_inputs": _gen_sum_list_inputs,
    },
    "sort_list": {
        "domain": "algorithms", "signature": "(lst: list) -> list",
        "description": "Return a new list with all elements sorted in ascending order",
        "ref": _ref_sort_list, "gen_inputs": _gen_sort_list_inputs,
    },
    "count_vowels": {
        "domain": "strings", "signature": "(s: str) -> int",
        "description": "Count the number of vowels (a,e,i,o,u) in the string (case-insensitive)",
        "ref": _ref_count_vowels, "gen_inputs": _gen_count_vowels_inputs,
    },
    "is_palindrome": {
        "domain": "strings", "signature": "(s: str) -> bool",
        "description": "Check if the string reads the same forwards and backwards",
        "ref": _ref_is_palindrome, "gen_inputs": _gen_is_palindrome_inputs,
    },
    "power": {
        "domain": "math", "signature": "(base: int, exp: int) -> int",
        "description": "Compute base raised to the power of exp",
        "ref": _ref_power, "gen_inputs": _gen_power_inputs,
    },
    "digit_sum": {
        "domain": "math", "signature": "(n: int) -> int",
        "description": "Compute the sum of digits of n (absolute value)",
        "ref": _ref_digit_sum, "gen_inputs": _gen_digit_sum_inputs,
    },
    "collatz_steps": {
        "domain": "algorithms", "signature": "(n: int) -> int",
        "description": "Count Collatz steps to reach 1 (even: n/2, odd: 3n+1). Return 0 if n<=0.",
        "ref": _ref_collatz_steps, "gen_inputs": _gen_collatz_inputs,
    },
    "word_count": {
        "domain": "strings", "signature": "(s: str) -> int",
        "description": "Count the number of words in the string (whitespace-separated)",
        "ref": _ref_word_count, "gen_inputs": _gen_word_count_inputs,
    },
    "max_element": {
        "domain": "algorithms", "signature": "(lst: list) -> int",
        "description": "Find the maximum element in a list. Return 0 if list is empty.",
        "ref": _ref_max_element, "gen_inputs": _gen_max_element_inputs,
    },
    "linear_search": {
        "domain": "algorithms", "signature": "(lst: list, target: int) -> int",
        "description": "Find the index of target in lst. Return -1 if not found.",
        "ref": _ref_linear_search, "gen_inputs": _gen_linear_search_inputs,
    },
}


# ─── Adaptive difficulty controller ───────────────────────────────────

class AdaptiveDifficulty:
    """Tracks per-domain success rate and adjusts difficulty (EvoCurr/GASP).

    Maintains a rolling window of recent results per domain:
      - success > 0.8 → escalate (easy→medium→hard)
      - success < 0.3 → ease (hard→medium→easy)
      - otherwise → maintain
    """

    LEVELS = ["easy", "medium", "hard"]
    WINDOW_SIZE = 20

    def __init__(self):
        self._results: dict[str, list[bool]] = {}  # domain -> [success, ...]
        self._level: dict[str, int] = {}            # domain -> level index

    def record(self, domain: str, success: bool):
        results = self._results.setdefault(domain, [])
        results.append(success)
        if len(results) > self.WINDOW_SIZE:
            results = results[-self.WINDOW_SIZE:]
        self._results[domain] = results
        self._adjust(domain)

    def _adjust(self, domain: str):
        results = self._results.get(domain, [])
        if len(results) < 5:
            return
        rate = sum(results) / len(results)
        idx = self._level.get(domain, 0)
        if rate > 0.8 and idx < len(self.LEVELS) - 1:
            self._level[domain] = idx + 1
        elif rate < 0.3 and idx > 0:
            self._level[domain] = idx - 1

    def get_difficulty(self, domain: str) -> str:
        idx = self._level.get(domain, 0)
        return self.LEVELS[idx]


# ─── Goal task generator ──────────────────────────────────────────────

class GoalTaskGenerator:
    """Generates GoalTasks with adaptive difficulty and I/O verification pairs.

    Trusted references compute expected outputs ONLY (never shown to model).
    The stress_index marks the largest input for efficiency scoring.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.difficulty_ctrl = AdaptiveDifficulty()
        self._task_counter = 0
        self._recent_archetypes: list[str] = []  # novelty filter (ANCORA)

    def generate(self, domain: str | None = None,
                 difficulty: str | None = None,
                 archetype: str | None = None) -> GoalTask:
        """Generate a single GoalTask.

        Args:
            domain: filter by domain (None = any)
            difficulty: override adaptive difficulty (None = adaptive)
            archetype: specific archetype (None = random within domain)
        """
        # Select archetype
        if archetype:
            arch = ARCHETYPES[archetype]
            arch_name = archetype
        else:
            candidates = {k: v for k, v in ARCHETYPES.items()
                          if domain is None or v["domain"] == domain}
            if not candidates:
                raise ValueError(f"No archetypes for domain '{domain}'")
            # Novelty filter: prefer archetypes not recently used
            fresh = [k for k in candidates if k not in self._recent_archetypes[-5:]]
            pool = fresh if fresh else list(candidates.keys())
            arch_name = self.rng.choice(pool)
            arch = candidates[arch_name]

        self._recent_archetypes.append(arch_name)

        # Determine difficulty
        eff_domain = arch["domain"]
        if difficulty is None:
            difficulty = self.difficulty_ctrl.get_difficulty(eff_domain)

        # Generate inputs and compute expected outputs
        inputs = arch["gen_inputs"](difficulty)
        ref = arch["ref"]

        test_cases = []
        for args in inputs:
            try:
                expected = ref(*args)
            except Exception:
                expected = None
            test_cases.append({"args": args, "expected": expected})

        # Identify stress index (last/largest input)
        stress_index = len(test_cases) - 1 if len(test_cases) > 2 else None

        self._task_counter += 1
        task_id = f"{arch_name}_{difficulty}_{self._task_counter}"

        return GoalTask(
            id=task_id,
            domain=eff_domain,
            difficulty=difficulty,
            description=arch["description"],
            input_signature=arch["signature"],
            solve_name="solve",
            test_cases=test_cases,
            stress_index=stress_index,
            archetype=arch_name,
        )

    def generate_batch(self, n: int, domain: str | None = None) -> list[GoalTask]:
        """Generate n tasks, distributing across archetypes for diversity."""
        tasks = []
        for _ in range(n):
            tasks.append(self.generate(domain=domain))
        return tasks

    def record_result(self, domain: str, success: bool):
        """Feed result back to adaptive difficulty controller."""
        self.difficulty_ctrl.record(domain, success)


def build_goal_prompt(task: GoalTask) -> str:
    """Build the prompt shown to the model for a goal task.

    States the GOAL and the I/O contract, not the implementation.
    The model must define solve(...) and is free to implement it any way.
    """
    lines = [
        f"# Goal: {task.description}",
        f"# Define a function `{task.solve_name}{task.input_signature}`.",
        "# It must produce the correct output for ALL of these test inputs:",
        "",
    ]
    for i, tc in enumerate(task.test_cases):
        args_str = ", ".join(repr(a) for a in tc["args"])
        expected = tc["expected"]
        marker = " (stress test)" if i == task.stress_index else ""
        lines.append(f"#   {task.solve_name}({args_str}) == {expected!r}{marker}")
    lines.append("")
    lines.append(f"# Implement {task.solve_name} any way you choose.")
    lines.append("# The function must return (not print) the result.")
    lines.append(f"def {task.solve_name}{task.input_signature.split(' -> ')[0]}:")
    lines.append(f'    """{task.description}"""')
    lines.append("    ")
    return "\n".join(lines)

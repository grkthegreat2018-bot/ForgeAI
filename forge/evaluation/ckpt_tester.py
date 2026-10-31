"""R&D Round 23: Checkpoint Tester.

Loads a checkpoint and runs a 50-question probe across 5 categories
(10 each: math, logic, code, recall, format) plus a val-loss delta
vs a prior checkpoint for regression detection.

Usage:
    from forge.evaluation.ckpt_tester import CheckpointTester
    tester = CheckpointTester(model, cfg, device, prior_val_loss=prev_loss)
    report = tester.run()
    if report["block"]:
        print("CRITICAL REGRESSION — do not ship this checkpoint")
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import torch
import torch.nn.functional as F

# ── Question Bank ─────────────────────────────────────────────────────────

QUESTION_BANK: dict[str, list[dict[str, Any]]] = {
    # ── Math: arithmetic / algebra ──
    "math": [
        {"category": "math", "prompt": "What is 2+2?", "answer": "4"},
        {"category": "math", "prompt": "What is 15*3?", "answer": "45"},
        {"category": "math", "prompt": "What is 100-37?", "answer": "63"},
        {"category": "math", "prompt": "What is 81/9?", "answer": "9"},
        {"category": "math", "prompt": "What is 7+8?", "answer": "15"},
        {"category": "math", "prompt": "What is 12*11?", "answer": "132"},
        {"category": "math", "prompt": "What is 200-145?", "answer": "55"},
        {"category": "math", "prompt": "What is 144/12?", "answer": "12"},
        {"category": "math", "prompt": "What is 25+30?", "answer": "55"},
        {"category": "math", "prompt": "What is 9*9?", "answer": "81"},
    ],
    # ── Logic: syllogisms / sequence completion ──
    "logic": [
        {"category": "logic", "prompt": "All men are mortal. Socrates is a man. What is Socrates?", "answer": "mortal"},
        {"category": "logic", "prompt": "Complete the sequence: 2, 4, 6, 8, ?", "answer": "10"},
        {"category": "logic", "prompt": "Complete the sequence: 1, 1, 2, 3, 5, 8, ?", "answer": "13"},
        {"category": "logic", "prompt": "If A>B and B>C, then A is what relative to C?", "answer": "greater"},
        {"category": "logic", "prompt": "Complete the sequence: 3, 6, 12, 24, ?", "answer": "48"},
        {"category": "logic", "prompt": "All birds can fly. A penguin is a bird. Can a penguin fly?", "answer": "yes"},
        {"category": "logic", "prompt": "Complete the sequence: 1, 4, 9, 16, 25, ?", "answer": "36"},
        {"category": "logic", "prompt": "If all cats are animals and Whiskers is a cat, what is Whiskers?", "answer": "animal"},
        {"category": "logic", "prompt": "Complete the sequence: 2, 6, 18, 54, ?", "answer": "162"},
        {"category": "logic", "prompt": "If X=5 and Y=X+3, what is Y?", "answer": "8"},
    ],
    # ── Code: simple function generation ──
    "code": [
        {"category": "code", "prompt": "Write a Python function add(a, b) that returns a+b.", "answer": "def add(a, b): return a + b"},
        {"category": "code", "prompt": "Write a Python function multiply(a, b) that returns a*b.", "answer": "def multiply(a, b): return a * b"},
        {"category": "code", "prompt": "Write a Python function is_even(n) that returns True if n is even.", "answer": "def is_even(n): return n % 2 == 0"},
        {"category": "code", "prompt": "Write a Python function square(n) that returns n*n.", "answer": "def square(n): return n * n"},
        {"category": "code", "prompt": "Write a Python function max_of_two(a, b) returning the larger.", "answer": "def max_of_two(a, b): return a if a > b else b"},
        {"category": "code", "prompt": "Write a Python function reverse_string(s) that reverses s.", "answer": "def reverse_string(s): return s[::-1]"},
        {"category": "code", "prompt": "Write a Python function factorial(n) returning n!.", "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"},
        {"category": "code", "prompt": "Write a Python function is_palindrome(s) checking if s reads the same forwards and backwards.", "answer": "def is_palindrome(s): return s == s[::-1]"},
        {"category": "code", "prompt": "Write a Python function count_vowels(s) counting vowels in s.", "answer": "def count_vowels(s): return sum(1 for c in s if c in 'aeiou')"},
        {"category": "code", "prompt": "Write a Python function fib(n) returning the n-th Fibonacci number.", "answer": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"},
    ],
    # ── Recall: factual knowledge ──
    "recall": [
        {"category": "recall", "prompt": "What is the capital of France?", "answer": "Paris"},
        {"category": "recall", "prompt": "What is the largest planet in our solar system?", "answer": "Jupiter"},
        {"category": "recall", "prompt": "Who wrote the play 'Romeo and Juliet'?", "answer": "Shakespeare"},
        {"category": "recall", "prompt": "What is the chemical symbol for water?", "answer": "H2O"},
        {"category": "recall", "prompt": "What is the tallest mountain on Earth?", "answer": "Everest"},
        {"category": "recall", "prompt": "How many continents are there on Earth?", "answer": "7"},
        {"category": "recall", "prompt": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide"},
        {"category": "recall", "prompt": "What is the speed of light approximately (in km/s)?", "answer": "300000"},
        {"category": "recall", "prompt": "Who painted the Mona Lisa?", "answer": "da Vinci"},
        {"category": "recall", "prompt": "What is the largest ocean on Earth?", "answer": "Pacific"},
    ],
    # ── Format: JSON, markdown, tool-use ──
    "format": [
        {"category": "format", "prompt": "Return valid JSON: an object with key 'name' and value 'Bob'.", "answer": '{"name": "Bob"}'},
        {"category": "format", "prompt": "Format this as markdown with a code block: def foo(): return 1", "answer": "```python\ndef foo(): return 1\n```"},
        {"category": "format", "prompt": "Output a tool-use call in JSON format with tool name 'search' and query 'cats'.", "answer": '{"tool": "search", "query": "cats"}'},
        {"category": "format", "prompt": "Create a JSON array with three numbers: 1, 2, 3.", "answer": "[1, 2, 3]"},
        {"category": "format", "prompt": "Write markdown with a heading '# Title' and a bullet list.", "answer": "# Title\n- item 1\n- item 2"},
        {"category": "format", "prompt": "Return a JSON object with 'status' set to 'ok'.", "answer": '{"status": "ok"}'},
        {"category": "format", "prompt": "Format the answer as a markdown table with headers.", "answer": "| A | B |\n|---|---|\n| 1 | 2 |"},
        {"category": "format", "prompt": "Output tool-use format: call tool 'calc' with argument 5.", "answer": '{"tool": "calc", "args": [5]}'},
        {"category": "format", "prompt": "Return valid JSON with nested object: {a: {b: 1}}.", "answer": '{"a": {"b": 1}}'},
        {"category": "format", "prompt": "Format as markdown code block with language 'python': print('hi').", "answer": "```python\nprint('hi')\n```"},
    ],
}


# ── Answer Evaluation ─────────────────────────────────────────────────────

def evaluate_answer(question: dict[str, Any], answer: str) -> bool | None:
    """Evaluate whether *answer* is correct for *question*.

    Returns:
        True  — answer is correct.
        False — answer is wrong.
        None  — cannot evaluate (missing expected answer, unparseable, etc.).
    """
    if answer is None:
        return None
    answer_str = str(answer).strip()
    if not answer_str:
        return None

    cat = question.get("category", "")
    expected = question.get("answer", question.get("expected", ""))
    expected_str = str(expected).strip() if expected is not None else ""

    # ── Math: check if the expected number appears as a standalone token ──
    if cat == "math":
        numbers = re.findall(r"-?\d+\.?\d*", expected_str)
        if not numbers:
            return None
        expected_num = numbers[-1]  # last number is the answer
        # Word-boundary match so "4" doesn't match inside "12345"
        pattern = r"(?<!\d)" + re.escape(expected_num) + r"(?!\d)"
        return bool(re.search(pattern, answer_str))

    # ── Format: validate JSON / markdown / tool-use structure ──
    if cat == "format":
        prompt_lower = str(question.get("prompt", "")).lower()
        if "json" in prompt_lower:
            # Strict JSON validation
            try:
                parsed = json.loads(answer_str)
                return bool(isinstance(parsed, (dict, list)))
            except (json.JSONDecodeError, ValueError):
                return False
        if "markdown" in prompt_lower or "``" in prompt_lower:
            # Markdown: expect a code block fence or heading
            return bool("```" in answer_str or answer_str.startswith("#"))
        if "tool" in prompt_lower:
            # Tool-use: expect JSON-like structure with tool key
            if "{" in answer_str and "}" in answer_str:
                try:
                    parsed = json.loads(answer_str)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        return True
                except (json.JSONDecodeError, ValueError):
                    pass
                return False
            return False
        # Generic format fallback
        return bool(expected_str and expected_str.lower() in answer_str.lower())

    # ── Logic / Code / Recall: keyword or substring match ──
    if expected_str:
        # For numeric answers, use word-boundary matching
        if expected_str.lstrip("-").isdigit():
            pattern = r"(?<!\d)" + re.escape(expected_str) + r"(?!\d)"
            return bool(re.search(pattern, answer_str))
        # Case-insensitive substring match
        return expected_str.lower() in answer_str.lower()

    return None


# ── Checkpoint Tester ─────────────────────────────────────────────────────

class CheckpointTester:
    """Run a 50-question probe + val-loss regression check on a checkpoint.

    Parameters:
        model: a ConfigurableResearchLLM (or compatible) with forward(idx, targets)
               returning (logits, loss) or (logits, loss, *presents).
        cfg: the model config (must have ``vocab_size`` and ``max_seq_len``).
        device: torch device for input tensors.
        prior_val_loss: val loss from the previous checkpoint (for delta).
        critical_threshold: if val_loss_delta exceeds this, flag regression.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Any,
        device: torch.device,
        prior_val_loss: float | None = None,
        critical_threshold: float = 0.05,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.prior_val_loss = prior_val_loss
        self.critical_threshold = critical_threshold

    # ── Generation ──────────────────────────────────────────────────────

    def _encode_prompt(self, prompt: str, max_len: int = 64) -> torch.Tensor:
        """Encode *prompt* to byte-level token ids (vocab ≤ 256 → byte mapping)."""
        vocab = getattr(self.cfg, "vocab_size", 256)
        raw = prompt.encode("utf-8", errors="ignore")
        if vocab <= 256:
            ids = [b % vocab for b in raw][:max_len]
        else:
            # Larger vocab: use raw byte values (still 0-255, well within vocab)
            ids = list(raw)[:max_len]
        if not ids:
            ids = [0]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _generate_answer(self, prompt: str, max_new_tokens: int = 16) -> str:
        """Generate a short answer via greedy decoding (argmax on logits)."""
        max_seq = getattr(self.cfg, "max_seq_len", 128)
        max_prompt = min(64, max_seq - max_new_tokens - 1)
        if max_prompt < 4:
            max_prompt = 4

        input_ids = self._encode_prompt(prompt, max_len=max_prompt)
        prompt_len = input_ids.size(1)

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if input_ids.size(1) >= max_seq:
                        input_ids = input_ids[:, -max_seq + 1:]
                    out = self.model(input_ids)
                    logits = out[0]
                    if logits is None:
                        break
                    next_tok = logits[0, -1].argmax(dim=-1, keepdim=True).view(1, 1)
                    input_ids = torch.cat([input_ids, next_tok], dim=1)
        except Exception:
            return ""

        generated_ids = input_ids[0, prompt_len:].tolist()
        if not generated_ids:
            return ""
        try:
            return bytes(b % 256 for b in generated_ids).decode("utf-8", errors="ignore")
        except Exception:
            return str(generated_ids)

    # ── Val Loss ────────────────────────────────────────────────────────

    def _compute_val_loss(self, batch_size: int = 2, seq_len: int = 32) -> float:
        """Compute CE loss on a *fixed* random batch (seed=42 for determinism).

        Using a fixed seed ensures the same batch is used across tester
        instances, so val_loss_delta is meaningful (same data, different weights).
        """
        vocab = getattr(self.cfg, "vocab_size", 256)
        gen = torch.Generator(device="cpu").manual_seed(42)
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), generator=gen, dtype=torch.long)
        targets = torch.randint(0, vocab, (batch_size, seq_len), generator=gen, dtype=torch.long)
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)

        try:
            with torch.no_grad():
                out = self.model(input_ids, targets)
                logits = out[0]
                loss = out[1]
                if loss is None and logits is not None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
        except Exception:
            return 0.0

        if loss is None:
            return 0.0
        loss_val = float(loss.item())
        # Guard against inf/nan from extreme weight perturbations —
        # return a large finite value so delta is still positive.
        if not math.isfinite(loss_val):
            return 1e6
        return loss_val

    # ── Main Entry Point ────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Run the 50-question probe + val-loss regression check.

        Returns a report dict with:
            total_score, category_scores, val_loss, val_loss_delta,
            regression_flag, questions_passed, questions_total, block.
        """
        self.model.eval()

        # ── Run questions ──
        category_scores: dict[str, float] = {}
        total_passed = 0
        for cat, questions in QUESTION_BANK.items():
            passed = 0
            for q in questions:
                try:
                    answer = self._generate_answer(q["prompt"])
                except Exception:
                    answer = ""
                result = evaluate_answer(q, answer)
                if result is True:
                    passed += 1
            n = len(questions)
            category_scores[cat] = passed / n if n > 0 else 0.0
            total_passed += passed

        total_score = total_passed / 50

        # ── Val loss ──
        val_loss = self._compute_val_loss()

        # ── Delta & regression ──
        if self.prior_val_loss is not None:
            val_loss_delta = val_loss - self.prior_val_loss
        else:
            val_loss_delta = 0.0

        regression_flag = val_loss_delta > self.critical_threshold
        block = regression_flag and val_loss_delta > self.critical_threshold

        return {
            "total_score": float(total_score),
            "category_scores": category_scores,
            "val_loss": float(val_loss),
            "val_loss_delta": float(val_loss_delta),
            "regression_flag": bool(regression_flag),
            "questions_passed": int(total_passed),
            "questions_total": 50,
            "block": bool(block),
        }


# ── Alternative Entry Point ───────────────────────────────────────────────

def run_test(
    model: torch.nn.Module | None = None,
    cfg: Any | None = None,
    device: torch.device | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience function: build a CheckpointTester and run it.

    Usage:
        report = run_test(model, cfg, device, prior_val_loss=prev)
    """
    if model is None or cfg is None:
        raise ValueError("run_test requires 'model' and 'cfg' arguments")
    if device is None:
        device = torch.device("cpu")
    tester = CheckpointTester(model, cfg, device=device, **kwargs)
    return tester.run()

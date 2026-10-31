"""Self-Play Sandbox — model generates code, executes it, logs everything.

Architecture:
  1. Model generates a solution (Python code) for a task
  2. Sandbox executes the code safely (subprocess + timeout + resource limits)
  3. Telemetry logs: execution time, memory, output, errors, file size, correctness
  4. Successful runs → KnowledgePacket (for later weight injection)
  5. Failed runs → error analysis (model learns from mistakes)

Data packet format (JSON, one per run):
  {
    "task": "fibonacci(10)",
    "task_type": "python",
    "prompt": "Write a function to compute fibonacci(10)",
    "generated_code": "def fib(n): ...",
    "execution": {
      "stdout": "55",
      "stderr": "",
      "returncode": 0,
      "exec_time_ms": 0.42,
      "peak_memory_kb": 12300,
      "file_size_bytes": 156,
      "output_matches_expected": true
    },
    "model_telemetry": {
      "gen_time_ms": 320,
      "tokens_generated": 45,
      "tokens_per_second": 140.6,
      "mean_logprob": -0.12,
      "confidence": 0.88
    },
    "quality_score": 0.92,
    "knowledge_text": "fibonacci(10) = 55. def fib(n): a,b=0,1; ...",
    "timestamp": "2026-01-15T12:00:00"
  }

Usage:
    from forge.self_play.self_play_sandbox import SelfPlaySandbox
    sandbox = SelfPlaySandbox(model, tokenizer, log_dir="research/data/self_play")
    sandbox.run_task("Compute fibonacci(10) in Python", expected_output="55")
"""
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from research.training_free.solver import TrainingFreeSolver

from forge.engine.scheduler.async_d2h import AsyncTokenReader, StreamedGenerator
from forge.self_play.io_match import io_match, io_similarity
from research.json_compat import dumps
import contextlib

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns (avoids recompilation per task).
_RE_DEF_FUNC = re.compile(r'def (\w+)')
_RE_DEF_CLASS = re.compile(r'class (\w+)')
_RE_FUNC_SIG = re.compile(r'def\s+(\w+)\s*\(([^)]*)\)')
_RE_DIGITS = re.compile(r'\d+')

# resource is Unix-only; on Windows we skip memory limits
try:
    import resource  # noqa: F401
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


class SandboxExecutor:
    """Safe Python code executor with full telemetry.

    Executes code in a subprocess with:
      - Timeout (default 5s)
      - Memory limit (default 256MB)
      - No network access (no imports of socket/urllib at OS level)
      - Captures stdout, stderr, return code, peak memory, file size

    Uses PersistentCodeExecutor by default (avoids 200ms Python startup per
    execution on Windows). Falls back to subprocess.run if persistent fails.

    ``temp_dir`` controls where scratch files are written (defaults to the
    system temp dir). Pass a fast/short-path directory on Windows to avoid
    MAX_PATH issues and slow temp volumes.
    """

    def __init__(self, timeout_s: float = 5.0, memory_limit_mb: int = 256,
                 use_persistent: bool = True, temp_dir: str | None = None,
                 workers: int = 1):
        self.timeout_s = timeout_s
        self.memory_limit_mb = memory_limit_mb
        self.temp_dir = temp_dir or tempfile.gettempdir()
        os.makedirs(self.temp_dir, exist_ok=True)
        self._persistent = None
        # Worker pool: N persistent executors, round-robin. A single
        # PersistentCodeExecutor serializes on its pipe lock, so a pool is
        # required for genuinely parallel verification from threads. Each
        # worker respawns independently on crash/timeout (stall isolation).
        self._pool: list = []
        self._pool_rr = 0
        import threading as _th
        self._pool_lock = _th.Lock()
        if use_persistent:
            try:
                from research.persistent_executor import PersistentCodeExecutor
                n = max(1, int(workers))
                self._pool = [PersistentCodeExecutor(
                    timeout_s=timeout_s, memory_limit_mb=memory_limit_mb,
                    fs_root=self.temp_dir)
                    for _ in range(n)]
                self._persistent = self._pool[0]
            except Exception as e:
                logger.warning("PersistentCodeExecutor unavailable, using "
                               "subprocess fallback (~200ms/exec slower): %s", e)
                self._persistent = None
                self._pool = []

    def _next_worker(self):
        with self._pool_lock:
            w = self._pool[self._pool_rr % len(self._pool)]
            self._pool_rr += 1
            return w

    def close(self) -> None:
        """Terminate all persistent workers."""
        for w in self._pool:
            try:
                w.close()
            except Exception:
                logger.debug("Error closing worker in sandbox pool", exc_info=True)
        self._pool = []
        self._persistent = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.debug("Error during sandbox cleanup", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def execute(self, code: str, expected_output: str | None = None) -> dict:
        """Execute Python code and return full telemetry.

        Args:
            code: Python code to execute
            expected_output: expected stdout (for correctness checking)

        Returns:
            Telemetry dict with execution results
        """
        # Fast path: use a persistent worker (avoids 200ms startup per call).
        if self._pool:
            result = self._next_worker().execute(code, expected_output)
            # Semantic I/O scoring: keep both paths consistent (subprocess
            # fallback applies the same tolerant matching below).
            if expected_output is not None and "io_score" not in result:
                result["io_score"] = io_similarity(expected_output, result.get("stdout", ""))
                result["output_matches_expected"] = result["io_score"] >= 0.99
            return result

        # Fallback: one-shot sandboxed worker (~200ms startup per execution).
        # The once-script enforces the SAME containment as the persistent
        # worker (audit hook, jail dir, env scrub, no pipe fds) — the old
        # two-stage wrapper applied its import hook to the wrong process,
        # leaving the executed code fully unrestricted.
        from research.persistent_executor import (
            assign_to_job, build_once_script, close_job, scrub_environ)
        import threading
        import uuid
        unique_suffix = f'_{threading.get_ident()}_{uuid.uuid4().hex[:8]}.py'
        with tempfile.NamedTemporaryFile(mode='w', suffix=unique_suffix,
                                          delete=False, encoding='utf-8',
                                          dir=self.temp_dir) as f:
            f.write(code)
            temp_path = f.name

        file_size = os.path.getsize(temp_path)
        once_path = temp_path + '_once.py'
        result_path = temp_path + '_result.json'
        with open(once_path, 'w', encoding='utf-8') as f:
            f.write(build_once_script())

        t0 = time.time()
        result = {
            "stdout": "", "stderr": "", "returncode": -1,
            "exec_time_ms": 0, "peak_memory_kb": 0,
            "file_size_bytes": file_size, "timed_out": False,
            "output_matches_expected": False, "io_score": 0.0,
        }

        proc = None
        job = None
        try:
            proc = subprocess.Popen(
                [sys.executable, once_path, self.temp_dir, self.temp_dir,
                 '--once', temp_path, result_path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.temp_dir, env=scrub_environ())
            handle = getattr(proc, '_handle', None)
            if handle is not None:
                job = assign_to_job(
                    handle, mem_bytes=self.memory_limit_mb * 1024 * 1024,
                    allow_children=True)
            rc = proc.wait(timeout=self.timeout_s)
            result["exec_time_ms"] = (time.time() - t0) * 1000
            if os.path.exists(result_path):
                import json as _json
                with open(result_path, encoding='utf-8') as f:
                    worker_result = _json.load(f)
                result["stdout"] = worker_result.get("stdout", "")
                result["stderr"] = worker_result.get("stderr", "")
                result["returncode"] = worker_result.get("returncode", rc)
            else:
                result["returncode"] = rc if rc is not None else -1
                result["stderr"] = "Sandbox worker produced no result"

            if expected_output is not None:
                # Semantic I/O match: tolerant to float formatting, ordering
                # of unordered answers, and whitespace differences.
                result["io_score"] = io_similarity(expected_output, result["stdout"])
                result["output_matches_expected"] = io_match(
                    expected_output, result["stdout"])

        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["stderr"] = f"Execution timed out after {self.timeout_s}s"
            result["exec_time_ms"] = self.timeout_s * 1000
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Sandbox execution error: %s", e)
            result["stderr"] = f"Sandbox error: {e}"
            result["exec_time_ms"] = (time.time() - t0) * 1000
        finally:
            close_job(job)
            # Cleanup
            try:
                os.unlink(temp_path)
                os.unlink(once_path)
                if os.path.exists(result_path):
                    os.unlink(result_path)
            except OSError as e:
                import warnings
                warnings.warn(f"temp file cleanup: {e}", RuntimeWarning, stacklevel=2)

        return result


class SelfPlaySandbox:
    """Self-play sandbox: model generates code → execute → log → package.

    The model generates solutions to tasks, the sandbox executes them,
    and all telemetry is logged into structured data packets for later
    knowledge injection.
    """

    # Task templates by domain
    TASK_TEMPLATES = {
        "python_basics": [
            ("Compute fibonacci({n})", "fib", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib({n}))"),
            ("Compute factorial({n})", "factorial", "def fact(n):\n    r = 1\n    for i in range(1, n+1):\n        r *= i\n    return r\nprint(fact({n}))"),
            ("Check if {n} is prime", "prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\nprint(is_prime({n}))"),
            ("Reverse the string '{s}'", "reverse", f"print('{''.join(reversed('hello'))}')"),
            ("Sum of list {lst}", "sum_list", "print(sum({lst}))"),
            ("Sort the list {lst}", "sort_list", "print(sorted({lst}))"),
            ("Count vowels in '{s}'", "vowels", "s = '{s}'\nprint(sum(1 for c in s if c in 'aeiouAEIOU'))"),
            ("Check if '{s}' is a palindrome", "palindrome", "s = '{s}'\nprint(s == s[::-1])"),
        ],
        "math": [
            ("Compute {a} * {b}", "multiply", "print({a} * {b})"),
            ("Compute {a} + {b}", "add", "print({a} + {b})"),
            ("Compute gcd({a}, {b})", "gcd", "import math\nprint(math.gcd({a}, {b}))"),
            ("Compute {a} ** {b}", "power", "print({a} ** {b})"),
            ("Is {a} divisible by {b}?", "divisible", "print({a} % {b} == 0)"),
            ("Compute sqrt of {a}", "sqrt", "print({a} ** 0.5)"),
            ("Solve: x^2 + {a}*x + {b} = 0", "quadratic", "a, b = {a}, {b}\nd = a**2 - 4*b\nprint((-a + d**0.5)/2, (-a - d**0.5)/2)"),
        ],
        "algorithms": [
            ("Implement bubble sort for {lst}", "bubble_sort", "arr = {lst}\nfor i in range(len(arr)):\n    for j in range(len(arr)-1-i):\n        if arr[j] > arr[j+1]:\n            arr[j], arr[j+1] = arr[j+1], arr[j]\nprint(arr)"),
            ("Binary search for {target} in sorted list", "binary_search", "arr = sorted([1,3,5,7,9,11,13])\ntarget = {target}\nlo, hi = 0, len(arr)-1\nwhile lo <= hi:\n    mid = (lo+hi)//2\n    if arr[mid] == target: print(mid); break\n    elif arr[mid] < target: lo = mid+1\n    else: hi = mid-1\nelse: print(-1)"),
            ("Implement a stack with push/pop", "stack", "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)\n    def pop(self): return self.items.pop() if self.items else None\ns = Stack()\ns.push(1); s.push(2)\nprint(s.pop(), s.pop())"),
        ],
        "string_manipulation": [
            ("Count words in: '{text}'", "word_count", "text = '{text}'\nprint(len(text.split()))"),
            ("Replace 'a' with 'X' in '{s}'", "replace", "print('{s}'.replace('a', 'X'))"),
            ("Uppercase '{s}'", "upper", "print('{s}'.upper())"),
            ("Find longest word in '{text}'", "longest_word", "words = '{text}'.split()\nprint(max(words, key=len) if words else '')"),
        ],
    }

    def __init__(self, model, tokenizer, log_dir: str | None = None,
                 device: str = "cuda", max_gen_tokens: int = 200,
                 temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
                 temp_dir: str | None = None, verify_workers: int = 1,
                 kv_int8: bool = False,
                 training_free: "TrainingFreeSolver | None" = None):
        if log_dir is None:
            log_dir = os.getenv("FORGE_SELF_PLAY_DIR", "research/data/self_play")
        self.model = model
        self.tokenizer = tokenizer
        # Optional forward-only adaptation (TrainingFreeSolver): when set,
        # run_task styles prompts with URIAL + Reflexion memory and records
        # every outcome so a task vector can be extracted from successful
        # runs — no weight updates, no optimizer, no backward passes.
        self.training_free = training_free
        self.device = device
        self.max_gen_tokens = max_gen_tokens
        # Sampling params for solution diversity (anti-overfitting across epochs)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # INT8 KV cache quantization: 2x KV memory reduction (near-lossless).
        # Research: KVTuner (ICML 2025), LMDeploy INT8 KV — "almost lossless".
        # Disabled automatically if the model doesn't use HF-style past_key_values.
        self.kv_int8 = kv_int8
        self._kv_dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.float16
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.executor = SandboxExecutor(temp_dir=temp_dir, workers=verify_workers)
        self.packets: list[dict] = []
        # Forward cache for repeated prompts (C2 — 20-40% fewer forward passes)
        from forge.runtime.forward_cache import ForwardCache
        self.fwd_cache = ForwardCache(max_entries=500, device=device)
        # Conformal sampler: per-query calibrated temperature (optional)
        try:
            from forge.keys.training.conformal_sampler_key import ConformalSampler
            self.conformal_sampler = ConformalSampler()
        except (ImportError, ModuleNotFoundError):
            self.conformal_sampler = None
        self.conformal_calibrated = False
        # Self-modeling: confidence-based retry (D5)
        from forge.runtime.self_model import ConfidenceScorer
        self.confidence_scorer = ConfidenceScorer()
        # Async D2H generator for single-sequence decode (reduces CPU spikes).
        self._streamed_gen = StreamedGenerator(model, device=device)
        self.stats = {
            "total_tasks": 0, "successful": 0, "failed": 0,
            "timed_out": 0, "correct": 0, "total_gen_time_ms": 0,
            "total_exec_time_ms": 0, "total_tokens_generated": 0,
        }

    def calibrate_conformal(self, calibration_prompts: list[str] | None = None,
                            alpha: float = 0.1) -> None:
        """Calibrate conformal sampler on held-out prompts.

        Runs the model on a few prompts, records confidence scores (max softmax
        prob of first generated token), then calibrates a conformal threshold.
        After calibration, generate_code() uses per-query temperature instead
        of the fixed self.temperature.

        Args:
            calibration_prompts: list of prompt strings (default: use task templates)
            alpha: miscoverage rate (0.1 = 90% coverage guarantee)
        """
        if calibration_prompts is None:
            # Use built-in task templates as calibration data
            calibration_prompts = []
            for tasks in self.TASK_TEMPLATES.values():
                for task_desc, _, _ in tasks[:3]:  # 3 per category
                    calibration_prompts.append(task_desc)

        scores = []
        self.model.eval()
        with torch.inference_mode():
            for prompt in calibration_prompts[:20]:  # cap at 20 for speed
                full_prompt = f'"""{prompt}"""\n'
                input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
                try:
                    out = self.model(input_ids, use_cache=False)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    scores.append(probs.max().item())
                except Exception as e:
                    logger.debug("conformal calibration: skipping prompt: %s", e)
                    continue

        if self.conformal_sampler and len(scores) >= 5:
            self.conformal_sampler.calibrate(scores, alpha=alpha)
            self.conformal_calibrated = True
            print(f"  [ConformalSampler] Calibrated on {len(scores)} prompts, "
                  f"threshold={self.conformal_sampler.threshold:.3f}, alpha={alpha}")
        else:
            print(f"  [ConformalSampler] Only {len(scores)} scores, skipping calibration")

    def _get_query_temperature(self, logits: torch.Tensor) -> float:
        """Get per-query temperature from conformal sampler or fall back to fixed.

        Args:
            logits: first-token logits from the model (used to compute confidence)

        Returns:
            Temperature for this query
        """
        if not self.conformal_calibrated or not self.conformal_sampler or self.temperature <= 0.0:
            return self.temperature

        # Compute query confidence: max softmax probability
        probs = torch.softmax(logits[0, -1], dim=-1)
        query_score = probs.max().item()

        # Conformal temperature: confident → low T (exploit), uncertain → high T (explore)
        return self.conformal_sampler.get_temperature(query_score)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        """Sample next token with temperature/top-k/top-p, or greedy if temp=0.

        Args:
            logits: next-token logits
            temperature: override temperature (default: use self.temperature)
        """
        temp = temperature if temperature is not None else self.temperature
        if temp <= 0.0:
            return logits.argmax()

        scaled = logits / temp
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        if 0.0 < self.top_p < 1.0:
            from research.sampling_utils import top_p_sample_probs
            probs = top_p_sample_probs(probs, self.top_p)

        return torch.multinomial(probs, num_samples=1).squeeze()

    def generate_code(self, prompt: str) -> tuple[str, dict]:
        """Generate code from a prompt using the model.

        Uses KV cache for O(n) generation instead of O(n²) reprocessing.
        Returns (generated_text, model_telemetry).
        """
        # If the prompt is already a code stub (contains def/class/import),
        # use it directly for code completion. Otherwise wrap in docstring.
        if any(kw in prompt for kw in ["def ", "class ", "import ", "from "]):
            # Add stop instruction + efficiency hint
            full_prompt = prompt + "# Write concise, efficient code. End with [stop]\n    "
        else:
            full_prompt = f'"""{prompt}"""\n'
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids.shape[1]

        t0 = time.time()
        log_probs = []

        # Token IDs for stopping
        newline_token = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_token[0] if newline_token else 198

        # Build stop string tokens — check for "[stop]" in generated text
        stop_str = "[stop]"

        # ── KV-cached generation (O(n) instead of O(n²)) ──────────
        # First pass: process the full prompt, cache KV pairs.
        # Subsequent passes: feed only the new token, reuse cached KV.
        with torch.inference_mode():
            generated_ids = []
            eos_id = self.tokenizer.eos_token_id

            # Pre-allocate KV cache for O(1) append per token.
            from forge.model_loader import create_kv_cache
            cache = create_kv_cache(
                self.model, input_ids.shape[1] + self.max_gen_tokens,
                batch=1, device=self.device,
            )

            # Initial forward pass on the full prompt
            logits, _ = self.model(input_ids, preallocated_cache=cache)
            next_logits = logits[0, -1]

            # Conformal sampler: per-query temperature from first-token confidence
            query_temp = self._get_query_temperature(logits)

            next_token = self._sample_next_token(next_logits, temperature=query_temp)

            # Async D2H: issue non-blocking copy instead of blocking .item().
            # The copy runs on the DMA engine while the GPU starts the next forward.
            reader = AsyncTokenReader(1, next_logits.shape[-1], self.device)
            reader.issue(next_token.unsqueeze(0), next_logits.unsqueeze(0))

            # Pre-allocate a single-token buffer to avoid per-step tensor creation.
            cur_token = torch.zeros(1, 1, dtype=input_ids.dtype, device=self.device)

            # Sync once to get the first token (needed for EOS check + first output).
            token_list = reader.get_tokens()
            lp_list = reader.get_logprobs()
            token_id = token_list[0]
            log_probs.append(lp_list[0])
            generated_ids.append(token_id)

            # Stop on EOS or [stop] after first token
            stop = False
            if token_id == eos_id:
                stop = True
            else:
                gen_text = self.tokenizer.decode(
                    torch.tensor(generated_ids), skip_special_tokens=True)
                if stop_str in gen_text:
                    stop = True

            # Subsequent tokens: feed only the last token, reuse pre-allocated KV cache.
            # Async D2H pattern: issue copy for step N, launch forward for step N+1,
            # then sync to get step N's result. The GPU works while the copy runs.
            while not stop and len(generated_ids) < self.max_gen_tokens:
                cur_token[0, 0] = next_token
                # Launch forward (GPU starts working immediately).
                # Use compiled decode step (CUDA Graphs) if available, else main model.
                if hasattr(self.model, '_compiled_decode') and self.model._compiled_decode is not None:
                    logits, _ = self.model._compiled_decode(
                        cur_token, preallocated_cache=cache)
                else:
                    logits, _ = self.model(cur_token, preallocated_cache=cache)
                next_logits = logits[0, -1]
                next_token = self._sample_next_token(next_logits, temperature=query_temp)

                # Issue async D2H for this step's token + log-prob (non-blocking).
                reader.issue(next_token.unsqueeze(0), next_logits.unsqueeze(0))

                # Sync to get the token (by now GPU has finished the copy — near-free).
                token_list = reader.get_tokens()
                lp_list = reader.get_logprobs()
                token_id = token_list[0]
                log_probs.append(lp_list[0])
                generated_ids.append(token_id)

                if token_id == eos_id:
                    break

                # Check for [stop] every few tokens (cheaper than every token)
                if len(generated_ids) % 4 == 0:
                    gen_text = self.tokenizer.decode(
                        torch.tensor(generated_ids), skip_special_tokens=True)
                    if stop_str in gen_text:
                        break

            # Free KV cache + logits aggressively (prevents VRAM accumulation
            # across repeated generate_code calls in the same process).
            del cache, logits, next_logits, next_token, cur_token
            with contextlib.suppress(NameError):
                del reader
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        gen_time_ms = (time.time() - t0) * 1000
        tokens_generated = len(generated_ids)

        generated_text = self.tokenizer.decode(
            torch.tensor(generated_ids), skip_special_tokens=True)

        # Extract code — strip [stop] marker and instruction comment
        code = generated_text.strip()
        # Remove [stop] and everything after it
        if stop_str in code:
            code = code[:code.index(stop_str)].strip()
        # Remove the instruction comment line if present
        code = code.replace("# Write concise, efficient code. End with [stop]\n", "")
        # Strip markdown fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)

        # If the code doesn't contain print(), add it for tasks that need output
        # (This helps with base models that write functions but don't call them)

        telemetry = {
            "gen_time_ms": gen_time_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / (gen_time_ms / 1000) if gen_time_ms > 0 else 0,
            "mean_logprob": sum(log_probs) / max(len(log_probs), 1),
            "confidence": pow(2.71828, sum(log_probs) / max(len(log_probs), 1)),
            "query_temperature": query_temp if self.conformal_calibrated else self.temperature,
            "conformal": self.conformal_calibrated,
        }

        return code, telemetry

    def _safe_batch_size(self, requested: int, max_tokens: int) -> int:
        """Estimate a VRAM-safe batch size for batched generation.

        Uses the actual model architecture (layers, KV heads, head_dim) to
        compute precise KV cache cost per sequence, then probes free VRAM.
        Accounts for INT8 KV quantization (2x reduction) when enabled.

        KV cache formula (per sequence, per token):
          2 × n_layers × n_kv_heads × head_dim × bytes_per_elem
        The factor 2 accounts for K and V. For GQA, n_kv_heads < n_heads.
        """
        try:
            if not torch.cuda.is_available():
                return requested
            free_bytes = torch.cuda.mem_get_info()[0]
            cfg = self.model.config
            n_layers = getattr(cfg, 'n_layers', 24)
            n_kv_heads = getattr(cfg, 'n_kv_heads', cfg.n_heads) if hasattr(cfg, 'attn_type') and cfg.attn_type == "gqa" else getattr(cfg, 'n_heads', 12)
            head_dim = getattr(cfg, 'd_model', 2048) // getattr(cfg, 'n_heads', 12)
            # INT8 KV quantization halves the bytes per element (2→1)
            bytes_per_elem = 1 if self.kv_int8 else 2
            kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem
            est_seq_len = 256 + max_tokens
            kv_per_seq = kv_bytes_per_token * est_seq_len
            # 20% safety margin for activations, logits, attention matrices
            kv_per_seq = int(kv_per_seq * 1.2)
            if kv_per_seq <= 0:
                return requested
            usable = max(0, free_bytes - 512 * 1024 * 1024)
            safe = max(1, usable // kv_per_seq)
            return min(requested, safe)
        except Exception:
            try:
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                kv_per_seq_mb = max_tokens * 0.08 * 1.3
                safe = max(1, int(free_mb / kv_per_seq_mb))
                return min(requested, safe)
            except Exception:
                return requested

    def _quantize_kv_cache_int8(self, past_kvs):
        """Quantize HF-style past_key_values to INT8 using KIVI-style approach.

        Based on KIVI paper (ICML 2024) + HuggingFace QuantizedCache:
        - Keys: per-channel quantization (along head_dim) — keys have channel outliers
        - Values: per-token quantization (along seq_len) — values are smoother
        - Residual buffer: keep last N tokens in fp16, only quantize older tokens
        - Attention sinks: never quantize first 4 tokens

        past_kvs is a list of (K, V) tuples per layer, each (B, n_heads, T, head_dim).
        Returns a list with mixed fp16 (recent) + int8 (old) tensors per layer.
        """
        if not self.kv_int8 or past_kvs is None:
            return past_kvs
        residual_len = 32  # keep last 32 tokens in fp16 (HF default)
        sink_len = 4       # never quantize first 4 tokens (attention sinks)
        try:
            new_kvs = []
            for layer_kvs in past_kvs:
                if layer_kvs is None:
                    new_kvs.append(layer_kvs)
                    continue
                k, v = layer_kvs
                T = k.shape[2]  # sequence length
                # Only quantize if we have enough tokens beyond sinks + residual
                if sink_len + residual_len >= T:
                    new_kvs.append((k, v))  # all in fp16
                    continue
                # Split: [sinks | quantizable | residual]
                k_sink, k_mid, k_res = k[:, :, :sink_len], k[:, :, sink_len:T-residual_len], k[:, :, T-residual_len:]
                v_sink, v_mid, v_res = v[:, :, :sink_len], v[:, :, sink_len:T-residual_len], v[:, :, T-residual_len:]
                # Per-channel key quantization: scale along seq_len dim (dim=2)
                # shape: (B, n_heads, 1, head_dim)
                k_scale = k_mid.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
                k_q = torch.round(k_mid / k_scale).clamp(-128, 127).to(torch.int8)
                # Per-token value quantization: scale along head_dim (dim=-1)
                # shape: (B, n_heads, T_mid, 1)
                v_scale = v_mid.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
                v_q = torch.round(v_mid / v_scale).clamp(-128, 127).to(torch.int8)
                # Store scales as tuples alongside the tensors
                new_kvs.append(((k_sink, k_q, k_res, k_scale),
                                (v_sink, v_q, v_res, v_scale)))
            return new_kvs
        except Exception:
            return past_kvs  # fallback: keep fp16

    def _dequantize_kv_cache(self, past_kvs):
        """Dequantize INT8 KV cache back to float for the forward pass.

        Reconstructs full fp16 K/V by concatenating sinks + dequantized mid + residual.
        Only called before model.forward() — the model expects contiguous float tensors.
        """
        if not self.kv_int8 or past_kvs is None:
            return past_kvs
        try:
            new_kvs = []
            for layer_kvs in past_kvs:
                if layer_kvs is None:
                    new_kvs.append(layer_kvs)
                    continue
                k_pack, v_pack = layer_kvs
                # Check if this layer is quantized (tuple of 4) or plain fp16
                if isinstance(k_pack, tuple):
                    k_sink, k_q, k_res, k_scale = k_pack
                    v_sink, v_q, v_res, v_scale = v_pack
                    dtype = self._kv_dtype if hasattr(self, '_kv_dtype') else torch.float16
                    # Dequantize: int8 → float, multiply by scale
                    k_mid = k_q.to(dtype) * k_scale.to(dtype)
                    v_mid = v_q.to(dtype) * v_scale.to(dtype)
                    # Concatenate: sinks + mid + residual
                    k_full = torch.cat([k_sink, k_mid, k_res], dim=2)
                    v_full = torch.cat([v_sink, v_mid, v_res], dim=2)
                    new_kvs.append((k_full, v_full))
                else:
                    # Plain fp16 tensor (not enough tokens to quantize)
                    new_kvs.append((k_pack, v_pack))
            return new_kvs
        except Exception:
            return past_kvs

    def generate_code_batch(self, prompts: list[str], raw: bool = False,
                            stop_markers: list[str] | None = None,
                            max_tokens: int | None = None) -> list[tuple[str, dict]]:
        """Generate code for multiple prompts in a single batched forward pass.

        Uses left-padding with proper attention_mask and position_ids to ensure
        pad tokens don't corrupt generation. Each sequence generates independently,
        stopping on EOS or [stop].

        Args:
            prompts: task prompts (or full prompts when raw=True)
            raw: use prompts verbatim (no instruction suffix / docstring wrap)
            stop_markers: extra substrings that end a sequence (checked every
                4 tokens, same cadence as [stop]); completion is truncated at
                the first marker occurrence
            max_tokens: per-call override of self.max_gen_tokens

        Returns: list of (code, telemetry) tuples, same length as prompts.
        """
        if len(prompts) == 1 and not raw:
            return [self.generate_code(prompts[0])]

        # VRAM-aware batch splitting: batched generation holds B KV caches
        # simultaneously. On 12GB GPUs, B=8 with 200 tokens can OOM. Estimate
        # safe batch size from free VRAM and fall back to sequential sub-batches.
        B = len(prompts)
        max_gen = max_tokens or self.max_gen_tokens
        safe_batch = self._safe_batch_size(B, max_gen)
        if safe_batch < B:
            # Process in sub-batches to stay within VRAM
            results: list[tuple[str, dict]] = []
            for i in range(0, B, safe_batch):
                sub = prompts[i:i + safe_batch]
                results.extend(self.generate_code_batch(
                    sub, raw=raw, stop_markers=stop_markers, max_tokens=max_tokens))
            return results

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id or 0
        stop_str = "[stop]"
        extra_stops = [m for m in (stop_markers or []) if m]

        # Build full prompts (same logic as generate_code)
        if raw:
            full_prompts = list(prompts)
        else:
            full_prompts = []
            for prompt in prompts:
                if any(kw in prompt for kw in ["def ", "class ", "import ", "from "]):
                    full_prompts.append(prompt + "# Write concise, efficient code. End with [stop]\n    ")
                else:
                    full_prompts.append(f'"""{prompt}"""\n')

        # Tokenize all prompts
        all_ids = []
        for fp in full_prompts:
            enc = self.tokenizer(fp, return_tensors="pt", truncation=True, max_length=512)
            all_ids.append(enc.input_ids[0])

        # Left-pad to max prompt length
        max_prompt_len = max(ids.shape[0] for ids in all_ids)
        padded = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=self.device)
        # Position IDs: pad tokens get position 0, real tokens get 0..T-1
        # (left-padded: real tokens are at the end, so positions count from
        # the start of the real sequence, not the start of the padded row)
        position_ids = torch.zeros(B, max_prompt_len, dtype=torch.long, device=self.device)
        for i, ids in enumerate(all_ids):
            t = ids.shape[0]
            padded[i, max_prompt_len - t:] = ids.to(self.device)
            attention_mask[i, max_prompt_len - t:] = 1
            # Real tokens get positions 0..t-1 (aligned to the right side)
            position_ids[i, max_prompt_len - t:] = torch.arange(t, device=self.device)

        t0 = time.time()
        all_generated = [[] for _ in range(B)]
        all_log_probs = [[] for _ in range(B)]

        with torch.inference_mode():
            # Prefill: all prompts at once. The model computes position_ids
            # internally from attention_mask (cumsum-1 for left-padding).
            logits, _, past_kvs = self.model(
                padded, past_key_values=None, use_cache=True, attention_mask=attention_mask)
            next_logits = logits[:, -1]  # (B, vocab)

            # Sample first token for each sequence
            next_tokens = self._sample_next_token_batch(next_logits)  # (B,)

            # Async D2H: issue non-blocking copy for all B tokens at once.
            reader = AsyncTokenReader(B, next_logits.shape[-1], self.device)
            reader.issue(next_tokens, next_logits)

            # Sync once for first token (needed for EOS check).
            token_list = reader.get_tokens()
            lp_list = reader.get_logprobs()
            for b in range(B):
                all_log_probs[b].append(lp_list[b])
                all_generated[b].append(token_list[b])

            # Decode loop
            finished = [tid == eos_id for tid in token_list]
            step = 0
            # Pre-allocate attention_mask buffer to avoid torch.cat per step (CPU spike fix).
            max_total = max_prompt_len + max_gen
            mask_buf = torch.zeros(B, max_total, dtype=torch.long, device=self.device)
            mask_buf[:, :max_prompt_len] = attention_mask
            while step < max_gen and not all(finished):
                cur = next_tokens.unsqueeze(-1)  # (B, 1)
                # Update mask buffer in-place (O(1) write vs O(n) torch.cat).
                mask_buf[:, max_prompt_len + step] = 1
                # Launch forward (GPU starts working immediately).
                logits, _, past_kvs = self.model(
                    cur, past_key_values=past_kvs, use_cache=True,
                    attention_mask=mask_buf[:, :max_prompt_len + step + 1])
                next_logits = logits[:, -1]
                next_tokens = self._sample_next_token_batch(next_logits)

                # Async D2H: non-blocking copy of tokens + log-probs.
                reader.issue(next_tokens, next_logits)

                # Sync to get tokens (GPU has finished copy by now — near-free).
                token_list = reader.get_tokens()
                lp_list = reader.get_logprobs()
                for b in range(B):
                    if not finished[b]:
                        all_log_probs[b].append(lp_list[b])
                        all_generated[b].append(token_list[b])
                        if token_list[b] == eos_id:
                            finished[b] = True

                # Check for [stop] + extra markers every 4 tokens
                if step % 4 == 3:
                    for b in range(B):
                        if not finished[b]:
                            gen_text = self.tokenizer.decode(
                                torch.tensor(all_generated[b]), skip_special_tokens=True)
                            if stop_str in gen_text or any(m in gen_text for m in extra_stops):
                                finished[b] = True

                # Early KV cleanup: zero out KV entries for finished sequences.
                # This frees VRAM mid-batch so remaining sequences can continue
                # without OOM. The finished sequence's output is already captured
                # in all_generated[], so its KV is no longer needed.
                if any(finished) and past_kvs is not None:
                    for _layer_idx, layer_kvs in enumerate(past_kvs):
                        if layer_kvs is None:
                            continue
                        k, v = layer_kvs
                        for b in range(B):
                            if finished[b]:
                                k[b] = 0
                                v[b] = 0

                step += 1

            # Aggressive VRAM cleanup: delete KV cache + logits immediately
            # after generation, before decode/post-process. This frees the
            # GPU memory for the next sub-batch or verification step.
            del past_kvs, next_logits, next_tokens, logits, cur
            del mask_buf
            with contextlib.suppress(NameError):
                del reader
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        gen_time_ms = (time.time() - t0) * 1000

        # Decode and post-process each sequence
        results = []
        for b in range(B):
            tokens_generated = len(all_generated[b])
            generated_text = self.tokenizer.decode(
                torch.tensor(all_generated[b]), skip_special_tokens=True)
            code = generated_text.strip()
            if stop_str in code:
                code = code[:code.index(stop_str)].strip()
            for m in extra_stops:
                if m and m in code:
                    code = code[:code.index(m)].strip()
            code = code.replace("# Write concise, efficient code. End with [stop]\n", "")
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                code = "\n".join(lines)

            lps = all_log_probs[b]
            telemetry = {
                "gen_time_ms": gen_time_ms / B,  # amortized
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / (gen_time_ms / 1000) if gen_time_ms > 0 else 0,
                "mean_logprob": sum(lps) / max(len(lps), 1),
                "confidence": pow(2.71828, sum(lps) / max(len(lps), 1)),
                "query_temperature": self.temperature,
                "conformal": False,
            }
            results.append((code, telemetry))

        return results

    def _sample_next_token_batch(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next tokens for a batch. logits: (B, vocab) -> (B,).

        Each sequence in the batch gets a different random seed so identical
        prompts produce different outputs (prevents response cloning).
        """
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[-1]:
            topk_vals, _ = torch.topk(probs, self.top_k, dim=-1)
            mask = probs < topk_vals[:, -1:]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            keep = cumsum <= self.top_p
            keep[..., 0] = True
            probs = torch.zeros_like(probs, dtype=probs.dtype)
            probs.scatter_(-1, sorted_idx, sorted_probs * keep.to(probs.dtype))
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Per-sequence sampling with different seeds to prevent cloning.
        # Each row b gets its own generator seeded with (base_seed + b + step)
        # so identical prompts in the same batch produce different tokens.
        B = probs.shape[0]
        out = torch.empty(B, dtype=torch.long, device=probs.device)
        for b in range(B):
            # Seed per-sequence: mix global RNG state + batch index for diversity
            seed = int(torch.randint(0, 2**31 - 1, (1,), device=probs.device).item())
            gen = torch.Generator(device=probs.device)
            gen.manual_seed(seed)
            out[b] = torch.multinomial(probs[b], num_samples=1, generator=gen).squeeze()
        return out

    def generate_reasoning(self, prompt: str) -> str:
        """Generate chain-of-thought reasoning before code.

        Prompts the model to think about the problem first,
        producing a reasoning trace that improves code quality.
        """
        # Extract function name from prompt
        match = _RE_DEF_FUNC.search(prompt)
        func_name = match.group(1) if match else "this function"
        match_cls = _RE_DEF_CLASS.search(prompt)
        cls_name = match_cls.group(1) if match_cls else None
        target = cls_name or func_name

        reasoning_prompt = (
            f"# Think step by step about implementing {target}.\n"
            f"# 1. What does it do?\n"
            f"# 2. What are the edge cases?\n"
            f"# 3. What approach will you use?\n"
            f"# "
        )
        ids = self.tokenizer(reasoning_prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            gen_ids = []
            eos_id = self.tokenizer.eos_token_id

            # Pre-allocated KV cache
            from forge.model_loader import create_kv_cache
            cache = create_kv_cache(
                self.model, ids.shape[1] + 60, batch=1, device=self.device,
            )

            # Initial pass on full prompt
            logits, _ = self.model(ids, preallocated_cache=cache)
            nxt = self._sample_next_token(logits[0, -1])

            # Async D2H for reasoning generation.
            reader = AsyncTokenReader(1, logits.shape[-1], self.device)
            reader.issue(nxt.unsqueeze(0), logits[0, -1].unsqueeze(0))
            token_list = reader.get_tokens()
            token_id = token_list[0]
            gen_ids.append(token_id)

            cur_tok = torch.zeros(1, 1, dtype=ids.dtype, device=self.device)
            for step in range(60):  # short reasoning
                if token_id == eos_id:
                    break
                # Feed only the new token with cached KV
                cur_tok[0, 0] = nxt
                logits, _ = self.model(cur_tok, preallocated_cache=cache)
                nxt = self._sample_next_token(logits[0, -1])
                # Async D2H (non-blocking).
                reader.issue(nxt.unsqueeze(0), logits[0, -1].unsqueeze(0))
                token_list = reader.get_tokens()
                token_id = token_list[0]
                gen_ids.append(token_id)
                # Stop at double newline (end of reasoning)
                if step > 10 and step % 4 == 0:
                    decoded = self.tokenizer.decode(
                        torch.tensor(gen_ids), skip_special_tokens=True)
                    if "\n\n" in decoded:
                        break

            del cache

        reasoning = self.tokenizer.decode(
            torch.tensor(gen_ids), skip_special_tokens=True).strip()
        # Clean up — keep only the reasoning lines (starting with #)
        lines = reasoning.split("\n")
        reasoning_lines = [l for l in lines if l.strip().startswith("#") or l.strip() == ""]
        return "\n".join(reasoning_lines).strip()

    def generate_with_reasoning(self, prompt: str) -> tuple[str, str, dict]:
        """Generate reasoning + code. Returns (reasoning, code, telemetry)."""
        reasoning = self.generate_reasoning(prompt)
        # Prepend reasoning as comments to the code prompt
        if reasoning:
            enhanced_prompt = reasoning + "\n" + prompt
        else:
            enhanced_prompt = prompt
        code, telemetry = self.generate_code(enhanced_prompt)
        return reasoning, code, telemetry

    def self_critique(self, code: str, prompt: str) -> tuple[str, list[dict]]:
        """Self-critique loop: generate edge case tests, run them, fix bugs.

        Returns (improved_code, test_results).
        """
        match = _RE_DEF_FUNC.search(prompt)
        func_name = match.group(1) if match else None

        if not func_name:
            return code, []

        # Generate edge case test calls
        critique_prompt = (
            f"{code}\n\n"
            f"# Write test calls for {func_name} with edge cases.\n"
            f"# Test: empty input, negative, zero, large, typical.\n"
            f"# Format: assert {func_name}(...) == expected\n"
            f"assert "
        )
        ids = self.tokenizer(critique_prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            cur_ids = ids
            for step in range(80):
                logits, _ = self.model(cur_ids)
                nxt = logits[0, -1].argmax()
                cur_ids = torch.cat([cur_ids, nxt.unsqueeze(0).unsqueeze(0)], dim=1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break
                decoded = self.tokenizer.decode(cur_ids[0, ids.shape[1]:],
                                                skip_special_tokens=True)
                if "\n\n" in decoded and step > 10:
                    break

        test_text = self.tokenizer.decode(cur_ids[0, ids.shape[1]:],
                                          skip_special_tokens=True).strip()
        # Extract assert statements
        test_lines = [l.strip() for l in test_text.split("\n") if l.strip().startswith("assert")]

        # Run the tests — through the sandboxed executor, never in-process:
        # `code` and `test_lines` are both raw model output (EC-sec).
        test_results = []
        all_code = code + "\n" + "\n".join(test_lines)
        res = self.executor.execute(all_code)
        if res.get("returncode", -1) == 0:
            test_results.append({"passed": True, "tests": len(test_lines)})
        else:
            test_results.append({"passed": False,
                                 "error": res.get("stderr", "execution failed"),
                                 "tests": len(test_lines)})

        return code, test_results

    def run_task(self, task_prompt: str, expected_output: str | None = None,
                 task_type: str = "python", task_name: str = "",
                 reference_code: str | None = None,
                 code_prefix: str = "") -> dict:
        """Run a single self-play task: generate → execute → log.

        Args:
            task_prompt: the task description
            expected_output: expected stdout for correctness check
            task_type: "python", "math", "algorithm"
            task_name: short name for the task
            reference_code: known-correct reference solution
            code_prefix: code to prepend (function signature, imports)

        Returns:
            Complete data packet with all telemetry
        """
        self.stats["total_tasks"] += 1

        # Phase 1: Generate code (optionally through the training-free
        # adapter: URIAL styling + Reflexion memory of past attempts).
        if self.training_free is not None:
            task_prompt = self.training_free.style(task_prompt)

        full_prompt = task_prompt
        if code_prefix:
            full_prompt = f"{code_prefix}\n# {task_prompt}"

        code, model_telem = self.generate_code(full_prompt)

        # Prepend the prefix to the generated code
        if code_prefix:
            code = code_prefix + "\n" + code

        # Auto-append print() call if code defines a function but doesn't call it
        # (base models often write functions without calling them)
        if "def " in code and "print(" not in code and expected_output is not None:
            # Try to extract function name and call it
            func_match = _RE_FUNC_SIG.search(code)
            if func_match:
                func_name = func_match.group(1)
                args = func_match.group(2)
                # Try to call with values from the prompt
                # Extract numbers from the prompt
                numbers = _RE_DIGITS.findall(task_prompt)
                if numbers:
                    call_args = ", ".join(numbers[:len(args.split(',')) if args else 0])
                    code += f"\nprint({func_name}({call_args}))"

        # Phase 2: Execute in sandbox
        exec_result = self.executor.execute(code, expected_output)

        # Phase 3: Build data packet
        packet = {
            "task": task_name or task_prompt[:50],
            "task_type": task_type,
            "prompt": task_prompt,
            "generated_code": code,
            "reference_code": reference_code,
            "expected_output": expected_output,
            "execution": exec_result,
            "model_telemetry": model_telem,
            "timestamp": datetime.now().isoformat(),
        }

        # Phase 4: Score the result
        quality_score = self._score_packet(packet)
        packet["quality_score"] = quality_score

        # Phase 5: Build knowledge text (for later injection)
        if exec_result["returncode"] == 0 and exec_result["stdout"].strip():
            packet["knowledge_text"] = (
                f"Task: {task_prompt}\n"
                f"Solution: {code[:200]}\n"
                f"Output: {exec_result['stdout'].strip()[:100]}\n"
                f"Execution time: {exec_result['exec_time_ms']:.1f}ms\n"
                f"Correct: {exec_result['output_matches_expected']}"
            )
        else:
            packet["knowledge_text"] = ""

        # Update stats
        if exec_result["timed_out"]:
            self.stats["timed_out"] += 1
        elif exec_result["returncode"] == 0:
            self.stats["successful"] += 1
            if exec_result["output_matches_expected"]:
                self.stats["correct"] += 1
        else:
            self.stats["failed"] += 1

        self.stats["total_gen_time_ms"] += model_telem["gen_time_ms"]
        self.stats["total_exec_time_ms"] += exec_result["exec_time_ms"]
        self.stats["total_tokens_generated"] += model_telem["tokens_generated"]

        # Training-free feedback: push success/failure into Reflexion memory
        # (+ collect activations for later task-vector extraction).
        if self.training_free is not None:
            ok = (exec_result.get("returncode") == 0
                  and exec_result.get("output_matches_expected", False))
            error = "" if ok else (exec_result.get("stderr", "")
                                   or "output mismatch")
            self.training_free.record(
                packet["prompt"], output=code, error=error, success=ok)

        self.packets.append(packet)
        return packet

    # ── Training-free adapters (no weight updates) ────────────────

    def enable_training_free(self, max_tokens: int | None = None,
                             capture_activations: bool = True,
                             memory_kwargs: dict | None = None):
        """Lazily build a TrainingFreeSolver for this sandbox.

        From here on, run_task styles prompts (URIAL + Reflexion memory) and
        records every outcome — success/failure feeds the in-context buffer
        and the activation sets used for task vectors.
        """
        from research.training_free import TrainingFreeSolver
        if self.training_free is None:
            self.training_free = TrainingFreeSolver(
                self.model, self.tokenizer, device=self.device,
                max_tokens=max_tokens or self.max_gen_tokens,
                capture_activations=capture_activations,
                memory_kwargs=memory_kwargs,
            )
        return self.training_free

    def build_task_vector(self, normalize: bool = True) -> dict:
        """Extract the success-minus-failure task vector from recorded runs."""
        if self.training_free is None:
            return {}
        return self.training_free.build_task_vector(normalize=normalize)

    def apply_steering(self, alpha: float = 1.0) -> bool:
        """Apply the task vector at strength alpha (hooks, no weight edits)."""
        if self.training_free is None:
            return False
        return self.training_free.apply_steering(alpha=alpha)

    def clear_steering(self) -> None:
        """Remove steering hooks — back to baseline behavior."""
        if self.training_free is not None:
            self.training_free.clear_steering()

    def _score_packet(self, packet: dict) -> float:
        """Score a data packet on a 0-1 scale (I/O-focused weighting).

        Factors (revamped for input-output focus — correctness dominates):
          - I/O correctness (60%): semantic io_score (tolerant to float
            formatting / unordered answers), not just exact string match
          - Execution success (15%): did it run without errors?
          - Speed (10%): faster execution = higher score
          - Code quality (5%): shorter code = higher score (efficiency)
          - Model confidence (5%): mean logprob
          - No timeout (5%): didn't time out
        """
        exec_r = packet["execution"]
        model_t = packet["model_telemetry"]

        score = 0.0

        # I/O correctness (60%) — semantic match gives partial credit
        io_score = exec_r.get("io_score")
        if io_score is None:
            # Legacy packets without io_score: fall back to boolean match
            io_score = 1.0 if exec_r["output_matches_expected"] else 0.0
        score += io_score * 0.6
        if io_score < 0.99 and exec_r["returncode"] == 0 and exec_r["stdout"].strip():
            score += 0.05  # ran and produced output, but wrong answer

        # Execution success (15%)
        if exec_r["returncode"] == 0:
            score += 0.15

        # Speed (10%) — faster is better, cap at 1s
        exec_time = exec_r["exec_time_ms"]
        speed_score = max(0, 1.0 - exec_time / 1000.0) * 0.10
        score += speed_score

        # Code quality (5%) — shorter code is more efficient
        code_len = len(packet["generated_code"])
        quality = max(0, 1.0 - code_len / 500.0) * 0.05
        score += quality

        # Model confidence (5%)
        conf = model_t.get("confidence", 0)
        score += min(conf, 1.0) * 0.05

        # No timeout (5%)
        if not exec_r["timed_out"]:
            score += 0.05

        return round(score, 4)

    def run_domain(self, domain: str, n_tasks: int = 10) -> list[dict]:
        """Run multiple tasks from a domain (python_basics, math, algorithms, etc.).

        Generates task parameters randomly and runs each through the sandbox.
        """
        import random
        templates = self.TASK_TEMPLATES.get(domain, [])
        if not templates:
            print(f"  Unknown domain: {domain}")
            return []

        packets = []
        for i in range(n_tasks):
            template = random.choice(templates)
            task_desc, task_name, reference_code = template

            # Generate random parameters
            n = random.randint(3, 20)
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            s = random.choice(["hello", "world", "python", "racecar", "madam"])
            lst = [random.randint(1, 100) for _ in range(random.randint(3, 8))]
            target = random.randint(1, 13)
            text = " ".join(random.choice(["the", "quick", "brown", "fox",
                                            "jumps", "over", "lazy", "dog"])
                            for _ in range(random.randint(3, 6)))

            # Format the task prompt
            try:
                prompt = task_desc.format(n=n, a=a, b=b, s=s, lst=lst,
                                          target=target, text=text)
            except KeyError:
                prompt = task_desc

            # Compute expected output from reference code
            expected = None
            if reference_code:
                try:
                    ref_code = reference_code.format(n=n, a=a, b=b, s=s, lst=lst,
                                                      target=target, text=text)
                    ref_result = self.executor.execute(ref_code)
                    if ref_result["returncode"] == 0:
                        expected = ref_result["stdout"].strip()
                except Exception as e:
                    import warnings
                    warnings.warn(f"confidence scoring: {e}", RuntimeWarning, stacklevel=2)

            print(f"\n  [{domain} {i+1}/{n_tasks}] {prompt[:60]}")

            # Build code prefix to help the model (function signature style)
            code_prefix = ""
            if domain == "math":
                code_prefix = "import math"
            elif domain == "algorithms":
                code_prefix = ""

            packet = self.run_task(prompt, expected_output=expected,
                                   task_type=domain, task_name=task_name,
                                   reference_code=reference_code,
                                   code_prefix=code_prefix)

            status = "✓" if packet["execution"]["output_matches_expected"] else \
                     "○" if packet["execution"]["returncode"] == 0 else "✗"
            print(f"    {status} score={packet['quality_score']:.2f} "
                  f"exec={packet['execution']['exec_time_ms']:.0f}ms "
                  f"gen={packet['model_telemetry']['gen_time_ms']:.0f}ms")
            if packet["execution"]["stdout"].strip():
                print(f"    Output: {packet['execution']['stdout'].strip()[:80]}")
            if packet["execution"]["stderr"].strip():
                print(f"    Error: {packet['execution']['stderr'].strip()[:80]}")

            packets.append(packet)

        return packets

    def save_packets(self, filename: str | None = None, max_logs: int = 10) -> str:
        """Save all data packets to a JSONL file for later knowledge injection.

        Implements log rotation: keeps only the `max_logs` most recent JSONL
        files in the log directory, deleting older ones.

        Args:
            filename: custom filename (default: timestamped)
            max_logs: maximum number of JSONL files to retain (default 10)
        """
        if filename is None:
            filename = f"self_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        path = self.log_dir / filename

        with open(path, 'w', encoding='utf-8') as f:
            for packet in self.packets:
                f.write(dumps(packet, ensure_ascii=False) + "\n")

        # Log rotation: remove oldest JSONL files beyond max_logs.
        # Single sorted glob + bulk unlink; failures are logged, not silent.
        try:
            jsonl_files = sorted(
                self.log_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_file in jsonl_files[max_logs:]:
                try:
                    old_file.unlink()
                except OSError as e:
                    logger.warning("log rotation: cannot delete %s: %s", old_file, e)
        except OSError as e:
            logger.warning("log rotation failed (best-effort): %s", e)

        n_saved = len(self.packets)
        # Free accumulated packets — without this, long self-play runs leak
        # memory (each packet holds full code + reasoning + telemetry).
        self.packets.clear()

        print(f"\n  Saved {n_saved} packets to {path}")
        return str(path)

    def get_stats(self) -> dict:
        """Return summary statistics."""
        s = self.stats
        s["success_rate"] = s["successful"] / max(s["total_tasks"], 1)
        s["correct_rate"] = s["correct"] / max(s["total_tasks"], 1)
        s["avg_gen_time_ms"] = s["total_gen_time_ms"] / max(s["total_tasks"], 1)
        s["avg_exec_time_ms"] = s["total_exec_time_ms"] / max(s["total_tasks"], 1)
        s["avg_tokens_per_second"] = (
            s["total_tokens_generated"] / max(s["total_gen_time_ms"] / 1000, 0.001)
        )
        return s

    def print_stats(self):
        """Print summary statistics."""
        s = self.get_stats()
        print(f"\n{'='*60}")
        print("Self-Play Sandbox Statistics")
        print(f"{'='*60}")
        print(f"  Total tasks:      {s['total_tasks']}")
        print(f"  Successful runs:  {s['successful']} ({s['success_rate']:.1%})")
        print(f"  Correct outputs:  {s['correct']} ({s['correct_rate']:.1%})")
        print(f"  Failed runs:      {s['failed']}")
        print(f"  Timed out:        {s['timed_out']}")
        print(f"  Avg gen time:     {s['avg_gen_time_ms']:.0f}ms")
        print(f"  Avg exec time:    {s['avg_exec_time_ms']:.0f}ms")
        print(f"  Avg tokens/sec:   {s['avg_tokens_per_second']:.0f}")
        print(f"  Packets logged:   {len(self.packets)}")
        print(f"{'='*60}")


def main():
    """Run self-play sandbox with ForgeLM."""
    from forge.config import get_config
    from forge.model_loader import ModelLoader
    from research.tokenizer_cache import get_tokenizer

    print("=" * 60)
    print("Self-Play Sandbox — ForgeLM")
    print("=" * 60)

    # Load model
    print("\n[1] Loading ForgeLM V10 (IRI-FP4 + BitNet QAT + TITAN + MoD)...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/ForgeLM_V2.safetensors")
    model.to("cuda").eval()
    tokenizer = get_tokenizer("research/checkpoints/forgelm_v2_tokenizer")

    # Create sandbox
    print("\n[2] Creating sandbox...")
    sandbox = SelfPlaySandbox(model, tokenizer,
                              log_dir="research/data/self_play",
                              max_gen_tokens=100)

    # Run domains
    print("\n[3] Running self-play tasks...")

    print("\n--- Python Basics ---")
    sandbox.run_domain("python_basics", n_tasks=5)

    print("\n--- Math ---")
    sandbox.run_domain("math", n_tasks=5)

    print("\n--- Algorithms ---")
    sandbox.run_domain("algorithms", n_tasks=3)

    print("\n--- String Manipulation ---")
    sandbox.run_domain("string_manipulation", n_tasks=3)

    # Save packets
    print("\n[4] Saving data packets...")
    path = sandbox.save_packets()

    # Print stats
    sandbox.print_stats()

    print(f"\n  Data packet file: {path}")
    print("  Ready for knowledge injection via fact_injection_key or knowledge_pack_key")


if __name__ == "__main__":
    main()

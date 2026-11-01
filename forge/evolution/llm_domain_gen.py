"""LLM domain generator — auto-creates evolution domains from uncovered topics.

Based on FunSearch (Romera-Paredes et al. 2024), EoH (Liu et al. 2024),
and LLaMEA (van Stein & Bäck 2024):
  Use an LLM to propose new search domains by generating Python code for
  BaseDomain subclasses. The LLM sees:
    1. Uncovered optimization topics from the topic scanner
    2. Existing domain code as examples
    3. The BaseDomain interface
  And generates a new domain class that covers the uncovered topic.

This removes the manual domain creation bottleneck entirely. The system
auto-discovers what can be optimized (topic scanner), then auto-creates
the domains to optimize them (LLM generator), then runs them (engine).

Safety:
  - Generated code is sandboxed (executed in a restricted namespace)
  - Only BaseDomain subclasses are accepted
  - Generated domains are validated before registration
  - A fallback "GenericDomain" is used if LLM generation fails

Usage:
  from forge.evolution.llm_domain_gen import LLMDomainGenerator
  gen = LLMDomainGenerator(engine=forge_engine)
  new_domains = gen.generate_for_uncovered_topics(scanner.get_uncovered_topics())
"""
from __future__ import annotations

import ast
import logging
import textwrap
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from forge.evolution.domains import DOMAINS, BaseDomain
from forge.evolution.topic_scanner import OptimizationTopic


# ── AST safety gate for generated code ───────────────────────────────────────
# Generated domain.py is exec'ed IN-PROCESS (torch CUDA speed — a subprocess
# sandbox would re-init CUDA per eval). In-process exec cannot fully contain
# torch-level abuse (huge allocs, CUDA context corruption), so this gate only
# filters obvious escapes — it is NOT a security boundary.

# Call targets (dotted or bare) that indicate an escape attempt.
_DANGEROUS_CALLS = frozenset({
    'os.system', 'os.popen', 'os.remove', 'os.unlink', 'os.rmdir',
    'os.kill', 'os._exit', 'os.environ',
    'subprocess.call', 'subprocess.run', 'subprocess.Popen',
    'subprocess.check_output',
    'sys.exit', 'eval', 'exec', 'compile', 'open', 'input',
    '__import__', 'getattr', 'setattr', 'delattr', 'breakpoint',
    'exit', 'quit',
})

# Modules whose import gives generated code OS/network/interp access.
_BLOCKED_IMPORTS = frozenset({
    'os', 'sys', 'subprocess', 'socket', 'ctypes', 'multiprocessing',
    'shutil', 'pathlib', 'importlib', 'pickle', 'requests', 'urllib',
    'signal', 'builtins', 'traceback', 'inspect', 'gc', 'io',
})

# Attribute names that reach frame/function/module internals.
_DANGEROUS_ATTRS = frozenset({
    '__globals__', '__subclasses__', '__builtins__', 'f_globals',
    'f_builtins', 'func_globals', 'gi_frame', 'cr_frame', 'tb_frame',
})


def _check_domain_code(source: str) -> tuple[bool, str]:
    """AST safety gate run on LLM-generated domain code before exec.

    Returns (True, "") when the source passes; (False, reason) on any
    SyntaxError, blocked import, dangerous call, or internals-reaching
    attribute. Allows torch/numpy/math/random/typing/collections/
    functools/itertools/dataclasses and anything else not on the blocklist.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    def _dotted(node: ast.AST) -> str:
        """Resolve a Name/Attribute chain to its dotted form
        (``os.system`` -> 'os.system', ``open`` -> 'open')."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return False, f"blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _BLOCKED_IMPORTS:
                return False, f"blocked import: {node.module}"
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            # Match the full dotted name AND every prefix of it, so both
            # 'os.system(...)' and 'os.environ.get(...)' are caught while
            # namespaced lookalikes ('re.compile', 'x.open') stay allowed.
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                if ".".join(parts[:i]) in _DANGEROUS_CALLS:
                    return False, f"dangerous call: {name}"
        elif isinstance(node, ast.Attribute):
            if node.attr in _DANGEROUS_ATTRS:
                return False, f"dangerous attribute: {node.attr}"
    return True, ""


@dataclass
class GeneratedDomain:
    """A domain generated by the LLM, ready for registration."""
    name: str
    code: str
    topic: OptimizationTopic
    validated: bool = False
    error: str = ""


class GenericDomain(BaseDomain):
    """Fallback domain for uncovered topics — generic parameter sweep.

    When the LLM can't generate a custom domain, this provides a basic
    parameter sweep over the topic's parameters. It doesn't have a real
    evaluator (returns a heuristic score based on parameter values), but
    it ensures the topic is at least explored.
    """

    def __init__(self, topic_name: str, params: list[str],
                 source_file: str = "", description: str = ""):
        self._topic_name = topic_name
        self._params = params
        self._source_file = source_file
        self._description = description
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def name(self) -> str:
        # Sanitize name for DOMAINS registry
        safe = self._topic_name.replace(" ", "_").replace(".", "_")
        return f"Generic_{safe}"

    def output_dim(self) -> int:
        return max(1, len(self._params))

    def decode(self, params) -> dict[str, Any]:
        p = params.detach().cpu().numpy() if hasattr(params, 'detach') else params
        config = {}
        for i, param_name in enumerate(self._params):
            if i < len(p):
                config[param_name] = float(p[i])
        return config

    def encode(self, config) -> Any:
        import torch
        vals = []
        for param_name in self._params:
            v = config.get(param_name, 0.5)
            vals.append(float(v))
        return torch.tensor(vals, device=self._device)

    def evaluate(self, config) -> dict:
        """Heuristic evaluation: diversity-based score for generic exploration.

        Since we don't have a real evaluator for uncovered topics, we use
        a diversity-based score that encourages exploring different parameter
        combinations. This feeds into the MAP-Elites archive for later
        refinement when a real evaluator becomes available.
        """
        # Score based on parameter diversity (encourages exploration)
        vals = list(config.values())
        if not vals:
            return {"score": 0.0, "behavioral": (0.0, 0.0), "metadata": {}}

        # Heuristic: prefer configs that use non-default values
        diversity = sum(abs(v - 0.5) for v in vals) / len(vals)
        score = 50.0 + diversity * 30.0  # base 50 + diversity bonus

        behavioral = (float(diversity), float(score / 100.0))
        return {
            "score": float(score),
            "behavioral": behavioral,
            "metadata": {"topic": self._topic_name, "generic": True},
        }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [
            ("diversity", 10, 0.0, 1.0),
            ("quality", 10, 0.0, 1.0),
        ]

    def to_cpu(self) -> GenericDomain:
        import torch
        cpu = GenericDomain(self._topic_name, self._params, self._source_file, self._description)
        cpu._device = torch.device("cpu")
        return cpu


class LLMDomainGenerator:
    """Generates evolution domains using an LLM (ForgeEngine).

    For each uncovered topic, asks the LLM to generate a BaseDomain subclass
    that covers it. Falls back to GenericDomain if LLM generation fails.

    The LLM prompt includes:
      - The BaseDomain interface
      - 2-3 example domain implementations
      - The uncovered topic's name, source file, and parameters
      - Instructions to generate a valid Python class

    Generated code is validated:
      1. Must parse as valid Python (ast.parse)
      2. Must define a class that inherits from BaseDomain
      3. Must implement all abstract methods
      4. Must not contain dangerous imports (os.system, subprocess, etc.)
    """

    def __init__(self, engine=None, max_domains_per_run: int = 10):
        self.engine = engine
        self.max_domains_per_run = max_domains_per_run
        self.generated: list[GeneratedDomain] = []

    def generate_for_uncovered_topics(
        self, topics: list[OptimizationTopic]
    ) -> list[GeneratedDomain]:
        """Generate domains for uncovered optimization topics.

        Returns list of GeneratedDomain objects (validated or not).
        """
        results = []
        for topic in topics[:self.max_domains_per_run]:
            domain = self._generate_one(topic)
            if domain is not None:
                results.append(domain)
        return results

    def _generate_one(self, topic: OptimizationTopic) -> GeneratedDomain | None:
        """Generate a domain for a single topic.

        Tries LLM generation first, falls back to GenericDomain.
        """
        # Try LLM generation if engine is available
        rejected = ""
        if self.engine is not None:
            try:
                code = self._generate_with_llm(topic)
                if code:
                    safe, rejected = _check_domain_code(code)
                    if not safe:
                        logger.warning(
                            "LLM domain code rejected by AST gate: %s",
                            rejected)
                        code = None
                if (code and self._validate_code(code)
                        and self._sanity_check(code, topic)):
                    domain = GeneratedDomain(
                        name=topic.name,
                        code=code,
                        topic=topic,
                        validated=True,
                    )
                    self.generated.append(domain)
                    return domain
            except Exception:
                # Fall back to GenericDomain
                logger.debug("LLM domain generation failed, falling back to GenericDomain", exc_info=True)

        # Fallback: create a GenericDomain
        domain = GeneratedDomain(
            name=f"Generic_{topic.name}",
            code="",  # GenericDomain doesn't need generated code
            topic=topic,
            validated=True,
            error=f"rejected: {rejected}" if rejected else "",
        )
        self.generated.append(domain)
        return domain

    def _generate_with_llm(self, topic: OptimizationTopic) -> str | None:
        """Use the ForgeEngine to generate a domain class.

        Constructs a prompt with the BaseDomain interface + example domains
        + the topic description, and asks the LLM to generate a valid class.
        """
        prompt = self._build_prompt(topic)
        try:
            # ChatML wrap — engine.generate takes literal text; a raw
            # prompt to a chat-tuned ckpt is what made the proposer emit
            # token soup before the _wrap fix (same failure here).
            # \x3c keeps the literal consistent with LLMProposer._wrap.
            i = prompt.rfind("```python")
            prime = prompt[i + len("```python"):] if i >= 0 else ""
            user = prompt[:i] if i >= 0 else prompt
            # LFM3-family engines need the official template shape:
            # startoftext BOS + the empty think-prefill its jinja
            # renders for enable_thinking=False. Detected from the
            # tokenizer BOS (renamed ckpt dirs still carry it).
            _tok = getattr(self.engine, "tokenizer", None)
            _bos = getattr(_tok, "bos_token", "") or ""
            _lfm3 = ("startoftext" in _bos
                     or "lfm3" in str(getattr(
                         self.engine, "checkpoint_path", "")).lower())
            wrapped = (("\x3c|startoftext|>" if _lfm3 else "")
                       + "\x3c|im_start|>user\n" + user
                       + "\x3c|im_end|>\n\x3c|im_start|>assistant\n"
                       + ("\x3cthink>\x3c/think>" if _lfm3 else "")
                       + ("```python" + prime if prime else ""))
            response = self.engine.generate(
                wrapped, max_new_tokens=2048, temperature=0.3,
                stop=["\x3c|im_end|>"])
            # Body continuations start with 'def'/docstring — the
            # extractor would treat that as a restart and drop the
            # class-line prime, so splice it back on ourselves.
            code = self._extract_code(
                (prime + response) if prime else response, "",
                required=[self._safe_class_name(topic.name)])
            return code
        except Exception:
            return None

    def _build_prompt(self, topic: OptimizationTopic) -> str:
        """Build the LLM prompt for domain generation.

        Ends mid-class (unclosed fence + header) — the model continues
        the body rather than deciding whether to fence, restart, or
        chatter. Same primed-continuation trick as the CodeEvolve
        proposer."""
        return textwrap.dedent(f"""\
            You are an expert AI optimization researcher. Write a Python class
            that defines an evolution search domain for optimizing: {topic.name}

            Category: {topic.category}
            Source file: {topic.source_file}
            Description: {topic.description}
            Parameters to optimize: {topic.parameters}

            The class must inherit from BaseDomain and implement these methods:
              - name() -> str
              - output_dim() -> int
              - decode(params: torch.Tensor) -> dict
              - encode(config: dict) -> torch.Tensor
              - evaluate(config: dict) -> dict (returns score, behavioral, metadata)
              - behavioral_dims() -> list of (name, n_bins, min, max) tuples

            The evaluate() method should compute a meaningful score for the config.
            For feature flags, score based on expected speedup/quality tradeoff.
            For config params, score based on known-good ranges from literature.

            Use torch for tensor operations. Keep it under 100 lines.
            Complete the class body below — do not restart, no explanations.

            ```python
            # Generated domain for {topic.name}
            from forge.evolution.domains import BaseDomain
            import torch
            import numpy as np

            class {self._safe_class_name(topic.name)}(BaseDomain):
            """)

    def _safe_class_name(self, name: str) -> str:
        """Convert a topic name to a valid Python class name — scanner
        topics can carry spaces/punctuation/leading digits that would
        make the emitted `class <name>(BaseDomain):` line unparseable
        (guaranteed validation failure -> silent GenericDomain fallback)."""
        import re
        parts = [p for p in re.sub(r"[^0-9a-zA-Z]+", "_", name)
                 .strip("_").split("_") if p]
        cls = "".join(p.capitalize() for p in parts) or "Generic"
        if not cls[0].isalpha():
            cls = "T" + cls
        return cls + "Domain"

    def _extract_code(self, response: str, prime: str = "",
                      required: list | None = None) -> str | None:
        """Extract Python code from an LLM response - delegates to the
        hardened CodeEvolve extractor (think-strip, unclosed fences,
        primed continuations, prose salvage, required-name preference)."""
        from forge.evolution.code_evolve import _extract_code
        code = _extract_code(response, prime, required=required)
        if code and "class" in code:
            return code
        # Try without code fence
        if "class " in response and "BaseDomain" in response:
            return response
        return None

    def _validate_code(self, code: str) -> bool:
        """Validate that generated code is safe and correct."""
        # Check for dangerous patterns
        dangerous = ["os.system", "subprocess", "eval(", "exec(",
                     "open(", "__import__", "os.remove", "shutil"]
        for pattern in dangerous:
            if pattern in code:
                return False

        # Check it parses as valid Python
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        # Check it defines a class inheriting from BaseDomain
        has_domain_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseDomain":
                        has_domain_class = True
                        break

        return has_domain_class

    def _sanity_check(self, code: str, topic: OptimizationTopic) -> bool:
        """Execute-gate: a parseable BaseDomain subclass can still be
        garbage — constant evaluate(), evaluate() that crashes on plain
        configs, or a class that can't instantiate. Exec it, run
        evaluate() on two spread configs, and require finite scores
        with SOME variance before it counts as a real domain."""
        safe, reason = _check_domain_code(code)
        if not safe:
            logger.warning("sanity-check rejected generated code: %s", reason)
            return False
        try:
            import math
            namespace = {"BaseDomain": BaseDomain,
                         "torch": __import__("torch"),
                         "np": __import__("numpy", fromlist=["numpy"])}
            # NOTE: in-process exec keeps torch/CUDA fast but cannot fully
            # contain torch-level abuse (VRAM exhaustion, CUDA faults) —
            # _check_domain_code above is a filter, not a sandbox.
            exec(code, namespace)
            cls = next(
                (o for o in namespace.values()
                 if isinstance(o, type) and issubclass(o, BaseDomain)
                 and o is not BaseDomain), None)
            if cls is None:
                return False
            try:
                inst = cls()
            except TypeError:
                return True         # arg'd __init__ — can't smoke it;
                                    # accept and let registration decide
            scores = []
            for v in (0.1, 0.9):
                cfg = {p: v for p in (topic.parameters or ["x"])}
                try:
                    r = inst.evaluate(cfg)
                    s = (r or {}).get("score")
                    if not isinstance(s, (int, float)) \
                            or not math.isfinite(s):
                        return False
                    scores.append(float(s))
                except Exception:
                    return False
            # constant-score domain = no gradient to search on;
            # GenericDomain is the better fallback
            return len(set(scores)) > 1
        except Exception:
            return False

    def register_domains(self, domains: list[GeneratedDomain]) -> list[str]:
        """Register validated domains in the global DOMAINS registry.

        Returns list of registered domain names.
        """
        registered = []
        for domain in domains:
            if not domain.validated:
                continue

            if domain.code:
                # Execute generated code to get the class — re-gate here
                # too: domain.code can be exec'ed long after generation.
                # Residual risk: in-process exec cannot contain torch-level
                # abuse (VRAM exhaustion, CUDA faults); the AST gate is a
                # filter, not a sandbox.
                safe, reason = _check_domain_code(domain.code)
                if not safe:
                    domain.error = f"rejected: {reason}"
                    logger.warning("register_domains rejected %s: %s",
                                   domain.name, reason)
                    generic = GenericDomain(domain.topic.name, domain.topic.parameters,
                                           domain.topic.source_file, domain.topic.description)
                    DOMAINS[generic.name()] = lambda g=generic: g
                    registered.append(generic.name())
                    continue
                try:
                    namespace = {"BaseDomain": BaseDomain, "torch": __import__("torch"),
                                 "np": __import__("numpy", fromlist=["numpy"])}
                    exec(domain.code, namespace)
                    # Find the domain class in the namespace
                    for name, obj in namespace.items():
                        if (isinstance(obj, type) and issubclass(obj, BaseDomain)
                                and obj is not BaseDomain):
                            DOMAINS[obj.__name__] = obj
                            registered.append(obj.__name__)
                            break
                except Exception:
                    # Fall back to GenericDomain
                    generic = GenericDomain(domain.topic.name, domain.topic.parameters,
                                           domain.topic.source_file, domain.topic.description)
                    DOMAINS[generic.name()] = lambda g=generic: g
                    registered.append(generic.name())
            else:
                # GenericDomain fallback
                generic = GenericDomain(domain.topic.name, domain.topic.parameters,
                                       domain.topic.source_file, domain.topic.description)
                DOMAINS[generic.name()] = lambda g=generic: g
                registered.append(generic.name())

        return registered

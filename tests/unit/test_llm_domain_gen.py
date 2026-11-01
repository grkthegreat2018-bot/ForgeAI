"""LLMDomainGenerator contracts — ChatML wrap, primed-continuation
prompt, prime-splice extraction, validation gate, DOMAINS registration,
and the GenericDomain fallback path."""

from __future__ import annotations

import pytest

from forge.evolution.domains import DOMAINS
from forge.evolution.llm_domain_gen import GeneratedDomain, LLMDomainGenerator
from forge.evolution.topic_scanner import OptimizationTopic


def _topic(name: str = "use_fused_rope", **kw) -> OptimizationTopic:
    return OptimizationTopic(
        name=name,
        category=kw.pop("category", "feature_flag"),
        source_file=kw.pop("source_file", "forge/engine/x.py"),
        description=kw.pop("description", "fuse rope into qk norm"),
        parameters=kw.pop("parameters", ["enabled"]),
        **kw,
    )


class _FakeEngine:
    """Records the exact (prompt, kwargs) the generator sends."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, **kw) -> str:
        self.calls.append((prompt, kw))
        return self.response


@pytest.fixture
def _restore_domains():
    before = dict(DOMAINS)
    yield
    for k in [k for k in DOMAINS if k not in before]:
        DOMAINS.pop(k, None)


def test_safe_class_name():
    gen = LLMDomainGenerator()
    assert gen._safe_class_name("use_fused_rope") == "UseFusedRopeDomain"
    assert gen._safe_class_name("w8a8-quant.mode") == "W8a8QuantModeDomain"


def test_safe_class_name_sanitizes_spaces_and_digits():
    gen = LLMDomainGenerator()
    # Topic names with spaces/punctuation must still yield a legal
    # identifier or the generated `class X(BaseDomain):` line can never
    # parse and every topic silently degrades to GenericDomain.
    cls = gen._safe_class_name("fast softmax")
    assert cls.isidentifier(), cls
    cls = gen._safe_class_name("4bit quant")
    assert cls.isidentifier(), cls
    assert cls.endswith("Domain")


def test_build_prompt_ends_mid_class():
    gen = LLMDomainGenerator()
    p = gen._build_prompt(_topic())
    # Primed contract: exactly one unclosed ```python fence, ending on
    # the class header line — the model continues the BODY.
    assert p.count("```python") == 1
    assert "```" not in p.rsplit("```python", 1)[1]
    assert p.rstrip().endswith("class UseFusedRopeDomain(BaseDomain):")
    assert "Complete the class body" in p


def test_generate_with_llm_wraps_chatml_and_stops():
    eng = _FakeEngine("    def name(self):\n        return 'd'\n")
    gen = LLMDomainGenerator(engine=eng)
    code = gen._generate_with_llm(_topic())
    assert code is not None
    prompt, kw = eng.calls[0]
    assert prompt.startswith("<|im_start|>user\n")
    assert "<|im_end|>\n<|im_start|>assistant\n```python" in prompt
    assert kw["stop"] == ["<|im_end|>"]
    assert kw["max_new_tokens"] == 2048


def test_generate_with_llm_splices_prime_back():
    # Body continuations start indented — without the splice the class
    # header is lost and the required-name preference can't fire.
    eng = _FakeEngine(
        "    def name(self):\n        return 'use_fused_rope'\n    def output_dim(self):\n        return 1\n"
    )
    gen = LLMDomainGenerator(engine=eng)
    code = gen._generate_with_llm(_topic())
    assert code is not None
    assert "class UseFusedRopeDomain(BaseDomain):" in code
    assert "def name" in code


def test_generate_with_llm_restart_block_still_wins():
    # If the model ignores the prime and restarts with its own fenced
    # class, the required-name preference must pick that block.
    eng = _FakeEngine(
        "Sure! Here you go:\n```python\n"
        "class UseFusedRopeDomain(BaseDomain):\n"
        "    def name(self):\n        return 'x'\n```\nDone."
    )
    gen = LLMDomainGenerator(engine=eng)
    code = gen._generate_with_llm(_topic())
    assert code is not None
    assert "class UseFusedRopeDomain(BaseDomain):" in code


def test_extract_code_rejects_non_class_output():
    gen = LLMDomainGenerator()
    assert gen._extract_code("```python\ndef f():\n    return 1\n```", required=["X"]) is None


def test_extract_code_bare_class_fallback():
    gen = LLMDomainGenerator()
    raw = "class FooDomain(BaseDomain):\n    pass"
    assert gen._extract_code(raw) is not None


def test_validate_code_gates():
    gen = LLMDomainGenerator()
    good = "from forge.evolution.domains import BaseDomain\nclass D(BaseDomain):\n    pass\n"
    assert gen._validate_code(good)
    assert not gen._validate_code("class D(BaseDomain):\n    pass\nimport os\nos.system('x')\n")
    assert not gen._validate_code("class D(BaseDomain):\n    x = eval('1')\n")
    assert not gen._validate_code("class D(NotBase):\n    pass\n")
    assert not gen._validate_code("def broken(:\n")


def test_generate_one_fallback_without_engine():
    gen = LLMDomainGenerator(engine=None)
    d = gen._generate_one(_topic("mystery"))
    assert d.validated and d.name.startswith("Generic_") and d.code == ""


def test_generate_one_fallback_on_soup():
    gen = LLMDomainGenerator(engine=_FakeEngine("I cannot help with that."))
    d = gen._generate_one(_topic("mystery"))
    assert d.name.startswith("Generic_")


def test_generate_for_uncovered_topics_cap():
    eng = _FakeEngine("junk")
    gen = LLMDomainGenerator(engine=eng, max_domains_per_run=2)
    out = gen.generate_for_uncovered_topics([_topic(f"t{i}") for i in range(5)])
    assert len(out) == 2
    assert len(eng.calls) <= 2


_VALID_DOMAIN = (
    "from forge.evolution.domains import BaseDomain\n"
    "import torch\n"
    "class WiredDomain(BaseDomain):\n"
    "    def name(self): return 'wired'\n"
    "    def output_dim(self): return 1\n"
    "    def decode(self, p): return {}\n"
    "    def encode(self, c): return torch.zeros(1)\n"
    "    def evaluate(self, c): return {'score': 1.0, 'behavioral': (0.0, 0.0), 'metadata': {}}\n"
    "    def behavioral_dims(self): return [('b', 10, 0.0, 1.0)]\n"
)


def test_register_domains_real_and_fallback(_restore_domains):
    gen = LLMDomainGenerator()
    good = GeneratedDomain(name="t1", code=_VALID_DOMAIN, topic=_topic(), validated=True)
    # exec-able-AST but runtime-broken code -> GenericDomain fallback
    bad = GeneratedDomain(
        name="t2", code="class BrokenDomain(BaseDomain):\n    x = undefined_name\n", topic=_topic("t2"), validated=True
    )
    names = gen.register_domains([good, bad])
    assert "WiredDomain" in names
    assert DOMAINS["WiredDomain"].__name__ == "WiredDomain"
    assert any(n.startswith("Generic_") for n in names)

"""Research helper for the evolve loop — gather candidate optimization
techniques from the web for the registered @self_improve targets.

Keys come from the repo ``.env`` (TAVILY_API_KEY / EXA_API_KEY —
``engine_tools.py`` uses the same file). Each topic maps a real
forge/ function to a search query; results land in
``docs/evolve_research.md`` as {technique, url, applicability} rows the
next R&D round or evolve prompt can cite.

Usage:
    python -m forge.evolution.research              # all topics
    python -m forge.evolution.research jsd mask     # topic-name filter
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "evolve_research.md"

# topic -> (forge target/function, search query)
TOPICS: dict[str, tuple[str, str]] = {
    "top_p_sort": (
        "research.sampling_utils.top_p_filter_logits / auto top-p path",
        "fast top-p nucleus sampling kernel CUDA torch partial sort"),
    "min_k": (
        "forge.engine.engine_common._min_k_filter",
        "efficient top-k threshold filtering logits vocab GPU without full sort"),
    "block_diag_mask": (
        "forge.model.attention_ops._build_block_diag_causal_mask",
        "vectorized block diagonal causal attention mask document packing varlen"),
    "causal_mask": (
        "forge.model.attention_ops._causal_mask",
        "causal attention mask construction fastest triu vs arange compare pytorch"),
    "w8a8_act": (
        "forge.engine.quant.w8a8_quant.w8a8_quantize_activation",
        "per-token dynamic INT8 activation quantization kernel absmax torch._int_mm"),
    "exp_discretize": (
        "forge.engine.mamba3.exp_trapezoidal_discretize",
        "mamba SSM discretization numerical stability expm1 fused kernel complex"),
    "complex_norm": (
        "forge.engine.mamba3.complex_rmsnorm",
        "complex valued RMSNorm implementation view_as_real optimization"),
    "ngram_draft": (
        "forge.engine.decoding._ngram_lookup",
        "n-gram speculative decoding draft lookup suffix automaton prompt lookup decoding"),
    "jsd": (
        "forge.engine.decoding._jsd (DoLa layer contrast)",
        "vectorized Jensen Shannon divergence pytorch single pass softmax"),
    "ece": (
        "forge.engine.decision_head.expected_calibration_error",
        "expected calibration error vectorized histogram bucketize pytorch"),
    "kv_slice": (
        "forge.engine.prefix_cache._slice_past_kv",
        "KV cache slice clone memory efficiency prefix cache reuse LMCache"),
    "stop_trunc": (
        "forge.engine.engine_common._truncate_at_stop",
        "streaming stop sequence detection token generation early truncation efficient"),
}


def _env_key(name: str) -> str | None:
    """API key from env or the repo .env (same loader semantics as
    engine_tools._get_api_key)."""
    v = os.environ.get(name)
    if v:
        return v
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _post(url: str, payload: dict, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tavily_search(query: str, n: int = 4) -> list[dict]:
    key = _env_key("TAVILY_API_KEY")
    if not key:
        return []
    try:
        r = _post("https://api.tavily.com/search", {
            "api_key": key, "query": query, "max_results": n,
            "search_depth": "basic"})
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": x.get("content", "")[:400]}
                for x in r.get("results", [])]
    except Exception as e:
        return [{"title": "", "url": "", "snippet": f"tavily error: {e}"}]


def ddgs_search(query: str, n: int = 4) -> list[dict]:
    """Keyless multi-engine search via the ddgs package (DuckDuckGo,
    Bing, Brave, Yahoo auto-fallback). No API key, no quota — burst
    throttling is possible, so the caller's cache/budget still applies."""
    try:
        from ddgs import DDGS
        with DDGS() as d:
            return [{"title": r.get("title", ""), "url": r.get("href", ""),
                     "snippet": (r.get("body") or "")[:400]}
                    for r in d.text(query, max_results=n)]
    except Exception as e:
        return [{"title": "", "url": "", "snippet": f"ddgs error: {e}"}]


def arxiv_search(query: str, n: int = 4) -> list[dict]:
    """arXiv Atom API — free, official, ~1req/3s polite. Best for
    paper/technique queries (kernels, sampling, quantization)."""
    import xml.etree.ElementTree as ET
    try:
        q = urllib.parse.quote_plus(
            'all:"' + query.replace('"', '') + '"')
        url = ("http://export.arxiv.org/api/query?search_query="
               f"{q}&start=0&max_results={n}&sortBy=relevance")
        req = urllib.request.Request(
            url, headers={"User-Agent": "forgeai-evolve/1.0"})
        with urllib.request.urlopen(req, timeout=20.0) as r:
            root = ET.fromstring(r.read().decode())
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall("a:entry", ns):
            title = " ".join(
                (e.findtext("a:title", "", ns) or "").split())
            link = ""
            for l in e.findall("a:link", ns):
                if l.get("type") == "text/html":
                    link = l.get("href", "")
                    break
            if not link:
                link = e.findtext("a:id", "", ns)
            summ = " ".join(
                (e.findtext("a:summary", "", ns) or "").split())[:400]
            out.append({"title": title, "url": link, "snippet": summ})
        return out
    except Exception as e:
        return [{"title": "", "url": "", "snippet": f"arxiv error: {e}"}]


def s2_search(query: str, n: int = 4) -> list[dict]:
    """Semantic Scholar Graph API — free public tier (~1rps), returns
    abstracts + TLDRs — richer snippets than raw titles for technique
    hunting."""
    try:
        q = urllib.parse.quote_plus(query)
        url = ("https://api.semanticscholar.org/graph/v1/paper/search"
               f"?query={q}&limit={n}"
               "&fields=title,url,abstract,tldr")
        req = urllib.request.Request(
            url, headers={"User-Agent": "forgeai-evolve/1.0"})
        with urllib.request.urlopen(req, timeout=20.0) as r:
            d = json.loads(r.read().decode())
        out = []
        for p in d.get("data", []):
            tldr = (p.get("tldr") or {}).get("text") or ""
            snip = tldr or (p.get("abstract") or "")
            out.append({"title": p.get("title", ""),
                        "url": p.get("url", ""),
                        "snippet": snip[:400]})
        return out
    except Exception as e:
        return [{"title": "", "url": "", "snippet": f"s2 error: {e}"}]


_PAPER_HINTS = ("paper", "arxiv", "algorithm", "novel", "technique",
                "attention", "sampling", "quantiz", "kernel", "spec",
                "cache", "divergence", "ssm", "mamba", "rope", "kv ")


def free_search(query: str, n: int = 4) -> list[dict]:
    """Keyless chain: ddgs general web first; for paper-ish queries,
    arXiv + Semantic Scholar fill any shortfall. Returns hits in the
    standard {title,url,snippet} shape — same contract as tavily/exa."""
    hits = [h for h in ddgs_search(query, n) if h.get("url")]
    if len(hits) >= 2 or not any(
            h in query.lower() for h in _PAPER_HINTS):
        return hits
    for backend in (arxiv_search, s2_search):
        if len(hits) >= n:
            break
        hits += [h for h in backend(query, n) if h.get("url")]
    return hits[:n]


def jina_fetch(url: str, timeout: float = 20.0) -> str:
    """Keyless Jina reader — https://r.jina.ai/<url> proxies the page
    and returns clean markdown-ish text. Free tier ~20 RPM, no API
    key. Shared fallback for engine_tools._web_scrape. Returns "" on
    any failure (caller chains the next backend)."""
    try:
        req = urllib.request.Request(
            "https://r.jina.ai/" + url,
            headers={"User-Agent": "forgeai-evolve/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def trafilatura_fetch(url: str, timeout: float = 20.0) -> str:
    """requests.get + trafilatura main-content extraction — only when
    the optional package is installed (lazy import; "" otherwise)."""
    try:
        import trafilatura
    except ImportError:
        return ""
    try:
        import requests
        html = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "forgeai-evolve/1.0"}).text
        return trafilatura.extract(html) or ""
    except Exception:
        return ""


def exa_search(query: str, n: int = 4) -> list[dict]:
    """Exa semantic search — key goes in the x-api-key header."""
    key = _env_key("EXA_API_KEY")
    if not key:
        return []
    try:
        req = urllib.request.Request(
            "https://api.exa.ai/search",
            data=json.dumps({"query": query, "numResults": n,
                             "contents": {"text": {"maxCharacters": 400}}}
                            ).encode(),
            headers={"Content-Type": "application/json",
                     "x-api-key": key})
        with urllib.request.urlopen(req, timeout=20.0) as r:
            d = json.loads(r.read().decode())
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": (x.get("text") or "")[:400]}
                for x in d.get("results", [])]
    except Exception as e:
        return [{"title": "", "url": "", "snippet": f"exa error: {e}"}]


def research(topics: dict | None = None, n: int = 4,
             out_path: Path = OUT) -> dict[str, list[dict]]:
    """Query each topic; write the markdown digest. Returns raw hits.

    Engine fallback: tavily first, exa when tavily yields only errors
    (dead key, quota) — reports which engine actually produced hits."""
    topics = TOPICS if topics is None else topics

    def search(query: str):
        hits = [h for h in free_search(query, n) if h.get("url")]
        if hits:
            return hits, "free"
        hits = tavily_search(query, n)
        good = [h for h in hits if h.get("url")]
        return (hits, "tavily") if good else (exa_search(query, n), "exa")

    engines: set[str] = set()
    all_hits: dict[str, list[dict]] = {}
    body: list[str] = []
    for topic, (target, query) in topics.items():
        hits, eng = search(query)
        engines.add(eng)
        all_hits[topic] = hits
        body.append(f"## {topic} — `{target}`")
        body.append(f"*query ({eng}): {query}*")
        body.append("")
        if not hits:
            body.append("- (no results)")
        for h in hits:
            title = (h.get("title") or h.get("url") or "?").strip()
            url = h.get("url", "").strip()
            snip = " ".join((h.get("snippet") or "").split())[:300]
            body.append(f"- **{title}** — {url}")
            if snip:
                body.append(f"  {snip}")
        body.append("")
    engine = "+".join(sorted(engines)) or "none"
    lines = [
        "# Evolve-loop research digest",
        "",
        f"Source: {engine} API (keys from repo .env). One row per "
        "candidate technique; applicability notes tie each hit to the "
        "real forge/ function it could improve. Regenerate:",
        "`python -m forge.evolution.research`",
        "",
        *body,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[research] {sum(len(v) for v in all_hits.values())} hits "
          f"across {len(all_hits)} topics -> {out_path}")
    return all_hits


if __name__ == "__main__":
    sel = sys.argv[1:]
    topics = TOPICS if not sel else {
        k: v for k, v in TOPICS.items()
        if any(s in k for s in sel)}
    if not topics:
        print(f"no topics match {sel}; have: {sorted(TOPICS)}")
        sys.exit(1)
    research(topics)

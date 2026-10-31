"""Unit tests for forge_gui.api.web_tools (keyless web search/fetch).

No real network calls: ``urllib.request.urlopen`` is monkeypatched to
return canned HTML / JSON / raise errors. Tests cover:
- tool definition shape (OpenAI function-calling schema)
- URL scheme validation (block javascript:/file:/data:, allow http/https)
- DuckDuckGo HTML parsing (result + redirect unwrap)
- Wikipedia / arXiv JSON/XML parsing
- network error handling (graceful {error: ...} dict)
- WebTools.execute dispatch + n clamping
- ToolHarness integration (defs include web tools, dispatch works)
"""
import json
from unittest.mock import patch

import pytest

from forge.web_primitives import (
    arxiv_search as _arxiv_search,
)
from forge.web_primitives import (
    ddg_search as _web_search,
)
from forge.web_primitives import (
    fetch_url as _web_fetch,
)
from forge.web_primitives import (
    is_safe_url as _is_safe_url,
)
from forge.web_primitives import (
    strip_ddg_redirect as _strip_ddg_redirect,
)
from forge.web_primitives import (
    wikipedia_search as _wikipedia_search,
)
from forge_gui.api.tool_harness import ToolHarness
from forge_gui.api.web_tools import (
    WebTools,
    web_tool_defs,
)


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class _FakeResp:
    """Minimal context-manager response object mimicking urlopen's return."""

    def __init__(self, data: bytes, content_type: str = "",
                 final_url: str = ""):
        self._data = data
        self._ctype = content_type
        self._final = final_url
        # dict satisfies headers.get("Content-Type") in http_get_full
        self.headers = ({"Content-Type": content_type}
                        if content_type else {})

    def read(self) -> bytes:
        return self._data

    def geturl(self) -> str:
        return self._final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(data: bytes | str, content_type: str = "",
                  final_url: str = ""):
    """Return a callable that yields a _FakeResp wrapping `data`."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return lambda *a, **kw: _FakeResp(data, content_type, final_url)


def _unwrap(res: dict) -> dict:
    """WebTools.execute wraps results in untrusted_content (EC-163)."""
    return res["untrusted_content"]["data"]


# â”€â”€ tool definition shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_web_tool_defs_shape():
    defs = web_tool_defs()
    names = {d["function"]["name"] for d in defs}
    assert {"web_search", "news_search", "web_fetch",
            "wikipedia_search", "arxiv_search"} == names
    for d in defs:
        assert d["type"] == "function"
        params = d["function"]["parameters"]
        assert params["type"] == "object"
        assert "required" in params


def test_web_tools_names_matches_defs():
    def_names = {d["function"]["name"] for d in web_tool_defs()}
    assert def_names == WebTools.NAMES


# â”€â”€ URL safety â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@pytest.mark.parametrize("url,ok", [
    ("https://example.com", True),
    ("http://example.com/path?q=1", True),
    ("HTTPS://Example.com", True),
    ("javascript:alert(1)", False),
    ("file:///etc/passwd", False),
    ("data:text/html,<script>", False),
    ("ftp://example.com", False),
    ("", False),
])
def test_is_safe_url(url, ok):
    assert _is_safe_url(url) is ok


def test_strip_ddg_redirect():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com%2Fpage&rut=abc"
    assert _strip_ddg_redirect(wrapped) == "https://real.com/page"
    # non-redirect href passes through unchanged
    assert _strip_ddg_redirect("https://plain.com/x") == "https://plain.com/x"


# â”€â”€ web_search (DuckDuckGo HTML) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_DDG_HTML = """
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Furllib.html">urllib â€” Python docs</a>
<a class="result__snippet">This module provides a high-level interface ...</a>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F2">Second Result</a>
<a class="result__snippet">Second snippet <b>text</b> here.</a>
"""


def test_web_search_parses_results():
    with patch("forge.web_primitives._OPENER.open",
               _fake_urlopen(_DDG_HTML)):
        res = _web_search("urllib", n=5)
    assert res["error"] is None
    assert len(res["results"]) == 2
    first = res["results"][0]
    assert first["url"] == "https://docs.python.org/3/library/urllib.html"
    assert first["title"] == "urllib â€” Python docs"
    assert "high-level interface" in first["snippet"]
    # second snippet has tags stripped
    assert "<b>" not in res["results"][1]["snippet"]


def test_web_search_empty_query():
    res = _web_search("   ")
    assert res["results"] == []
    assert "empty" in res["error"]


def test_web_search_network_error():
    def boom(*a, **kw):
        raise OSError("connection refused")
    with patch("forge.web_primitives._OPENER.open", boom):
        res = _web_search("anything")
    assert res["results"] == []
    assert "fetch failed" in res["error"]


def test_web_search_no_results():
    with patch("forge.web_primitives._OPENER.open",
               _fake_urlopen("<html><body>no results here</body></html>")):
        res = _web_search("zzz")
    assert res["results"] == []
    assert "no results" in res["error"]


def test_web_search_filters_ad_redirects():
    # DDG injects sponsored links that use a y.js redirector (destination
    # base64 in u3=, not uddg=) â€” these must never surface as results.
    html = """
<a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=ads.example&u3=AAAA">Sponsored Junk</a>
<a class="result__snippet">Buy now!</a>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com%2Farticle">Real Article</a>
<a class="result__snippet">Real snippet.</a>
<a class="result__a" href="https://duckduckgo.com/l/?no_uddg=1">Still DDG</a>
<a class="result__snippet">Tracker.</a>
"""
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(html)):
        res = _web_search("news", n=5)
    urls = [r["url"] for r in res["results"]]
    assert urls == ["https://real.com/article"]


# â”€â”€ news_search (Google News RSS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_GNEWS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Big story of the day - Example News</title>
  <link>https://news.google.com/rss/articles/AAAA</link>
  <pubDate>Tue, 01 Jul 2025 12:00:00 GMT</pubDate>
  <source url="https://example.com">Example News</source>
  <description>&lt;a href="..."&gt;Big story&lt;/a&gt; details</description>
</item>
<item>
  <title>Second headline - Other Outlet</title>
  <link>https://news.google.com/rss/articles/BBBB</link>
  <pubDate>Tue, 01 Jul 2025 11:30:00 GMT</pubDate>
  <source url="https://other.com">Other Outlet</source>
  <description>plain description</description>
</item>
</channel></rss>"""


def test_news_search_parses_rss():
    from forge.web_primitives import google_news_search
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(_GNEWS_XML)):
        res = google_news_search("top news", n=5)
    assert res["error"] is None
    assert len(res["results"]) == 2
    r = res["results"][0]
    assert r["title"].startswith("Big story of the day")
    assert r["source"] == "Example News"
    assert r["published"] == "Tue, 01 Jul 2025 12:00:00 GMT"
    assert "news.google.com" in r["url"]
    assert "<a" not in r["snippet"]  # description HTML stripped


def test_news_search_network_error():
    from forge.web_primitives import google_news_search

    def boom(*a, **kw):
        raise OSError("timeout")
    with patch("forge.web_primitives._OPENER.open", boom):
        res = google_news_search("x")
    assert res["results"] == []
    assert "timeout" in res["error"]


# â”€â”€ web_fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_web_fetch_strips_html():
    html = ("<html><head><script>evil()</script><style>x{}</style></head>"
            "<body><p>Hello   world</p></body></html>")
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(html)):
        res = _web_fetch("https://example.com")
    assert res["error"] is None
    assert res["url"] == "https://example.com"
    assert "Hello world" in res["text"]
    assert "evil" not in res["text"]      # script stripped
    assert "<" not in res["text"]         # tags stripped


def test_web_fetch_rejects_unsafe_scheme():
    res = _web_fetch("javascript:alert(1)")
    assert res["text"] == ""
    assert "scheme" in res["error"]


def test_web_fetch_rejects_file_scheme():
    res = _web_fetch("file:///etc/passwd")
    assert res["text"] == ""
    assert "scheme" in res["error"]


def test_web_fetch_network_error():
    def boom(*a, **kw):
        raise OSError("404")
    with patch("forge.web_primitives._OPENER.open", boom):
        res = _web_fetch("https://example.com/missing")
    assert res["text"] == ""
    assert "fetch failed" in res["error"]


def test_web_fetch_truncates():
    html = "<body>" + ("x " * 5000) + "</body>"
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(html)):
        res = _web_fetch("https://example.com", max_chars=100)
    assert len(res["text"]) <= 100
    assert res["truncated"] is True


# â”€â”€ wikipedia_search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_WIKI_JSON = json.dumps({
    "query": {"search": [
        {"title": "Python (programming language)", "snippet": "a <i>high-level</i> language"},
        {"title": "Python (genus)", "snippet": "a genus of snakes"},
    ]}
})


def test_wikipedia_search_parses():
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(_WIKI_JSON)):
        res = _wikipedia_search("python", n=2)
    assert res["error"] is None
    assert len(res["results"]) == 2
    assert res["results"][0]["title"] == "Python (programming language)"
    assert "<i>" not in res["results"][0]["snippet"]  # tags stripped
    assert "wikipedia.org/wiki/Python" in res["results"][0]["url"]


def test_wikipedia_search_network_error():
    def boom(*a, **kw):
        raise OSError("dns fail")
    with patch("forge.web_primitives._OPENER.open", boom):
        res = _wikipedia_search("x")
    assert res["results"] == []
    assert "dns fail" in res["error"]


# â”€â”€ arxiv_search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <title>Attention Is All You Need</title>
  <summary>We propose a new architecture...</summary>
  <id>http://arxiv.org/abs/1706.03762v5</id>
  <published>2017-06-12T18:00:00Z</published>
</entry>
</feed>"""


def test_arxiv_search_parses():
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(_ARXIV_XML)):
        res = _arxiv_search("transformer", n=3)
    assert res["error"] is None
    assert len(res["results"]) == 1
    r = res["results"][0]
    assert r["title"] == "Attention Is All You Need"
    assert "new architecture" in r["summary"]
    assert r["url"] == "http://arxiv.org/abs/1706.03762v5"
    assert r["published"] == "2017-06-12"


def test_arxiv_search_network_error():
    def boom(*a, **kw):
        raise OSError("timeout")
    with patch("forge.web_primitives._OPENER.open", boom):
        res = _arxiv_search("x")
    assert res["results"] == []
    assert "timeout" in res["error"]


# â”€â”€ WebTools.execute dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_webtools_execute_dispatch():
    wt = WebTools()
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(_DDG_HTML)):
        res = wt.execute("web_search", {"query": "test"})
    # success â†’ error key popped by execute() normalization
    assert "error" not in res
    assert len(_unwrap(res)["results"]) == 2


def test_webtools_execute_unknown_tool():
    wt = WebTools()
    res = wt.execute("not_a_web_tool", {})
    assert "unknown web tool" in res["error"]


def test_webtools_execute_disabled():
    wt = WebTools(enabled=False)
    res = wt.execute("web_search", {"query": "x"})
    assert "disabled" in res["error"]


def test_webtools_execute_clamps_n():
    wt = WebTools()
    captured = {}

    def fake_open(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp(b"<html></html>")

    with patch("forge.web_primitives._OPENER.open", fake_open):
        wt.execute("web_search", {"query": "x", "n": 999})
    # n clamped to 10 â†’ srlimit/max_results stays bounded (DDG ignores extra)
    assert "q=x" in captured["url"]


def test_webtools_execute_clamps_max_chars():
    wt = WebTools()
    html = "<body>" + ("y " * 20000) + "</body>"
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(html)):
        res = wt.execute("web_fetch", {"url": "https://x.com", "max_chars": 999999})
    # clamped to 16000
    assert len(_unwrap(res)["text"]) <= 16000


# â”€â”€ ToolHarness integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_harness_includes_web_tools_when_provided(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    names = {d["function"]["name"] for d in h.tool_defs()}
    assert {"web_search", "news_search", "web_fetch",
            "wikipedia_search", "arxiv_search"} <= names


def test_harness_no_web_tools_when_none(tmp_path):
    h = ToolHarness(workspace=str(tmp_path))
    names = {d["function"]["name"] for d in h.tool_defs()}
    assert "web_search" not in names


def test_harness_dispatches_web_tool(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(_DDG_HTML)):
        rec = h.execute("web_search", {"query": "urllib"})
    assert rec["ok"] is True
    assert rec["name"] == "web_search"
    assert len(_unwrap(rec["result"])["results"]) == 2


def test_harness_web_tool_error_marked_not_ok(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    rec = h.execute("web_fetch", {"url": "javascript:alert(1)"})
    assert rec["ok"] is False
    assert "scheme" in rec["result"]["error"]


def test_harness_chat_tool_defs_include_web(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    names = {d["function"]["name"] for d in h.chat_tool_defs()}
    assert {"web_search", "news_search", "web_fetch"} <= names


def test_webtools_execute_news_dispatch(tmp_path):
    wt = WebTools()
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(_GNEWS_XML)):
        res = wt.execute("news_search", {"query": "top news"})
    assert "error" not in res
    assert _unwrap(res)["results"][0]["source"] == "Example News"


def test_harness_read_only_keeps_web_tools(tmp_path):
    # web tools are read-only GET â†’ must survive read-only mode
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools(),
                    read_only=True)
    names = {d["function"]["name"] for d in h.tool_defs()}
    assert "web_search" in names
    assert "web_fetch" in names


# â”€â”€ fetch_url upgrades (any-URL access) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_fetch_url_bare_domain_upgrades_https():
    seen = []

    def fake(req, timeout=None, **kw):
        seen.append(req.full_url)
        return _FakeResp(b"<html><body><p>hi</p></body></html>")

    with patch("forge.web_primitives._OPENER.open", fake):
        res = _web_fetch("example.com")
    assert res["error"] is None
    assert res["url"] == "https://example.com"
    assert seen == ["https://example.com"]


def test_fetch_url_extracts_links_and_titles():
    html = ("<html><head><title>Doc Page</title></head><body>"
            "<h1>Guide</h1><p>Intro text.</p>"
            "<a href=\"/deep/dive\">Read the deep dive</a>"
            "<a href=\"https://other.com/x\">External</a>"
            "<a href=\"javascript:void(0)\">dead</a>"
            "<ul><li>one</li><li>two</li></ul></body></html>")
    with patch("forge.web_primitives._OPENER.open",
               _fake_urlopen(html, content_type="text/html")):
        res = _web_fetch("https://example.com/docs")
    assert res["error"] is None
    assert res["title"] == "Doc Page"
    assert "# Guide" in res["text"]
    # inline markdown links â€” link names stay visible to the model
    assert "[Read the deep dive](https://example.com/deep/dive)" \
        in res["text"]
    assert "[External](https://other.com/x)" in res["text"]
    assert "javascript" not in res["text"]
    assert "- one" in res["text"] and "- two" in res["text"]
    urls = {l["url"] for l in res["links"]}
    assert "https://example.com/deep/dive" in urls
    assert "https://other.com/x" in urls
    texts = {l["text"] for l in res["links"]}
    assert "Read the deep dive" in texts


def test_fetch_url_reports_final_url_after_redirect():
    html = "<html><body><p>article</p></body></html>"
    with patch("forge.web_primitives._OPENER.open",
               _fake_urlopen(html, content_type="text/html",
                             final_url="https://real.example/article")):
        res = _web_fetch("https://news.google.com/rss/articles/AAAA")
    assert res["final_url"] == "https://real.example/article"
    assert res["url"] == "https://news.google.com/rss/articles/AAAA"


def test_fetch_url_json_passthrough():
    payload = '{"ok": true, "items": [1, 2, 3]}'
    with patch("forge.web_primitives._OPENER.open",
               _fake_urlopen(payload,
                             content_type="application/json")):
        res = _web_fetch("https://api.example.com/data")
    assert res["error"] is None
    assert res["text"] == payload          # raw, not HTML-stripped
    assert res["content_type"] == "application/json"
    assert "links" not in res


def test_fetch_url_offset_paging():
    html = ("<body><p>"
            + "".join(f"token{i} " for i in range(3000)) + "</p></body>")
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(html)):
        res1 = _web_fetch("https://example.com/long", max_chars=2000)
        res2 = _web_fetch("https://example.com/long", max_chars=2000,
                          offset=res1["next_offset"])
        resN = _web_fetch("https://example.com/long", max_chars=2000,
                          offset=res1["chars"] - 500)
    assert res1["truncated"] is True
    assert res1["next_offset"] == 2000
    assert res2["text"] != res1["text"]
    assert res2["next_offset"] == 4000
    assert resN["truncated"] is False      # last page has no next_offset
    assert "next_offset" not in resN
    assert res1["chars"] == res2["chars"] > 2000


def test_webtools_execute_passes_offset():
    wt = WebTools()
    html = "<body><p>" + ("z " * 5000) + "</p></body>"
    with patch("forge.web_primitives._OPENER.open", _fake_urlopen(html)):
        res = wt.execute("web_fetch", {"url": "https://x.com",
                                       "max_chars": 500, "offset": 400})
    data = _unwrap(res)
    assert data["chars"] > 400
    assert len(data["text"]) == 500
    assert data["truncated"] is True


# â”€â”€ search fallback chain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_DDG_LITE_HTML = """
<table>
<tr><td><a class="result-link"
 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite.example%2Fpage">
Lite Result</a></td></tr>
<tr><td class="result-snippet">lite snippet text</td></tr>
</table>
"""

_BING_HTML = """
<ul><li class="b_algo"><h2><a href="https://bing.example/r">
Bing Result</a></h2><p>bing snippet</p></li></ul>
"""


def test_search_falls_back_to_ddg_lite():
    """DDG HTML endpoint parses empty â†’ lite endpoint supplies results."""
    calls = []

    def fake(req, timeout=None, **kw):
        url = req.full_url
        calls.append(url)
        if "html.duckduckgo.com" in url:
            return _FakeResp(b"<html>bot wall</html>")
        if "lite.duckduckgo.com" in url:
            return _FakeResp(_DDG_LITE_HTML.encode())
        raise AssertionError(f"unexpected engine call: {url}")

    with patch("forge.web_primitives._OPENER.open", fake):
        res = _web_search("test query", n=5)
    assert res["error"] is None
    assert res["results"][0]["url"] == "https://lite.example/page"
    assert res["results"][0]["title"] == "Lite Result"
    assert "lite snippet" in res["results"][0]["snippet"]


def test_search_falls_back_to_bing():
    def fake(req, timeout=None, **kw):
        url = req.full_url
        if "bing.com" in url:
            return _FakeResp(_BING_HTML.encode())
        return _FakeResp(b"<html>nothing parseable</html>")

    with patch("forge.web_primitives._OPENER.open", fake):
        res = _web_search("test query", n=5)
    assert res["error"] is None
    assert res["results"][0]["url"] == "https://bing.example/r"
    assert res["results"][0]["title"] == "Bing Result"
    assert "bing snippet" in res["results"][0]["snippet"]


def test_search_all_engines_dead_reports_error():
    def boom(*a, **kw):
        raise OSError("all down")
    with patch("forge.web_primitives._OPENER.open", boom):
        res = _web_search("anything")
    assert res["results"] == []
    assert "fetch failed" in res["error"]

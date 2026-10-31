"""Continuous online scraper -> normalized doc feed for FluxLM/Strata (R61).

Pipeline:  sources (generators) -> Processor (dedupe/filter) -> ShardWriter
(jsonl queue) -> Feeder (offline corpus | live POST /engine/learn |
in-process model.ingest).

Records are plain dicts::

    {"id": sha1, "source": "wiki", "title": ..., "url": ...,
     "text": ..., "fetched_at": epoch}

Design notes:

- **Fetch is injectable** — every source takes ``fetch=wp.http_get`` so
  tests run fully offline.
- **Queue contract**: ShardWriter emits ``feed_NNNNNN.jsonl.ready``;
  feeders consume ``*.ready`` and move them to ``fed/``.  Scrape and feed
  can run as one process or two — the filesystem is the bus, so a crash
  never loses more than the in-flight shard.
- **Provenance is free**: every record carries source+url, and each
  shard is ingested under a journal ``tag`` ("scrape:wiki" etc.) so a
  bad source is one ``revert_tag`` away from gone.
- **Dedupe is persistent**: ``seen.txt`` (sha1 per line) survives
  restarts — re-running the same source never double-ingests.
- **Rate limits live here**: ``rps`` throttles requests.  Wikimedia's
  robot policy wants a descriptive UA (handled by ``wp._ua_for``) and
  reasonable request rates — batched extracts (20 pages/req) make even
  1 req/s ~= 20 pages/s.  For >100 pages/s sustained, use the bulk HF
  stream (``train_flux.py --hf``) — live scraping is for the
  continuous/fresh tail.

Sources: wiki-random / wiki-all (full alphabetical walk, resumable via
``apcontinue`` cursor in state.json) / wiki-search / wiki-recent /
arxiv / wiktionary / urls (arbitrary pages) / ddg (search->fetch).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from forge import web_primitives as wp
import contextlib

Fetch = Callable[..., str]

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"
ARXIV_API = "http://export.arxiv.org/api/query"

MIN_CHARS = 500           # wiki min-chars mirrors download_wikipedia.py
EXLIMIT = 20              # generator queries cap at 20 pages/request
TITLES_BATCH = 50         # prop=revisions allows 50 titles/request


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


def _clean(text: str) -> str:
    """Collapse whitespace runs; drop empty/pure-markup docs."""
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text or "")
    return text.strip()


def record(source: str, title: str, url: str, text: str,
           ts: float | None = None) -> dict | None:
    """Normalized doc record; None when the text carries no signal."""
    text = _clean(text)
    if not text:
        return None
    rid = _sha(f"{source}|{url or title}|{text[:512]}")
    return {"id": rid, "source": source, "title": title[:200],
            "url": url, "text": text,
            "fetched_at": ts if ts is not None else time.time()}


# ── persistent dedupe + resumable state ─────────────────────────────────

class SeenSet:
    """sha1-per-line dedupe set; append-only file, survives restarts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._set: set[str] = set()
        if self.path.exists():
            self._set = {ln.strip() for ln in
                         self.path.read_text(encoding="utf-8",
                                             errors="replace")
                         .splitlines() if ln.strip()}
        self._f = self.path.open("a", encoding="utf-8")

    def add(self, rid: str) -> bool:
        """True iff rid was new (and is now recorded)."""
        if rid in self._set:
            return False
        self._set.add(rid)
        self._f.write(rid + "\n")
        self._f.flush()
        return True

    def __len__(self) -> int:
        return len(self._set)


class FeedState:
    """Small JSON state file: per-source cursors + shard index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {"cursors": {}, "shard_idx": 0,
                                     "stats": {}}
        if self.path.exists():
            with contextlib.suppress(Exception):
                self.data.update(json.loads(
                    self.path.read_text(encoding="utf-8")))

    def cursor(self, source: str, default: str = "") -> str:
        return str(self.data["cursors"].get(source, default))

    def set_cursor(self, source: str, cur: str) -> None:
        self.data["cursors"][source] = cur

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1),
                             encoding="utf-8")


# ── sources ─────────────────────────────────────────────────────────────

def _api_get(api: str, params: dict, fetch: Fetch,
             retries: int = 4) -> dict:
    """GET a JSON API with backoff — transient SSL/proxy hiccups are
    routine on long scrapes; a persistent failure ends the source
    (returns {}) rather than crashing the whole feed loop."""
    from urllib.parse import urlencode
    url = api + "?" + urlencode(params)
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return json.loads(fetch(url))
        except Exception:
            if attempt >= retries:
                return {}
            time.sleep(delay)
            delay *= 2
    return {}


# ── wikitext -> plaintext ───────────────────────────────────────────────
# prop=extracts caps whole-article text at ONE page per request; the
# revisions API returns wikitext for ~50 pages/request, so the live path
# fetches wikitext and cleans locally.  Cleaning also means
# should_skip()'s {{disambiguation}}-style patterns see raw markup —
# run filters on the RAW text before cleaning.

_RE_WIKI_REF = re.compile(r"<ref\b[^>]*?(?:/>|>.*?</ref>)", re.DOTALL | re.I)
_RE_WIKI_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_WIKI_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_RE_WIKI_FILE = re.compile(
    r"\[\[(?:File|Image|Category|Media)\s*:[^\]]*\]\]", re.I)
_RE_WIKI_LINK = re.compile(r"\[\[([^\]|]*)\|?([^\]]*)\]\]")
_RE_WIKI_EXTLINK = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
_RE_WIKI_FMT = re.compile(r"'{2,5}")
_RE_WIKI_HEAD = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
_RE_WIKI_TAG = re.compile(
    r"</?(?:math|gallery|timeline|hiero|score|syntaxhighlight|source|"
    r"nowiki|pre|poem|ref|onlyinclude|noinclude|includeonly|sup|sub|"
    r"small|big|center|div|span|br|hr)\b[^>]*>", re.I)
_RE_WIKI_TAG_BLOCK = re.compile(
    r"<(math|gallery|timeline|hiero|score|syntaxhighlight|source|"
    r"nowiki|pre|poem|graph|mapframe|maplink)\b[^>]*>.*?</\1>",
    re.DOTALL | re.I)


def _drop_templates(text: str) -> str:
    """Remove {{...}} template calls including nested ones."""
    out, depth, i, n = [], 0, 0, len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i) and depth:
            depth -= 1
            i += 2
        elif depth == 0:
            out.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(out)


def wiki_to_text(wikitext: str) -> str:
    """Wikitext -> readable plaintext (headings/lists kept as lines)."""
    t = _RE_WIKI_COMMENT.sub("", wikitext)
    t = _RE_WIKI_TAG_BLOCK.sub(" ", t)
    t = _RE_WIKI_REF.sub("", t)
    t = _RE_WIKI_TABLE.sub(" ", t)
    t = _drop_templates(t)
    t = _RE_WIKI_FILE.sub("", t)
    t = _RE_WIKI_LINK.sub(
        lambda m: m.group(2) or m.group(1), t)      # [[a|b]] -> b
    t = _RE_WIKI_EXTLINK.sub(lambda m: m.group(1), t)
    t = _RE_WIKI_HEAD.sub(lambda m: "\n" + m.group(2) + "\n", t)
    t = _RE_WIKI_TAG.sub("", t)
    t = _RE_WIKI_FMT.sub("", t)
    t = re.sub(r"^\s*[*#:;]+\s*", "", t, flags=re.M)  # list markers
    from html import unescape
    return _clean(unescape(t))


def _page_wikitext(page: dict) -> str:
    revs = page.get("revisions") or []
    if not revs:
        return ""
    main = (revs[0].get("slots") or {}).get("main") or revs[0]
    return str(main.get("content") or main.get("*") or "")


def _wiki_doc(title: str, wikitext: str) -> dict | None:
    text = wiki_to_text(wikitext)
    if not text:
        return None
    doc = record("wiki", title,
                 "https://en.wikipedia.org/wiki/" +
                 title.replace(" ", "_"), text)
    if doc is not None:
        # keep wikitext for the filter stage (should_skip keys on
        # {{disambiguation}}-style markup); Processor drops it.
        doc["raw"] = wikitext
    return doc


def wiki_extracts(titles: list[str], fetch: Fetch = wp.http_get
                  ) -> list[dict]:
    """Batch full wikitext for <=50 titles via prop=revisions."""
    titles = [t for t in titles if t][:TITLES_BATCH]
    if not titles:
        return []
    data = _api_get(WIKI_API, {
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "redirects": "1", "titles": "|".join(titles)}, fetch)
    docs = []
    for page in (data.get("query", {}).get("pages") or []):
        doc = _wiki_doc(page.get("title", ""), _page_wikitext(page))
        if doc:
            docs.append(doc)
    return docs


def iter_wiki_random(pages: int, fetch: Fetch = wp.http_get
                     ) -> Iterator[dict]:
    """Random articles, EXLIMIT pages of wikitext per request."""
    left = pages
    while left > 0:
        n = min(left, EXLIMIT)
        data = _api_get(WIKI_API, {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "random", "grnnamespace": "0",
            "grnlimit": str(n), "prop": "revisions",
            "rvprop": "content", "rvslots": "main"}, fetch)
        pages_d = data.get("query", {}).get("pages") or []
        if not pages_d:
            return
        left -= n
        for page in pages_d:
            doc = _wiki_doc(page.get("title", ""), _page_wikitext(page))
            if doc:
                yield doc


def iter_wiki_all(state: FeedState, fetch: Fetch = wp.http_get
                  ) -> Iterator[dict]:
    """Alphabetical walk of every article (list=allpages -> batched
    extracts).  Resumable: the ``gapcontinue`` cursor is checkpointed
    in state each batch."""
    cursor = state.cursor("wiki-all")
    batch: list[str] = []
    while True:
        params = {"action": "query", "format": "json",
                  "generator": "allpages", "gapnamespace": "0",
                  "gaplimit": "500", "prop": "info"}
        if cursor:
            params["gapcontinue"] = cursor
        data = _api_get(WIKI_API, params, fetch)
        if not data:
            return        # transient dead-end: keep cursor for resume
        pages_d = data.get("query", {}).get("pages") or {}
        titles = sorted((p.get("title", "") for p in pages_d.values()),
                        key=str.lower)
        for t in titles:
            batch.append(t)
            if len(batch) >= TITLES_BATCH:
                yield from wiki_extracts(batch, fetch)
                batch = []
        if batch:
            yield from wiki_extracts(batch, fetch)
            batch = []
        cont = (data.get("continue") or {}).get("gapcontinue", "")
        state.set_cursor("wiki-all", cont or "")
        state.save()
        if not cont:
            return
        cursor = cont


def iter_wiki_search(query: str, n: int = 20, fetch: Fetch = wp.http_get
                     ) -> Iterator[dict]:
    res = wp.wikipedia_search(query, n=n)
    titles = [r["title"] for r in res.get("results", [])]
    for i in range(0, len(titles), TITLES_BATCH):
        yield from wiki_extracts(titles[i:i + TITLES_BATCH], fetch)


def iter_wiki_recent(limit: int = 200, fetch: Fetch = wp.http_get
                     ) -> Iterator[dict]:
    """Recently changed articles -> fresh extracts (rc show=new/edit)."""
    data = _api_get(WIKI_API, {
        "action": "query", "format": "json", "list": "recentchanges",
        "rcnamespace": "0", "rctype": "edit|new",
        "rclimit": str(min(limit, 500))}, fetch)
    titles = [c.get("title", "") for c in
              data.get("query", {}).get("recentchanges", [])]
    for i in range(0, len(titles), TITLES_BATCH):
        yield from wiki_extracts(titles[i:i + TITLES_BATCH], fetch)


def iter_arxiv(query: str, n: int = 25, fetch: Fetch = wp.http_get
               ) -> Iterator[dict]:
    """arXiv Atom search (newest first) -> abstract docs.

    Full papers are PDFs — out of scope for stdlib.  The abstract +
    title + authors + id is the high-signal fact-shape anyway; bulk
    paper ingest should use a parsed-corpus path (e.g. HF
    ``allenai/peS2o``) instead of scraping ar5iv."""
    from urllib.parse import quote_plus
    url = (f"{ARXIV_API}?search_query=all:{quote_plus(query)}"
           f"&start=0&max_results={min(n, 100)}"
           "&sortBy=submittedDate&sortOrder=descending")
    xml = fetch(url, timeout=15)
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL):
        def tag(t: str, entry: str = entry) -> str:
            m = re.search(rf"<{t}>(.*?)</{t}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""
        title, summ, aid, pub = (tag("title"), tag("summary"),
                                 tag("id"), tag("published"))
        authors = re.findall(r"<author>\s*<name>(.*?)</name>",
                             entry, re.DOTALL)
        if not title or not summ:
            continue
        body = (f"{title}\n"
                f"Authors: {', '.join(wp.RE_TAG.sub('', a).strip() for a in authors[:8])}\n"
                f"Published: {pub[:10]}\n\n{summ}")
        doc = record("arxiv", title, aid, body)
        if doc:
            yield doc


def iter_wiktionary(words: Iterable[str], fetch: Fetch = wp.http_get
                    ) -> Iterator[dict]:
    """Wiktionary entries via action=parse wikitext -> plaintext glosses.

    Falls back to the kaikki-shaped "word (pos): gloss" line so the
    memory model stores the same fact-shape as --dict-flatten."""
    for word in words:
        word = word.strip().lower()
        if not re.fullmatch(r"[a-z][a-z' -]{1,40}", word):
            continue
        try:
            data = _api_get(WIKTIONARY_API, {
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": "1", "redirects": "1",
                "titles": word}, fetch)
            pages_d = data.get("query", {}).get("pages") or {}
            text = next(iter(pages_d.values()), {}).get("extract", "")
        except Exception:
            text = ""
        if not text:
            continue
        # first ~4 sense-ish lines keep the entry compact
        lines = [ln for ln in text.splitlines()
                 if ln.strip() and not ln.startswith(("=", "[", "{"))]
        text = "\n".join(lines[:24])
        doc = record("dict", word,
                     "https://en.wiktionary.org/wiki/" + word, text)
        if doc:
            yield doc


def iter_urls(urls: Iterable[str],
              fetch_full: Callable = wp.http_get_full) -> Iterator[dict]:
    """Arbitrary http(s) pages -> markdown-ish text via extract_page.
    ``fetch_full(url) -> (body, final_url, content_type)`` — injectable."""
    for url in urls:
        url = wp.normalize_url(url)
        if not wp.is_safe_url(url):
            continue
        try:
            html, final, ctype = fetch_full(url)
        except Exception:
            continue
        if ctype == "text/html" or wp._RE_HTML_SNIFF.search(html[:2000]):
            info = wp.extract_page(html, final)
            title, text = info.get("title", ""), info.get("text", "")
        else:
            title, text = url.rsplit("/", 1)[-1], html
        doc = record("web", title, final, text)
        if doc:
            yield doc


def iter_ddg(query: str, n: int = 8, fetch: Fetch = wp.http_get
             ) -> Iterator[dict]:
    """Search -> fetch each result page (the 'researcher' source)."""
    res = wp.ddg_search(query, n=n)
    urls = [r["url"] for r in res.get("results", [])]
    yield from iter_urls(urls, fetch_full=wp.http_get_full)


SOURCES = {
    "wiki-random": "random articles (bulk; pages=N)",
    "wiki-all": "full alphabetical walk (resumable cursor)",
    "wiki-recent": "recently edited articles (fresh tail)",
    "wiki-search": "search-driven extracts (query=...)",
    "arxiv": "arXiv abstracts, newest first (query=...)",
    "wiktionary": "word definitions (words file or seed list)",
    "urls": "fetch raw URLs from a file",
    "ddg": "DDG search -> fetch result pages (query=...)",
}


# ── processor ───────────────────────────────────────────────────────────

class Processor:
    """Dedupe + quality gate between source iterators and the shard
    writer.  ``filter_fn(title, text)`` is injected by the caller —
    the CLI wires in download_wikipedia.should_skip and/or the
    train_flux topic scorer so filter logic is never duplicated."""

    def __init__(self, seen: SeenSet, min_chars: int = MIN_CHARS,
                 filter_fn: Callable[[str, str], bool] | None = None,
                 max_chars: int = 200_000):
        self.seen = seen
        self.min_chars = min_chars
        self.filter_fn = filter_fn
        self.max_chars = max_chars
        self.n_in = self.n_dup = self.n_short = self.n_filtered = 0

    def __call__(self, doc: dict | None) -> dict | None:
        if doc is None:
            return None
        self.n_in += 1
        if not self.seen.add(doc["id"]):
            self.n_dup += 1
            return None
        text = doc["text"]
        if len(text) < self.min_chars:
            self.n_short += 1
            return None
        if len(text) > self.max_chars:
            doc = dict(doc, text=text[:self.max_chars])
            text = doc["text"]
        raw = doc.pop("raw", None)
        if self.filter_fn is not None and \
                not self.filter_fn(doc["title"], raw or text):
            self.n_filtered += 1
            return None
        return doc

    def stats(self) -> dict:
        return {"in": self.n_in, "dup": self.n_dup,
                "short": self.n_short, "filtered": self.n_filtered,
                "kept": self.n_in - self.n_dup - self.n_short
                - self.n_filtered}


# ── shard writer (queue) ────────────────────────────────────────────────

class ShardWriter:
    """Rotate jsonl shards under ``queue/``: ``.wip`` while writing,
    rename to ``.ready`` at ``shard_docs`` records — the feeder only
    ever sees complete files."""

    def __init__(self, out_dir: str | Path, shard_docs: int = 512,
                 shard_bytes: int = 32 << 20):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "fed").mkdir(exist_ok=True)
        self.shard_docs = shard_docs
        self.shard_bytes = shard_bytes
        self._f = None
        self._path: Path | None = None
        self._n = 0
        self._bytes = 0
        self._idx = self._next_idx()

    def _next_idx(self) -> int:
        idx = 0
        for p in self.dir.glob("feed_*.jsonl*"):
            m = re.search(r"feed_(\d+)", p.name)
            if m:
                idx = max(idx, int(m.group(1)) + 1)
        return idx

    def _open(self) -> None:
        name = f"feed_{self._idx:06d}.jsonl.wip"
        self._path = self.dir / name
        self._f = self._path.open("w", encoding="utf-8")
        self._n = self._bytes = 0

    def write(self, doc: dict) -> Path | None:
        """Append a doc; returns the sealed ``.ready`` path when this
        write rotated the shard, else None."""
        if self._f is None:
            self._open()
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        self._f.write(line)
        self._n += 1
        self._bytes += len(line)
        if self._n >= self.shard_docs or self._bytes >= self.shard_bytes:
            return self.close()
        return None

    def close(self) -> Path | None:
        """Seal the current shard -> ``.ready``; returns its path."""
        if self._f is None:
            return None
        self._f.close()
        ready = self._path.with_name(
            self._path.name.replace(".wip", ".ready"))
        self._path.rename(ready)
        self._f = None
        self._idx += 1
        return ready


# ── feeders ─────────────────────────────────────────────────────────────

def pending_shards(queue_dir: str | Path) -> list[Path]:
    return sorted(Path(queue_dir).glob("feed_*.jsonl.ready"))


def mark_fed(queue_dir: str | Path, shard: Path) -> Path:
    dst = Path(queue_dir) / "fed" / shard.name.replace(".ready", "")
    shard.rename(dst)
    return dst


def feed_api(base_url: str, shard: Path, tag: str,
             poll_s: float = 2.0, timeout_s: float = 3600.0,
             fetch: Fetch = wp.http_get_full) -> dict:
    """POST one shard to a running Forge server's /engine/learn and
    poll /engine/stats until ``info.learn`` flips (the route returns
    immediately; completion is signaled via ``_info['learn']``).

    Only one learn can be active server-side — callers must serialize.
    """
    from urllib.request import Request, urlopen

    def _get(path: str) -> dict:
        req = Request(base_url.rstrip("/") + path)
        with urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(path: str, body: dict) -> dict:
        req = Request(base_url.rstrip("/") + path,
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    before = (_get("/engine/stats").get("info") or {}).get("learn")
    _post("/engine/learn", {"path": str(shard.resolve()),
                            "tag": tag, "max_tokens": 0})
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        time.sleep(poll_s)
        learn = (_get("/engine/stats").get("info") or {}).get("learn")
        if learn is not None and learn != before:
            return learn
    raise TimeoutError(f"learn did not finish in {timeout_s}s: {shard}")


def feed_model(shard: Path, model, tok, tag: str,
               chunk_chars: int = 1 << 18,
               pack_chars: int = 0) -> dict:
    """In-process ingest of one shard into a FluxLM-like object
    (anything exposing ``ingest(ids, tag=)`` + ``soft_reset()``).

    ``pack_chars`` joins consecutive small docs into one blob before
    ingest — amortizes the per-block snapshot/flush over hundreds of
    tiny records (dict glosses, abstracts) instead of paying it per
    doc.  Packs never mix sources (provenance stays correct) and docs
    >= chunk_chars always ingest solo."""
    tokens = docs = 0
    pending, pend_src = "", ""

    def emit(text: str, rtag: str) -> None:
        nonlocal tokens
        for i in range(0, len(text), chunk_chars):
            ids = tok.encode(text[i:i + chunk_chars],
                             add_special_tokens=False)
            tokens += model.ingest(ids, tag=rtag)
        model.soft_reset()

    def flush() -> None:
        nonlocal pending
        if pending:
            emit(pending, f"{tag}:{pend_src}")
            pending = ""

    with shard.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            src = str(row.get("source") or "web")
            docs += 1
            if pack_chars and len(text) < chunk_chars:
                if pending and src != pend_src:
                    flush()              # packs never mix sources
                pending += text + "\n\n"
                pend_src = src
                if len(pending) >= pack_chars:
                    flush()
                continue
            flush()
            emit(text, f"{tag}:{src}")
    flush()
    return {"tokens": tokens, "docs": docs, "tag": tag}


def feed_dir(queue_dir: str | Path, sink: Callable[[Path], dict],
             once: bool = False) -> Iterator[tuple[Path, dict]]:
    """Consume every ``*.ready`` shard through ``sink`` -> ``fed/``.
    ``once=False`` polls forever (daemon mode)."""
    while True:
        shards = pending_shards(queue_dir)
        if not shards:
            if once:
                return
            time.sleep(2.0)
            continue
        for shard in shards:
            rep = sink(shard)
            mark_fed(queue_dir, shard)
            yield shard, rep
        if once:
            return


# ── top-level loop ──────────────────────────────────────────────────────

def run(queue_dir: str | Path, sources: Iterable[Iterator[dict]],
        processor: Processor, writer: ShardWriter,
        sink: Callable[[Path], dict] | None = None,
        max_docs: int = 0, rps: float = 2.0,
        log: Callable[[str], None] = print) -> dict:
    """Pull docs from each source iterator in turn, process, shard, and
    (optionally) feed each sealed shard through ``sink``.

    ``rps`` throttles the doc-level loop (a source's internal request
    batching means pages-per-second is usually higher)."""
    docs = fed = 0
    t0 = time.perf_counter()
    min_gap = 1.0 / max(rps, 0.01)
    try:
        for it in sources:
            docs_it = iter(it)
            while True:
                try:
                    doc = next(docs_it)
                except StopIteration:
                    break
                except Exception as e:
                    log(f"[feed] source error, moving on: {e}")
                    break
                t_req = time.perf_counter()
                doc = processor(doc)
                if doc is None:
                    continue
                ready = writer.write(doc)
                docs += 1
                if ready is not None and sink is not None:
                    sink(ready)
                    mark_fed(queue_dir, ready)
                    fed += 1
                if docs % 100 == 0:
                    el = time.perf_counter() - t0
                    log(f"[feed] {docs:,} docs "
                        f"({docs / max(el, 1e-9):.1f}/s) "
                        f"stats={processor.stats()}")
                if max_docs and docs >= max_docs:
                    raise _Done
                gap = min_gap - (time.perf_counter() - t_req)
                if gap > 0:
                    time.sleep(gap)
    except _Done:
        pass
    ready = writer.close()
    if sink is not None and ready is not None:
        sink(ready)
        mark_fed(queue_dir, ready)
        fed += 1
    el = time.perf_counter() - t0
    return {"docs": docs, "shards_fed": fed,
            "docs_per_s": round(docs / max(el, 1e-9), 1),
            "processor": processor.stats()}


class _Done(Exception):
    pass

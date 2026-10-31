"""Offline tests for forge.scrape_feed — every fetch is injected, no
network.  Covers record normalization, dedupe persistence, processor
gates, shard rotation, queue contract, and the run() loop."""
import json

from forge import scrape_feed as sf


def _doc(text="a" * 800, **kw):
    return sf.record(kw.pop("source", "wiki"),
                     kw.pop("title", "T"), kw.pop("url", "u"),
                     text, **kw)


class TestRecord:
    def test_normalizes_and_ids(self):
        d = _doc(text="x " * 600 + "\n\n\n\n tail")
        assert d["id"] and d["source"] == "wiki"
        assert "\n\n\n" not in d["text"]
        # same content -> same id
        assert _doc(text="x " * 600 + "\n\n\n\n tail")["id"] == d["id"]

    def test_empty_text_rejected(self):
        assert sf.record("wiki", "T", "u", "   \n  ") is None


class TestSeenSet:
    def test_dedupe_persists(self, tmp_path):
        p = tmp_path / "seen.txt"
        s = sf.SeenSet(p)
        assert s.add("abc") is True
        assert s.add("abc") is False
        del s
        s2 = sf.SeenSet(p)          # reload from disk
        assert "abc" in s2._set
        assert s2.add("abc") is False
        assert len(s2) == 1


class TestProcessor:
    def test_gates(self, tmp_path):
        seen = sf.SeenSet(tmp_path / "s.txt")
        p = sf.Processor(seen, min_chars=100, filter_fn=lambda t, x: "ok" in t)
        kept = _doc(title="ok page", url="u1")
        assert p(kept) is not None
        assert p(_doc(title="ok dup", url="u1",
                      text=kept["text"])) is None                # dup id
        assert p(_doc(title="ok short", url="u2",
                      text="tiny")) is None
        assert p(_doc(title="filtered out", url="u3")) is None
        st = p.stats()
        assert st["kept"] == 1 and st["dup"] == 1 \
            and st["short"] == 1 and st["filtered"] == 1

    def test_max_chars_clip(self, tmp_path):
        p = sf.Processor(sf.SeenSet(tmp_path / "s.txt"),
                         min_chars=10, max_chars=100)
        d = p(_doc(text="z" * 5000))
        assert len(d["text"]) == 100


class TestShardWriter:
    def test_rotation_and_ready_contract(self, tmp_path):
        w = sf.ShardWriter(tmp_path, shard_docs=3)
        for i in range(7):
            w.write(_doc(title=f"ok {i}", text=f"t{i}" * 400))
        ready = list(tmp_path.glob("feed_*.jsonl.ready"))
        assert len(ready) == 2                     # 3 + 3 sealed
        wips = list(tmp_path.glob("*.wip"))
        assert len(wips) == 1                      # 1 in flight
        w.close()
        assert len(list(tmp_path.glob("feed_*.jsonl.ready"))) == 3
        rows = (tmp_path / "feed_000000.jsonl.ready").read_text()
        assert len(rows.strip().splitlines()) == 3
        # resume: next writer continues the index, doesn't clobber
        w2 = sf.ShardWriter(tmp_path, shard_docs=1)
        w2.write(_doc(title="ok new", text="n" * 600))
        w2.close()
        assert (tmp_path / "feed_000003.jsonl.ready").exists()


class _FakeModel:
    def __init__(self):
        self.calls = []

    def ingest(self, ids, tag=""):
        self.calls.append((len(ids), tag))
        return len(ids)

    def soft_reset(self):
        pass


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 4)))


class TestFeeders:
    def _shard(self, tmp_path, n=3):
        p = tmp_path / "feed_000000.jsonl.ready"
        with p.open("w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"text": "d" * 600,
                                    "source": "wiki",
                                    "title": f"t{i}"}) + "\n")
        return p

    def test_feed_model(self, tmp_path):
        m, p = _FakeModel(), self._shard(tmp_path)
        rep = sf.feed_model(p, m, _FakeTok(), "scrape")
        assert rep["docs"] == 3 and rep["tokens"] > 0
        assert all(tag.startswith("scrape:wiki") for _, tag in m.calls)

    def test_feed_model_packing(self, tmp_path):
        # 6 same-source tiny docs -> ONE packed ingest
        m, p = _FakeModel(), self._shard(tmp_path, n=6)
        rep = sf.feed_model(p, m, _FakeTok(), "scrape",
                            pack_chars=1 << 18)
        assert rep["docs"] == 6 and len(m.calls) == 1

        # source switch splits the pack (provenance stays exact)
        p2 = tmp_path / "feed_000001.jsonl.ready"
        with p2.open("w", encoding="utf-8") as f:
            for i, src in enumerate(["wiki", "wiki", "arxiv", "arxiv"]):
                f.write(json.dumps({"text": "d" * 600, "source": src,
                                    "title": f"t{i}"}) + "\n")
        m2 = _FakeModel()
        sf.feed_model(p2, m2, _FakeTok(), "scrape",
                      pack_chars=1 << 18)
        tags = [t for _, t in m2.calls]
        assert tags == ["scrape:wiki", "scrape:arxiv"]

    def test_feed_dir_consumes_ready(self, tmp_path):
        self._shard(tmp_path)
        (tmp_path / "fed").mkdir()
        out = list(sf.feed_dir(tmp_path, lambda s: {"ok": True},
                               once=True))
        assert len(out) == 1
        assert not sf.pending_shards(tmp_path)
        assert (tmp_path / "fed" / "feed_000000.jsonl").exists()


class TestWikiToText:
    def test_strips_markup(self):
        wt = ("'''Alpha''' is a [[letter|Greek letter]].<ref>cite</ref>\n"
              "== History ==\n* first item\n{{infobox|x=1}}\n"
              "[[File:pic.jpg]] [http://x.com a link]\n"
              "<!-- note --> {| class='t'|row|}")
        t = sf.wiki_to_text(wt)
        assert "Alpha is a Greek letter." in t
        assert "History" in t and "first item" in t and "a link" in t
        assert "cite" not in t and "infobox" not in t
        assert "pic.jpg" not in t and "note" not in t
        assert "[[" not in t and "{{" not in t

    def test_nested_templates_dropped(self):
        assert "{{" not in sf.wiki_to_text("a {{x {{y|z}} }} b")


class TestSources:
    WIKI_RESP = json.dumps({"query": {"pages": [
        {"title": "Alpha", "revisions": [
            {"slots": {"main": {"content": "alpha text " * 100}}}]},
        {"title": "Beta", "revisions": [
            {"slots": {"main": {"content": "beta text " * 100}}}]},
    ]}})

    def test_wiki_extracts(self):
        docs = sf.wiki_extracts(["Alpha", "Beta"],
                                fetch=lambda url, **k: self.WIKI_RESP)
        assert {d["title"] for d in docs} == {"Alpha", "Beta"}
        assert all(d["source"] == "wiki" for d in docs)
        assert "wikipedia.org/wiki/Alpha" in docs[0]["url"]

    def test_wiki_random_uses_generator(self):
        calls = []

        def fetch(url, **k):
            calls.append(url)
            return self.WIKI_RESP
        docs = list(sf.iter_wiki_random(5, fetch=fetch))
        assert len(docs) == 2
        assert "generator=random" in calls[0]

    def test_arxiv_parses_atom(self):
        xml = """<feed><entry>
          <title>Attention Is All You Need</title>
          <summary>We propose the transformer...</summary>
          <id>http://arxiv.org/abs/1706.03762</id>
          <published>2017-06-12T00:00:00Z</published>
          <author><name>Ashish Vaswani</name></author>
        </entry></feed>"""
        docs = list(sf.iter_arxiv("attention",
                                  fetch=lambda url, **k: xml))
        assert len(docs) == 1
        d = docs[0]
        assert d["source"] == "arxiv"
        assert "transformer" in d["text"].lower()
        assert "Vaswani" in d["text"]
        assert "1706.03762" in d["url"]

    def test_urls_extracts_html(self):
        html = ("<html><head><title>P</title></head><body>"
                "<p>" + "content " * 200 + "</p></body></html>")

        def fetch_full(url, **k):
            return html, url, "text/html"
        docs = list(sf.iter_urls(["example.com/x"],
                                 fetch_full=fetch_full))
        assert len(docs) == 1
        assert docs[0]["source"] == "web"
        assert "content" in docs[0]["text"]

    def test_urls_skip_unsafe(self):
        assert list(sf.iter_urls(["javascript:alert(1)"])) == []

    def test_api_get_retries_then_empty(self):
        calls = []

        def flaky(url, **k):
            calls.append(1)
            if len(calls) < 3:
                raise OSError("transient")
            return '{"query": {"pages": []}}'
        assert sf._api_get("http://x", {}, flaky) == \
            {"query": {"pages": []}}
        assert len(calls) == 3

        def dead(url, **k):
            raise OSError("down")
        assert sf._api_get("http://x", {}, dead, retries=2) == {}


class TestRun:
    def test_end_to_end(self, tmp_path):
        seen = sf.SeenSet(tmp_path / "seen.txt")
        proc = sf.Processor(seen, min_chars=50)
        w = sf.ShardWriter(tmp_path, shard_docs=2)
        src = iter(_doc(title=f"ok {i}", text=f"body{i} " * 80)
                   for i in range(5))
        fed = []
        rep = sf.run(tmp_path, [src], proc, w,
                     sink=lambda p: fed.append(p) or {"ok": 1},
                     rps=1000, log=lambda s: None)
        assert rep["docs"] == 5
        # 2 sealed shards fed mid-run + final shard fed at close
        assert len(fed) >= 2
        assert not sf.pending_shards(tmp_path)
        assert len(list((tmp_path / "fed").glob("*.jsonl"))) >= 3

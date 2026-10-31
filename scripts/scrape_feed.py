"""Scrape online sources -> processed jsonl shards -> feed the model.

Three output modes:

    # 1. build a corpus queue only (train later with train_flux.py)
    python scripts/scrape_feed.py --sources wiki-random,arxiv \
        --query "machine learning" --pages 2000 \
        --queue research/data/feed --to shards

    # 2. feed a running Forge GUI server (load a .flux model first)
    python scripts/scrape_feed.py --sources wiki-recent,wiki-search \
        --query "quantum computing" --topic science \
        --queue research/data/feed --to api --api http://127.0.0.1:7860

    # 3. ingest straight into a .flux checkpoint in-process
    python scripts/scrape_feed.py --sources wiki-random --pages 500 \
        --queue research/data/feed --to model \
        --model research/checkpoints/flux/model.flux --device cuda

    # feed an existing queue without scraping
    python scripts/scrape_feed.py --queue research/data/feed \
        --to api --api http://127.0.0.1:7860

Sources: wiki-random | wiki-all | wiki-recent | wiki-search | arxiv |
wiktionary | urls | ddg.  Dedupe is persistent (seen.txt); wiki-all is
resumable (state.json cursor).  Provenance: every shard is ingested
under tag "<tag>:<source>" so a bad source is one revert_tag away.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings

from forge import scrape_feed as sf


def _seed_words() -> list[str]:
    from forge.self_play.flux_goals import _DICT_SEED_WORDS
    return list(_DICT_SEED_WORDS)


def _build_sources(args, state):
    srcs = []
    for name in args.sources.split(","):
        name = name.strip()
        if name == "wiki-random":
            srcs.append(sf.iter_wiki_random(args.pages))
        elif name == "wiki-all":
            srcs.append(sf.iter_wiki_all(state))
        elif name == "wiki-recent":
            srcs.append(sf.iter_wiki_recent(limit=args.pages))
        elif name == "wiki-search":
            if not args.query:
                raise SystemExit("wiki-search needs --query")
            srcs.append(sf.iter_wiki_search(args.query, n=args.pages))
        elif name == "arxiv":
            if not args.query:
                raise SystemExit("arxiv needs --query")
            srcs.append(sf.iter_arxiv(args.query, n=args.pages))
        elif name == "wiktionary":
            words = ([ln.strip() for ln in
                      Path(args.words).read_text(encoding="utf-8")
                      .splitlines() if ln.strip()]
                     if args.words else _seed_words())
            srcs.append(sf.iter_wiktionary(words))
        elif name == "urls":
            if not args.urls:
                raise SystemExit("urls needs --urls <file>")
            srcs.append(sf.iter_urls(
                ln.strip() for ln in
                Path(args.urls).read_text(encoding="utf-8")
                .splitlines() if ln.strip()))
        elif name == "ddg":
            if not args.query:
                raise SystemExit("ddg needs --query")
            srcs.append(sf.iter_ddg(args.query, n=args.pages))
        else:
            raise SystemExit(f"unknown source: {name} "
                             f"(choices: {sorted(sf.SOURCES)})")
    return srcs


def _build_filter(args):
    """Compose wiki-filter (should_skip) + topic scorer -> accept fn."""
    fns = []
    if args.wiki_filter:
        try:
            from download_wikipedia import should_skip
            fns.append(lambda t, x: not should_skip(t, x))
        except ImportError:
            print("warn: download_wikipedia.py not importable — "
                  "--wiki-filter ignored")
    if args.topic:
        from train_flux import _resolve_topics, _topic_scorer
        fns.append(_topic_scorer(_resolve_topics(args.topic)))
    if not fns:
        return None
    return lambda title, text: all(fn(title, text) for fn in fns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="",
                    help="comma list — empty = feed existing queue only")
    ap.add_argument("--queue", default="research/data/feed",
                    help="shard queue dir (the scrape/feed bus)")
    ap.add_argument("--to", choices=["shards", "api", "model"],
                    default="shards")
    ap.add_argument("--api", default="http://127.0.0.1:7860")
    ap.add_argument("--model", default=None, help=".flux ckpt for --to model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cuda-primary", action="store_true")
    ap.add_argument("--tag", default="scrape",
                    help="journal tag prefix -> '<tag>:<source>'")
    ap.add_argument("--pages", type=int, default=200,
                    help="pages/rows per bounded source")
    ap.add_argument("--query", default="")
    ap.add_argument("--words", default=None, help="wordlist file for wiktionary")
    ap.add_argument("--urls", default=None, help="url list file for urls")
    ap.add_argument("--topic", default=None,
                    help="domain filter, same syntax as train_flux --topic")
    ap.add_argument("--wiki-filter", action="store_true")
    ap.add_argument("--min-chars", type=int, default=sf.MIN_CHARS)
    ap.add_argument("--shard-docs", type=int, default=512)
    ap.add_argument("--pack-docs", type=int, default=0, metavar="CHARS",
                    help="pack small docs into CHARS blobs before "
                         "ingest (--to model; e.g. 262144) — big win "
                         "on tiny records like dict glosses")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--rps", type=float, default=2.0,
                    help="doc-level throttle (sources batch internally)")
    args = ap.parse_args()

    queue = Path(args.queue)
    queue.mkdir(parents=True, exist_ok=True)
    seen = sf.SeenSet(queue / "seen.txt")
    state = sf.FeedState(queue / "state.json")
    proc = sf.Processor(seen, min_chars=args.min_chars,
                        filter_fn=_build_filter(args))

    sink = None
    if args.to == "api":
        def sink(p):
            rep = sf.feed_api(args.api, p, tag=args.tag)
            print(f"    [fed] {p.name}: {rep}")
            return rep
    elif args.to == "model":
        if not args.model:
            ap.error("--to model needs --model <path.flux>")
        from forge.engine.engine_common import _tokenizer_for_vocab
        from forge.model.flux import FluxLM
        from research.tokenizer_cache import get_tokenizer
        model = FluxLM.load(args.model, device=args.device,
                            cuda_primary=args.cuda_primary or None)
        tok = get_tokenizer(_tokenizer_for_vocab(model.vocab_size))
        print(f"model: {args.model} vocab={model.vocab_size}")

        def sink(p):
            rep = sf.feed_model(p, model, tok, tag=args.tag,
                                pack_chars=args.pack_docs)
            model.snapshot(args.model)
            print(f"    [fed] {p.name}: {rep['docs']} docs "
                  f"{rep['tokens']:,} tok")
            return rep

    if args.sources:
        sources = _build_sources(args, state)
        rep = sf.run(queue, sources, proc,
                     sf.ShardWriter(queue, shard_docs=args.shard_docs),
                     sink=sink, max_docs=args.max_docs, rps=args.rps)
        state.save()
        print(f"done: {rep}")
    else:
        # feed-only pass over the existing queue
        if sink is None:
            n = len(sf.pending_shards(queue))
            print(f"{n} shard(s) queued under {queue}")
            return 0
        for shard, rep in sf.feed_dir(queue, sink, once=True):
            print(f"    [fed] {shard.name}: {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Filter Wikipedia JSONL to keep only STEM/coding/logic/reasoning articles.

Removes politics, geography, entertainment, sports, religion, biographies of
non-scientists, etc. Keeps computer science, mathematics, physics, chemistry,
biology, engineering, logic, algorithms, AI/ML, and related topics.

Strategy:
  1. Title-based filter: reject titles matching exclusion patterns (politicians,
     sports figures, geographic places, movies/songs/TV shows, etc.)
  2. Content-based filter: score the article text by counting STEM keyword
     occurrences. Keep articles with a high STEM signal.
  3. Category inference: check for Wikipedia category markers in the text
     (e.g. "This article is about a ...", "is a film/song/game", etc.)

Usage:
  python scripts/filter_wikipedia.py
  python scripts/filter_wikipedia.py --input research/data/pretrain/wikipedia --output research/data/pretrain/wikipedia_filtered
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── STEM inclusion keywords (high signal for coding/reasoning) ──
STEM_KEYWORDS = {
    # Computer science / programming
    "algorithm", "programming", "software", "computer", "code", "compiler",
    "data structure", "function", "variable", "debug", "api", "framework",
    "library", "runtime", "bytecode", "interpreter", "syntax", "semantics",
    "recursion", "iteration", "loop", "array", "hash", "tree", "graph",
    "pointer", "memory", "cache", "buffer", "thread", "process", "kernel",
    "operating system", "database", "query", "index", "transaction", "sql",
    "python", "javascript", "java", "rust", "golang", "typescript", "c++",
    "assembly", "machine code", "instruction set", "register", "pipeline",
    "parallel", "concurrent", "asynchronous", "distributed", "protocol",
    "tcp", "udp", "http", "dns", "encryption", "cipher", "key", "hash",
    "digital signature", "certificate", "authentication", "authorization",
    "machine learning", "neural network", "deep learning", "tensor",
    "gradient", "optimization", "loss function", "backpropagation",
    "transformer", "attention", "embedding", "tokenization", "language model",
    "artificial intelligence", "reinforcement learning", "inference",
    "training", "dataset", "feature", "classification", "regression",
    "clustering", "dimensionality", "vector", "matrix", "linear algebra",
    # Mathematics
    "mathematics", "theorem", "proof", "lemma", "corollary", "axiom",
    "algebra", "calculus", "geometry", "topology", "trigonometry",
    "probability", "statistics", "combinatorics", "number theory",
    "differential", "integral", "derivative", "equation", "polynomial",
    "function", "limit", "convergence", "series", "sequence", "set theory",
    "logic", "boolean", "proposition", "predicate", "quantifier",
    "inference", "deduction", "induction", "abduction", "syllogism",
    "discrete mathematics", "graph theory", "game theory", "optimization",
    "linear programming", "constraint", "objective", "feasible",
    "modular", "prime", "factor", "divisor", "congruence", "permutation",
    "combination", "binomial", "exponential", "logarithm", "matrix",
    "eigenvalue", "eigenvector", "determinant", "inverse", "transpose",
    # Physics / Chemistry / Biology (reasoning-relevant)
    "physics", "quantum", "relativity", "mechanics", "thermodynamics",
    "entropy", "energy", "force", "momentum", "velocity", "acceleration",
    "particle", "wave", "frequency", "wavelength", "amplitude",
    "chemistry", "molecule", "atom", "electron", "proton", "neutron",
    "reaction", "catalyst", "bond", "compound", "element", "periodic",
    "biology", "cell", "dna", "rna", "protein", "enzyme", "gene",
    "evolution", "organism", "neuron", "synapse", "brain",
    # Engineering / Technology
    "engineering", "circuit", "signal", "frequency", "amplifier",
    "transistor", "semiconductor", "integrated circuit", "microprocessor",
    "digital", "analog", "sensor", "actuator", "robotics", "control system",
    "feedback", "stability", "oscillation", "resonance",
    # Logic / Philosophy of mind / cognition
    "cognition", "cognitive", "reasoning", "rational", "logic",
    "inference", "deductive", "inductive", "abductive",
    "philosophy of mind", "consciousness", "computation", "turing",
    "complexity", "computability", "decidability", "formal language",
    "automaton", "finite state", "pushdown", "turing machine",
}

# ── Exclusion patterns (title-based) ──
# These are checked against the title. If matched, the article is rejected
# unless it also has strong STEM content (high keyword density).
EXCLUDE_TITLE_PATTERNS = [
    # Politics
    r"\b(president|prime minister|senator|congress|parliament|election|"
    r"political party|democrat|republican|conservative|liberal|"
    r"socialist|communist|fascist|monarch|king|queen|prince|duke|"
    r"dynasty|empire|war|battle|treaty|revolution|coup|regime)\b",
    # Geography
    r"\b(country|nation|state|province|territory|county|city|town|"
    r"village|region|continent|island|mountain|river|lake|ocean|"
    r"sea|desert|valley|plateau|climate|weather|season)\b",
    # Sports
    r"\b(football|soccer|basketball|baseball|tennis|golf|boxing|"
    r"wrestling|swimming|cycling|skiing|skating|cricket|rugby|hockey|"
    r"olympic|championship|tournament|league|match|game|player|"
    r"athlete|coach|team|season|playoff|fixture)\b",
    # Entertainment
    r"\b(film|movie|cinema|actor|actress|director|producer|screenplay|"
    r"television|tv series|sitcom|drama|soap opera|reality show|"
    r"music|song|album|band|singer|musician|guitarist|drummer|"
    r"rapper|dj|composer|orchestra|symphony|concert|festival|"
    r"video game|gaming|playstation|xbox|nintendo|esports|"
    r"novel|fiction|poetry|poem|author|writer|novelist|character|"
    r"superhero|comic|manga|anime|cartoon)\b",
    # Religion / mythology
    r"\b(god|goddess|deity|temple|church|mosque|cathedral|bible|quran|"
    r"torah|prayer|worship|faith|saint|prophet|apostle|miracle|"
    r"mythology|legend|fable|folklore)\b",
    # Fashion / food / lifestyle
    r"\b(fashion|model|designer|clothing|dress|costume|jewelry|"
    r"recipe|cuisine|cooking|restaurant|chef|dish|ingredient|"
    r"wine|beer|cocktail|bar|cafe|hotel|tourism|travel)\b",
]

# ── Exclusion patterns (content-based, first 500 chars) ──
EXCLUDE_CONTENT_PATTERNS = [
    r"is a (film|movie|song|album|video game|television series|tv series)",
    r"is a (politician|political party|member of parliament|senator)",
    r"is a (footballer|soccer player|basketball player|athlete|sportsperson)",
    r"is a (singer|musician|actor|actress|director|producer|author|novelist)",
    r"is the (capital|largest city|second-largest|most populous) (of|in)",
    r"is a (country|sovereign state|nation|republic|monarchy|kingdom)",
    r"is a (geographic|geographical|topographical) (feature|region|area)",
    r"is a (cuisine|dish|recipe|food|beverage|drink|wine|cocktail)",
    r"is a (fashion|clothing|garment|accessory) (brand|item|designer)",
    r"is a (religion|religious|denomination|sect|faith|belief)",
    r"is a (mythological|legendary|fictional) (figure|character|creature)",
]

# Compile patterns
_EXCLUDE_TITLE_RE = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_TITLE_PATTERNS]
_EXCLUDE_CONTENT_RE = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_CONTENT_PATTERNS]


def is_stem_article(title: str, text: str) -> bool:
    """Decide if a Wikipedia article is STEM/coding/reasoning relevant.

    Returns True if the article should be kept.
    """
    # Quick rejection: very short articles
    if len(text) < 500:
        return False

    # Check content patterns (first 500 chars — usually the summary)
    summary = text[:500]
    for pat in _EXCLUDE_CONTENT_RE:
        if pat.search(summary):
            return False

    # Additional content-based exclusion: check for politics/war/geography
    # markers in the summary that the regex patterns might miss
    summary_lower = summary.lower()
    extra_exclude = [
        "demographics of", "armed forces", "military", "navy", "army",
        "air force", "civil war", "revolutionary war", "world war",
        "election", "political party", "government of", "parliament of",
        "geography of", "climate of", "economy of", "history of",
        "culture of", "religion in", "sport in", "music of",
        "cinema of", "literature of", "cuisine of",
    ]
    for marker in extra_exclude:
        if marker in summary_lower:
            return False

    # Check title patterns — if title matches exclusion, reject immediately
    # unless the title ALSO contains a strong STEM keyword (e.g. "Turing machine"
    # contains "machine" but is clearly STEM; "Quantum mechanics" has "mechanics"
    # but is physics). We check the title against STEM keywords first.
    # Use word-boundary matching to avoid false positives (e.g. "demographics"
    # contains "graph" but is not a STEM keyword match).
    title_lower = title.lower()
    title_words = set(re.findall(r'\b\w+\b', title_lower))
    has_stem_title = any(kw in title_lower for kw in STEM_KEYWORDS
                         if ' ' not in kw and kw in title_words
                         or ' ' in kw and kw in title_lower)

    if has_stem_title:
        # Title itself is STEM — keep it
        return True

    # Title is not STEM — check if it matches exclusion patterns
    for pat in _EXCLUDE_TITLE_RE:
        if pat.search(title):
            return False  # non-STEM title + matches exclusion → reject

    # Also reject titles starting with demographic/geographic/political markers
    title_markers = [
        "demographics of", "geography of", "history of", "economy of",
        "politics of", "culture of", "religion in", "sport in",
        "music of", "cinema of", "literature of", "cuisine of",
        "armed forces", "military of", "foreign relations",
        "list of countries", "list of cities", "list of rivers",
    ]
    for marker in title_markers:
        if title_lower.startswith(marker):
            return False

    # Title doesn't match exclusion but isn't STEM either.
    # Require strong STEM content in the article body to keep it.
    sample = text[:2000].lower()
    stem_count = sum(1 for kw in STEM_KEYWORDS if kw in sample)

    # Higher threshold: need at least 8 STEM keyword hits to keep a
    # non-STEM-titled article. This filters out tangentially-related content.
    return stem_count >= 8


def filter_shard(input_path: Path, output_path: Path,
                 stats: dict) -> int:
    """Filter one JSONL shard. Returns number of articles kept."""
    kept = 0
    rejected = 0
    with open(input_path, encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8", buffering=1024*1024) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    stats["parse_errors"] += 1
                    continue
                title = obj.get("title", "")
                text = obj.get("text", "")
                if is_stem_article(title, text):
                    fout.write(line + "\n")
                    kept += 1
                else:
                    rejected += 1
    stats["kept"] += kept
    stats["rejected"] += rejected
    return kept


def main():
    parser = argparse.ArgumentParser(description="Filter Wikipedia to STEM/coding articles")
    parser.add_argument("--input", default="research/data/pretrain/wikipedia",
                        help="Input directory with wikipedia_*.jsonl shards")
    parser.add_argument("--output", default="research/data/pretrain/wikipedia_filtered",
                        help="Output directory for filtered shards")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(input_dir.glob("wikipedia_*.jsonl"))
    if not shards:
        print(f"No wikipedia_*.jsonl files found in {input_dir}")
        return

    print(f"Filtering {len(shards)} shards from {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Keeping: STEM, coding, math, logic, algorithms, AI/ML, physics, chemistry, biology, engineering")
    print(f"Removing: politics, geography, sports, entertainment, religion, fashion, food\n")

    stats = {"kept": 0, "rejected": 0, "parse_errors": 0}
    t0 = time.time()

    for i, shard in enumerate(shards):
        out_shard = output_dir / shard.name
        kept = filter_shard(shard, out_shard, stats)
        elapsed = time.time() - t0
        total = stats["kept"] + stats["rejected"]
        pct = stats["kept"] / max(total, 1) * 100
        print(f"  [{i+1}/{len(shards)}] {shard.name}: kept {kept:>6d} "
              f"| total kept {stats['kept']:>7d} | rejected {stats['rejected']:>7d} "
              f"| {pct:.1f}% kept | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = stats["kept"] + stats["rejected"]
    pct = stats["kept"] / max(total, 1) * 100
    print(f"\n{'='*60}")
    print(f"  FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"  Input articles:  {total}")
    print(f"  Kept (STEM):     {stats['kept']} ({pct:.1f}%)")
    print(f"  Rejected:        {stats['rejected']} ({100-pct:.1f}%)")
    print(f"  Parse errors:    {stats['parse_errors']}")
    print(f"  Time:            {elapsed:.0f}s")
    print(f"  Output:          {output_dir}")


if __name__ == "__main__":
    main()

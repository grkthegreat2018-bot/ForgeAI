# STRATA vs FluxLM — division of labor (A12)

The competitive audit (COMPETITIVE_AUDIT_2026-10-31.md, A12) flags that
`forge/model/strata/` and `forge/model/flux/` are two live
non-transformer associative-memory families that duplicate: suffix-hash
order cells (same `_suffix_key` FNV rolling-hash construction over the
same 10-order set), a hedge-weighted geometric channel mixture
(AdaHedge, `backoff`/`recency`/`fatigue`/`uni` channels, sleeping-
experts abstention, `uni_escape`), `teach`/`write_counts`/`ingest`
position-free tagged writes, journaled tag/position revert
(`revert_tag`, `revert_since`, `ws`/`we` blocks), and snapshot/restore.
Per AGENTS.md rule E, two files doing the same thing needs a decision.

## What each actually is today

**FluxLM** (`forge/model/flux/`, R52→R70): the research vehicle. One
`FluxConfig` surface (~100 knobs) over ~25 channel families — order
cells, archived episodic `deep`, approximate `knn` fingerprint bank,
`topic`/`sem`/`csem` Hebbian readouts, `gcell`/`gsyn`/`gsd`/`dnet`
learned-key/delta-net channels, `rev`/`gv`, domain lanes, cold-tier
`FluxBank`, GPU `cuda_primary` fused paths, `FluxJournal` bit-exact
undo, `.flux` snapshots. Its moat (per A6): journaled bit-exact revert
+ `explain()`/`audit()`/`inspect_cell` provenance — a *revertible
memory layer*, positioned against agent-memory (mem0/MemGPT/Letta),
not against LMs. Head-to-head QA vs RAG:
`scripts/bench_flux_vs_rag.py`.

**STRATA** (`forge/model/strata/`, R60, M1): the deployment-artifact
format. `StrataLM` is a deliberately small CPU-eager reference
(~86 tok/s learn) — order cells + uni/backoff/recency/fatigue only —
whose real content is the *container*: WEIGHTS `.strata` (labeled
`CellBlock`s of packed 48B cells: tag32+meta+8×u32 followers+raw64,
log-quantized counts, 2-choice cuckoo hash, active/sealed/cold
lifecycle, per-block undo arenas, memmap container IO), PARAMS `.iq`
(seeded tensors stored as `{seed,shape}`), KNOBS `.cond` (named scalar
overlays). Design: docs/STRATA_M1_DESIGN.md.

## Decision

- **FluxLM = research vehicle.** New memory capabilities (channels,
  retrieval, revert semantics) land in `flux/` first, where the journal
  makes every experiment auditable and undoable. The journal/revert/
  explain surface is the product (A6 repositioning), not the LM score.
- **STRATA = deployment artifact.** Its value is the format — packed
  bounded cells, three-artifact separation, memmap load — not its
  channel set. It does NOT grow research channels.
- **Rule going forward:** a capability may exist in STRATA only after
  it has stabilized in FluxLM and been ported as packed cells; nothing
  is prototyped in STRATA. This kills the drift where both files evolve
  parallel implementations of the same idea.
- **Duplicated components** (order cells, hedge mixture, teach/ingest,
  journal/revert, snapshot): kept in both for now — research divergence
  allowed; no consolidation without a bit-exact comparison between the
  two implementations.

## Consolidation trigger

If by next round STRATA's container still ships no consumer that
`.flux` snapshots can't serve (no external artifact contract, no
footprint win that matters), fold `blocks.py`/`cells.py`/`io.py` into
`forge/model/flux/` as the packed-cell substrate and retire the
`strata` package — one substrate, one format.

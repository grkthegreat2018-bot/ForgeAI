# LiveT — A Transformer That Can Be Live-Trained In Anything That Matters

R&D round following the FLUX literature audit (.devin/scratchpad.md, 2025-10).
Verdict from that audit: every working test-time-memory system (Titans,
DeltaNet/Gated DeltaNet-2, LaCT, SR-TTT) trains the memory substrate
end-to-end with the backbone. Live learning works *inside* a trained model
as journaled state writes + fast-weight updates — not as a substitute for
a trained generator, and not as sequential edits to trained weights
(ROME/MEMIT degrade catastrophically under sequential edits).

## The split: what learns live vs. what stays frozen

| surface | mechanism | live? | why |
|---|---|---|---|
| embed / head | vocab codec | frozen | fluency prior; drift = vocab corruption |
| attention (QKVO) | exact short-term context | frozen | fluency prior; chunk-windowed at ingest |
| FFN | parametric knowledge | frozen | sequential weight edits = catastrophic forgetting (lit) |
| norms | distribution shaping | frozen | cheap to move, hard to bound — keep still |
| GDN2 `_state` S | delta-rule recurrent state | **live** | state writes, not weight writes; per-sequence working memory |
| TitanMemory `memory`/`u`,`v` | fast weights M | **live** | cross-sequence persistent memory; inner-loop Titans update |
| `gate` scalars | branch mixing | bootstrap | opened by outer loop; live drift optional later |
| memory k/q/v/o projections | write/read format | bootstrap* | TTT lit: projections must be co-trained; `live_projections` flag available, off by default |

* "bootstrap" = ordinary optimizer steps on any corpus (or a borrowed
  checkpoint). One-time outer loop; cheap because the model is small.

## Memory semantics

- **GDN2 `_state`** (per block): working/episodic memory, S_t delta-rule
  state updated every forward under `use_cache`. Sequence-local; reset on
  `session.reset()`. Journaled per ingest call for exact revert.
- **TitanMemory fast weights** (per block): persistent long-term memory
  M : R^mk → R^mv trained by inner-loop associative recall —
  `L = Σ_t s_t ‖M(k_t) − v_t‖²`, `k = normalize(k_proj x)`,
  `v = v_proj x`, read `o = o_proj(q @ M)`, gated into the residual.
  Chunkwise update (LaCT-style batching):
  `fmom ← μ·fmom + Σ s_t k_t⊗err_t / N;  M ← γ·M − η·fmom`.
  Persists across `reset()` (unlike GDN2 state); only `revert` or
  `snapshot`/`load` moves it back.
- **Surprise weighting**: `s_t = clip(CE_t / mean(CE), 0, cap)` from the
  model's own next-token loss — write harder where the substrate is
  surprised (SR-TTT's admission result, independently reinvented in FLUX).
- **Inner-loop solver** (`LiveConfig.solver`): `"gd"` (default momentum
  step), `"cg"` (conjugate gradient), or `"solve"` (dense ridge solve of
  the normal equations, regularized toward the decayed prior —
  forgetting IS the regularization; MesaNet/RLS-style). Measured:
  `solve` λ=0.1 beats 1-step GD by ~0.05–0.1 NLL on novel-pair recall.

## Exact-suffix sidecar (NGramMemory)

`session.attach_sparse(NGramMemory(...))` adds a session-level exact
store: per-order suffix-hash → successor count tables, gated logit lift
(`lift` applied to logits in ingest + eval), journaled and reverted with
the rest of the live surface. Motivation (measured, probe_livet.py):
hidden-state keys blur across contexts — exact keys carry ~2 NLL of
verbatim recall that fuzzy fast weights cannot reach, while the fuzzy
channel still adds ~0.3 NLL on contexts where the suffix does not
repeat. Hybrid (solve + ngram) = 2.66 vs fuzzy-only 4.68 vs exact-only
2.96 on the probe. Arbitration is automatic: the sparse lift enters
logits before CE surprise is computed, so positions the exact channel
already answers produce low surprise and fuzzy channels skip them.
All-block TitanMemory beats last-block-only (2.66 vs 2.89). Host-side
dicts at prototype scale; `_GPUCellTables`-style backing if it
graduates.

## Ingest semantics (the live contract)

`session.ingest(ids)`:
1. Push one journal entry: pre-call snapshot of every live buffer
   (per-layer `memory`/`u`,`v` + momentum, GDN2 `_state` + conv states).
2. Chunked forward (`use_cache=True`, `past_key_values=[None]*L`):
   attention sees only the intra-chunk window (bounded short-term);
   GDN2 states carry across chunks (long-term recurrent state).
3. Per-token CE → surprise weights → inner-loop memory updates.
4. Stats: `{tokens, mean_loss, writes, mean_surprise}`.

`revert(mark)` restores journaled pre-call tensors exactly — same
guarantee as FLUX's journal, at ingest-call granularity.

`eval_nll(ids, use_live_state)` snapshots live state, evaluates, restores —
measurement never mutates memory.

## What this does NOT do (honest limits)

- Random-init substrate + live memory ≠ LLM quality. The memory readout
  path (`o_proj`, gates, head alignment) must be shaped by bootstrap
  before live writes are decodable. Tested.
- Intra-chunk attention window: token-level induction across chunk
  boundaries rides on GDN2/memory, not KV. Long-context exact recall of
  arbitrary verbatim spans is FLUX's job — the two systems complement.
- `live_projections` drifts k/v/q format params online — journaled but
  unbounded over long horizons (decay-to-init not yet implemented).

## Files

- `forge/keys/architecture/titan_memory_key.py` — TitanMemory module +
  key (upgraded: learned k/q/v/o projections, factorized or full fast
  weights, momentum + forget + surprise-gated inner loop).
- `forge/keys/attention/gdn2_key.py` — `GDN2Layer.restore_state`
  (journal-side counterpart to `snapshot_state`).
- `forge/model/live.py` — `LiveConfig`, `LiveSession`, `livet_config()`.
- `forge/model/ngram_mem.py` — `NGramMemory` exact-suffix sidecar
  (session-attached, journaled, gated logit lifts).
- `tests/unit/test_livet.py` — lossless init, live signal, frozen
  surface, exact revert, persistence, determinism.

## Bootstrap paths (fluency donor)

1. Train the small config on any corpus with the ordinary trainer —
   cheap at ≤ ~100M params.
2. Borrow an existing small checkpoint (Qwen2.5-0.5B preset exists:
   `qwen25_05b` + `use_gdn2` + `use_titan_memory` via keys — port-first
   path already proven by V13 lineage).
3. FLUX remains the standalone live-memory engine; LiveT is the
   trained-substrate counterpart for when generation quality matters.
   A bridge (FLUX logit interpolation into LiveT/ForgeLM) is a separate
   decision — nothing here couples them.

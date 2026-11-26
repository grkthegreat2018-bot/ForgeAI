# STRATA — M1 Model Dynamics: Design + Calculations

Successor to FluxLM (R52). M0 substrate is committed (905cec2):
`forge/model/strata/` — CellBlock/BlockStore/ParamSet/KnobSet/IO,
22 tests in `tests/unit/test_strata.py`. M1 adds the model itself:
context-key derivation, channel reads, mixture, learning writes, decode.

Requirement → mechanism map:

| Requirement | Mechanism |
|---|---|
| minimal mem | packed cells (48B), bounded block budgets, O(1) state |
| blazing gen | fixed-shape GPU slabs → whole step in 1 CUDA graph |
| flexibility | .cond knob overlays + swappable .iq params |
| learn while working | every predict is also a write opportunity |
| transparency | every score traceable: cell → block → tag → session |
| human-like learning | one-shot episodic write + spaced consolidation + distill |
| weak start, improves | cells empty at t=0; memory + params fill with use |

## 1. Memory math

Cell (M=8): tag 4B + meta 4B + 8×4B followers + raw 8B = **48B**
(40B lean). Block = 2^18 cells = **12.6MB raw / 10.5MB lean**.
Counts log-quantized to u8 (~4.4% steps); totals derived, never stored.

Kind budgets (defaults: 8 hot + 16 cold blocks/kind):

- hot/kind: 8 × 12.6MB = 100MB; 10 order kinds = **~1.0GB GPU**
- cold/kind: 16 × 12.6MB = 201MB; ×10 = **~2.0GB host**
- +epi/sparse/class/hybrid kinds (~8 more): worst case ~2GB GPU / ~4GB host
- `slots_per_block` 2^18→2^16 quarters everything (~250MB GPU hot)

vs Flux: 152B/cell dict-of-dict → 48B = **3.2× denser**, and fixed
slabs are GPU-resident (Flux's dicts forced host-side probes).
vs ForgeLM V2: 6.4GB weights + KV cache O(context). STRATA state is
O(1): sketch + rings ≈ ~1MB regardless of context length.

Dense aux slabs (also .strata tensors): unigram counts u8[V]=64KB,
recency ring i32[64], fatigue ring i32[16]. Trivial.

**.iq params total ~6–10MB** (below). R fingerprint table is 8 bytes
on disk when seeded (derived at load); +12.6MB if teacher-grafted int8.

## 2. Per-token compute — measured on RTX 5070 (`.tmp/bench_strata_step.py`)

| Op (eager) | Latency | Why |
|---|---|---|
| probe B=10 keys | 345µs | ~17 tiny kernels × WDDM ~20µs launch |
| read B=10 | 480µs | gather+LUT decode, launch-bound |
| add B=10 (undo on) | **3.1ms** | ~40 ops + sync branches + repair path |
| dense step (proj+delta+V readout) | 311µs | int8→f32 cast = 67MB temp |
| **CUDA graph: 10 probes + dense** | **585µs → ~1700 tok/s** | 1 replay |
| eager full step (10 r + 10 w + dense) | ~36ms → ~28 tok/s | launch-bound |

Conclusions the design MUST obey:

1. **Eager is dead.** Predict path must be one fused kernel/graph —
   fixed-shape slabs make the whole read side graphable (Flux needed
   host dicts; Strata does not — this is the substrate's real win).
   `CellBlock.pk_of` allocates the salt tensor per call → capture
   failure; cache it at init (M1 fix #1).
2. **add() is not graphable** (data-dependent cuckoo/repair, `.tolist()`
   loops). Two-tier write path: graphable *thin-write* (probe+merge,
   discard-slot on collision — flux dok1 precedent) inside the step;
   deferred repair/eviction flush every K tokens outside the graph.
   Amortized eager flush: every 64 tok → ~935 tok/s; thin-write target
   keeps learning inside the graph → ≥1k tok/s.
3. Dense readout: keep R fp16/int8-native (`torch._int_mm`), no
   fp32 temp — ~33MB traffic → ~50µs, or candidate-gated (~µs).
4. Realistic fused target: read+mix+sample+thin-write ≈ 100–250µs →
   **~4k–10k tok/s predict-only, ~1k–3k tok/s live-learning**
   (Flux: 692 tok/s live @V=4096). Validate with the bench before
   claiming; sweep slots_per_block / kind count / readout mode.

## 3. The .iq dense channel — param-tier math (measured)

Params are a separate artifact from weights, so param count is a
free dial.  The binding constraint is NOT VRAM — it's bytes/token:
at batch=1 every param's bytes cross the bus once per decode step.

**Measured bandwidth reality** (`.tmp/bench_strata_bw.py`, 5070):
fp16 GEMV hits **612 GB/s ≈ 91% of the 672 GB/s peak** — torch,
compiled, and a hand Triton kernel all agree (~600-630).  The kernel
is NOT the bottleneck; earlier bench-2 times moved 2× labeled bytes
(row-count bug) and understated ceilings ~2×.  Lever ranking:

1. **Storage width** — bytes/param is the whole game: bf16→int8→
   int4/NVFP4 (native SM120, in-tree)→ternary-packed.  At ~600 GB/s a
   500µs/token budget = ~300MB/token → 150M bf16 / 300M int8 /
   **600M int4** / ~1.2B ternary params.
2. **Batch decode** — measured: the same 200MB weight sweep is ~335µs
   at B=1 or B=16 streams → **~47k tok/s aggregate at B=16**.
   Per-stream param budget scales ×B.  Self-play/agentic batch
   attempts get this free (flux_batch_workers precedent).
3. **MoE** — params host/disk-resident, top-k fetched (AirMoE
   precedent): unbounded total, only top-k bytes/token.
4. Overlap — dense GEMV and cell probes on separate streams partially
   hide each other (different subsystems).

Corrected tiers (600 GB/s, single stream; ×B under batch decode):

| .iq tier | params | store | bytes/tok | dense cost | ceiling |
|---|---|---|---|---|---|
| nano | ~5M   | bf16 | 10MB  | ~17µs  | ~58k |
| base | ~150M | bf16 | 300MB | ~500µs | ~2.0k |
| pro  | ~300M | int8 | 300MB | ~500µs | ~2.0k |
| **xl** | **~600M** | **int4/NVFP4** | 300MB | ~500µs | ~2.0k |
| max  | ~1B   | int4 | 500MB | ~830µs | ~1.2k |
|      | ~2B   | int4 | 1GB   | ~1.7ms | ~0.6k |

VRAM stays trivial: 600M int4 = 300MB file next to ~1-2GB cells.

- **Recommendation: `iq-pro` ≈ 300-600M int4** (d≈1024, L≈16 GDN2
  delta stack) — 10-20× the params of the first draft at the same
  ~2k tok/s envelope; ~1-1.5k tok/s combined with the cell graph.
- **Invariant**: the dense core must be O(1)-state (delta-net/linear
  attn/SSM), never softmax attention — a KV cache would break the
  format's fixed-memory guarantee.  GDN2 (per R60 lit sweep) is the
  target: strict generalization of KDA/GDN, decoupled erase/write
  gates, chunkwise Triton exists upstream.
- Params train via the existing V2 distill path (track_gaps selective
  scoring); .iq swaps are load-time ops — ship tiers side by side.
- Readout to V=65536: low-rank V×64 int8 (+4.2M) or candidate-gated
  (score only sparse-channel candidates, ~µs on warm contexts).
  At 300M+ params the readout is a rounding error in the budget.

### 3a. The param floor — how small can .iq get?

Sub-BitNet is possible and mostly already in-tree; pick per axis:

| Axis | Mechanism | effective bytes/param |
|---|---|---|
| width | binary {-1,+1} | 0.125B (1 bit) — vs ternary 0.2B |
| width | int2 packed | 0.25B |
| sharing | layer looping (Hyperloop key, in-tree, near-lossless per paper) | ÷loop_iters |
| factorization | NLRQ/Kron/TT (in-tree) | ~0.05-0.4B at rank/8-bit |
| seeds | ParamSet.set_seeded — a whole tensor costs **8 bytes on disk** and regenerates bit-exact | ~0 |

Hard floor framing: disk size can approach ZERO — seeded random
projections cost nothing.  But params are read once per token, so
VRAM+bytes/token is the real cost; and params must hold enough
learned structure to generalize — a fully-seeded .iq contributes
mixing only (cells carry 100% of knowledge).  Two sweet spots:

Implemented (M0+): `ParamSet.set_packed(name, t, bits, group)` —
1/2/4-bit group-scaled codes packed into int32 words + fp16 scales
(`params.py`: `quantize_param`/`dequantize_param`).  bits=1 binary
(mean|w| scale), bits=2 BitNet-ternary {-1,0,1}, bits=4 FP4 E2M1
(NVFP4-native, matches in-tree `use_nvfp4`).

**Measured PTQ floors** (gaussian weights, group=64, this code):
fp4 → ~0.106 rel err, ternary → ~0.51, binary → ~0.60.
"Separate artifact" does NOT mean free quality: quantizing .iq
cannot corrupt the .strata cells (knowledge stays byte-exact), but
it DOES degrade the dense channel's own math at these rates.
The recovery path is structural, not numerical:

- **QAT at distill time** — params come from a distill pass anyway;
  train with the quantizer in the loop (STE, the codebase's BitNet
  precedent) → int4 ≈ lossless, ternary near-lossless.  Post-hoc
  packing of a trained fp .iq is the lossy path — pack a *copy*,
  keep the fp16 master on host as the re-distill source.
- **Cells-heavier mixture** — the more of the prediction carried by
  unaffected memory channels, the less the param error matters;
  quant is a knob on how much quality lives where.

### 3c. Nullifying the loss — measured residual stacking

`set_packed(..., residual=k)` stores W as k packed levels, each
quantizing the previous level's error (NLRQ-residual trick).
Measured, gaussian, group=64:

| config | rel err | B/param |
|---|---|---|
| fp4 x1 | 0.106 | 0.53 |
| **fp4 x2** | **0.011** | **1.06** |
| fp4 x3 | 0.0012 | 1.59 |
| ternary x2 | 0.33 | 0.56 |
| binary x2 | 0.36 | 0.31 |

fp4x2 ≈ int8-class quality at int4+int4 bytes; fp4x3 is
near-lossless (0.1%).  Error is now a continuous dial:
bits × residual-depth.  Combined with the levers:

1. **QAT at distill** (primary): quantizer-in-loop training drives
   residual error toward 0 because the weights adapt around it —
   do this for whichever (bits, residual) tier ships.
2. **Residual levels** (implemented): post-hoc, deterministic,
   +0.5B/param per level.
3. **Sensitivity-mixed widths** (cheap config): keep the readout head
   / gate vectors at int8 while bulk matrices ride fp4 — a handful
   of tensors carry outsized logit influence.
4. **Hedge self-calibration** (free): a degraded param channel reads
   as a weaker expert — AdaHedge down-weights it automatically;
   quant noise is absorbed by the mixture, not amplified.
5. **Hadamard rotation** (option for real distilled weights):
   spreads outliers pre-quant — big win on heavy-tailed trained
   weights, ~nothing on gaussian; keep as a flag when .iq is real.

- **Minimalist**: seeded feature tables + binary/int2 mixer +
  gcell+gsyn online-learned (they self-train at inference) →
  .iq ~1-10MB, decodes ~free.  Quality floor = cells alone until
  online learning shapes the sparse learned pieces.
- **Folding trick (STRATA-native)**: params can be periodic, not
  permanent — distill teacher knowledge into params offline, then
  `consolidate()`-style promotion folds hot param behavior INTO
  cells; at deploy you can even run cells-only.  Params become a
  training-time organ whose learned content migrates into the
  auditable weight store.  Novel vs literature — no other format
  lets you unload the params and keep the knowledge.

### 3b. Is SFT/training needed? — Three-tier answer

- **SFT: no.**  What SFT buys a transformer (format, chat behavior,
  instruction-following) maps to `teach()` + knob overlays + a few
  tagged ingest runs.  No labeled dataset needed.
- **One-time distill bootstrap: optional, recommended.**  Without it,
  gcell's coder W_k and gsyn's projections start random → the dense
  channel is noise until online delta-rule learning shapes it (slow —
  this is the "below par at start" window, possibly days of use).
  With one V2-teacher pass over a modest corpus (~10-100M scored
  tokens via track_gaps), .iq starts with real generalization.  The
  bootstrap touches ONLY .iq — weights and knobs stay clean, so the
  "params decide how, weights hold what" contract is intact.
- **Continual online learning: built in, not training.**  Order/epi/
  knn cells write every token; gcell/gsyn update by delta rule;
  hedge re-weights channels; reinforce/revert give RL-shaped credit;
  consolidate() distills episodic→order.  That IS the model working.

So: SFT never.  Distill bootstrap = how fast you want the floor to
rise — skippable entirely if pure self-bootstrapping is the goal.

**Validated in FLUX (R60d)**: `gsyn_deep` shipped the exact pattern
this doc predicts for the dense channel — a 2-layer MLP memory
trained by surprise-scaled online gradient steps (TTT-MLP/Titans),
kept as a *separate* channel beside linear gsyn rather than a
replacement.  Two findings transfer directly to STRATA's `.iq`
design: (1) the expressivity gap is real and measurable — linear
f(-u)=-f(u) structurally fails antipodal same-class keys (2/4 = chance
on the XOR-corner probe) where the MLP solves it (4/4); (2) undo for
non-invertible gradient steps = whole-theta snapshots scoped to
journal spans, and any span-scoped restore op MUST be added to the
rewind boundary set or a revert overshoots into the previous span
(latent bug, fixed).  STRATA's dense param tier should follow the
same separation: keep the linear/delta-rule path auditable, add the
nonlinear map as its own probeable channel.

**R60e follow-through — `dnet`**: the reasoning primitive is now a
channel too.  Stacked gated delta-rule layers (linear attention) =
a "transformer" whose learning IS inference: seeded projections,
online recurrent states, no KV cache, no optimizer, ~0.34MB.  The
stack gives iterated content routing — layer-l reads relu(o_{l-1}),
so one layer's recall addresses the next's memory.  Depth proof:
antipodal-key task 1-layer=2/4 (linear ceiling) vs 2-layer=3/4.
For STRATA's `.iq` tier this is the architecture answer: depth via
stacked delta layers, nonlinearity via inter-layer relu, audit via
state snapshots — all in the same journaled-channel pattern.

## 4. Learning math ("learns like a human")

- Surprise gate s = 1−p_mix per token (Flux-proven): predictable
  tokens barely write; novel ones commit hard. Thin-write ≈ probe +
  follower merge ≈ few µs fused; ~12 cell ops/token worst case.
- teach(ctx, A): one batched add per kind over the span — a 50-token
  fact lands in <1ms, journaled, revertible by tag. Instant.
- Undo arena: 200k cells × ~48B snapshot ≈ 10MB/block, bounded.
- consolidate(): episodic→order promotion (batch copy + drop).
- Learning curve shape: t=0 → unigram+distilled params (weak, by
  design). +1 exposure → verbatim recall ~exact (episodic). +repeats →
  counts sharpen, hedge rebalances, backoff confidence rises.
  +RSI epochs (R52-9 machinery unchanged: tag/revert/reinforce/teach)
  → gcell generalization + gsyn accumulation. +offline distill →
  teacher V2 re-scores track_gaps positions into .iq — params ratchet
  up WITHOUT touching the memory file (separate artifacts = the whole
  point of the format).

## 5. Corpus ingest — "100+ wiki pages/s" math (measured)

Target ≈ 100 pages/s × ~700 tok/page ≈ **70k tok/s sustained**.

Per-token ingest work: rolling suffix-hash keys (vectorized over the
whole doc — measured 3.3M tok/s even for a naive O(T·Σk) fold; a
prefix-hash form is O(T)) + writes to ~10-20 kinds (orders, uni,
rings, epi, knn, gcell) ≈ ~1.4M cell-writes/s needed.

Measured batched `add()` on the substrate (slots=2^18):

| batch | writes/s |
|---|---|
| 1k    | 0.3M  (launch-bound — too slow) |
| 10k   | 2.6M  |
| 100k  | **13.9M** |

→ bulk slices of ≥64k-100k tokens (bump Flux's 8k `_ingest_bulk`
slices) give **~10-20× headroom** over the 1.4M/s requirement.
Tokenization (~300KB/s input) and Python plumbing are not the
bottleneck.  Verdict: **100 pages/s is comfortably reachable**; the
design constraint is "always batch", not speed.

Caveats that matter at corpus scale:

- Cells store **statistics, not documents** — counts of top-8
  followers per context.  Provenance (which doc wrote a fact) comes
  free via tags; verbatim page recall needs the epi ring + a host doc
  index (trivial RAM).
- Sustained 70k tok/s churns blocks: a kind seals a block every ~3s
  under full-write load.  Retention window = hot/cold budgets; add a
  third tier — **cold blocks on disk as frozen memmap containers**
  (the IO format already supports it: `frozen=True` zero-copy reads)
  → capacity becomes GPU ~2GB / host ~4GB / disk unbounded, with deep
  history costing one mmap probe.
- All of English Wikipedia ≈ ~4.5B tok ≈ ~1h at target rate — but
  what persists is what consolidation promotes, not raw residency.
- Surprise gating in bulk uses the closed-form batch approx
  (flux `fast_ingest` precedent); skipping it entirely is the
  fallback if throughput ever dips.

## 6. Transparency — the differentiator

explain(tok) = exact additive decomposition (mixture is additive in
log space): each candidate's score → list of (kind, block_id, slot,
key, wtag). wtag → tag_ids → "written by session/doc/teach event X".
aux nibble = sub-provenance. Sealed blocks checksummed → bit-exact
audit. stats() = per-block writes/hits/evictions/load report card.
No transformer can answer "why did you emit that" at this precision;
STRATA answers it with a database query.

## 7. Honest limits + falsifiable targets

- Won't reach V2 open-domain reasoning at ~5M dense params. Reachable:
  fluent domain gen, instant factual learning (already exceeds LLMs),
  in-context recall, fast adaptation. Distill cycles raise the floor.
- Targets (must hit before claiming): ≥800 tok/s live CUDA @V=65536;
  single-exposure teach-recall ≥95% on repeated-span eval; NLL
  monotone-improving per epoch on fixed held-out set; ≤3GB VRAM
  all-resident default; predict-only ≥1.5k tok/s.

## 8. Build order

- M1 StrataLM skeleton: rolling suffix-hash keys (10 orders), channel
  registry (order/uni/backoff/recency/fatigue), geometric mixture,
  surprise-gated writes, teach/learn/observe. CPU-eager reference
  first — tests mirror test_flux learn/recall probes.
- M2 Hedge/AdaHedge weights (stored as .strata "meta" vec slabs) +
  explain()/memory_report + thin/warm gating knobs (.cond).
- M3 epi/sparse/class/knn/gcell kinds on the substrate (port Flux
  channel math; epi payload = 4×u64 (pos,succ,cert) per cell row).
- M4 gsyn dense channel: GDN2 params in .iq, S journaled for revert,
  candidate-gated vs full-V readout knob.  NOTE: no distill_gen —
  ship FLUX's `r_live` pattern instead (implemented 2026-10-01 in
  forge/model/flux): seeded features + an additive co-occurrence
  drift table (`_Rl`) whose rows EMA toward the pre-token context
  sketch, vocab-facing reads use the blend `_Reff = _R + _Rl`, frozen
  `_R` keeps EMA-inverse undo bit-exact.  STRATA should mirror it as
  a ParamSet `.iq` table + `r_eta` knob — the generator learns live
  instead of SFT.  R60c adds the recall→gen counterpart: `gen_recall`
  (warm-context write floor — recall-proven successors transfer even
  when unsurprising) and `consolidate_gen()` (closed-form episode
  replay → order-cell successor distribution → top-m share-weighted
  writes into gcell/gsyn/`_Rl`, one journaled block, revert-exact) —
  the gen channels distill what recall already proved without
  re-ingesting the stream.  STRATA M4 should carry both.
- M5 fused step: one CUDA graph (probe+mix+sample+thin-write),
  deferred repair flush; ForgeEngine.from_strata; .strata/.iq/.cond
  in models_index + GUI; self-play mode reuses R52-9 loop.

Measured bench (2026-09-30, RTX 5070, slots=2^18, load 0.57):
probe 345µs, read 480µs, add 3.1ms eager; graph(10 probes+dense)
585µs → 1709 tok/s predict floor; write flush K=64 → ~935 tok/s.
Batched add: 0.3M/2.6M/13.9M writes/s at B=1k/10k/100k; keygen
3.3M tok/s naive.  Scripts: `.tmp/bench_strata_step.py`,
`.tmp/bench_strata_scale.py`.

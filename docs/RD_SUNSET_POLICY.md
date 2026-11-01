# R&D Sunset Policy — use_* Flags Earn Residency

## Problem

`ActivationConfig` (`forge/engine/activation.py:28`) carries **55 `use_*`
feature flags** (:49-110) plus 44 `FeatureSpec` entries in
`_FEATURE_REGISTRY` (`forge/engine/feature_registry.py:607-664`). The
activation path — `_auto_activate_optimal`
(`forge/engine/engine_checkpoints.py:326-465`) →
`ActivationConfig.optimal_for` (`activation.py:211-223`) — turns on only
**~8-11 of them** on any given model (~15-20% ship rate). The rest are
dormant: real code, real maintenance cost during every refactor, and —
worse — they masquerade as capability (audit A2). "Leave it off by
default" is how the zoo grew; this policy makes non-shipping flags
self-deleting instead.

## Policy

A flag that has not **beaten the incumbent on its fixed bench within
N=3 R&D rounds of introduction is REMOVED** — `FeatureSpec` + handler +
`ActivationConfig` field + any plumbing — not merely left defaulted-off.
The deletion is recorded in that round's notes and the CHANGELOG
"Removed (consolidation)" section. Shelf the analysis of why it lost in
`.devin/scratchpad.md` (AGENTS.md §C — dead ends are worth recording).

## Mechanics

### (a) Introduction bookkeeping

- `FeatureSpec` gained `introduced_round: int | None = None`
  (`feature_registry.py:53`). **New specs MUST set it**; legacy specs
  keep `None` — no backfill, and `None` means "predates the policy", not
  exempt-from-review forever.
- Convention: `# introduced: R<nn>` comment next to the
  `ActivationConfig` field, mirroring the existing `R&D14:`/`R35:` tags
  at `activation.py:56-57,107-110`.

### (b) The evaluation bar

"Beat the incumbent on the fixed bench" — the incumbent is whatever the
production activation profile turns on today for that model class, never
a strawman (AGENTS.md §F):

- Decode-path flags: `forge/evolution/targets/decode_step.py` —
  `METRIC="us_per_token"` (:18), gate = logits equal the reference
  bit-for-bit and the cache contains the appended step (:11-13,
  `test()` :108, `bench()` :133). Beat the incumbent step's µs/token.
- KV/prefill/attention flags: the corresponding evolution target/domain
  plus the engine-level bench (e.g. `scripts/bench_lfm3_evolve.py`
  equality + tok/s) on the real model — improvement on the metric that
  maps to production tok/s or quality, not a synthetic score.
- A win must be reproducible: the round notes record bench, incumbent,
  and delta. Winning once and regressing later still sunsets.

### (c) Enforcement

- Each R&D round's close-out includes a flag review alongside the
  `scripts/ledger_audit.py` promotion audit: list flags with
  `introduced_round <= current_round - 3` and no recorded win → delete
  them in a cleanup commit (one logical change, AGENTS.md §H).
- While awaiting a verdict a dormant seam is marked, not hidden — the
  precedent is `kv_backend._LIVE_KV_STRATEGIES` (:405-407): non-live
  strategies build bookkeeping objects whose `info()` reports
  `decode_live: False` + `dormant` (:421-434). Do the same in
  `diagnostics()`/docs for flags — never let requested state read as
  effective state.

## Exceptions

A flag may be **pinned** only by citing its live consumer in the flag
comment — e.g. required by a named preset's validated profile
(`ActivationConfig.mamba_hybrid` :173-209) or a hardware-specific path.
"May be useful someday", "paper parity", and "was expensive to write"
are not pins. A pinned flag that loses its consumer loses its pin.

## Adding a flag (checklist)

1. `# introduced: R<nn>` on the `ActivationConfig` field.
2. `FeatureSpec(..., introduced_round=<nn>)`.
3. Bench-vs-incumbent number in the SAME round's notes (no number = the
   flag starts its 3-round clock as a loser).
4. If dormant until re-tested, mark it dormant honestly (§c).

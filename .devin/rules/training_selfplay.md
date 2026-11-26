# Training/Self-Play Rules — forge/training/, forge/self_play/

## Entry points
- `forge/training/runners/sft_train.py` — canonical SFT runner
  (`_resolve_lora_targets` detects fused `qkv_proj`/`gate_proj`/
  `gate_up_proj`/conv `in_proj` — the unfused default silently matches
  nothing on spark; missing `gate_up_proj`+`in_proj` silently de-adapted
  LFM3 conv blocks + fused FFN before the 2026-11-23 fix).
- `forge/self_play/infinite_loop.py` + `infinite_curriculum.py` —
  RSI quota-driven self-play loop (propose → solve → verify → export).
- `forge/self_play/grpo_trainer.py` — GRPO trainer.
- Sandbox: ALL model-generated code routes through
  `research/persistent_executor.py` (PEP 578 jail). Never exec raw.

## Hard-won rules (from AGENTS.md self-play notes — read before changes)
- **GRPO needs real old_log_probs** — call `collect_log_probs` on the
  pre-update policy and feed per-batch `old_log_probs` into
  `train_step`; ratio≡1.0 means PPO clip is dead (regression:
  `tests/unit/test_grpo_offpolicy.py`). Ref model is eliminated —
  `lora_disabled(model)` ctx does the ref forward (~3.4GB saved).
- **Tokenizer-derived ids, never hardcoded** — eos/stop/keep/`</think>`
  come from the loaded tokenizer (spark eos=1 vs jamba {2,519,542} —
  hardcoding banned spark's real eos and ran every gen to cap).
- **Chat format must match inference** — `_finetune` passes
  `--chat-format spark` (was ChatML: trained markup inference never sees).
- Spark defaults: `--kv-cache standard`, `--ft-pack-sequences` off,
  `--solve-group-size` 4, `--ft-grad-mixup` 3.
- `_fuzz_verify` seeds per-task (`7919 ^ crc32(description)`) — one
  fixed seed gives every task the same mutation trajectory.
- SIGINT handlers must weakref model/optimizer — a captured closure
  pins every epoch's model (commit 51c93e0).
- Launchers/builders for the sft chain live in `.devin/launch_sft*.cmd`
  + `build_sft*.py` + `pack_sft*.py` + `merge_*.py` (untracked — treat
  as the canonical pipeline).

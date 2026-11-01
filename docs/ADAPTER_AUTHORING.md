# Adapter Authoring — Adding a New HF Model to ForgeAI

Recipe for porting a HuggingFace checkpoint to a ForgeAI preset, in
dependency order. Ground rules (AGENTS.md §A): **port-first, train-second**
— the lossless conversion and a bit-exact forward test land BEFORE any
training experiment; **no silent regressions** — every divergence from a
parent preset is documented. Read the sibling of whatever arch is closest
to yours end-to-end first; each existing port is the template for the next.

## The four pieces

| Piece | Where | Example |
|---|---|---|
| `ModelConfig` preset | `forge/config.py` `MODEL_CONFIGS` | `lfm3_1p7b` (:1135) |
| HF→ForgeAI key remap | `forge/engine/compat/arch_adapters.py` + `forge/model/loader.py::_remap_hf_keys` (:358) | `LFM3_KEY_MAP` (:305) |
| `forge_convert.json` manifest | checkpoint dir | `research/checkpoints/lfm3_1p7b_cc_fq/forge_convert.json` |
| Bit-exact port test | `scripts/test_lfm3_port.py` + a `tests/unit/` twin | `tests/unit/test_rocinante_port.py` |

## 1. Preset — `forge/config.py`

`MODEL_CONFIGS` is an inline dict (:604–728) extended by assignment-style
entries below it: `forgelm_v12_jamba` (:746), `forgelm_v13_gdn2` (:787),
`forgelm_v14_rsi` (:828), `qwen25_05b` (:898), `qwen3_4b` (:988),
`llama31_8b` (:1020), `llama32_3b` (:1056), `rocinante_12b` (:1097),
`lfm3_1p7b` (:1135). Add `MODEL_CONFIGS["your_arch"] = ModelConfig(...)`.

Copy the model's **true HF shapes** — do not guess:

- `vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`,
  `intermediate_size`, `attn_bias`, `norm_eps`, `rope_base`,
  `tie_word_embeddings`, `use_qk_norm`.
- `head_dim` **explicitly** when `d_model // n_heads` is wrong
  (qwen3_4b: 2560/32=80, real head_dim=128 — `config.py:994`).
- Hybrids: `layer_types` list per block (`lfm3_1p7b` :1161-1167),
  `sliding_window`, `rope_layer_params` for per-type theta/partial-rotary
  (:1172-1177), `use_attn_out_gate`, `use_sandwich_norm` +
  `sandwich_norm_mode` for pre-residual norm sandwiches.
- `max_seq_len` is a VRAM/RoPE-table budget on 12GB, not the HF
  `max_position_embeddings` — cap it (llama31_8b :1040, lfm3 :1153-1157).

**Lineage contract** (`config.py:547-554`): `parent` names the preset this
one derives from; `dropped_keys` must list EVERY field whose value differs
from the parent — it really means "changed/overridden keys", with a reason
comment per key. `validate_preset_lineage()` (:1203) runs at module import
(:1257-1265): an undocumented divergence fails `import forge.config`.
HF-family ports are **root presets** — leave `parent=None` (they are not
derived from the ForgeLM chain). `get_config(name)` (:1187) always returns
a fresh `ModelConfig`, so callers may mutate fields freely.

## 2. Arch adapter — `forge/engine/compat/arch_adapters.py`

Each arch contributes four functions; copy the conventions of the closest
existing one (LFM3 for hybrids, Qwen3 for plain GQA+SwiGLU):

- `XXX_KEY_MAP` — dict of top-level renames only:
  `"model.embed_tokens.weight" → "embed.weight"`, final norm →
  `"final_norm.weight"` or `"ln_f.weight"` (LFM3's `embedding_norm` IS the
  final norm — :305-310). Per-layer keys do NOT go here.
- `_xxx_layer_map(hf_key, layer_idx) -> str | None` — strips the
  `model.layers.{i}.` prefix, routes each suffix to a `blocks.{i}.*` key.
  Conventions (:316-348): `self_attn.*` → `attn.*` with `o_proj` →
  `out_proj`; `mlp.{gate,up,down}_proj` → `ffn.{w_gate,w_up,w_down}`;
  `input_layernorm` → `ln1`, `post_attention_layernorm` → `ln2`;
  arch-specific norms → `ln_ffn_pre/post` (Gemma3 :126-128) or
  `post_attn_norm`/`post_ffn_norm` (LFM3 sandwich :344-347); conv/MoE
  sublayers hang off `attn.`/`moe.` as the layer module dictates.
- `convert_xxx_checkpoint(hf_state_dict, n_layers)` — applies the dict map
  plus the per-layer map; pure key rename + tensor passthrough (lossless).
- `xxx_layer_types(...)` when layers are non-uniform —
  `gemma3_layer_types` (:168) alternates `attention_swa`/`attention`;
  `lfm3_layer_types` (:371) maps HF names → ForgeAI names.

Then register in `detect_architecture` (:383 — add a key-signature check;
order matters, unique markers win) and `convert_checkpoint` (:409).
`detect_n_layers` (:268) counts `model.layers.{i}.` indices for you.

**Critical**: the engine's fast load path does NOT call `convert_checkpoint`
— it calls `ModelLoader._remap_hf_keys` (`forge/model/loader.py:358-417`),
a regex remap auto-applied to any state dict whose keys start with `model.`
(:756-758, :1179). Add your patterns there too — the `else` branch passes
unknown keys through verbatim (:415-416), so a missing pattern surfaces as
a shape/load failure, not a clean error. Keep the two maps in agreement.

## 3. `forge_convert.json` — checkpoint-dir manifest

A JSON file inside the converted checkpoint dir. Readers:

- `forge_config` — preset name; overrides the default/None `config_name`
  in `ForgeEngine.from_checkpoint` (`engine_checkpoints.py:91-105`).
- `primed` — forces completion-style proposer prompting in CodeEvolve
  (`code_evolve.py:586-592`); set when the tokenizer ships a chat template
  but the weights were SFT'd on primed prompts.
- `fq_rotation` — Hadamard-rotated ForgeQuant pack (`loader.py:794-806`).
- `source_gguf` — resolve the tokenizer from the source GGUF
  (`engine_checkpoints.py:135-143`).
- `weight_format`, `dtype`, `source`, `source_base`, `notes`, plus
  port-specific metadata (`d_c`/`experts`/`d_ff` in `proposer_mla_moe`).

Tokenizer: a `tokenizer.json` inside the ckpt dir wins over the
vocab-size dispatch (`engine_checkpoints.py:120-123`). Single-file
`.safetensors` without a dir/manifest are auto-paired by
`_detect_config_from_header` (`engine_checkpoints.py:674-705`) — header
(vocab, d_model, n_layers) must match a preset exactly.

## 4. Port-first contract — the bit-exact test

Before ANY training (AGENTS.md §A):

1. Convert the checkpoint (lossless key remap; for NEW arch keys, the
   `XxxKey` class — `forge/keys/misc/base.py:37` — gives zero/identity-init
   warm start: `forward` data→weights, `reverse` weights→data;
   `GDN2Key.convert_model_state` (`gdn2_key.py:402`) is the
   whole-checkpoint wrapper pattern incl. in-place `_kda`→`_gdn2`
   upgrades; `Mamba3Key` (`mamba3_key.py:86`) is the zero-init-imaginary
   exemplar).
2. Bit-exact forward test — harness: `scripts/test_lfm3_port.py`
   (venv_train, transformers ≥5.x): loads the HF reference + the ForgeAI
   port on the same input, reports max|Δlogit| (0.0 = BIT-EXACT, <0.05 =
   kernel-level fp pass, else FAIL).
3. CPU unit twin — `tests/unit/test_rocinante_port.py` is the template:
   preset-shape asserts + name-map asserts, no weights needed.
4. Engine load smoke — `ForgeEngine.from_checkpoint(<dir>)` picks the
   preset up from `forge_convert.json`; generate a few tokens.

## Checklist

- [ ] `MODEL_CONFIGS["arch_size"]` with true HF shapes (`head_dim`!)
- [ ] `XXX_KEY_MAP` + `_xxx_layer_map` + `convert_xxx_checkpoint` +
      `xxx_layer_types` (if hybrid)
- [ ] `detect_architecture` signature + `convert_checkpoint` dispatch
- [ ] `ModelLoader._remap_hf_keys` patterns (the live path)
- [ ] `forge_convert.json` in the ckpt dir (`forge_config` minimum)
- [ ] Bit-exact forward script passing (<0.05 max|Δlogit|)
- [ ] `tests/unit/test_<arch>_port.py` preset + map asserts
- [ ] `from_checkpoint` smoke + AGENTS.md preset-lineage note

"""Port Jamba Reasoning 3B -> ForgeLM V2 checkpoint.

Downloads Jamba Reasoning 3B from HuggingFace, converts weights to ForgeAI
internal format using MambaKey, and saves as ForgeLM_V2.safetensors.

Jamba arch:
  - 28 layers: 26 Mamba + 2 attention (at offset 7, period 14 → layers 7, 21)
  - hidden_size=2560, vocab=65536, d_state=16, d_conv=4, expand=2, dt_rank=160
  - 20 attention heads, 1 KV head (GQA 20:1)
  - tie_word_embeddings=True
  - MoE: 1 expert (effectively dense)

Usage:
    python scripts/port_jamba_to_forgelm_v2.py
    python scripts/port_jamba_to_forgelm_v2.py --verify  # also run HF reference
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.keys.architecture.mamba_key import MambaKey

# Jamba Reasoning 3B config
JAMBA_REPO = "ai21labs/AI21-Jamba-Reasoning-3B"
JAMBA_CONFIG = {
    "hidden_size": 2560,
    "num_hidden_layers": 28,
    "num_attention_heads": 20,
    "num_key_value_heads": 1,
    "intermediate_size": 8192,
    "vocab_size": 65536,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "mamba_dt_rank": 160,
    "mamba_conv_bias": True,
    "mamba_proj_bias": False,
    "attn_layer_offset": 7,
    "attn_layer_period": 14,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 262144,
    "bos_token_id": 1,
    "eos_token_id": [2, 519],
}

# Output paths
FORGE_DIR = Path("research/checkpoints")
OUTPUT_CHECKPOINT = FORGE_DIR / "ForgeLM_V2.safetensors"
OUTPUT_CONFIG = FORGE_DIR / "ForgeLM_V2_config.json"
TOKENIZER_DIR = FORGE_DIR / "forgelm_v2_tokenizer"


def build_layer_types(n_layers: int, attn_offset: int, attn_period: int) -> list[str]:
    """Build the layer_types list from Jamba's attention pattern."""
    types = []
    for i in range(n_layers):
        if (i - attn_offset) % attn_period == 0 and i >= attn_offset:
            types.append("attention")
        else:
            types.append("mamba")
    return types


def download_jamba_weights() -> dict[str, torch.Tensor]:
    """Download Jamba Reasoning 3B weights from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Downloading {JAMBA_REPO}...")
    model_dir = snapshot_download(
        repo_id=JAMBA_REPO,
        allow_patterns=["model-*.safetensors", "config.json",
                       "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json"],
    )
    model_dir = Path(model_dir)

    # Load all safetensors shards
    state = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        print(f"  Loading {shard.name}...")
        shard_state = load_file(str(shard))
        state.update(shard_state)

    print(f"Loaded {len(state)} weights ({sum(t.numel() for t in state.values())/1e9:.2f}B params)")
    return state, model_dir


def convert_to_forgeai(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert HuggingFace Jamba weights to ForgeAI internal format."""
    n_layers = JAMBA_CONFIG["num_hidden_layers"]
    layer_types = build_layer_types(
        n_layers,
        JAMBA_CONFIG["attn_layer_offset"],
        JAMBA_CONFIG["attn_layer_period"],
    )

    print(f"Layer types: {layer_types.count('mamba')} Mamba, "
          f"{layer_types.count('attention')} attention")
    print(f"  Attention layers: {[i for i, t in enumerate(layer_types) if t == 'attention']}")

    key = MambaKey(
        n_layers=n_layers,
        layer_types=layer_types,
        mamba_prefix="mamba",
        ffn_style="jamba",
    )

    print("Converting weights via MambaKey...")
    result = key.forward(hf_state)

    if not result.success:
        raise RuntimeError(f"Key conversion failed: {result.error}")

    print(f"Converted: {result.metadata['n_mapped']}/{result.metadata['n_input']} keys")
    if result.metadata["unmapped_keys"]:
        print(f"Unmapped keys ({len(result.metadata['unmapped_keys'])}):")
        for k in result.metadata["unmapped_keys"][:10]:
            print(f"  {k}")
        if len(result.metadata["unmapped_keys"]) > 10:
            print(f"  ... and {len(result.metadata['unmapped_keys']) - 10} more")

    return result.weights


def save_checkpoint(state: dict[str, torch.Tensor], config: dict):
    """Save converted weights and config."""
    FORGE_DIR.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    from safetensors.torch import save_file
    print(f"Saving checkpoint to {OUTPUT_CHECKPOINT}...")
    save_file(state, str(OUTPUT_CHECKPOINT))

    # Save config
    print(f"Saving config to {OUTPUT_CONFIG}...")
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done! Checkpoint: {OUTPUT_CHECKPOINT} ({OUTPUT_CHECKPOINT.stat().st_size / 1e9:.2f} GB)")


def build_forgelm_v2_config() -> dict:
    """Build ForgeLM V2 config from Jamba config."""
    n_layers = JAMBA_CONFIG["num_hidden_layers"]
    layer_types = build_layer_types(
        n_layers,
        JAMBA_CONFIG["attn_layer_offset"],
        JAMBA_CONFIG["attn_layer_period"],
    )

    config = {
        "config_name": "forgelm_v2",
        "architecture": "jamba_hybrid",
        "parent_model": JAMBA_REPO,
        "parent_conversion": "lossless_key_mamba",
        "vocab_size": JAMBA_CONFIG["vocab_size"],
        "d_model": JAMBA_CONFIG["hidden_size"],
        "n_layers": n_layers,
        "n_heads": JAMBA_CONFIG["num_attention_heads"],
        "n_kv_heads": JAMBA_CONFIG["num_key_value_heads"],
        "intermediate_size": JAMBA_CONFIG["intermediate_size"],
        "max_seq_len": JAMBA_CONFIG["max_position_embeddings"],
        "layer_types": layer_types,
        "mamba_d_state": JAMBA_CONFIG["mamba_d_state"],
        "mamba_d_conv": JAMBA_CONFIG["mamba_d_conv"],
        "mamba_expand": JAMBA_CONFIG["mamba_expand"],
        "mamba_dt_rank": JAMBA_CONFIG["mamba_dt_rank"],
        "mamba_bias": JAMBA_CONFIG["mamba_proj_bias"],
        "mamba_conv_bias": JAMBA_CONFIG["mamba_conv_bias"],
        "norm_type": "rmsnorm",
        "norm_eps": JAMBA_CONFIG["rms_norm_eps"],
        "tie_word_embeddings": JAMBA_CONFIG["tie_word_embeddings"],
        "use_final_norm": True,
        "use_embed_norm": False,
        "bos_token_id": JAMBA_CONFIG["bos_token_id"],
        "eos_token_ids": JAMBA_CONFIG["eos_token_id"],
        "rope_base": 1_000_000.0,
    }
    return config


def copy_tokenizer(model_dir: Path):
    """Copy tokenizer files to ForgeAI tokenizer directory."""
    import shutil
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, TOKENIZER_DIR / fname)
            print(f"  Copied {fname}")


def verify_conversion(hf_state: dict[str, torch.Tensor],
                      forge_state: dict[str, torch.Tensor]):
    """Verify that the conversion is lossless (all tensors preserved)."""
    print("\n--- Verification ---")
    n_layers = JAMBA_CONFIG["num_hidden_layers"]
    layer_types = build_layer_types(
        n_layers,
        JAMBA_CONFIG["attn_layer_offset"],
        JAMBA_CONFIG["attn_layer_period"],
    )

    key = MambaKey(
        n_layers=n_layers,
        layer_types=layer_types,
        mamba_prefix="mamba",
        ffn_style="jamba",
    )

    # Reverse: ForgeAI -> HF
    rev = key.reverse(forge_state)
    if not rev.success:
        print(f"Reverse failed: {rev.error}")
        return False

    # Check round-trip
    mismatches = 0
    for hf_key, tensor in hf_state.items():
        if hf_key not in rev.data:
            print(f"  Missing in reverse: {hf_key}")
            mismatches += 1
        elif not torch.equal(tensor, rev.data[hf_key]):
            print(f"  Value mismatch: {hf_key}")
            mismatches += 1

    if mismatches == 0:
        print(f"  Round-trip lossless: YES ({len(hf_state)} keys verified)")
        return True
    else:
        print(f"  Round-trip lossless: NO ({mismatches} mismatches)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Port Jamba Reasoning 3B to ForgeLM V2")
    parser.add_argument("--verify", action="store_true",
                        help="Verify lossless conversion after porting")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if weights already cached")
    args = parser.parse_args()

    # Step 1: Download
    hf_state, model_dir = download_jamba_weights()

    # Step 2: Convert
    forge_state = convert_to_forgeai(hf_state)

    # Step 3: Verify
    if args.verify:
        success = verify_conversion(hf_state, forge_state)
        if not success:
            print("WARNING: Conversion verification failed!")

    # Step 4: Save
    config = build_forgelm_v2_config()
    save_checkpoint(forge_state, config)

    # Step 5: Copy tokenizer
    print(f"\nCopying tokenizer to {TOKENIZER_DIR}...")
    copy_tokenizer(model_dir)

    print("\n✓ ForgeLM V2 port complete!")
    print(f"  Checkpoint: {OUTPUT_CHECKPOINT}")
    print(f"  Config: {OUTPUT_CONFIG}")
    print(f"  Tokenizer: {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()

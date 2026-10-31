"""Quick generation test: ForgeLM V2 (Jamba Reasoning 3B parent) via ForgeEngine.

Loads the converted checkpoint and generates text to verify the model works.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from forge.config import ModelConfig
from forge.model_loader import ConfigurableResearchLLM


def main():
    ckpt_path = "research/checkpoints/ForgeLM_V2.safetensors"
    config_path = "research/checkpoints/ForgeLM_V2_config.json"
    tokenizer_path = "research/checkpoints/forgelm_v2_tokenizer"

    import json
    with open(config_path) as f:
        cfg_dict = json.load(f)

    print(f"Loading tokenizer from {tokenizer_path}...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"  Vocab: {tok.vocab_size}, BOS={tok.bos_token_id}, EOS={tok.eos_token_id}")

    print(f"\nLoading checkpoint from {ckpt_path}...")
    from safetensors.torch import load_file
    state = load_file(ckpt_path)
    print(f"  {len(state)} weights, {sum(t.numel() for t in state.values())/1e9:.2f}B params")

    print("\nBuilding model config...")
    config = ModelConfig(
        vocab_size=cfg_dict["vocab_size"],
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        n_kv_heads=cfg_dict["n_kv_heads"],
        intermediate_size=cfg_dict["intermediate_size"],
        max_seq_len=2048,
        layer_types=cfg_dict["layer_types"],
        mamba_d_state=cfg_dict["mamba_d_state"],
        mamba_d_conv=cfg_dict["mamba_d_conv"],
        mamba_expand=cfg_dict["mamba_expand"],
        mamba_dt_rank=cfg_dict["mamba_dt_rank"],
        mamba_bias=cfg_dict["mamba_bias"],
        mamba_conv_bias=cfg_dict["mamba_conv_bias"],
        norm_eps=cfg_dict["norm_eps"],
        tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        use_final_norm=cfg_dict["use_final_norm"],
        use_embed_norm=cfg_dict["use_embed_norm"],
    )

    print("Building model...")
    model = ConfigurableResearchLLM(config)

    print("Loading weights...")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nMoving model to {device}...")
    model = model.to(device).eval()

    # Test prompts
    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    ",
        "Write a haiku about the ocean:",
    ]

    print("\n=== Generation Tests ===\n")
    for prompt in prompts:
        print(f"Prompt: {prompt!r}")
        input_ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        print(f"  Input tokens: {input_ids.shape[1]}")

        t0 = time.time()
        with torch.no_grad():
            output = model(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output.logits if hasattr(output, "logits") else output

        # Greedy decode a few tokens
        generated = input_ids
        with torch.no_grad():
            for _ in range(30):
                out = model(generated)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_token = logits[0, -1].argmax(keepdim=True)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
                if next_token.item() in (tok.eos_token_id, 2, 519):
                    break

        elapsed = time.time() - t0
        new_tokens = generated.shape[1] - input_ids.shape[1]
        text = tok.decode(generated[0], skip_special_tokens=False)
        print(f"  Output: {text!r}")
        print(f"  {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
        print()

    # Test with chat template
    print("=== Chat Template Test ===\n")
    from forge.self_play.discovery.chat_template import apply_chat_template
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt_text = apply_chat_template(messages, add_generation_prompt=True)
    print(f"Rendered prompt:\n{prompt_text}\n")

    input_ids = tok.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    print(f"Input tokens: {input_ids.shape[1]}")

    t0 = time.time()
    generated = input_ids
    with torch.no_grad():
        for _ in range(80):
            out = model(generated)
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
            next_token = logits[0, -1].argmax(keepdim=True)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            if next_token.item() in (tok.eos_token_id, 2, 519):
                break

    elapsed = time.time() - t0
    new_tokens = generated.shape[1] - input_ids.shape[1]
    response = tok.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=False)
    print(f"Response ({new_tokens} tokens, {elapsed:.2f}s):")
    print(response)
    print(f"\n{new_tokens/elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()

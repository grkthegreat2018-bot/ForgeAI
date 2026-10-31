"""Test TextToWeightsKey on full 1.2B config with a larger corpus."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(m): print(m, flush=True)

def main():
    from forge.config import get_config
    from research.tokenizer_cache import get_tokenizer_no_wrap as load_tokenizer
    from tests.fixtures.keys.architecture.text_to_weights_key import TextToWeightsKey

    # Use the full 1.2B config
    config_name = "forgelm_v2_light"
    log(f"Using config: {config_name}")
    cfg = get_config(config_name)
    log(f"Config: d={cfg.d_model}, L={cfg.n_layers}, vocab={cfg.vocab_size}, inter={cfg.intermediate_size}")

    # Load tokenizer
    tokenizer = load_tokenizer()
    log(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # Create a larger test corpus with diverse patterns
    corpus_path = "D:/windsurf/ForgeAI/.devin/test_corpus_large.txt"
    if not os.path.exists(corpus_path):
        log("Creating large test corpus...")
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        corpus = []
        # Diverse factual patterns
        facts = [
            "The capital of France is Paris.",
            "The capital of England is London.",
            "The capital of Japan is Tokyo.",
            "The capital of Germany is Berlin.",
            "The capital of Italy is Rome.",
            "The capital of Spain is Madrid.",
            "The capital of Russia is Moscow.",
            "The capital of China is Beijing.",
            "The capital of India is New Delhi.",
            "The capital of Brazil is Brasilia.",
            "What is 2+2? The answer is 4.",
            "What is 3+3? The answer is 6.",
            "What is 5+5? The answer is 10.",
            "Water is H2O.",
            "Salt is NaCl.",
            "Carbon dioxide is CO2.",
            "Methane is CH4.",
            "After Monday comes Tuesday.",
            "After Tuesday comes Wednesday.",
            "After Wednesday comes Thursday.",
            "After Thursday comes Friday.",
            "After Friday comes Saturday.",
            "After Saturday comes Sunday.",
            "After Sunday comes Monday.",
            "The sun rises in the east.",
            "The sun sets in the west.",
            "Romeo and Juliet was written by Shakespeare.",
            "Hamlet was written by Shakespeare.",
            "The chemical symbol for gold is Au.",
            "The chemical symbol for silver is Ag.",
        ]
        for _ in range(500):
            corpus.extend(facts)
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(corpus))
        log(f"Corpus: {len(corpus)} lines")

    # Synthesize weights
    log("\n=== Synthesizing weights from text ===")
    key = TextToWeightsKey(max_vocab=cfg.vocab_size, cooc_window=4, svd_rank=cfg.d_model)
    t0 = time.time()
    state_dict = key.synthesize(
        text_path=corpus_path,
        config=cfg,
        tokenizer=tokenizer,
        device="cpu",
        max_lines=0,
    )
    log(f"Synthesis took {time.time()-t0:.1f}s")

    # Save checkpoint
    out_path = "D:/windsurf/ForgeAI/research/checkpoints/ForgeLM_TextToWeights.safetensors"
    log(f"\nSaving to {out_path}...")
    from safetensors.torch import save_file
    save_dict = {k: (v.to(torch.int8).contiguous() if v.dtype == torch.int8
                     else v.contiguous())
                 for k, v in state_dict.items()}
    save_file(save_dict, out_path)
    log(f"Saved ({os.path.getsize(out_path)/1e9:.2f} GB)")

    # Build model and load weights using fast build
    log("\n=== Building model ===")
    from forge.model_loader import load_default_model
    model, _ = load_default_model("forgelm_v2_light",
                                   checkpoint_path=out_path,
                                   device="cuda", dtype=torch.bfloat16)
    model.eval()
    log("Model loaded on GPU")

    # Test generation
    log("\n=== Generation test ===")
    test_prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "What is 2+2?",
        "Water is",
        "After Monday comes",
        "The chemical symbol for gold is",
    ]

    for prompt in test_prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        log(f"\nPrompt: '{prompt}' (tokens: {ids.shape[1]})")

        with torch.no_grad():
            for _ in range(15):
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
                if next_id.item() == 7:  # EOS
                    break

        generated = tokenizer.decode(ids[0], skip_special_tokens=True)
        log(f"  Output: '{generated}'")

    # Embedding similarity analysis
    log("\n=== Embedding similarity analysis ===")
    embed = state_dict['embed.weight']
    test_pairs = [
        ("Paris", "France"),
        ("London", "England"),
        ("Tokyo", "Japan"),
        ("water", "H2O"),
        ("Monday", "Tuesday"),
        ("Paris", "London"),  # both capitals
        ("Paris", "water"),   # unrelated
    ]
    for w1, w2 in test_pairs:
        id1 = tokenizer.encode(w1, add_special_tokens=False)
        id2 = tokenizer.encode(w2, add_special_tokens=False)
        if id1 and id2:
            e1 = embed[id1[0]]
            e2 = embed[id2[0]]
            cos = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            log(f"  cos({w1}, {w2}) = {cos:.3f}")

if __name__ == "__main__":
    main()

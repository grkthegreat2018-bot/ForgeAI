"""Test TextToWeightsKey — build a model from raw text, no training."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(m): print(m, flush=True)

def main():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    from research.tokenizer_cache import get_tokenizer_no_wrap as load_tokenizer
    from tests.fixtures.keys.architecture.text_to_weights_key import TextToWeightsKey

    # Use a small config for fast testing, but with real vocab size
    config_name = "lfm25_tiny"
    log(f"Using config: {config_name}")

    cfg = get_config(config_name)
    # Override vocab to match tokenizer
    cfg.vocab_size = 65536
    cfg.intermediate_size = 512
    cfg.max_seq_len = 256
    log(f"Config: d={cfg.d_model}, L={cfg.n_layers}, vocab={cfg.vocab_size}, inter={cfg.intermediate_size}")

    # Load tokenizer
    tokenizer = load_tokenizer()
    log(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # Create a small test corpus
    corpus_path = "D:/windsurf/ForgeAI/.devin/test_corpus.txt"
    if not os.path.exists(corpus_path):
        log("Creating test corpus...")
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        # Write a small corpus with repeated patterns
        corpus = []
        for _ in range(1000):
            corpus.append("The capital of France is Paris.")
            corpus.append("The capital of England is London.")
            corpus.append("The capital of Japan is Tokyo.")
            corpus.append("What is 2+2? The answer is 4.")
            corpus.append("Water is H2O. Salt is NaCl.")
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

    # Build model and load weights
    log("\n=== Building model ===")
    with torch.device('meta'):
        model = ConfigurableResearchLLM(cfg)
    # Move from meta to real device
    model = model.to_empty(device='cpu')
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log(f"Missing keys: {len(missing)}")
        for k in missing[:5]:
            log(f"  {k}")
    if unexpected:
        log(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            log(f"  {k}")

    model = model.cuda().eval()
    log("Model loaded on GPU")

    # Test generation
    log("\n=== Generation test ===")
    test_prompts = [
        "The capital of France is",
        "What is 2+2?",
        "Water is",
    ]

    for prompt in test_prompts:
        # Tokenize
        ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        log(f"\nPrompt: '{prompt}' (tokens: {ids.shape[1]})")

        with torch.no_grad():
            # Generate 10 tokens greedily
            for _ in range(10):
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

        generated = tokenizer.decode(ids[0], skip_special_tokens=True)
        log(f"  Output: '{generated}'")

    # Compare with random init model
    log("\n=== Random init comparison ===")
    torch.manual_seed(42)
    with torch.device('meta'):
        random_model = ConfigurableResearchLLM(cfg)
    random_model = random_model.to_empty(device='cuda')
    # Initialize with random weights
    for p in random_model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
        else:
            torch.nn.init.ones_(p)
    random_model.eval()

    for prompt in test_prompts[:1]:
        ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        with torch.no_grad():
            for _ in range(10):
                out = random_model(ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
        generated = tokenizer.decode(ids[0], skip_special_tokens=True)
        log(f"  Random: '{generated}'")

    log("\n=== Embedding analysis ===")
    # Check if similar tokens have similar embeddings
    embed = state_dict['embed.weight']
    # Find token IDs for some words
    test_words = ["capital", "France", "Paris", "England", "London", "water", "H2O"]
    for word in test_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            tid = ids[0]
            if tid < embed.shape[0]:
                log(f"  '{word}' (id={tid}): norm={embed[tid].norm():.3f}")

if __name__ == "__main__":
    main()

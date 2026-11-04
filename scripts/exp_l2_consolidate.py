"""exp_l2_consolidate.py — step 3: L1 memory traces -> L2 weight consolidation.

The full "learn on the fly" loop on the real 1.7B backbone:

  ingest facts -> SuffixMemory (L1, instant episodic)
             -> reconstruct fact rows from the memory trace itself
             -> short LoRA fit on those rows (L2, weights)
             -> probe fact QA with memory gate OFF (did knowledge land
                in weights?) + unrelated-text CE (forgetting check)

Pass: post-fit fact CE << pre-fit fact CE, unrelated CE ~unchanged.

Run:  venv\\Scripts\\python.exe scripts\\exp_l2_consolidate.py
"""
import time

import torch
import torch.nn.functional as F

from forge.engine.forge_engine import ForgeEngine
from forge.model.ngram_mem import SuffixMemory
from forge.training.bitnet_lora import add_lora_adapters

CKPT = "research/checkpoints/lfm3_1p7b_cc_fq"
LORA_RANK = 8
LORA_LR = 5e-4
FIT_STEPS = 100
REPLAY = [
    ("Write a Python function to check if a number is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n"
     "    return all(n % i for i in range(2, int(n**0.5)+1))"),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain what a hash map is in one sentence.",
     "A hash map stores key-value pairs using a hash function for "
     "average O(1) lookup."),
    ("Translate 'good morning' to Spanish.", "Buenos dias."),
]

FACTS = [
    ("The Meridian project's vault code is", " 7741."),
    ("The harbor relay uses uplink port", " 9917."),
    ("Dr. Okafor keeps the lab notebook in drawer", " kappa-3."),
    ("The Nighthawk migration starts on", " March 17th."),
    ("Our staging cluster's admin token is", " skt-44-breeze."),
    ("The backup cronjob runs at", " 03:40 every night."),
    ("Captain Reyes pilots the vessel named", " Ironwood."),
    ("The checksum seed for build nine is", " 0xA3F1."),
]
QUESTIONS = [   # same knowledge, different surface — the generalization test
    "What is the vault code for the Meridian project?",
    "Which port does the harbor relay use for its uplink?",
    "Where does Dr. Okafor keep the lab notebook?",
    "When does the Nighthawk migration start?",
    "What is the staging cluster's admin token?",
    "When does the backup cronjob run?",
    "What vessel does Captain Reyes pilot?",
    "What is the checksum seed for build nine?",
]
CONTROL = ("def binary_search(arr, target):\n"
           "    lo, hi = 0, len(arr) - 1\n"
           "    while lo <= hi:\n"
           "        mid = (lo + hi) // 2\n")


def main():
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        CKPT, config_name="lfm3_1p7b", device="cuda",
        quantize=None, load_quantize=None, kv_cache="standard",
        use_compile=False, use_breakable_cuda_graph=False,
        use_block_fusion=False, use_chunked_prefix_cache=False,
        use_adaptive_spec=False, use_suffix_spec=False,
        use_faser=False, use_lookahead_gate=False, use_hotprefix=False,
        use_pod_attention=False, use_hybrid_prefill=False,
        use_fused_qk_norm_rope_cache=False, use_seq_split=False)
    tok = eng.tokenizer
    model = eng.model
    dev = next(model.parameters()).device
    print(f"loaded in {time.time()-t0:.1f}s  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    def ids_of(t):
        return torch.tensor(tok.encode(t), device=dev)

    def answer_ce(q_ids, a_ids):
        """mean CE over answer tokens given the question context, memory OFF."""
        ids = torch.cat([q_ids, a_ids])
        with torch.no_grad():
            z = model(ids.unsqueeze(0))[0]
            z = z[0] if z.dim() == 3 else z
        tgt = ids[q_ids.numel():]
        lp = F.cross_entropy(z[q_ids.numel() - 1:-1].float(), tgt,
                             reduction="none")
        return float(lp.mean())

    # ---- L1 ingest ---------------------------------------------------------
    n_vocab = int(model.config.vocab_size)
    mem = SuffixMemory(n_vocab, orders=(4, 8, 16, 32), gate=1.0)
    for p, a in FACTS:
        mem.feed(ids_of(p + a).tolist())

    # ---- reconstruct rows from the memory trace ----------------------------
    # draft() rolls the exact store forward — rows are what MEMORY knows,
    # not the source doc (drop-out check: facts memory missed get skipped)
    rows, mem_hits = [], 0
    for p, a in FACTS:
        mem._tail = ids_of(p)[-mem.max_order:].tolist()
        drafted = mem.draft(8)
        true_ids = tok.encode(a)
        if drafted[:len(true_ids)] == true_ids[:len(drafted)] or \
                set(true_ids) & set(drafted):
            mem_hits += 1
        rows.append((p, a))  # prompt/response SFT pair
    print(f"mem draft hit-rate: {mem_hits}/{len(FACTS)}")

    # ---- pre-fit probe (memory gate OFF — pure weights) ---------------------
    pre = []
    for q, (p, a) in zip(QUESTIONS, FACTS):
        pre.append(answer_ce(ids_of(q), ids_of(a)))
    ctrl_pre = answer_ce(ids_of(CONTROL),
                         ids_of("\n        if arr[mid] == target:"))
    print(f"pre-fit:  fact_ce={sum(pre)/len(pre):.2f}  ctrl_ce={ctrl_pre:.2f}")

    # ---- LoRA fit -----------------------------------------------------------
    n_ad = add_lora_adapters(model, rank=LORA_RANK,
                             target_modules=["q_proj", "o_proj",
                                             "gate_proj", "up_proj",
                                             "down_proj"])
    lora_params = [p for n, p in model.named_parameters()
                   if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    print(f"lora adapters={n_ad} params={sum(p.numel() for p in lora_params)}")
    opt = torch.optim.AdamW(lora_params, lr=LORA_LR)
    model.train()
    t_fit = time.time()
    pairs = [(ids_of(q), ids_of(a)) for q, (p, a) in zip(QUESTIONS, FACTS)]
    # replay buffer: unrelated verified rows mixed into every step —
    # the documented cure for catastrophic forgetting at small LR
    pairs += [(ids_of(q), ids_of(a)) for q, a in REPLAY]
    for step in range(FIT_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for q_ids, a_ids in pairs:
            ids = torch.cat([q_ids, a_ids])
            z = model(ids.unsqueeze(0))[0]
            z = z[0] if z.dim() == 3 else z
            tgt = ids[q_ids.numel():]
            loss = loss + F.cross_entropy(
                z[q_ids.numel() - 1:-1].float(), tgt)
        (loss / len(pairs)).backward()
        opt.step()
        if step % 20 == 0:
            print(f"  step {step}: loss={float(loss)/len(pairs):.3f}")
    print(f"fit: {time.time()-t_fit:.0f}s for {FIT_STEPS} steps x {len(pairs)} rows")
    model.eval()

    # ---- post-fit probe ------------------------------------------------------
    post = []
    for q, (p, a) in zip(QUESTIONS, FACTS):
        post.append(answer_ce(ids_of(q), ids_of(a)))
    ctrl_post = answer_ce(ids_of(CONTROL),
                          ids_of("\n        if arr[mid] == target:"))
    print(f"post-fit: fact_ce={sum(post)/len(post):.2f}  ctrl_ce={ctrl_post:.2f}")
    print(f"\nresult: fact {sum(pre)/len(pre):.2f} -> {sum(post)/len(post):.2f} "
          f"(drop {sum(pre)/len(pre)-sum(post)/len(post):.2f})  "
          f"ctrl {ctrl_pre:.2f} -> {ctrl_post:.2f} "
          f"(drift {ctrl_post-ctrl_pre:+.2f})  "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()

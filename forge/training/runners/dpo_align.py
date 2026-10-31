"""DPO/ORPO/KTO alignment for the ForgeAI research model.

Minimal standalone implementation that works with our custom nn.Module model
(no HF PreTrainedModel wrapping required). Uses TRL only for dataset loading
conventions; the DPO/ORPO/KTO loss is computed directly.

DPO loss (Rafailov et al. 2023):
    L = -log sigmoid(beta * ((logp_w - logp_l) - (logp_w_ref - logp_l_ref)))
ORPO loss (Hong et al. 2024):
    L = NLL(chosen) + lambda * log_odds_ratio_penalty(chosen, rejected)
KTO loss (Ethayarajh et al. 2024, arXiv:2402.01306):
    Good:  L = -log sigmoid(beta * (log_ratio - KL_ref))
    Bad:   L = -log sigmoid(beta * (KL_ref - log_ratio))
    KTO needs only binary good/bad labels — no paired preference data.
"""
import argparse
import os
import signal
import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv

from research.tokenizer_cache import get_tokenizer

load_dotenv()
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from forge.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    load_training_state,
    save_training_checkpoint,
    step_checkpoint_path,
)
from forge.config import get_config
from forge.model_loader import ModelLoader
from forge.runtime.task_logger import task_scope
from forge.training.training_utils import (
    add_safeguard_args,
    configure_optimizer,
    get_lr,
    has_nan_params,
    oom_guard,
    patch_triton_cache_for_windows,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
)
from research.json_compat import loads


def _logp_for_completion(model, input_ids, completion_start, device):
    """Sum log p(completion_tokens | prompt) for one sample.

    input_ids: 1D LongTensor [seq_len] (prompt+completion, already concatenated)
    completion_start: int index where completion begins
    Returns scalar logp on device.
    """
    if completion_start <= 0:
        raise ValueError(
            f"completion_start must be > 0 (got {completion_start}); "
            "the completion must be preceded by at least one prompt token so "
            "its first token can be predicted by the preceding prompt position."
        )
    ids = input_ids.unsqueeze(0).to(device)  # [1, T]
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out  # [1, T, V]
    # Predict token t+1 from position t. Completion tokens are at positions
    # [completion_start, T-1], predicted by logits at [completion_start-1, T-2].
    comp_logits = logits[0, completion_start - 1 : -1, :]  # [comp_len, V]
    comp_tokens = ids[0, completion_start:]  # [comp_len]
    logp = F.log_softmax(comp_logits.float(), dim=-1)
    token_logp = logp.gather(-1, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [comp_len]
    return token_logp.sum()


def build_preference_sample(tokenizer, prompt, chosen, rejected, max_length,
                            use_chat_template=True):
    """Tokenize one preference triple into two (ids, comp_start) pairs.

    When use_chat_template=True, wraps each in Qwen chat format so the model
    sees proper role markers (<|im_start|>user/assistant). The completion_start
    is computed from the chat-template-rendered prompt prefix.
    """
    if use_chat_template:
        # Render prompt with generation prompt for assistant.
        prompt_msgs = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Render full chosen/rejected conversations.
        chosen_msgs = prompt_msgs + [{"role": "assistant", "content": chosen}]
        rejected_msgs = prompt_msgs + [{"role": "assistant", "content": rejected}]
        chosen_text = tokenizer.apply_chat_template(
            chosen_msgs, tokenize=False, add_generation_prompt=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_msgs, tokenize=False, add_generation_prompt=False
        )
        chosen_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"][:max_length]
        rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"][:max_length]
    else:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            c_ids = c_ids + [tokenizer.eos_token_id]
            r_ids = r_ids + [tokenizer.eos_token_id]
        chosen_ids = (p_ids + c_ids)[:max_length]
        rejected_ids = (p_ids + r_ids)[:max_length]

    chosen_start = min(len(p_ids), len(chosen_ids) - 1)
    rejected_start = min(len(p_ids), len(rejected_ids) - 1)
    return {
        "chosen_ids": chosen_ids,
        "chosen_start": chosen_start,
        "rejected_ids": rejected_ids,
        "rejected_start": rejected_start,
    }


def dpo_loss(chosen_logp, rejected_logp, chosen_logp_ref, rejected_logp_ref, beta=0.1):
    """Standard DPO loss. Ref logits are detached."""
    pi_logratios = chosen_logp - rejected_logp
    ref_logratios = chosen_logp_ref - rejected_logp_ref
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def orpo_loss(chosen_logp, rejected_logp, chosen_nll, lam=1.0):
    """ORPO: NLL on chosen + log-odds-ratio penalty.

    chosen_logp, rejected_logp: per-token average logp (for the odds ratio).
    chosen_nll: total NLL on chosen completion (the SFT signal).
    """
    # Log-odds ratio = log( exp(chosen_logp) / (exp(chosen_logp) + exp(rejected_logp)) )
    lor = chosen_logp - rejected_logp
    penalty = -F.logsigmoid(lor).mean()
    return chosen_nll + lam * penalty


class KLReferencePoint:
    """Running EMA of log-ratios for KTO (the 'KL reference point').

    KTO (arXiv:2402.01306) uses a running average of log(pi/pi_ref) as the
    reference point that defines gains/losses in the prospect-theory loss.
    Updated once per training step with the detached batch-mean log-ratio.
    """

    def __init__(self, ema_beta=0.95):
        self.ema_beta = ema_beta
        self.value = 0.0
        self.initialized = False

    def update(self, log_ratio):
        """Update EMA with current log-ratios (detached). Returns new value."""
        with torch.no_grad():
            batch_mean = log_ratio.detach().float().mean().item()
            if not self.initialized:
                self.value = batch_mean
                self.initialized = True
            else:
                self.value = (self.ema_beta * self.value
                              + (1.0 - self.ema_beta) * batch_mean)
        return self.value


def kto_loss(policy_logp, ref_logp, labels, beta=0.1, kl_ref=0.0):
    """Kahneman-Tversky Optimization loss (arXiv:2402.01306, ICML 2024).

    Unlike DPO, KTO needs only binary good/bad labels — no paired preference
    data.  Each example is (prompt, response, label).

    For desirable (good, label=True) outputs:
        L = -log sigmoid(beta * (log_ratio - KL_ref))
    For undesirable (bad, label=False) outputs:
        L = -log sigmoid(beta * (KL_ref - log_ratio))

    Args:
        policy_logp: sum log p(y|x) under policy  [N]
        ref_logp:    sum log p(y|x) under reference (detached) [N]
        labels:      bool tensor, True=desirable / False=undesirable [N]
        beta:        temperature parameter (default 0.1)
        kl_ref:      running EMA of log-ratios (scalar KL reference point)

    Returns:
        scalar mean loss.
    """
    log_ratio = policy_logp - ref_logp  # [N]
    lbl = labels.to(log_ratio.dtype)
    # label=1 (good): factor = +1  -> beta * (log_ratio - kl_ref)
    # label=0 (bad):  factor = -1  -> beta * (kl_ref  - log_ratio)
    x = beta * (log_ratio - kl_ref) * (2.0 * lbl - 1.0)
    return -F.logsigmoid(x).mean()


def build_kto_sample(tokenizer, prompt, response, label, max_length,
                     use_chat_template=True):
    """Tokenize one KTO example into (ids, comp_start, label).

    KTO data format: {"prompt": ..., "response": ..., "label": true/false}
    Unlike build_preference_sample, there is no chosen/rejected pair — just
    a single response with a binary good/bad label.
    """
    if use_chat_template:
        prompt_msgs = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]
    else:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            r_ids = r_ids + [tokenizer.eos_token_id]
        full_ids = (p_ids + r_ids)[:max_length]

    comp_start = min(len(p_ids), len(full_ids) - 1)
    return {
        "ids": full_ids,
        "comp_start": comp_start,
        "label": bool(label),
    }


def self_reward_generate(model, tokenizer, prompts, n_samples=2,
                         max_new_tokens=128, device="cuda", use_chat_template=True):
    """Generate self-rewarding preference pairs via LLM-as-judge.

    For each prompt:
    1. Generate N candidate responses (sampling with temperature)
    2. Use the model itself as judge: score each response
    3. Highest score = chosen, lowest = rejected

    This is reward-model-free — no external preference data needed.
    Reference: Yuan et al. (ICML 2024) "Self-Rewarding Language Models".

    When use_chat_template=True, generation and scoring use the Qwen chat
    template so the model is prompted as an assistant (not raw text continuation).

    Args:
        model: the LLM (used for both generation and judging)
        tokenizer: tokenizer
        prompts: list of prompt strings
        n_samples: candidates per prompt
        max_new_tokens: generation length
        device: cuda or cpu
        use_chat_template: wrap prompts in chat format for generation + scoring

    Returns:
        list of {prompt, chosen, rejected} dicts
    """
    import torch.nn.functional as F
    pairs = []
    model.eval()

    for i, prompt in enumerate(prompts):
        # Build input ids — with chat template, this includes the assistant turn marker.
        if use_chat_template:
            prompt_msgs = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        prompt_len = input_ids.shape[1]

        # Generate N candidates with sampling.
        candidates = []
        with torch.no_grad():
            for _ in range(n_samples):
                generated = input_ids
                for _ in range(max_new_tokens):
                    out = model(generated)
                    logits = out[0] if isinstance(out, tuple) else out
                    next_logits = logits[:, -1, :] / 0.7  # temperature for diversity
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                response = tokenizer.decode(generated[0, prompt_len:],
                                           skip_special_tokens=True)
                candidates.append(response)

        if len(candidates) < 2:
            continue

        # Judge: score each candidate by computing log-likelihood of the response.
        scores = []
        for cand in candidates:
            if use_chat_template:
                full_msgs = [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": cand}]
                full_text = tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
                ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
                # Compute the prompt-only length to isolate response tokens.
                prompt_only_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True
                )
                prompt_only_ids = tokenizer(prompt_only_text, add_special_tokens=False)["input_ids"]
                resp_len = ids.shape[1] - len(prompt_only_ids)
            else:
                full = prompt + cand
                ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
                resp_len = ids.shape[1] - prompt_len

            with torch.no_grad():
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else out
                log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
                targets = ids[0, 1:]
                token_lps = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                if resp_len > 0:
                    scores.append(token_lps[-resp_len:].mean().item())
                else:
                    scores.append(-1e9)

        # Best = chosen, worst = rejected.
        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))
        if best_idx == worst_idx:
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": candidates[best_idx],
            "rejected": candidates[worst_idx],
        })

        if (i + 1) % 20 == 0:
            print(f"  [self-reward] {i+1}/{len(prompts)} prompts | {len(pairs)} pairs")

    model.train()
    return pairs


def main():
    p = argparse.ArgumentParser(description="DPO/ORPO alignment for ForgeAI model")
    p.add_argument("--config", default="360m_mla")
    p.add_argument("--checkpoint", default="research/checkpoints/pretrained_llm.pt")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized", help="HF preference dataset")
    p.add_argument("--data", default="",
                   help="Local JSONL file with {prompt, chosen, rejected} — overrides --dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--method", choices=["dpo", "orpo", "self-reward"], default="orpo",
                   help="ORPO needs no reference model; self-reward uses LLM-as-judge (no preference data needed)")
    p.add_argument("--loss-type", choices=["dpo", "kto"], default=None,
                   help="Loss function override: 'dpo' (paired) or 'kto' (binary labels, arXiv:2402.01306). "
                        "When set, takes precedence over --method for loss computation.")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--kto-beta", type=float, default=0.1,
                   help="KTO temperature parameter (default 0.1)")
    p.add_argument("--kto-kl-ema", type=float, default=0.95,
                   help="EMA decay for KTO KL reference point (default 0.95)")
    p.add_argument("--lambda-orpo", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--optimizer", default="bnb",
                   help="Optimizer: cpu_offload (full precision on 12GB), bnb (8-bit), fused")
    p.add_argument("--save", default="research/checkpoints/dpo_llm.pt")
    # Self-rewarding args.
    p.add_argument("--self-reward-prompts", default=None,
                   help="JSONL file with prompts for self-rewarding (if None, generates from model)")
    p.add_argument("--self-reward-n", type=int, default=200, help="Number of self-generated preference pairs")
    p.add_argument("--self-reward-samples", type=int, default=2, help="Samples per prompt for preference pairs")
    p.add_argument("--use-chat-template", action="store_true", default=True,
                   help="Wrap prompts/responses in Qwen chat template (default: ON)")
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false",
                   help="Disable chat template (use plain text concatenation)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume weights + training state from this checkpoint")
    add_safeguard_args(p)
    args = p.parse_args()

    patch_triton_cache_for_windows()
    cfg = get_config(args.config)
    cfg.seq_len = args.max_seq_length
    cfg.max_seq_len = max(cfg.max_seq_len, args.max_seq_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")

    weights_path = args.resume or args.checkpoint
    print(f"Loading base model from {weights_path}...")
    model = ModelLoader.build_model(cfg, checkpoint_path=weights_path).to(device).eval()
    # Enable training mode + grads.
    model.train()
    for p_ in model.parameters():
        p_.requires_grad_(True)

    # Resolve effective loss type: --loss-type takes precedence over --method.
    effective_loss = args.loss_type if args.loss_type else args.method

    # Reference model for DPO/KTO (frozen, no grads).
    ref_model = None
    if effective_loss in ("dpo", "kto"):
        ref_model = ModelLoader.build_model(cfg, checkpoint_path=args.checkpoint).to(device).eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)

    optimizer = configure_optimizer(model, args.lr, cfg.weight_decay, args.optimizer)

    # Resume training state (optimizer + RNG + step) if requested.
    start_step = 0
    if args.resume:
        ts = load_training_state(args.resume, optimizer=optimizer)
        if ts["step"] is not None:
            start_step = ts["step"] + 1
            print(f"Resuming from {args.resume} at step {start_step}")

    # Ctrl-C -> emergency checkpoint before dying.
    current = {"step": start_step}
    _old_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig, frame):
        emergency_save(model, args.save, "interrupt", current["step"], optimizer=optimizer)
        signal.signal(signal.SIGINT, _old_sigint)  # restore default before re-raising
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    # Self-rewarding: generate preference pairs from the model itself (no external data).
    if args.method == "self-reward":
        print(f"[self-reward] Generating {args.self_reward_n} preference pairs via LLM-as-judge...")
        prompts = []
        if args.self_reward_prompts:
            import json
            with open(args.self_reward_prompts) as f:
                content = f.read()
            for line in content.splitlines():
                d = loads(line)
                prompts.append(d.get("prompt", d.get("text", "")))
        else:
            # Default prompts for self-rewarding.
            default_prompts = [
                "Explain the concept of recursion in programming.",
                "Write a short poem about the ocean.",
                "What is the capital of France?",
                "How does photosynthesis work?",
                "Write a Python function to reverse a string.",
                "Explain the difference between TCP and UDP.",
                "What causes rainbows to form?",
                "Describe the water cycle.",
                "Write a haiku about autumn.",
                "What is machine learning in simple terms?",
            ]
            prompts = (default_prompts * (args.self_reward_n // 10 + 1))[:args.self_reward_n]

        pairs = self_reward_generate(
            model, tokenizer, prompts,
            n_samples=args.self_reward_samples,
            max_new_tokens=128, device=device,
            use_chat_template=args.use_chat_template,
        )
        print(f"[self-reward] Generated {len(pairs)} preference pairs.")
        samples = [build_preference_sample(tokenizer, p["prompt"], p["chosen"], p["rejected"],
                                          args.max_seq_length, use_chat_template=args.use_chat_template) for p in pairs]
        # Switch to ORPO for actual training (no reference model needed).
        args.method = "orpo"
    else:
        # ── KTO data loading (binary labels: prompt, response, label) ──
        if effective_loss == "kto":
            kto_samples = []
            if args.data:
                print(f"Loading local KTO data: {args.data}")
                with open(args.data, encoding="utf-8") as f:
                    content = f.read()
                for i, line in enumerate(content.splitlines()):
                    if i >= args.max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = obj.get("prompt", "")
                    response = obj.get("response", "")
                    label = obj.get("label", None)
                    if not prompt or not response or label is None:
                        continue
                    kto_samples.append(build_kto_sample(
                        tokenizer, prompt, response, label,
                        args.max_seq_length,
                        use_chat_template=args.use_chat_template))
                print(f"  Loaded {len(kto_samples)} KTO examples from local file")
            else:
                print("KTO requires --data with {prompt, response, label} JSONL; using synthetic smoke-test.")
            if not kto_samples:
                kto_samples = [
                    build_kto_sample(tokenizer, "The sky is", " blue.", True, args.max_seq_length,
                                     use_chat_template=args.use_chat_template),
                    build_kto_sample(tokenizer, "The sky is", " green.", False, args.max_seq_length,
                                     use_chat_template=args.use_chat_template),
                    build_kto_sample(tokenizer, "2 + 2 =", " 4.", True, args.max_seq_length,
                                     use_chat_template=args.use_chat_template),
                    build_kto_sample(tokenizer, "2 + 2 =", " 5.", False, args.max_seq_length,
                                     use_chat_template=args.use_chat_template),
                ]
            samples = kto_samples
        # ── Local JSONL data (from dpo_data_gen.py) takes priority ──
        elif args.data:
            print(f"Loading local preference data: {args.data}")
            samples = []
            with open(args.data, encoding="utf-8") as f:
                content = f.read()
            for i, line in enumerate(content.splitlines()):
                if i >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = obj.get("prompt", "")
                chosen = obj.get("chosen", "")
                rejected = obj.get("rejected", "")
                if not prompt or not chosen or not rejected:
                    continue
                samples.append(build_preference_sample(
                    tokenizer, prompt, chosen, rejected,
                    args.max_seq_length,
                    use_chat_template=args.use_chat_template))
            print(f"  Loaded {len(samples)} preference pairs from local file")
        else:
            print(f"Loading preference dataset {args.dataset}[{args.split}]...")
            try:
                ds = load_dataset(args.dataset, split=args.split)
            except Exception as e:
                print(f"Failed to load {args.dataset}: {e}")
                print("Falling back to synthetic preference samples for smoke test.")
                ds = None

            samples = []
            if ds is not None:
                for i, row in enumerate(ds):
                    if i >= args.max_samples:
                        break
                    # TRL ultrafeedback_binarized schema: prompt, chosen (list[message]), rejected (list[message])
                    prompt = row.get("prompt", "")
                    chosen = row.get("chosen", "")
                    rejected = row.get("rejected", "")
                    # If chosen/rejected are message lists, extract the assistant content.
                    if isinstance(chosen, list):
                        chosen = next((m["content"] for m in chosen if m.get("role") == "assistant"), "")
                    if isinstance(rejected, list):
                        rejected = next((m["content"] for m in rejected if m.get("role") == "assistant"), "")
                    if not prompt or not chosen or not rejected:
                        continue
                    samples.append(build_preference_sample(tokenizer, prompt, chosen, rejected,
                                                          args.max_seq_length,
                                                          use_chat_template=args.use_chat_template))
    if not samples and effective_loss != "kto":
        print("No usable samples; using synthetic smoke-test samples.")
        samples = [
            build_preference_sample(tokenizer, "The sky is", " blue.", " green.", args.max_seq_length,
                                    use_chat_template=args.use_chat_template),
            build_preference_sample(tokenizer, "Paris is the capital of", " France.", " Germany.", args.max_seq_length,
                                    use_chat_template=args.use_chat_template),
            build_preference_sample(tokenizer, "2 + 2 =", " 4.", " 5.", args.max_seq_length,
                                    use_chat_template=args.use_chat_template),
        ]

    # Recompute effective loss after data loading (self-reward switches to orpo).
    effective_loss = args.loss_type if args.loss_type else args.method

    print(f"Prepared {len(samples)} samples. Training {effective_loss} for {args.max_steps} steps.")

    aborted = False
    # KTO KL reference point (running EMA of log-ratios).
    kl_ref_point = KLReferencePoint(ema_beta=args.kto_kl_ema) if effective_loss == "kto" else None
    with task_scope("dpo") as log:
        step = start_step
        for _epoch in range(100):
            for s in samples:
                if step >= args.max_steps:
                    break
                if vram_exceeded(args.vram_limit_gb, device):
                    print("VRAM limit exceeded; emergency save + abort.")
                    emergency_save(model, args.save, "emergency", step, optimizer=optimizer)
                    aborted = True
                    break
                current["step"] = step

                with oom_guard(str(device), label="dpo_fwd") as safe:
                    if effective_loss == "kto":
                        # KTO: single (prompt, response, label) example.
                        ids = torch.tensor(s["ids"], dtype=torch.long)
                        policy_logp = _logp_for_completion(model, ids, s["comp_start"], device)
                        with torch.no_grad():
                            ref_logp = _logp_for_completion(ref_model, ids, s["comp_start"], device)
                        label_t = torch.tensor([s["label"]], dtype=torch.bool, device=device)
                        loss = kto_loss(
                            policy_logp.unsqueeze(0), ref_logp.unsqueeze(0),
                            label_t, beta=args.kto_beta, kl_ref=kl_ref_point.value)
                        # Update KL reference point with detached log-ratio.
                        kl_ref_point.update(policy_logp - ref_logp)
                    else:
                        chosen_ids = torch.tensor(s["chosen_ids"], dtype=torch.long)
                        rejected_ids = torch.tensor(s["rejected_ids"], dtype=torch.long)
                        chosen_logp = _logp_for_completion(model, chosen_ids, s["chosen_start"], device)
                        rejected_logp = _logp_for_completion(model, rejected_ids, s["rejected_start"], device)

                        if effective_loss == "dpo":
                            with torch.no_grad():
                                chosen_ref = _logp_for_completion(ref_model, chosen_ids, s["chosen_start"], device)
                                rejected_ref = _logp_for_completion(ref_model, rejected_ids, s["rejected_start"], device)
                            loss = dpo_loss(chosen_logp, rejected_logp, chosen_ref, rejected_ref, beta=args.beta)
                        else:
                            # ORPO: per-token average logp + NLL on chosen.
                            comp_len_c = max(1, len(s["chosen_ids"]) - s["chosen_start"])
                            comp_len_r = max(1, len(s["rejected_ids"]) - s["rejected_start"])
                            chosen_logp_pt = chosen_logp / comp_len_c
                            rejected_logp_pt = rejected_logp / comp_len_r
                            chosen_nll = -chosen_logp / comp_len_c
                            loss = orpo_loss(chosen_logp_pt, rejected_logp_pt, chosen_nll, lam=args.lambda_orpo)

                    optimizer.zero_grad()
                    loss.backward()
                if safe.skipped:
                    optimizer.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                lr = get_lr(step, args.max_steps, args.lr, args.lr * 0.1, max(1, args.max_steps // 10))
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.step()

                if step > 0 and step % 50 == 0 and has_nan_params(model):
                    print("NaN/Inf detected in parameters; emergency save + abort.")
                    emergency_save(model, args.save, "nan", step, optimizer=optimizer)
                    aborted = True
                    break

                if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                    ckpt = step_checkpoint_path(args.save, step)
                    save_training_checkpoint(model, ckpt, optimizer=optimizer, step=step)
                    cleanup_step_checkpoints(args.save, args.keep_checkpoints)

                if step % 5 == 0 or step == args.max_steps - 1:
                    kl_str = f" | kl_ref {kl_ref_point.value:.4f}" if kl_ref_point is not None else ""
                    msg = f"Step {step+1}/{args.max_steps} | {effective_loss} loss {loss.item():.4f}{kl_str} | lr {lr:.2e} | vram {torch.cuda.memory_allocated()/1e9:.2f} GB"
                    print(msg)
                    log.log(msg)
                    write_status_json(args.status_file, {
                        "step": step + 1, "max_steps": args.max_steps,
                        "loss": loss.item(), "lr": lr, "method": effective_loss,
                    })
                    write_heartbeat(args.heartbeat_file)
                step += 1
            if aborted or step >= args.max_steps:
                break

        if not aborted:
            save_training_checkpoint(model, args.save, optimizer=optimizer, step=step,
                                     meta={"config": cfg.__dict__})
            print(f"Saved aligned model to {args.save}")


if __name__ == "__main__":
    main()

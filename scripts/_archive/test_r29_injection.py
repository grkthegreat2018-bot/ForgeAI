"""R29: LoRA knowledge injection with param growth (Qwen 2.5 0.5B via ForgeEngine).

User spec: load Qwen2.5-0.5B through ForgeEngine -> freeze parent -> create
minimal LoRA shaped to match parent's pretrained weights -> train ONLY the
LoRA on randomly generated knowledge -> stabilization process -> unfreeze
parent -> merge -> test each knowledge fact.

Goals:
  1. Find the best injection process: >90% new-knowledge correctness,
     0% regression on existing knowledge (no known/held fact flips).
  2. Golden ratio: params needed per fact across knowledge sizes
     (rank x n_facts sweep -> params = P0 + c*n fit).

OPERATING POINT (established by the R29 probes + quant matrix, see
scripts/r29_quant_speed.json):
  - rank-16 LoRA on the L12 FFN trio (gate/up/down), scale 2.0
  - batch 16, bf16 autocast, lr 1e-3, 90 epochs  -> 96/100 exact in 31 s
  - single down_proj or small ranks learn format but not exact values;
    batched-bf16 is 12.7x faster than the fp32 batch-1 loop; quantized
    (int8/NF4/unsloth-4bit) bases are SLOWER here and cost base quality.

VRAM budget: fp32 0.5B weights ~2.0 GB + short-seq activations <1 GB
=> ~3 GB peak on RTX 5070 12 GB. No CPU offload needed.

Invariants (unit-tested in tests/unit/test_r29_lora.py):
  - LoRA zero-init forward is a bit-exact no-op; parent restored pristine
    between conditions (asserted via baseline PPL equality).
  - Merge: bitwise weight reconstruction (W0 + scale*B@A) per module +
    layer-level relative equivalence < 1e-3; zero-delta noop merge is
    fully bit-exact (logit diff 0.0).
  - KV-cached greedy eval is asserted equivalent to the slow re-forward
    path during the sanity phase before it is trusted.

Usage:
  python scripts/test_r29_injection.py --phase sanity
  python scripts/test_r29_injection.py --phase 1
  python scripts/test_r29_injection.py --phase 2 --process full
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from contextlib import ExitStack, contextmanager

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from forge.engine.forge_engine import ForgeEngine

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
QWEN_DIR = glob.glob(os.path.join(
    r"C:\Users\tmk68\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots", "*"))[0]
FACTS_PATH = os.path.join(os.path.dirname(__file__), "r29_facts.json")
HERE = os.path.dirname(__file__)

with open(FACTS_PATH) as f:
    _F = json.load(f)
KNOWN = [tuple(x) for x in _F["known"]]
HELD = [tuple(x) for x in _F["held"]]
INJECT = [tuple(x) for x in _F["inject"]]
ANCHORS = _F["anchors"]
QA_ANCHORS = [f"Q: {q}\nA: {a}." for q, a in _F.get("qa_anchors", [])]

LAYER_IDX = 12            # single mid layer, FFN trio = winning placement
TRIO = ["gate_proj", "up_proj", "down_proj"]
EFFECTIVE_SCALE = 2.0     # alpha = 2*rank -> scale=2 at every rank (controlled)
BATCH = 16
LR_DEFAULT = 1e-3
EPOCHS_DEFAULT = 90
MERGE_LOGIT_TOL = 0.5     # info bound on full-model logit diff
LAYER_REL_TOL = 1e-3      # hard gate: merged layer output vs adapter output

ORIG = {}                 # pristine parent FFN trio weights (captured once)
BASE_PPL = [None]         # first baseline PPL (reset sanity assert)
USE_CACHED_EVAL = [None]  # set during sanity after equivalence proof


# ═══ Minimal LoRA (shape-matched to parent) ═════════════════════════════════
class MinimalLoRA(nn.Module):
    """LoRA wrapped around a frozen parent Linear.

    delta W = B @ A has exactly the parent weight shape (shape-matched).
    B is zero-init => forward is a bit-exact no-op at start.
    Only A/B train; W0/bias are frozen clones.
    """
    def __init__(self, frozen_linear, rank=16, scale=2.0):
        super().__init__()
        in_f, out_f = frozen_linear.weight.shape[1], frozen_linear.weight.shape[0]
        self.rank, self.scale = rank, scale
        self.W0 = nn.Parameter(frozen_linear.weight.data.clone(), requires_grad=False)
        self.has_bias = frozen_linear.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(frozen_linear.bias.data.clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self._bypass = False
        self.reset(orthogonal=False)

    def reset(self, orthogonal=False):
        """Fresh init. B=0 => no-op regardless of A init."""
        if orthogonal and self.rank > 1:
            nn.init.orthogonal_(self.lora_A)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        with torch.no_grad():
            self.lora_B.zero_()

    def forward(self, a):
        base = F.linear(a, self.W0, self.bias if self.has_bias else None)
        if self._bypass:
            return base
        delta = F.linear(F.linear(a, self.lora_A), self.lora_B) * self.scale
        return base + delta

    @contextmanager
    def bypass(self):
        """Temporarily disable the delta (exact parent-only forward)."""
        self._bypass = True
        try:
            yield
        finally:
            self._bypass = False

    def delta_W(self):
        return (self.lora_B @ self.lora_A) * self.scale

    def merge_into_base(self):
        merged = nn.Linear(self.W0.shape[1], self.W0.shape[0],
                           bias=self.has_bias, device=self.W0.device,
                           dtype=self.W0.dtype)
        merged.weight.data = (self.W0 + self.delta_W()).data.clone()
        if self.has_bias:
            merged.bias.data = self.bias.data.clone()
        return merged

    def trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


@contextmanager
def bypass_all(loras):
    with ExitStack() as st:
        for lm in loras:
            st.enter_context(lm.bypass())
        yield


# ═══ Parent pristine-state management (FFN trio) ════════════════════════════
def capture_original(model):
    mlp = model.model.layers[LAYER_IDX].mlp
    for name in TRIO:
        lin = getattr(mlp, name)
        assert isinstance(lin, nn.Linear), f"parent {name} must be a plain Linear"
        ORIG[name] = (lin.weight.data.clone(),
                      lin.bias.data.clone() if lin.bias is not None else None)


def fresh_trio(model, rank, orth=False):
    """Restore the PRISTINE parent FFN trio, wrapped in fresh MinimalLoRAs.

    Called at the start of every condition so merges from previous
    conditions never contaminate the parent.
    """
    mlp = model.model.layers[LAYER_IDX].mlp
    loras = []
    for name in TRIO:
        w, b = ORIG[name]
        lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None,
                        device=w.device, dtype=w.dtype)
        lin.weight.data = w.clone()
        if b is not None:
            lin.bias.data = b.clone()
        lm = MinimalLoRA(lin, rank=rank, scale=EFFECTIVE_SCALE)
        if orth and lm.rank > 1:
            with torch.no_grad():
                nn.init.orthogonal_(lm.lora_A)
        setattr(mlp, name, lm.to(model.device))
        loras.append(lm)
    return loras


def freeze_except_lora(model):
    for name, p in model.named_parameters():
        p.requires_grad_(any(k in name for k in ["lora_A", "lora_B"]))


def unfreeze_parent(model):
    for p in model.parameters():
        p.requires_grad_(True)


def merge_trio(model):
    mlp = model.model.layers[LAYER_IDX].mlp
    for name in TRIO:
        if isinstance(getattr(mlp, name), MinimalLoRA):
            setattr(mlp, name, getattr(mlp, name).merge_into_base())


def trio_params(loras):
    return sum(lm.trainable_params() for lm in loras)


# ═══ Measurement ════════════════════════════════════════════════════════════
def make_text(q, a):
    return f"Q: {q}\nA: {a}."


def encode(engine, text):
    return engine.tokenizer(text, return_tensors="pt")["input_ids"].to(engine.device)


def exact_match_slow(engine, q, a):
    """Strict: greedy-generate exactly len(answer_tokens) tokens; ALL must match.

    'Q:/A:' format: debug showed Qwen2.5-0.5B answers directly after 'A:'
    but restates the question after 'Answer:'.
    """
    p_ids = encode(engine, f"Q: {q}\nA:")
    a_ids = encode(engine, " " + a)
    a_len = a_ids.shape[1]
    cur, gen = p_ids, []
    with torch.no_grad():
        for _ in range(a_len):
            logits = engine.model(cur).logits[0, -1, :]
            nxt = logits.argmax().item()
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=engine.device)], dim=1)
    return gen == a_ids[0].tolist()


def exact_match_cached(engine, q, a):
    """KV-cached greedy (same decisions, ~len(answer)x fewer token-forwards)."""
    p_ids = encode(engine, f"Q: {q}\nA:")
    a_ids = encode(engine, " " + a)
    a_len = a_ids.shape[1]
    gen = []
    with torch.no_grad():
        cache = DynamicCache()
        out = engine.model(p_ids, use_cache=True, past_key_values=cache)
        nxt = out.logits[0, -1, :].argmax().item()
        gen.append(nxt)
        for _ in range(a_len - 1):
            tok = torch.tensor([[gen[-1]]], device=engine.device)
            out = engine.model(tok, use_cache=True, past_key_values=cache)
            gen.append(out.logits[0, -1, :].argmax().item())
    return gen == a_ids[0].tolist()


def exact_match(engine, q, a):
    if USE_CACHED_EVAL[0]:
        return exact_match_cached(engine, q, a)
    return exact_match_slow(engine, q, a)


def greedy_gen(engine, q, a, extra=3):
    """Greedy string of len(answer)+extra tokens (diagnostics for flips)."""
    p_ids = encode(engine, f"Q: {q}\nA:")
    a_ids = encode(engine, " " + a)
    n = a_ids.shape[1] + extra
    cur, gen = p_ids, []
    with torch.no_grad():
        for _ in range(n):
            nxt = engine.model(cur).logits[0, -1, :].argmax().item()
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=engine.device)], dim=1)
    return engine.tokenizer.decode(gen)


def anchor_ppl(engine):
    """Mean-NLL perplexity across the anchor corpus (generic prose)."""
    tot_nll, n_tok = 0.0, 0
    with torch.no_grad():
        for text in ANCHORS:
            ids = encode(engine, text)
            logits = engine.model(ids).logits[..., :-1, :].contiguous()
            lb = ids[..., 1:].contiguous()
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)), lb.view(-1),
                                  reduction="sum")
            tot_nll += nll.item()
            n_tok += lb.numel()
    return math.exp(tot_nll / n_tok)


def measure(engine, label, inject_subset, base=None):
    res = {}
    detail = {}
    for name, facts in [("known", KNOWN), ("held", HELD), ("inject", inject_subset)]:
        flags = [exact_match(engine, q, a) for q, a in facts]
        detail[name] = flags
        res[name] = {"exact": sum(flags), "total": len(facts)}
    res["ppl"] = anchor_ppl(engine)
    if base is not None:
        res["known_reg"] = base["known"]["exact"] - res["known"]["exact"]
        res["held_reg"] = base["held"]["exact"] - res["held"]["exact"]
        res["ppl_delta_pct"] = 100.0 * (res["ppl"] - base["ppl"]) / base["ppl"]
        for name, facts in [("known", KNOWN), ("held", HELD)]:
            flipped = [q for (q, a), was, now in zip(facts, base["_detail"][name],
                                                     detail[name]) if was and not now]
            if flipped:
                print(f"    flipped {name}: {flipped}", flush=True)
                for q, a in facts:
                    if q in flipped:
                        print(f"      '{q}' -> now generates "
                              f"{greedy_gen(engine, q, a)!r} (want {a!r})", flush=True)
    res["_detail"] = {"known": detail["known"], "held": detail["held"]}
    i_pct = res["inject"]["exact"] / max(res["inject"]["total"], 1)
    extra = ""
    if base is not None:
        extra = (f" Kreg={res['known_reg']} Hreg={res['held_reg']} "
                 f"dPPL={res['ppl_delta_pct']:+.2f}%")
    print(f"  [{label}] K={res['known']['exact']}/{res['known']['total']} "
          f"H={res['held']['exact']}/{res['held']['total']} "
          f"I={res['inject']['exact']}/{res['inject']['total']} ({i_pct:.0%}) "
          f"PPL={res['ppl']:.3f}{extra}", flush=True)
    return res


# ═══ Training (batched, autocast, process knobs) ════════════════════════════
def make_batch(engine, texts):
    tok = engine.tokenizer
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="right")
    ids = enc["input_ids"].to(engine.device)
    attn = enc["attention_mask"].to(engine.device)
    labels = ids.clone()
    labels[attn == 0] = -100
    return ids, attn, labels


def ce_loss(model, ids, attn, labels):
    logits = model(ids, attention_mask=attn).logits
    sl = logits[..., :-1, :].contiguous()
    lb = labels[..., 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)


def train_lora(engine, loras, inject_facts, cfg):
    """Train only the LoRA (batched, bf16 autocast) with process knobs."""
    inject_texts = [make_text(q, a) for q, a in inject_facts]
    known_texts = [make_text(q, a) for q, a in KNOWN]
    params = [p for p in engine.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)

    n_batches = (len(inject_texts) + cfg["batch"] - 1) // cfg["batch"]
    n_steps = cfg["epochs"] * n_batches
    warmup = max(1, int(0.2 * n_steps)) if cfg["sched"] else 0
    ema = [] if not cfg["ema"] else [
        (p, p.detach().clone()) for p in params]

    def set_lr(t):
        if not cfg["sched"]:
            return
        if t < warmup:
            lr = cfg["lr"] * (t + 1) / warmup
        else:
            p = (t - warmup) / max(1, n_steps - warmup)
            lr = cfg["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))
        for g in opt.param_groups:
            g["lr"] = lr

    engine.model.train()
    step = 0
    t0 = time.time()
    final_loss = float("nan")
    for ep in range(cfg["epochs"]):
        order = list(range(len(inject_texts)))
        random.shuffle(order)
        for b in range(n_batches):
            set_lr(step)
            chunk = [inject_texts[i] for i in
                     order[b * cfg["batch"]:(b + 1) * cfg["batch"]]]
            if not chunk:
                continue
            ids, attn, labels = make_batch(engine, chunk)
            with torch.set_grad_enabled(True), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce_loss(engine.model, ids, attn, labels)

                # Replay: every k-th step, ADD a known-facts batch (rehearsal
                # without reducing inject exposure)
                if cfg["replay_ratio"] > 0 and \
                        step % max(1, round(1 / cfg["replay_ratio"])) == 0:
                    r_chunk = [known_texts[(step + j) % len(known_texts)]
                               for j in range(min(cfg["batch"], len(known_texts)))]
                    r_ids, r_attn, r_labels = make_batch(engine, r_chunk)
                    loss = loss + ce_loss(engine.model, r_ids, r_attn, r_labels)

                # KL anchor to frozen parent (per-token mean). With qa_anchor,
                # rotate through prose + QA-format anchors: phase-1 showed the
                # held-fact flip comes from QA-answer format drift, which
                # prose-only KL does not cover.
                if cfg["kl_weight"] > 0:
                    pool = ANCHORS + QA_ANCHORS if cfg.get("qa_anchor") else ANCHORS
                    a_ids = encode(engine, pool[step % len(pool)])
                    with bypass_all(loras), torch.no_grad():
                        t_logits = engine.model(a_ids).logits[..., :-1, :]
                        t_flat = t_logits.reshape(-1, t_logits.size(-1))
                    s_logits = engine.model(a_ids).logits[..., :-1, :]
                    s_flat = s_logits.reshape(-1, s_logits.size(-1))
                    loss = loss + cfg["kl_weight"] * F.kl_div(
                        F.log_softmax(s_flat.float(), dim=-1),
                        F.log_softmax(t_flat.float(), dim=-1),
                        reduction="batchmean", log_target=True)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if ema:
                with torch.no_grad():
                    for p, shadow in ema:
                        shadow.mul_(cfg["ema"]).add_(p.detach(), alpha=1 - cfg["ema"])
            final_loss = loss.item()
            step += 1
        if cfg.get("early_stop") and (ep + 1) % 10 == 0:
            engine.model.eval()
            got = sum(exact_match(engine, q, a) for q, a in inject_facts[:20])
            engine.model.train()
            if got >= cfg["early_stop"]:
                print(f"  early-stop at ep{ep + 1} ({got}/20 probe correct)", flush=True)
                break
    if ema:
        with torch.no_grad():
            for p, shadow in ema:
                p.copy_(shadow)
    engine.model.eval()
    return {"steps": step, "train_time": time.time() - t0, "final_loss": final_loss}


# ═══ Condition runner ═══════════════════════════════════════════════════════
def run_condition(engine, tag, process, rank, n_facts, epochs=EPOCHS_DEFAULT,
                  lr=LR_DEFAULT, seed=42, train=True):
    """One full injection cycle on the shared engine (pristine parent restored)."""
    random.seed(seed)
    torch.manual_seed(seed)
    loras = fresh_trio(engine.model, rank, orth=process.get("orth", False))
    n_params = trio_params(loras)
    print(f"\n{'=' * 64}\n  {tag}  (rank={rank}, n_facts={n_facts}, "
          f"params={n_params}, p/f={n_params / n_facts:.1f})\n{'=' * 64}", flush=True)

    freeze_except_lora(engine.model)
    inject_subset = INJECT[:n_facts]

    base = measure(engine, "BASE", inject_subset)
    if BASE_PPL[0] is None:
        BASE_PPL[0] = base["ppl"]
    else:
        assert abs(base["ppl"] - BASE_PPL[0]) < 1e-9, \
            f"parent restore not lossless: {base['ppl']} vs {BASE_PPL[0]}"

    stats = {"steps": 0, "train_time": 0.0, "final_loss": float("nan")}
    if train:
        cfg = {"lr": lr, "epochs": epochs, "batch": BATCH,
                   "replay_ratio": process.get("replay", 0.0),
                   "kl_weight": process.get("kl", 0.0),
                   "sched": process.get("sched", False),
                   "ema": process.get("ema", 0.0),
                   "early_stop": process.get("early_stop", 0),
                   "qa_anchor": process.get("qa_anchor", False)}
        stats = train_lora(engine, loras, inject_subset, cfg)

    # Merge verification (per module):
    #   1. bitwise weight reconstruction: merged W == W0 + scale*B@A
    #   2. layer-level relative equivalence on captured inputs < LAYER_REL_TOL
    #   3. full-model logit diff reported (fp32 GEMM reassociation amplifies
    #      downstream; only the zero-delta noop merge is bit-exact)
    probe = encode(engine, "Q: What is the capital of France?\nA:")
    mlp = engine.model.model.layers[LAYER_IDX].mlp
    feats = {n: {} for n in TRIO}

    def _hook(name):
        def h(mod, inp, out):
            feats[name]["in"] = inp[0].detach().clone()
            feats[name]["out"] = out.detach().clone()
        return h

    handles = [getattr(mlp, n).register_forward_hook(_hook(n)) for n in TRIO]
    with torch.no_grad():
        pre_logits = engine.model(probe).logits.clone()
    for h in handles:
        h.remove()
    expected = {n: loras[i].W0 + loras[i].delta_W() for i, n in enumerate(TRIO)}
    expected_b = {n: (loras[i].bias.data.clone() if loras[i].has_bias else None)
                  for i, n in enumerate(TRIO)}

    unfreeze_parent(engine.model)          # user flow: unfreeze parent ...
    merge_trio(engine.model)               # ... then merge delta into parent
    max_rel = 0.0
    for _i, n in enumerate(TRIO):
        merged_lin = getattr(mlp, n)
        assert torch.equal(merged_lin.weight.data, expected[n].data), \
            f"merge weight reconstruction failed on {n}"
        if expected_b[n] is not None:
            assert torch.equal(merged_lin.bias.data, expected_b[n])
        with torch.no_grad():
            out_merged = F.linear(feats[n]["in"], merged_lin.weight, merged_lin.bias)
        rel = ((feats[n]["out"] - out_merged).abs().max()
               / feats[n]["out"].abs().max().clamp_min(1e-9)).item()
        max_rel = max(max_rel, rel)
    with torch.no_grad():
        post_logits = engine.model(probe).logits
    max_diff = (pre_logits - post_logits).abs().max().item()
    if train:
        assert max_rel < LAYER_REL_TOL, f"merge layer mismatch: rel={max_rel}"
        assert max_diff < MERGE_LOGIT_TOL, f"merge logit diverged: {max_diff}"
    else:
        assert max_rel == 0.0 and max_diff == 0.0, \
            f"zero-delta merge not bit-exact: rel={max_rel} logit={max_diff}"
    print(f"  merge check: layer_rel={max_rel:.2e}  max_logit_diff={max_diff:.2e}")

    final = measure(engine, "MERGED", inject_subset, base=base)
    recall = final["inject"]["exact"] / max(final["inject"]["total"], 1)
    ok = (recall >= 0.9 and final["known_reg"] <= 0 and final["held_reg"] <= 0)
    return {"tag": tag, "process": process.get("name", "custom"), "rank": rank,
            "n_facts": n_facts, "lora_params": n_params,
            "params_per_fact": n_params / n_facts,
            "baseline": base, "final": final, "recall": recall,
            "known_reg": final["known_reg"], "held_reg": final["held_reg"],
            "ppl_delta_pct": final["ppl_delta_pct"], "targets_met": ok,
            "merge_max_diff": max_diff, **stats}


# ═══ Phases ═════════════════════════════════════════════════════════════════
PROCESSES = {
    "noop": {"name": "noop"},
    "plain": {"name": "plain"},
    "replay": {"name": "replay", "replay": 0.25},
    "kl": {"name": "kl", "kl": 1.0},
    "orth": {"name": "orth", "orth": True},
    "sched": {"name": "sched", "sched": True},
    "ema": {"name": "ema", "ema": 0.99},
    "early": {"name": "early", "early_stop": 19},   # stop at >=95% of 20-probe
    "full": {"name": "full", "orth": True, "sched": True, "replay": 0.25, "kl": 1.0},
    # Phase-1 follow-ups targeting the single held-fact flip under "full":
    "full_ema": {"name": "full_ema", "orth": True, "sched": True, "replay": 0.25,
                 "kl": 1.0, "ema": 0.99},
    "full_qa": {"name": "full_qa", "orth": True, "sched": True, "replay": 0.25,
                "kl": 1.0, "qa_anchor": True},
    "full_ema_qa": {"name": "full_ema_qa", "orth": True, "sched": True,
                    "replay": 0.25, "kl": 1.0, "ema": 0.99, "qa_anchor": True},
}


def phase_sanity(engine):
    """No-op cycle + cached-vs-slow eval equivalence proof."""
    # 1. Eval equivalence on a fact sample (before trusting the cached path)
    eq = all(exact_match_slow(engine, q, a) == exact_match_cached(engine, q, a)
             for q, a in (INJECT[:10] + KNOWN[:5] + HELD[:5]))
    print(f"  cached-vs-slow eval equivalence: {'PASS' if eq else 'FAIL'}")
    if eq:
        USE_CACHED_EVAL[0] = True
    res = run_condition(engine, "sanity_noop", PROCESSES["noop"], rank=16,
                        n_facts=100, train=False)
    assert res["final"]["inject"]["exact"] == res["baseline"]["inject"]["exact"], \
        "noop changed inject facts!"
    assert res["known_reg"] == 0 and res["held_reg"] == 0, "noop regressed facts!"
    assert res["merge_max_diff"] == 0.0
    print(f"\n  SANITY PASS: zero-delta merge bit-exact, "
          f"ppl_delta={res['ppl_delta_pct']:+.4f}%")
    return res


def phase1(engine, n_facts=100, epochs=EPOCHS_DEFAULT):
    """Process comparison at the established operating point (r16 trio, b16)."""
    print(f"\n{'#' * 70}\n  PHASE 1: process comparison (r16 trio, {n_facts} facts, "
          f"{epochs} epochs)\n{'#' * 70}", flush=True)
    res = [phase_sanity(engine)]
    for key in ["plain", "replay", "kl", "orth", "sched", "ema", "early", "full"]:
        res.append(run_condition(engine, f"p1_{key}", PROCESSES[key], rank=16,
                                 n_facts=n_facts, epochs=epochs))
        with open(os.path.join(HERE, "r29_phase1.json"), "w") as f:
            json.dump(res, f, indent=2)  # checkpoint after each condition
    res.sort(key=lambda r: (not r["targets_met"], -r["recall"], r["ppl_delta_pct"]))
    print(f"\n{'=' * 92}\n  PHASE 1 RESULTS (targets: recall>=90%, 0 known/held "
          f"regression)\n{'=' * 92}")
    print(f"  {'process':<8} {'recall':>7} {'Kreg':>5} {'Hreg':>5} {'dPPL%':>8} "
          f"{'steps':>6} {'train_s':>8} {'params':>8} {'p/f':>7} {'met':>4}")
    for r in res:
        print(f"  {r['process']:<8} {r['recall']:>6.0%} {r['known_reg']:>5} "
              f"{r['held_reg']:>5} {r['ppl_delta_pct']:>+8.2f} "
              f"{r['steps']:>6} {r['train_time']:>8.1f} "
              f"{r['lora_params']:>8} {r['params_per_fact']:>7.1f} "
              f"{'YES' if r['targets_met'] else 'no':>4}")
    with open(os.path.join(HERE, "r29_phase1.json"), "w") as f:
        json.dump(res, f, indent=2)
    best = next((r for r in res if r["targets_met"]), None)
    print(f"\n  BEST PROCESS: {best['process'] if best else 'NONE met targets'}")
    return res


def phase2(engine, process_key="full", ranks=(2, 4, 8, 16, 32),
           sizes=(25, 50, 100, 200, 400), epochs=EPOCHS_DEFAULT):
    """Golden ratio sweep: min params per knowledge size at the winning process."""
    print(f"\n{'#' * 70}\n  PHASE 2: golden ratio (process={process_key}, "
          f"epochs={epochs})\n{'#' * 70}", flush=True)
    res = []
    for n_facts in sizes:
        for rank in ranks:
            r = run_condition(engine, f"p2_r{rank}_n{n_facts}",
                              PROCESSES[process_key], rank=rank,
                              n_facts=n_facts, epochs=epochs)
            res.append(r)
            with open(os.path.join(HERE, "r29_phase2.json"), "w") as f:
                json.dump(res, f, indent=2)  # checkpoint after each condition
    analyze_golden_ratio(res)
    return res


def analyze_golden_ratio(res):
    print(f"\n{'=' * 90}\n  GOLDEN RATIO ANALYSIS (min params meeting: recall>=90%, "
          f"0 regression)\n{'=' * 90}")
    frontier = {}
    for r in res:
        if r["targets_met"]:
            n = r["n_facts"]
            if n not in frontier or r["lora_params"] < frontier[n]["lora_params"]:
                frontier[n] = r
    if not frontier:
        print("  NO condition met targets — full grid:")
        for r in sorted(res, key=lambda r: (r["n_facts"], -r["recall"])):
            print(f"  r{r['rank']:<3} n{r['n_facts']:<4} recall={r['recall']:.0%} "
                  f"Kreg={r['known_reg']} Hreg={r['held_reg']} "
                  f"dPPL={r['ppl_delta_pct']:+.1f}%")
        return
    print(f"  {'n_facts':>7} {'min_rank':>9} {'params':>8} {'params/fact':>12} {'recall':>7}")
    ns, ps = [], []
    for n in sorted(frontier):
        r = frontier[n]
        ns.append(n)
        ps.append(r["lora_params"])
        print(f"  {n:>7} {r['rank']:>9} {r['lora_params']:>8} "
              f"{r['lora_params'] / n:>12.1f} {r['recall']:>7.0%}")
    if len(ns) >= 2:
        import numpy as np
        A = np.vstack([ns, np.ones(len(ns))]).T
        c, p0 = np.linalg.lstsq(A, ps, rcond=None)[0]
        print(f"\n  FIT: params = {p0:.0f} + {c:.1f} * n_facts")
        print(f"  GOLDEN RATIO (marginal params/fact) = {c:.1f} "
              f"({c * 4:.1f} bytes/fact fp32, {c:.1f} bytes int8)")
        print(f"  Structural floor P0 = {p0:.0f} params")
        # Each inject fact is a random 6-digit code ~= log2(10^6) = 19.93 bits
        # of TRUE new information (worst case: uniform, no prior to lean on).
        bits_per_fact = math.log2(10 ** 6)
        print(f"  INFO DENSITY: {bits_per_fact:.1f} bits/fact at c={c:.1f} "
              f"params/fact -> {bits_per_fact / c:.3f} bits per LoRA param "
              f"(capacity-law reference: ~2 bits/param full training)")


# ═══ Main ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="1", choices=["sanity", "1", "2", "cond"])
    ap.add_argument("--process", default="full")
    ap.add_argument("--n-facts", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    print("=== R29: LoRA knowledge injection (Qwen 2.5 0.5B via ForgeEngine) ===")
    print(f"Device: {DEV}  facts: {len(INJECT)} inject / {len(KNOWN)} known / "
          f"{len(HELD)} held / {len(ANCHORS)} anchors", flush=True)

    tok = AutoTokenizer.from_pretrained(QWEN_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWEN_DIR, dtype=DTYPE).to(DEV)
    model.eval()
    engine = ForgeEngine(model, tok, device=str(DEV))
    capture_original(model)
    parent_params = sum(p.numel() for p in model.parameters())
    print(f"Parent params: {parent_params / 1e6:.1f}M  "
          f"target: L{LAYER_IDX} FFN trio {TRIO}")
    print(f"LoRA param growth: rank r -> 3 * r * (896+4864) = r*{3 * (896 + 4864)} "
          f"({100 * 16 * 3 * (896 + 4864) / parent_params:.4f}% at r=16)")

    if args.phase == "sanity":
        phase_sanity(engine)
    elif args.phase == "1":
        phase1(engine, n_facts=args.n_facts, epochs=args.epochs)
    elif args.phase == "2":
        phase2(engine, process_key=args.process, epochs=args.epochs)
    elif args.phase == "cond":
        # Single extra condition (phase-1 follow-ups); appends to r29_phase1b.json
        out = os.path.join(HERE, "r29_phase1b.json")
        res = []
        if os.path.exists(out):
            with open(out) as f:
                res = json.load(f)
        tag = f"cond_{args.process}_r{args.rank}_n{args.n_facts}_e{args.epochs}"
        if any(r["tag"] == tag for r in res):
            print(f"  {tag} already recorded")
        else:
            r = run_condition(engine, tag, PROCESSES[args.process], rank=args.rank,
                              n_facts=args.n_facts, epochs=args.epochs)
            res.append(r)
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
    print("\n=== R29 complete ===")


if __name__ == "__main__":
    main()

"""EXP-1 train-less generalization: live memory surfaces on stock Qwen2.5-0.5B.

Attaches TitanMemory fast-weight side-paths (zero-init gate -> lossless)
and an NGramMemory exact-suffix sidecar, then measures held-out NLL before
vs after ingest — generalization without a single gradient step on params.

Phases:
  A) lossless-attach verification (gates closed -> NLL identical)
  B) ingest wikitext train -> eval test NLL live vs frozen
  C) verbatim recall: ingest a doc, eval that doc
  D) engine-level SuffixMemory generation sanity check
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from forge.exp.harness import QWEN_ID, load_engine, log_result, wikitext_docs

MAX_LEN = 256


def docs_ids(eng, texts):
    out = []
    for t in texts:
        ids = eng.tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=MAX_LEN).input_ids.long().to("cuda")
        if ids.shape[1] >= 32:
            out.append(ids)
    return out


@torch.no_grad()
def model_nll(model, ids_list):
    tot, n = 0.0, 0
    for ids in ids_list:
        loss = model(ids[:, :-1], targets=ids[:, 1:])[1]
        tot += loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return tot / max(n, 1)


class CastTitan(torch.nn.Module):
    """fp32 TitanMemory behind a dtype shim — bf16 models feed it bf16
    activations, but Titan's inner loop is fp32 end-to-end."""

    def __init__(self, mem, out_dtype):
        super().__init__()
        self.mem = mem
        self._out_dtype = out_dtype

    @property
    def gate(self):
        return self.mem.gate

    def forward(self, x):
        return self.mem(x.float()).to(self._out_dtype)

    def update(self, x, **kw):
        return self.mem.update(x.float(), **kw)

    def snapshot_state(self):
        return self.mem.snapshot_state()

    def restore_state(self, s):
        return self.mem.restore_state(s)

    def reset_state(self):
        return self.mem.reset_state()


def attach_titan(model, layers, solver="solve"):
    from forge.keys.architecture.titan_memory_key import TitanMemory
    d = model.config.d_model
    p = next(model.parameters())
    for i in layers:
        mem = TitanMemory(d, update_lr=0.5, forget=0.99,
                          solver=solver, ridge=1.0).to(p.device)
        model.blocks[i]._memory = CastTitan(mem, p.dtype).to(p.device)
        model.blocks[i]._mem_gate_zero = None


@torch.no_grad()
def eval_live(sess, ids, use_live=True):
    """eval_nll without live-state snapshots: eval forwards don't mutate
    TitanMemory/NGramMemory state (writes happen only in ingest), so the
    deep-copy snapshot — several GB at 151936 vocab — is unnecessary."""
    import torch.nn.functional as F
    ids = ids.reshape(1, -1).to(next(sess.model.parameters()).device)
    saved = {}
    if not use_live:
        for i, mem in sess.mem_layers.items():
            saved[i] = float(mem.gate.item())
            mem.gate.data.zero_()
            sess._blocks[i]._mem_gate_zero = None
    try:
        sess.reset()
        losses = []
        chunk = max(sess.cfg.chunk, 2)
        for start in range(0, ids.shape[1] - 1, chunk):
            seg = ids[:, start:start + chunk + 1]
            logits = sess.model(
                seg, use_cache=True,
                past_key_values=[None] * len(sess._blocks))[0]
            if use_live and sess.sparse is not None:
                logits = logits + sess.sparse.lift(seg)
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                seg[:, 1:].reshape(-1), reduction="none")
            losses.append(float(ce.mean()))
        return sum(losses) / max(len(losses), 1)
    finally:
        for i, v in saved.items():
            sess.mem_layers[i].gate.data.fill_(v)
            sess._blocks[i]._mem_gate_zero = None


def main():
    eng = load_engine(QWEN_ID, warmup=False)
    model, tok = eng.model, eng.tokenizer
    n_layers = len(model.blocks)
    test_ids = docs_ids(eng, wikitext_docs(6, split="test"))
    train_ids = docs_ids(eng, wikitext_docs(6, split="train"))
    test_cat = torch.cat(test_ids, dim=1)

    nll_base = model_nll(model, test_ids)
    log_result("trainless", phase="baseline", nll=round(nll_base, 4))

    from forge.model.live import LiveConfig, LiveSession
    from forge.model.ngram_mem import NGramMemory

    for n_mem in (8, n_layers):
        layers = list(range(n_layers - n_mem, n_layers))
        # fresh attach per setting (reset previous)
        for b in model.blocks:
            b._memory = None
            b._mem_gate_zero = None
        attach_titan(model, layers, solver="solve")

        # Phase A: gates closed -> must be lossless
        nll_closed = model_nll(model, test_ids)
        log_result("trainless", phase="lossless_check", n_mem=n_mem,
                   nll=round(nll_closed, 4),
                   delta=round(nll_closed - nll_base, 6))

        sess = LiveSession(model, LiveConfig(
            chunk=128, solver="solve", ridge=1.0, journal=False))
        ng = NGramMemory(model.config.vocab_size, orders=(4, 8), gate=1.0)
        sess.attach_sparse(ng)

        # Phase B: ingest train docs, eval held-out test
        t0 = time.perf_counter()
        for ids in train_ids:
            sess.ingest(ids)
        ing_s = time.perf_counter() - t0
        sess.open_gates(1.0)
        nll_live = eval_live(sess, test_cat, use_live=True)
        nll_frozen = eval_live(sess, test_cat, use_live=False)
        log_result("trainless", phase="generalization", n_mem=n_mem,
                   nll_live=round(nll_live, 4),
                   nll_frozen=round(nll_frozen, 4),
                   delta=round(nll_frozen - nll_live, 4),
                   ingest_s=round(ing_s, 1))

        # Phase C: verbatim recall — ingest the eval docs themselves
        for ids in test_ids:
            sess.ingest(ids)
        nll_verbatim = eval_live(sess, test_cat, use_live=True)
        log_result("trainless", phase="verbatim", n_mem=n_mem,
                   nll=round(nll_verbatim, 4),
                   delta_vs_frozen=round(nll_frozen - nll_verbatim, 4))
        del sess

    # Phase D: engine SuffixMemory sanity
    eng.enable_suffix_memory(gate=1.0)
    r = eng.benchmark("a b c a b c a b c a b c a b", max_new_tokens=32,
                      n_runs=2)
    log_result("trainless", phase="suffix_mem_engine",
               tok_s=round(r["tokens_per_sec"], 1))
    print("[trainless] done")


if __name__ == "__main__":
    main()

"""Offline teacher-logprob scorer for on-policy distillation.

Scores each SFT row's completion with a teacher model (per-token logprobs
in the TEACHER's own vocab), then maps those logprobs back onto STUDENT
completion tokens via character-offset overlap. Writes `teacher_logp` —
one float per student completion token (-99 = unscored) — onto each row.

`sft_train --distill` consumes the cached array as an advantage signal
(ChunkedDistillCE): reinforce tokens the teacher likes more than the
student, suppress tokens the student is overconfident on. Sequence-level
scoring means teacher and student tokenizers/architectures never have to
match (Qwen 151K teacher → LFM3 128K student works).

Teacher sources:
    --teacher-hf    any HF causal-LM id/dir (bnb int4/int8/bf16)
    --teacher-forge a Forge checkpoint dir (ForgeEngine port, int4 default)

GPU budget: run this standalone (GPU<->API serialized rule). An int4 7B
teacher is ~5.5GB — do not keep a student engine resident.

Usage:
    venv\\Scripts\\python.exe scripts/score_teacher_logprobs.py \
        --in results/qwen_distill_v10.jsonl \
        --out results/qwen_distill_v10_scored.jsonl \
        --student-tokenizer research/checkpoints/lfm3_1p7b_cc \
        --teacher-hf Qwen/Qwen2.5-Coder-7B-Instruct --teacher-quant int4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from forge.training.losses.chunked_ce import ChunkedDistillCE  # noqa: E402
from forge.training.runners.sft_train import _chat_render, render_single_turn  # noqa: E402

MISSING = ChunkedDistillCE.TEACHER_LOGP_MISSING  # -99.0
_GATHER_CHUNK = 512  # positions per log_softmax slice (VRAM bound at 151K vocab)


# ── Tokenizer adapters ────────────────────────────────────────────────────

def _load_tokenizer(path_or_id):
    """Return (kind, tokenizer) where kind is 'hf' or 'tokenizers'."""
    p = Path(path_or_id)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(p) if p.exists() else path_or_id, trust_remote_code=True)
        if tok.is_fast:
            return "hf", tok
        return "hf_slow", tok
    except Exception:
        pass
    tok_file = p / "tokenizer.json"
    if tok_file.exists():
        from tokenizers import Tokenizer
        return "tokenizers", Tokenizer.from_file(str(tok_file))
    from transformers import AutoTokenizer
    return "hf", AutoTokenizer.from_pretrained(path_or_id,
                                             trust_remote_code=True)


def _encode(kind, tok, text):
    """Tokenize text -> (ids, offsets). Offsets may be None if unsupported."""
    if kind == "hf":
        try:
            enc = tok(text, add_special_tokens=False,
                      return_offsets_mapping=True)
            return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        except (TypeError, KeyError):
            return list(tok(text, add_special_tokens=False)["input_ids"]), None
    if kind == "tokenizers":
        enc = tok.encode(text)
        return list(enc.ids), [tuple(o) for o in enc.offsets]
    # hf_slow
    return list(tok(text, add_special_tokens=False)["input_ids"]), None


def _decode(kind, tok, ids):
    if kind == "tokenizers":
        return tok.decode(list(ids))
    return tok.decode(list(ids), skip_special_tokens=False)


def _fallback_offsets(kind, tok, text, ids):
    """Sequential decode-search alignment when offset mapping is missing.
    Approximate (BPE boundary whitespace), last-resort only."""
    spans, cur = [], 0
    for tid in ids:
        s = _decode(kind, tok, [tid])
        idx = text.find(s, cur) if s else -1
        if idx < 0:
            idx = cur
        spans.append((idx, idx + len(s)))
        cur = idx + len(s)
    return spans


# ── Teacher backends ──────────────────────────────────────────────────────

class _HFTeacher:
    def __init__(self, model_id, quant):
        from transformers import AutoModelForCausalLM
        kw = dict(trust_remote_code=True, dtype=torch.bfloat16,
                  device_map="auto")
        if quant == "int4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        elif quant == "int8":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        self.name = model_id
        self.dev = next(self.model.parameters()).device

    def forward(self, ids, mask):
        with torch.inference_mode():
            return self.model(input_ids=ids, attention_mask=mask).logits


class _ForgeTeacher:
    """Forge-native checkpoint (e.g. a ported ForgeLM/LFM model)."""

    def __init__(self, ckpt, quant):
        from forge.engine.forge_engine import ForgeEngine
        self.eng = ForgeEngine.from_checkpoint(
            ckpt, load_quantize=quant if quant != "none" else "auto",
            auto_activate=False)
        self.name = f"forge:{ckpt}"
        self.dev = next(self.eng.model.parameters()).device

    def forward(self, ids, mask):
        with self.eng._gen_lock, torch.inference_mode():
            out = self.eng.model(ids, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out


# ── Scoring core ──────────────────────────────────────────────────────────

def _token_logps(logits, ids_row, mask_row):
    """Per-token logp for one padded row: logp[i] = P(id[i] | ids[:i]).

    Chunked over positions to avoid a [T, V] fp32 softmax spike.
    Returns a list aligned to ids_row (index 0 gets 0.0 — no context).
    """
    T = int(mask_row.sum().item())
    ids_row = ids_row[:T]
    out = torch.zeros(T, dtype=torch.float32)
    for s in range(0, T - 1, _GATHER_CHUNK):
        e = min(s + _GATHER_CHUNK, T - 1)
        lp = torch.log_softmax(logits[s:e].float(), dim=-1)  # predicts pos s+1
        tgt = ids_row[s + 1:e + 1].to(lp.device)
        out[s + 1:e + 1] = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).cpu()
    return out.tolist(), T


def _map_to_student(student_spans, comp_start_tok, teacher_logps,
                    teacher_offsets):
    """Overlap-weighted mean of teacher logps over each student comp token."""
    res = []
    n_tok = len(teacher_offsets)
    for i in range(comp_start_tok, len(student_spans)):
        s, e = student_spans[i]
        if e <= s:
            res.append(MISSING)
            continue
        num = den = 0.0
        for t in range(n_tok):
            ts, te = teacher_offsets[t]
            ov = min(e, te) - max(s, ts)
            if ov > 0:
                w = ov / (e - s)
                num += teacher_logps[t] * w
                den += w
        res.append(num / den if den > 0 else MISSING)
    return res


def score_rows(rows, teacher, t_kind, t_tok, s_kind, s_tok,
               batch_size=4, max_seq=8192):
    """Annotate rows with teacher_logp lists. Returns (rows, stats)."""
    # ── Pre-tokenize student side (CPU, cheap) ──
    prepared = []
    for row in rows:
        try:
            full_text, _ = render_single_turn(row["prompt"], row["response"])
        except Exception:
            continue
        s_ids, s_off = _encode(s_kind, s_tok, full_text)
        # Match tokenize_example: prompt = rendered prompt_text portion.
        prompt_text = _chat_render([{"role": "user", "content": row["prompt"]}],
                                   add_generation_prompt=True)
        p_ids, _ = _encode(s_kind, s_tok, prompt_text)
        comp_tok = len(p_ids)
        if comp_tok >= len(s_ids) - 1 or len(s_ids) < 4:
            continue
        if s_off is None:
            s_off = _fallback_offsets(s_kind, s_tok, full_text, s_ids)
        prepared.append((row, full_text, s_ids, s_off, comp_tok))
    if not prepared:
        return rows, {"scored": 0, "missing_rows": len(rows)}

    # ── Teacher tokenize + batch forward ──
    n_scored = n_missing_tok = 0
    order = sorted(range(len(prepared)),
                   key=lambda i: len(prepared[i][2]))
    for bstart in range(0, len(order), batch_size):
        idxs = order[bstart:bstart + batch_size]
        t_ids_list, t_off_list = [], []
        for i in idxs:
            t_ids, t_off = _encode(t_kind, t_tok, prepared[i][1])
            t_ids = t_ids[:max_seq]
            t_off = t_off[:max_seq] if t_off else None
            t_ids_list.append(t_ids)
            t_off_list.append(t_off)
        max_t = max(len(x) for x in t_ids_list)
        pad_id = getattr(t_tok, "pad_token_id", None) or 0
        ids = torch.full((len(idxs), max_t), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idxs), max_t), dtype=torch.long)
        for j, t_ids in enumerate(t_ids_list):
            ids[j, :len(t_ids)] = torch.tensor(t_ids, dtype=torch.long)
            mask[j, :len(t_ids)] = 1
        logits = teacher.forward(ids.to(teacher.dev), mask.to(teacher.dev))
        for j, i in enumerate(idxs):
            row, full_text, s_ids, s_off, comp_tok = prepared[i]
            logps, T = _token_logps(logits[j], ids[j], mask[j])
            t_off = t_off_list[j]
            if t_off is None:
                t_off = _fallback_offsets(t_kind, t_tok, full_text,
                                          t_ids_list[j])
            tlp = _map_to_student(s_off, comp_tok, logps[:len(t_off)], t_off)
            n_scored += 1
            n_missing_tok += sum(1 for v in tlp if v <= MISSING)
            row["teacher_logp"] = [round(float(v), 4) for v in tlp]
            row["teacher_model"] = teacher.name
        done = min(bstart + batch_size, len(order))
        print(f"  scored {done}/{len(order)} rows "
              f"({n_missing_tok} unmapped tokens so far)", flush=True)
    return rows, {"scored": n_scored, "missing_tokens": n_missing_tok}


def _row_key(row):
    h = hashlib.sha1()
    h.update((row.get("prompt", "") + "ÿ" + row.get("response", "")).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="Input JSONL files ({prompt, response, ...} rows)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--student-tokenizer", required=True,
                    help="Student tokenizer dir (e.g. research/checkpoints/lfm3_1p7b_cc)")
    ap.add_argument("--teacher-hf", default=None,
                    help="HF teacher model id/dir")
    ap.add_argument("--teacher-forge", default=None,
                    help="Forge checkpoint dir teacher (port preset)")
    ap.add_argument("--teacher-quant", default="int4",
                    choices=["none", "int4", "int8"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq", type=int, default=8192,
                    help="Teacher context cap per row (longer rows truncated)")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--only-verified", action="store_true",
                    help="Skip rows with verified == False")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already present (and scored) in --out")
    args = ap.parse_args()

    if not args.teacher_hf and not args.teacher_forge:
        ap.error("one of --teacher-hf / --teacher-forge is required")

    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prompt" not in obj or "response" not in obj:
                    continue
                if args.only_verified and obj.get("verified") is False:
                    continue
                rows.append(obj)
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"Loaded {len(rows)} rows")

    done_keys = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    prev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if prev.get("teacher_logp"):
                    done_keys.add(_row_key(prev))
        rows = [r for r in rows if _row_key(r) not in done_keys]
        print(f"Resume: {len(done_keys)} already scored, {len(rows)} remaining")
    if not rows:
        print("Nothing to score.")
        return

    s_kind, s_tok = _load_tokenizer(args.student_tokenizer)
    if args.teacher_hf:
        t_kind, t_tok = _load_tokenizer(args.teacher_hf)
        teacher = _HFTeacher(args.teacher_hf, args.teacher_quant)
    else:
        t_kind, t_tok = _load_tokenizer(args.teacher_forge)
        teacher = _ForgeTeacher(args.teacher_forge, args.teacher_quant)
    print(f"Teacher: {teacher.name}  student-tok: {s_kind}  teacher-tok: {t_kind}")

    t0 = time.time()
    rows, stats = score_rows(rows, teacher, t_kind, t_tok, s_kind, s_tok,
                             batch_size=args.batch_size, max_seq=args.max_seq)
    mode = "a" if args.resume else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for row in rows:
            if row.get("teacher_logp"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Done in {time.time() - t0:.1f}s — stats: {stats} -> {out_path}")


if __name__ == "__main__":
    main()

"""QLoRA SFT: distill Roc-style RP patterns into Gemma-4-12B-OBLITERATED.

Reads JSONL transcripts from gen_roc_data.py, renders each conversation with
the student's own chat template, and masks loss to assistant turns only.

Run with the training venv (transformers >=5.10 required for gemma4):

  venv_train\\Scripts\\python.exe scripts\\distill\\train_qlora.py ^
      --data data\\roc_distill.jsonl --out adapters\\gemma4_roc_lora

VRAM (RTX 5070 12GB): 4bit base ~8GB + LoRA ~0.6GB + activations ~1.5GB.
"""
from __future__ import annotations

import argparse, json, random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

MODEL = "OBLITERATUS/Gemma-4-12B-OBLITERATED"
# Gemma4 turn markers, built by concat to keep them literal-safe.
T_OPEN = "<" + "|" + "turn>model\n"       # opens an assistant turn
T_THINK = "<" + "|" + "channel>thought\n<" + "channel" + "|" + ">"  # empty thought block
T_CLOSE = "<" + "turn" + "|" + ">"       # closes a turn


def build_features(tok, path, max_len, seed=1234):
    """JSONL -> token ids with labels masked outside assistant turns."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(rows)
    feats, skipped = [], 0
    for r in rows:
        msgs = r["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > max_len:
            skipped += 1
            continue
        # locate assistant spans in the RENDERED string, then map to tokens
        # via incremental re-tokenization of prefixes (offsets are unreliable
        # across template whitespace, so do it on the id level):
        labels = [-100] * len(ids)
        open_ids = tok(T_OPEN, add_special_tokens=False).input_ids
        close_ids = tok(T_CLOSE, add_special_tokens=False).input_ids
        i = 0
        while i < len(ids):
            if ids[i:i + len(open_ids)] == open_ids:
                j = i + len(open_ids)
                # optional empty-thought block right after opener
                th = tok(T_THINK, add_special_tokens=False).input_ids
                if ids[j:j + len(th)] == th:
                    j += len(th)
                # mask from content start through the next turn-close (incl.)
                start = j
                while j < len(ids) and ids[j:j + len(close_ids)] != close_ids:
                    j += 1
                end = min(j + len(close_ids), len(ids))
                for k in range(start, end):
                    labels[k] = ids[k]
                i = j
            else:
                i += 1
        feats.append({"input_ids": ids, "labels": labels})
    print(f"dataset: {len(feats)} kept, {skipped} dropped (> {max_len} tok)")
    return feats


class SelectedLossTrainer(Trainer):
    """CE only on labeled positions — avoids materializing T x 262k logits.

    Standard causal-LM loss computes lm_head over the whole sequence; with a
    262144-token vocab that is ~6GB fp32 for len 4096 -> OOM on 12GB. Here we
    run the base model for hidden states, gather positions whose NEXT token is
    a labeled assistant token, and apply lm_head to just those rows.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs["labels"]
        inner = model.model           # PeftModel -> ForCausalLM
        out = inner.model(input_ids=inputs["input_ids"],
                          attention_mask=inputs["attention_mask"])
        h = out.last_hidden_state                          # [B, T, D]
        shift = labels[:, 1:]
        sel = shift != -100
        h_sel = h[:, :-1, :][sel]                          # [N, D]
        tgt = shift[sel]
        logits = inner.lm_head(h_sel).float()              # [N, V]
        cap = getattr(inner.config, "final_logit_softcapping", None)
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return torch.nn.functional.cross_entropy(logits, tgt)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        m = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * m)
        labs.append(b["labels"] + [-100] * m)
        att.append([1] * len(b["input_ids"]) + [0] * m)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data\roc_distill.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=r"adapters\gemma4_roc_lora")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    feats = build_features(tok, args.data, args.max_len)
    ds = Dataset.from_list(feats)

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.mbs,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=30,
        logging_steps=10, save_steps=args.save_steps, save_total_limit=3,
        bf16=True, optim="paged_adamw_8bit", report_to=[],
        gradient_checkpointing=True, remove_unused_columns=False)
    trainer = SelectedLossTrainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved adapter ->", args.out)


if __name__ == "__main__":
    main()

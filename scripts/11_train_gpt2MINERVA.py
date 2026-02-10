from __future__ import annotations

from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Base PH/Tagalog GPT-2
BASE_MODEL = "jcblaise/gpt2-tagalog"

CORPUS_DIR = Path("data/gpt2")
TRAIN_FILE = CORPUS_DIR / "train.txt"
VAL_FILE = CORPUS_DIR / "val.txt"

OUT_DIR = Path("models/gpt2_tagalog_finetuned")
LOG_DIR = Path("logs/gpt2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 256

EPOCHS = 3
LR = 5e-5

PER_DEVICE_BATCH = 4
GRAD_ACCUM = 8  # 4 * 8 = 32 effective batch


def main() -> None:
    for p in [TRAIN_FILE, VAL_FILE]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 10_prepare_gpt2MINERVA.py first.")

    raw = load_dataset(
        "text",
        data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)},
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"])

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i: i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        logging_dir=str(LOG_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.05,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        tokenizer=tok,
        data_collator=collator,
    )

    print(f"[GPT-2] Fine-tuning {BASE_MODEL} on MINERVA corpus...")
    trainer.train()

    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    print(f"[OK] Saved GPT-2 model -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "jcblaise/roberta-tagalog-base"

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

OUT_DIR = Path("models/roberta_finetuned")
LOG_DIR = Path("logs/roberta")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 256
SEED = 42

# Train RoBERTa only on Tagalog (recommended)
FILTER_LANG = "tl"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    set_seed(SEED)

    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 03_split_dataset.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if FILTER_LANG:
        train_df = train_df[train_df["lang"] ==
                            FILTER_LANG].reset_index(drop=True)
        val_df = val_df[val_df["lang"] == FILTER_LANG].reset_index(drop=True)
        test_df = test_df[test_df["lang"] ==
                          FILTER_LANG].reset_index(drop=True)

    if len(train_df) < 10:
        raise RuntimeError(
            "Too few Tagalog rows after filtering. Check your corpus preparation step.")

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    for ds in (train_ds, val_ds, test_ds):
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)
    test_ds.set_format("torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(LOG_DIR),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(test_ds)
    print("RoBERTa Tagalog Final Test Results:", results)

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

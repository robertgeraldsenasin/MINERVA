from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_MAP = {"real": 0, "true": 0, "credible": 0,
             "fake": 1, "false": 1, "not_credible": 1}


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def normalize_label_series(s: pd.Series) -> pd.Series:
    """Coerce labels to int {0,1}. Supports string labels."""
    if s.dtype.kind in ("i", "u"):
        y = s.astype(int)
    else:
        v = s.astype(str).str.strip().str.lower()
        y = v.map(lambda x: LABEL_MAP.get(x, x)).astype(int)

    uniq = set(y.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels found: {sorted(list(uniq))} (expected only 0/1)")
    return y


def pick_col(df: pd.DataFrame, preferred: Optional[str], candidates: Tuple[str, ...]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def torch_determinism(seed: int) -> None:
    """Best-effort determinism for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic settings (can slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


@torch.no_grad()
def predict_proba_fake(trainer: Trainer, ds: Dataset) -> np.ndarray:
    """Return P(fake) for a dataset."""
    out = trainer.predict(ds)
    logits = out.predictions
    # softmax for class 1
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    return probs[:, 1]


@dataclass
class TrainReport:
    task: str
    model_name: str
    seed: int
    run_id: str
    output_dir: str
    train_csv: str
    val_csv: str
    test_csv: str
    text_col: str
    label_col: str
    n_train: int
    n_val: int
    n_test: int
    best_eval_f1: Optional[float]
    eval_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    test_confusion_matrix: list


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINERVA Script 16: Generic transformer classifier trainer (seed/run aware).")
    ap.add_argument("--task", required=True,
                    help="e.g. roberta or distilbert (used for output paths).")
    ap.add_argument("--model_name", required=True,
                    help="HF model id, e.g. jcblaise/roberta-tagalog-base")
    ap.add_argument("--run_id", default=None,
                    help="Run identifier for grouping; default = timestamp.")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--splits_dir", default="data/processed",
                    help="Folder containing train/val/test CSVs.")
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--test_csv", default=None)

    ap.add_argument("--text_col", default=None)
    ap.add_argument("--label_col", default=None)

    ap.add_argument("--output_base", default="models",
                    help="Base dir for model outputs.")
    ap.add_argument("--logs_base", default="logs", help="Base dir for logs.")

    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch*grad_accum).")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--fp16", action="store_true",
                    help="Enable fp16 if CUDA available.")
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="auto | none | <checkpoint_path>",
    )
    args = ap.parse_args()

    run_id = args.run_id or now_id()

    # Resolve split paths
    def _default_paths(base: str) -> Tuple[str, str, str]:
        p1 = (os.path.join(base, "train.csv"), os.path.join(
            base, "val.csv"), os.path.join(base, "test.csv"))
        if all(os.path.exists(p) for p in p1):
            return p1
        p2 = (os.path.join(base, "train_split.csv"), os.path.join(
            base, "val_split.csv"), os.path.join(base, "test_split.csv"))
        if all(os.path.exists(p) for p in p2):
            return p2
        raise FileNotFoundError(
            f"Could not find train/val/test CSVs in {base}")

    train_csv, val_csv, test_csv = (
        (args.train_csv, args.val_csv, args.test_csv)
        if (args.train_csv and args.val_csv and args.test_csv)
        else _default_paths(args.splits_dir)
    )

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    text_col = pick_col(df_train, args.text_col,
                        ("text", "content", "article", "body", "post"))
    label_col = pick_col(df_train, args.label_col,
                         ("label", "class", "target", "y"))
    if not text_col or not label_col:
        raise ValueError(
            f"Could not infer text/label columns. Columns: {list(df_train.columns)}")

    for df in (df_train, df_val, df_test):
        df[label_col] = normalize_label_series(df[label_col])

    # Seeds
    set_seed(args.seed)
    torch_determinism(args.seed)

    # Output directories
    out_dir = os.path.join(args.output_base, args.task,
                           f"run_{run_id}", f"seed_{args.seed}")
    log_dir = os.path.join(args.logs_base, args.task,
                           f"run_{run_id}", f"seed_{args.seed}")
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    # Model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2)

    # HF datasets
    def to_ds(df: pd.DataFrame) -> Dataset:
        sub = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "labels"}).reset_index(drop=True)
        return Dataset.from_pandas(sub, preserve_index=False)

    ds_train = to_ds(df_train)
    ds_val = to_ds(df_val)
    ds_test = to_ds(df_test)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    ds_train = ds_train.map(tok, batched=True)
    ds_val = ds_val.map(tok, batched=True)
    ds_test = ds_test.map(tok, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())

    # TrainingArguments (Transformers 4.33.x uses evaluation_strategy)
    train_args = TrainingArguments(
        output_dir=out_dir,
        logging_dir=log_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,

        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        save_total_limit=args.save_total_limit,

        fp16=use_fp16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Resume logic
    resume = None
    if str(args.resume_from_checkpoint).lower() == "auto":
        ckpts = [p for p in Path(out_dir).glob("checkpoint-*") if p.is_dir()]
        if ckpts:
            # highest step
            ckpts_sorted = sorted(
                ckpts, key=lambda p: int(p.name.split("-")[-1]))
            resume = str(ckpts_sorted[-1])
    elif str(args.resume_from_checkpoint).lower() == "none":
        resume = None
    else:
        resume = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume)

    # Save a complete HuggingFace bundle at the root out_dir so downstream scripts can load it:
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Evaluate
    eval_metrics = trainer.evaluate(ds_val)
    test_pred = trainer.predict(ds_test)
    test_metrics = compute_metrics(
        (test_pred.predictions, test_pred.label_ids))
    cm = confusion_matrix(test_pred.label_ids, np.argmax(
        test_pred.predictions, axis=-1)).tolist()

    best_eval_f1 = float(eval_metrics.get("eval_f1")
                         ) if "eval_f1" in eval_metrics else None

    report = TrainReport(
        task=args.task,
        model_name=args.model_name,
        seed=args.seed,
        run_id=run_id,
        output_dir=out_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        text_col=text_col,
        label_col=label_col,
        n_train=len(df_train),
        n_val=len(df_val),
        n_test=len(df_test),
        best_eval_f1=best_eval_f1,
        eval_metrics={k: float(v) for k, v in eval_metrics.items(
        ) if isinstance(v, (int, float))},
        test_metrics=test_metrics,
        test_confusion_matrix=cm,
    )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print(f"[OK] Saved model bundle -> {out_dir}")
    print(f"[OK] Saved metrics -> {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()

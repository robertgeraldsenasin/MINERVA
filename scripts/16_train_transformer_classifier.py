import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_MAP = {"real": 0, "true": 0, "legit": 0, "fake": 1, "false": 1}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pick_col(df: pd.DataFrame, preferred: Optional[str], candidates: Tuple[str, ...]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _normalize_labels(series: pd.Series) -> pd.Series:
    if series.dtype.kind in ("i", "u"):
        return series.astype(int)
    # strings
    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda x: LABEL_MAP.get(x, x)).astype(int)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _find_splits(base_dir: str) -> Tuple[str, str, str]:
    # prefer train.csv/val.csv/test.csv, fallback to *_split.csv
    p1 = (os.path.join(base_dir, "train.csv"),
          os.path.join(base_dir, "val.csv"),
          os.path.join(base_dir, "test.csv"))
    if all(os.path.exists(p) for p in p1):
        return p1
    p2 = (os.path.join(base_dir, "train_split.csv"),
          os.path.join(base_dir, "val_split.csv"),
          os.path.join(base_dir, "test_split.csv"))
    if all(os.path.exists(p) for p in p2):
        return p2
    raise FileNotFoundError(
        f"Could not find train/val/test CSVs in {base_dir}. "
        "Expected train.csv/val.csv/test.csv or *_split.csv."
    )


def _torch_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is best-effort; can reduce speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


@dataclass
class TrainReport:
    task: str
    model_name: str
    seed: int
    train_csv: str
    val_csv: str
    test_csv: str
    text_col: str
    label_col: str
    output_dir: str
    best_metric: Optional[float]
    eval: Dict[str, float]
    test: Dict[str, float]
    confusion_matrix: Optional[list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g., roberta, distilbert")
    ap.add_argument("--model_name", required=True, help="HF model id")
    ap.add_argument("--splits_dir", default="data/processed",
                    help="directory with train/val/test CSVs")

    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--test_csv", default=None)

    ap.add_argument("--text_col", default=None)
    ap.add_argument("--label_col", default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    ap.add_argument("--run_id", default=None,
                    help="optional run identifier; default timestamp")
    ap.add_argument("--output_base", default="models",
                    help="base output directory")
    ap.add_argument("--resume_from_checkpoint", default="auto",
                    help="'auto', 'none', or a checkpoint path")

    args = ap.parse_args()

    run_id = args.run_id or _now_id()
    train_csv, val_csv, test_csv = (
        (args.train_csv, args.val_csv, args.test_csv)
        if (args.train_csv and args.val_csv and args.test_csv)
        else _find_splits(args.splits_dir)
    )

    df_train = _load_csv(train_csv)
    df_val = _load_csv(val_csv)
    df_test = _load_csv(test_csv)

    text_col = _pick_col(df_train, args.text_col,
                         ("text", "content", "post", "body"))
    label_col = _pick_col(df_train, args.label_col,
                          ("label", "class", "target", "y"))
    if not text_col or not label_col:
        raise ValueError(
            f"Could not infer columns. Found columns: {list(df_train.columns)}. "
            "Provide --text_col and --label_col."
        )

    for df in (df_train, df_val, df_test):
        df[label_col] = _normalize_labels(df[label_col])

    # Seed control
    set_seed(args.seed)
    _torch_determinism(args.seed)

    out_dir = os.path.join(args.output_base, args.task,
                           f"run_{run_id}", f"seed_{args.seed}")
    log_dir = os.path.join(
        "logs", args.task, f"run_{run_id}", f"seed_{args.seed}")
    _ensure_dir(out_dir)
    _ensure_dir(log_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2)

    def tok(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=args.max_len,
        )

    ds_train = Dataset.from_pandas(
        df_train[[text_col, label_col]].rename(columns={label_col: "labels"}))
    ds_val = Dataset.from_pandas(
        df_val[[text_col, label_col]].rename(columns={label_col: "labels"}))
    ds_test = Dataset.from_pandas(
        df_test[[text_col, label_col]].rename(columns={label_col: "labels"}))

    ds_train = ds_train.map(tok, batched=True)
    ds_val = ds_val.map(tok, batched=True)
    ds_test = ds_test.map(tok, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fp16 = torch.cuda.is_available()

    tr_args = TrainingArguments(
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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        seed=args.seed,
        data_seed=args.seed,

        fp16=fp16,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    resume = None
    if args.resume_from_checkpoint.lower() == "auto":
        # find latest checkpoint-*
        candidates = [d for d in os.listdir(
            out_dir) if d.startswith("checkpoint-")]
        if candidates:
            candidates.sort(key=lambda x: int(x.split("-")[-1]))
            resume = os.path.join(out_dir, candidates[-1])
    elif args.resume_from_checkpoint.lower() == "none":
        resume = None
    else:
        resume = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume)

    eval_metrics = trainer.evaluate(ds_val)
    test_pred = trainer.predict(ds_test)
    test_metrics = _compute_metrics(
        (test_pred.predictions, test_pred.label_ids))
    cm = confusion_matrix(test_pred.label_ids, np.argmax(
        test_pred.predictions, axis=-1)).tolist()

    report = TrainReport(
        task=args.task,
        model_name=args.model_name,
        seed=args.seed,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        text_col=text_col,
        label_col=label_col,
        output_dir=out_dir,
        best_metric=float(eval_metrics.get("eval_f1")
                          ) if "eval_f1" in eval_metrics else None,
        eval={k: float(v) for k, v in eval_metrics.items()
              if isinstance(v, (int, float))},
        test=test_metrics,
        confusion_matrix=cm,
    )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print(f"[OK] Saved model+metrics -> {out_dir}")


if __name__ == "__main__":
    main()

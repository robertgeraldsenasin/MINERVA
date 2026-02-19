from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


def load_split(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, label")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINERVA Baseline: TF-IDF + Logistic Regression on JCBlaise splits.")
    ap.add_argument("--train_csv", default="data/processed/train.csv")
    ap.add_argument("--val_csv", default="data/processed/val.csv")
    ap.add_argument("--test_csv", default="data/processed/test.csv")
    ap.add_argument(
        "--out_model", default="models/baseline_tfidf_logreg.joblib")
    ap.add_argument(
        "--out_report", default="logs/baseline_tfidf_logreg_report.json")
    args = ap.parse_args()

    train_p = Path(args.train_csv)
    val_p = Path(args.val_csv)
    test_p = Path(args.test_csv)

    X_train, y_train = load_split(train_p)
    X_val, y_val = load_split(val_p)
    X_test, y_test = load_split(test_p)

    # Strong baseline for text classification: TF-IDF + linear classifier
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2
        )),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="saga",
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    report = {
        "model": "tfidf_logreg",
        "val": metrics(y_val, y_val_pred),
        "test": metrics(y_test, y_test_pred),
        "data": {
            "train_csv": str(train_p),
            "val_csv": str(val_p),
            "test_csv": str(test_p),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
        }
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, out_model)

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[baseline] saved:", out_model.resolve())
    print("[baseline] report:", out_report.resolve())
    print("[baseline] val:", report["val"])
    print("[baseline] test:", report["test"])


if __name__ == "__main__":
    main()

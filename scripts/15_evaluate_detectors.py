from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, label")
    texts = df["text"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()
    return texts, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


@torch.no_grad()
def predict_probs(model, tok, texts: list[str], batch_size: int = 16, max_len: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[:, 1]  # P(fake)
        out_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(out_probs)


def eval_transformer(model_dir: Path, texts: list[str], y_true: np.ndarray, batch_size: int, max_len: int) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    probs = predict_probs(
        model, tok, texts, batch_size=batch_size, max_len=max_len)
    y_pred = (probs >= 0.5).astype(int)
    return {
        "model_dir": str(model_dir),
        "metrics": metrics(y_true, y_pred),
        "mean_p_fake": float(np.mean(probs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate trained MINERVA detectors on test set and write a consolidated report.")
    ap.add_argument("--test_csv", default="data/processed/test.csv")
    ap.add_argument("--roberta_dir", default="models/roberta_finetuned")
    ap.add_argument(
        "--distil_dir", default="models/distilbert_multilingual_finetuned")
    ap.add_argument("--out_json", default="logs/eval_detectors_report.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    texts, y = load_split(Path(args.test_csv))

    report: dict[str, Any] = {
        "test_csv": args.test_csv,
        "n_test": int(len(y)),
        "detectors": {}
    }

    ro_dir = Path(args.roberta_dir)
    di_dir = Path(args.distil_dir)

    if ro_dir.exists():
        report["detectors"]["roberta"] = eval_transformer(
            ro_dir, texts, y, args.batch_size, args.max_len)
    else:
        report["detectors"]["roberta"] = {"error": f"missing: {ro_dir}"}

    if di_dir.exists():
        report["detectors"]["distilbert"] = eval_transformer(
            di_dir, texts, y, args.batch_size, args.max_len)
    else:
        report["detectors"]["distilbert"] = {"error": f"missing: {di_dir}"}

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[eval] wrote:", out.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

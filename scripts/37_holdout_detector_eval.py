#!/usr/bin/env python3
"""Holdout detector evaluation. Deferred to external Filipino fact-checker validation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("minerva.holdout_eval")


# Metric computation (no sklearn dependency to keep the script lightweight)

def compute_binary_metrics(y_true: list[int], y_pred: list[int],
                           label_for_positive: int = 1) -> dict:
    """Compute precision/recall/F1/accuracy for a binary classification.

    `y_true` and `y_pred` are 0/1 arrays. Positive class is `label_for_positive`.
    Returns a dict with the standard metrics. Defensively handles the
    edge cases of zero-true-positives or empty arrays.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) "
            f"must have equal length")
    if len(y_true) == 0:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    tp = sum(1 for yt, yp in zip(y_true, y_pred)
             if yt == label_for_positive and yp == label_for_positive)
    fp = sum(1 for yt, yp in zip(y_true, y_pred)
             if yt != label_for_positive and yp == label_for_positive)
    tn = sum(1 for yt, yp in zip(y_true, y_pred)
             if yt != label_for_positive and yp != label_for_positive)
    fn = sum(1 for yt, yp in zip(y_true, y_pred)
             if yt == label_for_positive and yp != label_for_positive)

    n = len(y_true)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "n": n, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# Holdout loader

def load_holdout(path: Path) -> list[dict]:
    """Load (id, text, true_label) from CSV. true_label normalized to 0/1.

    REAL → 0, FAKE → 1. UNCERTAIN cards are reported separately and excluded
    from the binary metrics (per standard fake-news-detection convention).
    """
    if not path.exists():
        raise FileNotFoundError(f"Holdout CSV not found: {path}")

    import csv
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("true_label") or "").strip().lower()
            label_int = {"real": 0, "fake": 1}.get(label, None)
            out.append({
                "id": row["id"].strip(),
                "text": row["text"],
                "true_label_str": label,
                "true_label_int": label_int,
            })
    return out


# Detector loaders + predictors

def predict_roberta(texts: list[str], model_dir: Path) -> list[float]:
    """Returns p_fake per text using the trained RoBERTa-Tagalog detector."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        logger.warning("transformers/torch not available — RoBERTa predictions "
                       "will be skipped. Install via `pip install -r requirements.txt`.")
        return [float("nan")] * len(texts)

    if not model_dir.exists():
        logger.error("RoBERTa model dir not found: %s", model_dir)
        return [float("nan")] * len(texts)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    probs = []
    with torch.no_grad():
        # Batch in groups of 8 to be memory-safe on T4
        for i in range(0, len(texts), 8):
            batch = texts[i:i + 8]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            logits = model(**enc).logits
            p_fake = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            probs.extend(p_fake)
    return probs


def predict_distilbert(texts: list[str], model_dir: Path) -> list[float]:
    """Returns p_fake per text using the trained DistilBERT detector."""
    # Same architecture as RoBERTa — reuse the loader path
    return predict_roberta(texts, model_dir)


# Driver

def evaluate(holdout_path: Path, report_path: Path,
             roberta_dir: Path, distilbert_dir: Path,
             threshold: float = 0.5) -> dict:
    holdout = load_holdout(holdout_path)
    logger.info("Loaded %d holdout cards from %s", len(holdout), holdout_path)

    # Class balance audit
    n_real = sum(1 for h in holdout if h["true_label_int"] == 0)
    n_fake = sum(1 for h in holdout if h["true_label_int"] == 1)
    n_unk = sum(1 for h in holdout if h["true_label_int"] is None)
    logger.info("Class balance: REAL=%d, FAKE=%d, UNCERTAIN=%d",
                n_real, n_fake, n_unk)

    # Filter to binary-labeled rows for metric computation
    binary = [h for h in holdout if h["true_label_int"] is not None]
    texts = [h["text"] for h in binary]
    y_true = [h["true_label_int"] for h in binary]

    # Predict per detector
    p_roberta = predict_roberta(texts, roberta_dir)
    p_distil = predict_distilbert(texts, distilbert_dir)
    p_ensemble = [
        (a + b) / 2 if not (a != a or b != b)  # NaN-safe average
        else (a if not (a != a) else b)
        for a, b in zip(p_roberta, p_distil)
    ]

    def _to_pred(probs, thr):
        return [1 if (p == p and p >= thr) else 0 for p in probs]

    pred_roberta = _to_pred(p_roberta, threshold)
    pred_distil = _to_pred(p_distil, threshold)
    pred_ensemble = _to_pred(p_ensemble, threshold)

    metrics_roberta = compute_binary_metrics(y_true, pred_roberta)
    metrics_distil = compute_binary_metrics(y_true, pred_distil)
    metrics_ens = compute_binary_metrics(y_true, pred_ensemble)

    # Per-card audit
    per_card = []
    for h, pr, pd_, pe in zip(binary, p_roberta, p_distil, p_ensemble):
        per_card.append({
            "id": h["id"],
            "true_label": h["true_label_str"],
            "p_roberta_fake": pr,
            "p_distil_fake": pd_,
            "p_ensemble_fake": pe,
            "verdict_at_threshold": "fake" if pe >= threshold else "real",
            "correct": (pe >= threshold) == (h["true_label_int"] == 1),
        })

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": "v2.9.4",
        "holdout": str(holdout_path),
        "n_total": len(holdout),
        "n_real": n_real,
        "n_fake": n_fake,
        "n_uncertain_excluded": n_unk,
        "threshold": threshold,
        "detector_metrics": {
            "p_roberta_fake": metrics_roberta,
            "p_distil_fake": metrics_distil,
            "p_ensemble_fake": metrics_ens,
        },
        "per_card_predictions": per_card,
        "note": (
            "This is held-out evaluation on hand-labeled GPT-2-generated cards "
            "(NOT JCBlaise test data). It measures detector generalization to "
            "the deployment distribution. Compare against reports/det.json "
            "(which is on JCBlaise test) to assess generalization gap."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    logger.info("Holdout evaluation complete:")
    logger.info("  RoBERTa F1     : %.4f", metrics_roberta["f1"])
    logger.info("  DistilBERT F1  : %.4f", metrics_distil["f1"])
    logger.info("  Ensemble F1    : %.4f", metrics_ens["f1"])
    return report


def main():
    p = argparse.ArgumentParser(
        description="v2.9.0 — Held-out detector evaluation on hand-labeled cards."
    )
    p.add_argument("--holdout", required=True,
                   help="CSV with columns: id, text, true_label")
    p.add_argument("--roberta_dir", default="models/roberta_finetuned")
    p.add_argument("--distilbert_dir", default="models/distilbert_multilingual_finetuned")
    p.add_argument("--report_out", default="reports/holdout_detector_eval.json")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate(
        holdout_path=Path(args.holdout),
        report_path=Path(args.report_out),
        roberta_dir=Path(args.roberta_dir),
        distilbert_dir=Path(args.distilbert_dir),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

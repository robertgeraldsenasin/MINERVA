#!/usr/bin/env python3
"""
M.I.N.E.R.V.A. v2.8 — Script 07: Random Forest classifier.

ARCHITECTURE (per BATB §3.5.2):
  > "DE-GNN aggregates ... features to produce a latent representation
  >  that reflects not only what the post states but also how it is
  >  positioned within a propagation network. THIS REPRESENTATION IS
  >  PASSED TO A RANDOM FOREST CLASSIFIER, which outputs a credibility
  >  label and probability score."

The paper specifies a SEQUENTIAL DE-GNN → RF pipeline. Earlier MINERVA
versions trained RF on PCA features in parallel with DE-GNN, treating
RF as an independent baseline. This was a deviation from the paper.

v2.8 corrects this:
  - RF is trained on PCA features PLUS the DE-GNN per-row confidence
    `p_degnn_fake` from data/features/degnn_preds.csv.
  - DE-GNN output is the dominant feature; PCA features remain to
    preserve the linguistic signal.
  - If DE-GNN preds are absent, RF falls back to PCA-only with a
    warning (so this script remains runnable in isolation for tests).

Output:
  models/random_forest.joblib                  the trained RF
  models/random_forest_feature_importance.csv  per-feature importance
  logs/random_forest_report.txt                test-set classification report
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

MINERVA_VERSION = "v2.8"

FEAT_DIR = Path("data/features")
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = FEAT_DIR / "train_tabular.csv"
VAL = FEAT_DIR / "val_tabular.csv"
TEST = FEAT_DIR / "test_tabular.csv"
DEGNN_PREDS = FEAT_DIR / "degnn_preds.csv"

OUT_MODEL = MODEL_DIR / "random_forest.joblib"
OUT_IMPORT = MODEL_DIR / "random_forest_feature_importance.csv"
OUT_REPORT = LOG_DIR / "random_forest_report.txt"


def _print_banner() -> None:
    print("=" * 60)
    print(f"  M.I.N.E.R.V.A. {MINERVA_VERSION} — Script 07")
    print(f"  Random Forest (sequential DE-GNN → RF per BATB §3.5.2)")
    print("=" * 60)


def _join_degnn_signal(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Add p_degnn_fake column to df from data/features/degnn_preds.csv.

    The DE-GNN preds file is expected to have a 'split' column identifying
    which rows belong to train/val/test, plus 'p_degnn_fake'. Rows are
    matched by position within the split (DE-GNN preserves input order).
    """
    if "p_degnn_fake" in df.columns:
        # Already joined upstream — nothing to do.
        return df
    if not DEGNN_PREDS.exists():
        print(
            f"[07] WARNING: {DEGNN_PREDS} not found.\n"
            f"     Falling back to PCA-only Random Forest (paper requires DE-GNN→RF).\n"
            f"     Run scripts/09_train_degnn.py first for the full architecture."
        )
        return df

    pred_df = pd.read_csv(DEGNN_PREDS)
    if "split" in pred_df.columns:
        sub = pred_df[pred_df["split"] == split].reset_index(drop=True)
    else:
        # No split column — assume preds match the split passed in 1:1
        sub = pred_df.copy().reset_index(drop=True)

    if len(sub) != len(df):
        print(
            f"[07] WARNING: DE-GNN preds for split={split} have {len(sub)} rows "
            f"but {split} features have {len(df)}. Skipping DE-GNN feature."
        )
        return df

    df = df.copy().reset_index(drop=True)
    df["p_degnn_fake"] = sub["p_degnn_fake"].values
    return df


def main() -> None:
    _print_banner()

    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise SystemExit(
                f"\n[FATAL] Missing {p}. Run scripts/06_extract_features.py first."
            )

    train_df = _join_degnn_signal(pd.read_csv(TRAIN), "train")
    val_df = _join_degnn_signal(pd.read_csv(VAL), "val")
    test_df = _join_degnn_signal(pd.read_csv(TEST), "test")

    excluded = {"id", "label", "dataset", "lang", "split"}
    feature_cols = [c for c in train_df.columns if c not in excluded]
    print(f"[07] Feature columns ({len(feature_cols)}): {feature_cols}")

    if "p_degnn_fake" in feature_cols:
        print(f"[07] DE-GNN signal present — running SEQUENTIAL DE-GNN → RF")
    else:
        print(f"[07] DE-GNN signal absent  — running PCA-only RF (degraded)")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluate
    for name, X, y in [("VAL", X_val, y_val), ("TEST", X_test, y_test)]:
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary"
        )
        print(f"[{name}] acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")

    dump(clf, OUT_MODEL)

    # Feature importance — what the panel will want to see
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUT_IMPORT, index=False)
    print(f"\n[07] Top 5 features by importance:")
    print(importance.head().to_string(index=False))

    # Test-set report
    test_preds = clf.predict(X_test)
    report = classification_report(y_test, test_preds, digits=4)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"\n[OK] Saved RF model       -> {OUT_MODEL}")
    print(f"[OK] Saved feat importance -> {OUT_IMPORT}")
    print(f"[OK] Saved report          -> {OUT_REPORT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback

        print("\n[FATAL] Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)

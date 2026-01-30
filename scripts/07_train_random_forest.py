from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from joblib import dump

FEAT_DIR = Path("data/features")
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TRAIN = FEAT_DIR / "train_tabular.csv"
VAL = FEAT_DIR / "val_tabular.csv"
TEST = FEAT_DIR / "test_tabular.csv"

OUT_MODEL = MODEL_DIR / "random_forest.joblib"
OUT_REPORT = LOG_DIR / "random_forest_report.txt"


def main():
    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 06_extract_features.py first.")

    train_df = pd.read_csv(TRAIN)
    val_df = pd.read_csv(VAL)
    test_df = pd.read_csv(TEST)

    feature_cols = [c for c in train_df.columns if c not in {
        "id", "label", "dataset", "lang"}]

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
            y, preds, average="binary")
        print(f"[{name}] acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")

    dump(clf, OUT_MODEL)

    # write report
    test_preds = clf.predict(X_test)
    report = classification_report(y_test, test_preds, digits=4)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"[OK] Saved RF model -> {OUT_MODEL}")
    print(f"[OK] Saved report -> {OUT_REPORT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

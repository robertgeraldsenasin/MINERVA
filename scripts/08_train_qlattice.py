from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

FEAT_DIR = Path("data/features")
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TRAIN = FEAT_DIR / "train_tabular.csv"
VAL = FEAT_DIR / "val_tabular.csv"
TEST = FEAT_DIR / "test_tabular.csv"

OUT_EQ = MODEL_DIR / "qlattice_equation.txt"
OUT_NOTE = LOG_DIR / "qlattice_notes.txt"


def main():
    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 06_extract_features.py first.")

    try:
        from feyn import QLattice
    except Exception as e:
        msg = (
            "Could not import feyn (Qlattice Python API).\n"
            "You can still proceed with RoBERTa/DistilBERT/RF/DE-GNN.\n"
            f"Error: {repr(e)}\n"
            "Try: python -m pip install feyn\n"
        )
        OUT_NOTE.write_text(msg, encoding="utf-8")
        print(msg)
        return

    train_df = pd.read_csv(TRAIN)
    # Keep Qlattice feature set small and stable (no ids/dataset/lang)
    feature_cols = [c for c in train_df.columns if c not in {
        "id", "label", "dataset", "lang"}]

    # Qlattice typically works well when you don't feed thousands of features.
    # Here we already reduced embeddings via PCA, so it is feasible.
    data = train_df[feature_cols + ["label"]].copy()

    ql = QLattice()
    # The exact API varies by feyn version; the pattern below is common.
    # If your installed feyn uses slightly different args, the error message will show it.
    models = ql.auto_run(
        data=data,
        output_name="label",
        n_epochs=20,
    )

    best = models[0]
    eq = str(best.sympify())
    OUT_EQ.write_text(eq, encoding="utf-8")

    print("[OK] Best Qlattice equation saved to:", OUT_EQ)
    print("Equation:")
    print(eq)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

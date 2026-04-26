from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

FEAT_DIR = Path("data/features")
MODEL_DIR = Path("models")
QL_DIR = MODEL_DIR / "qlattice"
LOG_DIR = Path("logs")

TRAIN = FEAT_DIR / "train_tabular.csv"
VAL = FEAT_DIR / "val_tabular.csv"
TEST = FEAT_DIR / "test_tabular.csv"

OUT_EQ = MODEL_DIR / "qlattice_equation.txt"
OUT_JSON_FLAT = MODEL_DIR / "best_qlattice.json"
OUT_JSON_NESTED = QL_DIR / "best_qlattice.json"
OUT_NOTE = LOG_DIR / "qlattice_notes.txt"

MODEL_DIR.mkdir(exist_ok=True)
QL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 06_extract_features.py first.")

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
    val_df = pd.read_csv(VAL)
    test_df = pd.read_csv(TEST)

    feature_cols = [c for c in train_df.columns if c not in {"id", "label", "dataset", "lang", "split"}]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found for Qlattice.")

    data = train_df[feature_cols + ["label"]].copy()
    ql = QLattice()

    models = ql.auto_run(
        data=data,
        output_name="label",
        n_epochs=20,
    )
    if not models:
        raise RuntimeError("Qlattice returned zero models.")

    best = models[0]
    eq = str(best.sympify()).strip()
    if not eq:
        raise RuntimeError("Qlattice returned an empty equation.")

    OUT_EQ.write_text(eq, encoding="utf-8")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "equation": eq,
        "feature_cols": feature_cols,
        "train_file": str(TRAIN),
        "val_file": str(VAL),
        "test_file": str(TEST),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "notes": [
            "The plain-text equation is still written to models/qlattice_equation.txt for backward compatibility.",
            "This JSON file improves reproducibility by storing the equation and the feature schema used to discover it.",
            "Model ranking still follows the installed feyn/QLattice API behavior."
        ],
    }

    write_json(OUT_JSON_FLAT, payload)
    write_json(OUT_JSON_NESTED, payload)

    print("[OK] Qlattice equation saved to:", OUT_EQ)
    print("[OK] Structured metadata saved to:", OUT_JSON_FLAT)
    print("[OK] Structured metadata saved to:", OUT_JSON_NESTED)
    print("Equation:")
    print(eq)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

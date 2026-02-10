from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "corpus.csv"

OUT_TRAIN = PROC_DIR / "train.csv"
OUT_VAL = PROC_DIR / "val.csv"
OUT_TEST = PROC_DIR / "test.csv"

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}. Run 02_prepare_dataset.py first.")

    df = pd.read_csv(IN_PATH)

    # With JCBlaise-only corpus, stratifying by dataset is unnecessary.
    # Stratify by label to preserve class balance across splits.
    strat = df["label"].astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=RANDOM_STATE,
        stratify=strat,
    )

    # Split temp into val and test
    temp_strat = temp_df["label"].astype(int)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_FRAC / (VAL_FRAC + TEST_FRAC)),
        random_state=RANDOM_STATE,
        stratify=temp_strat,
    )

    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    print("Dataset split completed:")
    print(f"Train: {len(train_df)} -> {OUT_TRAIN}")
    print(f"Val:   {len(val_df)} -> {OUT_VAL}")
    print(f"Test:  {len(test_df)} -> {OUT_TEST}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

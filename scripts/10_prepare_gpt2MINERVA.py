from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Input splits from MINERVA detector pipeline
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")

# Output corpus for GPT-2 fine-tuning
OUT_DIR = Path("data/gpt2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple control tokens to condition generation on target label
REAL_TOKEN = "<|label=real|>"
FAKE_TOKEN = "<|label=fake|>"


def row_to_line(text: str, label: int) -> str:
    tok = FAKE_TOKEN if int(label) == 1 else REAL_TOKEN
    text = (text or "").replace("\n", " ").strip()
    return f"{tok} {text}"


def main() -> None:
    for p in [TRAIN_PATH, VAL_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 03_split_dataset.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    train_lines = [row_to_line(t, y) for t, y in zip(
        train_df["text"], train_df["label"])]
    val_lines = [row_to_line(t, y)
                 for t, y in zip(val_df["text"], val_df["label"])]

    (OUT_DIR / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote GPT-2 corpora to {OUT_DIR.resolve()}:")
    print(f" - train.txt ({len(train_lines)} lines)")
    print(f" - val.txt   ({len(val_lines)} lines)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

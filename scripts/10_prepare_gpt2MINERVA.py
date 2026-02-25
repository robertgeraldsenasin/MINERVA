from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow importing repo-root modules when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva_privacy import pseudonymize_texts  # noqa: E402


# Input splits from MINERVA detector pipeline
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")

# Output corpus for GPT-2 fine-tuning
OUT_DIR = Path("data/gpt2")


# Simple control tokens to condition generation on target label
REAL_TOKEN = "<|label=real|>"
FAKE_TOKEN = "<|label=fake|>"


def row_to_line(text: str, label: int) -> str:
    tok = FAKE_TOKEN if int(label) == 1 else REAL_TOKEN
    text = (text or "").replace("\n", " ").strip()
    return f"{tok} {text}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINERVA Script 10: Build GPT-2 fine-tuning corpora (train.txt/val.txt)."
    )
    ap.add_argument("--train_csv", default=str(TRAIN_PATH))
    ap.add_argument("--val_csv", default=str(VAL_PATH))
    ap.add_argument("--out_dir", default=str(OUT_DIR))

    # Safety: pseudonymize exported corpora so the generator is less likely to reproduce real names.
    ap.add_argument(
        "--no_pseudonymize",
        action="store_true",
        help="Disable name pseudonymization (NOT recommended for game exports).",
    )
    ap.add_argument(
        "--placeholder_prefix",
        default="Candidate",
        help="Prefix for pseudonyms, e.g., 'Candidate' -> 'Candidate A'.",
    )

    args = ap.parse_args()

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [train_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 03_split_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_texts = [str(t) if isinstance(
        t, str) else "" for t in train_df["text"]]
    val_texts = [str(t) if isinstance(t, str) else "" for t in val_df["text"]]

    if not args.no_pseudonymize:
        train_texts, _ = pseudonymize_texts(
            train_texts, placeholder_prefix=args.placeholder_prefix)
        val_texts, _ = pseudonymize_texts(
            val_texts, placeholder_prefix=args.placeholder_prefix)

    train_lines = [
        row_to_line(t, y) for t, y in zip(train_texts, train_df["label"])
    ]
    val_lines = [
        row_to_line(t, y) for t, y in zip(val_texts, val_df["label"])
    ]

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote GPT-2 corpora to {out_dir.resolve()}:")
    print(f" - train.txt ({len(train_lines)} lines)")
    print(f" - val.txt   ({len(val_lines)} lines)")
    if not args.no_pseudonymize:
        print("[OK] Pseudonymization: ENABLED (placeholders like 'Candidate A').")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

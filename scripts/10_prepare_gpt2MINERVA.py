from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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

# Optional DE-GNN conditioning token (graph-confidence bin)
GRAPH_HIGH = "<|graph=high|>"
GRAPH_MID = "<|graph=mid|>"
GRAPH_LOW = "<|graph=low|>"
GRAPH_UNK = "<|graph=unk|>"  # used when DE-GNN preds are unavailable


def _graph_token_from_conf(conf: float | None, t_mid: float, t_high: float) -> str:
    if conf is None or not (conf == conf):  # NaN
        return GRAPH_UNK
    if float(conf) >= float(t_high):
        return GRAPH_HIGH
    if float(conf) >= float(t_mid):
        return GRAPH_MID
    return GRAPH_LOW


def row_to_line(text: str, label: int, graph_tok: str | None = None) -> str:
    label_tok = FAKE_TOKEN if int(label) == 1 else REAL_TOKEN
    text = (text or "").replace("\n", " ").strip()
    if graph_tok:
        return f"{label_tok} {graph_tok} {text}"
    return f"{label_tok} {text}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINERVA Script 10: Build GPT-2 fine-tuning corpora (train.txt/val.txt)."
    )
    ap.add_argument("--train_csv", default=str(TRAIN_PATH))
    ap.add_argument("--val_csv", default=str(VAL_PATH))
    ap.add_argument("--out_dir", default=str(OUT_DIR))

    # Optional: include DE-GNN confidence tokens in the corpus.
    ap.add_argument(
        "--degnn_preds",
        default="data/features/degnn_preds.csv",
        help="Path to DE-GNN predictions (from Script 09). If missing, graph tokens default to <|graph=unk|>.",
    )
    ap.add_argument(
        "--no_degnn_tokens",
        action="store_true",
        help="Disable adding <|graph=...|> tokens even if DE-GNN preds exist.",
    )
    ap.add_argument(
        "--graph_bins",
        default="0.60,0.80",
        help="Comma-separated thresholds for label-confidence bins: mid,high (default: 0.60,0.80).",
    )
    ap.add_argument(
        "--min_graph_conf",
        type=float,
        default=None,
        help="Optional: drop rows whose DE-GNN confidence for the *true label* is below this value.",
    )

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

    # -------------------------
    # Optional DE-GNN join
    # -------------------------
    use_degnn = (not args.no_degnn_tokens)
    preds_path = Path(args.degnn_preds)
    if use_degnn and preds_path.exists():
        preds = pd.read_csv(preds_path)
        preds["id"] = preds["id"].astype(str)
        if "p_degnn_fake" not in preds.columns:
            print(
                "[WARN] DE-GNN preds found but missing column 'p_degnn_fake'. "
                "Re-run 09_train_degnn.py (updated) to export probabilities. Falling back to <|graph=unk|>."
            )
            preds = preds[["id"]].copy()
            preds["p_degnn_fake"] = np.nan

        train_df["id"] = train_df["id"].astype(str)
        val_df["id"] = val_df["id"].astype(str)

        train_df = train_df.merge(
            preds[["id", "p_degnn_fake"]], on="id", how="left")
        val_df = val_df.merge(
            preds[["id", "p_degnn_fake"]], on="id", how="left")

        # Parse thresholds
        try:
            t_mid, t_high = [float(x.strip())
                             for x in args.graph_bins.split(",")]
        except Exception:
            raise ValueError("--graph_bins must look like '0.60,0.80'")

        # Confidence of DE-GNN for the *true* label
        def _conf(row) -> float | None:
            p_fake = row.get("p_degnn_fake")
            if p_fake is None or not (p_fake == p_fake):
                return None
            y = int(row["label"])
            return float(p_fake) if y == 1 else float(1.0 - float(p_fake))

        train_df["degnn_conf"] = train_df.apply(_conf, axis=1)
        val_df["degnn_conf"] = val_df.apply(_conf, axis=1)

        train_df["graph_tok"] = train_df["degnn_conf"].apply(
            lambda c: _graph_token_from_conf(c, t_mid, t_high)
        )
        val_df["graph_tok"] = val_df["degnn_conf"].apply(
            lambda c: _graph_token_from_conf(c, t_mid, t_high)
        )

        # Optional filtering for a cleaner GPT corpus
        if args.min_graph_conf is not None:
            before_tr, before_va = len(train_df), len(val_df)
            train_df = train_df[(train_df["degnn_conf"].isna()) | (
                train_df["degnn_conf"] >= args.min_graph_conf)].copy()
            val_df = val_df[(val_df["degnn_conf"].isna()) | (
                val_df["degnn_conf"] >= args.min_graph_conf)].copy()
            print(
                f"[10] Filtered by --min_graph_conf={args.min_graph_conf}: "
                f"train {before_tr}->{len(train_df)} | val {before_va}->{len(val_df)}"
            )
    else:
        # No preds -> use unknown token if tokens are enabled.
        if use_degnn:
            train_df["graph_tok"] = GRAPH_UNK
            val_df["graph_tok"] = GRAPH_UNK

    train_texts = [str(t) if isinstance(
        t, str) else "" for t in train_df["text"]]
    val_texts = [str(t) if isinstance(t, str) else "" for t in val_df["text"]]

    if not args.no_pseudonymize:
        train_texts, _ = pseudonymize_texts(
            train_texts, placeholder_prefix=args.placeholder_prefix)
        val_texts, _ = pseudonymize_texts(
            val_texts, placeholder_prefix=args.placeholder_prefix)

    train_lines = [
        row_to_line(
            t, y, (train_df["graph_tok"].iloc[i] if use_degnn else None))
        for i, (t, y) in enumerate(zip(train_texts, train_df["label"]))
    ]
    val_lines = [
        row_to_line(t, y, (val_df["graph_tok"].iloc[i] if use_degnn else None))
        for i, (t, y) in enumerate(zip(val_texts, val_df["label"]))
    ]

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote GPT-2 corpora to {out_dir.resolve()}:")
    print(f" - train.txt ({len(train_lines)} lines)")
    print(f" - val.txt   ({len(val_lines)} lines)")
    if not args.no_pseudonymize:
        print("[OK] Pseudonymization: ENABLED (placeholders like 'Candidate A').")
    if use_degnn:
        print("[OK] DE-GNN tokens: ENABLED (<|graph=...|>).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

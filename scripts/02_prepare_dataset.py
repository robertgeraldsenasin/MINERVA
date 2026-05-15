from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Unified label convention for the whole project:
# 1 = FAKE
# 0 = REAL
# For Fake News Filipino on Hugging Face, the ClassLabel is commonly: 0=real, 1=fake.
# Keep it configurable in case your exported CSV has labels flipped for any reason.
JCBLAISE_ONE_MEANS_FAKE = True


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_label(series: pd.Series) -> pd.Series:
    """Coerce labels to {0,1} where 1=fake and 0=real."""
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        mapped = s.map(lambda v: 1 if v ==
                       "fake" else 0 if v == "real" else None)
        if mapped.isna().any():
            bad = s[mapped.isna()].value_counts().head(10)
            raise ValueError(f"Unrecognized label values: {bad.to_dict()}")
        y = mapped.astype(int)
    else:
        y = series.astype(int)

    uniq = set(y.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Unexpected numeric labels: {sorted(list(uniq))} (expected only 0/1)")

    if not JCBLAISE_ONE_MEANS_FAKE:
        y = 1 - y
    return y


def load_jcblaise() -> pd.DataFrame:
    """Load JCBlaise CSV — accepts multiple filenames written by different
    script-01 versions (v2.7 wrote 'jcblaise_fake_news_filipino_train.csv';
    v2.8 writes the canonical 'jcblaise.csv')."""
    candidate_paths = [
        RAW_DIR / "jcblaise.csv",                              # v2.8 canonical
        RAW_DIR / "jcblaise_fake_news_filipino_train.csv",     # v2.7 datasets-lib path
        RAW_DIR / "jcblaise_fake_news_filipino.csv",           # v2.5 fallback
    ]
    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No JCBlaise CSV found. Tried: {[str(p) for p in candidate_paths]}.\n"
            f"Run scripts/01_download_dataset.py first."
        )

    df = pd.read_csv(path)
    print(f"[02] Loaded {path.name}: {len(df)} rows, columns={list(df.columns)}")

    # Accept either schema: (article, label) [v2.5] or (text, label) [v2.8 canonical].
    text_col = "article" if "article" in df.columns else "text"
    if text_col not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Unexpected columns in {path.name}: {df.columns.tolist()} "
            f"(expected: text or article + label)")

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str).map(clean_text)
    out["label"] = normalize_label(df["label"])
    out["dataset"] = "jcblaise_fake_news_filipino"
    out["lang"] = "tl"
    return out


def main() -> None:
    print("Loading and normalizing dataset (JCBlaise only)...")

    df = load_jcblaise()

    # Basic cleanup
    df = df[df["text"].astype(str).str.len() > 10].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Stable IDs
    df["id"] = df["dataset"] + "_" + df.index.astype(str)

    out_path = PROC_DIR / "corpus.csv"
    df[["id", "dataset", "lang", "text", "label"]].to_csv(
        out_path, index=False)

    print(f"[OK] Saved normalized corpus -> {out_path} ({len(df)} rows)")
    print("\nCounts by label:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

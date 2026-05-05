#!/usr/bin/env python3
"""
M.I.N.E.R.V.A. v2.8 — Script 01: Download JCBlaise Fake News Filipino dataset.

Three-tier fallback chain:
  Tier 1: datasets.load_dataset()                 (fastest if compatible)
  Tier 2: direct parquet via pandas               (bypasses fsspec issues)
  Tier 3: direct CSV via urllib                   (last resort)

This robustness was added in v2.8 after v2.7 runs hit the
`NotImplementedError: Loading a dataset cached in a LocalFileSystem
is not supported` failure on Colab images with mismatched fsspec/datasets.

Citation:
  Cruz, J. C. B., Tan, J. A., & Cheng, C. K. (2020).
  Localization of Fake News Detection via Multitask Transfer Learning.
  LREC 2020.

Output:
  data/raw/jcblaise_fake_news_filipino_train.csv  (and val/test if upstream provides)
  data/raw/jcblaise.csv                           (canonical alias for downstream scripts)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

MINERVA_VERSION = "v2.8"
JCBLAISE_ID = "jcblaise/fake_news_filipino"
JCBLAISE_PARQUET_URL = (
    "https://huggingface.co/datasets/jcblaise/fake_news_filipino/"
    "resolve/main/data/train-00000-of-00001.parquet"
)
JCBLAISE_CSV_URL = (
    "https://huggingface.co/datasets/jcblaise/fake_news_filipino/"
    "resolve/main/fake_news_filipino.csv"
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _print_banner() -> None:
    print("=" * 60)
    print(f"  M.I.N.E.R.V.A. {MINERVA_VERSION} — Script 01")
    print(f"  Download JCBlaise Fake News Filipino")
    print(f"  Dataset: {JCBLAISE_ID}")
    print(f"  Output:  {RAW_DIR.resolve()}")
    print("=" * 60)


def _save_canonical(df: pd.DataFrame, source: str) -> Path:
    """Write the canonical CSV that downstream scripts (02, 03) expect."""
    out_path = RAW_DIR / "jcblaise.csv"
    # Normalize column names to a canonical schema downstream expects
    if "label" not in df.columns and "labels" in df.columns:
        df = df.rename(columns={"labels": "label"})
    if "text" not in df.columns and "article" in df.columns:
        df = df.rename(columns={"article": "text"})
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'label' in dataset; found {list(df.columns)}"
        )
    df.to_csv(out_path, index=False)
    print(f"[OK] {source} → {out_path} ({len(df)} rows)")
    print(f"     columns: {list(df.columns)}")
    if "label" in df.columns:
        print(f"     label distribution: {dict(df['label'].value_counts())}")
    return out_path


def _try_datasets_library() -> Optional[pd.DataFrame]:
    """Tier 1: HuggingFace `datasets` library. Sensitive to fsspec version."""
    print("\n[Tier 1] Trying datasets.load_dataset()...")
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(JCBLAISE_ID)
        # Multiple splits possible; concat them all (pipeline does its own re-split in script 03)
        if hasattr(ds, "keys"):
            frames = []
            for split in ds.keys():
                frames.append(ds[split].to_pandas())
                print(f"       [+] split '{split}': {len(frames[-1])} rows")
            df = pd.concat(frames, ignore_index=True)
        else:
            df = ds.to_pandas()
        print(f"  [OK] Tier 1 succeeded — {len(df)} total rows")
        return df
    except NotImplementedError as e:
        print(f"  [SKIP] Tier 1 failed: NotImplementedError ({e})")
        print(f"         (fsspec/datasets version mismatch — falling back)")
        return None
    except ImportError as e:
        print(f"  [SKIP] Tier 1 failed: ImportError ({e})")
        return None
    except Exception as e:
        print(f"  [SKIP] Tier 1 failed: {type(e).__name__}: {e}")
        return None


def _try_direct_parquet() -> Optional[pd.DataFrame]:
    """Tier 2: direct parquet read via pandas + pyarrow. Bypasses datasets/fsspec."""
    print("\n[Tier 2] Trying direct parquet via pandas...")
    try:
        df = pd.read_parquet(JCBLAISE_PARQUET_URL)
        print(f"  [OK] Tier 2 succeeded — {len(df)} rows from parquet")
        return df
    except Exception as e:
        print(f"  [SKIP] Tier 2 failed: {type(e).__name__}: {e}")
        return None


def _try_direct_csv() -> Optional[pd.DataFrame]:
    """Tier 3: direct urllib download of CSV. Last resort."""
    print("\n[Tier 3] Trying direct CSV via urllib...")
    try:
        local_csv = RAW_DIR / "_jcblaise_direct.csv"
        urllib.request.urlretrieve(JCBLAISE_CSV_URL, local_csv)
        df = pd.read_csv(local_csv)
        local_csv.unlink()  # cleanup intermediate
        print(f"  [OK] Tier 3 succeeded — {len(df)} rows from direct CSV")
        return df
    except Exception as e:
        print(f"  [SKIP] Tier 3 failed: {type(e).__name__}: {e}")
        return None


def main() -> None:
    _print_banner()

    df: Optional[pd.DataFrame] = None
    sources_tried = []

    # Tier 1
    df = _try_datasets_library()
    sources_tried.append("datasets.load_dataset()")

    # Tier 2
    if df is None:
        df = _try_direct_parquet()
        sources_tried.append("direct parquet")

    # Tier 3
    if df is None:
        df = _try_direct_csv()
        sources_tried.append("direct CSV")

    if df is None or len(df) == 0:
        print("\n" + "=" * 60)
        print("[FATAL] All download tiers failed.")
        print(f"        Tried: {sources_tried}")
        print(f"        Check internet connectivity and try again.")
        print(f"        Manual fallback: download fake_news_filipino.csv from")
        print(f"          https://huggingface.co/datasets/{JCBLAISE_ID}/tree/main")
        print(f"        and place it at data/raw/jcblaise.csv.")
        print("=" * 60)
        sys.exit(1)

    saved = _save_canonical(df, source=sources_tried[-1])
    print(f"\n[OK] Wrote canonical CSV: {saved}")
    print("Done.")


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

#!/usr/bin/env python3
"""
M.I.N.E.R.V.A. v2.8 — Script 01: Download JCBlaise Fake News Filipino dataset.

The dataset (jcblaise/fake_news_filipino) uses a custom Python loading script,
NOT static parquet/CSV files. There's no realistic direct download fallback —
HuggingFace is the only canonical source. So this script focuses on making the
HuggingFace path robust:

  Tier 1: load_dataset(..., trust_remote_code=True)         (canonical)
  Tier 2: same but with disable_caching to bypass fsspec     (fallback)

The earlier v2.8 attempt to fall back to parquet/CSV URLs was based on the wrong
assumption that the dataset was static. It isn't. Per HF's notice on the dataset
card: "The viewer is disabled because this dataset repo requires arbitrary
Python code execution." That's a custom loader, not a parquet table.

Citation:
  Cruz, J. C. B., Tan, J. A., & Cheng, C. K. (2020).
  Localization of Fake News Detection via Multitask Transfer Learning.
  LREC 2020.

Output:
  data/raw/jcblaise.csv  (canonical CSV the rest of the pipeline reads)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

MINERVA_VERSION = "v2.8"
JCBLAISE_ID = "jcblaise/fake_news_filipino"

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
    if "label" not in df.columns and "labels" in df.columns:
        df = df.rename(columns={"labels": "label"})
    if "text" not in df.columns and "article" in df.columns:
        df = df.rename(columns={"article": "text"})
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'label' in dataset; found {list(df.columns)}"
        )
    df.to_csv(out_path, index=False)
    print(f"\n[OK] {source} → {out_path} ({len(df)} rows)")
    print(f"     columns: {list(df.columns)}")
    if "label" in df.columns:
        print(f"     label distribution: {dict(df['label'].value_counts())}")
    return out_path


def _try_tier_1_standard() -> Optional[pd.DataFrame]:
    """Tier 1: standard load_dataset with trust_remote_code=True."""
    print("\n[Tier 1] load_dataset(trust_remote_code=True) ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(JCBLAISE_ID, trust_remote_code=True)
        return _datasetdict_to_df(ds, source="Tier 1 (standard)")
    except NotImplementedError as e:
        print(f"  [SKIP] Tier 1 failed: NotImplementedError ({e})")
        print(f"         (fsspec/datasets caching incompatibility — falling back)")
        return None
    except ImportError as e:
        print(f"  [SKIP] Tier 1 failed: ImportError ({e})")
        return None
    except Exception as e:
        print(f"  [SKIP] Tier 1 failed: {type(e).__name__}: {e}")
        return None


def _try_tier_2_no_cache() -> Optional[pd.DataFrame]:
    """Tier 2: same as Tier 1 but with caching disabled.

    The NotImplementedError from Tier 1 happens during the cache write step.
    Disabling caching bypasses it — at the cost of re-downloading on every
    invocation, but for a 1.32 MB dataset that's fine.
    """
    print("\n[Tier 2] load_dataset with caching disabled ...")
    try:
        from datasets import disable_caching, load_dataset

        disable_caching()
        ds = load_dataset(JCBLAISE_ID, trust_remote_code=True)
        return _datasetdict_to_df(ds, source="Tier 2 (no cache)")
    except Exception as e:
        print(f"  [SKIP] Tier 2 failed: {type(e).__name__}: {e}")
        return None


def _try_tier_3_streaming() -> Optional[pd.DataFrame]:
    """Tier 3: streaming mode — bypasses the cache entirely by streaming rows."""
    print("\n[Tier 3] load_dataset in streaming mode ...")
    try:
        from datasets import load_dataset

        ds_stream = load_dataset(JCBLAISE_ID, trust_remote_code=True, streaming=True)
        rows = []
        for split_name in ds_stream:
            print(f"       [+] streaming split '{split_name}'...")
            for i, row in enumerate(ds_stream[split_name]):
                rows.append(row)
                if i > 0 and i % 500 == 0:
                    print(f"           {i} rows...")
        if not rows:
            return None
        df = pd.DataFrame(rows)
        print(f"  [OK] Tier 3 succeeded — {len(df)} total rows from streaming")
        return df
    except Exception as e:
        print(f"  [SKIP] Tier 3 failed: {type(e).__name__}: {e}")
        return None


def _datasetdict_to_df(ds, source: str) -> pd.DataFrame:
    """Convert a DatasetDict (or single Dataset) to a pandas DataFrame."""
    if hasattr(ds, "keys"):
        frames = []
        for split in ds.keys():
            frames.append(ds[split].to_pandas())
            print(f"       [+] split '{split}': {len(frames[-1])} rows")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = ds.to_pandas()
    print(f"  [OK] {source} succeeded — {len(df)} total rows")
    return df


def main() -> None:
    _print_banner()

    df: Optional[pd.DataFrame] = None
    sources_tried = []

    df = _try_tier_1_standard()
    sources_tried.append("Tier 1 (standard)")

    if df is None:
        df = _try_tier_2_no_cache()
        sources_tried.append("Tier 2 (no cache)")

    if df is None:
        df = _try_tier_3_streaming()
        sources_tried.append("Tier 3 (streaming)")

    if df is None or len(df) == 0:
        print("\n" + "=" * 60)
        print("[FATAL] All download tiers failed.")
        print(f"        Tried: {sources_tried}")
        print()
        print("Possible causes:")
        print("  1. fsspec/datasets version mismatch — check the install cell.")
        print("     Required: fsspec<=2024.6.1, datasets>=2.14,<3.0")
        print("  2. No internet connectivity in this Colab session.")
        print("  3. Dataset was renamed or made private.")
        print()
        print("Manual fallback:")
        print(f"  1. Download from https://huggingface.co/datasets/{JCBLAISE_ID}/tree/main")
        print(f"  2. Place the CSV at data/raw/jcblaise.csv with columns 'text','label'")
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

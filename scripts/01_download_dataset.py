#!/usr/bin/env python3
"""Download the JCBlaise Filipino fake-news dataset from HuggingFace into data/raw/."""

from __future__ import annotations

import csv
import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

MINERVA_VERSION = "v2.8.2"
JCBLAISE_ID = "jcblaise/fake_news_filipino"

# Verified URL from the HF loader script (fake_news_filipino.py).
# HF redirects this to a CAS-bridge / Xet S3 blob; urllib follows the redirect.
JCBLAISE_ZIP_URL = (
    "https://huggingface.co/datasets/jcblaise/fake_news_filipino/"
    "resolve/main/fakenews.zip"
)
# The CSV inside the zip — confirmed via loader script:
#   data_dir = dl_manager.download_and_extract(_URL)
#   train_path = os.path.join(data_dir, "fakenews", "full.csv")
JCBLAISE_INNER_CSV = "fakenews/full.csv"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Banner / canonical-save helpers

def _print_banner() -> None:
    print("=" * 60)
    print(f"  M.I.N.E.R.V.A. {MINERVA_VERSION} — Script 01")
    print(f"  Download JCBlaise Fake News Filipino")
    print(f"  Dataset: {JCBLAISE_ID}")
    print(f"  Output:  {RAW_DIR.resolve()}")
    print("=" * 60)


def _save_canonical(df: pd.DataFrame, source: str) -> Path:
    """Write the canonical CSV that downstream scripts (02, 03) expect.

    Schema normalization:
      - "labels" -> "label"
      - "article" -> "text"  (so script 02 can read either via pick_col)
      - validates required columns exist
    """
    out_path = RAW_DIR / "jcblaise.csv"

    if "label" not in df.columns and "labels" in df.columns:
        df = df.rename(columns={"labels": "label"})
    if "text" not in df.columns and "article" in df.columns:
        df = df.rename(columns={"article": "text"})

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'label' in dataset; "
            f"found {list(df.columns)}"
        )

    # Coerce label to int 0/1. Handle both pandas 2.x (object dtype) and
    # pandas 3.x (str dtype). JCBlaise stores ClassLabel as string "0"/"1".
    if not pd.api.types.is_integer_dtype(df["label"]):
        try:
            df["label"] = df["label"].astype(int)
        except (ValueError, TypeError):
            df["label"] = df["label"].astype(str).str.strip().astype(int)

    df.to_csv(out_path, index=False)

    print(f"\n[OK] {source} -> {out_path} ({len(df)} rows)")
    print(f"     columns: {list(df.columns)}")
    print(f"     label distribution: {dict(df['label'].value_counts().sort_index())}")
    return out_path


# Tier 1 — direct ZIP download, stdlib only (PRIMARY)

def parse_zip_bytes_to_df(zip_bytes: bytes,
                          inner_csv: str = JCBLAISE_INNER_CSV) -> pd.DataFrame:
    """Extract the JCBlaise CSV from in-memory ZIP bytes and parse to DataFrame.

    Uses csv.QUOTE_ALL with skipinitialspace=True to match the original
    loader's behaviour exactly:
        csv.reader(csv_file, quotechar='"', delimiter=",",
                   quoting=csv.QUOTE_ALL, skipinitialspace=True)
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if inner_csv not in names:
            # Defensive: search case-insensitively, allow nested
            cand = [n for n in names if n.lower().endswith("/full.csv")
                    or n.lower() == "full.csv"]
            if not cand:
                raise FileNotFoundError(
                    f"'{inner_csv}' not found in zip. "
                    f"Members: {names[:10]}{'...' if len(names) > 10 else ''}"
                )
            inner_csv = cand[0]

        with zf.open(inner_csv) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.reader(
                text,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"'{inner_csv}' is empty")
            rows = [row for row in reader if row]

    # Normalize header to {label, article} regardless of casing/whitespace
    header_clean = [h.strip().lower() for h in header]
    df = pd.DataFrame(rows, columns=header_clean)

    # Trim whitespace on every string-like column. Use is_string_dtype so
    # this works on pandas 2.x (object dtype) AND pandas 3.x (str dtype).
    # Avoid applymap — removed in pandas 2.2+.
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _try_tier_1_direct_zip() -> Optional[pd.DataFrame]:
    """Tier 1: download fakenews.zip directly via urllib + parse with stdlib."""
    print("\n[Tier 1] Direct ZIP download from HuggingFace (stdlib bypass) ...")
    print(f"         URL: {JCBLAISE_ZIP_URL}")
    try:
        req = urllib.request.Request(
            JCBLAISE_ZIP_URL,
            headers={"User-Agent": f"MINERVA/{MINERVA_VERSION} (urllib)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        print(f"         downloaded {len(content):,} bytes")

        # Sniff: ZIP magic is 'PK\x03\x04' (50 4B 03 04). If it's gzip
        # (1f 8b ...), defer to Tier 3.
        if not content.startswith(b"PK"):
            print(f"         [SKIP] Tier 1: response is not a ZIP "
                  f"(first bytes {content[:4].hex()}). Trying Tier 3.")
            return None

        df = parse_zip_bytes_to_df(content, JCBLAISE_INNER_CSV)
        if len(df) == 0:
            print("  [SKIP] Tier 1 returned 0 rows.")
            return None

        print(f"  [OK] Tier 1 succeeded — {len(df)} rows from "
              f"{JCBLAISE_INNER_CSV}")
        return df

    except urllib.error.HTTPError as e:
        print(f"  [SKIP] Tier 1 HTTPError {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"  [SKIP] Tier 1 URLError: {e.reason}")
        return None
    except (zipfile.BadZipFile, ValueError, FileNotFoundError) as e:
        print(f"  [SKIP] Tier 1 parse failed: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"  [SKIP] Tier 1 failed: {type(e).__name__}: {e}")
        return None


# Tier 2 — datasets library (kept as fallback)

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


def _try_tier_2_datasets_lib() -> Optional[pd.DataFrame]:
    """Tier 2: datasets.load_dataset(..., trust_remote_code=True).

    Kept as a fallback in case Tier 1's URL changes. As of v2.8.1 this
    fails on Colab with a gzip UnicodeDecodeError due to a version
    mismatch between the JCBlaise loader script (older) and modern
    `datasets ≥2.14` (newer). If the dataset publisher updates the
    loader, this tier will work again.
    """
    print("\n[Tier 2] datasets.load_dataset(trust_remote_code=True) ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(JCBLAISE_ID, trust_remote_code=True)
        return _datasetdict_to_df(ds, source="Tier 2 (datasets lib)")
    except ImportError as e:
        print(f"  [SKIP] Tier 2: datasets not installed ({e})")
        return None
    except Exception as e:
        print(f"  [SKIP] Tier 2 failed: {type(e).__name__}: {e}")
        return None


# Tier 3 — generic compressed-blob handler (last resort)

def _try_tier_3_gzip_or_unknown() -> Optional[pd.DataFrame]:
    """Tier 3: re-fetch the URL and try to decompress as gzip / tar.gz.

    Only relevant if HF starts serving a different format at the same URL.
    """
    print("\n[Tier 3] Generic decompression of fetched blob ...")
    try:
        req = urllib.request.Request(
            JCBLAISE_ZIP_URL,
            headers={"User-Agent": f"MINERVA/{MINERVA_VERSION} (urllib)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = resp.read()

        magic2 = blob[:2]
        magic4 = blob[:4]

        # gzip magic
        if magic2 == b"\x1f\x8b":
            print("         detected gzip; decompressing")
            inner = gzip.decompress(blob)
            text = inner.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
            print(f"  [OK] Tier 3 (gzip) succeeded — {len(df)} rows")
            return df

        # zip magic (handled by Tier 1 normally; included for completeness)
        if magic4[:2] == b"PK":
            print("         detected zip; parsing")
            return parse_zip_bytes_to_df(blob, JCBLAISE_INNER_CSV)

        print(f"  [SKIP] Tier 3: unrecognized magic bytes {magic4.hex()}")
        return None

    except Exception as e:
        print(f"  [SKIP] Tier 3 failed: {type(e).__name__}: {e}")
        return None


# Main

def main() -> None:
    _print_banner()

    df: Optional[pd.DataFrame] = None
    sources_tried = []

    # Tier 1 — direct ZIP (primary)
    df = _try_tier_1_direct_zip()
    sources_tried.append("Tier 1 (direct ZIP)")

    if df is None:
        df = _try_tier_2_datasets_lib()
        sources_tried.append("Tier 2 (datasets lib)")

    if df is None:
        df = _try_tier_3_gzip_or_unknown()
        sources_tried.append("Tier 3 (generic decompress)")

    if df is None or len(df) == 0:
        print("\n" + "=" * 60)
        print("[FATAL] All download tiers failed.")
        print(f"        Tried: {sources_tried}")
        print()
        print("Possible causes:")
        print("  1. No internet connectivity in this Colab session.")
        print("     -> Check by running: !curl -I https://huggingface.co")
        print("  2. HuggingFace is rate-limiting or temporarily unreachable.")
        print("     -> Wait a minute and retry.")
        print("  3. The dataset URL has changed.")
        print(f"     -> Verify: {JCBLAISE_ZIP_URL}")
        print("  4. (Tier 2 only) fsspec/datasets version mismatch.")
        print("     -> Required: fsspec<=2024.6.1, datasets>=2.14,<3.0")
        print()
        print("Manual fallback:")
        print(f"  1. In a browser, download {JCBLAISE_ZIP_URL}")
        print("  2. Extract; copy 'fakenews/full.csv' to data/raw/jcblaise.csv")
        print("  3. Rename columns: 'article'->'text' (script 02 also accepts 'article')")
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
    except Exception:
        import traceback

        print("\n[FATAL] Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)

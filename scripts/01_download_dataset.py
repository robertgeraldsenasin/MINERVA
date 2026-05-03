from __future__ import annotations

from pathlib import Path
import sys
from datasets import load_dataset

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Philippine-context dataset (Cruz et al., 2020): Fake News Filipino
JCBLAISE_ID = "jcblaise/fake_news_filipino"


def save_datasetdict_as_csv(ds, out_prefix: str) -> None:
    """Save a Hugging Face Dataset/DatasetDict to CSV(s) in data/raw/.

    We keep this as CSV to make the rest of the pipeline (pandas-based) deterministic
    and easy to inspect/debug.
    """
    if hasattr(ds, "keys"):
        for split in ds.keys():
            df = ds[split].to_pandas()
            out_path = RAW_DIR / f"{out_prefix}_{split}.csv"
            df.to_csv(out_path, index=False)
            print(
                f"[OK] Saved {out_prefix}:{split} -> {out_path} ({len(df)} rows)")
    else:
        df = ds.to_pandas()
        out_path = RAW_DIR / f"{out_prefix}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_prefix} -> {out_path} ({len(df)} rows)")


def main() -> None:
    print(f"Saving dataset into: {RAW_DIR.resolve()}")

    # JCBlaise Fake News Filipino (PH-local, 2020)
    print(f"\n=== Downloading JCBlaise dataset: {JCBLAISE_ID} ===")
    ds_jc = load_dataset(JCBLAISE_ID)
    save_datasetdict_as_csv(ds_jc, "jcblaise_fake_news_filipino")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

from __future__ import annotations

from pathlib import Path
import sys
import zipfile
import io
import requests
import pandas as pd
from datasets import load_dataset

RAW_DIR = Path("data/raw")
SEACROWD_ZIP_DIR = RAW_DIR / "seacrowd_ph_fake_news_corpus_zip"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SEACROWD_ZIP_DIR.mkdir(parents=True, exist_ok=True)

JCBLAISE_ID = "jcblaise/fake_news_filipino"
WELFAKE_ID = "davanstrien/WELFake"
SEACROWD_HF_ID = "SEACrowd/ph_fake_news_corpus"

# Fallback ZIP URL(s) for the Philippine Fake News Corpus (the dataset SEACrowd wraps)
SEACROWD_ZIP_URLS = [
    # try main then master
    "https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus/raw/main/Philippine%20Fake%20News%20Corpus.zip",
    "https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus/raw/master/Philippine%20Fake%20News%20Corpus.zip",
]


def save_datasetdict_as_csv(ds, out_prefix: str) -> None:
    # ds can be DatasetDict or Dataset
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


def try_download_seacrowd_from_hf() -> bool:
    print(f"\n=== Trying SEACrowd via Hugging Face: {SEACROWD_HF_ID} ===")
    try:
        # trust_remote_code exists in newer datasets; handle both
        try:
            ds = load_dataset(SEACROWD_HF_ID, trust_remote_code=True)
        except TypeError:
            ds = load_dataset(SEACROWD_HF_ID)
        save_datasetdict_as_csv(ds, "seacrowd_ph_fake_news_corpus")
        return True
    except Exception as e:
        print("[WARN] Could not load SEACrowd from HF. Reason:")
        print(" ", repr(e))
        return False


def download_seacrowd_zip_fallback() -> None:
    print("\n=== Fallback: Downloading Philippine Fake News Corpus ZIP from GitHub ===")
    last_err = None
    content = None

    for url in SEACROWD_ZIP_URLS:
        try:
            print(f"Trying: {url}")
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and r.content:
                content = r.content
                print("[OK] Downloaded ZIP.")
                break
            else:
                print(f"[WARN] HTTP {r.status_code}")
        except Exception as e:
            last_err = e
            print("[WARN]", repr(e))

    if content is None:
        raise RuntimeError(
            f"Failed to download SEACrowd ZIP from all URLs. Last error: {last_err}")

    # Extract ZIP
    z = zipfile.ZipFile(io.BytesIO(content))
    z.extractall(SEACROWD_ZIP_DIR)
    print(f"[OK] Extracted ZIP -> {SEACROWD_ZIP_DIR}")

    # Just list extracted files so the next script can parse them reliably
    extracted = list(SEACROWD_ZIP_DIR.rglob("*"))
    print(f"[INFO] Extracted files: {len(extracted)}")
    for p in extracted[:25]:
        print(" -", p)


def main() -> None:
    print(f"Saving all datasets into: {RAW_DIR.resolve()}")

    # 1) JCBlaise Fake News Filipino
    print(f"\n=== Downloading JCBlaise dataset: {JCBLAISE_ID} ===")
    # pinned datasets version supports script datasets
    ds_jc = load_dataset(JCBLAISE_ID)
    save_datasetdict_as_csv(ds_jc, "jcblaise_fake_news_filipino")

    # 2) WELFake
    print(f"\n=== Downloading WELFake dataset: {WELFAKE_ID} ===")
    ds_wel = load_dataset(WELFAKE_ID)
    save_datasetdict_as_csv(ds_wel, "welfake")

    # 3) SEACrowd / PH Fake News Corpus
    ok = try_download_seacrowd_from_hf()
    if not ok:
        download_seacrowd_zip_fallback()

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

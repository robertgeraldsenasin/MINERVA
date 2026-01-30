from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Unified label convention for the whole project:
# 1 = FAKE / NOT-CREDIBLE
# 0 = REAL / CREDIBLE

# JCBlaise mapping is not explicitly documented in the dataset card;
# keep it configurable so you can flip it if needed later.
JCBLAISE_ONE_MEANS_FAKE = True

# WELFake is documented as: 0=fake, 1=real -> we invert to our convention
WELFAKE_NEEDS_INVERT = True


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_jcblaise() -> pd.DataFrame:
    path = RAW_DIR / "jcblaise_fake_news_filipino_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 01_download_datasets.py first.")
    df = pd.read_csv(path)
    # Expected columns based on your earlier check: label, article
    if "article" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Unexpected JCBlaise columns: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["text"] = df["article"].astype(str).map(clean_text)
    out["label"] = df["label"].astype(int)

    if not JCBLAISE_ONE_MEANS_FAKE:
        out["label"] = 1 - out["label"]

    out["dataset"] = "jcblaise_fake_news_filipino"
    out["lang"] = "tl"
    return out


def load_welfake() -> pd.DataFrame:
    path = RAW_DIR / "welfake_train.csv"
    if not path.exists():
        # sometimes dataset may create different split names; try any welfake_*.csv
        candidates = sorted(RAW_DIR.glob("welfake_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "No welfake_*.csv found. Run 01_download_datasets.py first.")
        path = candidates[0]

    df = pd.read_csv(path)

    # Common column patterns in WELFake: title, text, label (plus ids)
    if "label" not in df.columns:
        raise ValueError(
            f"WELFake missing 'label'. Columns: {df.columns.tolist()}")

    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else None
    if text_col is None:
        # fallback guesses
        for c in df.columns:
            if c.lower() in {"content", "article", "body"}:
                text_col = c
                break
    if text_col is None:
        raise ValueError(
            f"Could not find text column in WELFake. Columns: {df.columns.tolist()}")

    combined_text = df[text_col].astype(str)
    if title_col is not None:
        combined_text = df[title_col].astype(str) + "\n\n" + combined_text

    out = pd.DataFrame()
    out["text"] = combined_text.map(clean_text)
    out["label"] = df["label"].astype(int)

    if WELFAKE_NEEDS_INVERT:
        out["label"] = 1 - out["label"]

    out["dataset"] = "welfake"
    out["lang"] = "en"
    return out


def load_seacrowd() -> pd.DataFrame:
    # Option A: downloaded from HF in CSV form
    hf_path = RAW_DIR / "seacrowd_ph_fake_news_corpus_train.csv"
    if hf_path.exists():
        df = pd.read_csv(hf_path)
        # Try to guess columns
        cols = {c.lower(): c for c in df.columns}

        # label column candidates
        label_col = None
        for key in ["label", "credibility", "class", "category"]:
            if key in cols:
                label_col = cols[key]
                break

        # text column candidates
        text_cols = []
        for key in ["text", "content", "article", "body"]:
            if key in cols:
                text_cols.append(cols[key])
        if "headline" in cols:
            text_cols.insert(0, cols["headline"])
        if "title" in cols:
            text_cols.insert(0, cols["title"])

        if not text_cols:
            raise ValueError(
                f"SEACrowd HF CSV: could not find text columns. Columns: {df.columns.tolist()}")

        # combine text
        combined = df[text_cols[0]].astype(str)
        for c in text_cols[1:]:
            combined = combined + "\n\n" + df[c].astype(str)

        out = pd.DataFrame()
        out["text"] = combined.map(clean_text)

        if label_col is None:
            raise ValueError(
                f"SEACrowd HF CSV: could not find label column. Columns: {df.columns.tolist()}")

        # If labels are strings like Credible / Not Credible, map them.
        if df[label_col].dtype == object:
            s = df[label_col].astype(str).str.lower()
            out["label"] = s.map(
                lambda v: 1 if "not" in v and "credible" in v else 0 if "credible" in v else None)
        else:
            out["label"] = df[label_col].astype(int)

        out["dataset"] = "seacrowd_ph_fake_news_corpus"
        out["lang"] = "en"
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(int)
        return out

    # Option B: fallback ZIP extracted
    zip_dir = RAW_DIR / "seacrowd_ph_fake_news_corpus_zip"
    if not zip_dir.exists():
        raise FileNotFoundError(
            "SEACrowd not found. Run 01_download_datasets.py first.")

    files = list(zip_dir.rglob("*.csv")) + list(zip_dir.rglob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX found under {zip_dir}")

    frames = []
    for f in files:
        try:
            if f.suffix.lower() == ".csv":
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception:
            continue

        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}

        # detect label via filename OR column
        fname = f.name.lower()
        file_label = None
        if "not credible" in fname or "not_credible" in fname or "fake" in fname:
            file_label = 1
        if "credible" in fname or "real" in fname:
            # if both match, keep 'not credible' priority
            if file_label is None:
                file_label = 0

        label_col = None
        for key in ["label", "credibility", "class", "category"]:
            if key in cols:
                label_col = cols[key]
                break

        # choose text columns
        title_col = cols.get("headline") or cols.get("title")
        body_col = cols.get("content") or cols.get(
            "text") or cols.get("article") or cols.get("body")

        if body_col is None and title_col is None:
            continue

        text = df[body_col].astype(str) if body_col is not None else ""
        if title_col is not None and body_col is not None:
            text = df[title_col].astype(
                str) + "\n\n" + df[body_col].astype(str)
        elif title_col is not None:
            text = df[title_col].astype(str)

        out = pd.DataFrame()
        out["text"] = text.map(clean_text)

        if label_col is not None:
            if df[label_col].dtype == object:
                s = df[label_col].astype(str).str.lower()
                out["label"] = s.map(
                    lambda v: 1 if "not" in v and "credible" in v else 0 if "credible" in v else None)
            else:
                out["label"] = df[label_col].astype(int)
        else:
            if file_label is None:
                # if we cannot deduce label, skip this file
                continue
            out["label"] = file_label

        out["dataset"] = "seacrowd_ph_fake_news_corpus"
        out["lang"] = "en"
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(int)
        frames.append(out)

    if not frames:
        raise RuntimeError(
            "Could not parse any usable SEACrowd/PH corpus files from ZIP fallback.")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("Loading and normalizing datasets...")

    jc = load_jcblaise()
    wel = load_welfake()
    sea = load_seacrowd()

    # Optional: reduce WELFake if you want faster training
    # (You can set to None to keep everything.)
    MAX_WEL_PER_CLASS = 10000
    if MAX_WEL_PER_CLASS is not None:
        wel0 = wel[wel["label"] == 0].sample(
            n=min(MAX_WEL_PER_CLASS, (wel["label"] == 0).sum()), random_state=42)
        wel1 = wel[wel["label"] == 1].sample(
            n=min(MAX_WEL_PER_CLASS, (wel["label"] == 1).sum()), random_state=42)
        wel = pd.concat([wel0, wel1], ignore_index=True)
        print(
            f"[INFO] Downsampled WELFake to {len(wel)} rows (balanced by class).")

    df = pd.concat([jc, wel, sea], ignore_index=True)

    # basic cleanup
    df = df[df["text"].astype(str).str.len() > 10]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # stable IDs
    df["id"] = df["dataset"] + "_" + df.index.astype(str)

    out_path = PROC_DIR / "corpus.csv"
    df[["id", "dataset", "lang", "text", "label"]].to_csv(
        out_path, index=False)

    print(f"[OK] Saved normalized corpus -> {out_path} ({len(df)} rows)")
    print("\nCounts by dataset/label:")
    print(df.groupby(["dataset", "label"]).size())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

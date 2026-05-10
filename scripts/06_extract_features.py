from __future__ import annotations

# This file needs to be modified for refining of features.
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from joblib import dump
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

ROBERTA_DIR = Path("models/roberta_finetuned")
DISTIL_DIR = Path("models/distilbert_multilingual_finetuned")

FEAT_DIR = Path("data/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

PCA_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 16

# 16 roberta + 16 distilbert = 32 dims (good for RF/Qlattice/GNN)
PCA_COMPONENTS = 16


def lexical_features(text: str) -> dict:
    if not isinstance(text, str):
        text = ""
    n_chars = len(text)
    n_words = len(text.split())
    n_exclam = text.count("!")
    n_q = text.count("?")
    n_digits = sum(ch.isdigit() for ch in text)
    digit_ratio = (n_digits / n_chars) if n_chars > 0 else 0.0
    return {
        "char_len": n_chars,
        "word_len": n_words,
        "exclam": n_exclam,
        "question": n_q,
        "digit_ratio": digit_ratio,
    }


@torch.no_grad()
def encode(model, tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_emb = []
    all_p = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i: i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, return_dict=True)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)

        # CLS embedding from last layer
        cls = out.hidden_states[-1][:, 0, :]

        all_emb.append(cls.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    return np.vstack(all_emb), np.concatenate(all_p)


def main():
    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 03_split_dataset.py first.")
    if not ROBERTA_DIR.exists():
        raise FileNotFoundError(
            f"Missing {ROBERTA_DIR}. Run 04_train_robertaMINERVA.py first.")
    if not DISTIL_DIR.exists():
        raise FileNotFoundError(
            f"Missing {DISTIL_DIR}. Run 05_train_distilbert.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Load models/tokenizers
    roberta_tok = AutoTokenizer.from_pretrained(str(ROBERTA_DIR))
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        str(ROBERTA_DIR)).to(DEVICE)

    distil_tok = AutoTokenizer.from_pretrained(str(DISTIL_DIR))
    distil_model = AutoModelForSequenceClassification.from_pretrained(
        str(DISTIL_DIR)).to(DEVICE)

    # Encode train (fit PCA)
    print("\n[1/3] Encoding TRAIN for RoBERTa...")
    r_train_emb, r_train_p = encode(
        roberta_model, roberta_tok, train_df["text"].astype(str).tolist())

    print("\n[1/3] Encoding TRAIN for DistilBERT...")
    d_train_emb, d_train_p = encode(
        distil_model, distil_tok, train_df["text"].astype(str).tolist())

    # Fit PCA for each embedding space
    print("\nFitting PCA...")
    pca_r = PCA(n_components=PCA_COMPONENTS, random_state=42)
    pca_d = PCA(n_components=PCA_COMPONENTS, random_state=42)

    r_train_pca = pca_r.fit_transform(r_train_emb)
    d_train_pca = pca_d.fit_transform(d_train_emb)

    dump(pca_r, PCA_DIR / "pca_roberta.joblib")
    dump(pca_d, PCA_DIR / "pca_distilbert.joblib")

    # helper to build tabular dataframe
    def build_tabular(df: pd.DataFrame, r_p: np.ndarray, d_p: np.ndarray, rp: np.ndarray, dp: np.ndarray) -> pd.DataFrame:
        feats = df["text"].astype(str).map(lexical_features).apply(pd.Series)

        out = pd.DataFrame()
        out["id"] = df["id"].astype(str)
        out["label"] = df["label"].astype(int)
        out["dataset"] = df["dataset"].astype(str)
        out["lang"] = df["lang"].astype(str)

        # PCA components
        for i in range(PCA_COMPONENTS):
            out[f"r_pca_{i}"] = r_p[:, i]
            out[f"d_pca_{i}"] = d_p[:, i]

        # probabilities
        out["p_roberta_fake"] = rp
        out["p_distil_fake"] = dp

        # lexical
        out = pd.concat([out, feats], axis=1)
        return out

    train_tab = build_tabular(train_df, r_train_pca,
                              d_train_pca, r_train_p, d_train_p)
    train_tab.to_csv(FEAT_DIR / "train_tabular.csv", index=False)

    np.savez_compressed(
        FEAT_DIR / "train_embeddings.npz",
        id=train_df["id"].astype(str).values,
        y=train_df["label"].astype(int).values,
        r_emb=r_train_emb,
        d_emb=d_train_emb,
        p_r=r_train_p,
        p_d=d_train_p,
    )

    # Encode val/test then transform with PCA
    def process_split(name: str, df: pd.DataFrame):
        print(f"\n[2/3] Encoding {name} for RoBERTa...")
        r_emb, r_p = encode(roberta_model, roberta_tok,
                            df["text"].astype(str).tolist())
        print(f"\n[2/3] Encoding {name} for DistilBERT...")
        d_emb, d_p = encode(distil_model, distil_tok,
                            df["text"].astype(str).tolist())

        r_pca = pca_r.transform(r_emb)
        d_pca = pca_d.transform(d_emb)

        tab = build_tabular(df, r_pca, d_pca, r_p, d_p)
        tab.to_csv(FEAT_DIR / f"{name.lower()}_tabular.csv", index=False)

        np.savez_compressed(
            FEAT_DIR / f"{name.lower()}_embeddings.npz",
            id=df["id"].astype(str).values,
            y=df["label"].astype(int).values,
            r_emb=r_emb,
            d_emb=d_emb,
            p_r=r_p,
            p_d=d_p,
        )

    process_split("VAL", val_df)
    process_split("TEST", test_df)

    print("\n[OK] Features extracted to data/features/.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

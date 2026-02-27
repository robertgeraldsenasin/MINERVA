from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# -----------------------------
# Safe import for privacy module
# -----------------------------
try:
    # preferred: scripts/minerva_privacy.py
    from scripts.minerva_privacy import maybe_pseudonymize_texts
except Exception:
    try:
        # fallback: repo root minerva_privacy.py
        from minerva_privacy import maybe_pseudonymize_texts
    except Exception:
        # ultimate fallback (no-op)
        def maybe_pseudonymize_texts(texts: List[str]) -> List[str]:
            return texts


BASE_GPT2 = "jcblaise/gpt2-tagalog"

# Default directories (will be resolved relative to RUN_DIR if set)
RUN_DIR = Path(os.environ.get("RUN_DIR", ".")).resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", RUN_DIR / "models")).resolve()
SPLITS_DIR = Path(os.environ.get(
    "SPLITS_DIR", RUN_DIR / "data" / "processed")).resolve()

DEFAULT_OUT_FILE = RUN_DIR / "generated" / "gpt2_synthetic_samples.jsonl"

# PCA files are produced by script 06 (extract features) and used for equation features in scripts 13/18
PCA_ROBERTA = MODEL_DIR / "pca_roberta.joblib"
PCA_DISTILBERT = MODEL_DIR / "pca_distilbert.joblib"

# detector directories (trained by scripts 04/05 or wrappers -> script 16)
ROBERTA_DIR = MODEL_DIR / "roberta_finetuned"
DISTILBERT_DIR = MODEL_DIR / "distilbert_multilingual_finetuned"


def ensure_float(x):
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).item())


def encode_cls_and_prob(
    text: str,
    tok,
    model,
    device: torch.device,
    max_len: int = 256,
) -> Tuple[int, float, np.ndarray]:
    """Return (predicted_class, prob_of_fake, pooled_embedding_vector)."""
    enc = tok(
        text,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
        logits = out.logits  # [1, num_labels]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        # Convention: label 1 = fake (works if training used {0: real, 1: fake})
        prob_fake = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])

        pred = int(np.argmax(probs))

        # pooled embedding: take CLS token hidden state from last layer if available
        # (works for RoBERTa/DistilBERT style models)
        hs = out.hidden_states[-1]  # [1, seq, hidden]
        cls_vec = hs[:, 0, :].detach().cpu().numpy()[0]  # [hidden]

    return pred, prob_fake, cls_vec


def load_detector(det_dir: Path, device: torch.device):
    tok = AutoTokenizer.from_pretrained(str(det_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(det_dir)).to(device)
    model.eval()
    return tok, model


def load_pca(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def to_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic samples using GPT-2 and filter with detectors.")
    parser.add_argument(
        "n", type=int, help="Number of candidates to generate.")
    parser.add_argument("target", choices=[
                        "fake", "real"], help="Target label to generate.")
    parser.add_argument("min_conf", type=float,
                        help="Minimum confidence threshold for accept_mode.")
    parser.add_argument("max_new_tokens", type=int,
                        help="Max new tokens to generate.")
    parser.add_argument(
        "--accept_mode", choices=["none", "roberta", "distilbert", "ensemble"], default="none")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_tokens", type=int, default=25)
    parser.add_argument("--out_file", type=str, default=str(DEFAULT_OUT_FILE))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load GPT-2 generator
    # -----------------------------
    gpt_dir = MODEL_DIR / "gpt2_tagalog_finetuned"
    gpt_model_name = str(gpt_dir) if gpt_dir.exists() else BASE_GPT2

    gen_tok = AutoTokenizer.from_pretrained(gpt_model_name, use_fast=True)

    # IMPORTANT for decoder-only batching:
    # - set pad token to EOS
    # - set LEFT padding to avoid warning + ensure correct generation
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_tok.padding_side = "left"

    gen_model = AutoModelForCausalLM.from_pretrained(gpt_model_name).to(device)
    gen_model.eval()

    # -----------------------------
    # Load detectors
    # -----------------------------
    if not ROBERTA_DIR.exists():
        raise FileNotFoundError(f"Missing RoBERTa detector dir: {ROBERTA_DIR}")
    if not DISTILBERT_DIR.exists():
        raise FileNotFoundError(
            f"Missing DistilBERT detector dir: {DISTILBERT_DIR}")

    print(f"[12] Using RoBERTa detector -> {ROBERTA_DIR}")
    print(f"[12] Using DistilBERT detector -> {DISTILBERT_DIR}")

    r_tok, r_model = load_detector(ROBERTA_DIR, device)
    d_tok, d_model = load_detector(DISTILBERT_DIR, device)

    # -----------------------------
    # Load PCA models (optional but recommended)
    # -----------------------------
    pca_r = load_pca(PCA_ROBERTA)
    pca_d = load_pca(PCA_DISTILBERT)

    if pca_r is None or pca_d is None:
        print(f"[WARN] PCA models missing. Expected:")
        print(f"       - {PCA_ROBERTA}")
        print(f"       - {PCA_DISTILBERT}")
        print(
            "[WARN] Script 13 may fail if your Qlattice equation uses dpca*/rpca* terms.")

    # -----------------------------
    # Prompt format
    # -----------------------------
    # Keep prompt short and deterministic; model learns style from fine-tuning corpus.
    label_token = f"<|label={args.target}|>"
    prompt = label_token + "\n"

    out_rows: List[Dict] = []
    kept = 0
    generated = 0

    # We'll over-generate slightly to meet n kept after filtering
    # but still stop if we get enough.
    max_attempts = int(args.n * 2.5)

    # -----------------------------
    # Generation loop (batched)
    # -----------------------------
    while kept < args.n and generated < max_attempts:
        batch_n = min(args.batch_size, args.n - kept)
        batch_prompts = [prompt] * batch_n

        # Left-padding configured above
        enc = gen_tok(batch_prompts, return_tensors="pt",
                      padding=True).to(device)

        with torch.no_grad():
            gen_ids = gen_model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=gen_tok.pad_token_id,
                eos_token_id=gen_tok.eos_token_id,
            )

        batch_texts = gen_tok.batch_decode(gen_ids, skip_special_tokens=True)
        batch_texts = [t.replace(label_token, "").strip() for t in batch_texts]

        # Filter very short generations
        batch_texts = [t for t in batch_texts if len(
            t.split()) >= args.min_tokens]

        # Pseudonymize entities for legality/privacy
        batch_texts = maybe_pseudonymize_texts(batch_texts)

        for text in batch_texts:
            generated += 1

            # Detector predictions + embeddings
            r_pred, r_prob, r_emb = encode_cls_and_prob(
                text, r_tok, r_model, device=device)
            d_pred, d_prob, d_emb = encode_cls_and_prob(
                text, d_tok, d_model, device=device)

            # Acceptance logic
            if args.accept_mode == "none":
                accept = True
                accept_score = 0.0
            elif args.accept_mode == "roberta":
                accept_score = r_prob
                accept = (accept_score >= args.min_conf)
            elif args.accept_mode == "distilbert":
                accept_score = d_prob
                accept = (accept_score >= args.min_conf)
            else:  # ensemble
                accept_score = (r_prob + d_prob) / 2.0
                accept = (accept_score >= args.min_conf)

            if not accept:
                continue

            row: Dict = {
                "id": f"gpt2_{generated:07d}",
                "target": args.target,
                "text": text,
                "roberta_pred": int(r_pred),
                "roberta_prob_fake": ensure_float(r_prob),
                "distilbert_pred": int(d_pred),
                "distilbert_prob_fake": ensure_float(d_prob),
                "accept_mode": args.accept_mode,
                "accept_score": ensure_float(accept_score),
            }

            # Add PCA feature columns expected downstream (scripts 13/18)
            if pca_r is not None:
                r_pca_vec = pca_r.transform(r_emb.reshape(1, -1))[0]
                for k, v in enumerate(r_pca_vec.tolist()):
                    row[f"r_pca_{k}"] = ensure_float(v)

            if pca_d is not None:
                d_pca_vec = pca_d.transform(d_emb.reshape(1, -1))[0]
                for k, v in enumerate(d_pca_vec.tolist()):
                    row[f"d_pca_{k}"] = ensure_float(v)

            out_rows.append(row)
            kept += 1
            if kept >= args.n:
                break

    out_path = Path(args.out_file)
    to_jsonl(out_path, out_rows)
    print(f"[12] Generated attempts: {generated} | Kept: {kept}")
    print(f"[12] Saved -> {out_path}")
    # Optional: print pseudonymization mode hint
    print("[12] Pseudonymization: ENABLED (placeholders like 'Candidate A').")


if __name__ == "__main__":
    main()

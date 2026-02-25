from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

# Allow importing repo-root modules when running
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva_privacy import pseudonymize_texts  # noqa: E402


REAL_TOKEN = "<|label=real|>"
FAKE_TOKEN = "<|label=fake|>"


def lexical_features(text: str) -> dict:
    """Lexical/cheap features used in MINERVA tabular layer."""
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
        "digit_ratio": float(digit_ratio),
    }


def strip_control_token(s: str) -> str:
    s = (s or "").strip()
    if s.startswith(FAKE_TOKEN):
        return s[len(FAKE_TOKEN):].strip()
    if s.startswith(REAL_TOKEN):
        return s[len(REAL_TOKEN):].strip()
    return s


@torch.no_grad()
def encode_cls_and_prob(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 16,
    max_len: int = 256,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CLS embeddings and P(fake) per text."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_emb = []
    all_p = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Detector encoding"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, return_dict=True)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)
        cls = out.hidden_states[-1][:, 0, :]  # CLS embedding

        all_emb.append(cls.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    return np.vstack(all_emb), np.concatenate(all_p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MINERVA Script 12: Generate GPT-2 samples and compute detector/tabular features "
            "for Qlattice scoring."
        )
    )

    # Backward-compatible positional args:
    ap.add_argument("n", type=int, nargs="?", default=200)
    ap.add_argument("target", choices=[
                    "fake", "real"], nargs="?", default="fake")
    ap.add_argument("min_conf", type=float, nargs="?", default=0.70)
    ap.add_argument("max_new_tokens", type=int, nargs="?", default=120)

    # Paths
    ap.add_argument("--gen_dir", default="models/gpt2_tagalog_finetuned")
    ap.add_argument("--roberta_dir", default="models/roberta_finetuned")
    ap.add_argument(
        "--distil_dir", default="models/distilbert_multilingual_finetuned")
    ap.add_argument("--pca_roberta", default="models/pca_roberta.joblib")
    ap.add_argument("--pca_distil", default="models/pca_distilbert.joblib")
    ap.add_argument(
        "--out_file", default="generated/gpt2_synthetic_samples.jsonl")

    # Generation controls
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Safety: pseudonymize names
    ap.add_argument(
        "--no_pseudonymize",
        action="store_true",
        help="Disable name pseudonymization (NOT recommended for Unity/game exports).",
    )
    ap.add_argument(
        "--placeholder_prefix",
        default="Candidate",
        help="Prefix for pseudonyms, e.g., 'Candidate' -> 'Candidate A'.",
    )

    # Detector controls / filtering
    ap.add_argument(
        "--accept_mode",
        choices=["roberta_only", "distil_only", "both_agree", "ensemble"],
        default="ensemble",
        help="How to apply min_conf filtering before writing outputs.",
    )
    ap.add_argument("--w_roberta", type=float, default=0.60)
    ap.add_argument("--w_distil", type=float, default=0.40)
    ap.add_argument("--det_batch", type=int, default=16)
    ap.add_argument("--det_max_len", type=int, default=256)

    # Output behavior
    ap.add_argument(
        "--write_all",
        action="store_true",
        help="If set, writes all generated candidates (ignores min_conf filtering).",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    gen_dir = Path(args.gen_dir)
    ro_dir = Path(args.roberta_dir)
    di_dir = Path(args.distil_dir)

    if not gen_dir.exists():
        raise FileNotFoundError(
            f"Missing generator dir: {gen_dir} (run 11_train_gpt2MINERVA.py)")
    if not ro_dir.exists():
        raise FileNotFoundError(
            f"Missing RoBERTa dir: {ro_dir} (select/export best detector seed first)")
    if not di_dir.exists():
        raise FileNotFoundError(
            f"Missing DistilBERT dir: {di_dir} (select/export best detector seed first)")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load generator ---
    gen_tok = AutoTokenizer.from_pretrained(str(gen_dir), use_fast=True)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        str(gen_dir)).to(device).eval()

    prompt = FAKE_TOKEN if args.target == "fake" else REAL_TOKEN

    # --- Generate texts ---
    raw_texts: list[str] = []
    for _ in range(args.n):
        inp = gen_tok(prompt, return_tensors="pt").to(device)
        out_ids = gen_model.generate(
            **inp,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=gen_tok.eos_token_id,
        )
        decoded = gen_tok.decode(out_ids[0], skip_special_tokens=True)
        cleaned = strip_control_token(decoded)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        raw_texts.append(cleaned)

    # --- Pseudonymize names BEFORE scoring (so explanations match the shown text) ---
    pseudonymized = False
    if not args.no_pseudonymize:
        raw_texts, _ = pseudonymize_texts(
            raw_texts, placeholder_prefix=args.placeholder_prefix)
        pseudonymized = True

    # --- Load detectors ---
    ro_tok = AutoTokenizer.from_pretrained(str(ro_dir), use_fast=True)
    ro_model = AutoModelForSequenceClassification.from_pretrained(
        str(ro_dir)).to(device)

    di_tok = AutoTokenizer.from_pretrained(str(di_dir), use_fast=True)
    di_model = AutoModelForSequenceClassification.from_pretrained(
        str(di_dir)).to(device)

    # --- Extract detector features (CLS + p_fake) ---
    r_emb, r_p = encode_cls_and_prob(
        ro_model,
        ro_tok,
        raw_texts,
        batch_size=args.det_batch,
        max_len=args.det_max_len,
        device=device,
    )
    d_emb, d_p = encode_cls_and_prob(
        di_model,
        di_tok,
        raw_texts,
        batch_size=args.det_batch,
        max_len=args.det_max_len,
        device=device,
    )

    # --- PCA projection (optional; recommended for Qlattice compatibility) ---
    pca_r_path = Path(args.pca_roberta)
    pca_d_path = Path(args.pca_distil)

    r_pca = None
    d_pca = None
    if pca_r_path.exists() and pca_d_path.exists():
        pca_r = joblib_load(pca_r_path)
        pca_d = joblib_load(pca_d_path)
        r_pca = pca_r.transform(r_emb)
        d_pca = pca_d.transform(d_emb)
    else:
        print(
            "[WARN] PCA joblib files not found. Outputs will not include r_pca_* / d_pca_* "
            "and any Qlattice equation that references PCA features may not apply."
        )

    # --- Build output table with consistent column names ---
    rows = []
    for i, text in enumerate(raw_texts):
        feats = lexical_features(text)
        row = {
            "id": f"gen_{i}",
            "target_label": args.target,
            "text": text,
            # match MINERVA feature naming:
            "p_roberta_fake": float(r_p[i]),
            "p_distil_fake": float(d_p[i]),
            **feats,
            "pseudonymized": pseudonymized,
            "placeholder_prefix": args.placeholder_prefix if pseudonymized else None,
        }

        if r_pca is not None and d_pca is not None:
            for k in range(r_pca.shape[1]):
                row[f"r_pca_{k}"] = float(r_pca[i, k])
            for k in range(d_pca.shape[1]):
                row[f"d_pca_{k}"] = float(d_pca[i, k])

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Apply optional acceptance filter (min_conf) ---
    def accept_row(pr: float, pd_: float) -> bool:
        if args.write_all:
            return True
        need_fake = args.target == "fake"

        if args.accept_mode == "roberta_only":
            pf = pr
        elif args.accept_mode == "distil_only":
            pf = pd_
        elif args.accept_mode == "both_agree":
            if need_fake:
                return (pr >= args.min_conf) and (pd_ >= args.min_conf)
            return (pr <= (1.0 - args.min_conf)) and (pd_ <= (1.0 - args.min_conf))
        else:
            # ensemble
            wsum = max(1e-9, args.w_roberta + args.w_distil)
            pf = (args.w_roberta * pr + args.w_distil * pd_) / wsum

        if need_fake:
            return pf >= args.min_conf
        return pf <= (1.0 - args.min_conf)

    mask = [accept_row(r, d) for r, d in zip(
        df["p_roberta_fake"], df["p_distil_fake"])]
    kept = df[mask].reset_index(drop=True)

    # --- Write JSONL ---
    with open(out_path, "w", encoding="utf-8") as f:
        for j, rec in enumerate(kept.to_dict(orient="records"), start=1):
            rec["id"] = f"gen_keep_{j}"
            rec["accept_mode"] = args.accept_mode
            rec["min_conf"] = args.min_conf
            rec["temperature"] = args.temperature
            rec["top_p"] = args.top_p
            rec["seed"] = args.seed
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[12] Generated: {len(df)} | Kept: {len(kept)}")
    print(f"[12] Saved -> {out_path.resolve()}")
    if pseudonymized:
        print("[12] Pseudonymization: ENABLED (placeholders like 'Candidate A').")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------
# Optional privacy layer
# -----------------------------
try:
    # Expected to exist in repo root (or on PYTHONPATH)
    from minerva_privacy import pseudonymize_text  # type: ignore
except Exception:
    pseudonymize_text = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    # scripts/12_generate_gpt2MINERVA.py -> repo root is parents[1]
    return Path(__file__).resolve().parents[1]


def _detect_run_dir() -> Path:
    """
    Determine the active run directory.

    Priority:
      1) MINERVA_RUN_DIR env var (if set)
      2) latest subdir under /content/drive/MyDrive/MINERVA_RUNS (Colab default)
      3) repo root (fallback)
    """
    env = os.environ.get("MINERVA_RUN_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    colab_runs = Path("/content/drive/MyDrive/MINERVA_RUNS")
    if colab_runs.exists() and colab_runs.is_dir():
        subdirs = [p for p in colab_runs.iterdir() if p.is_dir()]
        if subdirs:
            subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return subdirs[0].resolve()

    return _repo_root()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _get_label_id(model: AutoModelForSequenceClassification, label: str) -> int:
    """
    Resolve a human label string ("fake"/"real") into a class id using model config.

    Falls back to common binary conventions if config is missing.
    """
    label = label.strip()
    cfg_map = getattr(model.config, "label2id", None) or {}
    if isinstance(cfg_map, dict) and cfg_map:
        for k in (label, label.lower(), label.upper(), label.capitalize()):
            if k in cfg_map:
                return int(cfg_map[k])

        # Some configs store {"LABEL_0":0,"LABEL_1":1}
        if label.lower() == "fake" and "LABEL_1" in cfg_map:
            return int(cfg_map["LABEL_1"])
        if label.lower() == "real" and "LABEL_0" in cfg_map:
            return int(cfg_map["LABEL_0"])

    # Final fallback: assume 1=fake, 0=real
    return 1 if label.lower() == "fake" else 0


def _predict_prob(
    tok: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    texts: List[str],
    label_id: int,
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Return probability for `label_id` for each text.
    """
    probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            p = _softmax_probs(out.logits)[:, label_id]
            probs.extend(p.detach().cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_mean_pool(
    tok: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    texts: List[str],
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute mean-pooled encoder embeddings from a SeqClassification model by using its base encoder.
    """
    base_prefix = getattr(model, "base_model_prefix", None)
    encoder = getattr(model, base_prefix) if base_prefix and hasattr(
        model, base_prefix) else model

    embs: List[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = encoder(**enc)
            pooled = _mean_pool_last_hidden(
                out.last_hidden_state, enc["attention_mask"])
            embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs).astype(np.float32)


def _alias_pca_columns(row: Dict[str, object], prefix: str, n_components: int) -> None:
    """
    Given long PCA columns (r_pca_0..), add short aliases (rpca0..).
    prefix: "r" or "d"
    """
    long_prefix = f"{prefix}_pca_"
    short_prefix = f"{prefix}pca"
    for j in range(n_components):
        long_name = f"{long_prefix}{j}"
        short_name = f"{short_prefix}{j}"
        if long_name in row and short_name not in row:
            row[short_name] = row[long_name]


# -----------------------------
# Paths
# -----------------------------
@dataclass
class Paths:
    run_dir: Path
    models_dir: Path
    pca_roberta: Path
    pca_distilbert: Path
    gpt2_model_dir: Path
    roberta_dir: Path
    distilbert_dir: Path
    out_jsonl: Path


def _resolve_paths(run_dir: Path) -> Paths:
    """
    Resolve model + PCA paths using run_dir-first search, then repo-root fallback.
    """
    root = _repo_root()

    def pick(*candidates: Path) -> Path:
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    models_dir = pick(run_dir / "models", root / "models")

    roberta_dir = pick(models_dir / "roberta_finetuned",
                       root / "models" / "roberta_finetuned")
    distilbert_dir = pick(
        models_dir / "distilbert_multilingual_finetuned",
        root / "models" / "distilbert_multilingual_finetuned",
    )
    gpt2_model_dir = pick(models_dir / "gpt2_tagalog_finetuned",
                          root / "models" / "gpt2_tagalog_finetuned")

    pca_roberta = pick(models_dir / "pca_roberta.joblib",
                       root / "models" / "pca_roberta.joblib")
    pca_distilbert = pick(models_dir / "pca_distilbert.joblib",
                          root / "models" / "pca_distilbert.joblib")

    out_dir = _ensure_dir(run_dir / "generated")
    out_jsonl = out_dir / "gpt2_synthetic_samples.jsonl"

    return Paths(
        run_dir=run_dir,
        models_dir=models_dir,
        pca_roberta=pca_roberta,
        pca_distilbert=pca_distilbert,
        gpt2_model_dir=gpt2_model_dir,
        roberta_dir=roberta_dir,
        distilbert_dir=distilbert_dir,
        out_jsonl=out_jsonl,
    )


def _load_pca(p: Path):
    if not p.exists():
        raise FileNotFoundError(
            f"Missing PCA file: {p}\n"
            f"Fix: run scripts/06_extract_features.py to (re)create PCA models, or copy them into {p.parent}."
        )
    obj = joblib.load(p)
    if not hasattr(obj, "transform"):
        raise TypeError(f"PCA object at {p} does not have .transform().")
    return obj


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\[(FAKE|REAL)\]\s*", "", text,
                  flags=re.IGNORECASE).strip()
    return text


def _should_pseudonymize() -> bool:
    if pseudonymize_text is None:
        return False
    v = os.environ.get("MINERVA_PSEUDONYMIZE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n_samples", type=int)
    ap.add_argument("target_label", type=str, choices=["fake", "real"])
    ap.add_argument("min_conf", type=float)
    ap.add_argument("max_new_tokens", type=int)
    ap.add_argument(
        "--accept_mode",
        type=str,
        default="ensemble",
        choices=["roberta", "distilbert", "ensemble"],
        help="Which detector(s) gate acceptance.",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--detector_batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_attempts", type=int, default=5000)
    ap.add_argument("--run_dir", type=str, default="auto")
    ap.add_argument("--out_jsonl", type=str, default="")
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = _detect_run_dir() if args.run_dir == "auto" else Path(
        args.run_dir).expanduser().resolve()
    paths = _resolve_paths(run_dir)

    if args.out_jsonl:
        paths.out_jsonl = Path(args.out_jsonl).expanduser().resolve()
        _ensure_dir(paths.out_jsonl.parent)

    # PCA
    pca_r = _load_pca(paths.pca_roberta)
    pca_d = _load_pca(paths.pca_distilbert)

    # Detectors
    print(f"[12] Using RoBERTa detector -> {paths.roberta_dir}")
    roberta_tok = AutoTokenizer.from_pretrained(
        str(paths.roberta_dir), use_fast=True)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        str(paths.roberta_dir)).to(device)

    print(f"[12] Using DistilBERT detector -> {paths.distilbert_dir}")
    distil_tok = AutoTokenizer.from_pretrained(
        str(paths.distilbert_dir), use_fast=True)
    distil_model = AutoModelForSequenceClassification.from_pretrained(
        str(paths.distilbert_dir)).to(device)

    roberta_target_id = _get_label_id(roberta_model, args.target_label)
    distil_target_id = _get_label_id(distil_model, args.target_label)

    # GPT-2
    gpt2_tok = AutoTokenizer.from_pretrained(
        str(paths.gpt2_model_dir), use_fast=True)
    gpt2_model = AutoModelForCausalLM.from_pretrained(
        str(paths.gpt2_model_dir)).to(device)

    # Decoder-only padding fix
    gpt2_tok.padding_side = "left"
    if gpt2_tok.pad_token is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_model.config.pad_token_id = gpt2_tok.pad_token_id

    pseudonymize = _should_pseudonymize()
    print("[12] Pseudonymization: ENABLED (placeholders like 'Candidate A')." if pseudonymize else "[12] Pseudonymization: DISABLED.")

    kept_rows: List[Dict[str, object]] = []
    attempts = 0
    n_keep = int(args.n_samples)

    pbar = tqdm(total=n_keep, desc="Accepted", unit="sample")

    while len(kept_rows) < n_keep and attempts < args.max_attempts:
        bsz = min(args.batch_size, n_keep - len(kept_rows))

        prompt = f"[{args.target_label.upper()}] "
        enc = gpt2_tok([prompt] * bsz, return_tensors="pt",
                       padding=True).to(device)

        with torch.no_grad():
            gen = gpt2_model.generate(
                **enc,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=gpt2_tok.pad_token_id,
                eos_token_id=gpt2_tok.eos_token_id,
            )

        decoded = gpt2_tok.batch_decode(gen, skip_special_tokens=True)
        candidates = [_clean_generated_text(t) for t in decoded]

        if pseudonymize and pseudonymize_text is not None:
            candidates = [pseudonymize_text(t) for t in candidates]

        # Detector probs for the TARGET label
        rob_p = _predict_prob(roberta_tok, roberta_model, candidates,
                              roberta_target_id, device, batch_size=args.detector_batch_size)
        dis_p = _predict_prob(distil_tok, distil_model, candidates,
                              distil_target_id, device, batch_size=args.detector_batch_size)

        if args.accept_mode == "roberta":
            gate_p = rob_p
        elif args.accept_mode == "distilbert":
            gate_p = dis_p
        else:
            gate_p = (rob_p + dis_p) / 2.0

        accepted_idx = [i for i, p in enumerate(
            gate_p) if float(p) >= float(args.min_conf)]

        if accepted_idx:
            accepted_texts = [candidates[i] for i in accepted_idx]

            r_emb = _encode_mean_pool(
                roberta_tok, roberta_model, accepted_texts, device, batch_size=args.detector_batch_size)
            d_emb = _encode_mean_pool(
                distil_tok, distil_model, accepted_texts, device, batch_size=args.detector_batch_size)

            r_pca = pca_r.transform(r_emb)
            d_pca = pca_d.transform(d_emb)

            for j, text in enumerate(accepted_texts):
                row: Dict[str, object] = {
                    "id": f"gpt2_{len(kept_rows)+1:06d}",
                    "text": text,
                    "target_label": args.target_label,
                    "accept_mode": args.accept_mode,
                    "min_conf": float(args.min_conf),
                    "roberta_prob": float(rob_p[accepted_idx[j]]),
                    "distilbert_prob": float(dis_p[accepted_idx[j]]),
                    "ensemble_prob": float(((rob_p + dis_p) / 2.0)[accepted_idx[j]]),
                    "created_at_unix": int(time.time()),
                }

                for k in range(r_pca.shape[1]):
                    row[f"r_pca_{k}"] = float(r_pca[j, k])
                for k in range(d_pca.shape[1]):
                    row[f"d_pca_{k}"] = float(d_pca[j, k])

                # Add short aliases used by some Qlattice equations
                _alias_pca_columns(row, "r", r_pca.shape[1])  # rpca0..rpca15
                _alias_pca_columns(row, "d", d_pca.shape[1])  # dpca0..dpca15

                kept_rows.append(row)
                pbar.update(1)
                if len(kept_rows) >= n_keep:
                    break

        attempts += bsz

    pbar.close()

    print(f"[12] Generated attempts: {attempts} | Kept: {len(kept_rows)}")
    if len(kept_rows) < n_keep:
        print(
            f"[WARN] Reached max_attempts={args.max_attempts} before collecting n_samples={n_keep}.")

    with open(paths.out_jsonl, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[12] Saved -> {paths.out_jsonl}")
    if pseudonymize:
        print("[12] Pseudonymization: ENABLED (placeholders like 'Candidate A').")


if __name__ == "__main__":
    main()

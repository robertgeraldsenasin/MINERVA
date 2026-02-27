from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# Allow importing repo-root modules when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from minerva_privacy import pseudonymize_texts  # type: ignore
except Exception:
    pseudonymize_texts = None

try:
    from minerva_degnn import load_degnn_artifacts, predict_p_fake_for_new_nodes  # type: ignore
except Exception:
    load_degnn_artifacts = None
    predict_p_fake_for_new_nodes = None


# -----------------------------------------------------------------------------
# Control tokens (must match Script 10 + Script 11)

REAL_TOKEN = "<|label=real|>"
FAKE_TOKEN = "<|label=fake|>"

GRAPH_HIGH = "<|graph=high|>"
GRAPH_MID = "<|graph=mid|>"
GRAPH_LOW = "<|graph=low|>"
GRAPH_UNK = "<|graph=unk|>"


# -----------------------------------------------------------------------------
# Feature helpers (match Script 06 schema)


def compute_lexical_features(text: str) -> Dict[str, float]:
    """Language-agnostic lexical features used across MINERVA."""

    if not isinstance(text, str):
        text = ""

    n_chars = len(text)
    n_words = len(text.split())
    n_exclam = text.count("!")
    n_q = text.count("?")
    n_digits = sum(ch.isdigit() for ch in text)
    digit_ratio = (n_digits / n_chars) if n_chars > 0 else 0.0

    return {
        "char_len": float(n_chars),
        "word_len": float(n_words),
        "exclam": float(n_exclam),
        "question": float(n_q),
        "digit_ratio": float(digit_ratio),
    }


@torch.no_grad()
def encode_texts_cls(
    tok: AutoTokenizer,
    mdl: AutoModelForSequenceClassification,
    texts: List[str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 16,
) -> np.ndarray:
    """Return CLS embeddings from the last hidden layer (matches Script 06)."""

    mdl.eval()
    out_chunks: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc, output_hidden_states=True, return_dict=True)
        cls = out.hidden_states[-1][:, 0, :]
        out_chunks.append(cls.detach().cpu().numpy())

    return np.vstack(out_chunks) if out_chunks else np.zeros((0, 768), dtype=np.float32)


@torch.no_grad()
def predict_prob_fake(
    tok: AutoTokenizer,
    mdl: AutoModelForSequenceClassification,
    texts: List[str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 32,
) -> np.ndarray:
    """Return P(fake) for each text."""

    mdl.eval()
    probs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, 1]
        probs.append(p.detach().cpu().numpy())

    return np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)


# -----------------------------------------------------------------------------
# Misc helpers


def is_hf_model_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and (p / "config.json").exists()


def _load_metrics(metrics_path: Path) -> Optional[dict]:
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric_value(report: dict, metric: str = "eval_f1") -> Optional[float]:
    eval_dict = report.get("eval", {}) or {}
    test_dict = report.get("test", {}) or {}

    if metric in eval_dict:
        return eval_dict.get(metric)

    if not metric.startswith("eval_"):
        k = f"eval_{metric}"
        if k in eval_dict:
            return eval_dict.get(k)

    if metric in test_dict:
        return test_dict.get(metric)
    if not metric.startswith("eval_"):
        k = f"test_{metric}"
        if k in test_dict:
            return test_dict.get(k)

    return None


def autodetect_best_detector(
    task: str,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    metric: str = "eval_f1",
    models_dir: Path = Path("models"),
) -> Path:
    """Pick the best HF detector dir under models/<task>/run_*/seed_* by metrics.json."""

    base = models_dir / task
    if not base.exists():
        raise FileNotFoundError(
            f"Detector base dir not found: {base}.\n"
            "Fix: run scripts/17_run_5seeds_detectors.py (recommended) or "
            "train via scripts/16_train_transformer_classifier.py."
        )

    if run_id:
        run_dirs = [base / f"run_{run_id}"]
    else:
        run_dirs = sorted(
            [p for p in base.glob("run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    candidates: List[Path] = []
    for rd in run_dirs:
        if not rd.exists():
            continue
        if seed is not None:
            sd = rd / f"seed_{seed}"
            if sd.exists():
                candidates.append(sd)
        else:
            candidates.extend(
                sorted([p for p in rd.glob("seed_*") if p.is_dir()]))

    candidates = [c for c in candidates if is_hf_model_dir(c)]
    if not candidates:
        raise FileNotFoundError(
            f"No saved detector model directories found under: {base}\n"
            "Fix: run scripts/17_run_5seeds_detectors.py first."
        )

    best_dir = candidates[0]
    best_score = float("-inf")

    for c in candidates:
        m = _load_metrics(c / "metrics.json")
        if not m:
            continue
        val = _metric_value(m, metric=metric)
        if val is None:
            continue
        if float(val) > best_score:
            best_score = float(val)
            best_dir = c

    return best_dir


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"^\[(FAKE|REAL)\]\s*", "", text,
                  flags=re.IGNORECASE).strip()
    return text


def _should_pseudonymize(no_privacy_flag: bool) -> bool:
    if no_privacy_flag:
        return False
    if pseudonymize_texts is None:
        return False
    v = os.environ.get("MINERVA_PSEUDONYMIZE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _graph_token_from_prompt(arg: str) -> str:
    v = (arg or "").strip().lower()
    if v == "high":
        return GRAPH_HIGH
    if v == "mid":
        return GRAPH_MID
    if v == "low":
        return GRAPH_LOW
    return GRAPH_UNK


def _confidence_for_target(p_fake: float, target: str) -> float:
    return float(p_fake) if target == "fake" else float(1.0 - float(p_fake))


# -----------------------------------------------------------------------------
# Paths


@dataclass
class Paths:
    models_dir: Path
    roberta_dir: Path
    distil_dir: Path
    gpt2_dir: Path
    pca_roberta: Path
    pca_distilbert: Path
    degnn_artifacts: Path


def resolve_paths(args) -> Paths:
    root = REPO_ROOT
    models_dir = Path(args.models_dir) if args.models_dir else (
        root / "models")

    return Paths(
        models_dir=models_dir,
        roberta_dir=Path(args.roberta_dir),
        distil_dir=Path(args.distil_dir),
        gpt2_dir=Path(args.gen_model_dir),
        pca_roberta=Path(args.pca_roberta),
        pca_distilbert=Path(args.pca_distilbert),
        degnn_artifacts=Path(args.degnn_artifacts),
    )


# -----------------------------------------------------------------------------
# Main


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MINERVA Script 12: generate GPT-2 synthetic samples.")

    p.add_argument(
        "n", type=int, help="Number of ACCEPTED samples to write to JSONL")
    p.add_argument("target", choices=[
                   "fake", "real"], help="Target label for conditioning + filtering")
    p.add_argument("min_conf", type=float,
                   help="Acceptance threshold for chosen accept_mode")
    p.add_argument("max_new_tokens", type=int,
                   help="Max new tokens per generation")

    p.add_argument(
        "--accept_mode",
        default="ensemble3",
        choices=["roberta", "distilbert", "ensemble", "degnn", "ensemble3"],
        help="Acceptance gate: roberta | distilbert | ensemble(avg) | degnn | ensemble3(avg of all three)",
    )

    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=16,
                   help="Generation batch size")

    p.add_argument("--out", default="generated/gpt2_synthetic_samples.jsonl")
    p.add_argument("--gen_model_dir", default="models/gpt2_tagalog_finetuned")
    p.add_argument("--models_dir", default="models",
                   help="Base models directory")

    # Detector selection
    p.add_argument("--roberta_dir", default="models/roberta_finetuned")
    p.add_argument(
        "--distil_dir", default="models/distilbert_multilingual_finetuned")
    p.add_argument("--run_id", default=None)
    p.add_argument("--detector_seed", type=int, default=None)
    p.add_argument("--detector_metric", default="eval_f1")

    # PCA paths
    p.add_argument("--pca_roberta", default="models/pca_roberta.joblib")
    p.add_argument("--pca_distilbert", default="models/pca_distilbert.joblib")

    # DE-GNN artifacts
    p.add_argument("--degnn_artifacts",
                   default="models/degnn_artifacts.joblib")

    # Prompt conditioning token for DE-GNN bins
    p.add_argument(
        "--graph_prompt",
        default="high",
        choices=["high", "mid", "low", "unk"],
        help="Which <|graph=...|> token to prepend (use 'unk' if you trained without graph tokens).",
    )

    p.add_argument("--no_privacy", action="store_true",
                   help="Disable pseudonymization if available")
    p.add_argument(
        "--max_batches",
        type=int,
        default=200,
        help="Hard cap on generation batches (prevents infinite loops if acceptance is too strict).",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeds
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    paths = resolve_paths(args)

    if not paths.gpt2_dir.exists():
        raise FileNotFoundError(
            f"Missing GPT-2 model dir: {paths.gpt2_dir}.\nFix: run scripts/11_train_gpt2MINERVA.py first."
        )

    # Resolve detectors
    roberta_dir = paths.roberta_dir
    if not is_hf_model_dir(roberta_dir):
        roberta_dir = autodetect_best_detector(
            task="roberta",
            run_id=args.run_id,
            seed=args.detector_seed,
            metric=args.detector_metric,
            models_dir=paths.models_dir,
        )

    distil_dir = paths.distil_dir
    if not is_hf_model_dir(distil_dir):
        distil_dir = autodetect_best_detector(
            task="distilbert",
            run_id=args.run_id,
            seed=args.detector_seed,
            metric=args.detector_metric,
            models_dir=paths.models_dir,
        )

    print(f"[12] Using RoBERTa detector -> {roberta_dir}")
    print(f"[12] Using DistilBERT detector -> {distil_dir}")

    # Load generator
    gen_tok = AutoTokenizer.from_pretrained(str(paths.gpt2_dir), use_fast=True)
    gen_mdl = AutoModelForCausalLM.from_pretrained(
        str(paths.gpt2_dir)).to(device)
    gen_mdl.eval()

    # Ensure control tokens exist
    special = {
        "additional_special_tokens": [
            REAL_TOKEN,
            FAKE_TOKEN,
            GRAPH_HIGH,
            GRAPH_MID,
            GRAPH_LOW,
            GRAPH_UNK,
        ]
    }
    gen_tok.add_special_tokens(special)
    gen_mdl.resize_token_embeddings(len(gen_tok))

    # Load detectors
    rob_tok = AutoTokenizer.from_pretrained(str(roberta_dir), use_fast=True)
    rob_mdl = AutoModelForSequenceClassification.from_pretrained(
        str(roberta_dir)).to(device)

    dis_tok = AutoTokenizer.from_pretrained(str(distil_dir), use_fast=True)
    dis_mdl = AutoModelForSequenceClassification.from_pretrained(
        str(distil_dir)).to(device)

    # PCA models
    if not paths.pca_roberta.exists() or not paths.pca_distilbert.exists():
        raise FileNotFoundError(
            "Missing PCA models.\n"
            f"Expected: {paths.pca_roberta} and {paths.pca_distilbert}\n"
            "Fix: run scripts/06_extract_features.py first."
        )
    pca_r = joblib.load(paths.pca_roberta)
    pca_d = joblib.load(paths.pca_distilbert)

    # Optional DE-GNN
    need_degnn = args.accept_mode in ("degnn", "ensemble3")
    degnn_art = None
    if need_degnn:
        if load_degnn_artifacts is None or predict_p_fake_for_new_nodes is None:
            raise RuntimeError(
                "DE-GNN utilities missing (minerva_degnn.py). Ensure the patch files are in repo root."
            )
        if not paths.degnn_artifacts.exists():
            raise FileNotFoundError(
                f"Missing DE-GNN artifacts: {paths.degnn_artifacts}\nFix: run scripts/09_train_degnn.py first."
            )
        degnn_art = load_degnn_artifacts(paths.degnn_artifacts)

    # Privacy
    do_priv = _should_pseudonymize(args.no_privacy)
    print(f"[12] Pseudonymization: {'ENABLED' if do_priv else 'DISABLED'}.")

    # Conditioning prompt
    label_tok = FAKE_TOKEN if args.target == "fake" else REAL_TOKEN
    graph_tok = _graph_token_from_prompt(args.graph_prompt)
    prompt_prefix = f"{label_tok} {graph_tok} "

    accepted_rows: List[dict] = []
    accepted = 0
    total_batches = 0

    while accepted < int(args.n) and total_batches < int(args.max_batches):
        total_batches += 1

        base_prompts = ["Ulat:", "Balita:", "Update:",
                        "Trending:", "BREAKING:", "Babala:", "Paalala:"]
        prompts = [
            prompt_prefix + rng.choice(base_prompts) + " " for _ in range(int(args.batch))]

        enc = gen_tok(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_ids = gen_mdl.generate(
            **enc,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            pad_token_id=gen_tok.eos_token_id,
        )

        texts = gen_tok.batch_decode(gen_ids, skip_special_tokens=True)
        texts = [_clean_generated_text(t) for t in texts]
        texts = [t for t in texts if len(t) > 0]
        if not texts:
            continue

        if do_priv and pseudonymize_texts is not None:
            texts, _ = pseudonymize_texts(texts)

        # Detector probabilities
        r_p_fake = predict_prob_fake(rob_tok, rob_mdl, texts, device=device)
        d_p_fake = predict_prob_fake(dis_tok, dis_mdl, texts, device=device)

        # Cheap pre-filter using roberta/distil ensemble (or single model)
        if args.accept_mode == "roberta":
            accept_score = np.array([_confidence_for_target(
                p, args.target) for p in r_p_fake], dtype=float)
        elif args.accept_mode == "distilbert":
            accept_score = np.array([_confidence_for_target(
                p, args.target) for p in d_p_fake], dtype=float)
        else:
            ens = 0.5 * (r_p_fake + d_p_fake)
            accept_score = np.array([_confidence_for_target(
                p, args.target) for p in ens], dtype=float)

        pre_mask = accept_score >= float(args.min_conf)
        if not np.any(pre_mask):
            continue

        kept_texts = [t for t, m in zip(texts, pre_mask) if bool(m)]
        kept_rp = r_p_fake[pre_mask]
        kept_dp = d_p_fake[pre_mask]

        # PCA features
        r_emb = encode_texts_cls(rob_tok, rob_mdl, kept_texts, device=device)
        d_emb = encode_texts_cls(dis_tok, dis_mdl, kept_texts, device=device)
        r_pca = pca_r.transform(r_emb)
        d_pca = pca_d.transform(d_emb)

        # Lexical features
        lex_rows = [compute_lexical_features(t) for t in kept_texts]

        # Optional DE-GNN acceptance refinement
        p_degnn_fake = np.array([np.nan] * len(kept_texts), dtype=float)
        if need_degnn and degnn_art is not None:
            feat = pd.DataFrame(lex_rows)
            feat["p_roberta_fake"] = kept_rp
            feat["p_distil_fake"] = kept_dp
            for j in range(r_pca.shape[1]):
                feat[f"r_pca_{j}"] = r_pca[:, j]
            for j in range(d_pca.shape[1]):
                feat[f"d_pca_{j}"] = d_pca[:, j]

            for c in degnn_art.feature_cols:
                if c not in feat.columns:
                    feat[c] = 0.0
            feat = feat[degnn_art.feature_cols]

            p_degnn_fake = predict_p_fake_for_new_nodes(degnn_art, feat)

            if args.accept_mode == "degnn":
                accept_score2 = np.array([_confidence_for_target(
                    p, args.target) for p in p_degnn_fake], dtype=float)
            else:
                ens3 = (kept_rp + kept_dp + p_degnn_fake) / 3.0
                accept_score2 = np.array([_confidence_for_target(
                    p, args.target) for p in ens3], dtype=float)

            keep_mask = accept_score2 >= float(args.min_conf)
        else:
            keep_mask = np.ones((len(kept_texts),), dtype=bool)

        if not np.any(keep_mask):
            continue

        for idx in np.where(keep_mask)[0].tolist():
            if accepted >= int(args.n):
                break

            row = {
                "id": f"gpt2_{args.target}_s{int(args.seed)}_{accepted:06d}",
                "target_label": args.target,
                "text": kept_texts[idx],
                "accept_mode": args.accept_mode,
                "min_conf": float(args.min_conf),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "seed": int(args.seed),
                "graph_prompt": args.graph_prompt,
                "p_roberta_fake": float(kept_rp[idx]),
                "p_distil_fake": float(kept_dp[idx]),
                "p_degnn_fake": (float(p_degnn_fake[idx]) if np.isfinite(p_degnn_fake[idx]) else None),
            }

            row.update({k: float(v) for k, v in lex_rows[idx].items()})

            for j in range(r_pca.shape[1]):
                row[f"r_pca_{j}"] = float(r_pca[idx, j])
            for j in range(d_pca.shape[1]):
                row[f"d_pca_{j}"] = float(d_pca[idx, j])

            accepted_rows.append(row)
            accepted += 1

        if accepted % 25 == 0 or accepted == int(args.n):
            print(
                f"[12] Accepted {accepted}/{int(args.n)} (batches={total_batches})")

    with out_path.open("w", encoding="utf-8") as f:
        for r in accepted_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if accepted < int(args.n):
        print(
            f"[WARN] Requested {int(args.n)} accepted samples but only produced {accepted}. "
            f"Try lowering min_conf or increasing max_batches."
        )

    print(f"[12] Output -> {out_path.resolve()} (rows={len(accepted_rows)})")


if __name__ == "__main__":
    main()

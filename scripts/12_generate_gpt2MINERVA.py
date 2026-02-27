from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import joblib
from typing import Dict, List, Optional
from pathlib import Path
import os
import json
mport argparse


try:
    from minerva_privacy import pseudonymize_texts
except Exception:
    pseudonymize_texts = None


def compute_lexical_features(text: str) -> Dict[str, float]:
    # Simple, cheap, language-agnostic features that work for Tagalog + mixed code-switching.
    n_chars = len(text)
    words = text.split()
    n_words = len(words)
    n_exclaims = text.count("!")
    n_questions = text.count("?")
    pct_upper = sum(1 for c in text if c.isupper()) / max(1, n_chars)
    has_url = 1.0 if (
        "http://" in text or "https://" in text or "www." in text) else 0.0
    has_number = 1.0 if any(c.isdigit() for c in text) else 0.0
    return {
        "n_chars": float(n_chars),
        "n_words": float(n_words),
        "n_exclaims": float(n_exclaims),
        "n_questions": float(n_questions),
        "pct_upper": float(pct_upper),
        "has_url": float(has_url),
        "has_number": float(has_number),
    }


def is_hf_model_dir(p: Path) -> bool:
    # A minimal check for a local Hugging Face model directory.
    return p.exists() and p.is_dir() and (p / "config.json").exists()


def _load_metrics(metrics_path: Path) -> Optional[dict]:
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric_value(report: dict, metric: str = "eval_f1") -> Optional[float]:
    # Prefer eval metrics (validation) for model selection.
    eval_dict = report.get("eval", {}) or {}
    test_dict = report.get("test", {}) or {}

    # Allow passing either "eval_f1" or "f1"
    if metric in eval_dict:
        return eval_dict.get(metric)

    if not metric.startswith("eval_"):
        k = f"eval_{metric}"
        if k in eval_dict:
            return eval_dict.get(k)

    # Fallback to test metrics if eval isn't present (should be rare)
    if metric in test_dict:
        return test_dict.get(metric)
    if not metric.startswith("eval_") and metric in test_dict:
        return test_dict.get(metric)

    return None


def autodetect_best_detector(
    task: str,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    metric: str = "eval_f1",
    models_dir: Path = Path("models"),
) -> Path:
    """Find the best available detector directory for a given task.

    Expected layouts:
      - models/<task>/run_<run_id>/seed_<seed>/
      - models/<task>/run_*/seed_*/
    """
    base = models_dir / task
    if not base.exists():
        raise FileNotFoundError(
            f"Detector base dir not found: {base}.\n"
            f"Fix: run scripts/17_run_5seeds_detectors.py (recommended) or train a single seed via scripts/16_train_transformer_classifier.py."
        )

    # Restrict to a specific run if provided.
    if run_id:
        rd = base / f"run_{run_id}"
        if not rd.exists():
            raise FileNotFoundError(
                f"No run directory for task='{task}' and run_id='{run_id}'. Expected: {rd}\n"
                f"Fix: ensure you trained with --run_id {run_id}."
            )
        run_dirs = [rd]
    else:
        # Prefer newest run_* by modified time
        run_dirs = sorted([p for p in base.glob("run_*") if p.is_dir()],
                          key=lambda p: p.stat().st_mtime, reverse=True)

    candidates: List[Path] = []
    for rd in run_dirs:
        if seed is not None:
            sd = rd / f"seed_{seed}"
            if sd.exists():
                candidates.append(sd)
        else:
            candidates.extend(
                sorted([p for p in rd.glob("seed_*") if p.is_dir()]))

    # Filter to directories that look like saved HF models.
    candidates = [c for c in candidates if is_hf_model_dir(c)]

    if not candidates:
        raise FileNotFoundError(
            f"No saved detector model directories found under: {base}\n"
            f"Fix: run scripts/17_run_5seeds_detectors.py first (it creates models/{task}/run_<RUN_ID>/seed_<SEED>/).\n"
            f"Tip: if you trained but directories are missing config.json, ensure scripts/16_train_transformer_classifier.py calls trainer.save_model(out_dir) and tokenizer.save_pretrained(out_dir)."
        )

    # Select best by metrics.json (highest eval metric).
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

    # If we never saw a metric, keep the first candidate (newest run + lowest seed order).
    return best_dir


def load_detector(model_dir: Path, device: torch.device):
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    mdl.to(device)
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def score_texts(tok, mdl, texts: List[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(batch, truncation=True, padding=True,
                  max_length=256, return_tensors="pt").to(device)
        logits = mdl(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, 1]  # class 1 = fake
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def encode_texts(tok, mdl, texts: List[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    # Mean-pooled last hidden states (simple, fast).
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(batch, truncation=True, padding=True,
                  max_length=256, return_tensors="pt").to(device)
        out = mdl.base_model(
            **enc, output_hidden_states=False, return_dict=True)
        last = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(pooled.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)


# ---------------------------
# Main
# ---------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "n", type=int, help="Number of candidates to generate (before filtering)")
    p.add_argument("target", choices=[
                   "fake", "real"], help="Label tag attached to generated samples")
    p.add_argument("min_conf", type=float,
                   help="Minimum confidence threshold for accept_mode")
    p.add_argument("max_new_tokens", type=int,
                   help="Max tokens to generate per sample")
    p.add_argument(
        "--accept_mode",
        default="ensemble",
        choices=["ensemble", "roberta", "distilbert"],
        help="Which detector(s) to use for filtering: ensemble=(roberta+distilbert avg)",
    )
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="generated/gpt2_synthetic_samples.jsonl")
    p.add_argument("--gen_model_dir", default="models/gpt2_tagalog_finetuned")

    # Detector selection knobs
    p.add_argument(
        "--roberta_dir",
        default="models/roberta_finetuned",
        help="Path to a fine-tuned RoBERTa detector dir. If missing/invalid, auto-detect is used.",
    )
    p.add_argument(
        "--distil_dir",
        default="models/distilbert_multilingual_finetuned",
        help="Path to a fine-tuned DistilBERT detector dir. If missing/invalid, auto-detect is used.",
    )
    p.add_argument("--run_id", default=None,
                   help="Optional: restrict auto-detection to models/<task>/run_<run_id>/seed_*/")
    p.add_argument("--detector_seed", type=int, default=None,
                   help="Optional: choose a specific seed directory if available.")
    p.add_argument("--detector_metric", default="eval_f1",
                   help="Metric key used to pick best seed (default: eval_f1).")

    p.add_argument("--no_privacy", action="store_true",
                   help="Disable pseudonymization if minerva_privacy is available.")
    return p


def main():
    args = build_argparser().parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    gen_model_dir = Path(args.gen_model_dir)
    if not gen_model_dir.exists():
        raise FileNotFoundError(
            f"Missing GPT-2 model dir: {gen_model_dir}.\n"
            f"Fix: run scripts/11_train_gpt2MINERVA.py first."
        )

    # Resolve detector directories (auto-detect if legacy paths are missing/invalid).
    roberta_dir = Path(args.roberta_dir)
    if not is_hf_model_dir(roberta_dir):
        roberta_dir = autodetect_best_detector(
            task="roberta",
            run_id=args.run_id,
            seed=args.detector_seed,
            metric=args.detector_metric,
            models_dir=Path("models"),
        )

    distil_dir = Path(args.distil_dir)
    if not is_hf_model_dir(distil_dir):
        distil_dir = autodetect_best_detector(
            task="distilbert",
            run_id=args.run_id,
            seed=args.detector_seed,
            metric=args.detector_metric,
            models_dir=Path("models"),
        )

    print(f"[12] Using RoBERTa detector -> {roberta_dir}")
    print(f"[12] Using DistilBERT detector -> {distil_dir}")

    # Load generator
    gen_tok = AutoTokenizer.from_pretrained(str(gen_model_dir), use_fast=True)
    gen_mdl = AutoModelForCausalLM.from_pretrained(
        str(gen_model_dir)).to(device)
    gen_mdl.eval()

    # Load detectors
    r_tok, r_mdl = load_detector(roberta_dir, device)
    d_tok, d_mdl = load_detector(distil_dir, device)

    # Load PCA (optional)
    models_dir = Path("models")
    ro_pca_path = models_dir / "pca_roberta.joblib"
    di_pca_path = models_dir / "pca_distilbert.joblib"
    pca_ro = joblib.load(ro_pca_path) if ro_pca_path.exists() else None
    pca_di = joblib.load(di_pca_path) if di_pca_path.exists() else None

    if pca_ro is None or pca_di is None:
        print("[WARN] PCA files not found. Generated samples will NOT include r_pca_* / d_pca_* features.")
        print("       If your Qlattice equation uses PCA terms (rpca*/dpca*), run scripts/06_extract_features.py to create PCA joblibs, then re-run this script.")

    # Generate candidates
    prompts = [
        "Ulat: ",
        "Balita: ",
        "Ayon sa mga ulat, ",
        "Ayon sa isang source, ",
        "Breaking: ",
    ]

    all_texts: List[str] = []
    gen_batch = args.batch
    to_generate = args.n

    while len(all_texts) < to_generate:
        b = min(gen_batch, to_generate - len(all_texts))
        batch_prompts = rng.choice(prompts, size=b, replace=True).tolist()
        enc = gen_tok(batch_prompts, return_tensors="pt",
                      padding=True).to(device)
        out_ids = gen_mdl.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=gen_tok.eos_token_id,
        )
        texts = gen_tok.batch_decode(out_ids, skip_special_tokens=True)
        all_texts.extend([t.strip() for t in texts])

    # Optional pseudonymization
    privacy_enabled = (not args.no_privacy) and (
        pseudonymize_texts is not None)
    if privacy_enabled:
        all_texts, _maps = pseudonymize_texts(
            all_texts, placeholder_prefix="Candidate")

    # Score with detectors
    r_probs = score_texts(r_tok, r_mdl, all_texts,
                          device=device, batch_size=32)
    d_probs = score_texts(d_tok, d_mdl, all_texts,
                          device=device, batch_size=32)

    if args.accept_mode == "roberta":
        accept_score = r_probs
    elif args.accept_mode == "distilbert":
        accept_score = d_probs
    else:
        accept_score = 0.5 * (r_probs + d_probs)

    # Filter by threshold
    keep_mask = accept_score >= args.min_conf
    kept_texts = [t for t, m in zip(all_texts, keep_mask) if m]
    kept_r = r_probs[keep_mask]
    kept_d = d_probs[keep_mask]
    kept_s = accept_score[keep_mask]

    # Encode & PCA
    rows: List[dict] = []
    if kept_texts:
        r_emb = encode_texts(r_tok, r_mdl, kept_texts,
                             device=device, batch_size=32)
        d_emb = encode_texts(d_tok, d_mdl, kept_texts,
                             device=device, batch_size=32)

        r_p = pca_ro.transform(r_emb) if pca_ro is not None else None
        d_p = pca_di.transform(d_emb) if pca_di is not None else None

        for i, txt in enumerate(kept_texts):
            row = {
                "text": txt,
                "target": args.target,
                "accept_mode": args.accept_mode,
                "accept_score": float(kept_s[i]),
                "p_roberta_fake": float(kept_r[i]),
                "p_distilbert_fake": float(kept_d[i]),
                "privacy_enabled": bool(privacy_enabled),
                "detector_roberta_dir": str(roberta_dir),
                "detector_distilbert_dir": str(distil_dir),
            }
            row.update(compute_lexical_features(txt))
            if r_p is not None:
                for k in range(r_p.shape[1]):
                    row[f"r_pca_{k}"] = float(r_p[i, k])
            if d_p is not None:
                for k in range(d_p.shape[1]):
                    row[f"d_pca_{k}"] = float(d_p[i, k])
            rows.append(row)

    # Write JSONL
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[12] Generated: {len(all_texts)} | Kept: {len(rows)}")
    print(f"[12] Saved -> {out_path}")
    if privacy_enabled:
        print("[12] Pseudonymization: ENABLED (placeholders like 'Candidate A').")
    else:
        print("[12] Pseudonymization: DISABLED.")


if __name__ == "__main__":
    main()

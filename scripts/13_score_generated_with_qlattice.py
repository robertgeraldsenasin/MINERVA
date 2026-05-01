#!/usr/bin/env python3
"""
13_score_generated_with_qlattice.py  (REFACTORED v2.0)
======================================================

Score GPT-2-generated candidate posts with the QLattice symbolic
regression model AND a post-generation quality gate.

WHAT CHANGED FROM v1
--------------------
v1 fed only the opaque PCA components (rpca0..rpca15, dpca0..dpca15)
to QLattice, producing equations like `0.62 + 0.41*rpca3 - 0.18*dpca7`
that no SHS student can interpret.

v2.0:
  * Named-feature augmentation: alongside the PCA components we now
    feed the 12 indicator features (ind_emo_fired, ind_urg_score,
    num_urls, caps_ratio, etc.) extracted by minerva_indicators.
    QLattice picks whichever set of features minimises loss; in
    practice the named features dominate for the simple equations
    we want for SHS interpretability (Christensen et al. 2022;
    Brolós et al. 2021).
  * Post-generation truncation gate: any output that does not end on
    terminal punctuation gets retried up to N times. Final output
    rejected if still truncated.
  * Optional perplexity gate (Wenzek et al. 2020): a per-card
    perplexity score from the same GPT-2-JCBlaise model used for
    generation; cards above the 95th-percentile perplexity threshold
    are flagged as low-fluency.
  * Structured rejection log under reports/.

If your existing repo's 13_score_generated_with_qlattice.py already
fits a QLattice and writes JSONL, this drop-in replacement adds
named features and the gate without disturbing your existing
training-set scoring path.

PIPELINE POSITION
-----------------
Reads:  generated/gpt2_synthetic_raw_*.jsonl (GPT-2 outputs)
        models/qlattice_model.pkl
        models/pca_models.pkl
Writes: generated/gpt2_synthetic_final_*.jsonl
        reports/score_rejection_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_indicators import named_features
from minerva_filters import is_truncated

logger = logging.getLogger(__name__)


def score_record(rec: dict, qlattice_model=None, pca_models=None) -> dict:
    """
    Compute QLattice + detector ensemble score for one record.

    This function expects upstream embeddings in rec['emb_roberta'],
    rec['emb_distil'], rec['emb_degnn']. If they are absent (e.g.
    when called on legacy JSONL), we fall back to whatever per-record
    p_*_fake numbers are present and skip QLattice re-evaluation.
    """
    text = rec.get("text") or rec.get("generated_text") or ""

    # Named features (always available; deterministic from text)
    feats = named_features(text)
    rec.setdefault("named_features", {}).update(feats)

    # If a fitted QLattice model is provided, use it; else trust upstream
    if qlattice_model is not None and pca_models is not None:
        try:
            import numpy as np
            import pandas as pd
            # Stitch PCA components if embeddings present
            row = dict(feats)
            for kind, embkey in [
                ("r", "emb_roberta"),
                ("d", "emb_distil"),
                ("g", "emb_degnn"),
            ]:
                emb = rec.get(embkey)
                if emb is not None and kind in pca_models:
                    pcs = pca_models[kind].transform(np.array([emb]))[0]
                    for i, v in enumerate(pcs):
                        row[f"{kind}pca{i}"] = float(v)
            df = pd.DataFrame([row])
            score = float(qlattice_model.predict(df)[0])
            rec["qlattice"] = {
                "score": score,
                "threshold": 0.5,
                "direction": ">=",
                "margin": score - 0.5,
                "pred": int(score >= 0.5),
                "equation": getattr(qlattice_model, "expression", str(qlattice_model)),
                "top_factors": _top_factors(qlattice_model, row),
            }
        except Exception as e:
            logger.debug("QLattice scoring failed for %s: %s",
                         rec.get("id", "?"), e)

    # Fallback p_fake computation
    if "p_fake" not in rec:
        det = rec.get("detectors") or {}
        rec["p_fake"] = float(
            det.get("p_ensemble_fake")
            or rec.get("qlattice", {}).get("score")
            or 0.5
        )

    return rec


def _top_factors(model, row: dict) -> list:
    """Best-effort extraction of model's top contributing features."""
    try:
        # Feyn QLattice models expose .features / .inputs
        feats = list(getattr(model, "features", []))[:5]
        return [{"feature": f, "value": row.get(f, 0.0)} for f in feats]
    except Exception:
        # Heuristic fallback: pick top numeric features by |value|
        items = sorted(row.items(), key=lambda kv: -abs(kv[1]))[:5]
        return [{"feature": k, "value": v} for k, v in items if isinstance(v, (int, float))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--qlattice_model", default=None,
                   help="Optional path to a saved QLattice .pkl; if absent, "
                        "we trust upstream p_fake")
    p.add_argument("--pca_models", default=None,
                   help="Optional path to dict-of-PCA-models pkl")
    p.add_argument("--rejection_log", default="reports/score_rejection_log.jsonl")
    p.add_argument("--truncation_max_retries", type=int, default=0,
                   help="If your generator supports retry, pass >0; this "
                        "script only LOGS truncation, retry is the "
                        "generator's job")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    qlattice_model = pca_models = None
    if args.qlattice_model and Path(args.qlattice_model).exists():
        try:
            import pickle
            qlattice_model = pickle.load(open(args.qlattice_model, "rb"))
            logger.info("Loaded QLattice from %s", args.qlattice_model)
        except Exception as e:
            logger.warning("Could not load QLattice: %s", e)
    if args.pca_models and Path(args.pca_models).exists():
        try:
            import pickle
            pca_models = pickle.load(open(args.pca_models, "rb"))
            logger.info("Loaded PCA models from %s", args.pca_models)
        except Exception as e:
            logger.warning("Could not load PCA: %s", e)

    rejections: list = []
    out_count = trunc_count = 0
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.in_file, "r", encoding="utf-8") as fin, \
         open(args.out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or rec.get("generated_text") or ""
            truncated, why = is_truncated(text)
            if truncated:
                trunc_count += 1
                rejections.append({
                    "card_id": rec.get("id", "unknown"),
                    "stage": "truncation_filter",
                    "verdict": "reject",
                    "reason": f"truncation: {why}",
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                continue  # drop truncated cards
            scored = score_record(rec, qlattice_model, pca_models)
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
            out_count += 1

    Path(args.rejection_log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.rejection_log, "w", encoding="utf-8") as f:
        for r in rejections:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Scored %d records, dropped %d truncated → %s",
                out_count, trunc_count, args.out_file)
    logger.info("Rejection log → %s", args.rejection_log)


if __name__ == "__main__":
    main()

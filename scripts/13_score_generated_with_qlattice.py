#!/usr/bin/env python3
"""Score generated candidates with the QLattice equation; drop low-confidence and out-of-policy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_indicators import named_features  # noqa: E402
from minerva_filters import is_truncated        # noqa: E402

logger = logging.getLogger(__name__)


def score_record(rec: dict, qlattice_model=None, pca_models=None) -> dict:
    """
    Compute QLattice + detector ensemble for one GPT-2 record.

    Reads detector probabilities from the actual script-12 output
    schema (top-level fields). Falls back gracefully if some are
    missing.
    """
    text = rec.get("text") or rec.get("generated_text") or ""

    # Named features (deterministic from text)
    rec.setdefault("named_features", {}).update(named_features(text))

    # Detector probabilities — read from TOP LEVEL (script 12 schema)
    p_roberta = rec.get("p_roberta_fake")
    p_distil  = rec.get("p_distil_fake")
    p_degnn   = rec.get("p_degnn_fake")

    # Forward-compat fallback
    det_in = rec.get("detectors") or {}
    if p_roberta is None:
        p_roberta = det_in.get("p_roberta_fake")
    if p_distil is None:
        p_distil = det_in.get("p_distil_fake")
    if p_degnn is None:
        p_degnn = det_in.get("p_degnn_fake")

    # Average available detectors
    detectors_present = [float(p) for p in (p_roberta, p_distil, p_degnn)
                         if p is not None]
    p_ensemble = sum(detectors_present) / len(detectors_present) \
        if detectors_present else 0.5

    rec["detectors"] = {
        "p_roberta_fake": float(p_roberta) if p_roberta is not None
        else float(p_ensemble),
        "p_distil_fake": float(p_distil) if p_distil is not None
        else float(p_ensemble),
        "p_degnn_fake": float(p_degnn) if p_degnn is not None
        else float(p_ensemble),
        "p_ensemble_fake": float(p_ensemble),
    }
    rec["p_fake"] = float(p_ensemble)

    # QLattice scoring (optional)
    if qlattice_model is not None:
        try:
            import pandas as pd
            row = dict(rec.get("named_features", {}))
            for k, v in rec.items():
                if k.startswith(("r_pca_", "d_pca_", "g_pca_")) \
                        and isinstance(v, (int, float)):
                    row[k] = float(v)
            df = pd.DataFrame([row])
            score = float(qlattice_model.predict(df)[0])
            rec["qlattice"] = {
                "score": score,
                "threshold": 0.5,
                "direction": ">=",
                "margin": score - 0.5,
                "pred": int(score >= 0.5),
                "equation": getattr(qlattice_model, "expression",
                                    str(qlattice_model)),
                "top_factors": _top_factors(qlattice_model, row),
            }
            rec["p_fake"] = score
            rec["detectors"]["p_ensemble_fake"] = score
        except Exception as e:
            logger.debug("QLattice scoring failed for %s: %s",
                         rec.get("id", "?"), e)

    truncated, why = is_truncated(text)
    rec["truncation_flag"] = {"is_truncated": bool(truncated), "reason": why}
    return rec


def _top_factors(model, row: dict) -> list:
    try:
        feats = list(getattr(model, "features", []))[:5]
        return [{"feature": f, "value": row.get(f, 0.0)} for f in feats]
    except Exception:
        items = sorted(
            ((k, v) for k, v in row.items() if isinstance(v, (int, float))),
            key=lambda kv: -abs(kv[1]),
        )[:5]
        return [{"feature": k, "value": v} for k, v in items]


def should_drop(rec: dict) -> tuple[bool, str]:
    """Lenient drop policy — only reject obviously broken records."""
    text = (rec.get("text") or "").strip()
    if not text:
        return True, "empty_text"
    if len(text) < 30:
        return True, "too_short"
    DANGLERS = {"ang", "ng", "sa", "at", "ay", "kay", "mga",
                "the", "a", "an", "of", "and", "or", "but",
                "with", "for", "to", "by"}
    last = text.split()[-1].lower().rstrip(".,;:!?\"\u201d")
    if last in DANGLERS:
        return True, "dangling_function_word"
    return False, "ok"


def main():
    p = argparse.ArgumentParser(
        description="QLattice-augmented scorer for GPT-2 synthetic posts (v2.1)"
    )
    p.add_argument("--in_file", "--in_jsonl", dest="in_file", required=True)
    p.add_argument("--out_file", "--out_final", dest="out_file", required=True)
    p.add_argument("--out_scored", default=None,
                   help="(legacy compat) optional copy of output")
    p.add_argument("--target", default=None, help="(legacy compat, ignored)")
    p.add_argument("--qlattice_model", default=None)
    p.add_argument("--pca_models", default=None)
    p.add_argument("--rejection_log",
                   default="reports/score_rejection_log.jsonl")
    p.add_argument("--strict_truncation", action="store_true",
                   help="(default off) drop any truncated record")
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

    rejections: list = []
    n_in = n_out = n_dropped = n_truncated_kept = 0
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.in_file, "r", encoding="utf-8") as fin, \
         open(args.out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_dropped += 1
                rejections.append({
                    "card_id": "unknown",
                    "stage": "score",
                    "verdict": "reject",
                    "reason": "json_decode_error",
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                continue

            scored = score_record(rec, qlattice_model, pca_models)
            drop, why = should_drop(scored)
            if not drop and args.strict_truncation and \
               scored.get("truncation_flag", {}).get("is_truncated"):
                drop = True
                why = "strict_truncation_" + scored["truncation_flag"]["reason"]
            if drop:
                n_dropped += 1
                rejections.append({
                    "card_id": rec.get("id", "unknown"),
                    "stage": "score",
                    "verdict": "reject",
                    "reason": why,
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                continue

            if scored.get("truncation_flag", {}).get("is_truncated"):
                n_truncated_kept += 1
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
            n_out += 1

    Path(args.rejection_log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.rejection_log, "w", encoding="utf-8") as f:
        for r in rejections:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.out_scored:
        Path(args.out_scored).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "rb") as src, \
             open(args.out_scored, "wb") as dst:
            dst.write(src.read())

    logger.info("=" * 60)
    logger.info("Score results: in=%d, passed=%d, dropped=%d, "
                "kept-but-flagged=%d", n_in, n_out, n_dropped, n_truncated_kept)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

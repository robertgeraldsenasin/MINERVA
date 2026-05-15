#!/usr/bin/env python3
"""Sanity-check detector predictions on template cards (internal consensus metric only)."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def predict_from_score(p_fake: float, threshold: float = 0.5,
                       uncertain_band: float = 0.05) -> str:
    """Convert a fake-probability score to a verdict label.

    Within +/- uncertain_band of the 0.5 midpoint, the label is
    UNCERTAIN. This matches how the detection pipeline treats
    border-line confidence scores.
    """
    if p_fake >= 0.5 + uncertain_band:
        return "FAKE"
    elif p_fake <= 0.5 - uncertain_band:
        return "REAL"
    else:
        return "UNCERTAIN"


def per_detector_accuracy(cards: list, detector_key: str,
                          threshold: float = 0.5) -> dict:
    """Compute accuracy / precision / recall / F1 for one detector."""
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    for card in cards:
        gold = card.get("verdict")
        if gold not in {"FAKE", "REAL", "UNCERTAIN"}:
            continue
        score = card.get("detectors", {}).get(detector_key)
        if score is None:
            continue
        pred = predict_from_score(score, threshold)
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1
        total += 1
    if total == 0:
        return {"accuracy": None, "n": 0}
    accuracy = correct / total

    # Per-class precision/recall (FAKE as the positive class)
    tp = confusion["FAKE"]["FAKE"]
    fp = confusion["REAL"]["FAKE"] + confusion["UNCERTAIN"]["FAKE"]
    fn = confusion["FAKE"]["REAL"] + confusion["FAKE"]["UNCERTAIN"]
    tn = confusion["REAL"]["REAL"] + confusion["UNCERTAIN"]["UNCERTAIN"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "n": total,
        "accuracy": round(accuracy * 100, 2),
        "precision_fake": round(precision * 100, 2),
        "recall_fake": round(recall * 100, 2),
        "f1_fake": round(f1 * 100, 2),
        "confusion_matrix": {
            gold: dict(preds) for gold, preds in confusion.items()
        },
    }


def per_strata_accuracy(cards: list, strata_key: str,
                        detector_key: str = "p_ensemble_fake",
                        threshold: float = 0.5) -> dict:
    """Accuracy of the ensemble detector, broken down by a strata
    (e.g., tier or tactic).
    """
    by_stratum = defaultdict(lambda: {"correct": 0, "total": 0})
    for card in cards:
        gold = card.get("verdict")
        if gold not in {"FAKE", "REAL", "UNCERTAIN"}:
            continue
        # Resolve stratum
        if strata_key in {"tier", "tactic"}:
            stratum = card.get("provenance", {}).get(strata_key, "unknown")
        else:
            stratum = card.get(strata_key, "unknown")
        score = card.get("detectors", {}).get(detector_key)
        if score is None:
            continue
        pred = predict_from_score(score, threshold)
        by_stratum[stratum]["total"] += 1
        if gold == pred:
            by_stratum[stratum]["correct"] += 1
    return {
        s: {
            "n": v["total"],
            "accuracy": round(100 * v["correct"] / v["total"], 2)
                       if v["total"] > 0 else None,
        }
        for s, v in by_stratum.items()
    }


def main():
    p = argparse.ArgumentParser(
        description="v2.6-final detector validation on template cards"
    )
    p.add_argument("--pool_file", required=True,
                   help="Curated pool JSON (output of script 24)")
    p.add_argument("--report_out",
                   default="reports/detector_validation_report.json")
    p.add_argument("--markdown_out",
                   default="reports/detector_validation_summary.md")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    payload = json.load(open(args.pool_file, encoding="utf-8"))
    if isinstance(payload, dict) and "cards" in payload:
        cards = payload["cards"]
    else:
        cards = payload
    logger.info("Loaded %d cards from %s", len(cards), args.pool_file)

    # Per-detector accuracy
    detectors = ["p_roberta_fake", "p_distil_fake", "p_degnn_fake",
                 "p_ensemble_fake"]
    detector_metrics = {}
    for d in detectors:
        detector_metrics[d] = per_detector_accuracy(
            cards, d, threshold=args.threshold)

    # Per-tier accuracy (ensemble only)
    tier_metrics = per_strata_accuracy(cards, "tier",
                                       detector_key="p_ensemble_fake",
                                       threshold=args.threshold)
    tactic_metrics = per_strata_accuracy(cards, "tactic",
                                         detector_key="p_ensemble_fake",
                                         threshold=args.threshold)
    candidate_metrics = per_strata_accuracy(cards, "candidate",
                                            detector_key="p_ensemble_fake",
                                            threshold=args.threshold)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        # The pool was constructed by filtering on these same detectors'
        # consensus, so 100% accuracy here is mathematically guaranteed.
        # This is an internal-consistency check, not generalization F1.
        # For real generalization metrics, see reports/holdout_detector_eval.json
        # (script 37, requires hand-labeled holdout CSV).
        "interpretation": (
            "INTERNAL-CONSISTENCY CHECK ONLY. The pool was curated to "
            "detector consensus, so 100% accuracy here is the expected, "
            "trivial outcome — NOT a generalization metric. See "
            "reports/holdout_detector_eval.json for off-distribution F1."
        ),
        "metric_kind": "internal_consensus",  # vs "off_distribution"
        "pool_file": args.pool_file,
        "n_cards": len(cards),
        "threshold": args.threshold,
        "verdict_distribution": dict(Counter(c.get("verdict") for c in cards)),
        "per_detector_metrics": detector_metrics,
        "per_tier_metrics": tier_metrics,
        "per_tactic_metrics": tactic_metrics,
        "per_candidate_metrics": candidate_metrics,
        "notes": [
            "Scores are template-encoded, not a fresh inference.",
            "For full validation, run scripts/15_evaluate_detectors.py "
            "on a held-out test set.",
            "Modirrousta-Galian & Higham (2023) emphasize per-tier "
            "calibration; per-tier accuracy here informs the "
            "credible-card quota.",
        ],
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Markdown summary
    md = ["# Detector Validation on v2.6-final Template Cards\n",
          f"**Generated:** {report['ts']}",
          f"**Pool:** `{args.pool_file}` ({len(cards)} cards)",
          f"**Threshold:** {args.threshold}\n",
          "## Per-detector accuracy\n",
          "| Detector | n | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) |",
          "|---|---|---|---|---|---|"]
    for d, m in detector_metrics.items():
        if m.get("n"):
            md.append(
                f"| `{d}` | {m['n']} | {m['accuracy']}% | "
                f"{m['precision_fake']}% | {m['recall_fake']}% | "
                f"{m['f1_fake']}% |"
            )

    md.append("\n## Ensemble accuracy by tier\n")
    md.append("| Tier | n | Accuracy |")
    md.append("|---|---|---|")
    for tier, m in tier_metrics.items():
        md.append(f"| {tier} | {m['n']} | {m.get('accuracy')}% |")

    md.append("\n## Ensemble accuracy by tactic\n")
    md.append("| Tactic | n | Accuracy |")
    md.append("|---|---|---|")
    for tactic, m in sorted(tactic_metrics.items()):
        md.append(f"| {tactic} | {m['n']} | {m.get('accuracy')}% |")

    md.append("\n## Ensemble accuracy by candidate\n")
    md.append("| Candidate | n | Accuracy |")
    md.append("|---|---|---|")
    for cand, m in sorted(candidate_metrics.items()):
        md.append(f"| {cand} | {m['n']} | {m.get('accuracy')}% |")

    md.append("\n## Notes\n")
    for n in report["notes"]:
        md.append(f"- {n}")

    Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
    open(args.markdown_out, "w", encoding="utf-8").write("\n".join(md) + "\n")

    logger.info("=" * 60)
    logger.info("Detector validation complete (v2.6-final)")
    logger.info("  Cards validated   : %d", len(cards))
    for d, m in detector_metrics.items():
        if m.get("n"):
            logger.info("  %-24s acc=%s%% F1(fake)=%s%% (n=%d)",
                        d, m["accuracy"], m["f1_fake"], m["n"])
    logger.info("=" * 60)
    logger.info("Report   -> %s", args.report_out)
    logger.info("Markdown -> %s", args.markdown_out)


if __name__ == "__main__":
    main()

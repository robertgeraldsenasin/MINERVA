#!/usr/bin/env python3
"""
18_verdict_explain.py  (REFACTORED v2.0)
========================================

Convert scored generations into Unity cards with content-aware,
varied, faithful explanations.

WHAT CHANGED FROM v1
--------------------
Legacy v1 (current repo):
  Every card got the same template: "Verdict: REAL/FAKE … The decision
  comes from the stored Qlattice equation applied to detector/embedding
  features." Per-card variation was zero. Signals were PCA-component
  jargon ("RoBERTa semantic component 0 deviates from baseline") that
  no SHS student can learn from. The 992-card unity_cards.json had 992
  identical explanation summaries — modulo the verdict word.

v2.0 changes:
  * Indicator extraction (12-cue taxonomy) replaces PCA-component prose.
  * Response bank (56 entries, 3 tiers) replaces single template.
  * Explanation is content-aware: only fired indicators get phrased.
  * Each card carries fired_indicators + bank_refs for audit.
  * Output validates against UnityCard pydantic schema.

PIPELINE POSITION
-----------------
Reads:  generated/gpt2_synthetic_final_{fake,real,both}.jsonl
Writes: generated/unity_cards.json
Next:   21_balance_unity_cards.py

CITATIONS (defended in MASTER_CODEBOOK.md):
  Barzilai & Stadtler (2025); Khosravi et al. (2022); Athira et al.
  (2023); Longo et al. (2024); Liu, Ye & Li (2024); Roozenbeek &
  van der Linden (2019); Caulfield (2019); Modirrousta-Galian &
  Higham (2023); Christensen et al. (2022).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_indicators import (extract_indicators, fired_codes,
                                 indicator_summary_for_card)
from minerva_response_bank import (assemble_explanation, BANK_VERSION,
                                    bank_hash, tier_for_card_index)
from minerva_schemas import (DetectorBlock, ExplanationBlock,
                              IndicatorDetail, ProvenanceBlock,
                              QlatticeBlock, ThemeFlags, UnityCard,
                              IndicatorPhrase)

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _difficulty(p_fake: float) -> str:
    """Bin p_fake into easy/medium/hard, matching the original spec."""
    margin = abs(p_fake - 0.5)
    if margin >= 0.35:
        return "easy"
    if margin >= 0.15:
        return "medium"
    return "hard"


def _build_card_dict(
    raw_record: dict,
    *,
    card_index: int,
    seed: int,
    git_sha: str,
) -> dict | None:
    """
    Convert a single scored generation record into the v2.0 unity-card dict.

    Returns None if the record cannot be promoted (e.g. no text).
    """
    text = raw_record.get("text") or raw_record.get("generated_text") or ""
    if not text or len(text.strip()) < 20:
        return None

    target_label = raw_record.get("target_label") or raw_record.get("target") or "fake"
    if target_label not in ("fake", "real"):
        target_label = "fake"

    # v2.1: read p_fake from any of the documented schemas
    p_fake_candidates = [
        raw_record.get("p_fake"),
        raw_record.get("fake_probability"),
        raw_record.get("ensemble_p_fake"),
        (raw_record.get("detectors") or {}).get("p_ensemble_fake"),
        (raw_record.get("qlattice") or {}).get("score"),
        # Last-resort: average top-level detector probabilities
        (
            sum(filter(None, [
                raw_record.get("p_roberta_fake"),
                raw_record.get("p_distil_fake"),
                raw_record.get("p_degnn_fake"),
            ])) / max(
                sum(1 for v in [
                    raw_record.get("p_roberta_fake"),
                    raw_record.get("p_distil_fake"),
                    raw_record.get("p_degnn_fake"),
                ] if v is not None),
                1,
            )
        ) if any(raw_record.get(k) is not None for k in
                  ("p_roberta_fake", "p_distil_fake", "p_degnn_fake")) else None,
    ]
    p_fake = float(next((p for p in p_fake_candidates if p is not None), 0.5))
    fake_pct = max(0.0, min(100.0, p_fake * 100))
    cred_pct = 100.0 - fake_pct

    # Verdict policy: 0..0.4 -> REAL, 0.4..0.6 -> UNCERTAIN, 0.6..1 -> FAKE
    if p_fake >= 0.6:
        verdict = "FAKE"
    elif p_fake <= 0.4:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    # Indicators (the new pedagogy core)
    indicator_summary = indicator_summary_for_card(text)
    fired = indicator_summary["fired_indicators"]

    # v2.2 + v2.4: VERDICT-RULE ALIGNMENT GUARD
    # If the symbolic-regression score says REAL but indicator rules
    # find >=3 misinformation cues, the two interpretation paths
    # disagree. v2.4 raised the threshold from 2 to 3 because GPT-2
    # output is naturally noisy (MISS/FAB fire on most outputs because
    # the synthetic posts lack URLs by construction); demoting at >=2
    # was too aggressive and produced only 4 REAL cards in the v2.3
    # run instead of the ~85 needed for the credible-card quota.
    if verdict == "REAL" and len(fired) >= 3:
        verdict = "UNCERTAIN"
        # Adjust fake_pct upward into the uncertainty band so the
        # displayed percentage doesn't contradict the new verdict.
        fake_pct = max(fake_pct, 41.0)
        cred_pct = 100.0 - fake_pct
    # Symmetric guard: if FAKE but zero indicators fired, the
    # detectors found something the rules didn't surface — still call
    # it FAKE but flag for audit.
    alignment_flag = "ok"
    if verdict == "FAKE" and len(fired) == 0:
        alignment_flag = "fake_no_indicators"
    elif verdict == "UNCERTAIN" and p_fake <= 0.4:
        alignment_flag = "demoted_real_to_uncertain"

    # Tier banding
    tier = tier_for_card_index(card_index)

    # Card id
    card_id = (
        raw_record.get("id")
        or f"C-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{card_index:06d}"
    )

    # Per-card stable seed for explanation rotation
    seed_str = f"{card_id}|{seed}"

    # Detector block — fall back gracefully if upstream didn't provide
    det_in = raw_record.get("detectors") or {}
    detectors = {
        "p_roberta_fake": float(det_in.get("p_roberta_fake", p_fake)),
        "p_distil_fake": float(det_in.get("p_distil_fake", p_fake)),
        "p_degnn_fake": float(det_in.get("p_degnn_fake", p_fake)),
        "p_ensemble_fake": float(det_in.get("p_ensemble_fake", p_fake)),
    }

    # QLattice block (carry through if present, else stub)
    qb_in = raw_record.get("qlattice") or {}
    qlattice = {
        "score": float(qb_in.get("score", p_fake)),
        "threshold": float(qb_in.get("threshold", 0.5)),
        "direction": str(qb_in.get("direction", ">=")),
        "margin": float(qb_in.get("margin", p_fake - 0.5)),
        "pred": int(qb_in.get("pred", 1 if p_fake >= 0.5 else 0)),
        "equation": str(qb_in.get("equation", "")),
        "top_factors": list(qb_in.get("top_factors", [])),
    }

    heuristics = raw_record.get("heuristics") or {}

    # Theme flags — script 23 may overwrite; for now compute baseline
    from minerva_filters import keyword_score
    theme_score, _ = keyword_score(text)
    theme_flags = {
        "is_electoral": theme_score >= 0.55,
        "electoral_score": theme_score,
        "is_neutral_volume": False,
        "classifier_label": "keyword_baseline",
    }

    # Build explanation
    explanation = assemble_explanation(
        fired_indicators=fired,
        verdict=verdict,
        fake_likelihood_percent=fake_pct,
        seed_str=seed_str,
        tier=tier,
        candidate_name=None,  # script 22 will fill candidate context
    )

    # Pseudonymisation candidate is filled by script 22; default NONE here
    candidate = "NONE"

    provenance = {
        "seed": seed,
        "git_sha": git_sha,
        "bank_version": BANK_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.2.0",
        "script_chain": ["13", "18"],
        "alignment_flag": alignment_flag,
    }

    card = {
        "id": card_id,
        "candidate": candidate,
        "text": text.strip(),
        "target_label": target_label,
        "verdict": verdict,
        "fake_likelihood_percent": fake_pct,
        "credibility_percent": cred_pct,
        "difficulty_bin": _difficulty(p_fake),
        "fired_indicators": fired,
        "indicator_details": {
            c: indicator_summary["indicator_details"][c]
            for c in indicator_summary["indicator_details"]
        },
        "named_features": indicator_summary["named_features"],
        "qlattice": qlattice,
        "detectors": detectors,
        "heuristics": heuristics,
        "theme_flags": theme_flags,
        "explanation": explanation,
        "provenance": provenance,
        "metadata": raw_record.get("metadata") or {},
    }
    return card


def _validate_or_log(card_dict: dict, audit_log: list) -> dict | None:
    """Validate a card against the schema; log + drop if invalid."""
    try:
        UnityCard.model_validate(card_dict)
        return card_dict
    except Exception as e:
        audit_log.append({
            "card_id": card_dict.get("id", "unknown"),
            "stage": "schema_validation",
            "verdict": "reject",
            "reason": str(e)[:300],
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return None


def main():
    p = argparse.ArgumentParser(description="Refactored verdict-explain (v2.0)")
    p.add_argument("--in_file", required=True,
                   help="JSONL input (e.g. gpt2_synthetic_final_both.jsonl)")
    p.add_argument("--out_file", required=True,
                   help="JSON output (unity_cards.json)")
    p.add_argument("--audit_out", default=None,
                   help="JSONL audit log of rejected cards")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    git_sha = _git_sha()
    logger.info("Bank version=%s hash=%s git=%s seed=%d",
                BANK_VERSION, bank_hash(), git_sha, args.seed)

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cards: list[dict] = []
    audit_log: list = []
    seen_ids: set[str] = set()

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: %s", line_no, e)
                continue
            card = _build_card_dict(rec, card_index=len(cards),
                                     seed=args.seed, git_sha=git_sha)
            if card is None:
                continue
            # Deduplicate by id (legacy data sometimes has duplicates)
            if card["id"] in seen_ids:
                continue
            valid = _validate_or_log(card, audit_log)
            if valid is None:
                continue
            seen_ids.add(card["id"])
            cards.append(valid)

    # Write output
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d cards to %s", len(cards), out_path)

    # Audit log
    if args.audit_out:
        ap = Path(args.audit_out)
        ap.parent.mkdir(parents=True, exist_ok=True)
        with ap.open("w", encoding="utf-8") as f:
            for entry in audit_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Wrote %d audit entries to %s", len(audit_log), ap)

    # Cohort-wide explanation diversity report
    explanations = [c["explanation"]["summary"] for c in cards]
    unique = len(set(explanations))
    logger.info("Cohort diversity: %d unique summaries / %d total (%.1f%%)",
                unique, len(cards),
                100.0 * unique / max(len(cards), 1))


if __name__ == "__main__":
    main()

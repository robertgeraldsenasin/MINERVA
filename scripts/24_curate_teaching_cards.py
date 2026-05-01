#!/usr/bin/env python3
"""
24_curate_teaching_cards.py  (REFACTORED v2.0)
==============================================

Promote themed unity cards into the teaching deck (story_cards.json),
applying:
  * Difficulty banding (novice → proficient → advanced) by card index
    (Dehghanzadeh et al. 2024; Almaki et al. 2024).
  * Bank-version-stamped explanations.
  * Mandatory inclusion of credible cards to counter conservative-bias
    drift (Modirrousta-Galian & Higham 2023).
  * Day assignment (1..N) for the daily-cycle gameplay.

WHAT CHANGED FROM v1
--------------------
v1 wrote story_cards.json with the same static template explanation
that 18_verdict_explain produced. It also sometimes promoted off-theme
posts and truncated text.

v2.0:
  * Tier banding tied to within-day position.
  * Explicit "credible-counter" pairing: each fake card is linked to
    a credible card the player has already seen, so VERIdict can
    show side-by-side comparison (Barzilai & Stadtler 2025).
  * Schema validation; reject + log invalid promotions.
  * Audit report.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_response_bank import (assemble_explanation, BANK_VERSION,
                                    bank_hash, tier_for_card_index)
from minerva_candidates import REGISTRY
from minerva_schemas import StoryCard

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--reject_out", default="generated/story_cards_rejected.json")
    p.add_argument("--report_out", default="reports/story_cards_curation_report.json")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--cards_per_day", type=int, default=8)
    p.add_argument("--min_credible_per_day", type=int, default=3,
                   help="At least this many REAL cards per day "
                        "(Modirrousta-Galian & Higham 2023 mandate)")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    rng = random.Random(args.seed)

    cards = json.load(open(args.in_file, encoding="utf-8"))
    fakes = [c for c in cards if c.get("verdict") == "FAKE"]
    reals = [c for c in cards if c.get("verdict") == "REAL"]
    uncertains = [c for c in cards if c.get("verdict") == "UNCERTAIN"]
    rng.shuffle(fakes)
    rng.shuffle(reals)
    rng.shuffle(uncertains)

    promoted: list[dict] = []
    rejected: list[dict] = []
    target_total = args.days * args.cards_per_day

    fake_idx = real_idx = unc_idx = 0
    for day in range(1, args.days + 1):
        day_card_count = 0
        # Reserve credible quota first
        for _ in range(args.min_credible_per_day):
            if real_idx >= len(reals):
                break
            promoted_card = _promote(reals[real_idx], day, len(promoted), args.seed)
            if promoted_card:
                promoted.append(promoted_card)
            else:
                rejected.append({"id": reals[real_idx].get("id"),
                                  "reason": "schema_validation_real"})
            real_idx += 1
            day_card_count += 1
        # Fill rest with fakes (and a sprinkle of uncertains)
        while day_card_count < args.cards_per_day:
            if rng.random() < 0.15 and unc_idx < len(uncertains):
                src = uncertains[unc_idx]; unc_idx += 1
            elif fake_idx < len(fakes):
                src = fakes[fake_idx]; fake_idx += 1
            elif real_idx < len(reals):
                src = reals[real_idx]; real_idx += 1
            elif unc_idx < len(uncertains):
                src = uncertains[unc_idx]; unc_idx += 1
            else:
                break
            promoted_card = _promote(src, day, len(promoted), args.seed)
            if promoted_card:
                promoted.append(promoted_card)
            else:
                rejected.append({"id": src.get("id"),
                                  "reason": "schema_validation_fake"})
            day_card_count += 1

    # Cross-link each FAKE to the most recent REAL the player has seen
    # (the credible-counter pairing for VERIdict)
    last_credible_id: str | None = None
    for c in promoted:
        if c["verdict"] == "REAL":
            last_credible_id = c["id"]
        elif c["verdict"] == "FAKE" and last_credible_id:
            c["explanation"]["credible_counter_card_id"] = last_credible_id

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(promoted, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    Path(args.reject_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rejected, open(args.reject_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    report = {
        "input_total": len(cards),
        "promoted": len(promoted),
        "rejected": len(rejected),
        "days": args.days,
        "cards_per_day": args.cards_per_day,
        "fake_count": sum(1 for c in promoted if c["verdict"] == "FAKE"),
        "real_count": sum(1 for c in promoted if c["verdict"] == "REAL"),
        "uncertain_count": sum(1 for c in promoted if c["verdict"] == "UNCERTAIN"),
        "candidate_distribution": {
            code: sum(1 for c in promoted if c.get("candidate") == code)
            for code in ["C-RM", "C-IB", "C-JS", "NONE"]
        },
        "tier_distribution": _tier_dist(promoted),
        "indicator_coverage": _indicator_coverage(promoted),
        "explanation_diversity": _diversity(promoted),
        "bank_version": BANK_VERSION,
        "bank_hash": bank_hash(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"), indent=2)
    logger.info("Curation report: %s", json.dumps(report, indent=2)[:1500])


def _promote(card: dict, day: int, idx_in_run: int, seed: int) -> dict | None:
    """Stamp tier + day, validate against StoryCard schema."""
    tier = tier_for_card_index(idx_in_run)
    cand_code = card.get("candidate", "NONE")
    cand = REGISTRY.get(cand_code)
    cand_name = cand.name if cand else None

    # Re-assemble explanation at the correct tier
    card["explanation"] = assemble_explanation(
        fired_indicators=card.get("fired_indicators", []),
        verdict=card.get("verdict", "UNCERTAIN"),
        fake_likelihood_percent=card.get("fake_likelihood_percent", 50.0),
        seed_str=f"{card['id']}|{seed}",
        tier=tier,
        candidate_name=cand_name,
    )
    card["day"] = day
    card.setdefault("provenance", {}).setdefault("script_chain", []).append("24")

    try:
        StoryCard.model_validate(card)
        return card
    except Exception as e:
        logger.warning("Card %s failed schema: %s", card.get("id"), str(e)[:200])
        return None


def _tier_dist(cards: list[dict]) -> dict:
    out: dict = {"novice": 0, "proficient": 0, "advanced": 0}
    for c in cards:
        t = c.get("explanation", {}).get("tier", "novice")
        out[t] = out.get(t, 0) + 1
    return out


def _indicator_coverage(cards: list[dict]) -> dict:
    counts: dict = {}
    for c in cards:
        for ind in c.get("fired_indicators", []):
            counts[ind] = counts.get(ind, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _diversity(cards: list[dict]) -> dict:
    summaries = [c.get("explanation", {}).get("summary", "") for c in cards]
    unique = len(set(summaries))
    return {
        "total_summaries": len(summaries),
        "unique_summaries": unique,
        "diversity_pct": round(100.0 * unique / max(len(summaries), 1), 1),
    }


if __name__ == "__main__":
    main()

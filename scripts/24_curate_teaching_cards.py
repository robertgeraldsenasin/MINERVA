#!/usr/bin/env python3
"""Final teaching-card curation: indicator coverage, difficulty balance, pedagogical structure."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_schemas import StoryCard
from minerva_response_bank import BANK_VERSION, bank_hash

logger = logging.getLogger(__name__)


def _promote(card: dict, pool_index: int, seed: int) -> dict | None:
    """Validate a unity-card and return its story-card form.

    Story cards in v2.3 do NOT carry a 'day' field anymore — that's
    assigned by the per-user draw script (28) at runtime. The pool is
    day-agnostic.
    """
    out = dict(card)
    # Strip any pre-existing day field (from v2.x runs)
    out.pop("day", None)
    out["pool_index"] = pool_index
    try:
        StoryCard.model_validate(out)
        return out
    except Exception as e:
        logger.debug("Card %s rejected by schema: %s",
                     card.get("id"), str(e)[:160])
        return None


def main():
    p = argparse.ArgumentParser(
        description="v2.3 — curate a POOL of teaching cards (per-user decks "
                    "are drawn separately by script 28)."
    )
    p.add_argument("--in_file", required=True,
                   help="Input: unity_cards_themed.json")
    p.add_argument("--out_file", required=True,
                   help="Output: unity_cards_pool.json (the POOL)")
    p.add_argument("--reject_out", default="generated/pool_rejected.json")
    p.add_argument("--report_out",
                   default="reports/curation_report.json")
    p.add_argument("--target_pool_size", type=int, default=500,
                   help="Target size of the POOL (default 500). "
                        "Each player will draw days*cards_per_day cards "
                        "from this pool.")
    p.add_argument("--min_real_share", type=float, default=0.35,
                   help="Minimum share of REAL cards in the pool "
                        "(default 35%%). Modirrousta-Galian & Higham 2023 "
                        "mandate ensures students see enough credible "
                        "examples to avoid over-suspicion drift.")
    # Legacy compat: keep --days/--cards_per_day flags so old scripts/CLI
    # keep working, but they're informational only in v2.3.
    p.add_argument("--days", type=int, default=7,
                   help="(informational) Days per player deck — used by "
                        "downstream draw script 28")
    p.add_argument("--cards_per_day", type=int, default=8,
                   help="(informational) Cards per day per player")
    p.add_argument("--min_credible_per_day", type=int, default=3,
                   help="(informational) Carried into pool metadata for "
                        "the draw script")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    rng = random.Random(args.seed)

    cards = json.load(open(args.in_file, encoding="utf-8"))
    rng.shuffle(cards)

    # Bucket by verdict so we can enforce min_real_share
    fakes = [c for c in cards if c.get("verdict") == "FAKE"]
    reals = [c for c in cards if c.get("verdict") == "REAL"]
    uncertains = [c for c in cards if c.get("verdict") == "UNCERTAIN"]

    target = args.target_pool_size
    n_real_min = max(1, int(target * args.min_real_share))
    n_unc_max = max(1, int(target * 0.10))   # up to 10% UNCERTAIN
    n_fake_max = target - n_real_min - n_unc_max

    promoted: list[dict] = []
    rejected: list[dict] = []
    pool_index = 0

    # 1) Reserve REAL quota first
    for c in reals[:n_real_min]:
        pc = _promote(c, pool_index, args.seed)
        if pc:
            promoted.append(pc); pool_index += 1
        else:
            rejected.append({"id": c.get("id"), "reason": "schema_real"})

    # 2) Add UNCERTAIN cap
    for c in uncertains[:n_unc_max]:
        pc = _promote(c, pool_index, args.seed)
        if pc:
            promoted.append(pc); pool_index += 1
        else:
            rejected.append({"id": c.get("id"), "reason": "schema_unc"})

    # 3) Fill remaining slots with FAKE
    need = target - len(promoted)
    for c in fakes[:need]:
        pc = _promote(c, pool_index, args.seed)
        if pc:
            promoted.append(pc); pool_index += 1
        else:
            rejected.append({"id": c.get("id"), "reason": "schema_fake"})

    # 4) If still short, top up from any leftover REAL/UNCERTAIN
    leftover = (reals[n_real_min:] + uncertains[n_unc_max:]
                + fakes[need:])
    rng.shuffle(leftover)
    while len(promoted) < target and leftover:
        c = leftover.pop()
        pc = _promote(c, pool_index, args.seed)
        if pc:
            promoted.append(pc); pool_index += 1
        else:
            rejected.append({"id": c.get("id"), "reason": "schema_topup"})

    # Sanity: cross-link every FAKE card to a randomly sampled REAL
    # from the pool. Used by VERIdict for the credible-counter pairing
    # *within a player's session*. The pairing is approximate at the
    # pool level — script 28 may re-link based on actual draw order.
    real_ids = [c["id"] for c in promoted if c["verdict"] == "REAL"]
    if real_ids:
        for c in promoted:
            if c["verdict"] == "FAKE":
                c.setdefault("explanation", {})["credible_counter_card_id"] = \
                    rng.choice(real_ids)

    # Stamp pool-level metadata
    pool_metadata = {
        "pool_version": "2.3",
        "pool_size": len(promoted),
        "default_days_per_player": args.days,
        "default_cards_per_day": args.cards_per_day,
        "default_min_credible_per_day": args.min_credible_per_day,
        "fake_count":      sum(1 for c in promoted if c["verdict"] == "FAKE"),
        "real_count":      sum(1 for c in promoted if c["verdict"] == "REAL"),
        "uncertain_count": sum(1 for c in promoted if c["verdict"] == "UNCERTAIN"),
        "candidate_distribution": dict(Counter(
            c.get("candidate", "NONE") for c in promoted)),
        "indicator_coverage": dict(Counter(
            ind for c in promoted for ind in c.get("fired_indicators", []))),
        "tier_distribution": dict(Counter(
            c.get("explanation", {}).get("tier", "?") for c in promoted)),
        "explanation_diversity": {
            "total": len(promoted),
            "unique": len(set(c.get("explanation", {}).get("summary", "")
                              for c in promoted)),
        },
        "bank_version": BANK_VERSION,
        "bank_hash": bank_hash(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    # Build the pool file with metadata header so Unity can self-validate
    pool_doc = {"_metadata": pool_metadata, "cards": promoted}

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pool_doc, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    Path(args.reject_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rejected, open(args.reject_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pool_metadata,
              open(args.report_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Capacity estimate: how many distinct decks can the pool support?
    cards_per_player = args.days * args.cards_per_day
    capacity_strict = len(promoted) // cards_per_player
    expected_overlap_pct = 100.0 * cards_per_player / max(len(promoted), 1)

    logger.info("=" * 60)
    logger.info("Pool curation complete (v2.3)")
    logger.info("  pool_size        : %d", len(promoted))
    logger.info("  rejected         : %d", len(rejected))
    logger.info("  cards_per_player : %d (=%d days * %d cards)",
                cards_per_player, args.days, args.cards_per_day)
    logger.info("  max non-overlapping decks : %d", capacity_strict)
    logger.info("  expected pairwise overlap : ~%.1f%% (lower is better)",
                expected_overlap_pct)
    logger.info("  verdicts         : %s", dict(Counter(
        c["verdict"] for c in promoted)))
    logger.info("  candidates       : %s", dict(Counter(
        c.get("candidate", "NONE") for c in promoted)))
    logger.info("=" * 60)
    logger.info("Pool       -> %s", args.out_file)
    logger.info("Rejections -> %s", args.reject_out)
    logger.info("Report     -> %s", args.report_out)
    logger.info("Next step  -> run scripts/28_draw_user_deck.py")


if __name__ == "__main__":
    main()

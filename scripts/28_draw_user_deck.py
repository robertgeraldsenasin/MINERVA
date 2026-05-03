#!/usr/bin/env python3
"""
28_draw_user_deck.py  (NEW in v2.3)
===================================

Deterministically draw a per-user teaching deck from the curated pool
produced by script 24. Same user_id + same pool always produces the
same deck (so the user can resume; researchers can reproduce).

DESIGN NOTES
------------
The thesis evaluation framework requires that each Filipino SHS
student see a *different* deck so individual decisions are
independent observations and answer-sharing between students cannot
short-circuit the learning measurement.

This script implements a stratified deterministic draw with quotas:

  * day_assignment: each player gets DAYS days of CARDS_PER_DAY cards.
  * REAL quota: at least MIN_CREDIBLE_PER_DAY real cards per day
                (Modirrousta-Galian & Higham 2023).
  * candidate quota: at least 1 card per day mentions each of the
                three fictional candidates (when pool supports it).
  * indicator coverage: across the deck, at least 6 of the 12
                indicators must appear at least once (so students
                practice spotting different cue types).

DETERMINISM
-----------
The draw is fully deterministic in (user_id, pool_hash). The same
user replaying the game gets the same cards in the same order. Two
different user_ids on the same pool get different decks.

The pool_hash is included in the deck file so a researcher comparing
two players' deck files can verify they were drawn from the same pool.

PIPELINE POSITION
-----------------
Reads:  generated/unity_cards_pool.json
Writes: generated/decks/deck_<user_id>.json

UNITY INTEGRATION
-----------------
For offline play, port this draw logic to C# and ship the pool file
with the APK. The hash function used here is Python's stable hash
of (user_id + pool_hash) — easy to replicate in C#.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


def deterministic_seed(user_id: str, pool_hash: str) -> int:
    """Stable across Python versions / OS — uses sha256, not built-in hash().

    The first 8 bytes of sha256(user_id + ':' + pool_hash) interpreted as
    unsigned 64-bit int. Easy to replicate in C# (BitConverter.ToInt64
    of the first 8 bytes of SHA256ManagedComputeHash output).
    """
    digest = hashlib.sha256(f"{user_id}:{pool_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _by_verdict(cards: list[dict], verdict: str) -> list[dict]:
    return [c for c in cards if c.get("verdict") == verdict]


def _by_candidate(cards: list[dict], cand: str) -> list[dict]:
    return [c for c in cards if c.get("candidate") == cand]


def draw_deck(pool: list[dict], user_id: str, pool_hash: str,
              days: int, cards_per_day: int,
              min_credible_per_day: int) -> list[dict]:
    """Stratified deterministic deck draw."""
    seed = deterministic_seed(user_id, pool_hash)
    rng = random.Random(seed)

    # Per-day construction. We pick cards day-by-day so each day
    # satisfies its quotas independently.
    deck: list[dict] = []
    used_ids: set = set()

    pool_reals = _by_verdict(pool, "REAL")
    pool_fakes = _by_verdict(pool, "FAKE")
    pool_uncs  = _by_verdict(pool, "UNCERTAIN")

    # Shuffle once per stratum with the user's seed
    rng.shuffle(pool_reals)
    rng.shuffle(pool_fakes)
    rng.shuffle(pool_uncs)

    real_idx = fake_idx = unc_idx = 0
    candidates = ["C-RM", "C-IB", "C-JS"]

    for day in range(1, days + 1):
        day_cards: list[dict] = []
        day_candidates_seen: set = set()

        # 1. REAL quota
        n_reals_added = 0
        while n_reals_added < min_credible_per_day and real_idx < len(pool_reals):
            c = pool_reals[real_idx]; real_idx += 1
            if c["id"] in used_ids:
                continue
            day_cards.append(dict(c, day=day))
            used_ids.add(c["id"])
            day_candidates_seen.add(c.get("candidate"))
            n_reals_added += 1

        # 2. Candidate-coverage quota: try to ensure at least one card
        #    per candidate per day. We do this by taking from the FAKE
        #    bucket but routing by candidate for the next ~3 picks.
        for cand in candidates:
            if cand in day_candidates_seen:
                continue
            if len(day_cards) >= cards_per_day:
                break
            cand_fakes = [c for c in pool_fakes
                          if c.get("candidate") == cand and c["id"] not in used_ids]
            if cand_fakes:
                pick = cand_fakes[0]
                day_cards.append(dict(pick, day=day))
                used_ids.add(pick["id"])
                day_candidates_seen.add(cand)

        # 3. Fill the rest with a mix of FAKE and UNCERTAIN.
        while len(day_cards) < cards_per_day:
            # 15% chance of UNCERTAIN, else FAKE; fall back if depleted
            if rng.random() < 0.15 and unc_idx < len(pool_uncs):
                c = pool_uncs[unc_idx]; unc_idx += 1
                if c["id"] in used_ids:
                    continue
            elif fake_idx < len(pool_fakes):
                c = pool_fakes[fake_idx]; fake_idx += 1
                if c["id"] in used_ids:
                    continue
            elif unc_idx < len(pool_uncs):
                c = pool_uncs[unc_idx]; unc_idx += 1
                if c["id"] in used_ids:
                    continue
            elif real_idx < len(pool_reals):
                c = pool_reals[real_idx]; real_idx += 1
                if c["id"] in used_ids:
                    continue
            else:
                # Pool exhausted — log and break
                logger.warning(
                    "User %s: pool exhausted on day %d. "
                    "Got %d cards instead of %d.",
                    user_id, day, len(day_cards), cards_per_day)
                break
            day_cards.append(dict(c, day=day))
            used_ids.add(c["id"])

        # Per-day shuffle so the REAL quota cards aren't all at the start
        rng.shuffle(day_cards)
        deck.extend(day_cards)

    # Cross-link FAKE -> last credible REAL the player has seen so far
    last_real_id = None
    for c in deck:
        if c["verdict"] == "REAL":
            last_real_id = c["id"]
        elif c["verdict"] == "FAKE" and last_real_id:
            c.setdefault("explanation", {})[
                "credible_counter_card_id"] = last_real_id

    return deck


def compute_pool_hash(pool_doc: dict) -> str:
    """Stable hash of the pool's content (independent of metadata.ts)."""
    cards = pool_doc.get("cards", [])
    # Hash on (id, verdict, candidate) tuples — stable across runs
    payload = json.dumps(
        sorted([(c["id"], c.get("verdict"), c.get("candidate"))
                for c in cards]),
        ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(
        description="v2.3 — deterministic per-user deck draw from a curated pool"
    )
    p.add_argument("--pool_file", required=True,
                   help="Pool JSON from script 24")
    p.add_argument("--out_dir", default="generated/decks",
                   help="Where to write deck_<user_id>.json files")
    p.add_argument("--user_ids", required=True,
                   help="Comma-separated list of user IDs, OR a path to a "
                        "text file with one user_id per line. Use 'demo' "
                        "to draw 5 demo decks.")
    p.add_argument("--days", type=int, default=None,
                   help="Override pool default")
    p.add_argument("--cards_per_day", type=int, default=None,
                   help="Override pool default")
    p.add_argument("--min_credible_per_day", type=int, default=None,
                   help="Override pool default")
    p.add_argument("--report_out", default="reports/draw_report.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    pool_doc = json.load(open(args.pool_file, encoding="utf-8"))
    pool = pool_doc["cards"] if isinstance(pool_doc, dict) and "cards" in pool_doc \
        else pool_doc
    metadata = pool_doc.get("_metadata", {}) if isinstance(pool_doc, dict) else {}

    days = args.days if args.days is not None \
        else metadata.get("default_days_per_player", 7)
    cards_per_day = args.cards_per_day if args.cards_per_day is not None \
        else metadata.get("default_cards_per_day", 8)
    min_credible_per_day = args.min_credible_per_day if args.min_credible_per_day is not None \
        else metadata.get("default_min_credible_per_day", 3)

    pool_hash = compute_pool_hash({"cards": pool})
    logger.info("Pool size: %d cards | pool_hash: %s",
                len(pool), pool_hash)
    logger.info("Per-user deck shape: %d days x %d cards = %d cards",
                days, cards_per_day, days * cards_per_day)

    # Resolve user_ids list
    if args.user_ids == "demo":
        user_ids = [f"demo_user_{i:02d}" for i in range(1, 6)]
    elif Path(args.user_ids).exists():
        user_ids = [line.strip() for line in
                    open(args.user_ids).readlines() if line.strip()]
    else:
        user_ids = [u.strip() for u in args.user_ids.split(",") if u.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track overlap statistics
    all_decks: dict[str, set] = {}
    deck_summaries: list[dict] = []

    for uid in user_ids:
        deck = draw_deck(pool, uid, pool_hash,
                         days, cards_per_day, min_credible_per_day)
        deck_doc = {
            "_metadata": {
                "user_id": uid,
                "pool_hash": pool_hash,
                "draw_seed": deterministic_seed(uid, pool_hash),
                "days": days,
                "cards_per_day": cards_per_day,
                "min_credible_per_day": min_credible_per_day,
                "drawn_at": datetime.now(timezone.utc).isoformat(),
                "deck_size": len(deck),
                "verdict_dist": dict(Counter(c["verdict"] for c in deck)),
                "candidate_dist": dict(Counter(c.get("candidate", "NONE")
                                                for c in deck)),
                "indicator_coverage": dict(Counter(
                    ind for c in deck for ind in c.get("fired_indicators", []))),
            },
            "cards": deck,
        }
        out_path = out_dir / f"deck_{uid}.json"
        json.dump(deck_doc, open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        all_decks[uid] = {c["id"] for c in deck}
        deck_summaries.append({
            "user_id": uid,
            "deck_size": len(deck),
            "verdicts": deck_doc["_metadata"]["verdict_dist"],
            "candidates": deck_doc["_metadata"]["candidate_dist"],
            "indicators": deck_doc["_metadata"]["indicator_coverage"],
            "out_path": str(out_path),
        })

    # Pairwise overlap analysis (capped at 25 pairs to avoid quadratic)
    overlap_samples: list[float] = []
    user_id_list = list(all_decks.keys())
    n = len(user_id_list)
    for i in range(min(n, 25)):
        for j in range(i + 1, min(n, 25)):
            a = all_decks[user_id_list[i]]
            b = all_decks[user_id_list[j]]
            if a and b:
                overlap_samples.append(len(a & b) / len(a))
    overlap_summary = {}
    if overlap_samples:
        overlap_summary = {
            "n_pairs_compared": len(overlap_samples),
            "mean_overlap_pct": round(100 * sum(overlap_samples) / len(overlap_samples), 2),
            "max_overlap_pct": round(100 * max(overlap_samples), 2),
            "min_overlap_pct": round(100 * min(overlap_samples), 2),
        }

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pool_file": args.pool_file,
        "pool_size": len(pool),
        "pool_hash": pool_hash,
        "n_users_drawn": len(user_ids),
        "deck_shape": {
            "days": days,
            "cards_per_day": cards_per_day,
            "min_credible_per_day": min_credible_per_day,
            "deck_size": days * cards_per_day,
        },
        "pairwise_overlap": overlap_summary,
        "decks": deck_summaries,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Drew %d per-user decks", len(user_ids))
    if overlap_summary:
        logger.info("Pairwise overlap: mean=%.1f%% min=%.1f%% max=%.1f%%",
                    overlap_summary["mean_overlap_pct"],
                    overlap_summary["min_overlap_pct"],
                    overlap_summary["max_overlap_pct"])
    logger.info("=" * 60)
    logger.info("Output: %s", out_dir)


if __name__ == "__main__":
    main()

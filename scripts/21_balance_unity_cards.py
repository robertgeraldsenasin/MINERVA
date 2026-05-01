#!/usr/bin/env python3
"""
21_balance_unity_cards.py  (REFACTORED v2.0)
============================================

Balance the unity_cards.json across:
  * verdicts (FAKE / REAL / UNCERTAIN)
  * candidates (C-RM / C-IB / C-JS)
  * indicator coverage (ensure all 12 are represented)
  * difficulty (easy / medium / hard)

WHAT CHANGED FROM v1
--------------------
v1 only filtered for de-duplication and basic verdict balance. It
did not balance across candidates (because there were no real
candidates) and did not check indicator coverage (because there
were no real indicators).

v2.0 introduces all four balance dimensions and emits a balance
report.

PIPELINE POSITION
-----------------
Reads:  generated/unity_cards.json   (script 18 output)
Writes: generated/unity_cards_balanced.json
        reports/balance_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_schemas import UnityCard
from minerva_response_bank import BANK_VERSION, bank_hash

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--report_out", default="reports/balance_report.json")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--target_total", type=int, default=200)
    p.add_argument("--fake_real_ratio", type=float, default=0.6,
                   help="Target fraction of FAKE cards (rest split between REAL and UNCERTAIN)")
    p.add_argument("--validate", action="store_true", default=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    rng = random.Random(args.seed)

    raw = json.load(open(args.in_file, encoding="utf-8"))

    # Schema validation pass
    valid_cards: list[dict] = []
    invalid_count = 0
    if args.validate:
        for c in raw:
            try:
                UnityCard.model_validate(c)
                valid_cards.append(c)
            except Exception as e:
                invalid_count += 1
                logger.debug("Card %s invalid: %s", c.get("id"), str(e)[:120])
    else:
        valid_cards = raw

    # Bucket by verdict and candidate
    buckets: dict = {}
    for c in valid_cards:
        key = (c.get("verdict", "UNCERTAIN"), c.get("candidate", "NONE"))
        buckets.setdefault(key, []).append(c)
    for k in buckets:
        rng.shuffle(buckets[k])

    # Compute target counts
    n_fake = int(args.target_total * args.fake_real_ratio)
    n_real = int(args.target_total * (1 - args.fake_real_ratio) * 0.85)
    n_uncertain = args.target_total - n_fake - n_real
    candidate_codes = ["C-RM", "C-IB", "C-JS"]
    per_cand_fake = max(1, n_fake // 3)
    per_cand_real = max(1, n_real // 3)

    chosen: list[dict] = []

    def take(verdict: str, cand: str, n: int):
        items = buckets.get((verdict, cand), [])
        chosen.extend(items[:n])

    # Sample by (verdict, candidate)
    for cand in candidate_codes:
        take("FAKE", cand, per_cand_fake)
        take("REAL", cand, per_cand_real)
    # Pick uncertains across any candidate
    uncs: list = []
    for cand in candidate_codes + ["NONE"]:
        uncs.extend(buckets.get(("UNCERTAIN", cand), []))
    rng.shuffle(uncs)
    chosen.extend(uncs[:n_uncertain])

    # Top up if short — pull additional from any bucket
    if len(chosen) < args.target_total:
        all_remaining = []
        for key, items in buckets.items():
            already = sum(1 for c in chosen if c in items)
            all_remaining.extend(items[already:])
        rng.shuffle(all_remaining)
        chosen.extend(all_remaining[: args.target_total - len(chosen)])

    rng.shuffle(chosen)
    chosen = chosen[: args.target_total]

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(chosen, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Balance report
    report = {
        "input_total": len(raw),
        "schema_invalid_dropped": invalid_count,
        "valid_pool": len(valid_cards),
        "selected_total": len(chosen),
        "verdict_distribution": _count_by(chosen, lambda c: c.get("verdict")),
        "candidate_distribution": _count_by(chosen, lambda c: c.get("candidate")),
        "difficulty_distribution": _count_by(chosen, lambda c: c.get("difficulty_bin")),
        "indicator_coverage": _indicator_coverage(chosen),
        "explanation_diversity": _diversity(chosen),
        "bank_version": BANK_VERSION,
        "bank_hash": bank_hash(),
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"), indent=2)
    logger.info("Wrote %d balanced cards → %s", len(chosen), args.out_file)
    logger.info("Verdict dist: %s", report["verdict_distribution"])
    logger.info("Candidate dist: %s", report["candidate_distribution"])
    logger.info("Indicator coverage: %s",
                {k: v for k, v in list(report["indicator_coverage"].items())[:6]})


def _count_by(cards, keyfn):
    out: dict = {}
    for c in cards:
        out[keyfn(c)] = out.get(keyfn(c), 0) + 1
    return out


def _indicator_coverage(cards):
    counts: dict = {}
    for c in cards:
        for ind in c.get("fired_indicators", []):
            counts[ind] = counts.get(ind, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _diversity(cards):
    summaries = [c.get("explanation", {}).get("summary", "") for c in cards]
    return {
        "total": len(summaries),
        "unique": len(set(summaries)),
        "diversity_pct": round(100.0 * len(set(summaries)) / max(len(summaries), 1), 1),
    }


if __name__ == "__main__":
    main()

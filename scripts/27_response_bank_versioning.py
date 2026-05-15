#!/usr/bin/env python3
"""Stamp the response bank version + hash into the pipeline for audit provenance."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_response_bank import (BANK, CREDIBLE_AFFIRMATIONS, BANK_VERSION,
                                    bank_hash, bank_stats,
                                    assemble_explanation,
                                    tier_for_card_index)
from minerva_candidates import REGISTRY

logger = logging.getLogger(__name__)

def _load_cards_or_pool(path: str) -> list:
    """accept either a flat list of cards or a pool doc
    {"_metadata": ..., "cards": [...]}.
    """
    import json
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict) and "cards" in payload:
        return payload["cards"]
    return payload


def cmd_stamp(args):
    cards = _load_cards_or_pool(args.in_file)
    h = bank_hash()
    for c in cards:
        c.setdefault("provenance", {})["bank_version"] = BANK_VERSION
        c["provenance"]["bank_hash"] = h
    json.dump(cards, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info("Stamped %d cards with bank %s (hash=%s)",
                len(cards), BANK_VERSION, h)


def cmd_diff(args):
    a = json.load(open(args.bank_a, encoding="utf-8"))
    b = json.load(open(args.bank_b, encoding="utf-8"))
    a_ids = {e["bank_id"]: e for e in a.get("entries", [])}
    b_ids = {e["bank_id"]: e for e in b.get("entries", [])}
    only_a = sorted(set(a_ids) - set(b_ids))
    only_b = sorted(set(b_ids) - set(a_ids))
    common = sorted(set(a_ids) & set(b_ids))
    changed = [bid for bid in common if a_ids[bid].get("phrase") != b_ids[bid].get("phrase")]
    diff = {
        "a_version": a.get("bank_version", "?"),
        "b_version": b.get("bank_version", "?"),
        "removed_in_b": only_a,
        "added_in_b": only_b,
        "changed_phrases": changed,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(diff, open(args.out_file, "w", encoding="utf-8"), indent=2)
    logger.info("Diff: -%d +%d ~%d → %s",
                len(only_a), len(only_b), len(changed), args.out_file)


def cmd_rerender(args):
    cards = _load_cards_or_pool(args.in_file)
    rendered = 0
    for idx, c in enumerate(cards):
        cand_obj = REGISTRY.get(c.get("candidate", "NONE"))
        cand_name = cand_obj.name if cand_obj else None
        tier = tier_for_card_index(idx)
        c["explanation"] = assemble_explanation(
            fired_indicators=c.get("fired_indicators", []),
            verdict=c.get("verdict", "UNCERTAIN"),
            fake_likelihood_percent=c.get("fake_likelihood_percent", 50.0),
            seed_str=f"{c['id']}|{args.seed}",
            tier=tier,
            candidate_name=cand_name,
        )
        c.setdefault("provenance", {})["bank_version"] = BANK_VERSION
        c["provenance"]["bank_hash"] = bank_hash()
        c.setdefault("provenance", {}).setdefault("script_chain", []).append("27-rerender")
        rendered += 1
    json.dump(cards, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info("Re-rendered %d cards under bank %s", rendered, BANK_VERSION)


def cmd_export(args):
    """Export the in-Python BANK as a JSON file."""
    entries = []
    for code, items in BANK.items():
        for e in items:
            entries.append({
                "code": e.code, "tier": e.tier, "phrase": e.phrase,
                "sift": e.sift, "bank_id": e.bank_id,
            })
    for e in CREDIBLE_AFFIRMATIONS:
        entries.append({
            "code": e.code, "tier": e.tier, "phrase": e.phrase,
            "sift": e.sift, "bank_id": e.bank_id,
        })
    payload = {
        "bank_version": BANK_VERSION,
        "bank_hash": bank_hash(),
        "stats": bank_stats(),
        "entries": entries,
    }
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.out_file, "w", encoding="utf-8"), indent=2)
    logger.info("Exported %d bank entries → %s", len(entries), args.out_file)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True, dest="cmd")

    s = sub.add_parser("stamp")
    s.add_argument("--in_file", required=True)
    s.add_argument("--out_file", required=True)
    s.set_defaults(fn=cmd_stamp)

    s = sub.add_parser("diff")
    s.add_argument("--bank_a", required=True)
    s.add_argument("--bank_b", required=True)
    s.add_argument("--out_file", required=True)
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("rerender")
    s.add_argument("--in_file", required=True)
    s.add_argument("--out_file", required=True)
    s.add_argument("--seed", type=int, default=1729)
    s.set_defaults(fn=cmd_rerender)

    s = sub.add_parser("export")
    s.add_argument("--out_file", required=True)
    s.set_defaults(fn=cmd_export)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reject cards whose content is not on-theme for the 2025 election scenario."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_filters import keyword_score, run_all_gates

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--report_out",
                   default="reports/theme_filter_report.json")
    p.add_argument("--rejection_log",
                   default="reports/theme_rejection_log.jsonl")
    p.add_argument("--theme_threshold", type=float, default=0.55)
    p.add_argument("--allow_neutral_volume", action="store_true", default=True)
    p.add_argument("--strict_no_neutral", action="store_true",
                   help="Override: reject all non-electoral content")
    p.add_argument("--require_candidate", action="store_true", default=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.strict_no_neutral:
        args.allow_neutral_volume = False

    cards = json.load(open(args.in_file, encoding="utf-8"))
    accepted: list[dict] = []
    rejection_log: list[dict] = []
    counters = {
        "total": len(cards),
        "accepted_electoral": 0,
        "accepted_neutral_volume": 0,
        "rejected_off_theme": 0,
        "rejected_legacy_pseudonym": 0,
        "rejected_truncated": 0,
        "rejected_no_candidate": 0,
    }

    for card in cards:
        text = card.get("text", "")
        gate = run_all_gates(
            text,
            theme_threshold=args.theme_threshold,
            allow_neutral_volume=args.allow_neutral_volume,
            require_candidate_mention=args.require_candidate,
        )
        # Update card with theme classification
        tflags = card.setdefault("theme_flags", {})
        tflags["is_electoral"] = gate.diagnostics.get("is_electoral", False)
        tflags["electoral_score"] = float(gate.diagnostics.get("theme_score", 0.0))
        tflags["is_neutral_volume"] = gate.diagnostics.get("is_neutral_volume", False)
        tflags["classifier_label"] = (
            "electoral" if tflags["is_electoral"]
            else ("neutral_volume" if tflags["is_neutral_volume"] else "off_theme")
        )

        prov = card.setdefault("provenance", {})
        prov.setdefault("script_chain", []).append("23")

        if gate.accepted:
            accepted.append(card)
            if tflags["is_electoral"]:
                counters["accepted_electoral"] += 1
            elif tflags["is_neutral_volume"]:
                counters["accepted_neutral_volume"] += 1
            continue

        # Log rejection
        for reason in gate.reasons:
            stage_map = {
                "theme:": "theme_filter",
                "truncation:": "truncation_filter",
                "legacy_pseudonyms": "pseudonym_filter",
                "no_candidate_mention": "candidate_filter",
            }
            stage = "theme_filter"
            for prefix, s in stage_map.items():
                if reason.startswith(prefix) or reason == prefix:
                    stage = s
                    break
            rejection_log.append({
                "card_id": card.get("id", "unknown"),
                "stage": stage,
                "verdict": "reject",
                "reason": reason,
                "diagnostics": gate.diagnostics,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            if "theme:" in reason:
                counters["rejected_off_theme"] += 1
            elif "truncation:" in reason:
                counters["rejected_truncated"] += 1
            elif "legacy_pseudonyms" in reason:
                counters["rejected_legacy_pseudonym"] += 1
            elif "no_candidate_mention" in reason:
                counters["rejected_no_candidate"] += 1

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(accepted, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(counters, open(args.report_out, "w", encoding="utf-8"), indent=2)
    Path(args.rejection_log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.rejection_log, "w", encoding="utf-8") as f:
        for entry in rejection_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info("Theme filter results:")
    for k, v in counters.items():
        logger.info("  %-30s %d", k, v)
    logger.info("=" * 60)
    logger.info("Accepted → %s", args.out_file)
    logger.info("Report   → %s", args.report_out)
    logger.info("Rejects  → %s", args.rejection_log)


if __name__ == "__main__":
    main()

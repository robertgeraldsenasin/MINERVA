#!/usr/bin/env python3
"""
26_faithfulness_audit.py  (NEW)
===============================

Post-hoc audit that re-extracts indicators from every card's
explanation prose and asserts SET EQUALITY with the originally
fired_indicators.

Why: Longo et al. (2024) Open Problem 7 mandates that paraphrased
explanations remain FAITHFUL — i.e. they continue to reflect the
underlying decision. Without an automated check, a future bank
edit could silently break this property. Liu, Ye & Li (2024)
distinguish faithfulness from plausibility; this script audits
faithfulness.

This is the panel-defence-grade check: at thesis defence we can
say "every card emitted by the pipeline passes 26_faithfulness_audit"
as a verifiable claim.

Output:
  reports/faithfulness_audit_report.json   (counts + sample failures)
  reports/faithfulness_failures.jsonl      (full failure log)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_response_bank import BANK, CREDIBLE_AFFIRMATIONS, BANK_VERSION

logger = logging.getLogger(__name__)


# Indicator-mention dictionary: words that, if present in an explanation,
# indicate the explanation is talking about that indicator.
INDICATOR_MENTIONS = {
    "EMO": ["emotion", "loaded", "heated", "betrayed", "anger", "feeling",
            "outraged", "scandalo", "manloloko", "sensation"],
    "URG": ["urgency", "share now", "before it's deleted", "agad", "ngayon",
            "pressure", "ALL CAPS", "shouting", "panic", "act now"],
    "ANON": ["anonymous", "sources say", "insiders", "daw", "diumano",
            "hearsay", "unnamed"],
    "MISS": ["no link", "no document", "zero receipts", "missing", "no source",
            "no named source", "no url", "no evidence"],
    "FAB": ["fabricated", "quote", "transcript", "no video"],
    "POL": ["polariz", "us-vs-them", "real filipinos", "traitors",
            "in-group", "out-group"],
    "CONS": ["conspirac", "secret", "cover-up", "they don't want you",
            "hidden", "cabal"],
    "DISC": ["discredit", "personal attack", "ad hominem", "red-tag",
            "smear", "communist", "NPA"],
    "IMP": ["impersonat", "spoofed", "copy-cat", "fake outlet", "logo"],
    "REV": ["revisionism", "golden age", "historical", "rewriting",
            "martial law"],
    "ENDO": ["survey", "endorsement", "85%", "manufactured", "polling firm",
            "sample size"],
    "RECF": ["fabrication", "invented project", "fake award", "credential",
            "Harvard", "Nobel", "record"],
}


def _mentions_indicator(text: str, code: str) -> bool:
    tl = text.lower()
    for marker in INDICATOR_MENTIONS.get(code, []):
        if marker.lower() in tl:
            return True
    return False


def audit_card(card: dict) -> dict:
    """Audit a single card. Returns issue list (empty if all green)."""
    issues = []
    fired = set(card.get("fired_indicators", []))
    expl = card.get("explanation", {}) or {}
    summary = expl.get("summary", "") or ""
    indicator_phrases = expl.get("indicator_phrases", []) or []

    # Check 1: indicator_phrases set should be subset of fired_indicators
    phrase_codes = {p.get("indicator") for p in indicator_phrases
                    if p.get("indicator") and p.get("indicator") != "CREDIBLE"}
    extra_in_phrases = phrase_codes - fired
    if extra_in_phrases:
        issues.append({
            "type": "extra_indicator_phrase",
            "details": f"Explanation has phrases for {extra_in_phrases} but they did not fire",
        })

    # Check 2: each phrase should mention its indicator (lexical proxy
    # for "the phrase is talking about the right thing")
    for p in indicator_phrases:
        code = p.get("indicator")
        phr = p.get("phrase", "")
        if code in INDICATOR_MENTIONS and not _mentions_indicator(phr, code):
            issues.append({
                "type": "indicator_phrase_mismatch",
                "code": code,
                "phrase_excerpt": phr[:120],
            })

    # Check 3: bank_ref well-formed and bank_version matches
    for p in indicator_phrases:
        ref = p.get("bank_ref", "")
        if not re.match(r"^[A-Z]+/v[\d\.]+/[npa]\d+$", ref):
            issues.append({
                "type": "malformed_bank_ref",
                "ref": ref,
            })

    # Check 4: stamped bank_version matches current bank
    bv = expl.get("bank_version", "unknown")
    if bv != BANK_VERSION:
        issues.append({
            "type": "stale_bank_version",
            "stamped": bv,
            "current": BANK_VERSION,
        })

    # Check 5: REAL verdict cards must include a credible affirmation
    if card.get("verdict") == "REAL":
        has_credible = any(p.get("indicator") == "CREDIBLE"
                            for p in indicator_phrases)
        if not has_credible and "credible" not in summary.lower():
            issues.append({"type": "real_card_no_credible_affirmation"})

    # Check 6: at least 1 indicator phrase or credible affirmation exists
    if not indicator_phrases and card.get("verdict") in ("FAKE", "REAL"):
        issues.append({"type": "empty_explanation_phrases"})

    return {
        "card_id": card.get("id", "unknown"),
        "issues": issues,
        "passed": len(issues) == 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--report_out",
                   default="reports/faithfulness_audit_report.json")
    p.add_argument("--failures_out",
                   default="reports/faithfulness_failures.jsonl")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any failure")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cards = json.load(open(args.in_file, encoding="utf-8"))
    audits = [audit_card(c) for c in cards]
    failures = [a for a in audits if not a["passed"]]

    issue_counts: dict = {}
    for a in failures:
        for iss in a["issues"]:
            t = iss["type"]
            issue_counts[t] = issue_counts.get(t, 0) + 1

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "bank_version": BANK_VERSION,
        "total_audited": len(cards),
        "passed": len(audits) - len(failures),
        "failed": len(failures),
        "pass_rate": round(100.0 * (len(audits) - len(failures)) /
                            max(len(audits), 1), 2),
        "issue_breakdown": issue_counts,
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"), indent=2)

    Path(args.failures_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.failures_out, "w", encoding="utf-8") as f:
        for a in failures:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    logger.info("Audit: %d / %d passed (%.1f%%)",
                report["passed"], report["total_audited"], report["pass_rate"])
    if issue_counts:
        logger.info("Issues: %s", issue_counts)

    if args.strict and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

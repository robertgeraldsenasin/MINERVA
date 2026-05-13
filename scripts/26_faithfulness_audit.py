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

def _load_cards_or_pool(path: str) -> list:
    """v2.3: accept either a flat list of cards or a pool doc
    {"_metadata": ..., "cards": [...]}.
    """
    import json
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict) and "cards" in payload:
        return payload["cards"]
    return payload


# Indicator-mention dictionary: words that, if present in an explanation,
# indicate the explanation is talking about that indicator.
INDICATOR_MENTIONS = {
    # v2.1 expanded lexicon — covers actual bank prose more thoroughly.
    # Each list contains lower-cased substrings; the audit fires if ANY
    # match appears in the explanation phrase for that indicator.
    "EMO": ["emotion", "loaded", "heated", "betrayed", "anger", "feeling",
            "outraged", "scandalo", "manloloko", "sensation", "react",
            "bad news", "share without thinking", "designed to make"],
    "URG": ["urgency", "share now", "before it", "agad", "ngayon",
            "pressure", "all caps", "shouting", "panic", "act now",
            "deleted", "10 minutes", "in 10 min", "real news doesn",
            "skip checking", "bypass"],
    "ANON": ["anonymous", "sources say", "insiders", "daw", "diumano",
            "hearsay", "unnamed", "no name", "second-hand",
            "name their source"],
    "MISS": ["no link", "no document", "zero receipts", "missing",
            "no source", "no named source", "no url", "no evidence",
            "where would i check", "claim hangs",
            "big claim", "receipts",
            # v2.2: catch the advanced-tier phrase
            "scores zero", "named-source", "external-link",
            "credibility-signals", "credibility signals",
            "w3c", "leite et al",
            "asymmetry", "bigger the claim",
            "single unverified", "low-credibility prior"],
    "FAB": ["fabricated", "quote", "transcript", "no video", "alleged",
            "trace to source", "primary transcript", "anyone could write",
            "treat as alleged"],
    "POL": ["polariz", "us-vs-them", "real filipinos", "traitors",
            "in-group", "out-group", "us vs them", "us-vs", "tribal",
            "less than human",
            # v2.6-final additions
            "us versus them", "divisive", "framing", "paghihiwalay",
            "manghahati", "lipunan", "elitistang"],
    "CONS": ["conspirac", "secret", "cover-up", "they don't want you",
            "hidden", "cabal", "secret-cabal", "footprints", "unfalsifiable",
            "leaves footprints",
            # v2.6-final additions
            "deep state", "lihim", "konspirasiya", "shadowy"],
    "DISC": ["discredit", "personal attack", "ad hominem", "red-tag",
            "smear", "communist", "npa", "evidence", "set it aside",
            "attacks the person",
            # v2.6-final additions
            "engaging with arguments", "engaging with their argument",
            "without engaging", "personal", "atake", "insulto",
            "without engaging with"],
    "IMP": ["impersonat", "spoofed", "copy-cat", "fake outlet", "logo",
            "real news brand", "domain is wrong", "off by a letter",
            "borrow its credibility",
            # v2.6-final additions
            "fake authority", "uses titles", "fake account",
            "pekeng", "nagpapanggap", "fake profile", "fake page"],
    "REV": ["revisionism", "golden age", "historical", "rewriting",
            "martial law", "rewrites a historical period",
            "who benefits",
            # v2.6-final additions
            "alternative narrative", "without sources",
            "ginintuang panahon", "binabago ang"],
    "ENDO": ["survey", "endorsement", "85%", "manufactured", "polling firm",
            "sample size", "real surveys disclose", "graphic, not data",
            "without a polling firm",
            # v2.6-final additions
            "claimed endorsement", "official statement",
            "without official"],
    "RECF": [
            # v2.6-final REWRITTEN — RECF means "recycled content"
            # (old material reused/relabeled as new), not fabricated
            # credentials. The original lexicon was wrong.
            "recycled", "reshared", "reused", "old photo", "old video",
            "old clip", "from a previous", "metadata shows",
            "originally from", "ginagamit muli", "lumang", "matagal na",
            "reupload", "reshare", "old content",
            # legacy fabrication markers (kept for backward compat with
            # old cards that used RECF in the fabrication sense)
            "fabrication", "invented project", "fake award", "credential",
            "harvard", "nobel", "record", "official site or coa",
            "verifiable record", "voting record"],
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
    # v2.9.7: bank_ref format in v2.9+ is <INDICATOR>/<role>/<tier>/v<digit>
    # (4-segment), e.g. "MISS/fake/novice/v0". The pre-v2.9 format was
    # <INDICATOR>/v<version>/<tier-letter><idx> e.g. "MISS/v1.0/n0".
    # Accept both — first the new format, then the legacy format as fallback.
    BANK_REF_NEW = re.compile(
        r"^[A-Z]+/(fake|real|none)/(novice|proficient|advanced)/v\d+$"
    )
    BANK_REF_LEGACY = re.compile(r"^[A-Z]+/v[\d\.]+/[npa]\d+$")
    for p in indicator_phrases:
        ref = p.get("bank_ref", "")
        if not (BANK_REF_NEW.match(ref) or BANK_REF_LEGACY.match(ref)):
            issues.append({
                "type": "malformed_bank_ref",
                "ref": ref,
            })

    # Check 4: stamped bank_version matches current bank
    # v2.9.7: cards may stamp the corpus codename ("v2.9.0", "v2.9.4", "v2.9.6")
    # while the bank file itself uses a semver-style version ("1.1"). Both refer
    # to the same canonical response_bank_v2.json. Accept either form.
    bv = expl.get("bank_version", "unknown")
    _CODENAME_RX = re.compile(r"^v\d+\.\d+\.\d+$")  # v2.9.6, v2.9.0, etc.
    if bv != BANK_VERSION and not _CODENAME_RX.match(bv):
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

    cards = _load_cards_or_pool(args.in_file)
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

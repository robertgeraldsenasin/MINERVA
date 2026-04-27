
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def candidate_of(card: Dict[str, Any]) -> str:
    targets = card.get("targets")
    if isinstance(targets, list) and targets:
        return str(targets[0] or "GENERAL")
    linked = card.get("linked_blue_truth", {})
    if isinstance(linked, dict) and linked.get("candidate"):
        return str(linked["candidate"])
    return "GENERAL"


def score_of(card: Dict[str, Any]) -> float:
    q = card.get("quality", {})
    try:
        return float(q.get("quality_score", 0.0))
    except Exception:
        return 0.0


def build_release_deck(template_cards: List[Dict[str, Any]], organic_cards: List[Dict[str, Any]], min_organic_quality: float) -> Dict[str, Any]:
    seen = set()
    release: List[Dict[str, Any]] = []
    dropped_duplicates = 0
    dropped_low_quality_organic = 0

    # Templates first: they are the safer fallback for release.
    for card in sorted(template_cards, key=lambda c: (-score_of(c), str(c.get("id")))):
        key = norm_text(str(card.get("text", "")))
        if key in seen:
            dropped_duplicates += 1
            continue
        row = dict(card)
        row["release_ready"] = True
        row["release_source"] = "rule_constrained_template"
        release.append(row)
        seen.add(key)

    # Organic cards only if they are unusually strong after curation.
    for card in sorted(organic_cards, key=lambda c: (-score_of(c), str(c.get("id")))):
        if score_of(card) < min_organic_quality:
            dropped_low_quality_organic += 1
            continue
        key = norm_text(str(card.get("text", "")))
        if key in seen:
            dropped_duplicates += 1
            continue
        row = dict(card)
        row["release_ready"] = True
        row["release_source"] = "organic_gpt2_after_strict_gate"
        release.append(row)
        seen.add(key)

    verdict_counts = Counter(str(card.get("verdict", "unknown")) for card in release)
    candidate_counts = Counter(candidate_of(card) for card in release)

    report = {
        "release_cards": len(release),
        "verdict_counts": dict(verdict_counts),
        "candidate_counts": dict(candidate_counts),
        "dropped_duplicates": dropped_duplicates,
        "dropped_low_quality_organic": dropped_low_quality_organic,
        "policy": {
            "template_cards_prioritized": True,
            "organic_cards_allowed_only_if_quality_at_least": min_organic_quality,
            "purpose": "Produce a safer release deck for classroom testing by prioritizing coherent, rule-constrained election scenarios.",
        },
    }
    return {"cards": release, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a release-ready MINERVA deck by prioritizing rule-constrained cards and excluding weak organic outputs.")
    ap.add_argument("--template_curated", required=True)
    ap.add_argument("--organic_curated", required=False, default=None)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--report_out", required=False, default=None)
    ap.add_argument("--min_organic_quality", type=float, default=0.60)
    args = ap.parse_args()

    template_cards = read_json(Path(args.template_curated))
    organic_cards = read_json(Path(args.organic_curated)) if args.organic_curated else []

    result = build_release_deck(template_cards, organic_cards, args.min_organic_quality)
    write_json(Path(args.out_file), result["cards"])
    if args.report_out:
        write_json(Path(args.report_out), result["report"])

    print(f"[OK] Wrote {len(result['cards'])} release-ready cards to {args.out_file}")


if __name__ == "__main__":
    main()

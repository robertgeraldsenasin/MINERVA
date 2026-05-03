#!/usr/bin/env python3
"""
25_build_candidate_scenarios.py  (REFACTORED v2.0)
==================================================

Build per-candidate VERIdex profile cards: each profile contains the
candidate's archetype-grounded biography, platform planks,
indicator-susceptibility heatmap, and counter-narrative anchors.

WHAT CHANGED FROM v1
--------------------
v1 emitted random "Candidate XXX" profiles populated from a small
hand-written list. There was no archetype grounding and no link to
the misinformation indicators the candidate was likely to attract.

v2.0:
  * Profiles drawn directly from minerva_candidates.REGISTRY.
  * Each profile lists indicator susceptibility (which misinformation
    cues this candidate's posts have most-frequently triggered) so
    that VERIdex can show learners "this is what to watch for when
    you scroll a Marquez/Bantayan/Salonga rumor".
  * Counter-narrative anchors: documented, verifiable real-evidence
    types the learner could use to fact-check rumors about this
    candidate (Mendoza et al. 2023; Caulfield 2019 — SIFT 'Find').

PIPELINE POSITION
-----------------
Reads:  generated/story_cards.json (for the empirical cue counts)
        templates/candidate_profiles_three_candidates.json (registry)
Writes: generated/candidate_scenarios.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_candidates import REGISTRY

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--story_cards", required=True,
                   help="Final story_cards.json to derive empirical cue counts")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cards = _load_cards_or_pool(args.story_cards)

    # Empirical: count how often each indicator fires per candidate
    empirical: dict = {}
    for c in cards:
        cand = c.get("candidate", "NONE")
        empirical.setdefault(cand, {"total": 0, "indicators": {}})
        empirical[cand]["total"] += 1
        for ind in c.get("fired_indicators", []):
            d = empirical[cand]["indicators"]
            d[ind] = d.get(ind, 0) + 1

    profiles = []
    for code, cand in REGISTRY.items():
        emp = empirical.get(code, {"total": 0, "indicators": {}})
        emp_total = max(emp["total"], 1)
        empirical_susceptibility = {
            ind: round(emp["indicators"].get(ind, 0) / emp_total, 3)
            for ind in cand.indicator_weights
        }
        profile = {
            "code": cand.code,
            "name": cand.name,
            "archetype": cand.archetype,
            "bio": _bio_paragraph(cand),
            "age": _archetype_age(cand.archetype),
            "region": cand.region,
            "party_acronym": cand.party_acronym,
            "party_name": cand.party_name,
            "platform_slogan": cand.platform_slogan,
            "policy_planks": list(cand.policy_planks),
            "indicator_susceptibility_prior": cand.indicator_weights,
            "indicator_susceptibility_empirical": empirical_susceptibility,
            "counter_narrative_anchors": list(cand.counter_anchors),
            "references": list(cand.references),
            "card_count_in_deck": emp["total"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        profiles.append(profile)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(profiles, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info("Wrote %d candidate scenarios → %s", len(profiles), args.out_file)


def _bio_paragraph(cand) -> str:
    if cand.archetype == "DYNASTIC":
        return (
            f"{cand.name} is a third-generation politician from {cand.region} "
            f"running under the {cand.party_name} ({cand.party_acronym}). "
            f"His campaign emphasises continuity, infrastructure, and "
            f"national-unity messaging captured in the slogan "
            f"\u201c{cand.platform_slogan}.\u201d"
        )
    if cand.archetype == "REFORMIST":
        return (
            f"{cand.name} is a lawyer-turned-local-executive from "
            f"{cand.region} running under the {cand.party_name} "
            f"({cand.party_acronym}). Her campaign foregrounds "
            f"transparency, accountability, and reformist governance, "
            f"branded under \u201c{cand.platform_slogan}.\u201d"
        )
    return (
        f"{cand.name} is a former broadcast personality and party-list "
        f"representative from {cand.region}, running under the "
        f"{cand.party_name} ({cand.party_acronym}). His campaign blends "
        f"populist mass-appeal with concrete pro-poor planks under "
        f"\u201c{cand.platform_slogan}.\u201d"
    )


def _archetype_age(arch: str) -> int:
    return {"DYNASTIC": 62, "REFORMIST": 47, "POPULIST": 54}.get(arch, 50)


if __name__ == "__main__":
    main()

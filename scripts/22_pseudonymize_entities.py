#!/usr/bin/env python3
"""
22_pseudonymize_entities.py  (REFACTORED v2.0)
==============================================

Deterministic pseudonymisation: replace any real Filipino political
name reference with one of the three fictional candidate names
(C-RM Marquez, C-IB Bantayan, C-JS Salonga), routed by archetype cue
co-occurrence and cached for session-scoped consistency.

WHAT CHANGED FROM v1
--------------------
v1 emitted random codes ("Candidate GQW", "Candidate DTQ", "Entity B")
that broke narrative coherence — students could not study a candidate
they encountered under a different code each card.

v2.0:
  * Fixed registry of 3 archetype-grounded fictional candidates.
  * Archetype router (cue-based) selects which candidate plays the
    role implied by the post's narrative content.
  * Session cache: same real-name input → same fictional candidate
    code throughout the run, satisfying Yermilov et al. (2023)'s
    consistency-preservation criterion.
  * Output: the card's `text` is rewritten in place; the `candidate`
    field is set to one of {C-RM, C-IB, C-JS, NONE}.

PIPELINE POSITION
-----------------
Reads:  generated/unity_cards.json   (or balanced version)
Writes: generated/unity_cards_pseudonymized.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_candidates import REGISTRY, pseudonymize, candidate_for_text
from minerva_filters import has_legacy_pseudonyms
from minerva_indicators import indicator_summary_for_card
from minerva_response_bank import assemble_explanation, tier_for_card_index

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--re_explain", action="store_true",
                   help="Recompute explanation now that candidate name is known")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cards = json.load(open(args.in_file, encoding="utf-8"))
    session_cache: dict[str, str] = {}

    out: list[dict] = []
    counts = {"C-RM": 0, "C-IB": 0, "C-JS": 0, "NONE": 0}

    for idx, card in enumerate(cards):
        text = card.get("text", "")
        if not text:
            continue
        # Step A: rewrite real-name references (e.g. Sen. Marcos -> Sen. Marquez)
        rewritten, code, replaced = pseudonymize(
            text, session_cache=session_cache, seed=args.seed
        )
        # Step B: also rewrite legacy "Candidate XXX" patterns to the
        # assigned candidate's full name. The legacy pipeline emitted
        # random codes; we map them all to the cue-routed candidate so
        # the resulting card mentions the candidate explicitly.
        has_legacy, offenders = has_legacy_pseudonyms(rewritten)
        if has_legacy:
            cand_name = REGISTRY[code].name
            import re as _re
            for off in offenders:
                rewritten = _re.sub(_re.escape(off), cand_name, rewritten)
            replaced.extend(offenders)
        card["text"] = rewritten
        card["candidate"] = code
        # Re-extract indicators on rewritten text (the rewrite may flip POL/IMP cues)
        card.update({
            k: v for k, v in indicator_summary_for_card(rewritten).items()
            if k in {"fired_indicators", "indicator_details", "named_features"}
        })
        # Optionally re-build explanation now that we have a real candidate name
        if args.re_explain:
            cand_obj = REGISTRY.get(code)
            cand_name = cand_obj.name if cand_obj else None
            tier = tier_for_card_index(idx)
            card["explanation"] = assemble_explanation(
                fired_indicators=card.get("fired_indicators", []),
                verdict=card.get("verdict", "UNCERTAIN"),
                fake_likelihood_percent=card.get("fake_likelihood_percent", 50.0),
                seed_str=f"{card['id']}|{args.seed}",
                tier=tier,
                candidate_name=cand_name,
            )

        # Provenance update
        prov = card.setdefault("provenance", {})
        prov.setdefault("script_chain", []).append("22")
        prov["pseudonym_session_size"] = len(session_cache)
        card.setdefault("metadata", {})["pseudonym_replacements"] = replaced

        counts[code if code in counts else "NONE"] = counts.get(code, 0) + 1
        out.append(card)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info("Pseudonymised %d cards → %s", len(out), args.out_file)
    logger.info("Candidate distribution: %s", counts)
    logger.info("Session-cache size: %d unique entities mapped", len(session_cache))


if __name__ == "__main__":
    main()

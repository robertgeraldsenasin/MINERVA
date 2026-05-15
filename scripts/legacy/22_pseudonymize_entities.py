#!/usr/bin/env python3
"""
22_pseudonymize_entities.py  (REFACTORED v2.2)

v2.2 changes from v2.1:
  * After replacing real names with the canonical fictional candidate,
    COLLAPSE repetition: keep the full name on first mention (e.g.
    'Sen. Reynaldo "Rey" Marquez') and use the short surname for
    subsequent mentions in the same card. Avoids the "name jammed
    everywhere" problem that made cards read as broken.
  * Catch a wider set of GPT-2-invented placeholder patterns:
    'Candidate W', 'Candidate AW', 'Candidate EK', 'Candidate UU' —
    any 'Candidate' followed by 1-3 capital letters. The v2.1 regex
    only caught 3-letter patterns.
  * Apply legacy-pseudonym rewriting BEFORE the candidate-name
    substitution so collapse logic sees a clean text.
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
    import re as _re

    # v2.4: UNIFIED placeholder regex — catches all GPT-2 / JCBlaise
    # placeholder families. The thesis specifies exactly THREE fictional
    # candidates (C-RM, C-IB, C-JS); any other "Candidate X" / "Entity Y"
    # / "Person Z" placeholder MUST be rewritten to one of those three.
    # Anything else is a leak.
    LEGACY_RE = _re.compile(
        r"\b(?:Candidate|Entity|Person)\s+[A-Z]{1,3}\b"
    )

    # v2.2: pre-built list of all canonical full names so we can clean
    # cross-candidate name pollution that may have leaked in earlier runs
    ALL_FULL_NAMES = [c.name for c in REGISTRY.values()]
    ALL_SHORT_NAMES = [c.short_name for c in REGISTRY.values()]

    out: list[dict] = []
    counts = {"C-RM": 0, "C-IB": 0, "C-JS": 0, "NONE": 0}
    repetition_collapses = 0
    cross_pollution_cleaned = 0

    for idx, card in enumerate(cards):
        text = card.get("text", "")
        if not text:
            continue

        # Step A: rewrite real political surnames -> fictional candidate
        rewritten, code, replaced = pseudonymize(
            text, session_cache=session_cache, seed=args.seed
        )

        # Step B: rewrite ANY "Candidate XX" placeholder
        legacy_matches = LEGACY_RE.findall(rewritten)
        if legacy_matches:
            cand_name = REGISTRY[code].name
            for off in set(legacy_matches):
                rewritten = _re.sub(_re.escape(off), cand_name, rewritten)
            replaced.extend(legacy_matches)

        # Step B2 (NEW v2.2): clean cross-candidate name pollution.
        # If the card was rewritten by a previous run that picked a
        # different candidate, OTHER candidates' full names may still
        # appear. Replace any non-assigned candidate's name with the
        # assigned one BEFORE running collapse, so we don't end up
        # with frankenstein names like "Salonga \"Rey\" Marquez".
        if code in REGISTRY:
            assigned_full = REGISTRY[code].name
            for other_code, other_cand in REGISTRY.items():
                if other_code == code:
                    continue
                other_full = other_cand.name
                if other_full in rewritten and assigned_full != other_full:
                    rewritten = rewritten.replace(other_full, assigned_full)
                    cross_pollution_cleaned += 1
                # Also strip dangling short-name fragments from other candidates
                other_short = other_cand.short_name
                if other_short in rewritten and other_short != REGISTRY[code].short_name:
                    # Only replace standalone occurrences (word-bounded)
                    rewritten = _re.sub(
                        rf'\b{_re.escape(other_short)}\b',
                        REGISTRY[code].short_name,
                        rewritten,
                    )

        # Step C: collapse repetition of the assigned name.
        # First mention: full name. Subsequent: short surname.
        if code in REGISTRY:
            full_name = REGISTRY[code].name
            short_name = REGISTRY[code].short_name
            full_pattern = _re.escape(full_name)
            occurrences = list(_re.finditer(full_pattern, rewritten))
            if len(occurrences) > 1:
                pieces = []
                last_end = 0
                for i, m in enumerate(occurrences):
                    pieces.append(rewritten[last_end:m.start()])
                    if i == 0:
                        pieces.append(full_name)
                    else:
                        pieces.append(short_name)
                    last_end = m.end()
                pieces.append(rewritten[last_end:])
                rewritten = "".join(pieces)
                repetition_collapses += 1

        # Step D (NEW v2.2): clean leftover quote artifacts from
        # the original raw GPT-2 output (e.g. stray '"Rey"' floating
        # after Marquez was replaced by Salonga). Remove orphaned
        # quoted-name fragments that don't belong to the assigned
        # candidate.
        if code == "C-JS":
            # Salonga assigned but text might still have '"Rey"' from Marquez
            rewritten = _re.sub(r'\s*"Rey"\s*', ' ', rewritten)
            rewritten = _re.sub(r'\s*"JM"\s*"Rey"\s*', ' "JM" ', rewritten)
        if code == "C-RM":
            # Marquez assigned but might have '"JM"' from Salonga
            rewritten = _re.sub(r'\s*"JM"\s*', ' ', rewritten)
        # Collapse any double spaces left behind
        rewritten = _re.sub(r'\s+', ' ', rewritten).strip()

        card["text"] = rewritten
        card["candidate"] = code

        # Re-extract indicators on rewritten text
        card.update({
            k: v for k, v in indicator_summary_for_card(rewritten).items()
            if k in {"fired_indicators", "indicator_details", "named_features"}
        })

        # v2.4: re-apply verdict-rule alignment guard here too,
        # since callers may run script 22 on already-verdicted cards
        # without re-running script 18. Threshold raised from >=2 to
        # >=3 to match v2.4 script 18 — see issue #03 in v2.3 audit.
        verdict = card.get("verdict", "UNCERTAIN")
        n_indicators = len(card.get("fired_indicators", []))
        if verdict == "REAL" and n_indicators >= 3:
            card["verdict"] = "UNCERTAIN"
            fake_pct = card.get("fake_likelihood_percent", 0.0)
            card["fake_likelihood_percent"] = max(fake_pct, 41.0)
            card["credibility_percent"] = 100.0 - card["fake_likelihood_percent"]
            card.setdefault("provenance", {})["alignment_flag"] = \
                "demoted_real_to_uncertain_at_22"

        if args.re_explain:
            cand_obj = REGISTRY.get(code)
            cand_name = cand_obj.name if cand_obj else None
            # v2.4: pass total card count so tier ratio is 40/35/25
            # proportionally, not using legacy absolute thresholds that
            # produced 91% advanced tier in the v2.3 run.
            tier = tier_for_card_index(idx, total_in_session=len(cards))
            card["explanation"] = assemble_explanation(
                fired_indicators=card.get("fired_indicators", []),
                verdict=card.get("verdict", "UNCERTAIN"),
                fake_likelihood_percent=card.get("fake_likelihood_percent", 50.0),
                seed_str=f"{card['id']}|{args.seed}",
                tier=tier,
                candidate_name=cand_name,
            )

        prov = card.setdefault("provenance", {})
        prov.setdefault("script_chain", []).append("22")
        prov["pseudonym_session_size"] = len(session_cache)
        card.setdefault("metadata", {})["pseudonym_replacements"] = replaced

        counts[code if code in counts else "NONE"] = counts.get(code, 0) + 1
        out.append(card)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info("Pseudonymised %d cards -> %s", len(out), args.out_file)
    logger.info("Candidate distribution: %s", counts)
    logger.info("Session-cache size: %d unique entities mapped", len(session_cache))
    logger.info("Repetition-collapse applied to %d cards (v2.2)",
                repetition_collapses)
    logger.info("Cross-candidate pollution cleaned in %d cards (v2.2)",
                cross_pollution_cleaned)


if __name__ == "__main__":
    main()

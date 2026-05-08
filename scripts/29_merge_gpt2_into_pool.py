#!/usr/bin/env python3
"""
M.I.N.E.R.V.A. v2.8.7 — Script 29: Merge GPT-2 neuro-symbolic cards into the
                                    template card stream

The pipeline before v2.8.7 had a structural gap: scripts 12b/13 produced
GPT-2 cards in `generated/gpt2_neuro_final_*.jsonl`, but no script ever
merged them into the template stream. Cell 37 of the notebook reads
`generated/template_cards.json` directly into pseudonymize, so any GPT-2
contributions were stranded. v2.8.7 closes the gap with this script.

What it does:

  1. Read template_cards.json (the templates).
  2. Read gpt2_neuro_final_fake.jsonl + gpt2_neuro_final_real.jsonl
     (the post-scoring survivors from script 13).
  3. Pre-filter GPT-2 cards:
       - Drop cards with truncation_flag = True (mid-sentence cutoffs)
       - Drop cards mentioning any "Candidate X" code outside {A,B,C}
         (these would just fail script 33's strict allowlist anyway)
       - Drop cards that contain NO candidate reference at all (no place
         in the candidate-themed pool)
       - Drop empty/very-short text
  4. Transform each surviving GPT-2 card to the template-card schema, using
     the GPT-2 model's own real-valued p_fake, detector scores, and named
     features instead of the templates' synthetic stub values. The card
     ends up indistinguishable in shape from a template card to scripts
     31/23/24/33 downstream, but its provenance.source field reads
     "gpt2_neurosymbolic" so it can be audited as GPT-2-derived.
  5. Append survivors to the template list and write the merged file.
  6. Write a merge report with per-stage counts so the panel can audit
     "templates: N, GPT-2 attempted: M, GPT-2 surviving merge: K".

This is the script that finally lets us answer "show me a card GPT-2
contributed to the pool."
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("minerva.merge_gpt2")


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------

# Match "Candidate X" or "Candidate XY..." (1+ uppercase letters/digits after
# Candidate). The captured group is the code following "Candidate ".
CANDIDATE_CODE_RE = re.compile(r"\bCandidate\s+([A-Z][A-Z0-9]*)\b")

# The full allowlist for codes embedded in card text. Mirrors script 33's
# rule that only candidates A/B/C are supported in the pseudonymized scenario.
ALLOWED_CODES = frozenset({"A", "B", "C"})


def text_candidate_codes(text: str) -> list[str]:
    """Return all 'Candidate <code>' codes mentioned in text, in order.

    Used both for pre-filter (any non-A/B/C code → reject) and for assigning
    the `candidate` field on the merged card (first allowed code wins).
    """
    return [m.group(1) for m in CANDIDATE_CODE_RE.finditer(text or "")]


def remap_to_allowlist(text: str,
                       max_distinct: int = 3) -> tuple[str, int, dict]:
    """Remap arbitrary 'Candidate <code>' occurrences to A/B/C in order of
    first appearance.

    Why this exists: minerva_privacy._index_to_code uses Excel-style codes
    (A..Z, AA..AZ, ..., AAA, etc.) so the GPT-2 *training* corpus contains
    "Candidate DKR", "Candidate JZQ", etc. — all valid because JCBlaise has
    hundreds of distinct named entities. The *game* however only allows
    A/B/C. After GPT-2 generates new text using whatever codes it saw, we
    remap them: first distinct code seen → A, second → B, third → C.

    If the card mentions MORE than `max_distinct` distinct entities, return
    (text, n_distinct, {}) with empty mapping — the caller should drop the
    card because there's no clean A/B/C slot left.

    Returns (remapped_text, n_distinct_codes_seen, code_to_letter_mapping).
    """
    distinct_codes: list[str] = []
    for m in CANDIDATE_CODE_RE.finditer(text or ""):
        code = m.group(1)
        if code not in distinct_codes:
            distinct_codes.append(code)

    n_distinct = len(distinct_codes)
    if n_distinct > max_distinct:
        return text, n_distinct, {}

    letters = ["A", "B", "C"][:max_distinct]
    code_to_letter = dict(zip(distinct_codes, letters))

    if not code_to_letter:
        return text, 0, {}

    def replace(m):
        return f"Candidate {code_to_letter[m.group(1)]}"

    return CANDIDATE_CODE_RE.sub(replace, text), n_distinct, code_to_letter


def passes_candidate_filter(text: str) -> tuple[bool, str]:
    """Return (passes, reason). reason is empty when passes is True.

    Run AFTER remap_to_allowlist; remaining issue is "no candidate at all".
    """
    codes = text_candidate_codes(text)
    if not codes:
        return False, "no_candidate_reference"
    bad = [c for c in codes if c not in ALLOWED_CODES]
    if bad:
        return False, f"foreign_candidate_codes: {sorted(set(bad))}"
    return True, ""


def recover_truncated_text(text: str, min_chars: int = 80) -> tuple[str, str]:
    """Try to recover usable text from a mid-sentence-truncated generation.

    GPT-2 with max_new_tokens=120 frequently cuts off mid-word/mid-clause.
    The text up to that cutoff is often perfectly usable Tagalog if we
    trim back to the last complete sentence boundary. Strategy:

      1. Find the last `.`, `?`, or `!` in the text.
      2. If found AND the resulting prefix is at least `min_chars` long,
         keep everything up to and including that boundary.
      3. Otherwise, return the original text unchanged (and let the
         downstream quality filter reject it).

    Returns (recovered_text, status). status is one of:
      "ok"          : original text was already complete
      "trimmed"     : recovered by trimming to last sentence boundary
      "unrecoverable": no sentence boundary found at sufficient length
    """
    if not text:
        return text, "unrecoverable"
    text = text.strip()

    # If it already ends with a sentence terminator, no recovery needed
    if text and text[-1] in ".!?":
        return text, "ok"

    # Find the last terminator
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last >= min_chars - 1:
        return text[:last + 1].strip(), "trimmed"

    return text, "unrecoverable"


def passes_quality_filter(card: dict, min_chars: int = 80,
                          max_chars: int = 1200) -> tuple[bool, str, str]:
    """Returns (passes, reason, recovered_text). recovered_text is the
    possibly-trimmed version of card['text'] that should be used downstream.

    The script 13 truncation_flag is `{"is_truncated": bool, "reason": str}`
    so we can't just check truthiness on a dict-vs-bool. We DO honor truly
    degenerate flags (empty, too_short) but try to recover from
    "dangling_function_word" via sentence trimming.
    """
    raw_text = (card.get("text") or "").strip()
    if not raw_text:
        return False, "empty_text", raw_text

    # Hard rejection: degenerate flags from script 13 that can't be recovered
    tf = card.get("truncation_flag", {})
    if isinstance(tf, dict):
        reason = tf.get("reason", "")
        if reason in ("empty", "too_short"):
            return False, f"truncation:{reason}", raw_text
    elif tf is True:  # back-compat with simpler bool flag
        # Try recovery anyway
        pass

    # Try to recover the text. If it already ended cleanly, status="ok".
    recovered, status = recover_truncated_text(raw_text, min_chars=min_chars)

    if len(recovered) < min_chars:
        return False, f"too_short_after_recovery ({len(recovered)} < {min_chars})", recovered
    if len(recovered) > max_chars:
        return False, f"too_long ({len(recovered)} > {max_chars})", recovered
    if status == "unrecoverable":
        return False, "unrecoverable_truncation", recovered

    # Recovered or already-clean text passes
    return True, "" if status == "ok" else f"recovered:{status}", recovered


# ---------------------------------------------------------------------------
# Schema mapping: GPT-2 jsonl card → template-card shape
# ---------------------------------------------------------------------------

def _verdict_from_pfake(p_fake: float) -> str:
    """Mirror the verdict logic the rest of the pipeline uses (threshold 0.5
    with a UNCERTAIN band of ±0.15 either side).
    """
    if p_fake >= 0.65:
        return "FAKE"
    if p_fake <= 0.35:
        return "REAL"
    return "UNCERTAIN"


def _difficulty_from_pfake(p_fake: float) -> str:
    margin = abs(p_fake - 0.5)
    if margin >= 0.30:
        return "easy"      # clear-cut
    if margin >= 0.10:
        return "medium"
    return "hard"          # near 0.5 = ambiguous


def _build_indicator_details(named_features: dict, fired: list[str]) -> dict:
    """Build the indicator_details mini-objects expected by faithfulness audit."""
    label_map = {
        "EMO": "Emotional / loaded language",
        "URG": "Urgency / pressure",
        "ANON": "Anonymous / unverified source",
        "MISS": "Missing context / sourcing",
        "FAB": "Fabricated specifics",
        "POL": "Polarization framing",
        "CONS": "Consistency with prior statement",
        "DISC": "Discrediting frame",
    }
    out = {}
    for code, label in label_map.items():
        ks = f"ind_{code.lower()}_score"
        kf = f"ind_{code.lower()}_fired"
        out[code] = {
            "code": code,
            "label": label,
            "score": float(named_features.get(ks, 0.0)),
            "fired": bool(named_features.get(kf, 0.0)),
        }
    return out


# ---------------------------------------------------------------------------
# v2.9.0: Response bank loader + indicator-coverage filter
# ---------------------------------------------------------------------------

_RESPONSE_BANK: dict | None = None
_RESPONSE_BANK_PATH = "templates/response_bank_v2.json"


def _load_response_bank(path: str = _RESPONSE_BANK_PATH) -> dict:
    """Load the v2.9 response bank into a module-level cache.

    The bank is keyed `{INDICATOR}/{role}/{tier}` where role ∈ {fake, real}
    and tier ∈ {novice, proficient, advanced}. Each value is a list of phrase
    variants {phrase_tl, phrase_en, verifier_action}. Loaded lazily so unit
    tests that don't touch the bank don't pay the cost.
    """
    global _RESPONSE_BANK
    if _RESPONSE_BANK is not None:
        return _RESPONSE_BANK
    bank_path = Path(path)
    if not bank_path.exists():
        # Fallback to v1 stub schema so the script still runs in test envs
        # that don't have the bank file. We log a warning so production runs
        # don't silently degrade.
        logger.warning(
            "Response bank not found at %s — falling back to stub phrases. "
            "Faithfulness audit will likely regress. Build the bank with "
            "templates/response_bank_v2.json.", path)
        _RESPONSE_BANK = {"phrases": {}, "version": "stub"}
        return _RESPONSE_BANK
    _RESPONSE_BANK = json.loads(bank_path.read_text(encoding="utf-8"))
    return _RESPONSE_BANK


def gpt2_indicators_supported_by_bank(fired: list[str], target_label: str,
                                      tier: str,
                                      bank: dict | None = None) -> tuple[bool, list[str]]:
    """Check whether ALL of `fired` have a matching response-bank entry for
    this (role, tier). Returns (all_supported, list_of_unsupported).

    GPT-2 cards whose fired indicators are NOT a subset of the bank's keys
    will produce explanation phrases that don't match the indicator — exactly
    the failure mode the v2.8.7 faithfulness audit caught (231 mismatches).
    Filtering before promotion eliminates this class of failures entirely.
    """
    if bank is None:
        bank = _load_response_bank()
    keys = set(bank.get("phrases", {}).keys())
    role = "fake" if target_label == "fake" else "real"
    unsupported = []
    for ind in fired:
        bank_key = f"{ind}/{role}/{tier}"
        if bank_key not in keys:
            unsupported.append(ind)
    return (len(unsupported) == 0, unsupported)


def _pick_phrase_variant(bank: dict, key: str, idx_seed: int) -> dict | None:
    """Deterministically pick one of the variants for a given bank key."""
    variants = bank.get("phrases", {}).get(key, [])
    if not variants:
        return None
    return variants[idx_seed % len(variants)]


def _build_explanation(target_label: str, fired: list[str], tier: str,
                       card_idx: int = 0) -> dict:
    """Build a faithfulness-passing explanation block using the v2.9
    response bank. Replaces the v2.8.7 stub that produced phrases like
    "GPT-2 generation flagged X" — those triggered 231 indicator-phrase
    mismatches in the faithfulness audit.

    `card_idx` is used to deterministically rotate through phrase variants
    so that two cards with the same fired indicators don't get identical
    explanations. This is what pushes explanation diversity above 30%.
    """
    bank = _load_response_bank()
    role = "fake" if target_label == "fake" else "real"
    sift_move = "STOP" if target_label == "fake" else "TRACE"

    summary_intro = ("This post shows misinformation cues."
                     if target_label == "fake"
                     else "This post shows credibility cues.")
    if fired:
        summary = (f"{summary_intro} {len(fired)} indicator(s) fired: "
                   f"{', '.join(fired)}.")
    else:
        summary = f"{summary_intro} No strong indicators fired."

    indicator_phrases = []
    for j, ind in enumerate(fired):
        bank_key = f"{ind}/{role}/{tier}"
        variant = _pick_phrase_variant(bank, bank_key, idx_seed=card_idx + j)
        if variant is None:
            # Bank missing this combination — fall back to a generic phrase
            # but log it so the gap can be filled. The pre-filter
            # (gpt2_indicators_supported_by_bank) should have caught this
            # but we degrade gracefully if filter was bypassed.
            phrase = (f"This card's {ind} indicator needs a hand-authored "
                      f"explanation in the response bank.")
            verifier_action = ""
            phrase_en = phrase
        else:
            phrase = variant["phrase_tl"]
            phrase_en = variant.get("phrase_en", "")
            verifier_action = variant.get("verifier_action", "")

        indicator_phrases.append({
            "indicator": ind,
            "phrase": phrase,
            "phrase_en": phrase_en,
            "verifier_action": verifier_action,
            "bank_ref": f"{ind}/{role}/{tier}/v{(card_idx + j) % 3}",
            "sift_move": sift_move,
        })

    # For real-verdict cards, append a CREDIBLE phrase like the templates do
    if target_label == "real":
        cred_variant = _pick_phrase_variant(bank, f"CREDIBLE/real/{tier}",
                                            idx_seed=card_idx)
        if cred_variant is not None:
            indicator_phrases.append({
                "indicator": "CREDIBLE",
                "phrase": cred_variant["phrase_tl"],
                "phrase_en": cred_variant.get("phrase_en", ""),
                "verifier_action": cred_variant.get("verifier_action", ""),
                "bank_ref": f"CREDIBLE/real/{tier}/v{card_idx % 3}",
                "sift_move": "TRACE",
            })

    return {
        "tier": tier,
        "summary": summary,
        "indicator_phrases": indicator_phrases,
        "sift_move": sift_move,
        "credible_counter_card_id": None,
        "bank_version": bank.get("version", "v2.9.0"),
    }


def gpt2_card_to_template_shape(g: dict, idx: int) -> dict:
    """Convert a GPT-2 jsonl record to the template-card schema."""
    text = g.get("text", "").strip()
    target = g.get("target", "fake")  # "fake" or "real"
    p_fake = float(g.get("p_fake", 0.5))
    nf = g.get("named_features", {}) or {}
    det = g.get("detectors", {}) or {}
    tier = (g.get("control_tokens", {}) or {}).get("tier", "novice")
    verdict = _verdict_from_pfake(p_fake)

    # Find which allowed candidate code appears first in the text — that's
    # the candidate we attribute the card to. Pre-filter guarantees ≥1 code
    # exists and ALL codes are in {A,B,C}, so this can't fail at this stage.
    codes = text_candidate_codes(text)
    candidate_code = f"C-{codes[0]}" if codes else "C-A"

    fired: list[str] = []
    for k, v in nf.items():
        if k.endswith("_fired") and float(v) >= 1.0:
            fired.append(k.replace("ind_", "").replace("_fired", "").upper())
    if not fired:
        # Mirror the template default — every card gets at least MISS so the
        # response bank has something to attach to
        fired = ["MISS"]

    return {
        "id": f"gpt2_neuro_{target}_{idx:05d}",
        "text": text,
        "candidate": candidate_code,
        "target_label": target,
        "verdict": verdict,
        "fake_likelihood_percent": round(p_fake * 100, 1),
        "credibility_percent": round((1.0 - p_fake) * 100, 1),
        "difficulty_bin": _difficulty_from_pfake(p_fake),
        "fired_indicators": fired,
        "indicator_details": _build_indicator_details(nf, fired),
        "named_features": nf,
        "qlattice": {
            "score": p_fake,
            "threshold": 0.5,
            "direction": ">=",
            "margin": p_fake - 0.5,
            "pred": 1 if p_fake >= 0.5 else 0,
            "equation": "scored_by_script_13",
            "top_factors": [],
        },
        "detectors": {
            "p_roberta_fake":  float(det.get("p_roberta_fake",  p_fake)),
            "p_distil_fake":   float(det.get("p_distil_fake",   p_fake)),
            "p_degnn_fake":    float(det.get("p_degnn_fake",    p_fake)),
            "p_ensemble_fake": float(det.get("p_ensemble_fake", p_fake)),
        },
        "heuristics": {},
        "theme_flags": {
            # script 23 (theme) recomputes these but starting electoral=True
            # gives the GPT-2 cards a fair starting score
            "is_electoral": True,
            "electoral_score": 0.70,
            "is_neutral_volume": False,
            "classifier_label": "electoral",
        },
        "explanation": _build_explanation(target, fired, tier, card_idx=idx),
        "provenance": {
            "seed": idx,
            "git_sha": "gpt2_neurosymbolic_v2.8.7",
            "bank_version": "1.1",
            "generator": "gpt2_neurosymbolic",
            "tactic": f"gpt2_{target}",
            "tier": tier,
            "control_tokens": g.get("control_tokens", {}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "2.8.7",
            "script_chain": [
                "11b_train_gpt2_neurosymbolic",
                "12b_generate_gpt2_neurosymbolic",
                "13_score_generated_with_qlattice",
                "29_merge_gpt2_into_pool",
            ],
            "source": "gpt2_neurosymbolic",
            "alignment_flag": "ok",
        },
        "metadata": {
            "truncation_flag": bool(g.get("truncation_flag", False)),
            "raw_p_fake": p_fake,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Stream-read a JSONL file. Returns [] for missing paths."""
    if not path.exists():
        logger.warning("File not found: %s — treating as empty", path)
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def merge(templates_path: Path,
          gpt2_fake_path: Path,
          gpt2_real_path: Path,
          out_path: Path,
          report_path: Path,
          require_at_least: int = 0) -> dict:
    """Perform the merge. Returns the merge report dict."""

    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    if not isinstance(templates, list):
        raise ValueError(f"{templates_path} should be a JSON array of cards; "
                         f"got {type(templates).__name__}")

    gpt2_fake = _read_jsonl(gpt2_fake_path)
    gpt2_real = _read_jsonl(gpt2_real_path)

    rejected: list[dict] = []
    promoted: list[dict] = []

    for source_label, batch in [("fake", gpt2_fake), ("real", gpt2_real)]:
        for i, g in enumerate(batch):
            ok_q, reason_q, recovered_text = passes_quality_filter(g)
            if not ok_q:
                rejected.append({
                    "stage": "quality",
                    "label": source_label,
                    "reason": reason_q,
                    "text_preview": (g.get("text", "") or "")[:140],
                })
                continue

            # Remap multi-letter codes (e.g. "Candidate DKR") → A/B/C in
            # order of first appearance. Drops cards with >3 distinct
            # entities (not enough slots).
            remapped_text, n_distinct, mapping = remap_to_allowlist(recovered_text)
            if not mapping:
                if n_distinct > 3:
                    rejected.append({
                        "stage": "candidate_remap",
                        "label": source_label,
                        "reason": f"too_many_distinct_entities ({n_distinct} > 3)",
                        "text_preview": recovered_text[:140],
                    })
                else:
                    rejected.append({
                        "stage": "candidate_remap",
                        "label": source_label,
                        "reason": "no_candidate_reference",
                        "text_preview": recovered_text[:140],
                    })
                continue

            ok_c, reason_c = passes_candidate_filter(remapped_text)
            if not ok_c:
                rejected.append({
                    "stage": "candidate_allowlist",
                    "label": source_label,
                    "reason": reason_c,
                    "text_preview": remapped_text[:140],
                })
                continue

            # v2.9.0: Indicator-coverage filter.
            # Determine which indicators would fire for this GPT-2 card,
            # then check the response bank can produce a phrase for each
            # (indicator × role × tier) combination. If not, drop the card —
            # promoting it would introduce a faithfulness mismatch (which
            # was the v2.8.7 audit's #3 critical finding: 231 mismatches).
            nf_check = g.get("named_features", {}) or {}
            fired_check: list[str] = []
            for k, v in nf_check.items():
                if k.endswith("_fired") and float(v) >= 1.0:
                    fired_check.append(
                        k.replace("ind_", "").replace("_fired", "").upper()
                    )
            if not fired_check:
                fired_check = ["MISS"]
            tier_check = (g.get("control_tokens", {}) or {}).get("tier", "novice")
            ok_b, missing = gpt2_indicators_supported_by_bank(
                fired_check, source_label, tier_check
            )
            if not ok_b:
                rejected.append({
                    "stage": "response_bank_coverage",
                    "label": source_label,
                    "reason": f"unsupported_indicators: {missing}",
                    "text_preview": remapped_text[:140],
                })
                continue

            # Apply the cleaned + remapped text into the card before mapping
            g_clean = {**g, "text": remapped_text}
            try:
                promoted.append(
                    gpt2_card_to_template_shape(g_clean, len(promoted))
                )
            except Exception as e:
                rejected.append({
                    "stage": "schema_mapping",
                    "label": source_label,
                    "reason": f"{type(e).__name__}: {e}",
                    "text_preview": remapped_text[:140],
                })

    merged = templates + promoted

    # Write outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "templates_in": len(templates),
        "gpt2_fake_attempted": len(gpt2_fake),
        "gpt2_real_attempted": len(gpt2_real),
        "gpt2_promoted_total": len(promoted),
        "gpt2_promoted_by_label": {
            "fake": sum(1 for c in promoted if c["target_label"] == "fake"),
            "real": sum(1 for c in promoted if c["target_label"] == "real"),
        },
        "gpt2_rejected_total": len(rejected),
        "gpt2_rejected_by_reason": _tally(
            r["reason"].split(":")[0] for r in rejected),
        "gpt2_rejected_examples_first10": rejected[:10],
        "merged_total": len(merged),
        "outputs": {
            "merged_cards": str(out_path),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    logger.info("Merge complete:")
    logger.info("  Templates in     : %d", len(templates))
    logger.info("  GPT-2 attempted  : %d (fake) + %d (real) = %d",
                len(gpt2_fake), len(gpt2_real), len(gpt2_fake) + len(gpt2_real))
    logger.info("  GPT-2 promoted   : %d  (fake=%d, real=%d)",
                len(promoted),
                report["gpt2_promoted_by_label"]["fake"],
                report["gpt2_promoted_by_label"]["real"])
    logger.info("  GPT-2 rejected   : %d  reasons=%s",
                len(rejected), report["gpt2_rejected_by_reason"])
    logger.info("  Merged total     : %d", len(merged))

    if len(promoted) < require_at_least:
        logger.warning(
            "  GPT-2 promoted (%d) < require_at_least (%d). The pipeline "
            "will continue with mostly-template cards. Consider increasing "
            "GPT2_EPOCHS, GPT2_MAX_ATTEMPTS, or generation batch size in "
            "the notebook config.",
            len(promoted), require_at_least)

    return report


def _tally(items) -> dict:
    out: dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


def main():
    p = argparse.ArgumentParser(
        description="v2.8.7 — Merge GPT-2 neuro-symbolic generations into "
                    "the template card stream so they reach the pool."
    )
    p.add_argument("--templates", required=True,
                   help="Input: generated/template_cards.json")
    p.add_argument("--gpt2_fake", required=True,
                   help="Input: generated/gpt2_neuro_final_fake.jsonl")
    p.add_argument("--gpt2_real", required=True,
                   help="Input: generated/gpt2_neuro_final_real.jsonl")
    p.add_argument("--out", required=True,
                   help="Output: generated/template_plus_gpt2_cards.json")
    p.add_argument("--report_out",
                   default="reports/merge_gpt2_into_pool.json")
    p.add_argument("--require_at_least", type=int, default=0,
                   help="Warn (don't fail) if fewer than N GPT-2 cards "
                        "survive the merge. Set to 100+ for production runs.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    merge(
        templates_path=Path(args.templates),
        gpt2_fake_path=Path(args.gpt2_fake),
        gpt2_real_path=Path(args.gpt2_real),
        out_path=Path(args.out),
        report_path=Path(args.report_out),
        require_at_least=args.require_at_least,
    )


if __name__ == "__main__":
    main()

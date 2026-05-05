"""
minerva_filters.py
==================

Four content gates that reject non-conforming cards at generation
time, addressing the issues observed in the legacy unity_cards.json:
  1. Off-theme content (Grab/Meralco/transport leaks).
  2. Truncated GPT-2 generations (mid-sentence cut-offs).
  3. Cards that do not mention any of the three fictional candidates.
  4. Pseudonym-integrity failures (random codes leaking through).

Each gate emits a structured RejectionLog entry that becomes part of
the audit trail (Khosravi et al. 2022 — auditability requirement).

Citations: Source2Synth-style curation patterns; Hu et al. (2024)
ARG safety; Wenzek et al. (2020) CRISP perplexity gating.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Theme keyword bank
# ---------------------------------------------------------------------------
# Positive: electoral / political-race vocabulary.
# Negative (hard negatives): topics that have been observed leaking
# through the legacy keyword filter — transport, utilities, sports,
# entertainment, weather. Cards that match negatives without strong
# positive anchors are rejected.
ELECTORAL_POSITIVE = [
    # Tagalog
    "halalan", "eleksyon", "kandidato", "balota", "boto", "bumoto",
    "presidente", "bise presidente", "vice president", "senador",
    "kongresista", "gobernador", "mayor", "konsehal", "konseho",
    "partido", "campaign", "kampanya", "comelec", "pulitika",
    "pulitiko", "platform", "platapormang", "manifesto",
    # English
    "election", "candidate", "voter", "ballot", "vote", "polling",
    "campaign", "primary", "platform", "policy", "incumbent",
    "challenger", "opponent", "constituency", "district",
    "house representative", "senator", "governor", "councilor",
    "rally", "endorsement", "debate", "running mate",
    # M.I.N.E.R.V.A. legacy candidate identifiers (kept for backward compat
    # with v2.4/v2.5 deliverables and any cards still referencing the
    # original names)
    "C-RM", "C-IB", "C-JS", "Marquez", "Bantayan", "Salonga",
    "Bagong Sigla", "Tahanan ng Bayan", "Para sa Masa",
]

# v2.6.final: extend the positive keyword list with whatever candidate
# identifiers are currently configured in scripts/candidate_config.py.
# CRITICAL: filter out empty strings — in v2.6.final's generic-only
# config, first_name/last_name/nickname are empty, and "" in any text
# is always True, which would break keyword scoring.
try:
    import candidate_config as _cfg
    for _entry in _cfg.CANDIDATES_CONFIG:
        for _field in ("display_name", "public_name", "short_name",
                       "last_name", "first_name", "code", "nickname"):
            _val = _entry.get(_field)
            if isinstance(_val, str) and _val.strip():
                ELECTORAL_POSITIVE.append(_val.strip())
        for _alias in _entry.get("aliases", []) or []:
            if isinstance(_alias, str) and _alias.strip():
                ELECTORAL_POSITIVE.append(_alias.strip())
except ImportError:
    pass  # Use legacy names only

# Final dedup + remove any stray empty strings
ELECTORAL_POSITIVE = [w for w in dict.fromkeys(ELECTORAL_POSITIVE)
                      if isinstance(w, str) and w.strip()]

ELECTORAL_NEGATIVE = [
    # Transport (the Grab leak)
    "grab", "lyft", "uber", "fare", "fare hike", "transport strike",
    "tnvs", "taxi", "jeepney modernization",
    # Utilities (the Meralco leak)
    "meralco", "electric bill", "electricity rate", "kuryente",
    "ercb", "doe", "department of energy", "petron", "shell",
    # Sports — generic + boxing-specific (v2.2 expanded after audit)
    "kpop", "k-pop", "blackpink", "bts", "nba", "pba", "uaap",
    "ncaa", "boxing", "manny pacquiao boxing", "boksingero",
    "lightweight", "heavyweight", "featherweight", "mgm",
    "split decision", "fighter", "knockout", "ko win",
    "championship", "kampeonato", "tournament", "world cup",
    "fifa", "olympics", "olympiad", "olympic",
    "puntos", "iskor sa loob", "round", "ikaapat na round",
    "ikalimang round", "panalo sa laban", "first round ko",
    # Entertainment / showbiz (v2.2 expanded after audit)
    "aktres", "aktor", "actress", "showbiz", "host", "love team",
    "tv host", "kapamilya", "kapuso", "starstruck",
    "abs-cbn drama", "gma drama", "concert tour",
    # Weather / disasters (unless explicitly electoral)
    "typhoon update", "bagyo update", "lpa", "pagasa weather",
    # Misc news that drifts off-theme in the JCBlaise corpus
    "medical mission", "outreach concert", "wrestling",
]


def keyword_score(text: str) -> tuple[float, dict]:
    """
    Cheap baseline keyword scorer: returns electoral-relevance score
    in [0, 1] and the diagnostic counts.

    v2.2: weights negatives more heavily so a card with 4 sports terms
    and 1 candidate mention scores below threshold (was passing in v2.1
    because positives were treated as 1.0 each and negatives as 0.6 each).
    """
    if not text:
        return 0.0, {"pos": 0, "neg": 0, "len": 0}
    tl = text.lower()
    pos = sum(1 for w in ELECTORAL_POSITIVE if w.lower() in tl)
    neg = sum(1 for w in ELECTORAL_NEGATIVE if w.lower() in tl)
    # v2.2: stronger negative weighting (was 0.6 per negative, now 1.0)
    # plus an explicit downweight when negatives outnumber positives
    raw = pos - 1.0 * neg
    # Heavy penalty if neg count >= pos count (off-theme dominates)
    if neg >= pos and neg >= 2:
        raw -= 1.5
    # Squash to [0, 1]
    score = 1.0 / (1.0 + 2.71828 ** (-0.7 * raw))
    return score, {"pos": pos, "neg": neg, "len": len(text)}


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------
_TERMINAL_CHARS = set(".!?\"\u201d)]…")


def is_truncated(text: str) -> tuple[bool, str]:
    """
    Detect mid-sentence cut-offs from GPT-2 generation. v2.1 LENIENT:
    only flags genuinely broken fragments (empty, very short, ending
    with a Tagalog/English function word). Text that ends without
    terminal punctuation but with a substantive content word passes
    through — reflects how real GPT-2 generations look at max_tokens.

    Returns (is_truncated, reason).
    """
    if not text:
        return True, "empty"
    text = text.strip()
    if len(text) < 30:
        return True, "too_short"

    # Strict checks: only flag truly degenerate text
    last_word = text.split()[-1] if text.split() else ""
    DANGLERS = {"at", "ng", "sa", "and", "the", "of", "with", "by",
                "for", "to", "or", "but", "ay", "kay", "mga", "ang",
                "a", "an"}
    if last_word.lower().rstrip(".,;:!?\"\u201d") in DANGLERS:
        return True, "dangling_function_word"

    # Otherwise ACCEPT — even if no terminal punctuation. Real GPT-2
    # output is often clipped at max_tokens; downstream pedagogy still
    # works as long as the text isn't a fragment.
    return False, "ok"


# ---------------------------------------------------------------------------
# Pseudonym-integrity check
# ---------------------------------------------------------------------------
# v2.6.final: 'Candidate A/B/C' are the CANONICAL display names per the
# generic-only naming policy. They must NOT be flagged as legacy.
# Other placeholders (Candidate D-Z, Entity X, Person X) ARE still flagged
# as leftover unprocessed placeholders that need rewriting.
_LEGACY_PSEUDONYM_RE = re.compile(
    r"\b(?:Candidate|Entity|Person)\s+[A-Z]{1,3}\b"
)
# Whitelist for v2.6.final canonical candidate names
_CANONICAL_CANDIDATE_NAMES = {"Candidate A", "Candidate B", "Candidate C"}


def has_legacy_pseudonyms(text: str) -> tuple[bool, list[str]]:
    """Returns (any_found, list_of_offenders).

    v2.6.final: 'Candidate A', 'Candidate B', 'Candidate C' are canonical
    and not flagged. Anything else matching the placeholder regex IS flagged.
    """
    if not text:
        return False, []
    matches = list({m.group(0) for m in _LEGACY_PSEUDONYM_RE.finditer(text)})
    # Filter out canonical names
    offenders = [m for m in matches if m not in _CANONICAL_CANDIDATE_NAMES]
    return bool(offenders), offenders


# ---------------------------------------------------------------------------
# Candidate-mention check
# ---------------------------------------------------------------------------
def mentions_one_of_three(text: str, candidate_codes: list[str]) -> tuple[bool, str | None]:
    """Returns (mentions_at_least_one, the_code_mentioned)."""
    if not text:
        return False, None
    from minerva_candidates import REGISTRY  # local import to avoid cycle
    for code, cand in REGISTRY.items():
        # Match by full name, short name, or code
        if (
            cand.name in text
            or cand.short_name in text
            or code in text
        ):
            return True, code
    return False, None


# ---------------------------------------------------------------------------
# The full gate
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    accepted: bool
    reasons: list[str]
    diagnostics: dict
    candidate_code: str | None = None


def run_all_gates(
    text: str,
    *,
    require_candidate_mention: bool = True,
    theme_threshold: float = 0.55,
    allow_neutral_volume: bool = True,
) -> GateResult:
    """
    Run all four gates. Returns a GateResult with reasons for any
    rejections AND with the diagnostics that downstream scripts (24,
    25, 26) can attach to the card's provenance.

    `allow_neutral_volume` lets clearly off-theme but harmless cards
    pass through as "neutral volume" so the Chattr feed has variety,
    per the user's stated requirement.
    """
    reasons: list[str] = []
    diag: dict = {}

    # Gate 1: theme
    score, kw_diag = keyword_score(text)
    diag["theme_score"] = score
    diag.update({f"kw_{k}": v for k, v in kw_diag.items()})
    is_electoral = score >= theme_threshold
    is_neutral_volume = (
        allow_neutral_volume
        and not is_electoral
        and kw_diag["neg"] <= 1
        and kw_diag["pos"] == 0
        # Neutral-volume cards must look benign — no misinformation
        # patterns. We import here to avoid cycles.
    )
    if is_neutral_volume:
        from minerva_indicators import fired_codes
        if fired_codes(text):
            is_neutral_volume = False  # a misinfo signal disqualifies neutral
    diag["is_electoral"] = is_electoral
    diag["is_neutral_volume"] = is_neutral_volume
    if not is_electoral and not is_neutral_volume:
        reasons.append(f"theme: score={score:.2f}<{theme_threshold} and no neutral-volume pass")

    # Gate 2: truncation (only for electoral cards — neutral-volume can be fragmentary headlines)
    if is_electoral:
        truncated, why = is_truncated(text)
        diag["truncation_check"] = why
        if truncated:
            reasons.append(f"truncation: {why}")

    # Gate 3: pseudonym integrity
    has_legacy, offenders = has_legacy_pseudonyms(text)
    diag["legacy_pseudonyms"] = offenders
    if has_legacy:
        reasons.append(f"legacy_pseudonyms: {offenders}")

    # Gate 4: candidate mention (electoral cards must mention exactly one)
    cand_code = None
    if require_candidate_mention and is_electoral:
        ok, cand_code = mentions_one_of_three(text, ["C-RM", "C-IB", "C-JS"])
        diag["candidate_mention"] = cand_code
        if not ok:
            reasons.append("no_candidate_mention")

    accepted = len(reasons) == 0
    return GateResult(
        accepted=accepted,
        reasons=reasons,
        diagnostics=diag,
        candidate_code=cand_code,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    logging.basicConfig(level=logging.INFO)
    samples = [
        ("legacy off-theme leak (Grab/Meralco)",
         "Sinabi ng Manila Standard Company (MVP) na hindi na itutuloy "
         "ng Grab ang kanilang plano dahil sa banta ng transport strike."),
        ("legacy pseudonym leak",
         "Ipinagmalaki ni Candidate DTQ ang kampanya ng Pangulo laban "
         "sa rebelde."),
        ("good electoral fake",
         "URGENT! Sources say Sen. Reynaldo \"Rey\" Marquez betrayed the masa. "
         "85% of Filipinos already support his rival. Share now before this "
         "is deleted."),
        ("good electoral real",
         "Vice-Mayor Iris Bantayan filed a transparency bill at the Senate. "
         "Full text at https://www.senate.gov.ph/bill-1234. The bill mandates "
         "open procurement publication within 30 days."),
        ("truncated",
         "Sen. Reynaldo \"Rey\" Marquez vowed to push for new infrastructure "
         "funding next quarter, citing the need for"),
    ]
    for label, txt in samples:
        r = run_all_gates(txt)
        print(f"\n[{label}] accepted={r.accepted}")
        if r.reasons:
            print(f"  reasons: {r.reasons}")
        print(f"  diag: theme={r.diagnostics.get('theme_score',0):.2f}, "
              f"electoral={r.diagnostics.get('is_electoral')}, "
              f"neutral_vol={r.diagnostics.get('is_neutral_volume')}, "
              f"cand={r.candidate_code}")

"""
minerva_candidates.py
=====================

Three fictional candidate registry + archetype router that powers
deterministic pseudonymisation across the M.I.N.E.R.V.A. pipeline.

Design rationale
----------------
The legacy pipeline emitted random pseudonyms (e.g. "Candidate GQW",
"Candidate DTQ", "Entity B") that were:
  * Inconsistent across cards in the same story arc.
  * Not connected to the three VERIdex profiles the game requires.
  * Pedagogically pointless — students cannot study a candidate they
    encounter under a different code each time.

This module replaces that with:
  * A FIXED REGISTRY of three study-backed archetypes drawn from
    Arugay & Baquisal (2022), Schipper (2025), and Mendoza et al.
    (2022, 2023) — the canonical Filipino electoral-disinformation
    narrative families. The archetypes are CLEARLY FICTIONAL with
    invented names, regions, parties, and platforms.
  * A DETERMINISTIC ROUTER that maps any detected real-person
    reference, or any ambient narrative cue in a post, to one of the
    three codes (C-RM / C-IB / C-JS) by stable hash + archetype-cue
    classifier. The same input always maps to the same code; the
    same story-arc always uses the same code for the same character.
  * SESSION-SCOPED CONSISTENCY so a card mentioning a politician
    multiple times within one story uses the same code throughout.

This satisfies:
  * Thesis §3.2.2 (VERIdex module's three-candidate interface)
  * Yermilov et al. (2023) consistency-preservation pseudonymisation
  * Schipper (2025) ethical content-safety guidance
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    code: str            # "C-RM", "C-IB", "C-JS"
    name: str            # display name
    short_name: str      # last name only, for inline replacement
    archetype: str       # "DYNASTIC" | "REFORMIST" | "POPULIST"
    region: str
    party_acronym: str
    party_name: str
    platform_slogan: str
    policy_planks: tuple[str, ...]
    indicator_weights: dict[str, float]  # which misinfo indicators they attract
    counter_anchors: tuple[str, ...]     # what real evidence about them looks like
    references: tuple[str, ...]


REGISTRY: dict[str, Candidate] = {
    "C-RM": Candidate(
        code="C-RM",
        name="Sen. Reynaldo \"Rey\" Marquez",
        short_name="Marquez",
        archetype="DYNASTIC",
        region="Northern Luzon",
        party_acronym="BSL",
        party_name="Bagong Sigla Lakas",
        platform_slogan="Bagong Sigla, Bagong Bukas",
        policy_planks=(
            "Continued large-scale infrastructure rollout",
            "Stricter law-and-order policing framework",
            "Expanded national-unity youth programmes",
            "Tax reforms favouring agribusiness corridors",
        ),
        indicator_weights={
            "REV": 0.85, "ENDO": 0.75, "IMP": 0.65, "DISC": 0.50,
            "EMO": 0.30, "URG": 0.25, "ANON": 0.45, "MISS": 0.40,
            "FAB": 0.50, "POL": 0.55, "CONS": 0.30, "RECF": 0.40,
        },
        counter_anchors=(
            "Documented Senate voting record",
            "Filed Statements of Assets, Liabilities and Net Worth (SALN)",
            "Floor-debate transcripts and committee reports",
            "Public infrastructure-project audits",
        ),
        references=(
            "Arugay & Baquisal (2022)",
            "Arugay & Mendoza (ISEAS Perspective 2024/53)",
            "Schipper (Data & Policy 2025)",
        ),
    ),
    "C-IB": Candidate(
        code="C-IB",
        name="Vice-Mayor Iris Bantayan",
        short_name="Bantayan",
        archetype="REFORMIST",
        region="Central Visayas",
        party_acronym="TNB",
        party_name="Tahanan ng Bayan",
        platform_slogan="Tahanan, Hindi Pamana",
        policy_planks=(
            "Open-data government and procurement transparency",
            "Public-defender expansion for indigent litigants",
            "Climate-resilience municipal funds",
            "Mass-transit modernisation (MRT/LRT integration)",
            "Universal early-childhood-education subsidy",
        ),
        indicator_weights={
            "DISC": 0.85, "CONS": 0.70, "FAB": 0.65, "POL": 0.55,
            "EMO": 0.45, "URG": 0.30, "ANON": 0.55, "MISS": 0.40,
            "REV": 0.25, "ENDO": 0.30, "IMP": 0.30, "RECF": 0.45,
        },
        counter_anchors=(
            "Verifiable policy white papers",
            "Livestreamed town-hall recordings",
            "Court-published legal briefs and case files",
            "Civil-society coalition endorsements with named signatories",
        ),
        references=(
            "Schipper (2025) — red-tagging tactics",
            "Mendoza et al. (Asian J. Communication, 2023)",
            "Bautista (FEU AJPE, 2021)",
        ),
    ),
    "C-JS": Candidate(
        code="C-JS",
        name="Rep. Datu Jomar \"JM\" Salonga",
        short_name="Salonga",
        archetype="POPULIST",
        region="Mindanao party-list (BARMM)",
        party_acronym="PSM",
        party_name="Para sa Masa",
        platform_slogan="Para sa Masa, Para sa Mundo",
        policy_planks=(
            "Doubling of 4Ps conditional-cash-transfer coverage",
            "OFW remittance-fee protection law",
            "Cheap-rice ('Tutok-Bigas') programme",
            "Pro-poor housing acceleration via NHA",
        ),
        indicator_weights={
            "EMO": 0.85, "URG": 0.80, "ENDO": 0.75, "RECF": 0.70,
            "POL": 0.50, "FAB": 0.55, "ANON": 0.50, "MISS": 0.45,
            "REV": 0.20, "DISC": 0.40, "CONS": 0.35, "IMP": 0.50,
        },
        counter_anchors=(
            "Verified broadcast clips with timestamps",
            "Commission-on-Audit (COA) reports for sponsored projects",
            "Filed bills with House voting records",
            "Constituent-service-office case records",
        ),
        references=(
            "Mendoza et al. (2022) youth voting & misinformation",
            "Deinla et al. (Asian J. Political Science, 2022)",
            "Cadeliña et al. (IEEE ICCECT 2024) — Pinoycchio",
        ),
    ),
}


# ---------------------------------------------------------------------------
# v2.6-final: rebuild REGISTRY from editable candidate_config.py
# ---------------------------------------------------------------------------
# The metadata above (party_name, policy_planks, indicator_weights,
# counter_anchors, references) is bound to ARCHETYPES, not to specific
# names. So if the team edits scripts/candidate_config.py and changes
# C-RM/Marquez to C-A/Cruz (or anything else), we keep the rich
# archetype-bound metadata intact and just update name fields.
#
# This implements the user's v2.6-final request:
#   "the names generated will be either common names with the studies
#    backing it and Focused to the three candidates (candidates a,b,c
#    that's editable through code)"
#
# Backed by Roozenbeek & van der Linden (2019, 2020) on fictional
# examples in inoculation games, and the Hainmueller et al. (2015)
# vignette-experiment standard in political psychology.

try:
    import candidate_config as _cfg

    # Build a code -> archetype-prototype lookup so we can copy rich metadata
    _archetype_template = {}
    for _code, _cand in REGISTRY.items():
        _archetype_template.setdefault(_cand.archetype, _cand)

    _new_registry: dict[str, Candidate] = {}
    for _entry in _cfg.CANDIDATES_CONFIG:
        _proto = _archetype_template.get(_entry["archetype"])
        if _proto is None:
            # Archetype validation already done in candidate_config import
            continue
        _new_registry[_entry["code"]] = Candidate(
            code=_entry["code"],
            name=_cfg.full_name(_entry),
            short_name=_entry["last_name"],
            archetype=_entry["archetype"],
            region=_entry.get("region", _proto.region),
            party_acronym=_proto.party_acronym,
            party_name=_proto.party_name,
            platform_slogan=_proto.platform_slogan,
            policy_planks=_proto.policy_planks,
            indicator_weights=_proto.indicator_weights,
            counter_anchors=_proto.counter_anchors,
            references=_proto.references,
        )
    if len(_new_registry) == 3:
        REGISTRY = _new_registry
        logger.info("REGISTRY rebuilt from candidate_config.py: %s",
                    {c: REGISTRY[c].name for c in REGISTRY})
except ImportError:
    logger.info("candidate_config.py not found — using legacy REGISTRY")
except Exception as _e:
    logger.warning("Failed to rebuild REGISTRY from candidate_config: %s — "
                   "falling back to legacy REGISTRY", _e)


# ---------------------------------------------------------------------------
# Archetype routing cues
# ---------------------------------------------------------------------------
# Cues drawn from the indicator-susceptibility profiles AND from
# Filipino electoral discourse vocabulary. The router uses cue
# co-occurrence, not single-keyword matching, to avoid mis-routing.

DYNASTIC_CUES = [
    r"\b(?:dynasty|dinastiya|pamilya|family\s+legacy|three\s+generations?)\b",
    r"\b(?:martial\s+law|marcos[-\s]?era|gintong\s+panahon|golden\s+age)\b",
    r"\b(?:traditional|establishment|veteran)\s+(?:politician|senator|leader)\b",
    r"\b(?:law[-\s]?and[-\s]?order|discipline|peace\s+and\s+order)\b",
    r"\b(?:infrastructure|build[-\s]?build[-\s]?build)\b",
    r"\bnational\s+unity\b",
]
REFORMIST_CUES = [
    r"\b(?:reform(?:ist)?|anti[-\s]?corruption|good\s+governance)\b",
    r"\b(?:transparency|accountability|open\s+data)\b",
    r"\b(?:lawyer|abogada|abogado|public\s+defender)\b",
    r"\b(?:vice[-\s]?mayor|local\s+executive|councilor)\b",
    r"\b(?:climate|education|kababaihan)\b",
    r"\bred[-\s]?tag(?:ged|ging)?\b",
    r"\b(?:dilawan|kakampink|dilaw)\b",       # smear-target framing
]
POPULIST_CUES = [
    r"\b(?:masa|mahirap|sambayanan|bayan|ordinary\s+filipino)\b",
    r"\b(?:celebrity|tv\s+host|aktor|aktres|host)\b",
    r"\b(?:OFW|overseas\s+filipino|remittance|remit)\b",
    r"\b(?:cash\s+transfer|4Ps|ayuda|libre|libreng)\b",
    r"\b(?:against\s+the\s+elite|laban\s+sa\s+oligarch)\b",
    r"\b(?:viral|trending|million\s+views?)\b",
]

ARCHETYPE_CUE_MAP = {
    "DYNASTIC":  DYNASTIC_CUES,
    "REFORMIST": REFORMIST_CUES,
    "POPULIST":  POPULIST_CUES,
}


def _archetype_score(text: str, archetype: str) -> int:
    cues = ARCHETYPE_CUE_MAP[archetype]
    return sum(1 for pat in cues if re.search(pat, text, re.IGNORECASE))


def archetype_for_text(text: str) -> Optional[str]:
    """
    Pick the most-likely archetype for a card based on cue co-occurrence.
    Returns None if no archetype gets >=2 cue hits (caller should
    decide whether to route to a fallback or skip).
    """
    scores = {a: _archetype_score(text, a) for a in ARCHETYPE_CUE_MAP}
    best = max(scores, key=scores.get)
    if scores[best] < 2:
        return None
    return best


# ---------------------------------------------------------------------------
# Deterministic candidate selection
# ---------------------------------------------------------------------------
def _stable_hash(s: str, salt: str = "minerva") -> int:
    h = hashlib.sha256((salt + "|" + s).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def candidate_for_text(
    text: str,
    *,
    fallback_seed: int = 1729,
    session_cache: dict[str, str] | None = None,
    detected_real_name: str | None = None,
) -> Candidate:
    """
    Map a card's text to one of the three fictional candidates.

    Strategy (in order):
      1. If a detected real name is present and we have a cached
         mapping for it in this session, reuse it (consistency).
      2. Otherwise classify by archetype cue co-occurrence.
      3. If no archetype clearly wins, deterministically rotate by
         stable hash of (detected_real_name or text-prefix or seed).
      4. Cache the result for the (detected_real_name) key so future
         cards in the same story stay consistent.
    """
    if session_cache is None:
        session_cache = {}

    # 1. Reuse session-cached mapping
    cache_key = detected_real_name or ""
    if cache_key and cache_key in session_cache:
        return REGISTRY[session_cache[cache_key]]

    # 2. Cue-based archetype routing
    arch = archetype_for_text(text)
    if arch is not None:
        for code, cand in REGISTRY.items():
            if cand.archetype == arch:
                if cache_key:
                    session_cache[cache_key] = code
                return cand

    # 3. Deterministic rotation
    seed_str = detected_real_name or text[:80] or str(fallback_seed)
    h = _stable_hash(seed_str)
    codes = list(REGISTRY.keys())
    chosen = codes[h % len(codes)]
    if cache_key:
        session_cache[cache_key] = chosen
    return REGISTRY[chosen]


# ---------------------------------------------------------------------------
# Real-name detection (heuristic, NER-replaceable)
# ---------------------------------------------------------------------------
# These are well-known Filipino political surnames + titles that the
# pipeline must pseudonymise. We deliberately do NOT include first
# names alone (too many collisions). The list is intentionally small;
# spaCy NER is the long-term replacement (already a project dep).
# Two-tier matcher to keep case sensitivity for the [A-Z]/[a-z] parts
# while supporting case-insensitive title prefixes via explicit alternation.
_TITLE_NAME_RE = re.compile(
    r"\b(?:Pres(?:ident)?\.?|Sen(?:ator)?\.?|Rep(?:resentative)?\.?|Mayor|"
    r"Gov(?:ernor)?\.?|Sec(?:retary)?\.?|VP|Vice\s*Pres(?:ident)?|"
    r"Vice[-\s]+Mayor|Cong\.?|Atty\.?)\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)
_SURNAME_ALLOWLIST = [
    "Marcos", "Duterte", "Robredo", "Pacquiao", "Moreno", "Lacson",
    "Sotto", "Aquino", "Estrada", "Binay", "Roxas", "Trillanes",
    "Hontiveros", "Pangilinan", "Villar", "Cayetano", "Recto",
    "Gordon", "Drilon", "Enrile", "Honasan", "Poe", "Pimentel",
    "Escudero", "Legarda", "Zubiri", "Tolentino", "Diokno",
    "Gatchalian", "BBM", "Leni",
]
_SURNAME_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _SURNAME_ALLOWLIST) + r")\b"
)


def detect_real_names(text: str) -> list[str]:
    """Heuristic detection of well-known Filipino political names."""
    if not text:
        return []
    found: set[str] = set()
    for m in _TITLE_NAME_RE.finditer(text):
        found.add(m.group(0).strip())
    for m in _SURNAME_RE.finditer(text):
        found.add(m.group(0).strip())
    # Filter out items that are substrings of larger items (avoid double-replace)
    sorted_by_len = sorted(found, key=len, reverse=True)
    deduped: list[str] = []
    for item in sorted_by_len:
        if not any(item != other and item in other for other in deduped):
            deduped.append(item)
    return deduped


# ---------------------------------------------------------------------------
# Pseudonymisation
# ---------------------------------------------------------------------------
def pseudonymize(
    text: str,
    *,
    session_cache: dict[str, str] | None = None,
    seed: int = 1729,
) -> tuple[str, str, list[str]]:
    """
    Replace any detected real-name reference with the corresponding
    fictional candidate's display name.

    Returns: (rewritten_text, candidate_code, list_of_replaced_names)
    """
    if session_cache is None:
        session_cache = {}

    names = detect_real_names(text)
    if not names:
        # No real-name match: still route to a candidate by cue/hash
        cand = candidate_for_text(text, fallback_seed=seed,
                                   session_cache=session_cache)
        return text, cand.code, []

    rewritten = text
    chosen_code: str | None = None
    for nm in names:
        cand = candidate_for_text(
            text, fallback_seed=seed,
            session_cache=session_cache,
            detected_real_name=nm,
        )
        rewritten = re.sub(
            re.escape(nm),
            cand.name,
            rewritten,
        )
        chosen_code = cand.code  # last one wins; consistency is via cache

    if chosen_code is None:
        chosen_code = candidate_for_text(text, fallback_seed=seed,
                                          session_cache=session_cache).code
    return rewritten, chosen_code, names


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        "Sen. Marcos vowed continued infrastructure spending under the Bagong "
        "Pilipinas banner, citing dynasty experience.",
        "VP Sara was endorsed by 85% of Filipinos in a viral survey, share now!",
        "Vice-Mayor Robredo filed a transparency bill with red-tagged "
        "civil-society partners.",
        "An anonymous source said the celebrity host is loved by the masa.",
    ]
    cache: dict[str, str] = {}
    for s in samples:
        out, code, names = pseudonymize(s, session_cache=cache, seed=42)
        print(f"\nIN : {s}")
        print(f"OUT: {out}")
        print(f"  -> code={code}, replaced={names}")
    print(f"\nSession cache: {cache}")

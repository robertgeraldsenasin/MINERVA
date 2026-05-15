#!/usr/bin/env python3
"""12-indicator definitions (EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF) and SIFT moves."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


# Lexicons
# Lexicons are deliberately conservative — high precision, lower recall.
# A missed indicator is far less harmful than a falsely-fired one because
# the latter would mislead a learner about WHY a post is suspect.
# Tagalog terms drawn from JCBlaise (Cruz et al. 2020) labelled examples
# and from the disinformation-narrative inventories of Arugay & Baquisal
# (2022) and Schipper (2025).

TAGALOG_EMOTIONAL = {
    "galit", "outraged", "ipinagkanulo", "ipinagkakanulo", "balasubas",
    "tarantado", "duwag", "patay-gutom", "nakakahiya", "nakakadiri",
    "kasuklam-suklam", "kahiya-hiya", "scandalo", "iskandalo", "lintik",
    "tarantado", "putang", "demonyo", "salbahe", "trapo", "epal",
    "manloloko", "magnanakaw", "kurakot",
}
ENGLISH_EMOTIONAL = {
    "outraged", "betrayed", "shocking", "horrific", "disgusting",
    "appalling", "scandal", "exposed", "destroyed", "crushed", "evil",
    "monster", "traitor", "thief", "liar", "corrupt", "scumbag",
    "disgrace", "shameful", "shameless", "atrocious", "vile",
    "abomination", "nightmare",
}
EMOTIONAL_LEXICON = TAGALOG_EMOTIONAL | ENGLISH_EMOTIONAL

URGENCY_TAGALOG = {
    "ngayon na", "bago mahuli", "i-share na", "ipakalat", "wag mong",
    "huwag mong", "sa lalong madaling panahon", "mabilisan",
    "agad-agad", "kaagad",
}
URGENCY_ENGLISH = {
    "share now", "before it's deleted", "before its deleted", "act now",
    "urgent", "breaking", "emergency", "must read", "must share",
    "spread the word", "wake up", "open your eyes", "do not delay",
    "share before", "delete before",
}

ANONYMOUS_PATTERNS = [
    r"\bsources?\s+say(s)?\b",
    r"\baccording to\s+(?:insiders?|sources?|witnesses?|experts?)\b",
    r"\binsiders?\s+(?:reveal(?:ed)?|claim(?:ed)?|told)\b",
    r"\b(?:a|some|certain)\s+(?:insider|witness|expert|official)\s+",
    r"\bdaw\b(?!\s+(?:po|opo))",            # Tagalog hearsay marker
    r"\branong\b",                           # Tagalog "supposedly"
    r"\bbalita\s+(?:diumano|umano|raw)\b",  # Tagalog "report says"
    r"\b(?:diumano|umano)\b",
    r"\banonymous\s+(?:source|tip|caller|leaker)\b",
    r"\bunnamed\s+(?:source|official|insider)\b",
]

CONSPIRATORIAL_PATTERNS = [
    r"\bthey\s+don'?t\s+want\s+you\s+to\s+know\b",
    r"\bayaw\s+nilang\s+malaman\b",
    r"\bhidden\s+(?:agenda|truth|plan|cabal)\b",
    r"\bcover[- ]?up\b",
    r"\bsikretong\b",
    r"\b(?:secret|hidden)\s+(?:society|government|elite|cabal)\b",
    r"\bdeep\s+state\b",
    r"\bnew\s+world\s+order\b",
    r"\bconspiracy\b",
    r"\bbinubura\s+ng\b",
    r"\bsinusupil\s+ng\b",
]

POLARIZING_PATTERNS = [
    r"\breal\s+filipinos?\b\s+(?:vs|versus|laban\s+sa)",
    r"\btrue\s+(?:patriots?|pinoy|pinoys?|citizens?)\b",
    r"\btraitor(s)?\s+to\s+the\s+(?:nation|country|flag)\b",
    r"\b(?:dilawan|dilaw|kakampink|kakam-pink|bbm|marcos\s+loyalist)s?\b",
    r"\bdayuhan(g\s+kaaway)?\b",
    r"\benemy\s+of\s+the\s+(?:state|people|nation)\b",
    r"\bus\s+(?:vs|versus|against)\s+them\b",
    r"\bmga\s+(?:tunay|totoo)ng\s+pilipino\b",
]

REVISIONIST_PATTERNS = [
    r"\bgolden\s+age\b",
    r"\bgintong\s+(?:panahon|taon)\b",
    r"\bedad\s+de\s+oro\b",
    r"\b(?:walang|no)\s+(?:martial\s+law\s+abuses?|human\s+rights\s+violations?)\b",
    r"\bmas\s+maganda\s+noon\b",
    r"\b(?:7|seven|pito)\s+thousand\s+ng\s+infrastructure\b",  # invented stats trope
    r"\brichest\s+country\s+in\s+asia\b",  # Marcos-era myth
    r"\bnumber\s+one\s+sa\s+asya\s+noong\b",
]

DISCREDITING_PATTERNS = [
    r"\bcommunist\s+sympath(?:y|izer)\b",
    r"\bnpa[-\s]+supporter\b",
    r"\bnpa\s+(?:funder|fund-raiser|recruit(?:er|or)?)\b",
    r"\bred[-\s]+tag(?:ged)?\b",
    r"\bterrorist\s+sympath\w+\b",
    r"\bka\s+(?:josephine|joma|juana)\b",  # invented red-tagging tropes
    r"\bpuppet\s+of\s+(?:china|america|us|the\s+west)\b",
    r"\b(?:tuta|alipin)\s+ng\b",
    r"\b(?:bayaran|bayad)\s+(?:lang|na|ng)\b",
]

IMPERSONATION_PATTERNS = [
    r"\b(?:abs[-\s]?cbn|gma|inquirer|rappler|cnn|bbc|reuters|ap)\.[a-z]{2,4}(?!\.ph|\.com|\.net|\.org)\b",
    r"\b(?:abs-cbn|gma|inquirer|rappler|reuters)\s+breaking\s+news\b",
    r"\bofficial\s+statement\s+from\b.*\(?(?:fake|leaked)\)?",
]

ENDORSEMENT_PATTERNS = [
    r"\b\d{2,3}\s*%\s+of\s+(?:filipinos?|voters?|pinoys?)\s+(?:already\s+)?support\b",
    r"\bsurvey\s+says?\s+\d{2,3}\s*%\b(?!.*(?:sws|pulse\s+asia|stratbase|octa|laylo|publicus|nuvoli))",
    r"\bnumber\s+one\s+choice\s+ng\s+(?:masa|sambayanan)\b",
    r"\b(?:hollywood|kpop|world)\s+star\s+endorses\b",
    r"\bunanimous\s+endorsement\s+from\b",
]

RECORD_FABRICATION_PATTERNS = [
    r"\bnobel\s+(?:peace\s+)?prize\s+winner\b",
    r"\bharvard|stanford|oxford|mit\s+(?:graduate|alumnus|alumna)\b(?!.*(?:not|never|fake))",
    r"\bphd\s+from\s+(?:harvard|stanford|yale|oxford|cambridge)\b",
    r"\bbuilt\s+\d{2,4}\s+(?:hospitals?|schools?|bridges?)\s+single[-\s]?handedly\b",
    r"\bawarded\s+by\s+(?:un|united\s+nations|world\s+bank)\b",
]


# DEPICT mapping
# Roozenbeek & van der Linden's six-technique taxonomy plus three
# Filipino-specific extensions (REV, ENDO, RECF) and one composite
# (MISS).
DEPICT = {
    "EMO":  "Emotion",
    "URG":  "Emotion+Trolling",
    "ANON": "Impersonation",
    "MISS": "Composite (no source / no link)",
    "FAB":  "Impersonation",
    "POL":  "Polarization",
    "CONS": "Conspiracy",
    "DISC": "Discrediting",
    "IMP":  "Impersonation",
    "REV":  "Filipino-specific (Arugay & Baquisal 2022)",
    "ENDO": "Filipino-specific (manufactured surveys/endorsements)",
    "RECF": "Filipino-specific (candidate-record fabrication)",
}


# Data classes
@dataclass
class IndicatorHit:
    """A single indicator detection result."""
    code: str                            # e.g. "EMO"
    label: str                           # student-facing label
    fired: bool
    evidence: list[str] = field(default_factory=list)   # actual text snippets
    score: float = 0.0                   # 0..1 confidence
    depict_family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Internal helpers
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
_QUOTE_RE = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d]{8,400})[\"\u201c\u201d]")
# Named source heuristic: capitalised phrase followed by reporting verb
_NAMED_SOURCE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+"
    r"(?:said|told|stated|announced|confirmed|sabi|sinabi|ayon\s+kay|nagsabi)\b"
)
_CAPS_TOKEN_RE = re.compile(r"\b[A-Z]{3,}\b")


def _word_count(text: str) -> int:
    return max(len(_WORD_RE.findall(text)), 1)


def _lex_hits(text_lower: str, lexicon: set[str]) -> list[str]:
    words = set(_WORD_RE.findall(text_lower))
    return sorted(words & lexicon)


def _phrase_hits(text_lower: str, phrases: set[str]) -> list[str]:
    return [p for p in phrases if p in text_lower]


def _regex_hits(text: str, patterns: list[str]) -> list[str]:
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            out.append(m.group(0))
    return out


# Indicator detectors
def detect_emo(text: str) -> IndicatorHit:
    """EMO — Emotional / loaded language."""
    tl = text.lower()
    hits = _lex_hits(tl, EMOTIONAL_LEXICON)
    wc = _word_count(text)
    density = len(hits) / wc
    # Two convergent cues raise confidence
    fired = density >= 0.025 or len(hits) >= 2
    return IndicatorHit(
        code="EMO",
        label="Emotional / loaded language",
        fired=fired,
        evidence=hits[:8],
        score=min(1.0, density * 20),
        depict_family=DEPICT["EMO"],
    )


def detect_urg(text: str) -> IndicatorHit:
    """URG — Urgency / panic cues."""
    tl = text.lower()
    phrase_hits = _phrase_hits(tl, URGENCY_TAGALOG | URGENCY_ENGLISH)
    caps_tokens = _CAPS_TOKEN_RE.findall(text)
    excl_count = text.count("!")
    fired = (
        bool(phrase_hits)
        or len(caps_tokens) >= 3
        or excl_count >= 3
    )
    evidence = phrase_hits[:5] + caps_tokens[:5]
    score = min(1.0, 0.3 * len(phrase_hits) + 0.1 * len(caps_tokens) + 0.1 * excl_count)
    return IndicatorHit(
        code="URG",
        label="Urgency / pressure cues",
        fired=fired,
        evidence=evidence,
        score=score,
        depict_family=DEPICT["URG"],
    )


def detect_anon(text: str) -> IndicatorHit:
    """ANON — Anonymous / vague attribution."""
    hits = _regex_hits(text, ANONYMOUS_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="ANON",
        label="Anonymous or vague attribution",
        fired=fired,
        evidence=hits[:5],
        score=min(1.0, 0.4 * len(hits)),
        depict_family=DEPICT["ANON"],
    )


def detect_miss(text: str) -> IndicatorHit:
    """MISS — Missing evidence / no link / no named source."""
    urls = _URL_RE.findall(text)
    named_sources = _NAMED_SOURCE_RE.findall(text)
    has_url = len(urls) > 0
    has_named_source = len(named_sources) > 0
    # MISS fires when BOTH absent — that's the "no receipts" condition
    fired = (not has_url) and (not has_named_source)
    return IndicatorHit(
        code="MISS",
        label="Missing evidence (no link, no named source)",
        fired=fired,
        evidence=[
            f"urls={len(urls)}",
            f"named_sources={len(named_sources)}",
        ],
        score=1.0 if fired else 0.0,
        depict_family=DEPICT["MISS"],
    )


def detect_fab(text: str) -> IndicatorHit:
    """FAB — Fabricated quote markers (long quote attributed without trace)."""
    quotes = _QUOTE_RE.findall(text)
    urls = _URL_RE.findall(text)
    # A long quote (>=8 words) with no URL that could anchor it
    long_quotes = [q for q in quotes if len(q.split()) >= 8]
    fired = bool(long_quotes) and not urls
    return IndicatorHit(
        code="FAB",
        label="Fabricated-quote risk (unverifiable attribution)",
        fired=fired,
        evidence=[q[:120] for q in long_quotes[:2]],
        score=1.0 if fired else 0.0,
        depict_family=DEPICT["FAB"],
    )


def detect_pol(text: str) -> IndicatorHit:
    """POL — Polarising in-group / out-group framing."""
    hits = _regex_hits(text, POLARIZING_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="POL",
        label="Polarising in-group / out-group framing",
        fired=fired,
        evidence=hits[:5],
        score=min(1.0, 0.5 * len(hits)),
        depict_family=DEPICT["POL"],
    )


def detect_cons(text: str) -> IndicatorHit:
    """CONS — Conspiratorial reasoning."""
    hits = _regex_hits(text, CONSPIRATORIAL_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="CONS",
        label="Conspiratorial reasoning",
        fired=fired,
        evidence=hits[:5],
        score=min(1.0, 0.5 * len(hits)),
        depict_family=DEPICT["CONS"],
    )


def detect_disc(text: str) -> IndicatorHit:
    """DISC — Discrediting / character attack (incl. red-tagging)."""
    hits = _regex_hits(text, DISCREDITING_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="DISC",
        label="Discrediting / personal attack",
        fired=fired,
        evidence=hits[:5],
        score=min(1.0, 0.5 * len(hits)),
        depict_family=DEPICT["DISC"],
    )


def detect_imp(text: str) -> IndicatorHit:
    """IMP — Impersonation / fake authority brand."""
    hits = _regex_hits(text, IMPERSONATION_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="IMP",
        label="Impersonation / fake authority",
        fired=fired,
        evidence=hits[:3],
        score=1.0 if fired else 0.0,
        depict_family=DEPICT["IMP"],
    )


def detect_rev(text: str) -> IndicatorHit:
    """REV — Historical revisionism (Filipino electoral context)."""
    hits = _regex_hits(text, REVISIONIST_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="REV",
        label="Historical revisionism",
        fired=fired,
        evidence=hits[:3],
        score=min(1.0, 0.6 * len(hits)),
        depict_family=DEPICT["REV"],
    )


def detect_endo(text: str) -> IndicatorHit:
    """ENDO — Manufactured survey / fake endorsement."""
    hits = _regex_hits(text, ENDORSEMENT_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="ENDO",
        label="Manufactured survey or fake endorsement",
        fired=fired,
        evidence=hits[:3],
        score=min(1.0, 0.6 * len(hits)),
        depict_family=DEPICT["ENDO"],
    )


def detect_recf(text: str) -> IndicatorHit:
    """RECF — Candidate-record fabrication."""
    hits = _regex_hits(text, RECORD_FABRICATION_PATTERNS)
    fired = bool(hits)
    return IndicatorHit(
        code="RECF",
        label="Candidate-record fabrication",
        fired=fired,
        evidence=hits[:3],
        score=min(1.0, 0.6 * len(hits)),
        depict_family=DEPICT["RECF"],
    )


_DETECTORS = [
    detect_emo, detect_urg, detect_anon, detect_miss, detect_fab,
    detect_pol, detect_cons, detect_disc, detect_imp, detect_rev,
    detect_endo, detect_recf,
]


# Public API
def extract_indicators(
    text: str, metadata: dict | None = None
) -> dict[str, IndicatorHit]:
    """Run all 12 detectors. Returns dict keyed by indicator code."""
    if not text or not isinstance(text, str):
        return {}
    return {fn.__name__.replace("detect_", "").upper(): fn(text)
            for fn in _DETECTORS}


def named_features(text: str) -> dict[str, float]:
    """
    Flat numeric feature dictionary for QLattice re-fitting.

    Replaces the opaque PCA components (rpca0, dpca1, ...) of the
    legacy pipeline with NAMED features that map 1:1 to indicators
    (Christensen et al. 2022; Brolós et al. 2021). These names appear
    directly in the discovered symbolic equation, satisfying the
    interpretability premise of the thesis (see §2.6, §2.7 of paper).
    """
    if not text:
        return {}
    hits = extract_indicators(text)
    feats: dict[str, float] = {}
    for code, hit in hits.items():
        feats[f"ind_{code.lower()}_fired"] = 1.0 if hit.fired else 0.0
        feats[f"ind_{code.lower()}_score"] = float(hit.score)
    # Additional cheap structural features
    wc = _word_count(text)
    feats["len_words"] = float(wc)
    feats["len_chars"] = float(len(text))
    feats["num_urls"] = float(len(_URL_RE.findall(text)))
    feats["num_quotes"] = float(len(_QUOTE_RE.findall(text)))
    feats["num_named_sources"] = float(len(_NAMED_SOURCE_RE.findall(text)))
    feats["caps_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    feats["excl_count"] = float(text.count("!"))
    feats["question_count"] = float(text.count("?"))
    return feats


def fired_codes(text: str) -> list[str]:
    """Convenience: list of indicator codes that fired for a post."""
    return [c for c, h in extract_indicators(text).items() if h.fired]


def indicator_summary_for_card(text: str) -> dict:
    """
    Summary block to attach to a unity_card.json.
    Contains both human-readable fired indicators and numeric feature
    vector for downstream QLattice / Random-Forest scoring.
    """
    hits = extract_indicators(text)
    fired = [code for code, h in hits.items() if h.fired]
    return {
        "fired_indicators": fired,
        "indicator_details": {c: h.to_dict() for c, h in hits.items()},
        "named_features": named_features(text),
        "depict_families": sorted({hits[c].depict_family for c in fired}),
    }


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        ("Sample fake-ish post",
         "URGENT! SHARE NOW BEFORE IT'S DELETED! Sources say Candidate "
         "betrayed the masa. They don't want you to know. Real Filipinos "
         "vs traitors!"),
        ("Sample credible post",
         "According to Sen. Maria Cruz, the bill passed third reading "
         "today. Full transcript at https://www.senate.gov.ph/journal/123."),
        ("Mixed Tagalog",
         "Diumano si Candidate ay nagpa-foul play. Walang link, walang "
         "patunay, pero share daw agad-agad."),
    ]
    for label, txt in samples:
        print(f"\n=== {label} ===")
        print(f"Text: {txt[:90]}...")
        for code in fired_codes(txt):
            h = extract_indicators(txt)[code]
            print(f"  [{code}] {h.label} | evidence={h.evidence[:2]}")

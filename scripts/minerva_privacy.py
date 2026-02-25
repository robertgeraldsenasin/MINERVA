"""MINERVA privacy utilities.

Goal
----
MINERVA generates *synthetic* educational content for a Unity game. To reduce legal / ethical risk
(e.g., generated text repeating real-world political names from training data), we pseudonymize
person-like entities into neutral placeholders such as:

    Candidate A, Candidate B, Candidate C, ...

This is **not** perfect anonymization/NER. It is a lightweight, dependency-free heuristic that
works reasonably well for Filipino/English political news style text and is suitable as a
"first line" safety step.

Recommended
-----------
- Keep pseudonymization ENABLED for any content exported outside the research environment.
- Still do a human review pass on the final JSON used by the game.

API
---
- pseudonymize_text(text, placeholder_prefix="Candidate") -> (text_pseudo, mapping)
- pseudonymize_texts(list_of_texts, placeholder_prefix="Candidate") -> (texts_pseudo, mappings)

Notes
-----
- The mapping contains original strings, so **do not export mapping files** if that is a concern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Common honorifics/titles in English + Filipino political/news text.
# We treat the token(s) after the title as a "person-like" entity.
_TITLES = [
    # English
    "President",
    "Vice President",
    "VP",
    "Senator",
    "Sen\.",
    "Rep\.",
    "Representative",
    "Congressman",
    "Mayor",
    "Governor",
    "Gov\.",
    "Secretary",
    "Sec\.",
    "Spokesperson",
    "Atty\.",
    "Attorney",
    "Dr\.",
    "General",
    "Gen\.",
    # Filipino/Tagalog
    "Pangulo",
    "Pangulong",
    "Presidente",
    "Bise Presidente",
    "Bise-Presidente",
    "Senador",
    "Kongresista",
    "Alkalde",
    "Gobernador",
    "Kalihim",
]

# Title + 1-3 capitalized tokens (e.g., "President Rodrigo Duterte", "Pangulong Duterte")
# We capture the title and the name part separately so we can preserve the title.
_TITLE_NAME_RE = re.compile(
    rf"\b(?P<title>{'|'.join(_TITLES)})\s+(?P<name>[A-Z][\w\-']*(?:\s+[A-Z][\w\-']*){{0,2}})\b"
)

# Multi-token capitalized phrases (2-4 tokens). This catches "Juan Dela Cruz", "Sara Duterte"
# but will also catch some org/location names; that's acceptable for safety-first exports.
_MULTI_CAP_RE = re.compile(
    r"\b(?P<phrase>[A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){1,3})\b")

# Some very common capitalized words/phrases we generally do NOT want to replace.
# (Keep this small; over-whitelisting defeats the purpose.)
_WHITELIST = {
    "Philippines",
    "Philippine",
    "Filipino",
    "Filipina",
    "Manila",
    "Quezon City",
    "COVID",
    "COVID-19",
}

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _index_to_label(i: int) -> str:
    """Convert 0->A, 1->B, ..., 25->Z, 26->AA, ..."""
    if i < 26:
        return _ALPHABET[i]
    # Excel-like: 26->AA, 27->AB, ...
    i0 = i
    letters = []
    while True:
        i, rem = divmod(i, 26)
        letters.append(_ALPHABET[rem])
        if i == 0:
            break
        i -= 1
    return "".join(reversed(letters))


@dataclass
class PseudonymizationResult:
    text: str
    mapping: Dict[str, str]


def pseudonymize_text(text: str, placeholder_prefix: str = "Candidate") -> PseudonymizationResult:
    """Pseudonymize one string.

    Returns
    -------
    PseudonymizationResult(text=<pseudonymized>, mapping=<original->placeholder>)
    """
    if not isinstance(text, str):
        text = ""

    # Local mapping per-text reduces cross-sample linking risk.
    mapping: Dict[str, str] = {}
    next_idx = 0

    def get_placeholder(original: str) -> str:
        nonlocal next_idx
        original = (original or "").strip()
        if not original:
            return original
        if original in _WHITELIST:
            return original
        if original not in mapping:
            label = _index_to_label(next_idx)
            mapping[original] = f"{placeholder_prefix} {label}"
            next_idx += 1
        return mapping[original]

    def maybe_reuse_from_lastname(name: str) -> str | None:
        """If we see a single-token name, try to reuse an existing multi-token mapping."""
        tokens = name.split()
        if len(tokens) != 1:
            return None
        last = tokens[0].lower()
        for k, v in mapping.items():
            ktoks = k.split()
            if len(ktoks) >= 2 and ktoks[-1].lower() == last:
                return v
        return None

    # 1) Replace title+name (preserve title).
    def _sub_title(m: re.Match) -> str:
        title = m.group("title")
        name = m.group("name")
        reuse = maybe_reuse_from_lastname(name)
        ph = reuse if reuse is not None else get_placeholder(name)
        # Ensure we do not create a multi-cap phrase that gets replaced again:
        # Multi-cap regex ignores 1-letter tokens, so 'Candidate A' is safe.
        return f"{title} {ph}"

    out = _TITLE_NAME_RE.sub(_sub_title, text)

    # 2) Replace multi-cap phrases.
    def _sub_multi(m: re.Match) -> str:
        phrase = m.group("phrase")
        if phrase in _WHITELIST:
            return phrase
        # Don't touch already-pseudonymized outputs (e.g., "President Candidate A")
        if placeholder_prefix in phrase:
            return phrase
        return get_placeholder(phrase)

    out = _MULTI_CAP_RE.sub(_sub_multi, out)

    return PseudonymizationResult(text=out, mapping=mapping)


def pseudonymize_texts(texts: List[str], placeholder_prefix: str = "Candidate") -> Tuple[List[str], List[Dict[str, str]]]:
    """Pseudonymize a list of strings.

    Returns
    -------
    (pseudonymized_texts, mappings_per_text)
    """
    pseudo: List[str] = []
    maps: List[Dict[str, str]] = []
    for t in texts:
        r = pseudonymize_text(t, placeholder_prefix=placeholder_prefix)
        pseudo.append(r.text)
        maps.append(r.mapping)
    return pseudo, maps
